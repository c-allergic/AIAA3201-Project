#!/usr/bin/env python3
"""Evaluate SR frame folders with PSNR/SSIM/LPIPS/FID/tLPIPS and temporal flow metrics.

Writes optional ``aggregation`` (frame + video/temporal min-max blend) when ``summary`` has
methods and ``--skip_aggregation`` is not set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch


def list_frames(folder: str) -> List[Path]:
    p = Path(folder)
    frames = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
    if not frames:
        raise FileNotFoundError(f"No frames found in {folder}")
    return frames


def to_float_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    return img[border:-border, border:-border, :]


def psnr(sr: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((sr - gt) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def ssim_rgb(sr: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("Please install scikit-image for SSIM: pip install scikit-image") from exc
    return float(structural_similarity(sr, gt, channel_axis=2, data_range=1.0))


def np_to_lpips_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def ensure_same_hw(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _luminance_u8(rgb: np.ndarray) -> np.ndarray:
    """rgb float HWC [0,1] -> uint8 gray."""
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(g * 255.0, 0.0, 255.0).astype(np.uint8)


def _dense_flow_farneback(prev_u8: np.ndarray, next_u8: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(
        prev_u8, next_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )


def _warp_rgb_forward(flow: np.ndarray, prev_rgb: np.ndarray) -> np.ndarray:
    """Warp prev_rgb toward next using forward flow (dx,dy) defined on prev grid."""
    import cv2

    h, w = flow.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    prev_bgr = cv2.cvtColor((np.clip(prev_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    warped_bgr = cv2.remap(prev_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return warped_rgb


def temporal_flow_metrics(srs: List[np.ndarray], gts: List[np.ndarray]) -> Dict[str, float]:
    """Temporal metrics from dense optical flow (Farneback on luminance).

    flow_epe: mean ||F_sr - F_gt||_2 (lower = SR motion field closer to GT).
    warp_l1_sr_self: mean L1(Warp(I_sr^t, F_sr^{t->t+1}), I_sr^{t+1}) (lower = self-consistent dynamics).
    warp_l1_sr_gt_flow: mean L1(Warp(I_sr^t, F_gt^{t->t+1}), I_sr^{t+1}) (lower = follows GT motion).
    """
    if len(srs) < 2:
        return {"flow_epe": 0.0, "warp_l1_sr_self": 0.0, "warp_l1_sr_gt_flow": 0.0}
    epes: List[float] = []
    wself: List[float] = []
    wgt: List[float] = []
    for i in range(len(srs) - 1):
        sp0 = _luminance_u8(srs[i])
        sp1 = _luminance_u8(srs[i + 1])
        gp0 = _luminance_u8(gts[i])
        gp1 = _luminance_u8(gts[i + 1])
        f_sr = _dense_flow_farneback(sp0, sp1)
        f_gt = _dense_flow_farneback(gp0, gp1)
        epes.append(float(np.mean(np.linalg.norm(f_sr - f_gt, axis=-1))))
        warped_s = _warp_rgb_forward(f_sr, srs[i])
        wself.append(float(np.mean(np.abs(warped_s - srs[i + 1]))))
        warped_g = _warp_rgb_forward(f_gt, srs[i])
        wgt.append(float(np.mean(np.abs(warped_g - srs[i + 1]))))
    return {
        "flow_epe": float(np.mean(epes)),
        "warp_l1_sr_self": float(np.mean(wself)),
        "warp_l1_sr_gt_flow": float(np.mean(wgt)),
    }


def evaluate_one(
    sr_dir: str,
    gt_dir: str,
    crop: int,
    lpips_model: "lpips.LPIPS",
    device: torch.device,
    skip_temporal_flow: bool = False,
) -> Dict[str, float]:
    sr_frames = list_frames(sr_dir)
    gt_frames = list_frames(gt_dir)
    n = min(len(sr_frames), len(gt_frames))
    if n == 0:
        raise ValueError(f"Empty matched frame count for {sr_dir}")

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    tlpips_vals: List[float] = []
    prev_sr_lpips: torch.Tensor = None
    sr_cropped: List[np.ndarray] = []
    gt_cropped: List[np.ndarray] = []
    for i in range(n):
        sr = crop_border(to_float_rgb(sr_frames[i]), crop)
        gt = crop_border(to_float_rgb(gt_frames[i]), crop)
        sr, gt = ensure_same_hw(sr, gt)
        sr_cropped.append(sr)
        gt_cropped.append(gt)
        psnr_vals.append(psnr(sr, gt))
        ssim_vals.append(ssim_rgb(sr, gt))
        sr_t = np_to_lpips_tensor(sr.astype(np.float32), device)
        gt_t = np_to_lpips_tensor(gt.astype(np.float32), device)
        with torch.no_grad():
            lpips_vals.append(float(lpips_model(sr_t, gt_t).mean().item()))
            if prev_sr_lpips is not None:
                tlpips_vals.append(float(lpips_model(prev_sr_lpips, sr_t).mean().item()))
        prev_sr_lpips = sr_t

    out: Dict[str, float] = {
        "num_frames": n,
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
        "tlpips": float(np.mean(tlpips_vals)) if tlpips_vals else 0.0,
        "num_pairs": max(n - 1, 0),
    }
    if not skip_temporal_flow and n >= 2:
        out.update(temporal_flow_metrics(sr_cropped, gt_cropped))
    elif not skip_temporal_flow:
        out.update({"flow_epe": 0.0, "warp_l1_sr_self": 0.0, "warp_l1_sr_gt_flow": 0.0})
    return out


def calculate_fid(sr_dir: str, gt_dir: str, device: torch.device, batch_size: int, dims: int) -> float:
    from pytorch_fid import fid_score

    return float(
        fid_score.calculate_fid_given_paths(
            [gt_dir, sr_dir],
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=4,
        )
    )


def _safe_link(src: str, dst: str) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _minmax_score(
    value: float, values: List[float], *, higher_is_better: bool
) -> float:
    """Map value to [0,1] within batch; 0.5 if all equal."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return 0.5
    t = (value - lo) / (hi - lo)
    return float(t if higher_is_better else 1.0 - t)


def compute_video_aggregation(
    summary: Dict[str, Dict[str, Any]],
    *,
    frame_weight: float = 0.45,
    temporal_weight: float = 0.55,
) -> Dict[str, Any]:
    """Cross-method min-max scores: frame (PSNR/SSIM/LPIPS) + video/temporal (tLPIPS + flow + warp).

    Higher is better for all returned subscores and ``overall``. Intended for **relative ranking**
    within one eval JSON (not comparable across different eval runs).

    If temporal flow metrics are missing (``None``), the temporal bucket uses only tLPIPS when
    pair count > 0; weights are renormalized so frame + temporal still sum to 1.
    """
    methods = sorted(summary.keys())
    wf = max(float(frame_weight), 0.0)
    wt = max(float(temporal_weight), 0.0)
    s = wf + wt
    if s < 1e-12:
        wf, wt = 0.5, 0.5
    else:
        wf, wt = wf / s, wt / s

    def collect(key: str) -> Tuple[List[str], List[float]]:
        names: List[str] = []
        vals: List[float] = []
        for m in methods:
            v = summary[m].get(key)
            if v is None:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                names.append(m)
                vals.append(float(v))
        return names, vals

    ps_m, ps_v = collect("psnr")
    ss_m, ss_v = collect("ssim")
    lp_m, lp_v = collect("lpips")
    tl_m, tl_v = collect("tlpips")
    fe_m, fe_v = collect("flow_epe")
    ws_m, ws_v = collect("warp_l1_sr_self")
    wg_m, wg_v = collect("warp_l1_sr_gt_flow")

    per_method: Dict[str, Dict[str, float]] = {}
    for m in methods:
        parts_f: List[float] = []
        if m in ps_m:
            parts_f.append(_minmax_score(summary[m]["psnr"], ps_v, higher_is_better=True))
        if m in ss_m:
            parts_f.append(_minmax_score(summary[m]["ssim"], ss_v, higher_is_better=True))
        if m in lp_m:
            parts_f.append(_minmax_score(summary[m]["lpips"], lp_v, higher_is_better=False))
        frame_sub = float(np.mean(parts_f)) if parts_f else 0.5

        parts_t: List[float] = []
        if m in tl_m and summary[m].get("total_pairs", 0):
            parts_t.append(_minmax_score(summary[m]["tlpips"], tl_v, higher_is_better=False))
        if m in fe_m:
            parts_t.append(_minmax_score(summary[m]["flow_epe"], fe_v, higher_is_better=False))
        if m in ws_m:
            parts_t.append(
                _minmax_score(summary[m]["warp_l1_sr_self"], ws_v, higher_is_better=False)
            )
        if m in wg_m:
            parts_t.append(
                _minmax_score(summary[m]["warp_l1_sr_gt_flow"], wg_v, higher_is_better=False)
            )
        temporal_sub = float(np.mean(parts_t)) if parts_t else 0.5

        w_eff_f, w_eff_t = wf, wt
        if not parts_f and parts_t:
            w_eff_f, w_eff_t = 0.0, 1.0
        elif parts_f and not parts_t:
            w_eff_f, w_eff_t = 1.0, 0.0
        overall = w_eff_f * frame_sub + w_eff_t * temporal_sub

        per_method[m] = {
            "frame_subscore": frame_sub,
            "temporal_subscore": temporal_sub,
            "overall": float(overall),
        }

    ranked = sorted(per_method.items(), key=lambda x: x[1]["overall"], reverse=True)
    return {
        "description": (
            "Min-max 归一化后按权重合成：frame=PSNR↑ SSIM↑ LPIPS↓；"
            "temporal=tLPIPS↓ flow_epe↓ warp_l1↓。overall 仅在同一 eval 内可比。"
        ),
        "frame_bucket_metrics": ["psnr", "ssim", "lpips"],
        "temporal_bucket_metrics": [
            "tlpips",
            "flow_epe",
            "warp_l1_sr_self",
            "warp_l1_sr_gt_flow",
        ],
        "weights": {"frame": wf, "temporal": wt},
        "rank_by_overall": [m for m, _ in ranked],
        "per_method": per_method,
    }


def compute_global_fid_for_method(
    method_name: str,
    report: Dict[str, Dict[str, Dict[str, float]]],
    gt_root: str,
    results_root: str,
    device: torch.device,
    batch_size: int,
    dims: int,
) -> float:
    with tempfile.TemporaryDirectory(prefix="vsr_fid_gt_") as gt_tmp, tempfile.TemporaryDirectory(
        prefix="vsr_fid_sr_"
    ) as sr_tmp:
        idx = 0
        for seq, methods in report.items():
            if method_name not in methods:
                continue
            gt_dir = os.path.join(gt_root, seq)
            sr_dir = os.path.join(results_root, method_name)
            gt_frames = list_frames(gt_dir)
            sr_frames = list_frames(sr_dir)
            n = min(len(gt_frames), len(sr_frames))
            for i in range(n):
                ext = ".png"
                gt_dst = os.path.join(gt_tmp, f"{idx:08d}{ext}")
                sr_dst = os.path.join(sr_tmp, f"{idx:08d}{ext}")
                _safe_link(str(gt_frames[i]), gt_dst)
                _safe_link(str(sr_frames[i]), sr_dst)
                idx += 1
        if idx == 0:
            return float("nan")
        return calculate_fid(sr_tmp, gt_tmp, device, batch_size, dims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, required=True, help="GT root, e.g. data/GT")
    parser.add_argument("--results_root", type=str, required=True, help="Results root, e.g. results")
    parser.add_argument("--methods", nargs="+", required=True, help="Method dir templates, e.g. cleaned_{seq}_bicubic")
    parser.add_argument("--sequences", nargs="+", required=True, help="Sequence names, e.g. calendar city")
    parser.add_argument("--crop_border", type=int, default=4, help="Border crop for fair eval")
    parser.add_argument("--device", type=str, default="", help="cuda:0 / cpu")
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fid_dims", type=int, default=2048)
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID (avoids downloading Inception weights; PSNR/SSIM/LPIPS/tLPIPS still run).",
    )
    parser.add_argument(
        "--skip_temporal_flow",
        action="store_true",
        help="Skip Farneback optical-flow temporal metrics (faster eval).",
    )
    parser.add_argument("--save_json", type=str, default="", help="Optional output json path")
    parser.add_argument(
        "--skip_aggregation",
        action="store_true",
        help="Do not add cross-method video/frame aggregation block to JSON output.",
    )
    parser.add_argument(
        "--agg_frame_weight",
        type=float,
        default=0.45,
        help="Weight for frame bucket (PSNR/SSIM/LPIPS) in aggregation; temporal uses the remainder.",
    )
    parser.add_argument(
        "--agg_temporal_weight",
        type=float,
        default=0.55,
        help="Weight for temporal bucket (tLPIPS + flow + warp) in aggregation.",
    )
    args = parser.parse_args()

    import lpips

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seq in args.sequences:
        gt_dir = os.path.join(args.gt_root, seq)
        report[seq] = {}
        for m in args.methods:
            method_dirname = m.format(seq=seq)
            sr_dir = os.path.join(args.results_root, method_dirname)
            if not os.path.isdir(sr_dir):
                continue
            metrics = evaluate_one(
                sr_dir,
                gt_dir,
                args.crop_border,
                lpips_model,
                device,
                skip_temporal_flow=args.skip_temporal_flow,
            )
            if args.skip_fid:
                metrics["fid"] = None
            else:
                metrics["fid"] = calculate_fid(sr_dir, gt_dir, device, args.fid_batch_size, args.fid_dims)
            report[seq][method_dirname] = metrics

    method_rows: Dict[str, List[Dict[str, float]]] = {}
    for seq_data in report.values():
        for method, metrics in seq_data.items():
            method_rows.setdefault(method, []).append(metrics)

    summary: Dict[str, Dict[str, float | int | None]] = {}
    for method, rows in method_rows.items():
        total_frames = sum(int(r["num_frames"]) for r in rows)
        total_pairs = sum(int(r["num_pairs"]) for r in rows)
        w_psnr = sum(r["psnr"] * r["num_frames"] for r in rows) / total_frames
        w_ssim = sum(r["ssim"] * r["num_frames"] for r in rows) / total_frames
        w_lpips = sum(r["lpips"] * r["num_frames"] for r in rows) / total_frames
        w_tlpips = (
            sum(r["tlpips"] * r["num_pairs"] for r in rows) / total_pairs if total_pairs > 0 else 0.0
        )
        fid_vals = [r["fid"] for r in rows if r.get("fid") is not None]
        mean_fid = float(np.mean(fid_vals)) if fid_vals else None
        global_fid: float | None
        if args.skip_fid or not fid_vals:
            global_fid = None
        else:
            global_fid = float(
                compute_global_fid_for_method(
                    method_name=method,
                    report=report,
                    gt_root=args.gt_root,
                    results_root=args.results_root,
                    device=device,
                    batch_size=args.fid_batch_size,
                    dims=args.fid_dims,
                )
            )
        entry: Dict[str, float | int | None] = {
            "num_sequences": len(rows),
            "total_frames": total_frames,
            "total_pairs": total_pairs,
            "psnr": float(w_psnr),
            "ssim": float(w_ssim),
            "lpips": float(w_lpips),
            "tlpips": float(w_tlpips),
            "fid_mean_per_sequence": mean_fid,
            "fid_global": global_fid,
        }
        if not args.skip_temporal_flow and total_pairs > 0 and "flow_epe" in rows[0]:
            entry["flow_epe"] = float(
                sum(r["flow_epe"] * r["num_pairs"] for r in rows) / total_pairs
            )
            entry["warp_l1_sr_self"] = float(
                sum(r["warp_l1_sr_self"] * r["num_pairs"] for r in rows) / total_pairs
            )
            entry["warp_l1_sr_gt_flow"] = float(
                sum(r["warp_l1_sr_gt_flow"] * r["num_pairs"] for r in rows) / total_pairs
            )
        else:
            entry["flow_epe"] = None
            entry["warp_l1_sr_self"] = None
            entry["warp_l1_sr_gt_flow"] = None
        summary[method] = entry

    out: Dict[str, Any] = {"per_sequence": report, "summary": summary}
    if not args.skip_aggregation and summary:
        tw = max(float(args.agg_temporal_weight), 0.0)
        fw = max(float(args.agg_frame_weight), 0.0)
        out["aggregation"] = compute_video_aggregation(
            summary,
            frame_weight=fw,
            temporal_weight=tw,
        )

    if args.save_json:
        json_dir = os.path.dirname(os.path.abspath(args.save_json))
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
