#!/usr/bin/env python3
"""Merge per_sequence metrics from several eval JSONs; weighted over a chosen sequence set."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_over_sequences(
    per_sequence: Dict[str, Dict[str, Any]],
    seqs: List[str],
    method_key_fn,
) -> Dict[str, Any] | None:
    rows: List[Dict[str, Any]] = []
    for seq in seqs:
        if seq not in per_sequence:
            continue
        key = method_key_fn(seq)
        if key is None or key not in per_sequence[seq]:
            return None
        rows.append(per_sequence[seq][key])
    if not rows:
        return None
    tf = sum(int(r["num_frames"]) for r in rows)
    tp = sum(int(r["num_pairs"]) for r in rows)
    return {
        "num_sequences": len(rows),
        "total_frames": tf,
        "total_pairs": tp,
        "psnr": sum(r["psnr"] * r["num_frames"] for r in rows) / tf,
        "ssim": sum(r["ssim"] * r["num_frames"] for r in rows) / tf,
        "lpips": sum(r["lpips"] * r["num_frames"] for r in rows) / tf,
        "tlpips": sum(r["tlpips"] * r["num_pairs"] for r in rows) / tp if tp else 0.0,
        "flow_epe": sum(r["flow_epe"] * r["num_pairs"] for r in rows) / tp if tp else 0.0,
        "warp_l1_sr_self": sum(r["warp_l1_sr_self"] * r["num_pairs"] for r in rows) / tp,
        "warp_l1_sr_gt_flow": sum(r["warp_l1_sr_gt_flow"] * r["num_pairs"] for r in rows) / tp,
    }


def load_eval_pipeline():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_pipeline", root / "scripts" / "eval_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequences",
        nargs="+",
        default=["calendar", "foliage", "walk"],
        help="Sequences for fair compare (default matches sweep w/o city).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/eval_unified_compare_3seq.json"),
    )
    args = p.parse_args()
    seqs: List[str] = list(args.sequences)
    root = Path(__file__).resolve().parents[1]

    all_m = load_json(root / "results" / "eval_all_methods_vid4_temporal.json")
    ps_all = all_m["per_sequence"]

    v2 = load_json(root / "results" / "eval_part3_vid4_C_hybrid_v2.json")
    ps_v2 = v2["per_sequence"]

    sweep = load_json(root / "results" / "eval_part3_sweep_by_config_3seq.json")

    rows_out: Dict[str, Dict[str, Any]] = {}

    part12 = ("bicubic", "srcnn", "basicvsr", "realesrgan", "realesrnet")
    for name in part12:
        label = f"Part1/2 bi_{name}"
        m = weighted_over_sequences(ps_all, seqs, lambda s, n=name: f"bi_{s}_{n}")
        if m:
            rows_out[label] = m

    m = weighted_over_sequences(ps_all, seqs, lambda s: f"part3_{s}_C_hybrid")
    if m:
        rows_out["Part3 C_hybrid (早期默认)"] = m

    m = weighted_over_sequences(ps_v2, seqs, lambda s: f"part3_{s}_C_hybrid_v2")
    if m:
        rows_out["Part3 C_hybrid_v2"] = m

    for cfg, block in sweep["summary_by_config"].items():
        rows_out[f"Part3 sweep {cfg}"] = {k: v for k, v in block.items() if k != "num_sequences"}

    ep = load_eval_pipeline()
    agg = ep.compute_video_aggregation(rows_out, frame_weight=0.45, temporal_weight=0.55)

    out = {
        "meta": {
            "sequences": seqs,
            "sources": [
                "results/eval_all_methods_vid4_temporal.json (Part1/2 + Part3 C_hybrid)",
                "results/eval_part3_vid4_C_hybrid_v2.json",
                "results/eval_part3_sweep_by_config_3seq.json",
            ],
            "note": "同一序列子集上按 num_frames / num_pairs 加权；与全 VID4 四序列表不可直接比绝对值。",
        },
        "summary": rows_out,
        "aggregation": agg,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
