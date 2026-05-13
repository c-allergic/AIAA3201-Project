#!/usr/bin/env python3
"""Part 3 inference: B_only (SCST) and C_hybrid (BasicVSR++ + SCST + fusion)."""

import argparse
import os
import sys

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from configs import Config
from models import build_model
from models.uncertainty_fusion import build_fusion_training_input, fuse_with_weight, rule_based_weight, FusionWeightCNN
from scripts.temporal_refine import temporal_refine
from utils import ensure_weights, load_frames, save_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=["B_only", "C_hybrid"], required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--use_stcm", action="store_true", help="Use SCST STCM branch instead of LocalAttention.")
    parser.add_argument("--fusion_ckpt", type=str, default="", help="Optional fusion CNN checkpoint path.")
    parser.add_argument("--skip_temporal_refine", action="store_true")
    parser.add_argument(
        "--temporal_blend",
        type=float,
        default=0.24,
        help="temporal_refine neighbor blend; larger often improves temporal metrics but may soften detail.",
    )
    parser.add_argument(
        "--fusion_gen_scale",
        type=float,
        default=1.0,
        help="Scale SCST branch weight after fusion rule/CNN in [0,1]; <1 biases toward BasicVSR++ (often raises PSNR/SSIM).",
    )
    parser.add_argument(
        "--rule_alpha",
        type=float,
        default=4.0,
        help="rule_based_weight: temporal-variance term scale.",
    )
    parser.add_argument("--rule_beta", type=float, default=6.0, help="rule_based_weight: disagreement term scale.")
    parser.add_argument("--rule_gamma", type=float, default=3.0, help="rule_based_weight: edge penalty scale.")
    parser.add_argument(
        "--rule_zeta",
        type=float,
        default=2.5,
        help="rule_based_weight: penalize SCST temporal flicker (higher -> more BasicVSR++ when gen is unstable).",
    )
    return parser.parse_args()


def _ensure_part3_weights(cfg: Config, use_stcm: bool) -> dict:
    out = {}
    out["basicvsr_pp"] = ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "basicvsr_plusplus_x4")
    try:
        out["scst_localatten"] = ensure_weights(cfg.weights.scst_root_dir, cfg.weights.urls, "scst_localatten_unet")
    except Exception as exc:
        print(f"[warn] SCST localatten checkpoint unavailable: {exc}")
        out["scst_localatten"] = os.path.join(cfg.weights.scst_root_dir, "localatten_unet.pth")
    try:
        out["scst_stcm"] = ensure_weights(cfg.weights.scst_root_dir, cfg.weights.urls, "scst_stcm_unet")
    except Exception as exc:
        print(f"[warn] SCST stcm checkpoint unavailable: {exc}")
        out["scst_stcm"] = os.path.join(cfg.weights.scst_root_dir, "stcm_unet.pth")
    chosen_unet = out["scst_stcm"] if use_stcm else out["scst_localatten"]
    out["scst_selected_unet"] = chosen_unet
    return out


def _load_fusion_model(path: str, device: torch.device):
    model = FusionWeightCNN()
    raw = torch.load(path, map_location="cpu")
    state = raw.get("state_dict", raw)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    ckpts = _ensure_part3_weights(cfg, args.use_stcm)
    lr = load_frames(args.input_dir).to(device)

    scst = build_model(
        "scst", scale=cfg.runtime.scale, scst_ckpt_root=cfg.weights.scst_root_dir
    ).to(device).eval()
    scst.temporal_mode = "stcm" if args.use_stcm else "localatten"
    scst.load_checkpoint(ckpts["scst_selected_unet"])

    with torch.no_grad():
        sr_gen = scst(lr).clamp(0.0, 1.0)
        if sr_gen.dim() == 4:
            sr_gen = sr_gen.unsqueeze(0)

    if args.mode == "B_only":
        out = sr_gen
    else:
        try:
            bvsrpp = build_model("basicvsr_pp", scale=cfg.runtime.scale).to(device).eval()
            bvsrpp.load_checkpoint(ckpts["basicvsr_pp"])
        except Exception as exc:
            print(f"[warn] BasicVSR++ unavailable, fallback to bicubic fidelity branch: {exc}")
            bvsrpp = build_model("bicubic", scale=cfg.runtime.scale).to(device).eval()
        with torch.no_grad():
            sr_fid = bvsrpp(lr).clamp(0.0, 1.0)
            if sr_fid.dim() == 4:
                sr_fid = sr_fid.unsqueeze(0)

            rb_kw = dict(alpha=args.rule_alpha, beta=args.rule_beta, gamma=args.rule_gamma, zeta=args.rule_zeta)
            if args.fusion_ckpt:
                fusion = _load_fusion_model(args.fusion_ckpt, device)
                w_rule, feat = rule_based_weight(sr_fid, sr_gen, **rb_kw)
                feat_tensor = build_fusion_training_input(feat, sr_fid, sr_gen)
                b, t, c, h, w = feat_tensor.shape
                w_pred = fusion(feat_tensor.reshape(b * t, c, h, w)).reshape(b, t, 1, h, w)
                w = torch.clamp(0.5 * w_rule + 0.5 * w_pred, 0.0, 1.0)
            else:
                w, _ = rule_based_weight(sr_fid, sr_gen, **rb_kw)
            if args.fusion_gen_scale != 1.0:
                w = torch.clamp(w * float(args.fusion_gen_scale), 0.0, 1.0)
            out = fuse_with_weight(sr_fid, sr_gen, w)
            if not args.skip_temporal_refine:
                out = temporal_refine(out, blend=float(args.temporal_blend))

    save_frames(out, args.output_dir)
    print(f"[done] mode={args.mode}: {args.input_dir} -> {args.output_dir}")


if __name__ == "__main__":
    main()
