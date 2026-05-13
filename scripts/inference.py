#!/usr/bin/env python3
"""Inference entrypoint (cleaned Part1/Part2, pretrained-first)."""

import argparse
import os
import sys

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from configs import Config
from models import build_model
from utils import ensure_weights, load_frames, save_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "bicubic",
            "srcnn",
            "basicvsr",
            "basicvsr_pp",
            "basicvsr_plusplus",
            "scst",
            "realesrgan",
            "real_esrgan",
            "realesrnet",
            "real_esrnet",
        ],
    )
    parser.add_argument("--input_dir", type=str, required=True, help="输入 LR 帧目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 SR 帧目录")
    parser.add_argument("--checkpoint", type=str, default="", help="可选：手动指定权重路径")
    parser.add_argument("--device", type=str, default="", help="可选覆盖，如 cuda:0 / cpu")
    return parser.parse_args()


def resolve_checkpoint(model_name: str, cfg: Config, user_ckpt: str) -> str:
    if user_ckpt:
        return user_ckpt
    if model_name == "srcnn":
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "srcnn_x4")
    if model_name == "basicvsr":
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "basicvsr_x4")
    if model_name in {"basicvsr_pp", "basicvsr_plusplus"}:
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "basicvsr_plusplus_x4")
    if model_name == "scst":
        key = "scst_localatten_unet"
        fallback = os.path.join(cfg.weights.scst_root_dir, os.path.basename(cfg.weights.urls[key]))
        try:
            return ensure_weights(cfg.weights.scst_root_dir, cfg.weights.urls, key)
        except Exception as exc:
            print(f"[warn] SCST checkpoint unavailable ({exc}); using {fallback} (may bicubic-fallback)")
            return fallback
    if model_name in {"realesrgan", "real_esrgan"}:
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "realesrgan_x4plus")
    if model_name in {"realesrnet", "real_esrnet"}:
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "realesrnet_x4plus")
    return ""


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    alias = {
        "real_esrgan": "realesrgan",
        "real_esrnet": "realesrnet",
        "basicvsr_plusplus": "basicvsr_pp",
    }
    normalized_name = alias.get(args.model_name, args.model_name)
    bm_kw = {}
    if normalized_name == "scst":
        bm_kw["scst_ckpt_root"] = cfg.weights.scst_root_dir
    model = build_model(normalized_name, scale=cfg.runtime.scale, **bm_kw).to(device).eval()
    ckpt_path = resolve_checkpoint(normalized_name, cfg, args.checkpoint)
    if ckpt_path and hasattr(model, "load_checkpoint"):
        print(f"[weights] loading: {ckpt_path}")
        model.load_checkpoint(ckpt_path)
    elif ckpt_path and hasattr(model, "load_openmmlab_checkpoint"):
        print(f"[weights] loading: {ckpt_path}")
        model.load_openmmlab_checkpoint(ckpt_path)

    lr = load_frames(args.input_dir).to(device)
    with torch.no_grad():
        sr = model(lr).clamp(0.0, 1.0)
        if sr.dim() == 4:
            sr = sr.unsqueeze(0)

    save_frames(sr, args.output_dir)
    print(f"[done] {normalized_name}: {args.input_dir} -> {args.output_dir}")


if __name__ == "__main__":
    main()
