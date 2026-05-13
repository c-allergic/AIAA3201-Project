#!/usr/bin/env python3
"""Wild video inference — loads model once, processes frames in small batches.

Usage:
  python scripts/wild_inference.py --model_name basicvsr --device cuda:0
"""

import argparse, os, sys

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from configs import Config
from models import build_model
from utils import ensure_weights, load_frames, save_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="data/wild_lr_frames")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device)

    alias = {
        "real_esrgan": "realesrgan", "real_esrnet": "realesrnet",
        "basicvsr_plusplus": "basicvsr_pp", "temporal_average": "temporal_avg",
    }
    name = alias.get(args.model_name, args.model_name)
    bm_kw = {}
    if name == "scst":
        bm_kw["scst_ckpt_root"] = cfg.weights.scst_root_dir

    model = build_model(name, scale=cfg.runtime.scale, **bm_kw).to(device).eval()

    # Load checkpoint if needed
    ckpt = args.checkpoint
    if not ckpt and name not in {"bicubic", "lanczos", "temporal_avg"}:
        ckpt = _resolve_checkpoint(name, cfg)
    if ckpt:
        print(f"[weights] loading: {ckpt}")
        if hasattr(model, "load_checkpoint"):
            model.load_checkpoint(ckpt)
        elif hasattr(model, "load_openmmlab_checkpoint"):
            model.load_openmmlab_checkpoint(ckpt)

    # Load all frames
    import numpy as np
    from PIL import Image
    from pathlib import Path

    paths = sorted(Path(args.input_dir).glob("*.png")) + sorted(Path(args.input_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames in {args.input_dir}")

    all_frames = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        all_frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    all_frames = torch.stack(all_frames, dim=0)  # T,C,H,W

    output_dir = args.output_dir or f"results/wild_{name}"
    os.makedirs(output_dir, exist_ok=True)

    T = len(all_frames)
    B = args.batch_size
    saved_count = 0

    print(f"[{name}] {T} frames, batch_size={B}, device={device}")
    with torch.no_grad():
        for start in range(0, T, B):
            end = min(start + B, T)
            batch = all_frames[start:end].to(device)  # b,c,h,w

            try:
                sr = model(batch).clamp(0.0, 1.0)
            except torch.OutOfMemoryError:
                # fallback to single frame
                torch.cuda.empty_cache()
                print(f"  OOM at {start}-{end}, falling back to single frame")
                sr_list = []
                for i in range(start, end):
                    single = all_frames[i:i+1].to(device)
                    sr_list.append(model(single).clamp(0.0, 1.0).cpu())
                    torch.cuda.empty_cache()
                sr = torch.cat(sr_list, dim=0).to(device)

            if sr.dim() == 3:
                sr = sr.unsqueeze(0)

            for j in range(sr.shape[0]):
                img = sr[j].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255.0).round().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(output_dir, f"frame_{start + j:06d}.png"))
                saved_count += 1

            if (start // B) % 10 == 0:
                print(f"  [{saved_count}/{T}]")

    print(f"[done] {name}: {saved_count} frames -> {output_dir}")


CHECKPOINT_FREE = {"bicubic", "lanczos", "temporal_avg", "temporal_average"}


def _resolve_checkpoint(model_name: str, cfg: Config) -> str:
    if model_name in CHECKPOINT_FREE:
        return ""
    if model_name == "srcnn":
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "srcnn_x4")
    if model_name == "basicvsr":
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "basicvsr_x4")
    if model_name in {"basicvsr_pp", "basicvsr_plusplus"}:
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "basicvsr_plusplus_x4")
    if model_name in {"realesrgan", "real_esrgan"}:
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "realesrgan_x4plus")
    if model_name in {"realesrnet", "real_esrnet"}:
        return ensure_weights(cfg.weights.root_dir, cfg.weights.urls, "realesrnet_x4plus")
    return ""


if __name__ == "__main__":
    main()
