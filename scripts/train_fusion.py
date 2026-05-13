#!/usr/bin/env python3
"""Train lightweight fusion CNN on precomputed branch outputs."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from models.uncertainty_fusion import FusionWeightCNN, build_fusion_training_input, compute_uncertainty_features, fuse_with_weight
from utils import load_frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fid_dir", type=str, required=True, help="BasicVSR++ output frame dir")
    p.add_argument("--gen_dir", type=str, required=True, help="SCST output frame dir")
    p.add_argument("--gt_dir", type=str, required=True, help="GT frame dir")
    p.add_argument("--save_path", type=str, default="checkpoints/pretrained/fusion_cnn.pth")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Only use first T frames (speed). 0 = use all aligned frames.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    sr_fid = load_frames(args.fid_dir).to(device)
    sr_gen = load_frames(args.gen_dir).to(device)
    gt = load_frames(args.gt_dir).to(device)
    n = min(sr_fid.shape[1], sr_gen.shape[1], gt.shape[1])
    if args.max_frames and args.max_frames > 0:
        n = min(n, args.max_frames)
    sr_fid, sr_gen, gt = sr_fid[:, :n], sr_gen[:, :n], gt[:, :n]
    print(f"[train_fusion] frames={n} spatial={tuple(sr_fid.shape[-2:])} device={device}", flush=True)

    model = FusionWeightCNN().to(device)
    l1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        feat = compute_uncertainty_features(sr_fid, sr_gen)
        x = build_fusion_training_input(feat, sr_fid, sr_gen)
        b, t, c, h, w = x.shape
        weight = model(x.reshape(b * t, c, h, w)).reshape(b, t, 1, h, w)
        fused = fuse_with_weight(sr_fid, sr_gen, weight)
        loss = l1(fused, gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"[step {step:04d}] loss={loss.item():.6f}", flush=True)

    Path(os.path.dirname(args.save_path) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.save_path)
    print(f"[done] saved: {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
