#!/usr/bin/env python3
"""STCM fusion sweep on city."""
import sys
sys.path.insert(0, ".")
import torch
from models.uncertainty_fusion import rule_based_weight, fuse_with_weight
from scripts.temporal_refine import temporal_refine
from utils import load_frames, save_frames

device = torch.device("cuda:0")
sr_fid = load_frames("results/bi_city_basicvsr").to(device)
sr_gen = load_frames("results/part3_city_B_only_stcm").to(device)
print(f"fid shape: {list(sr_fid.shape)}, gen(stcm) shape: {list(sr_gen.shape)}")

configs = [
    (0.1, 2.5, "stcm_g0.10"),
    (0.2, 2.5, "stcm_g0.20"),
    (0.3, 2.5, "stcm_g0.30"),
    (0.4, 2.5, "stcm_g0.40"),
    (0.5, 2.5, "stcm_g0.50"),
    (0.3, 4.0, "stcm_g0.30_z4"),
]

with torch.no_grad():
    for g, zeta, label in configs:
        out_dir = f"results/part3_city_C_hybrid{label}"
        rb_kw = dict(alpha=4.0, beta=6.0, gamma=3.0, zeta=zeta)
        w, _ = rule_based_weight(sr_fid, sr_gen, **rb_kw)
        w = torch.clamp(w * g, 0.0, 1.0)
        out = fuse_with_weight(sr_fid, sr_gen, w)
        out = temporal_refine(out, blend=0.24)
        save_frames(out, out_dir)
        print(f"  Saved: {out_dir}")
print("STCM fusion sweep done!")
