#!/usr/bin/env python3
"""Ablation: temporal refinement on/off."""
import sys; sys.path.insert(0, ".")
import torch
from models.uncertainty_fusion import rule_based_weight, fuse_with_weight
from utils import load_frames, save_frames

device = torch.device("cuda:0")
sr_fid = load_frames("results/bi_city_basicvsr").to(device)
sr_gen = load_frames("results/part3_city_B_only").to(device)

with torch.no_grad():
    rb_kw = dict(alpha=4.0, beta=6.0, gamma=3.0, zeta=2.5)
    w, _ = rule_based_weight(sr_fid, sr_gen, **rb_kw)
    w = torch.clamp(w * 0.3, 0.0, 1.0)
    out = fuse_with_weight(sr_fid, sr_gen, w)
    out_dir = "results/part3_city_C_hybridg0.3_noTR"
    save_frames(out, out_dir)
    print("Saved:", out_dir)
