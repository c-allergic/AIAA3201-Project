#!/usr/bin/env python3
"""Run P1/P2 baselines on wild2: bicubic, BasicVSR++, Real-ESRGAN."""
import os, sys
sys.path.insert(0, '/home/user/VSR_Project')
os.chdir('/home/user/VSR_Project')

import torch
import numpy as np
from PIL import Image
from configs import Config
from models import build_model
from utils import ensure_weights, load_frames, save_frames

INPUT_DIR = 'data/wild2_lr_frames'
OUTPUT_BASE = 'results/wild2'

cfg = Config.from_yaml('configs/default.yaml')
device = torch.device('cuda:0')
root = cfg.weights.root_dir

# (model_name_for_build, checkpoint_key, output_suffix)
tasks = [
    ('bicubic', None, 'bicubic'),
    ('basicvsr_pp', 'basicvsr_plusplus_x4', 'basicvsr_pp'),
    ('realesrgan', 'realesrgan_x4plus', 'realesrgan'),
]

for model_name, ckpt_key, out_name in tasks:
    out_dir = f'{OUTPUT_BASE}_{out_name}'
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    if existing >= 100:
        print(f'[{out_name}] Already done ({existing} frames), skipping')
        continue

    print(f'[{out_name}] Loading model...')
    model = build_model(model_name, scale=cfg.runtime.scale).to(device).eval()
    if ckpt_key:
        ckpt = ensure_weights(root, cfg.weights.urls, ckpt_key)
        model.load_checkpoint(ckpt)

    lr = load_frames(INPUT_DIR).to(device)
    print(f'[{out_name}] LR shape: {lr.shape}')

    with torch.no_grad():
        sr = model(lr).clamp(0, 1)

    print(f'[{out_name}] SR shape: {sr.shape}')
    save_frames(sr, out_dir)
    del model, lr, sr
    torch.cuda.empty_cache()
    print(f'[{out_name}] Done -> {out_dir}')

print('All baselines done.')
