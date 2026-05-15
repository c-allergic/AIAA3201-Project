#!/usr/bin/env python3
"""Generate comparison figure: 3 methods x 6 frames from a chosen sequence."""
import os, sys
import numpy as np
from PIL import Image

# --- Config ---
BASE = "/home/user/VSR_Project/results"
SEQUENCE = "city"  # Vid4 BIx4 city
FRAME_COUNT = 6
FRAME_START = 8
FRAME_STEP = 5  # frames: 8, 13, 18, 23, 28, 33 (34 total)

METHODS = [
    ("Bicubic", f"bi_{SEQUENCE}_bicubic"),
    ("BasicVSR", f"bi_{SEQUENCE}_basicvsr"),
    ("Real-ESRGAN", f"bi_{SEQUENCE}_realesrgan"),
]

OUTPUT = f"/home/user/VSR_Project/figures/frame_comparison_{SEQUENCE}.png"

# --- Load frames ---
method_frames = {}
for label, dirname in METHODS:
    d = os.path.join(BASE, dirname)
    if not os.path.isdir(d):
        print(f"MISSING: {d}")
        sys.exit(1)
    files = sorted([f for f in os.listdir(d) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"[{label}] {len(files)} frames in {dirname}")

    selected = []
    for i in range(FRAME_COUNT):
        idx = min(FRAME_START + i * FRAME_STEP, len(files) - 1)
        path = os.path.join(d, files[idx])
        img = np.array(Image.open(path).convert('RGB'))
        selected.append(img)
        print(f"  frame {idx}: {files[idx]}, shape={img.shape}")
    method_frames[label] = selected

# --- Build comparison grid ---
# Layout: 3 rows (methods) x FRAME_COUNT columns
# Each cell: clip to a consistent crop region
H, W = method_frames["Bicubic"][0].shape[:2]
CROP_H = min(320, H)
CROP_W = min(512, W)
crop_t = (H - CROP_H) // 2
crop_l = (W - CROP_W) // 2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(METHODS), FRAME_COUNT, figsize=(FRAME_COUNT * 2.2, len(METHODS) * 2.2))

for row_idx, (label, frames) in enumerate(method_frames.items()):
    for col_idx, frame in enumerate(frames):
        ax = axes[row_idx, col_idx]
        crop = frame[crop_t:crop_t + CROP_H, crop_l:crop_l + CROP_W]
        ax.imshow(crop)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(f"t={FRAME_START + col_idx * FRAME_STEP}", fontsize=9)
        if col_idx == 0:
            ax.text(-0.3, 0.5, label, transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='center', ha='right', rotation=0)

plt.subplots_adjust(wspace=0.02, hspace=0.02)
fig.savefig(OUTPUT, dpi=250, bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"\nSaved: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT) / 1024:.0f} KB")
