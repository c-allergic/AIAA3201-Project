#!/usr/bin/env python3
"""Wild video eval - sample frames to avoid OOM (SR frames are 5120x2880)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
from PIL import Image

METHODS = [
    ("wild_bicubic", "Bicubic"),
    ("wild_lanczos", "Lanczos"),
    ("wild_temporal_avg", "Temporal Avg"),
    ("wild_srcnn", "SRCNN"),
    ("wild_basicvsr", "BasicVSR"),
    ("wild_realesrgan", "Real-ESRGAN"),
    ("wild_realesrnet", "Real-ESRNet"),
]

results_root = Path("results")
report = {}

for d, name in METHODS:
    p = results_root / d
    if not p.exists():
        print(f"{name:20s}: MISSING")
        continue
    frames = sorted(list(p.glob("*.png")))
    if not frames:
        print(f"{name:20s}: empty")
        continue

    # Sample every 10th frame to avoid OOM
    sample_frames = frames[::10]
    if len(sample_frames) < 3:
        sample_frames = frames[:3]

    imgs = []
    for f in sample_frames:
        arr = np.array(Image.open(f), dtype=np.float32) / 255.0
        imgs.append(arr)
    stack = np.stack(imgs)

    report[name] = {
        "num_frames": len(frames),
        "resolution": f"{stack.shape[2]}x{stack.shape[1]}",
        "mean": float(stack.mean()),
        "std": float(stack.std()),
    }
    print(f"{name:20s}: {len(frames)} frames {stack.shape[2]}x{stack.shape[1]}, "
          f"mean={stack.mean():.4f} std={stack.std():.4f}")

with open("results/wild_summary.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved wild_summary.json ({len(report)} methods)")
