#!/usr/bin/env python3
"""Run Part 3 C_hybrid on Wild V2 (240p)."""
import os, sys

_repo_root = '/home/user/VSR_Project'
os.chdir(_repo_root)
sys.path.insert(0, _repo_root)

import subprocess, tempfile, time

INPUT_DIR = "/home/user/VSR_Project/data/wild2_lr_frames"
OUTPUT_DIR = "/home/user/VSR_Project/results/part3_wild2_C_hybrid_g0.3"

# Check input
if not os.path.isdir(INPUT_DIR):
    print(f"ERROR: Input dir not found: {INPUT_DIR}")
    frames = sorted([f for f in os.listdir("/home/user/VSR_Project/data") if 'wild2' in f.lower()])
    print(f"  wild2 candidates in data/: {frames}")
    import glob
    alt = glob.glob("/home/user/VSR_Project/**/wild2*", recursive=True)
    print(f"  wild2 anywhere: {alt[:5]}")
    sys.exit(1)

n_frames = len([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
print(f"Input: {n_frames} frames in {INPUT_DIR}")

# Run chunked inference
cmd = [
    "/home/user/.conda/envs/vsr_part3/bin/python",
    "/tmp/inference_part3_chunked.py",
    "--input_dir", INPUT_DIR,
    "--output_dir", OUTPUT_DIR,
    "--chunk_size", "8",
    "--overlap", "2",
    "--fusion_gen_scale", "0.3",
    "--temporal_blend", "0.24",
    "--scst_steps", "20",
    "--scst_guidance", "5.0",
]

print(f"Running: {' '.join(cmd)}")
print(f"Log: /tmp/rerun_wild2_part3.log")

t0 = time.time()
with open("/tmp/rerun_wild2_part3.log", "w") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    proc.wait()

elapsed = time.time() - t0
print(f"Exit code: {proc.returncode}")
print(f"Elapsed: {elapsed/60:.1f} min")

if proc.returncode == 0:
    out_frames = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    print(f"Output frames: {out_frames}")
else:
    print("FAILED - check /tmp/rerun_wild2_part3.log")
