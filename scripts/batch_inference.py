#!/usr/bin/env python3
"""Batch inference for large video sequences to avoid OOM."""
import argparse, os, sys, shutil, tempfile, subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    frames = sorted([f for f in os.listdir(args.input_dir) if f.endswith(('.png','.jpg','.jpeg'))])
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(0, len(frames), args.batch_size):
        batch_frames = frames[i:i + args.batch_size]
        tmp_dir = tempfile.mkdtemp(prefix="vsr_batch_")
        for f in batch_frames:
            shutil.copy2(os.path.join(args.input_dir, f), os.path.join(tmp_dir, f))

        tmp_out = tempfile.mkdtemp(prefix="vsr_batch_out_")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [
            sys.executable, os.path.join(repo_root, "scripts", "inference.py"),
            "--model_name", args.model_name,
            "--input_dir", tmp_dir,
            "--output_dir", tmp_out,
            "--device", args.device,
        ]
        print(f"[batch {i//args.batch_size + 1}] {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=repo_root)

        for f in os.listdir(tmp_out):
            shutil.move(os.path.join(tmp_out, f), os.path.join(args.output_dir, f))

        shutil.rmtree(tmp_dir)
        shutil.rmtree(tmp_out)

    print(f"[done] {len(frames)} frames -> {args.output_dir}")

if __name__ == "__main__":
    main()
