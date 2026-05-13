#!/usr/bin/env python3
"""
一键跑通 Part1/2 推理、Part3 推理（B_only / C_hybrid）与 eval_pipeline 汇总评估。

默认数据布局（与 PROJECT_PROGRESS.md 一致）：
  --lr_root/BIx4/<seq>/*.png   低分辨率输入帧
  --gt_root/<seq>/*.png        对齐的真值帧

结果目录：
  results/bi_<seq>_<model>
  results/part3_<seq>_B_only
  results/part3_<seq>_C_hybrid
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SEQUENCES = ("calendar", "city", "foliage", "walk")
PART12_MODELS = ("bicubic", "srcnn", "basicvsr", "realesrgan", "realesrnet")


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd or _REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full VSR benchmark: Part1/2 + Part3 + eval.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--lr_root", type=str, default="data/BIx4", help="LR frames root (contains seq subdirs)")
    p.add_argument("--gt_root", type=str, default="data/GT")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0 or cpu; forwarded to scripts")
    p.add_argument("--crop_border", type=int, default=4)
    p.add_argument(
        "--skip_fid",
        action="store_true",
        help="Forward to eval_pipeline: skip FID (no Inception download).",
    )
    p.add_argument(
        "--skip_temporal_flow",
        action="store_true",
        help="Forward to eval_pipeline: skip Farneback optical-flow metrics.",
    )
    p.add_argument("--save_json", type=str, default="results/eval_full_benchmark.json")
    p.add_argument("--fusion_ckpt", type=str, default="", help="Optional fusion CNN for C_hybrid")
    p.add_argument("--fusion_gen_scale", type=float, default=1.0, help="Part3 C_hybrid: SCST weight scale (see inference_part3.py).")
    p.add_argument("--temporal_blend", type=float, default=0.24, help="Part3 temporal_refine blend.")
    p.add_argument("--rule_zeta", type=float, default=2.5, help="Part3 C_hybrid: penalize SCST temporal flicker in fusion rule.")
    p.add_argument("--skip_temporal_refine", action="store_true")
    p.add_argument("--use_stcm", action="store_true", help="Part3 SCST STCM branch instead of LocalAttention")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--phase", choices=("all", "download", "part12", "part3", "eval"), default="all")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def _py() -> str:
    return sys.executable


def _device_args(device: str) -> list[str]:
    return ["--device", device] if device else []


def main() -> None:
    args = parse_args()
    os.chdir(_REPO_ROOT)

    def safe_run(cmd: list[str]) -> None:
        try:
            _run(cmd)
        except subprocess.CalledProcessError as exc:
            if args.continue_on_error:
                print(f"[warn] command failed (exit {exc.returncode}), continuing.", flush=True)
            else:
                raise

    if args.phase in ("all", "download") and not args.skip_download:
        safe_run([_py(), "scripts/download_pretrained.py", "--config", args.config])

    if args.phase in ("all", "part12"):
        for seq in args.sequences:
            in_dir = os.path.join(args.lr_root, seq)
            if not os.path.isdir(in_dir):
                msg = f"missing LR dir: {in_dir}"
                if args.continue_on_error:
                    print(f"[warn] {msg}", flush=True)
                    continue
                raise FileNotFoundError(msg)
            for model in PART12_MODELS:
                out_dir = os.path.join(args.results_root, f"bi_{seq}_{model}")
                cmd = [
                    _py(),
                    "scripts/inference.py",
                    "--config",
                    args.config,
                    "--model_name",
                    model,
                    "--input_dir",
                    in_dir,
                    "--output_dir",
                    out_dir,
                ]
                cmd.extend(_device_args(args.device))
                safe_run(cmd)

    if args.phase in ("all", "part3"):
        for seq in args.sequences:
            in_dir = os.path.join(args.lr_root, seq)
            if not os.path.isdir(in_dir):
                if args.continue_on_error:
                    print(f"[warn] skip part3, missing {in_dir}", flush=True)
                    continue
                raise FileNotFoundError(f"missing LR dir: {in_dir}")
            for mode in ("B_only", "C_hybrid"):
                out_dir = os.path.join(args.results_root, f"part3_{seq}_{mode}")
                cmd = [
                    _py(),
                    "scripts/inference_part3.py",
                    "--config",
                    args.config,
                    "--mode",
                    mode,
                    "--input_dir",
                    in_dir,
                    "--output_dir",
                    out_dir,
                ]
                cmd.extend(_device_args(args.device))
                if args.fusion_ckpt and mode == "C_hybrid":
                    cmd.extend(["--fusion_ckpt", args.fusion_ckpt])
                if args.skip_temporal_refine:
                    cmd.append("--skip_temporal_refine")
                if args.use_stcm:
                    cmd.append("--use_stcm")
                cmd.extend(["--fusion_gen_scale", str(args.fusion_gen_scale)])
                cmd.extend(["--temporal_blend", str(args.temporal_blend)])
                cmd.extend(["--rule_zeta", str(args.rule_zeta)])
                safe_run(cmd)

    if args.phase in ("all", "eval"):
        methods: list[str] = []
        for m in PART12_MODELS:
            methods.append(f"bi_{{seq}}_{m}")
        methods.extend(["part3_{seq}_B_only", "part3_{seq}_C_hybrid"])
        cmd = [
            _py(),
            "scripts/eval_pipeline.py",
            "--gt_root",
            args.gt_root,
            "--results_root",
            args.results_root,
            "--methods",
            *methods,
            "--sequences",
            *args.sequences,
            "--crop_border",
            str(args.crop_border),
            "--save_json",
            args.save_json,
        ]
        cmd.extend(_device_args(args.device))
        if args.skip_fid:
            cmd.append("--skip_fid")
        if args.skip_temporal_flow:
            cmd.append("--skip_temporal_flow")
        safe_run(cmd)

    print("[done] run_full_benchmark phase=", args.phase, flush=True)


if __name__ == "__main__":
    main()
