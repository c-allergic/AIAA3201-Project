#!/usr/bin/env python3
"""Part3 自检：配置路径解析、SCST 权重与目录、diffusers/hf/basicsr 关键依赖。

用法（在仓库根目录）：
  conda activate vsr_part3
  python scripts/check_part3_ready.py
  python scripts/check_part3_ready.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Part3 / SCST 环境与权重自检")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--skip-heavy-imports",
        action="store_true",
        help="跳过 torch/diffusers/basicsr 导入（仅检查路径与文件）",
    )
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    errors: list[str] = []
    warnings: list[str] = []

    from configs import Config, weights_base_dir

    if not os.path.isfile(cfg_path):
        print(f"[FAIL] 配置文件不存在: {cfg_path}")
        return 1

    cfg = Config.from_yaml(cfg_path)
    base = weights_base_dir(cfg_path)
    print(f"[ok] 配置: {cfg_path}")
    print(f"[ok] 权重基准目录: {base}")
    print(f"[ok] scst_root_dir -> {cfg.weights.scst_root_dir}")
    print(f"[ok] pretrained root_dir -> {cfg.weights.root_dir}")

    scst = cfg.weights.scst_root_dir
    checks = [
        (os.path.join(scst, "localatten_unet.pth"), "SCST LocalAttention UNet（B_only / 默认）"),
        (os.path.join(scst, "stable-diffusion-2-1-base"), "SD2.1 base 目录"),
        (os.path.join(scst, "controlnet"), "ControlNet 目录"),
    ]
    for path, label in checks:
        if os.path.isfile(path) or os.path.isdir(path):
            print(f"[ok] {label}: {path}")
        else:
            errors.append(f"缺失 {label}: {path}")

    stcm = os.path.join(scst, "stcm_unet.pth")
    if not os.path.isfile(stcm):
        warnings.append(f"可选（仅 --use_stcm）: {stcm}")

    bv_name = os.path.basename(cfg.weights.urls["basicvsr_plusplus_x4"])
    bv_path = os.path.join(cfg.weights.root_dir, bv_name)
    if not os.path.isfile(bv_path):
        warnings.append(f"C_hybrid 需要 BasicVSR++ 权重: {bv_path}")

    if args.skip_heavy_imports:
        print("[info] 已跳过重型 import 检查（--skip-heavy-imports）")
    else:
        try:
            import huggingface_hub as hh

            if not hasattr(hh, "cached_download"):
                errors.append(
                    "huggingface_hub 过新，diffusers 0.25 需要 cached_download；"
                    "请: pip install 'huggingface_hub>=0.19.4,<0.26'"
                )
            else:
                print("[ok] huggingface_hub 与 diffusers 0.25 兼容（含 cached_download）")
        except ImportError as e:
            errors.append(f"huggingface_hub: {e}")

        try:
            import diffusers  # noqa: F401

            print("[ok] diffusers 可导入")
        except ImportError as e:
            errors.append(f"diffusers: {e}")

        try:
            from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus  # noqa: F401

            print("[ok] basicsr BasicVSRPlusPlus 可导入")
        except ImportError as e:
            errors.append(f"basicsr BasicVSR++: {e}")

        try:
            import torch  # noqa: F401

            print("[ok] torch 可导入")
        except ImportError as e:
            errors.append(f"torch: {e}")

    # SCST 子进程参数：ckpt 必须为绝对路径（cwd=third_party/SCST）
    from models.scst_wrapper import SCSTVideoWrapper

    w = SCSTVideoWrapper(scale=4, scst_ckpt_root=cfg.weights.scst_root_dir)
    fake_ckpt = os.path.join(cfg.weights.scst_root_dir, "localatten_unet.pth")
    w.load_checkpoint(fake_ckpt)
    cmd = w._build_command("/tmp/scst_check_in", "/tmp/scst_check_out")
    i = cmd.index("--ckpt_model_path") + 1
    ckpt_arg = cmd[i]
    if not os.path.isabs(ckpt_arg):
        errors.append(f"SCST --ckpt_model_path 非绝对路径（会导致 FileNotFoundError）: {ckpt_arg}")
    else:
        print(f"[ok] SCST 子进程 ckpt 参数为绝对路径: {ckpt_arg}")

    pre_i = cmd.index("--pretrained_model_path") + 1
    ctrl_i = cmd.index("--controlnet_path") + 1
    if not os.path.isabs(cmd[pre_i]) or not os.path.isabs(cmd[ctrl_i]):
        errors.append("SCST pretrained/controlnet 路径应为绝对路径")

    for wmsg in warnings:
        print(f"[warn] {wmsg}")
    for err in errors:
        print(f"[FAIL] {err}")

    if errors:
        print("\n请先补齐权重或修正配置，再运行 scripts/inference_part3.py。")
        return 1
    print("\n[done] Part3 自检通过（路径与依赖就绪）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
