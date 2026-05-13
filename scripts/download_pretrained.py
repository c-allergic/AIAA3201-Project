#!/usr/bin/env python3
"""Download official open-source weights used by this project."""

import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from configs import Config
from utils import ensure_weights


def ensure_hf_repo_snapshot(repo_id: str, local_dir: str, allow_patterns=None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Please install huggingface_hub to download repository snapshots.") from exc

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    return local_dir


def _flatten_scst_controlnet_layout(scst_root: str) -> None:
    """MochunniaN1/SCST 上 ControlNet 在 checkpoints/controlnet/；推理需要 scst_root/controlnet/。"""
    import shutil

    nested = os.path.join(scst_root, "checkpoints", "controlnet")
    target = os.path.join(scst_root, "controlnet")
    if os.path.isdir(nested) and not os.path.isdir(target):
        shutil.move(nested, target)
    # 若仅剩空的 checkpoints 目录可删（避免误导）
    leftover = os.path.join(scst_root, "checkpoints")
    if os.path.isdir(leftover) and not os.listdir(leftover):
        os.rmdir(leftover)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "srcnn",
            "basicvsr",
            "basicvsr_plusplus",
            "realesrgan",
            "realesrnet",
            "scst",
            "sd21_base",
            "scst_controlnet",
        ],
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    key_map = {
        "srcnn": "srcnn_x4",
        "basicvsr": "basicvsr_x4",
        "basicvsr_plusplus": "basicvsr_plusplus_x4",
        "realesrgan": "realesrgan_x4plus",
        "realesrnet": "realesrnet_x4plus",
        "scst_localatten": "scst_localatten_unet",
        "scst_mococtrl": "scst_mococtrl_unet",
        "scst_stcm": "scst_stcm_unet",
    }
    for model in args.models:
        if model == "scst":
            for key in ("scst_localatten", "scst_mococtrl", "scst_stcm"):
                try:
                    path = ensure_weights(cfg.weights.scst_root_dir, cfg.weights.urls, key_map[key])
                    print(f"[ok] {key}: {path}")
                except Exception as exc:  # pragma: no cover - network dependent
                    print(f"[warn] {key}: {exc}")
            continue
        if model in {"sd21_base", "scst_controlnet"}:
            if model not in cfg.weights.repos:
                print(f"[skip] missing repo config for: {model}")
                continue
            repo_cfg = cfg.weights.repos[model]
            # Hub 上路径为 controlnet/...；local_dir 必须是 scst 根目录，否则会落成 .../controlnet/controlnet/ 且 allow_patterns 常匹配为 0。
            if model == "scst_controlnet":
                local_dir = cfg.weights.scst_root_dir
            else:
                local_dir = os.path.join(cfg.weights.scst_root_dir, repo_cfg["local_subdir"])
            try:
                path = ensure_hf_repo_snapshot(
                    repo_id=repo_cfg["repo_id"],
                    local_dir=local_dir,
                    allow_patterns=repo_cfg.get("allow_patterns"),
                )
                if model == "scst_controlnet":
                    _flatten_scst_controlnet_layout(cfg.weights.scst_root_dir)
                print(f"[ok] {model}: {path}")
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[warn] {model}: {exc}")
            continue
        if model not in key_map:
            print(f"[skip] unsupported model key: {model}")
            continue
        try:
            path = ensure_weights(cfg.weights.root_dir, cfg.weights.urls, key_map[model])
            print(f"[ok] {model}: {path}")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[warn] {model}: {exc}")


if __name__ == "__main__":
    main()
