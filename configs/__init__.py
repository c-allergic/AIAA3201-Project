"""Minimal configuration loader for inference-first VSR pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import yaml


def weights_base_dir(yaml_abs_path: str) -> str:
    """权重相对路径默认相对仓库根：若配置在 <repo>/configs/*.yaml 则取 <repo>，否则取 yaml 所在目录。"""
    d = os.path.dirname(os.path.abspath(yaml_abs_path))
    return os.path.dirname(d) if os.path.basename(d) == "configs" else d


def resolve_repo_path(yaml_abs_path: str, path: str | None, default: str) -> str:
    raw = default if path in (None, "") else path
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    base = weights_base_dir(yaml_abs_path)
    return os.path.normpath(os.path.join(base, raw))


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    scale: int = 4
    tile_size: int = 0
    tile_pad: int = 10


@dataclass
class WeightsConfig:
    root_dir: str = "./checkpoints/pretrained"
    urls: Dict[str, str] = field(default_factory=dict)
    scst_root_dir: str = "./checkpoints/scst"
    repos: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class Config:
    model_name: str = "bicubic"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        yaml_abs = os.path.abspath(yaml_path)
        with open(yaml_abs, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        runtime = RuntimeConfig(**raw.get("runtime", {}))
        wraw = dict(raw.get("weights", {}))
        wraw["root_dir"] = resolve_repo_path(
            yaml_abs, wraw.get("root_dir"), "./checkpoints/pretrained"
        )
        wraw["scst_root_dir"] = resolve_repo_path(
            yaml_abs, wraw.get("scst_root_dir"), "./checkpoints/scst"
        )
        weights = WeightsConfig(**wraw)
        return cls(
            model_name=raw.get("model_name", "bicubic"),
            runtime=runtime,
            weights=weights,
        )
