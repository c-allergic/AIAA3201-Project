"""Model factory: Part1/2 baselines + Part2 SOTA + Part3 wrappers (BasicVSR++, SCST)."""

from __future__ import annotations

from typing import Optional

from models.baselines import InterpolationBaseline, SRCNN
from models.basicvsr import BasicVSRWrapper
from models.basicvsr_pp import BasicVSRPlusPlusWrapper
from models.real_esrgan import RealESRGANVideo, RealESRNetVideo
from models.scst_wrapper import SCSTVideoWrapper


def build_model(model_name: str, scale: int = 4, scst_ckpt_root: Optional[str] = None):
    name = model_name.lower()
    if name == "bicubic":
        return InterpolationBaseline(scale=scale, mode="bicubic")
    if name == "srcnn":
        return SRCNN(scale=scale)
    if name == "basicvsr":
        return BasicVSRWrapper(scale=scale)
    if name in {"basicvsr_pp", "basicvsr_plusplus"}:
        return BasicVSRPlusPlusWrapper(scale=scale)
    if name == "scst":
        kw = {"scale": scale}
        if scst_ckpt_root is not None:
            kw["scst_ckpt_root"] = scst_ckpt_root
        return SCSTVideoWrapper(**kw)
    if name == "realesrgan":
        return RealESRGANVideo(scale=scale)
    if name == "realesrnet":
        return RealESRNetVideo(scale=scale)
    raise ValueError(f"Unknown model_name: {model_name}")


__all__ = [
    "InterpolationBaseline",
    "SRCNN",
    "BasicVSRWrapper",
    "BasicVSRPlusPlusWrapper",
    "RealESRGANVideo",
    "RealESRNetVideo",
    "SCSTVideoWrapper",
    "build_model",
]
