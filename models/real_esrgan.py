"""Part 2 Real-ESRGAN wrapper with official weights."""

import torch
import torch.nn as nn
import sys
import types
import os


def _patch_torchvision_compat() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as tvf

        patched = types.ModuleType(module_name)
        patched.rgb_to_grayscale = tvf.rgb_to_grayscale
        sys.modules[module_name] = patched


def _patch_basicsr_fallback() -> None:
    try:
        import basicsr  # type: ignore # noqa: F401
        return
    except Exception:
        pass
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, "third_party", "SCST")
    if os.path.isdir(os.path.join(candidate, "basicsr")) and candidate not in sys.path:
        sys.path.insert(0, candidate)


class RealESRGANVideo(nn.Module):
    """Frame-wise Real-ESRGAN x4plus."""

    def __init__(self, scale: int = 4):
        super().__init__()
        if scale != 4:
            raise ValueError("当前仅支持 x4 Real-ESRGAN 预训练权重。")
        _patch_torchvision_compat()
        _patch_basicsr_fallback()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as exc:
            raise ImportError(
                "未安装 basicsr。请先执行 `pip install --no-deps --no-build-isolation basicsr==1.4.2`。"
            ) from exc
        self.net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

    def load_checkpoint(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu")
        state = (
            raw.get("params_ema")
            or raw.get("params")
            or raw.get("state_dict")
            or raw
        )
        cleaned = {}
        for k, v in state.items():
            nk = k[len("generator.") :] if k.startswith("generator.") else k
            cleaned[nk] = v
        self.net.load_state_dict(cleaned, strict=True)

    def forward(self, lr_frames: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if lr_frames.dim() == 4:
            lr_frames = lr_frames.unsqueeze(0)
            squeeze = True
        b, t, c, h, w = lr_frames.shape
        flat = lr_frames.reshape(b * t, c, h, w)
        # Process one frame at a time to avoid OOM on long sequences
        sr_list = []
        for i in range(flat.shape[0]):
            sr_list.append(self.net(flat[i:i+1]).clamp(0.0, 1.0))
        sr = torch.cat(sr_list, dim=0)
        sr = sr.reshape(b, t, 3, sr.shape[-2], sr.shape[-1])
        return sr.squeeze(0) if squeeze else sr


class RealESRNetVideo(nn.Module):
    """Frame-wise RealESRNet x4plus (non-GAN counterpart)."""

    def __init__(self, scale: int = 4):
        super().__init__()
        if scale != 4:
            raise ValueError("当前仅支持 x4 RealESRNet 预训练权重。")
        _patch_torchvision_compat()
        _patch_basicsr_fallback()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as exc:
            raise ImportError(
                "未安装 basicsr。请先执行 `pip install --no-deps --no-build-isolation basicsr==1.4.2`。"
            ) from exc
        self.net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

    def load_checkpoint(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu")
        state = (
            raw.get("params")
            or raw.get("params_ema")
            or raw.get("state_dict")
            or raw
        )
        cleaned = {}
        for k, v in state.items():
            nk = k[len("generator.") :] if k.startswith("generator.") else k
            cleaned[nk] = v
        self.net.load_state_dict(cleaned, strict=True)

    def forward(self, lr_frames: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if lr_frames.dim() == 4:
            lr_frames = lr_frames.unsqueeze(0)
            squeeze = True
        b, t, c, h, w = lr_frames.shape
        flat = lr_frames.reshape(b * t, c, h, w)
        # Process one frame at a time to avoid OOM on long sequences
        sr_list = []
        for i in range(flat.shape[0]):
            sr_list.append(self.net(flat[i:i+1]).clamp(0.0, 1.0))
        sr = torch.cat(sr_list, dim=0)
        sr = sr.reshape(b, t, 3, sr.shape[-2], sr.shape[-1])
        return sr.squeeze(0) if squeeze else sr
