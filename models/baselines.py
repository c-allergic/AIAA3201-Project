"""Part 1 baselines with stable inference behavior."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolationBaseline(nn.Module):
    """Per-frame interpolation baseline."""

    def __init__(self, scale: int = 4, mode: str = "bicubic"):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, lr_frames: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if lr_frames.dim() == 4:
            lr_frames = lr_frames.unsqueeze(0)
            squeeze = True
        b, t, c, h, w = lr_frames.shape
        up = F.interpolate(
            lr_frames.reshape(b * t, c, h, w),
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=False if self.mode in {"bicubic", "bilinear"} else None,
        )
        out = up.reshape(b, t, c, up.shape[-2], up.shape[-1]).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out


class SRCNN(nn.Module):
    """Classic 3-layer SRCNN for x4 after bicubic pre-upsample."""

    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def load_openmmlab_checkpoint(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("state_dict", raw)
        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("generator."):
                nk = nk[len("generator.") :]
            cleaned[nk] = v
        self.load_state_dict(cleaned, strict=True)

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x.clamp(0.0, 1.0)

    def forward(self, lr_frames: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if lr_frames.dim() == 4:
            lr_frames = lr_frames.unsqueeze(0)
            squeeze = True
        b, t, c, h, w = lr_frames.shape
        sr = self._forward_4d(lr_frames.reshape(b * t, c, h, w))
        sr = sr.reshape(b, t, 3, sr.shape[-2], sr.shape[-1])
        return sr.squeeze(0) if squeeze else sr
