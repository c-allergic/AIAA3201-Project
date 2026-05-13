"""Part 1 baselines with stable inference behavior.

Bicubic / Lanczos / SRCNN / Temporal-Average + Unsharp Masking.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lanczos_resize(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Lanczos3 interpolation via PIL (per-frame, 4D NCHW input).

    Uses PIL.Image.resize(..., LANCZOS) which is widely available and
    does not depend on torch F.interpolate lanczos support.
    """
    from PIL import Image

    b, c, h, w = x.shape
    out_h, out_w = h * scale, w * scale
    # Convert to numpy HWC uint8 for PIL
    arr = x.detach().clamp(0.0, 1.0).cpu().numpy()  # B, C, H, W
    arr = (arr * 255.0).round().astype(np.uint8)
    out = np.empty((b, c, out_h, out_w), dtype=np.float32)
    for i in range(b):
        img = Image.fromarray(arr[i].transpose(1, 2, 0), mode="RGB")
        img = img.resize((out_w, out_h), Image.LANCZOS)
        out[i] = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(out).to(x.device)


class InterpolationBaseline(nn.Module):
    """Per-frame interpolation baseline (bicubic / lanczos)."""

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
        flat = lr_frames.reshape(b * t, c, h, w)

        if self.mode == "lanczos":
            up = _lanczos_resize(flat, self.scale)
        else:
            up = F.interpolate(
                flat,
                scale_factor=self.scale,
                mode=self.mode,
                align_corners=False if self.mode in {"bicubic", "bilinear"} else None,
            )

        out = up.reshape(b, t, c, up.shape[-2], up.shape[-1]).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out


class TemporalAverageBaseline(nn.Module):
    """Multi-frame averaging baseline with unsharp masking.

    1. Upsample each LR frame individually (bicubic or lanczos).
    2. Gaussian-weighted temporal average over neighbouring frames.
    3. Unsharp masking: sharpen = avg + amount * (avg - blur(avg)).
    """

    def __init__(self, scale: int = 4, mode: str = "bicubic",
                 temporal_radius: int = 2, temporal_sigma: float = 1.0,
                 unsharp_amount: float = 0.8, unsharp_sigma: float = 1.5):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.temporal_radius = temporal_radius
        self.temporal_sigma = temporal_sigma
        self.unsharp_amount = unsharp_amount
        self.unsharp_sigma = unsharp_sigma

    def _upsample(self, lr_frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = lr_frames.shape
        flat = lr_frames.reshape(b * t, c, h, w)
        if self.mode == "lanczos":
            up = _lanczos_resize(flat, self.scale)
        else:
            up = F.interpolate(flat, scale_factor=self.scale, mode=self.mode,
                               align_corners=False)
        return up.reshape(b, t, c, up.shape[-2], up.shape[-1])

    def _temporal_average(self, sr: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = sr.shape
        r = self.temporal_radius

        avg = []
        for t_idx in range(t):
            t0 = max(0, t_idx - r)
            t1 = min(t, t_idx + r + 1)
            k = t1 - t0
            # Compute Gaussian weights directly for this window
            offsets = torch.arange(t0, t1, dtype=torch.float32, device=sr.device) - t_idx
            g = torch.exp(-0.5 * (offsets / self.temporal_sigma) ** 2)
            w = (g / g.sum()).view(1, k, 1, 1, 1)
            window = sr[:, t0:t1, :, :, :]
            avg_t = (window * w).sum(dim=1, keepdim=True)
            avg.append(avg_t)
        return torch.cat(avg, dim=1).clamp(0.0, 1.0)

    def _gaussian_kernel(self) -> torch.Tensor:
        r = self.temporal_radius
        xs = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-0.5 * (xs / self.temporal_sigma) ** 2)
        return g / g.sum()

    def _unsharp_mask(self, avg: torch.Tensor) -> torch.Tensor:
        """Per-frame unsharp masking on the temporally averaged frames."""
        b, t, c, h, w = avg.shape
        flat = avg.reshape(b * t, c, h, w)
        ksize = max(3, int(4 * self.unsharp_sigma) | 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred = _gaussian_blur_2d(flat, ksize, self.unsharp_sigma)
        sharp = flat + self.unsharp_amount * (flat - blurred)
        return sharp.reshape(b, t, c, h, w).clamp(0.0, 1.0)

    def forward(self, lr_frames: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if lr_frames.dim() == 4:
            lr_frames = lr_frames.unsqueeze(0)
            squeeze = True
        sr = self._upsample(lr_frames)
        avg = self._temporal_average(sr)
        out = self._unsharp_mask(avg)
        return out.squeeze(0) if squeeze else out


def _gaussian_blur_2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """2D Gaussian blur with 'reflect' padding (channel-wise)."""
    ax = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
    ax -= kernel_size // 2
    g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1d /= g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    kernel = g2d.expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(F.pad(x, (kernel_size // 2,) * 4, mode="reflect"),
                    kernel, groups=x.shape[1])


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
                nk = nk[len("generator."):]
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
