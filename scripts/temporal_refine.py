#!/usr/bin/env python3
"""Temporal refinement with optical-flow-guided smoothing."""

from typing import Optional

import torch
import torch.nn.functional as F


def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def _flow_to_norm_grid(flow: torch.Tensor) -> torch.Tensor:
    _, _, h, w = flow.shape
    base = _make_base_grid(h, w, flow.device, flow.dtype)
    norm_flow = torch.zeros_like(flow)
    norm_flow[:, 0] = flow[:, 0] * (2.0 / max(w - 1, 1))
    norm_flow[:, 1] = flow[:, 1] * (2.0 / max(h - 1, 1))
    return base + norm_flow.permute(0, 2, 3, 1)


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    grid = _flow_to_norm_grid(flow)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _load_spynet():
    from models.basicvsr import _patch_torchvision_compat

    _patch_torchvision_compat()
    from basicsr.archs.spynet_arch import SpyNet

    return SpyNet(load_path=None)


def temporal_refine(sr: torch.Tensor, spynet: Optional[torch.nn.Module] = None, blend: float = 0.25) -> torch.Tensor:
    """Refine temporal consistency by flow-aligned neighbor averaging."""
    if sr.dim() != 5:
        raise ValueError(f"Expected sr shape (B,T,C,H,W), got {tuple(sr.shape)}")
    if sr.shape[1] < 2:
        return sr
    if spynet is None:
        spynet = _load_spynet().to(sr.device).eval()

    b, t, c, h, w = sr.shape
    out = sr.clone()
    with torch.no_grad():
        for i in range(1, t - 1):
            prev = sr[:, i - 1]
            cur = sr[:, i]
            nxt = sr[:, i + 1]
            flow_prev_to_cur = spynet(prev, cur)
            flow_next_to_cur = spynet(nxt, cur)
            prev_warp = _warp(prev, flow_prev_to_cur)
            next_warp = _warp(nxt, flow_next_to_cur)
            neigh = 0.5 * (prev_warp + next_warp)
            out[:, i] = (1.0 - blend) * cur + blend * neigh
        if t >= 2:
            cur0, nxt0 = sr[:, 0], sr[:, 1]
            f0 = spynet(nxt0, cur0)
            out[:, 0] = (1.0 - blend) * cur0 + blend * _warp(nxt0, f0)
            curL, prevL = sr[:, t - 1], sr[:, t - 2]
            fL = spynet(prevL, curL)
            out[:, t - 1] = (1.0 - blend) * curL + blend * _warp(prevL, fL)
    return out.clamp(0.0, 1.0)
