"""Uncertainty-aware fusion for fidelity/generative SR branches."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    return 0.2989 * x[:, :, 0:1] + 0.5870 * x[:, :, 1:2] + 0.1140 * x[:, :, 2:3]


def _sobel_edge_strength(x: torch.Tensor) -> torch.Tensor:
    gray = _rgb_to_gray(x)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    b, t, c, h, w = gray.shape
    g = gray.reshape(b * t, c, h, w)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag.reshape(b, t, 1, h, w)


class FusionWeightCNN(nn.Module):
    """Lightweight fusion network that predicts pixel-wise blending weight."""

    def __init__(self, in_channels: int = 4, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def compute_uncertainty_features(sr_fid: torch.Tensor, sr_gen: torch.Tensor) -> Dict[str, torch.Tensor]:
    if sr_fid.shape != sr_gen.shape:
        raise ValueError("sr_fid and sr_gen must have identical shapes.")
    if sr_fid.dim() != 5:
        raise ValueError("Expected 5D tensors: (B,T,C,H,W).")

    disagree = (sr_fid - sr_gen).abs().mean(dim=2, keepdim=True)
    edge = _sobel_edge_strength(sr_fid)
    var = torch.zeros_like(disagree)
    var[:, 1:-1] = ((sr_fid[:, 1:-1] - sr_fid[:, :-2]).abs() + (sr_fid[:, 1:-1] - sr_fid[:, 2:]).abs()).mean(
        dim=2, keepdim=True
    )
    var[:, 0] = (sr_fid[:, 0] - sr_fid[:, 1]).abs().mean(dim=1, keepdim=True)
    var[:, -1] = (sr_fid[:, -1] - sr_fid[:, -2]).abs().mean(dim=1, keepdim=True)
    var_gen = torch.zeros_like(disagree)
    var_gen[:, 1:-1] = ((sr_gen[:, 1:-1] - sr_gen[:, :-2]).abs() + (sr_gen[:, 1:-1] - sr_gen[:, 2:]).abs()).mean(
        dim=2, keepdim=True
    )
    var_gen[:, 0] = (sr_gen[:, 0] - sr_gen[:, 1]).abs().mean(dim=1, keepdim=True)
    var_gen[:, -1] = (sr_gen[:, -1] - sr_gen[:, -2]).abs().mean(dim=1, keepdim=True)
    return {"var_t": var, "var_gen_t": var_gen, "disagree": disagree, "edge_strength": edge}


def rule_based_weight(
    sr_fid: torch.Tensor,
    sr_gen: torch.Tensor,
    alpha: float = 4.0,
    beta: float = 6.0,
    gamma: float = 3.0,
    zeta: float = 2.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    feat = compute_uncertainty_features(sr_fid, sr_gen)
    raw = (
        alpha * feat["var_t"]
        + beta * feat["disagree"]
        - gamma * feat["edge_strength"]
        - zeta * feat["var_gen_t"]
    )
    w = torch.sigmoid(raw)
    return w, feat


def fuse_with_weight(sr_fid: torch.Tensor, sr_gen: torch.Tensor, w_gen: torch.Tensor) -> torch.Tensor:
    if w_gen.shape[2] != 1:
        raise ValueError("w_gen should be single-channel weight map.")
    return (1.0 - w_gen) * sr_fid + w_gen * sr_gen


def build_fusion_training_input(feat: Dict[str, torch.Tensor], sr_fid: torch.Tensor, sr_gen: torch.Tensor) -> torch.Tensor:
    l1 = (sr_fid - sr_gen).abs().mean(dim=2, keepdim=True)
    return torch.cat([feat["var_t"], feat["disagree"], feat["edge_strength"], l1], dim=2)
