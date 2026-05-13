"""Minimal utilities for cleaned VSR pipeline."""

import os
from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# Image Quality Metrics
# ============================================================

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
    if hr.dim() == 3:
        hr = hr.unsqueeze(0)
    mse = F.mse_loss(sr, hr)
    if mse.item() <= 0:
        return 99.0
    return float(-10.0 * torch.log10(mse).item())


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().clamp(0.0, 1.0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(tensor.shape)}")
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    return (image * 255.0).round().astype(np.uint8)


def load_frames(frame_dir: str) -> torch.Tensor:
    paths = sorted(Path(frame_dir).glob("*.png")) + sorted(Path(frame_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames found in: {frame_dir}")
    frames: List[torch.Tensor] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(frames, dim=0).unsqueeze(0)


def save_frames(sr: torch.Tensor, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if sr.dim() != 5:
        raise ValueError(f"Expected (B,T,C,H,W), got {tuple(sr.shape)}")
    for i in range(sr.shape[1]):
        img = tensor_to_image(sr[0, i])
        Image.fromarray(img).save(os.path.join(output_dir, f"frame_{i:06d}.png"))


def ensure_weights(weights_root: str, urls: Dict[str, str], key: str) -> str:
    if key not in urls:
        raise KeyError(f"weights.urls missing key: {key}")
    os.makedirs(weights_root, exist_ok=True)
    filename = os.path.basename(urls[key])
    ckpt_path = os.path.join(weights_root, filename)
    if os.path.exists(ckpt_path):
        return ckpt_path
    print(f"[download] {key} -> {ckpt_path}")
    urlretrieve(urls[key], ckpt_path)
    return ckpt_path
