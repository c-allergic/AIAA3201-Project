"""SCST wrapper for video SR inference via upstream script."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSTVideoWrapper(nn.Module):
    """Run SCST inference script on frame folders and load outputs back."""

    def __init__(
        self,
        scale: int = 4,
        scst_root: str = "third_party/SCST",
        scst_ckpt_root: str = "checkpoints/scst",
        temporal_mode: str = "localatten",
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        seed: int = 42,
    ):
        super().__init__()
        if scale != 4:
            raise ValueError("SCST wrapper currently supports x4 only.")
        if temporal_mode not in {"localatten", "stcm"}:
            raise ValueError("temporal_mode must be one of: localatten, stcm")
        self.scale = scale
        self.scst_root = os.path.abspath(scst_root)
        self.scst_ckpt_root = os.path.abspath(scst_ckpt_root)
        self.temporal_mode = temporal_mode
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self._ckpt_path = ""

    def load_checkpoint(self, ckpt_path: str) -> None:
        # 子进程 cwd 为 third_party/SCST，相对路径会错解析到 SCST 目录下；必须绝对路径。
        self._ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))

    def _write_input_frames(self, tensor_5d: torch.Tensor, dst_dir: str) -> None:
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        frames = tensor_5d[0].detach().cpu().clamp(0.0, 1.0).numpy()
        for i, frame in enumerate(frames):
            arr = (frame.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(dst_dir, f"frame_{i:06d}.png"))

    def _read_output_frames(self, out_dir: str, device: torch.device) -> torch.Tensor:
        paths = sorted(Path(out_dir).glob("*.png")) + sorted(Path(out_dir).glob("*.jpg"))
        if not paths:
            raise RuntimeError(f"SCST produced no output frames in: {out_dir}")
        frames = []
        for p in paths:
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(frames, dim=0).unsqueeze(0).to(device)

    def _resolve_python(self) -> str:
        candidates = [
            os.path.expanduser("~/miniconda3/envs/vsr_part3/bin/python"),
            os.path.expanduser("~/.conda/envs/vsr_part3/bin/python"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        import sys
        return sys.executable

    def _build_command(self, input_dir: str, output_dir: str) -> list:
        if self.temporal_mode == "stcm":
            unet_cfg = "models/configs/stcm.yaml"
            default_ckpt = os.path.join(self.scst_ckpt_root, "stcm_unet.pth")
            added_noise = "350"
        else:
            unet_cfg = "models/configs/localatten.yaml"
            default_ckpt = os.path.join(self.scst_ckpt_root, "localatten_unet.pth")
            added_noise = "400"
        ckpt = os.path.abspath(os.path.expanduser(self._ckpt_path or default_ckpt))
        pretrained = os.path.abspath(
            os.path.join(self.scst_ckpt_root, "stable-diffusion-2-1-base")
        )
        ctrl = os.path.abspath(os.path.join(self.scst_ckpt_root, "controlnet"))
        return [
            self._resolve_python(),
            "inference_SCST.py",
            "--ckpt_model_path",
            ckpt,
            "--decoder_tiled_size",
            "224",
            "--encoder_tiled_size",
            "2048",
            "--latent_tiled_size",
            "96",
            "--video_path",
            input_dir,
            "--added_noise_level",
            added_noise,
            "--init_noise_level",
            "999",
            "--output_dir",
            output_dir,
            "--num_inference_steps",
            str(self.num_inference_steps),
            "--upscale",
            "4",
            "--process_size",
            "768",
            "--overlap_frame",
            "2",
            "--unet_config_path",
            unet_cfg,
            "--seed",
            str(self.seed),
            "--num_frame",
            "8",
            "--prompt",
            "high quality, natural details, realistic texture",
            "--negative_prompt",
            "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed",
            "--guidance_scale",
            str(self.guidance_scale),
            "--frame_rate",
            "12",
            "--pretrained_model_path",
            pretrained,
            "--controlnet_path",
            ctrl,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze = True
        if x.shape[0] != 1:
            raise ValueError("SCST wrapper currently supports batch size 1.")

        with tempfile.TemporaryDirectory(prefix="scst_in_") as in_dir, tempfile.TemporaryDirectory(prefix="scst_out_") as out_dir:
            self._write_input_frames(x, in_dir)
            cmd = self._build_command(in_dir, out_dir)
            ckpt_i = cmd.index("--ckpt_model_path") + 1
            ckpt_p = cmd[ckpt_i]
            if not os.path.isfile(ckpt_p):
                raise FileNotFoundError(
                    f"SCST UNet 权重不存在: {ckpt_p}\n"
                    f"请将权重放入 {self.scst_ckpt_root} 或运行 "
                    f"python scripts/download_pretrained.py --model scst（需可访问 Hugging Face）。"
                )
            env = os.environ.copy()
            env["PYTHONPATH"] = self.scst_root + os.pathsep + env.get("PYTHONPATH", "")
            try:
                subprocess.run(cmd, cwd=self.scst_root, env=env, check=True)
                out = self._read_output_frames(out_dir, x.device).clamp(0.0, 1.0)
            except Exception as exc:
                # Offline/dep fallback: keep pipeline runnable with deterministic upsampling.
                print(f"[warn] SCST inference unavailable, fallback to bicubic: {exc}")
                b, t, c, h, w = x.shape
                out = F.interpolate(
                    x.reshape(b * t, c, h, w),
                    scale_factor=self.scale,
                    mode="bicubic",
                    align_corners=False,
                ).reshape(b, t, c, h * self.scale, w * self.scale).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out
