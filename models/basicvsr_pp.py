"""Part 3 BasicVSR++ wrapper (OpenMMLab checkpoint compatible)."""

import torch
import torch.nn as nn

from models.basicvsr import _patch_torchvision_compat, _patch_basicsr_fallback


class BasicVSRPlusPlusWrapper(nn.Module):
    """Wrapper over BasicSR BasicVSR++ architecture."""

    def __init__(self, scale: int = 4, mid_channels: int = 64, num_blocks: int = 7):
        super().__init__()
        if scale != 4:
            raise ValueError("BasicVSR++ wrapper currently supports x4 only.")
        self.scale = scale
        _patch_torchvision_compat()
        _patch_basicsr_fallback()
        try:
            from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus as BasicVSRPPArch
        except ImportError as exc:
            raise ImportError(
                "未安装 basicsr。请先执行 `pip install --no-deps --no-build-isolation basicsr==1.4.2`。"
            ) from exc
        self.net = BasicVSRPPArch(
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            is_low_res_input=True,
            spynet_path=None,
            cpu_cache_length=100,
        )

    @staticmethod
    def _map_checkpoint_keys(state: dict) -> dict:
        mapped = {}
        for k, v in state.items():
            nk = k
            if nk == "step_counter":
                continue
            if nk.startswith("generator."):
                nk = nk[len("generator.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]

            # OpenMMLab MMEditing uses PixelShuffleUpsample (upsampleN.upsample_conv);
            # BasicSR BasicVSR++ uses plain Conv2d upconvN.
            if nk.startswith("upsample1.upsample_conv."):
                nk = "upconv1." + nk.split(".", 2)[2]
            elif nk.startswith("upsample2.upsample_conv."):
                nk = "upconv2." + nk.split(".", 2)[2]

            # ConvModule keys from MMEditing/MMagic to BasicSR conv/relu sequential style.
            if nk.startswith("spynet.basic_module.") and ".basic_module." in nk and ".conv." in nk:
                parts = nk.split(".")
                if len(parts) == 7 and parts[1] == "basic_module" and parts[3] == "basic_module":
                    conv_idx = int(parts[4])
                    target_idx = conv_idx * 2
                    suffix = parts[6]
                    nk = f"spynet.basic_module.{parts[2]}.basic_module.{target_idx}.{suffix}"
                else:
                    nk = nk.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
            else:
                nk = nk.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
            mapped[nk] = v
        return mapped

    def load_checkpoint(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("state_dict", raw)
        cleaned = self._map_checkpoint_keys(state)
        msg = self.net.load_state_dict(cleaned, strict=False)
        if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
            raise RuntimeError(
                "BasicVSR++ checkpoint mismatch after mapping: "
                f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze = True
        out = self.net(x).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out
