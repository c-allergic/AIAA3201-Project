"""Part 2 BasicVSR wrappers (use official pretrained checkpoints)."""

import torch
import torch.nn as nn
import sys
import types
import os


def _patch_torchvision_compat() -> None:
    """BasicSR expects old torchvision module path."""
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
    """Fallback to vendored BasicSR in third_party/SCST when pip basicsr is unavailable."""
    try:
        import basicsr  # type: ignore # noqa: F401
        return
    except Exception:
        pass
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, "third_party", "SCST")
    if os.path.isdir(os.path.join(candidate, "basicsr")) and candidate not in sys.path:
        sys.path.insert(0, candidate)


class BasicVSRWrapper(nn.Module):
    """Wrapper over BasicSR BasicVSR architecture."""

    def __init__(self, scale: int = 4, mid_channels: int = 64, num_blocks: int = 30):
        super().__init__()
        self.scale = scale
        _patch_torchvision_compat()
        _patch_basicsr_fallback()
        try:
            from basicsr.archs.basicvsr_arch import BasicVSR as BasicVSRArch
        except ImportError as exc:
            raise ImportError(
                "未安装 basicsr。请先执行 `pip install --no-deps --no-build-isolation basicsr==1.4.2`。"
            ) from exc
        self.net = BasicVSRArch(num_feat=mid_channels, num_block=num_blocks, spynet_path=None)

    @staticmethod
    def _map_checkpoint_keys(state: dict) -> dict:
        """Map MMEditing/MMagic BasicVSR keys to BasicSR BasicVSR keys."""
        mapped = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("generator."):
                nk = nk[len("generator.") :]

            nk = nk.replace("backward_resblocks.", "backward_trunk.")
            nk = nk.replace("forward_resblocks.", "forward_trunk.")
            nk = nk.replace("upsample1.upsample_conv.", "upconv1.")
            nk = nk.replace("upsample2.upsample_conv.", "upconv2.")

            # SpyNet ConvModule -> Sequential(Conv+ReLU) key style in BasicSR.
            # Example:
            # spynet.basic_module.0.basic_module.1.conv.weight
            # -> spynet.basic_module.0.basic_module.2.weight
            if nk.startswith("spynet.basic_module.") and ".basic_module." in nk and ".conv." in nk:
                parts = nk.split(".")
                # spynet basic_module {lvl} basic_module {idx} conv {w/b}
                if len(parts) == 7 and parts[1] == "basic_module" and parts[3] == "basic_module":
                    conv_idx = int(parts[4])
                    target_idx = conv_idx * 2
                    suffix = parts[6]  # weight or bias
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
                "BasicVSR checkpoint mismatch after mapping: "
                f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze = True
        out = self.net(x).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out
