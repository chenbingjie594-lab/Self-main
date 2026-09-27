"""Stage9A frozen-backbone residual adapter.

The adapter never consumes a defect reference.  Its condition is restricted to
the binary target mask, the paired normal/background image and the diffusion
timestep.  It injects bounded, zero-initialized residuals into the final two
(highest-resolution) UNet up blocks.
"""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    freq = torch.exp(-scale * torch.arange(half, device=timesteps.device))
    args = timesteps.float()[:, None] * freq[None]
    emb = torch.cat((args.sin(), args.cos()), dim=1)
    return F.pad(emb, (0, dim - emb.shape[1]))


class FrozenResidualAdapter(nn.Module):
    """Small conditioning network with no reference or class-specific input."""

    def __init__(self, channels=(640, 320), hidden=64, max_rms_ratio=0.10):
        super().__init__()
        self.channels = tuple(channels)
        self.hidden = int(hidden)
        self.max_rms_ratio = float(max_rms_ratio)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, hidden // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 3, padding=1), nn.SiLU())
        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 3, padding=1), nn.SiLU())
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.projections = nn.ModuleList([nn.Conv2d(hidden, c, 1) for c in channels])
        for projection in self.projections:
            nn.init.zeros_(projection.weight); nn.init.zeros_(projection.bias)
        self._mask = self._context = self._timesteps = None
        self._handles = []

    def set_condition(self, mask, normal_context, timesteps):
        if mask.ndim != 4 or mask.shape[1] != 1: raise ValueError("mask must be Bx1xHxW")
        if normal_context.ndim != 4 or normal_context.shape[1] != 3: raise ValueError("context must be Bx3xHxW")
        self._mask, self._context, self._timesteps = mask, normal_context, timesteps

    def _residual(self, hidden, index):
        if self._mask is None: raise RuntimeError("Stage9A condition was not set")
        size = hidden.shape[-2:]
        mask = F.interpolate(self._mask.float(), size=size, mode="nearest")
        context = F.interpolate(self._context.float(), size=size, mode="bilinear", align_corners=False)
        feature = self.mask_encoder(mask) + self.context_encoder(context)
        time = self.time_mlp(timestep_embedding(self._timesteps, self.hidden)).to(feature.dtype)
        feature = F.silu(feature + time[:, :, None, None]) * mask
        residual = self.projections[index](feature).to(hidden.dtype)
        h_rms = hidden.float().square().mean((1,2,3), keepdim=True).sqrt().clamp_min(1e-6)
        r_rms = residual.float().square().mean((1,2,3), keepdim=True).sqrt().clamp_min(1e-6)
        scale = (self.max_rms_ratio * h_rms / r_rms).clamp(max=1.0).to(residual.dtype)
        return residual * scale

    def attach(self, unet):
        if self._handles: return len(self._handles)
        if len(unet.up_blocks) < 2: raise ValueError("UNet must expose at least two up blocks")
        for index, block in enumerate(unet.up_blocks[-2:]):
            def hook(_module, _inputs, output, i=index):
                if isinstance(output, tuple): return (output[0] + self._residual(output[0], i), *output[1:])
                return output + self._residual(output, i)
            self._handles.append(block.register_forward_hook(hook))
        return len(self._handles)

    def detach(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def save(self, path: Path, metadata: dict):
        torch.save({"state_dict":self.state_dict(), "metadata":metadata}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload=torch.load(path,map_location=device)
        obj=cls(**payload["metadata"]["architecture"]);obj.load_state_dict(payload["state_dict"])
        return obj, payload["metadata"]

