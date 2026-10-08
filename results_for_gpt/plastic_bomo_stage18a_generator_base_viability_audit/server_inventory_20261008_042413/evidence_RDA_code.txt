"""Reference-Decoupled Anomaly Adapter v4.

The adapter converts frozen-VAE reference latents into cross-attention tokens.
It never writes residuals into UNet feature maps, preventing brightness and
colour amplification while preserving reference morphology.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


RDA_FORMAT_VERSION = 4


def _crop_to_mask(feature: torch.Tensor, mask: torch.Tensor):
    crops = []
    mask = F.interpolate(mask.float(), feature.shape[-2:], mode="nearest")
    for index in range(feature.shape[0]):
        positions = torch.nonzero(mask[index, 0] > 0.5, as_tuple=False)
        if positions.numel() == 0:
            crops.append(feature[index : index + 1])
            continue
        y0, y1 = int(positions[:, 0].min()), int(positions[:, 0].max()) + 1
        x0, x1 = int(positions[:, 1].min()), int(positions[:, 1].max()) + 1
        crops.append(feature[index : index + 1, :, y0:y1, x0:x1])
    return crops


class ReferenceTokenAdapter(nn.Module):
    """Project frozen VAE defect crops into adaptively gated attention tokens."""

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_dim: int = 128,
        token_dim: int = 1024,
        token_grid: int = 4,
        max_token_scale: float = 0.25,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.token_grid = int(token_grid)
        self.max_token_scale = float(max_token_scale)
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
        )
        self.projection = nn.Linear(hidden_dim, token_dim)
        self.output_norm = nn.LayerNorm(token_dim)
        self.token_gate = nn.Parameter(torch.tensor(-2.0))
        gate_hidden = max(16, hidden_dim // 4)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(4, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.adaptive_gate[-1].weight)
        nn.init.zeros_(self.adaptive_gate[-1].bias)
        self.last_token_scale = None

    @property
    def num_tokens(self):
        return self.token_grid * self.token_grid

    def forward(self, reference_latents: torch.Tensor, reference_mask: torch.Tensor):
        reference_latents = reference_latents.float()
        latent_mask = F.interpolate(
            reference_mask.float(), reference_latents.shape[-2:], mode="nearest"
        ).clamp(0, 1)
        dilated = F.max_pool2d(latent_mask, 9, stride=1, padding=4)
        ring = (dilated - latent_mask).clamp(0, 1)
        inside_denominator = latent_mask.sum((2, 3)).clamp_min(1.0)
        ring_denominator = ring.sum((2, 3)).clamp_min(1.0)
        inside_mean = (
            reference_latents * latent_mask
        ).sum((2, 3)) / inside_denominator
        ring_mean = (
            reference_latents * ring
        ).sum((2, 3)) / ring_denominator
        global_std = reference_latents.std((1, 2, 3)).clamp_min(1e-3)
        contrast = (
            (inside_mean - ring_mean).abs().mean(1) / global_std
        ).clamp(0, 4) / 4
        centered = reference_latents - inside_mean[:, :, None, None]
        texture = (
            (centered.square() * latent_mask).sum((1, 2, 3))
            / (inside_denominator[:, 0] * reference_latents.shape[1])
        ).clamp_min(0).sqrt()
        texture = (texture / global_std).clamp(0, 4) / 4
        area = latent_mask.mean((1, 2, 3)).sqrt()
        crop_area = []
        for sample_mask in latent_mask:
            positions = torch.nonzero(sample_mask[0] > 0.5, as_tuple=False)
            if positions.numel() == 0:
                crop_area.append(reference_latents.new_tensor(1.0))
            else:
                height = positions[:, 0].max() - positions[:, 0].min() + 1
                width = positions[:, 1].max() - positions[:, 1].min() + 1
                crop_area.append(
                    (height * width).float()
                    / (reference_latents.shape[-2] * reference_latents.shape[-1])
                )
        descriptors = torch.stack(
            [contrast, texture, area, torch.stack(crop_area).sqrt()], dim=1
        )

        crops = _crop_to_mask(reference_latents, reference_mask)
        pooled = []
        for crop in crops:
            mean = crop.mean(dim=(2, 3), keepdim=True)
            std = crop.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
            normalized = ((crop - mean) / std).clamp(-4, 4)
            encoded = self.encoder(normalized)
            pooled.append(
                F.adaptive_avg_pool2d(encoded, (self.token_grid, self.token_grid))
            )
        feature = torch.cat(pooled, dim=0).flatten(2).transpose(1, 2)
        tokens = self.output_norm(self.projection(feature))
        gate_offset = self.adaptive_gate(descriptors)
        scale = self.max_token_scale * torch.sigmoid(self.token_gate + gate_offset)
        self.last_token_scale = scale
        return tokens * scale[:, None, :]


def append_reference_tokens(
    prompt_embeddings: torch.Tensor,
    reference_tokens: torch.Tensor,
    classifier_free_guidance: bool,
):
    """Append zeros to unconditional prompts and references to conditional."""
    reference_tokens = reference_tokens.to(
        device=prompt_embeddings.device, dtype=prompt_embeddings.dtype
    )
    if classifier_free_guidance:
        half = prompt_embeddings.shape[0] // 2
        unconditional, conditional = prompt_embeddings[:half], prompt_embeddings[half:]
        zero_tokens = torch.zeros_like(reference_tokens)
        unconditional = torch.cat([unconditional, zero_tokens], dim=1)
        conditional = torch.cat([conditional, reference_tokens], dim=1)
        return torch.cat([unconditional, conditional], dim=0)
    return torch.cat([prompt_embeddings, reference_tokens], dim=1)


def load_rda_adapter(path, device) -> ReferenceTokenAdapter:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RDA adapter not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    version = int(payload.get("format_version", 1))
    if version != RDA_FORMAT_VERSION:
        raise ValueError(
            f"RDA weight format v{version} is incompatible with code v"
            f"{RDA_FORMAT_VERSION}; use the validated v4 checkpoint"
        )
    model = ReferenceTokenAdapter(
        latent_channels=int(payload.get("latent_channels", 4)),
        hidden_dim=int(payload.get("hidden_dim", 128)),
        token_dim=int(payload.get("token_dim", 1024)),
        token_grid=int(payload.get("token_grid", 4)),
        max_token_scale=float(payload.get("max_token_scale", 0.25)),
    )
    model.load_state_dict(payload["state_dict"])
    model.requires_grad_(False)
    return model.to(device=device, dtype=torch.float32).eval()
