"""Multi-scale defect fidelity adapter for tiny industrial anomalies.

MSDF fuses a pre-VAE pixel residual branch with a latent semantic branch,
preserves the reference morphology while aligning it to the target coarse-mask
centre, and injects frequency-aware features into every UNet upsampling stage.
A scale/time gate makes high-resolution stages more important for tiny masks
and changes the injection over the denoising path.

The output projections are zero-initialized, so attaching a fresh adapter is
exactly equivalent to the unmodified UNet at the start of training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


MSDF_FORMAT_VERSION = 3


def _safe_rms(value: torch.Tensor, dims) -> torch.Tensor:
    """RMS with a finite derivative at an exactly-zero input."""
    return (value.float().square().mean(dim=dims) + 1e-8).sqrt()


def _block_output_channels(unet) -> list[int]:
    channels = []
    for block in unet.up_blocks:
        if not getattr(block, "resnets", None):
            raise ValueError("Every UNet up block must expose at least one resnet")
        channels.append(int(block.resnets[-1].out_channels))
    return channels


class MSDFAdapter(nn.Module):
    """Reference-conditioned multi-scale spatial residual adapter."""

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        block_channels: Sequence[int],
        context_scale: float = 1.75,
        max_injection: float = 0.75,
        max_residual_ratio: float = 0.25,
        branch_dropout: float = 0.20,
        pixel_support_weight: float = 0.50,
    ):
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if not block_channels:
            raise ValueError("block_channels cannot be empty")
        if context_scale < 1.0:
            raise ValueError("context_scale must be at least 1")
        if max_injection <= 0:
            raise ValueError("max_injection must be positive")
        if not 0 < max_residual_ratio <= 1:
            raise ValueError("max_residual_ratio must be in (0, 1]")
        if not 0 <= branch_dropout < 0.5:
            raise ValueError("branch_dropout must be in [0, 0.5)")
        if pixel_support_weight < 0:
            raise ValueError("pixel_support_weight cannot be negative")

        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.block_channels = tuple(int(value) for value in block_channels)
        self.context_scale = float(context_scale)
        self.max_injection = float(max_injection)
        self.max_residual_ratio = float(max_residual_ratio)
        self.branch_dropout = float(branch_dropout)
        self.pixel_support_weight = float(pixel_support_weight)

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups:
            groups -= 1
        self.encoder = nn.Sequential(
            nn.Conv2d(self.latent_channels + 1, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
        )
        pixel_mid = max(16, self.hidden_dim // 2)
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, pixel_mid, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(pixel_mid, self.hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.hidden_dim, self.hidden_dim, 1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
        )
        # This head gives the pre-VAE branch its own learning signal. Without
        # it, a jointly trained UNet can satisfy the diffusion objective while
        # the fusion layer silently ignores the new high-resolution branch.
        self.pixel_support_head = nn.Conv2d(self.hidden_dim, 1, 1)
        nn.init.zeros_(self.pixel_support_head.weight)
        nn.init.constant_(self.pixel_support_head.bias, -2.0)
        self.support_head = nn.Conv2d(self.hidden_dim, 1, 1)
        nn.init.zeros_(self.support_head.weight)
        nn.init.constant_(self.support_head.bias, -2.0)
        # Inputs: log mask area, normalized diffusion time, local response.
        self.gate = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(self.block_channels)),
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, channels, 1)
            for channels in self.block_channels
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        self._hook_handles = []
        self._feature = None
        self._support = None
        self._gates = None
        self._cfg = False
        self.last_gates = None
        self.last_aligned_support = None
        self.last_reference_support = None
        self.last_pixel_support = None

    @property
    def num_scales(self) -> int:
        return len(self.block_channels)

    def attach(self, unet) -> int:
        """Attach residual hooks to UNet up blocks once."""
        discovered = tuple(_block_output_channels(unet))
        if discovered != self.block_channels:
            raise ValueError(
                "MSDF/UNet channel mismatch: checkpoint="
                f"{self.block_channels}, unet={discovered}"
            )
        self.detach()
        for index, block in enumerate(unet.up_blocks):
            self._hook_handles.append(
                block.register_forward_hook(self._make_hook(index))
            )
        return len(self._hook_handles)

    def detach(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.clear()

    def clear(self):
        self._feature = None
        self._support = None
        self._gates = None
        self._cfg = False
        self.last_aligned_support = None
        self.last_reference_support = None
        self.last_pixel_support = None

    @staticmethod
    def _bbox(mask: torch.Tensor):
        coordinates = torch.nonzero(mask > 0.05, as_tuple=False)
        if coordinates.numel() == 0:
            return 0, mask.shape[-2], 0, mask.shape[-1]
        y0 = int(coordinates[:, -2].min())
        y1 = int(coordinates[:, -2].max()) + 1
        x0 = int(coordinates[:, -1].min())
        x1 = int(coordinates[:, -1].max()) + 1
        return y0, y1, x0, x1

    def _expand_bbox(self, bbox, height: int, width: int):
        y0, y1, x0, x1 = bbox
        center_y = 0.5 * (y0 + y1)
        center_x = 0.5 * (x0 + x1)
        box_h = max(1.0, (y1 - y0) * self.context_scale)
        box_w = max(1.0, (x1 - x0) * self.context_scale)
        new_y0 = max(0, int(round(center_y - 0.5 * box_h)))
        new_y1 = min(height, int(round(center_y + 0.5 * box_h)))
        new_x0 = max(0, int(round(center_x - 0.5 * box_w)))
        new_x1 = min(width, int(round(center_x + 0.5 * box_w)))
        return new_y0, max(new_y0 + 1, new_y1), new_x0, max(new_x0 + 1, new_x1)

    def _align_reference(
        self,
        reference_feature: torch.Tensor,
        reference_support: torch.Tensor,
        target_prior: torch.Tensor,
    ):
        """Place reference morphology without stretching it to the coarse mask.

        The target prior specifies only a centre and maximum search extent. The
        reference support supplies aspect ratio and occupied-area ratio, so a
        dot, elongated flash, or irregular spot does not collapse to the shape
        of the target coarse annotation.
        """
        reference_feature = torch.nan_to_num(
            reference_feature, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        reference_support = torch.nan_to_num(
            reference_support, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        target_prior = torch.nan_to_num(
            target_prior, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        feature_canvases = []
        support_canvases = []
        height, width = target_prior.shape[-2:]
        for batch_index in range(reference_feature.shape[0]):
            source_box = self._expand_bbox(
                self._bbox(reference_support[batch_index, 0].detach()), height, width
            )
            target_box = self._bbox(target_prior[batch_index, 0].detach())
            sy0, sy1, sx0, sx1 = source_box
            ty0, ty1, tx0, tx1 = target_box
            source_h = max(1, sy1 - sy0)
            source_w = max(1, sx1 - sx0)
            target_h = max(1, ty1 - ty0)
            target_w = max(1, tx1 - tx0)

            source_region = reference_support[
                batch_index : batch_index + 1, :, sy0:sy1, sx0:sx1
            ]
            occupied_ratio = torch.nan_to_num(
                source_region.mean().detach(), nan=0.08, posinf=0.85, neginf=0.08
            ).clamp(0.08, 0.85)
            desired_area = float(target_h * target_w) * float(occupied_ratio)
            aspect = source_w / float(source_h)
            placed_h = max(1, int(round((desired_area / max(aspect, 1e-6)) ** 0.5)))
            placed_w = max(1, int(round(placed_h * aspect)))
            fit = min(1.0, target_h / placed_h, target_w / placed_w)
            placed_h = max(1, int(round(placed_h * fit)))
            placed_w = max(1, int(round(placed_w * fit)))
            center_y = (ty0 + ty1) // 2
            center_x = (tx0 + tx1) // 2
            py0 = max(0, min(height - placed_h, center_y - placed_h // 2))
            px0 = max(0, min(width - placed_w, center_x - placed_w // 2))
            py1, px1 = py0 + placed_h, px0 + placed_w

            feature_crop = reference_feature[
                batch_index : batch_index + 1, :, sy0:sy1, sx0:sx1
            ]
            feature_crop = F.interpolate(
                feature_crop,
                size=(placed_h, placed_w),
                mode="bilinear",
                align_corners=False,
            )
            support_crop = F.interpolate(
                source_region,
                size=(placed_h, placed_w),
                mode="bilinear",
                align_corners=False,
            )
            feature_canvases.append(
                F.pad(feature_crop, (px0, width - px1, py0, height - py1))
            )
            support_canvases.append(
                F.pad(support_crop, (px0, width - px1, py0, height - py1))
            )
        return torch.cat(feature_canvases, dim=0), torch.cat(support_canvases, dim=0)

    def prepare(
        self,
        reference_latents: torch.Tensor,
        reference_mask: torch.Tensor,
        target_prior: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
        classifier_free_guidance: bool,
        reference_pixels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build one spatial feature pyramid context for the next UNet call."""
        latent_size = reference_latents.shape[-2:]
        reference_mask_pixels = reference_mask.float().clamp(0, 1)
        reference_mask = F.interpolate(
            reference_mask_pixels, size=latent_size, mode="nearest"
        ).clamp(0, 1)
        target_prior = F.interpolate(
            target_prior.float(), size=latent_size, mode="bilinear", align_corners=False
        ).clamp(0, 1)
        reference_latents = torch.nan_to_num(
            reference_latents.float(), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        outside = 1.0 - reference_mask
        kernel = 9
        outside_count = F.avg_pool2d(
            outside, kernel, stride=1, padding=kernel // 2
        )
        local_background = F.avg_pool2d(
            reference_latents * outside,
            kernel,
            stride=1,
            padding=kernel // 2,
        ) / outside_count.clamp_min(0.05)
        fallback_background = F.avg_pool2d(
            reference_latents, kernel, stride=1, padding=kernel // 2
        )
        reliable = (outside_count > 0.05).to(reference_latents.dtype)
        local_background = (
            reliable * local_background + (1.0 - reliable) * fallback_background
        ).clamp(-10.0, 10.0)
        defect_residual = (
            (reference_latents - local_background) * reference_mask
        ).clamp(-20.0, 20.0)
        latent_encoded = self.encoder(
            torch.cat([defect_residual, reference_mask], dim=1)
        )
        if reference_pixels is None:
            raise ValueError("MSDF v2 requires reference_pixels before VAE compression")
        reference_pixels = torch.nan_to_num(
            reference_pixels.float(), nan=0.0, posinf=1.0, neginf=-1.0
        ).clamp(-1.0, 1.0)
        pixel_mask = F.interpolate(
            reference_mask_pixels,
            size=reference_pixels.shape[-2:],
            mode="nearest",
        )
        pixel_kernel = 31
        pixel_outside = 1.0 - pixel_mask
        pixel_count = F.avg_pool2d(
            pixel_outside, pixel_kernel, stride=1, padding=pixel_kernel // 2
        )
        pixel_background = F.avg_pool2d(
            reference_pixels * pixel_outside,
            pixel_kernel,
            stride=1,
            padding=pixel_kernel // 2,
        ) / pixel_count.clamp_min(0.02)
        pixel_fallback = F.avg_pool2d(
            reference_pixels,
            pixel_kernel,
            stride=1,
            padding=pixel_kernel // 2,
        )
        pixel_reliable = (pixel_count > 0.02).to(reference_pixels.dtype)
        pixel_background = (
            pixel_reliable * pixel_background
            + (1.0 - pixel_reliable) * pixel_fallback
        ).clamp(-1.0, 1.0)
        pixel_residual = (
            (reference_pixels - pixel_background) * pixel_mask
        ).clamp(-2.0, 2.0)
        pixel_encoded = self.pixel_encoder(
            torch.cat([pixel_residual, pixel_mask], dim=1)
        )
        pixel_encoded = F.interpolate(
            pixel_encoded,
            size=latent_size,
            mode="bilinear",
            align_corners=False,
        )
        pixel_support = (
            torch.sigmoid(
                torch.nan_to_num(
                    self.pixel_support_head(pixel_encoded).float(),
                    nan=-2.0,
                    posinf=-2.0,
                    neginf=-2.0,
                ).clamp(-12.0, 12.0)
            )
            * reference_mask
        )
        self.last_pixel_support = pixel_support

        # Modality dropout makes both encoders useful: some training batches
        # must be solved from the pixel branch and some from the latent branch.
        # At inference both branches are always active.
        if self.training and self.branch_dropout > 0:
            choice = torch.rand(
                latent_encoded.shape[0], 1, 1, 1,
                device=latent_encoded.device,
            )
            latent_encoded = latent_encoded * (choice >= self.branch_dropout)
            pixel_encoded = pixel_encoded * (choice < 1.0 - self.branch_dropout)

        encoded = self.fusion(torch.cat([latent_encoded, pixel_encoded], dim=1))
        encoded = torch.nan_to_num(
            encoded, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(-8.0, 8.0)
        support_logits = torch.nan_to_num(
            self.support_head(encoded).float(),
            nan=-2.0,
            posinf=-2.0,
            neginf=-2.0,
        ).clamp(-12.0, 12.0)
        fused_support = torch.sigmoid(support_logits) * reference_mask
        # Pixel evidence cannot be bypassed at inference, while the fused head
        # still contributes semantic/latent support.
        reference_support = 0.5 * (fused_support + pixel_support)
        aligned, aligned_support = self._align_reference(
            encoded, reference_support, target_prior
        )

        # Only the reference-derived support determines morphology. A narrow
        # soft ring permits halo blending, while the coarse target prior remains
        # a location/maximum-extent constraint rather than an output shape.
        context = F.max_pool2d(aligned_support, 3, stride=1, padding=1)
        support = (aligned_support + 0.25 * (context - aligned_support)).clamp(0, 1)
        support = support * F.max_pool2d(target_prior, 3, stride=1, padding=1)
        aligned = aligned * support
        area = target_prior.flatten(1).mean(1).clamp_min(1e-6)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=area.device)
        timesteps = timesteps.to(device=area.device, dtype=torch.float32).flatten()
        if timesteps.numel() == 1 and area.numel() > 1:
            timesteps = timesteps.expand(area.numel())
        time_value = timesteps / max(float(num_train_timesteps - 1), 1.0)
        response = torch.nan_to_num(
            _safe_rms(aligned, dims=(1, 2, 3)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 20.0)
        gate_input = torch.stack([area.log(), time_value, response], dim=1)
        gate_logits = torch.nan_to_num(
            self.gate(gate_input).float(),
            nan=0.0,
            posinf=12.0,
            neginf=-12.0,
        ).clamp(-12.0, 12.0)
        gates = torch.sigmoid(gate_logits) * self.max_injection

        # Bias tiny masks toward later/high-resolution up blocks. The learned
        # gate remains free to suppress an unhelpful scale or timestep.
        smallness = (-area.log() / 9.0).clamp(0, 1).unsqueeze(1)
        resolution_bias = torch.linspace(
            0.65, 1.35, self.num_scales, device=gates.device
        ).unsqueeze(0)
        gates = gates * (1.0 + smallness * (resolution_bias - 1.0))
        # Resolution preference must not violate the advertised global cap.
        gates = gates.clamp(0.0, self.max_injection)

        self._feature = aligned
        self._support = support
        self._gates = gates
        self._cfg = bool(classifier_free_guidance)
        self.last_gates = gates.detach()
        self.last_reference_support = reference_support
        self.last_aligned_support = aligned_support
        return gates

    def morphology_support_loss(
        self,
        target_support: torch.Tensor,
        self_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise shape only when reference and diffusion target coincide."""
        if self.last_aligned_support is None:
            raise RuntimeError("Call MSDF prepare before morphology_support_loss")
        predicted = torch.nan_to_num(
            self.last_aligned_support,
            nan=1e-5,
            posinf=1.0 - 1e-5,
            neginf=1e-5,
        ).clamp(1e-5, 1.0 - 1e-5)
        target = F.interpolate(
            target_support.float(),
            size=predicted.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        selected = self_reference.to(
            device=predicted.device, dtype=torch.bool
        ).flatten()
        if not bool(selected.any()):
            return predicted.sum() * 0.0
        predicted = predicted[selected]
        target = target[selected]

        def support_objective(value, truth):
            value = value.clamp(1e-5, 1.0 - 1e-5)
            positive_weight = 4.0
            bce = -(
                positive_weight * truth * value.log()
                + (1.0 - truth) * (1.0 - value).log()
            ).mean()
            intersection = (value * truth).sum((1, 2, 3))
            denominator = value.sum((1, 2, 3)) + truth.sum((1, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
            return bce + dice.mean()

        loss = support_objective(predicted, target)
        if self.last_pixel_support is not None and self.pixel_support_weight > 0:
            pixel_support = F.interpolate(
                self.last_pixel_support[selected],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = loss + self.pixel_support_weight * support_objective(
                pixel_support, target
            )
        return loss

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            if self._feature is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            feature = F.interpolate(
                self._feature,
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            low_frequency = F.avg_pool2d(feature, 3, stride=1, padding=1)
            high_frequency = feature - low_frequency
            scale_fraction = index / max(self.num_scales - 1, 1)
            # Coarse decoder stages receive stable body/halo information;
            # high-resolution stages receive progressively stronger edges.
            feature = low_frequency + (0.25 + scale_fraction) * high_frequency
            support = F.interpolate(
                self._support,
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            residual = self.projections[index](feature.to(self.projections[index].weight.dtype))
            residual = torch.nan_to_num(
                residual, nan=0.0, posinf=0.0, neginf=0.0
            )
            residual = residual.to(hidden.dtype) * support.to(hidden.dtype)
            gate = self._gates[:, index].reshape(-1, 1, 1, 1).to(hidden.dtype)
            residual = residual * gate
            if hidden.shape[0] == 2 * residual.shape[0] and self._cfg:
                residual = torch.cat([torch.zeros_like(residual), residual], dim=0)
            elif hidden.shape[0] != residual.shape[0]:
                repeats = hidden.shape[0] // residual.shape[0]
                if repeats * residual.shape[0] != hidden.shape[0]:
                    raise RuntimeError("MSDF feature batch does not match UNet batch")
                residual = residual.repeat_interleave(repeats, dim=0)
            # Bound every injected pixel relative to the current UNet feature
            # RMS. A fixed absolute clamp is unsafe because activation scales
            # differ greatly between up blocks and defect classes.
            hidden_rms = _safe_rms(
                hidden, dims=(1, 2, 3)
            ).reshape(-1, 1, 1, 1).clamp_min(1e-4)
            limit = self.max_residual_ratio * hidden_rms
            residual = limit * torch.tanh(residual.float() / limit.clamp_min(1e-6))
            residual = torch.nan_to_num(
                residual, nan=0.0, posinf=0.0, neginf=0.0
            ).to(hidden.dtype)
            updated = hidden + residual
            if isinstance(output, tuple):
                return (updated,) + output[1:]
            return updated

        return hook


def build_msdf_adapter(
    unet,
    latent_channels: int,
    hidden_dim: int,
    context_scale: float,
    max_injection: float,
    max_residual_ratio: float = 0.25,
    branch_dropout: float = 0.20,
    pixel_support_weight: float = 0.50,
) -> MSDFAdapter:
    return MSDFAdapter(
        latent_channels=latent_channels,
        hidden_dim=hidden_dim,
        block_channels=_block_output_channels(unet),
        context_scale=context_scale,
        max_injection=max_injection,
        max_residual_ratio=max_residual_ratio,
        branch_dropout=branch_dropout,
        pixel_support_weight=pixel_support_weight,
    )


def load_msdf_adapter(path, unet, device) -> MSDFAdapter:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MSDF checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    version = int(checkpoint.get("format_version", 0))
    if version != MSDF_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported MSDF checkpoint format: v{version}; "
            "MSDF v3 changes stability and pixel supervision and must be retrained"
        )
    adapter = MSDFAdapter(
        latent_channels=int(checkpoint["latent_channels"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        block_channels=checkpoint["block_channels"],
        context_scale=float(checkpoint["context_scale"]),
        max_injection=float(checkpoint["max_injection"]),
        max_residual_ratio=float(checkpoint["max_residual_ratio"]),
        branch_dropout=float(checkpoint["branch_dropout"]),
        pixel_support_weight=float(checkpoint["pixel_support_weight"]),
    )
    state_dict = checkpoint["state_dict"]
    invalid = [
        name for name, value in state_dict.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if invalid:
        raise ValueError(
            "MSDF checkpoint contains NaN/Inf tensors: " + ", ".join(invalid[:8])
        )
    adapter.load_state_dict(state_dict, strict=True)
    adapter.to(device=device, dtype=torch.float32).eval()
    adapter.attach(unet)
    return adapter
