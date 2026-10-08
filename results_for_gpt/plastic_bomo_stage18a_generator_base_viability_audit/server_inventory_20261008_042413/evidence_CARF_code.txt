"""Coarse-Annotation Region Refinement for latent diffusion.

CARF treats an input mask as a search region rather than pixel-accurate ground
truth.  A small timestep-aware network predicts a soft latent-space support map
from the current noisy latent, masked-image latent, and coarse prior.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CARF_FORMAT_VERSION = 4
SUPPORTED_CARF_FORMAT_VERSIONS = {2, 3, 4}


class _LegacyCARFRegionRefiner(nn.Module):
    """Checkpoint-compatible implementation for CARF v2/v3 weights."""

    def __init__(self, latent_channels=4, hidden_dim=64, min_support=0.05):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.min_support = float(min_support)
        input_channels = self.latent_channels * 2 + 2
        groups = min(8, self.hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim // 2, 2, 1),
        )
        # Start close to the coarse prior. Training then learns only the useful
        # refinement instead of destabilising an already working inpaint model.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Near-zero spatial intervention makes the joint model start close to
        # RDA-only. Diffusion gradients can increase it when gating is useful.
        self.attention_gate_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, noisy_latents, masked_latents, coarse_prior, timesteps):
        coarse_prior = F.interpolate(
            coarse_prior.float(), noisy_latents.shape[-2:], mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=noisy_latents.device)
        timesteps = timesteps.to(device=noisy_latents.device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(noisy_latents.shape[0])
        if timesteps.shape[0] != noisy_latents.shape[0]:
            timesteps = timesteps[:1].expand(noisy_latents.shape[0])
        time_map = (timesteps / 999.0).clamp(0, 1)[:, None, None, None]
        time_map = time_map.expand(-1, 1, *noisy_latents.shape[-2:])
        inputs = torch.cat(
            [noisy_latents.float(), masked_latents.float(), coarse_prior, time_map],
            dim=1,
        )
        core_residual, context_residual = self.net(inputs).chunk(2, dim=1)
        core = torch.sigmoid(core_residual) * coarse_prior
        # A blurred support permits smooth halos and boundary transitions while
        # remaining inside the uncertain annotation. It is not forced to be a
        # compact copy of the defect core.
        context_support = (
            F.avg_pool2d(coarse_prior, 7, stride=1, padding=3) * coarse_prior
        )
        context = torch.sigmoid(context_residual - 1.1) * context_support
        refined = core + (1.0 - core) * context
        self.last_core_map = core
        self.last_context_map = context
        return refined.clamp(0, 1)

    def attention_gate(self, context_weight=0.5):
        if not hasattr(self, "last_core_map"):
            raise RuntimeError("Call CARF forward before requesting its gate")
        spatial = (
            self.last_core_map + float(context_weight) * self.last_context_map
        ).clamp(0, 1)
        alpha = torch.sigmoid(self.attention_gate_logit)
        self.last_attention_alpha = alpha
        return ((1.0 - alpha) + alpha * spatial).clamp(1e-4, 1.0)


class CARFRegionRefiner(nn.Module):
    """Residual CARF v3 refiner with sample-adaptive morphology gates.

    The coarse prior is an identity path. Consequently a newly initialized
    CARF is exactly equivalent to the baseline mask and learns only a bounded
    residual correction instead of replacing a useful prior at step zero.
    """

    def __init__(
        self,
        latent_channels=4,
        hidden_dim=64,
        min_support=0.35,
        max_core_adjustment=0.75,
        max_context_adjustment=0.50,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.min_support = float(min_support)
        self.max_core_adjustment = float(max_core_adjustment)
        self.max_context_adjustment = float(max_context_adjustment)
        input_channels = self.latent_channels * 2 + 2
        groups = min(8, self.hidden_dim)
        feature_dim = self.hidden_dim // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, feature_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.residual_head = nn.Conv2d(feature_dim, 2, 1)
        # area, residual mean, residual deviation, normalized timestep
        self.morphology_gate = nn.Sequential(
            nn.Linear(4, max(8, feature_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(8, feature_dim // 2), 2),
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.morphology_gate[-1].weight)
        nn.init.zeros_(self.morphology_gate[-1].bias)
        self.attention_gate_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, noisy_latents, masked_latents, coarse_prior, timesteps):
        coarse_prior = F.interpolate(
            coarse_prior.float(), noisy_latents.shape[-2:], mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=noisy_latents.device)
        timesteps = timesteps.to(device=noisy_latents.device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(noisy_latents.shape[0])
        if timesteps.shape[0] != noisy_latents.shape[0]:
            timesteps = timesteps[:1].expand(noisy_latents.shape[0])
        normalized_time = (timesteps / 999.0).clamp(0, 1)
        time_map = normalized_time[:, None, None, None].expand(
            -1, 1, *noisy_latents.shape[-2:]
        )
        inputs = torch.cat(
            [noisy_latents.float(), masked_latents.float(), coarse_prior, time_map],
            dim=1,
        )
        core_residual, context_residual = self.residual_head(
            self.encoder(inputs)
        ).chunk(2, dim=1)

        latent_residual = (noisy_latents.float() - masked_latents.float()).abs().mean(1)
        support = coarse_prior[:, 0]
        support_sum = support.sum((1, 2)).clamp_min(1.0)
        residual_mean = (latent_residual * support).sum((1, 2)) / support_sum
        residual_variance = (
            (latent_residual - residual_mean[:, None, None]).square() * support
        ).sum((1, 2)) / support_sum
        descriptors = torch.stack(
            [support.mean((1, 2)), residual_mean, residual_variance.sqrt(), normalized_time],
            dim=1,
        )
        morphology = torch.sigmoid(self.morphology_gate(descriptors))
        core_gate = morphology[:, 0, None, None, None]
        context_gate = morphology[:, 1, None, None, None]

        # Core learns bounded suppression. min_support is now an actual lower
        # bound, preventing a coarse box from collapsing to a tiny bright dot.
        core_scale = 1.0 - (
            self.max_core_adjustment * core_gate * torch.sigmoid(core_residual) * 2.0
        )
        # Subtract the initialization offset so zero logits yield scale=1.
        core_scale = core_scale + self.max_core_adjustment * core_gate
        core_scale = core_scale.clamp(min=self.min_support, max=1.0)
        core = coarse_prior * core_scale

        context_support = F.avg_pool2d(coarse_prior, 7, stride=1, padding=3)
        context_activation = (
            self.max_context_adjustment
            * context_gate
            * torch.tanh(context_residual)
        )
        context_correction = context_activation * context_support * coarse_prior
        refined = core + (coarse_prior - core) * context_correction
        # Positive map is used by weak supervision and optional joint attention;
        # the signed correction above keeps the zero-initialized identity path.
        context = torch.sigmoid(context_residual) * context_support * coarse_prior
        self.last_core_map = core
        self.last_context_map = context
        self.last_morphology_gate = morphology
        return refined.clamp(0, 1)

    def attention_gate(self, context_weight=0.5):
        if not hasattr(self, "last_core_map"):
            raise RuntimeError("Call CARF forward before requesting its gate")
        spatial = (
            self.last_core_map + float(context_weight) * self.last_context_map
        ).clamp(0, 1)
        alpha = torch.sigmoid(self.attention_gate_logit)
        self.last_attention_alpha = alpha
        return ((1.0 - alpha) + alpha * spatial).clamp(1e-4, 1.0)


def carf_weak_supervision_loss(
    refined,
    evidence,
    coarse_prior,
    second_view=None,
    core=None,
    context=None,
    target_support_ratio=None,
    evidence_weight=1.0,
    containment_weight=0.5,
    area_weight=0.1,
    smoothness_weight=0.05,
    consistency_weight=0.25,
    coverage_weight=0.5,
    min_core_coverage=0.90,
):
    """Weak supervision without pretending the coarse mask is exact GT."""
    evidence = F.interpolate(
        evidence.float(), refined.shape[-2:], mode="bilinear", align_corners=False
    ).clamp(0, 1)
    coarse_prior = F.interpolate(
        coarse_prior.float(), refined.shape[-2:], mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)
    # Reliable image evidence is positive supervision. Weak responses are not
    # treated as definite background, which is essential for imperfect masks.
    positive = -(evidence * refined.clamp_min(1e-6).log()).sum()
    positive = positive / evidence.sum().clamp_min(1.0)
    outside = (refined * (1.0 - coarse_prior)).mean()
    support_ratio = refined.sum((2, 3)) / coarse_prior.sum((2, 3)).clamp_min(1.0)
    if target_support_ratio is None:
        target_support_ratio = (
            evidence.sum((2, 3)) / coarse_prior.sum((2, 3)).clamp_min(1.0)
        ).clamp(0.03, 0.90)
    target_support_ratio = target_support_ratio.detach().to(support_ratio)
    area = F.smooth_l1_loss(support_ratio, target_support_ratio)
    smoothness = (
        (refined[..., 1:] - refined[..., :-1]).abs().mean()
        + (refined[..., 1:, :] - refined[..., :-1, :]).abs().mean()
    )
    consistency = refined.new_zeros(())
    if second_view is not None:
        consistency = F.smooth_l1_loss(refined, second_view)
    decomposition = refined.new_zeros(())
    coverage = refined.new_zeros(())
    coverage_penalty = refined.new_zeros(())
    if core is not None and context is not None:
        high_confidence = evidence.square()
        core_positive = -(
            high_confidence * core.clamp_min(1e-6).log()
        ).sum() / high_confidence.sum().clamp_min(1.0)
        # Context should explain lower-confidence transition evidence without
        # replacing the high-confidence core.
        transition = (evidence - high_confidence).clamp_min(0)
        context_positive = -(
            transition * context.clamp_min(1e-6).log()
        ).sum() / transition.sum().clamp_min(1.0)
        overlap = (context * high_confidence).mean()
        decomposition = core_positive + 0.5 * context_positive + 0.1 * overlap
        covered_fraction = (core * high_confidence).sum((2, 3)) / high_confidence.sum(
            (2, 3)
        ).clamp_min(1.0)
        coverage = covered_fraction.mean()
        coverage_penalty = F.relu(
            float(min_core_coverage) - covered_fraction
        ).mean()
    total = (
        evidence_weight * positive
        + containment_weight * outside
        + area_weight * area
        + smoothness_weight * smoothness
        + consistency_weight * consistency
        + 0.25 * decomposition
        + coverage_weight * coverage_penalty
    )
    return total, {
        "carf_evidence": positive.detach(),
        "carf_outside": outside.detach(),
        "carf_area": area.detach(),
        "carf_smoothness": smoothness.detach(),
        "carf_consistency": consistency.detach(),
        "carf_support": support_ratio.mean().detach(),
        "carf_target_support": target_support_ratio.mean().detach(),
        "carf_decomposition": decomposition.detach(),
        "carf_core_coverage": coverage.detach(),
        "carf_coverage_penalty": coverage_penalty.detach(),
    }


def load_carf_refiner(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CARF checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    version = int(payload.get("format_version", 0))
    if version not in SUPPORTED_CARF_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported CARF checkpoint format: v{version}")
    model_class = _LegacyCARFRegionRefiner if version in {2, 3} else CARFRegionRefiner
    kwargs = {
        "latent_channels": int(payload.get("latent_channels", 4)),
        "hidden_dim": int(payload.get("hidden_dim", 64)),
        "min_support": float(payload.get("min_support", 0.05)),
    }
    if version >= 4:
        kwargs.update(
            max_core_adjustment=float(payload.get("max_core_adjustment", 0.75)),
            max_context_adjustment=float(payload.get("max_context_adjustment", 0.50)),
        )
    model = model_class(**kwargs)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {"attention_gate_logit"} if version == 2 else set()
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            "CARF checkpoint state is incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.requires_grad_(False)
    return model.to(device=device, dtype=torch.float32).eval()
