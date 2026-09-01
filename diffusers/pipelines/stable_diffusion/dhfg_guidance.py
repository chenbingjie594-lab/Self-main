"""Detector-guided real-feature supervision for diffusion training.

The detector is exported to TorchScript ahead of time.  The exported teacher
returns neck feature maps and per-class response maps, so diffusion training
does not depend on Ultralytics and does not backpropagate through NMS.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DHFG_FORMAT_VERSION = 1


def _normalise_class_name(value: str) -> str:
    value = re.sub(r"^\d+", "", str(value).lower())
    return "".join(character for character in value if character.isalnum())


def crop_mask_regions(
    value: torch.Tensor,
    mask: torch.Tensor,
    output_size: int,
    context_scale: float = 4.0,
    minimum_extent_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop differentiable local regions around coarse masks.

    Bounding-box selection is intentionally non-differentiable; gradients are
    only required from the resized crop back to ``value``.  A minimum context
    extent prevents tiny masks from becoming unrealistically magnified.
    """

    if value.ndim != 4 or mask.ndim != 4:
        raise ValueError("value and mask must be BCHW tensors")
    if value.shape[0] != mask.shape[0]:
        raise ValueError("value and mask batch sizes must match")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    source_height, source_width = mask.shape[-2:]
    value_height, value_width = value.shape[-2:]
    crops, crop_masks = [], []
    for sample_index in range(value.shape[0]):
        coordinates = torch.nonzero(mask[sample_index, 0] > 0.5, as_tuple=False)
        if coordinates.numel() == 0:
            center_y, center_x = source_height / 2.0, source_width / 2.0
            box_height = source_height * minimum_extent_ratio
            box_width = source_width * minimum_extent_ratio
        else:
            y0, x0 = coordinates.amin(dim=0).tolist()
            y1, x1 = coordinates.amax(dim=0).tolist()
            center_y = (float(y0) + float(y1) + 1.0) / 2.0
            center_x = (float(x0) + float(x1) + 1.0) / 2.0
            box_height = max(
                (float(y1) - float(y0) + 1.0) * float(context_scale),
                source_height * float(minimum_extent_ratio),
            )
            box_width = max(
                (float(x1) - float(x0) + 1.0) * float(context_scale),
                source_width * float(minimum_extent_ratio),
            )
        box_height = min(box_height, float(source_height))
        box_width = min(box_width, float(source_width))
        top = max(0.0, min(center_y - box_height / 2.0, source_height - box_height))
        left = max(0.0, min(center_x - box_width / 2.0, source_width - box_width))
        bottom = top + box_height
        right = left + box_width

        value_top = int(round(top * value_height / source_height))
        value_left = int(round(left * value_width / source_width))
        value_bottom = int(round(bottom * value_height / source_height))
        value_right = int(round(right * value_width / source_width))
        value_top = min(max(value_top, 0), value_height - 1)
        value_left = min(max(value_left, 0), value_width - 1)
        value_bottom = min(max(value_bottom, value_top + 1), value_height)
        value_right = min(max(value_right, value_left + 1), value_width)

        mask_top = int(round(top))
        mask_left = int(round(left))
        mask_bottom = min(max(int(round(bottom)), mask_top + 1), source_height)
        mask_right = min(max(int(round(right)), mask_left + 1), source_width)
        mask_top = min(max(mask_top, 0), source_height - 1)
        mask_left = min(max(mask_left, 0), source_width - 1)

        crops.append(
            F.interpolate(
                value[sample_index : sample_index + 1, :, value_top:value_bottom, value_left:value_right],
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
        )
        crop_masks.append(
            F.interpolate(
                mask[sample_index : sample_index + 1, :, mask_top:mask_bottom, mask_left:mask_right].float(),
                size=(output_size, output_size),
                mode="nearest",
            )
        )
    return torch.cat(crops), torch.cat(crop_masks).clamp(0, 1)


def paired_confidence_loss(
    generated_confidence: torch.Tensor,
    reference_confidence: torch.Tensor,
    tolerance: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Match each generated score to its real reference within a tolerance.

    A symmetric tolerance prevents both invisible defects and confidence
    maximisation.  Huber growth caps the influence of an initially very large
    generated/reference logit gap.
    """

    if tolerance < 0:
        raise ValueError("confidence tolerance must be non-negative")
    difference = (generated_confidence - reference_confidence.detach()).abs()
    outside_tolerance = F.relu(difference - float(tolerance))
    zeros = torch.zeros_like(outside_tolerance)
    return F.smooth_l1_loss(
        outside_tolerance, zeros, beta=float(beta), reduction="none"
    ).mean()


def calibrated_interval_loss(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Penalise values only when they leave a calibrated real-data interval."""

    if beta <= 0:
        raise ValueError("interval beta must be positive")
    lower = lower.to(values)
    upper = upper.to(values)
    if bool((lower > upper).any()):
        raise ValueError("interval lower bound cannot exceed upper bound")
    distance = F.relu(lower - values) + F.relu(values - upper)
    return F.smooth_l1_loss(
        distance,
        torch.zeros_like(distance),
        beta=float(beta),
        reduction="mean",
    )


def _resize_support(mask: torch.Tensor, size) -> torch.Tensor:
    """Resize a tiny support without letting it vanish on coarse YOLO scales."""

    if size[0] <= mask.shape[-2] and size[1] <= mask.shape[-1]:
        return F.adaptive_max_pool2d(mask.float(), size).clamp(0, 1)
    return F.interpolate(mask.float(), size=size, mode="nearest").clamp(0, 1)


def _masked_topk_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    fraction: float,
    maximum_count: int,
    empty_value: float,
) -> torch.Tensor:
    """Average only the strongest valid responses for each batch sample."""

    results = []
    for sample_index in range(values.shape[0]):
        selected = values[sample_index][valid_mask[sample_index] > 0.5]
        if selected.numel() == 0:
            results.append(values.new_tensor(float(empty_value)))
            continue
        count = max(1, int(round(selected.numel() * float(fraction))))
        count = min(count, int(maximum_count), selected.numel())
        results.append(selected.topk(count).values.mean())
    return torch.stack(results)


def _masked_feature_statistics(
    feature: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mask-pooled direction and first two spatial feature moments.

    ``mask`` may be a coarse region or a reference-derived core/context map.
    Channel-normalised feature maps make detector scales with very different
    activation magnitudes comparable.
    """

    support = _resize_support(mask, feature.shape[-2:]).float()
    value = F.normalize(feature.float(), dim=1, eps=1e-6)
    denominator = support.sum((2, 3)).clamp_min(1.0)
    mean = (value * support).sum((2, 3)) / denominator
    centered = value - mean[:, :, None, None]
    variance = (centered.square() * support).sum((2, 3)) / denominator
    standard_deviation = variance.clamp_min(1e-8).sqrt()
    pooled = F.normalize(mean, dim=1, eps=1e-6)
    return pooled, mean, standard_deviation


def _local_photometric_signature(
    pixels: torch.Tensor,
    rough_mask: torch.Tensor,
    context_radius: int = 8,
    contrast_threshold: float = 1.0,
    contrast_temperature: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Measure polarity, colour change and soft morphology relative to context.

    Positive and negative luminance evidence are kept separately.  This makes
    the representation work for bright, dark and bipolar defects without a
    class-name switch.  The rough mask limits the search region but its border
    is never treated as the defect boundary.
    """

    if pixels.ndim != 4 or pixels.shape[1] != 3 or rough_mask.ndim != 4:
        raise ValueError("pixels must be RGB BCHW and rough_mask must be BCHW")
    if context_radius < 1:
        raise ValueError("context_radius must be at least 1")
    if contrast_temperature <= 0:
        raise ValueError("contrast_temperature must be positive")
    mask = F.interpolate(
        rough_mask.float(), pixels.shape[-2:], mode="nearest"
    ).clamp(0, 1)
    dilated = F.max_pool2d(
        mask,
        kernel_size=2 * int(context_radius) + 1,
        stride=1,
        padding=int(context_radius),
    )
    ring = (dilated - mask).clamp(0, 1)
    outside = 1.0 - mask
    ring_count = ring.sum((2, 3), keepdim=True)
    context = torch.where(ring_count > 0, ring, outside)
    context_count = context.sum((2, 3), keepdim=True).clamp_min(1.0)

    value = pixels.float()
    luminance = (
        0.299 * value[:, 0:1]
        + 0.587 * value[:, 1:2]
        + 0.114 * value[:, 2:3]
    )
    context_mean = (luminance * context).sum((2, 3), keepdim=True) / context_count
    context_variance = (
        (luminance - context_mean).square() * context
    ).sum((2, 3), keepdim=True) / context_count
    # A floor of two percent of the image range prevents nearly uniform
    # backgrounds from turning compression noise into enormous z-scores.
    context_std = context_variance.clamp_min(0.02**2).sqrt()
    signed_contrast = ((luminance - context_mean) / context_std).clamp(-8, 8)
    positive_map = F.relu(signed_contrast)
    negative_map = F.relu(-signed_contrast)
    positive = _masked_topk_mean(positive_map, mask, 0.05, 256, 0.0)
    negative = _masked_topk_mean(negative_map, mask, 0.05, 256, 0.0)

    opponent = torch.cat(
        (value[:, 0:1] - value[:, 1:2], value[:, 2:3] - value[:, 1:2]),
        dim=1,
    )
    opponent_mean = (opponent * context).sum((2, 3), keepdim=True) / context_count
    chroma_map = (opponent - opponent_mean).square().sum(1, keepdim=True).add(1e-8).sqrt()
    chroma = _masked_topk_mean(chroma_map, mask, 0.05, 256, 0.0)

    activation = torch.sigmoid(
        (signed_contrast.abs() - float(contrast_threshold))
        / float(contrast_temperature)
    ) * mask
    mask_count = mask.sum((2, 3)).clamp_min(1.0)
    coverage = activation.sum((2, 3))[:, 0] / mask_count[:, 0]

    height, width = pixels.shape[-2:]
    yy = torch.linspace(-1, 1, height, device=pixels.device, dtype=torch.float32)
    xx = torch.linspace(-1, 1, width, device=pixels.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    weight = activation[:, 0]
    weight_sum = weight.sum((1, 2)).clamp_min(1e-6)
    mean_x = (weight * grid_x).sum((1, 2)) / weight_sum
    mean_y = (weight * grid_y).sum((1, 2)) / weight_sum
    variance_x = (
        weight * (grid_x[None] - mean_x[:, None, None]).square()
    ).sum((1, 2)) / weight_sum
    variance_y = (
        weight * (grid_y[None] - mean_y[:, None, None]).square()
    ).sum((1, 2)) / weight_sum
    extent = (variance_x + variance_y).clamp_min(1e-8).sqrt()
    return {
        "positive": positive,
        "negative": negative,
        "chroma": chroma,
        "coverage": coverage,
        "extent": extent,
        "contrast_map": signed_contrast.abs(),
    }


@torch.no_grad()
def _reference_core_context_support(
    response: torch.Tensor,
    rough_mask: torch.Tensor,
    core_fraction: float = 0.25,
    maximum_core_locations: int = 8,
    context_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate the real defect core and a surrounding context ring.

    The support is selected exclusively from the frozen teacher response on
    the real reference.  The generated image therefore cannot move attention
    onto an adversarial high-response pixel.  Top-k selection also prevents a
    tiny defect from being averaged away by its much larger coarse mask.
    """

    if not 0 < core_fraction <= 1:
        raise ValueError("core_fraction must be in (0, 1]")
    if maximum_core_locations < 1:
        raise ValueError("maximum_core_locations must be positive")
    if context_radius < 1:
        raise ValueError("context_radius must be positive")
    support_mask = _resize_support(rough_mask, response.shape[-2:]) > 0.5
    cores = []
    for sample_index in range(response.shape[0]):
        valid_flat = support_mask[sample_index, 0].flatten()
        valid_indices = torch.nonzero(valid_flat, as_tuple=False).flatten()
        core_flat = torch.zeros_like(valid_flat, dtype=response.dtype)
        if valid_indices.numel() == 0:
            center = (valid_flat.numel() - 1) // 2
            core_flat[center] = 1.0
        else:
            count = max(1, int(round(valid_indices.numel() * float(core_fraction))))
            count = min(count, int(maximum_core_locations), valid_indices.numel())
            valid_scores = response[sample_index, 0].flatten()[valid_indices]
            selected = valid_indices[valid_scores.topk(count).indices]
            core_flat[selected] = 1.0
        cores.append(core_flat.reshape(1, *response.shape[-2:]))
    core = torch.stack(cores, dim=0)
    dilated = F.max_pool2d(
        core,
        kernel_size=2 * int(context_radius) + 1,
        stride=1,
        padding=int(context_radius),
    )
    context = (dilated - core).clamp(0, 1)
    return core, context


@torch.no_grad()
def _soft_reference_core_context_support(
    response: torch.Tensor,
    reference_contrast: torch.Tensor,
    rough_mask: torch.Tensor,
    context_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a continuous core from teacher response and physical contrast."""

    if context_radius < 1:
        raise ValueError("context_radius must be positive")
    support = _resize_support(rough_mask, response.shape[-2:]).float()
    contrast = _resize_support(reference_contrast, response.shape[-2:]).float()
    response_core = response.float() * support
    contrast_core = contrast * support
    response_scale = response_core.amax((2, 3), keepdim=True).clamp_min(1e-6)
    contrast_scale = contrast_core.amax((2, 3), keepdim=True).clamp_min(1e-6)
    response_core = response_core / response_scale
    contrast_core = contrast_core / contrast_scale
    # The product demands agreement, while the small additive path keeps a
    # useful core when a weak detector misses a physically clear dark defect.
    fused = (
        (response_core * contrast_core).clamp_min(0).sqrt()
        + 0.25 * contrast_core
    ) * support
    core = fused / fused.amax((2, 3), keepdim=True).clamp_min(1e-6)
    core = core.square()
    dilated = F.max_pool2d(
        core,
        kernel_size=2 * int(context_radius) + 1,
        stride=1,
        padding=int(context_radius),
    )
    context = (dilated - core).clamp(0, 1)
    return core, context


@torch.no_grad()
def _counterfactual_support(
    responses: list[torch.Tensor],
    reference_contrast: torch.Tensor,
    rough_mask: torch.Tensor,
    output_size: tuple[int, int],
    erasure_radius: int = 4,
) -> torch.Tensor:
    """Fuse all teacher scales with physical contrast into a soft erase support."""

    if not responses:
        raise ValueError("at least one teacher response is required")
    if erasure_radius < 0:
        raise ValueError("erasure_radius cannot be negative")
    resized = [
        F.interpolate(response.float(), output_size, mode="bilinear", align_corners=False)
        for response in responses
    ]
    teacher_response = torch.stack(resized, dim=0).amax(0)
    core, _ = _soft_reference_core_context_support(
        teacher_response,
        reference_contrast,
        rough_mask,
        context_radius=1,
    )
    if erasure_radius:
        core = F.max_pool2d(
            core,
            kernel_size=2 * int(erasure_radius) + 1,
            stride=1,
            padding=int(erasure_radius),
        )
    # Feathering prevents the erased image itself from acquiring a sharp edge
    # that a detector could mistake for a defect.
    support = F.avg_pool2d(core, kernel_size=5, stride=1, padding=2)
    mask = F.interpolate(rough_mask.float(), output_size, mode="nearest")
    return (support * mask).clamp(0, 1)


def counterfactual_erase(
    pixels: torch.Tensor,
    support: torch.Tensor,
    blur_radius: int = 8,
) -> torch.Tensor:
    """Replace a soft defect support with differentiable local context."""

    if blur_radius < 1:
        raise ValueError("blur_radius must be at least 1")
    if support.shape[-2:] != pixels.shape[-2:]:
        support = F.interpolate(support.float(), pixels.shape[-2:], mode="bilinear", align_corners=False)
    kernel = 2 * int(blur_radius) + 1
    padded = F.pad(
        pixels.float(),
        (blur_radius, blur_radius, blur_radius, blur_radius),
        mode="reflect",
    )
    local_context = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    erased = pixels.float() * (1.0 - support) + local_context * support
    # Calibration runs outside autocast.  Preserve the teacher-facing input
    # dtype so an fp16 TorchScript detector never receives a float32 erase.
    return erased.to(dtype=pixels.dtype)


def _masked_smooth_l1(
    generated: torch.Tensor,
    reference: torch.Tensor,
    support: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    support = _resize_support(support, generated.shape[-2:]).float()
    error = F.smooth_l1_loss(
        generated.float(), reference.detach().float(), beta=float(beta), reduction="none"
    )
    return (error * support).sum() / (
        support.sum() * generated.shape[1]
    ).clamp_min(1.0)


def boundary_preservation_loss(
    generated_pixels: torch.Tensor,
    reference_pixels: torch.Tensor,
    mask: torch.Tensor,
    radius: int = 8,
) -> torch.Tensor:
    """Preserve the real sample in a ring immediately outside the coarse mask."""

    if radius < 1:
        raise ValueError("boundary radius must be at least 1")
    if generated_pixels.shape != reference_pixels.shape:
        raise ValueError("generated and reference pixels must have equal shapes")
    local_mask = F.interpolate(mask.float(), generated_pixels.shape[-2:], mode="nearest")
    dilated = F.max_pool2d(
        local_mask,
        kernel_size=2 * int(radius) + 1,
        stride=1,
        padding=int(radius),
    )
    ring = (dilated - local_mask).clamp(0, 1)
    absolute_error = (generated_pixels.float() - reference_pixels.float()).abs()
    return (absolute_error * ring).sum() / (
        ring.sum() * generated_pixels.shape[1]
    ).clamp_min(1.0)


class DHFGTeacher(nn.Module):
    """Frozen TorchScript detector feature teacher.

    The archive must return ``features + class_logits`` and include a
    ``dhfg.json`` extra file written by ``tools/export_dhfg_teacher.py``.
    """

    def __init__(
        self,
        teacher_path: str | Path,
        class_name: str | None = None,
        class_id: int | None = None,
        topk_fraction: float = 0.05,
        maximum_object_locations: int = 4,
        background_topk_fraction: float = 0.01,
        maximum_background_locations: int = 16,
        minimum_teacher_probability: float = 0.01,
        confidence_beta: float = 1.0,
        boundary_radius: int = 8,
        contrast_threshold: float = 1.0,
        contrast_temperature: float = 0.25,
        erasure_radius: int = 4,
        erasure_blur_radius: int = 8,
        delta_tolerance: float = 0.25,
        minimum_teacher_delta: float = 0.01,
    ):
        super().__init__()
        extra_files = {"dhfg.json": ""}
        self.teacher = torch.jit.load(str(teacher_path), map_location="cpu", _extra_files=extra_files)
        raw_metadata = extra_files["dhfg.json"]
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        if not raw_metadata:
            raise ValueError("DHFG teacher archive has no dhfg.json metadata")
        metadata = json.loads(raw_metadata)
        if int(metadata.get("format_version", -1)) != DHFG_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported DHFG teacher format {metadata.get('format_version')}"
            )
        self.input_size = int(metadata["input_size"])
        self.num_scales = int(metadata["num_scales"])
        self.class_names = [str(name) for name in metadata["class_names"]]
        self.source_weights = str(metadata.get("source_weights", ""))
        if class_id is None:
            wanted = _normalise_class_name(class_name or "")
            matches = [
                index
                for index, name in enumerate(self.class_names)
                if _normalise_class_name(name) == wanted
                or _normalise_class_name(name) in wanted
                or wanted in _normalise_class_name(name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot uniquely map defect class {class_name!r}; "
                    f"teacher classes are {self.class_names}. Pass --dhfg_class_id."
                )
            class_id = matches[0]
        if not 0 <= int(class_id) < len(self.class_names):
            raise ValueError(f"dhfg_class_id {class_id} is out of range")
        self.class_id = int(class_id)
        self.topk_fraction = float(topk_fraction)
        self.maximum_object_locations = int(maximum_object_locations)
        self.background_topk_fraction = float(background_topk_fraction)
        self.maximum_background_locations = int(maximum_background_locations)
        self.minimum_teacher_probability = float(minimum_teacher_probability)
        self.confidence_beta = float(confidence_beta)
        self.boundary_radius = int(boundary_radius)
        self.contrast_threshold = float(contrast_threshold)
        self.contrast_temperature = float(contrast_temperature)
        self.erasure_radius = int(erasure_radius)
        self.erasure_blur_radius = int(erasure_blur_radius)
        self.delta_tolerance = float(delta_tolerance)
        self.minimum_teacher_delta = float(minimum_teacher_delta)
        if not 0 < self.topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1]")
        if not 0 < self.background_topk_fraction <= 1:
            raise ValueError("background_topk_fraction must be in (0, 1]")
        if self.maximum_object_locations < 1 or self.maximum_background_locations < 1:
            raise ValueError("DHFG top-k maximum counts must be positive")
        if not 0 <= self.minimum_teacher_probability < 1:
            raise ValueError("minimum_teacher_probability must be in [0, 1)")
        if self.confidence_beta <= 0:
            raise ValueError("confidence_beta must be positive")
        if self.boundary_radius < 1:
            raise ValueError("boundary_radius must be at least 1")
        if self.contrast_temperature <= 0:
            raise ValueError("contrast_temperature must be positive")
        if self.erasure_radius < 0 or self.erasure_blur_radius < 1:
            raise ValueError("invalid counterfactual erasure radii")
        if self.delta_tolerance < 0:
            raise ValueError("delta_tolerance must be non-negative")
        if self.minimum_teacher_delta < 0:
            raise ValueError("minimum_teacher_delta must be non-negative")
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self.register_buffer("confidence_lower", torch.tensor(float("nan")))
        self.register_buffer("confidence_upper", torch.tensor(float("nan")))
        self.register_buffer("background_upper", torch.tensor(float("nan")))
        self.register_buffer("confidence_delta_lower", torch.tensor(float("nan")))
        self.register_buffer("confidence_delta_upper", torch.tensor(float("nan")))

    def train(self, mode: bool = True):
        super().train(False)
        self.teacher.eval()
        return self

    def _extract(self, pixels: torch.Tensor):
        teacher_parameter = next(self.teacher.parameters(), None)
        if teacher_parameter is not None:
            pixels = pixels.to(
                device=teacher_parameter.device,
                dtype=teacher_parameter.dtype,
            )
        outputs = self.teacher(pixels)
        if not isinstance(outputs, (tuple, list)):
            raise RuntimeError("DHFG teacher must return a tuple of tensors")
        outputs = list(outputs)
        if len(outputs) != 2 * self.num_scales:
            raise RuntimeError(
                f"Expected {2 * self.num_scales} teacher outputs, got {len(outputs)}"
            )
        return outputs[: self.num_scales], outputs[self.num_scales :]

    def _measure(self, pixels: torch.Tensor, mask: torch.Tensor):
        if pixels.shape[-2:] != (self.input_size, self.input_size):
            pixels = F.interpolate(
                pixels, (self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
        features, logits = self._extract(pixels.clamp(0, 1))
        responses, confidences, backgrounds = [], [], []
        for feature, class_logits in zip(features, logits):
            class_logit = class_logits[:, self.class_id : self.class_id + 1].float()
            response = class_logit.sigmoid()
            local_mask = _resize_support(mask, response.shape[-2:])
            responses.append(response)

            # Use pre-sigmoid scores for the hardness interval.  Tiny-defect
            # teachers can assign very small probabilities; probability-space
            # squared error would then have almost no useful gradient.
            confidences.append(
                _masked_topk_mean(
                    class_logit,
                    local_mask,
                    self.topk_fraction,
                    self.maximum_object_locations,
                    empty_value=-1e4,
                )
            )
            outside = 1.0 - local_mask
            backgrounds.append(
                _masked_topk_mean(
                    response,
                    outside,
                    self.background_topk_fraction,
                    self.maximum_background_locations,
                    empty_value=0.0,
                )
            )
        confidence = torch.stack(confidences, dim=1).amax(1)
        # A false positive on any detector scale matters; averaging scales can
        # hide exactly the sparse response DHFG is intended to suppress.
        background = torch.stack(backgrounds, dim=1).amax(1)
        return {
            "features": features,
            "responses": responses,
            "confidence": confidence,
            "background": background,
        }

    @torch.no_grad()
    def calibrate(self, pixel_batches, mask_batches, low_quantile=0.20, high_quantile=0.80):
        confidence_values, confidence_delta_values, background_values = [], [], []
        for pixels, masks in zip(pixel_batches, mask_batches):
            measurement = self._measure(pixels, masks)
            confidence_values.append(measurement["confidence"].float().cpu())
            background_values.append(measurement["background"].float().cpu())
            signature = _local_photometric_signature(
                pixels,
                masks,
                context_radius=self.boundary_radius,
                contrast_threshold=self.contrast_threshold,
                contrast_temperature=self.contrast_temperature,
            )
            support = _counterfactual_support(
                measurement["responses"],
                signature["contrast_map"],
                masks,
                pixels.shape[-2:],
                erasure_radius=self.erasure_radius,
            )
            erased = counterfactual_erase(
                pixels, support, blur_radius=self.erasure_blur_radius
            )
            erased_measurement = self._measure(erased, masks)
            confidence_delta_values.append(
                (measurement["confidence"] - erased_measurement["confidence"])
                .float()
                .cpu()
            )
        if not confidence_values:
            raise ValueError("DHFG calibration received no samples")
        confidence = torch.cat(confidence_values).to(self.confidence_lower.device)
        confidence_delta = torch.cat(confidence_delta_values).to(
            self.confidence_lower.device
        )
        background = torch.cat(background_values).to(self.confidence_lower.device)
        self.confidence_lower.copy_(torch.quantile(confidence, float(low_quantile)))
        self.confidence_upper.copy_(torch.quantile(confidence, float(high_quantile)))
        self.confidence_delta_lower.copy_(
            torch.quantile(confidence_delta, float(low_quantile))
        )
        self.confidence_delta_upper.copy_(
            torch.quantile(confidence_delta, float(high_quantile))
        )
        self.background_upper.copy_(torch.quantile(background, 0.90))
        lower_probability = self.confidence_lower.float().sigmoid()
        upper_probability = self.confidence_upper.float().sigmoid()
        if float(upper_probability) < self.minimum_teacher_probability:
            raise ValueError(
                "DHFG teacher cannot recognise real defects at its exported input "
                f"scale: calibrated probability interval is "
                f"[{float(lower_probability):.6g}, {float(upper_probability):.6g}], "
                f"below minimum {self.minimum_teacher_probability:.6g}. "
                "Export the teacher at input_size=512 or train a real-only teacher "
                "on the 512x512 defect crops before diffusion training."
            )
        if float(self.confidence_delta_upper) < self.minimum_teacher_delta:
            raise ValueError(
                "DHFG teacher does not distinguish real defects from their "
                "counterfactual erasures: calibrated confidence delta upper "
                f"bound is {float(self.confidence_delta_upper):.6g}, below "
                f"{self.minimum_teacher_delta:.6g}. Check the teacher, mask, "
                "or erasure support before starting diffusion training."
            )
        return {
            "confidence_lower": float(self.confidence_lower.cpu()),
            "confidence_upper": float(self.confidence_upper.cpu()),
            "confidence_probability_lower": float(lower_probability.cpu()),
            "confidence_probability_upper": float(upper_probability.cpu()),
            "confidence_delta_lower": float(self.confidence_delta_lower.cpu()),
            "confidence_delta_upper": float(self.confidence_delta_upper.cpu()),
            "background_upper": float(self.background_upper.cpu()),
        }

    def guidance_losses(
        self,
        generated_pixels: torch.Tensor,
        reference_pixels: torch.Tensor,
        mask: torch.Tensor,
    ):
        if not torch.isfinite(self.confidence_lower):
            raise RuntimeError("DHFG teacher must be calibrated before training")
        with torch.no_grad():
            reference = self._measure(reference_pixels, mask)
            reference_signature = _local_photometric_signature(
                reference_pixels,
                mask,
                context_radius=self.boundary_radius,
                contrast_threshold=self.contrast_threshold,
                contrast_temperature=self.contrast_temperature,
            )
            support = _counterfactual_support(
                reference["responses"],
                reference_signature["contrast_map"],
                mask,
                reference_pixels.shape[-2:],
                erasure_radius=self.erasure_radius,
            )
            reference_erased = counterfactual_erase(
                reference_pixels, support, blur_radius=self.erasure_blur_radius
            )
            reference_erased_measurement = self._measure(reference_erased, mask)
            reference_residual = (
                reference_pixels.float() - reference_erased.float()
            ) * 0.5 + 0.5
            reference_residual_signature = _local_photometric_signature(
                reference_residual,
                mask,
                context_radius=self.boundary_radius,
                contrast_threshold=self.contrast_threshold,
                contrast_temperature=self.contrast_temperature,
            )
        generated = self._measure(generated_pixels, mask)
        # The erased branch is a counterfactual baseline, not a second target
        # generator.  Stop-gradient keeps only one detector forward graph and
        # makes v5 fit the same 24 GB training setup as v3/v4.
        generated_erased = counterfactual_erase(
            generated_pixels, support, blur_radius=self.erasure_blur_radius
        ).detach()
        with torch.no_grad():
            generated_erased_measurement = self._measure(generated_erased, mask)
        generated_residual = (
            generated_pixels.float() - generated_erased.float()
        ) * 0.5 + 0.5
        generated_residual_signature = _local_photometric_signature(
            generated_residual,
            mask,
            context_radius=self.boundary_radius,
            contrast_threshold=self.contrast_threshold,
            contrast_temperature=self.contrast_temperature,
        )

        generated_confidence_delta = (
            generated["confidence"] - generated_erased_measurement["confidence"]
        )
        reference_confidence_delta = (
            reference["confidence"] - reference_erased_measurement["confidence"]
        ).detach()
        clipped_reference_delta = reference_confidence_delta.clamp(
            min=self.confidence_delta_lower.to(reference_confidence_delta),
            max=self.confidence_delta_upper.to(reference_confidence_delta),
        )
        confidence_delta_loss = paired_confidence_loss(
            generated_confidence_delta,
            clipped_reference_delta,
            tolerance=self.delta_tolerance,
            beta=self.confidence_beta,
        )

        feature_delta_loss = generated_pixels.new_zeros((), dtype=torch.float32)
        response_delta_loss = generated_pixels.new_zeros((), dtype=torch.float32)
        for generated_feature, generated_erased_feature, reference_feature, reference_erased_feature in zip(
            generated["features"],
            generated_erased_measurement["features"],
            reference["features"],
            reference_erased_measurement["features"],
        ):
            feature_delta_loss = feature_delta_loss + _masked_smooth_l1(
                generated_feature - generated_erased_feature,
                reference_feature - reference_erased_feature,
                support,
            )
        for generated_response, generated_erased_response, reference_response, reference_erased_response in zip(
            generated["responses"],
            generated_erased_measurement["responses"],
            reference["responses"],
            reference_erased_measurement["responses"],
        ):
            response_delta_loss = response_delta_loss + _masked_smooth_l1(
                generated_response - generated_erased_response,
                reference_response - reference_erased_response,
                support,
            )
        scale_count = max(len(generated["features"]), 1)
        feature_delta_loss = feature_delta_loss / scale_count
        response_delta_loss = response_delta_loss / scale_count

        polarity_loss = sum(
            F.smooth_l1_loss(
                generated_residual_signature[name],
                reference_residual_signature[name].detach(),
                beta=0.1,
            )
            for name in ("positive", "negative", "chroma")
        )
        shape_loss = sum(
            F.smooth_l1_loss(
                generated_residual_signature[name],
                reference_residual_signature[name].detach(),
                beta=0.05,
            )
            for name in ("coverage", "extent")
        )
        background_loss = F.relu(
            generated["background"]
            - self.background_upper.to(generated["background"])
        ).square().mean()
        boundary_loss = boundary_preservation_loss(
            generated_pixels,
            reference_pixels.detach(),
            mask,
            radius=self.boundary_radius,
        )
        return {
            "feature_delta": feature_delta_loss,
            "response_delta": response_delta_loss,
            "confidence_delta_loss": confidence_delta_loss,
            "polarity": polarity_loss,
            "shape": shape_loss,
            "boundary": boundary_loss,
            "false_background": background_loss,
            "confidence": generated["confidence"].mean().detach(),
            "reference_confidence": reference["confidence"].mean().detach(),
            "confidence_delta": generated_confidence_delta.mean().detach(),
            "reference_confidence_delta": reference_confidence_delta.mean().detach(),
            "support": support.mean().detach(),
            "background_response": generated["background"].mean().detach(),
            "positive": generated_residual_signature["positive"].mean().detach(),
            "negative": generated_residual_signature["negative"].mean().detach(),
            "coverage": generated_residual_signature["coverage"].mean().detach(),
            "extent": generated_residual_signature["extent"].mean().detach(),
        }
