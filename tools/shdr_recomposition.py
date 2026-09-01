"""SHDR: scale-aware high-resolution defect recomposition.

SHDR restores the physical size of a locally generated defect before it is
inserted into a high-resolution image.  The high-confidence core is mapped to
a real training-box geometry, while the uncertain context is resized and
attenuated independently so that scale correction does not erase halos.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .hgrr_recomposition import support_box
except ImportError:
    from hgrr_recomposition import support_box


SHDR_FORMAT_VERSION = 2


def detector_aware_target(
    real_wh,
    detector_imgsz=1536,
    image_wh=(4096, 1024),
    min_visible_short_side=14.0,
    scale_multiplier=1.35,
    max_scale_multiplier=2.0,
):
    """Calibrate a real box for visibility after detector letterboxing.

    ``real_wh`` remains the geometry prior.  The multiplier supplies moderate
    scale diversity, while the visibility floor prevents a tiny real box from
    collapsing to only a few pixels at the detector input resolution.
    """
    real_width = max(1.0, float(real_wh[0]))
    real_height = max(1.0, float(real_wh[1]))
    image_width = max(1.0, float(image_wh[0]))
    image_height = max(1.0, float(image_wh[1]))
    detector_scale = float(detector_imgsz) / max(image_width, image_height)
    visible_short_side = min(real_width, real_height) * detector_scale
    visibility_multiplier = float(min_visible_short_side) / max(
        visible_short_side, 1e-6
    )
    applied_multiplier = min(
        max(float(scale_multiplier), visibility_multiplier, 1.0),
        max(1.0, float(max_scale_multiplier)),
    )
    target = (
        max(1, int(round(real_width * applied_multiplier))),
        max(1, int(round(real_height * applied_multiplier))),
    )
    return target, {
        "real_target_wh": [real_width, real_height],
        "detector_imgsz": int(detector_imgsz),
        "detector_scale": float(detector_scale),
        "visible_short_side_before": float(visible_short_side),
        "visible_short_side_after": float(
            min(target) * detector_scale
        ),
        "scale_multiplier": float(scale_multiplier),
        "visibility_multiplier": float(visibility_multiplier),
        "applied_multiplier": float(applied_multiplier),
    }


def _resize(array, width, height):
    shrinking = width < array.shape[1] or height < array.shape[0]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(array, (int(width), int(height)), interpolation=interpolation)


def _center_add(canvas, patch):
    y = (canvas.shape[0] - patch.shape[0]) // 2
    x = (canvas.shape[1] - patch.shape[1]) // 2
    canvas[y:y + patch.shape[0], x:x + patch.shape[1]] += patch
    return x, y


def recompose_scale_aware(
    residual,
    alpha,
    target_wh,
    core_threshold=0.50,
    context_scale=1.20,
    context_strength=0.40,
):
    """Return a physically calibrated residual, alpha, and core-only label map."""
    residual = np.asarray(residual, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32).clip(0, 1)
    target_width = max(1, int(round(float(target_wh[0]))))
    target_height = max(1, int(round(float(target_wh[1]))))
    x0, y0, x1, y1 = support_box(alpha, core_threshold)
    source_core_width = max(1, x1 - x0)
    source_core_height = max(1, y1 - y0)

    # A soft decomposition avoids a seam between the two independently scaled
    # branches.  Both branches are premultiplied before interpolation.
    core_alpha = np.clip(
        (alpha - float(core_threshold)) / max(1e-6, 1.0 - float(core_threshold)),
        0,
        1,
    )
    context_alpha = np.clip(alpha - core_alpha, 0, 1)
    core_crop = residual[y0:y1, x0:x1] * core_alpha[y0:y1, x0:x1, None]
    core_mask = core_alpha[y0:y1, x0:x1]
    label_mask = (alpha[y0:y1, x0:x1] >= float(core_threshold)).astype(
        np.float32
    )

    cx0, cy0, cx1, cy1 = support_box(alpha, 0.05)
    context_crop = (
        residual[cy0:cy1, cx0:cx1]
        * context_alpha[cy0:cy1, cx0:cx1, None]
    )
    context_mask = context_alpha[cy0:cy1, cx0:cx1]

    resized_core = _resize(core_crop, target_width, target_height)
    resized_core_mask = _resize(core_mask, target_width, target_height).clip(0, 1)
    resized_label_mask = cv2.resize(
        label_mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    context_width = max(target_width, int(round(target_width * float(context_scale))))
    context_height = max(target_height, int(round(target_height * float(context_scale))))
    resized_context = _resize(context_crop, context_width, context_height)
    resized_context_mask = _resize(
        context_mask, context_width, context_height
    ).clip(0, 1)

    canvas_width = max(target_width, context_width)
    canvas_height = max(target_height, context_height)
    premultiplied = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    blend_alpha = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    label_support = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    context_xy = _center_add(
        premultiplied, float(context_strength) * resized_context
    )
    cx, cy = context_xy
    blend_alpha[cy:cy + context_height, cx:cx + context_width] += (
        float(context_strength) * resized_context_mask
    )
    core_xy = _center_add(premultiplied, resized_core)
    kx, ky = core_xy
    blend_alpha[ky:ky + target_height, kx:kx + target_width] += resized_core_mask
    label_support[ky:ky + target_height, kx:kx + target_width] = resized_label_mask
    blend_alpha = blend_alpha.clip(0, 1)

    # build_glrr_stage1 composites residual_out * blend_alpha. Undo the
    # premultiplication here so the product equals the desired combined delta.
    residual_out = np.zeros_like(premultiplied)
    valid = blend_alpha > 1e-6
    residual_out[valid] = premultiplied[valid] / blend_alpha[valid, None]
    label_box = support_box(label_support, core_threshold)
    metadata = {
        "format_version": SHDR_FORMAT_VERSION,
        "target_wh": [target_width, target_height],
        "source_core_wh": [source_core_width, source_core_height],
        "scale_xy": [
            target_width / float(source_core_width),
            target_height / float(source_core_height),
        ],
        "context_scale": float(context_scale),
        "context_strength": float(context_strength),
        "output_patch_wh": [canvas_width, canvas_height],
        "label_box": list(label_box),
    }
    return residual_out, blend_alpha, label_support, metadata
