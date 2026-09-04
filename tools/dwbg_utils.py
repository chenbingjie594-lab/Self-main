"""Shared, detector-independent feature utilities for DWBG stages 1 and 2."""

from __future__ import annotations

import numpy as np


BIN_NAMES = {
    "scale": ("tiny", "medium", "large"),
    "contrast": ("low", "medium", "high"),
    "morphology": ("compact", "medium", "elongated"),
}


def compute_bbox_geometry(xyxy, width: int, height: int):
    """Return normalized size and direction-aware/-invariant shape features."""
    x0, y0, x1, y1 = map(float, xyxy)
    pixel_width = max(x1 - x0, 1e-12)
    pixel_height = max(y1 - y0, 1e-12)
    width_fraction = pixel_width / max(float(width), 1e-12)
    height_fraction = pixel_height / max(float(height), 1e-12)
    aspect = pixel_width / pixel_height
    return {
        "area_fraction": float(width_fraction * height_fraction),
        "width_fraction": float(width_fraction),
        "height_fraction": float(height_fraction),
        "aspect_ratio": float(aspect),
        "elongation": float(max(aspect, 1.0 / aspect)),
    }


def context_ring_masks(shape, xyxy, ring_scale=1.5):
    """Construct a clipped defect mask and a non-overlapping local context ring."""
    height, width = map(int, shape[:2])
    x0, y0, x1, y1 = map(float, xyxy)
    ix0 = int(np.clip(np.floor(x0), 0, max(width - 1, 0)))
    iy0 = int(np.clip(np.floor(y0), 0, max(height - 1, 0)))
    ix1 = int(np.clip(np.ceil(x1), ix0 + 1, width))
    iy1 = int(np.clip(np.ceil(y1), iy0 + 1, height))
    box_width, box_height = ix1 - ix0, iy1 - iy0
    target_width = max(box_width + 2, int(np.ceil(box_width * ring_scale)))
    target_height = max(box_height + 2, int(np.ceil(box_height * ring_scale)))
    cx, cy = (ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0
    ex0 = max(0, int(np.floor(cx - target_width / 2)))
    ey0 = max(0, int(np.floor(cy - target_height / 2)))
    ex1 = min(width, int(np.ceil(cx + target_width / 2)))
    ey1 = min(height, int(np.ceil(cy + target_height / 2)))
    defect = np.zeros((height, width), dtype=bool)
    context = np.zeros((height, width), dtype=bool)
    defect[iy0:iy1, ix0:ix1] = True
    context[ey0:ey1, ex0:ex1] = True
    context &= ~defect
    if not context.any():
        nx0, ny0 = max(0, ix0 - 1), max(0, iy0 - 1)
        nx1, ny1 = min(width, ix1 + 1), min(height, iy1 + 1)
        context[ny0:ny1, nx0:nx1] = True
        context &= ~defect
    if not context.any() and height * width > int(defect.sum()):
        yy, xx = np.argwhere(~defect)[0]
        context[yy, xx] = True
    return defect, context


def compute_local_contrast(gray, xyxy, ring_scale=1.5, eps=1e-6):
    """Normalized and raw contrast between bbox core and its local ring."""
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("compute_local_contrast expects a 2-D grayscale image")
    defect, context = context_ring_masks(gray.shape, xyxy, ring_scale)
    defect_values, context_values = gray[defect], gray[context]
    if not len(defect_values) or not len(context_values):
        return {"local_contrast": 0.0, "raw_contrast": 0.0,
                "mu_defect": 0.0, "mu_context": 0.0, "sigma_context": 0.0}
    mu_defect = float(defect_values.mean())
    mu_context = float(context_values.mean())
    sigma_context = float(context_values.std())
    difference = abs(mu_defect - mu_context)
    return {
        "local_contrast": float(difference / (sigma_context + eps)),
        "raw_contrast": float(difference / 255.0),
        "mu_defect": mu_defect,
        "mu_context": mu_context,
        "sigma_context": sigma_context,
    }


def quantile_labels(values, names):
    """Tertile labels, safely merging duplicate edges for small/discrete sets."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [], {"q1": None, "q2": None, "active_bins": [], "merge_mode": "empty"}
    q1, q2 = map(float, np.quantile(values, [1 / 3, 2 / 3]))
    if np.isclose(q1, q2):
        if np.allclose(values, q1):
            labels, merge_mode = [names[1]] * len(values), "all_equal"
        else:
            labels = [names[0] if value <= q1 else names[2] for value in values]
            merge_mode = "two_bins"
    else:
        labels = [names[0] if value <= q1 else names[1] if value <= q2 else names[2]
                  for value in values]
        merge_mode = "three_bins"
    return labels, {"q1": q1, "q2": q2,
                    "active_bins": [name for name in names if name in labels],
                    "merge_mode": merge_mode}


def assign_profile_bin(value, threshold, names):
    """Assign a candidate using a Stage-1 class-specific threshold record."""
    mode = threshold.get("merge_mode")
    active = threshold.get("active_bins", [])
    if mode is None:  # Backward compatibility with Stage-1 JSON written before v2.
        mode = "all_equal" if active == [names[1]] else "two_bins" if len(active) == 2 else "three_bins"
    q1, q2 = threshold.get("q1"), threshold.get("q2")
    if q1 is None or q2 is None:
        raise ValueError("Profile threshold is empty; no real instances for this class")
    value = float(value)
    if mode == "all_equal":
        if np.isclose(value, float(q1)):
            return names[1]
        return names[0] if value < float(q1) else names[2]
    if mode == "two_bins":
        return names[0] if value <= float(q1) else names[2]
    return names[0] if value <= float(q1) else names[1] if value <= float(q2) else names[2]


def native(value):
    """Recursively remove NumPy types before JSON serialization."""
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
