"""Pure DWBG-v2 scoring and selection primitives.

The functions in this module deliberately do not import Ultralytics.  This
makes the validity gate and the hard/anchor policy unit-testable on CPU and
keeps model-specific feature extraction in the two CLI tools.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


EPS = 1e-8


def clip_roi_xyxy(xyxy, image_width, image_height, map_width, map_height):
    """Map an image-space box to a valid non-empty feature-map ROI."""
    x0, y0, x1, y1 = map(float, xyxy)
    sx, sy = float(map_width) / image_width, float(map_height) / image_height
    x0, x1 = sorted((x0 * sx, x1 * sx))
    y0, y1 = sorted((y0 * sy, y1 * sy))
    x0 = int(np.clip(math.floor(x0), 0, max(map_width - 1, 0)))
    y0 = int(np.clip(math.floor(y0), 0, max(map_height - 1, 0)))
    x1 = int(np.clip(math.ceil(x1), x0 + 1, map_width))
    y1 = int(np.clip(math.ceil(y1), y0 + 1, map_height))
    return x0, y0, x1, y1


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, EPS)


def cosine_knn_distance(query, bank, k=5, exclude_index=None):
    """Mean cosine distance to k nearest rows; optional LOO exclusion."""
    query = l2_normalize(query)
    bank = np.asarray(bank, dtype=np.float32)
    if bank.ndim != 2 or not len(bank):
        raise ValueError("Feature bank must be a non-empty [N,D] matrix")
    bank = np.asarray([l2_normalize(row) for row in bank], dtype=np.float32)
    distances = 1.0 - np.clip(bank @ query, -1.0, 1.0)
    if exclude_index is not None:
        if not 0 <= int(exclude_index) < len(distances):
            raise IndexError("exclude_index outside feature bank")
        distances[int(exclude_index)] = np.inf
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        raise ValueError("No neighbor remains after leave-one-out exclusion")
    return float(np.mean(np.partition(finite, min(int(k), len(finite)) - 1)[:min(int(k), len(finite))]))


def leave_one_out_distances(bank, k=5):
    bank = np.asarray(bank, dtype=np.float32)
    if len(bank) < 2:
        raise ValueError("At least two real instances are required for leave-one-out kNN")
    return np.asarray([cosine_knn_distance(row, bank, k=k, exclude_index=i)
                       for i, row in enumerate(bank)], dtype=np.float32)


def manifold_reference(bank, k=5, quantile=0.95):
    distances = leave_one_out_distances(bank, k=k)
    return {
        "real_real_distances": distances,
        "median_real_distance": float(np.median(distances)),
        "q90_real_distance": float(np.quantile(distances, .90)),
        "q95_real_distance": float(np.quantile(distances, .95)),
        "threshold_real_distance": float(np.quantile(distances, quantile)),
        "quantile": float(quantile),
        "k": int(k),
    }


def consensus_score(std_confidence, std_iou, sigma_conf, sigma_iou):
    conf = math.exp(-float(std_confidence) / max(float(sigma_conf), EPS))
    iou = math.exp(-float(std_iou) / max(float(sigma_iou), EPS))
    return float(math.sqrt(conf * iou))


def logit(value):
    value = float(np.clip(value, EPS, 1.0 - EPS))
    return math.log(value / (1.0 - value))


def flash_boundary_score(confidence, iou, threshold, sigma_boundary, min_iou, conf_floor=0.05):
    """High only at the decision boundary; invalid localization gets zero."""
    if float(iou) < float(min_iou) or float(confidence) < float(conf_floor):
        return 0.0
    return float(math.exp(-abs(logit(confidence) - logit(threshold)) /
                          max(float(sigma_boundary), EPS)))


def interval_typicality(value, low, high):
    """One in [low,high], smoothly decreases outside the real typical interval."""
    value, low, high = float(value), float(low), float(high)
    if low <= value <= high:
        return 1.0
    width = max(high - low, EPS)
    return float(math.exp(-abs(value - (low if value < low else high)) / width))


def geometric_score(weakness, hardness, manifold, consensus, exponents):
    return float(
        max(float(weakness), EPS) ** float(exponents["weakness_exp"]) *
        max(float(hardness), EPS) ** float(exponents["boundary_exp"]) *
        max(float(manifold), EPS) ** float(exponents["manifold_exp"]) *
        max(float(consensus), EPS) ** float(exponents["consensus_exp"])
    )


def diversity_allowed(candidate, selected, max_per_source, max_per_seed):
    source_count = sum(item["reference_image"] == candidate["reference_image"] for item in selected)
    seed_count = sum(item["seed"] == candidate["seed"] for item in selected)
    return source_count < int(max_per_source) and seed_count < int(max_per_seed)


def diversity_bonus(candidate, selected):
    if not selected:
        return 1.0
    geometry = (candidate["scale_bin"], candidate["contrast_bin"], candidate["morphology_bin"])
    seen = {(x["scale_bin"], x["contrast_bin"], x["morphology_bin"]) for x in selected}
    return 0.05 if geometry not in seen else 0.0


def greedy_unique(pool, count, score_key, selected_prefix, max_per_source, max_per_seed):
    selected, remaining = list(selected_prefix), [dict(x) for x in pool]
    output = []
    while len(output) < count:
        eligible = [x for x in remaining if x["candidate_id"] not in {y["candidate_id"] for y in selected}
                    and diversity_allowed(x, selected, max_per_source, max_per_seed)]
        if not eligible:
            break
        chosen = max(eligible, key=lambda x: (float(x.get(score_key, 0.0)) + diversity_bonus(x, selected), x["candidate_id"]))
        output.append(chosen)
        selected.append(chosen)
    return output


def distribution(items):
    return {key: dict(Counter(x[key] for x in items))
            for key in ("scale_bin", "contrast_bin", "morphology_bin")}
