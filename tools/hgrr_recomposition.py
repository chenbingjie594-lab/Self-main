"""HGRR: high-resolution geometry-consistent residual recomposition.

The module learns a category-specific physical-size prior from real YOLO boxes,
then maps a generated local residual back to full resolution while preserving
its aspect ratio. A high-confidence core defines the detector box; the softer
context is retained only for seamless appearance blending.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RealBoxPrior:
    boxes_by_defect: dict[str, tuple[tuple[float, float], ...]]

    @classmethod
    def from_yolo(
        cls,
        label_dir,
        class_ids,
        defects,
        image_width=4096,
        image_height=1024,
    ):
        collected = {defect: [] for defect in defects}
        id_to_defect = {
            int(class_id): defect for class_id, defect in zip(class_ids, defects)
        }
        for path in sorted(Path(label_dir).glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if not fields:
                    continue
                defect = id_to_defect.get(int(fields[0]))
                if defect is None:
                    continue
                width = float(fields[3]) * float(image_width)
                height = float(fields[4]) * float(image_height)
                if width > 0 and height > 0:
                    collected[defect].append((width, height))
        missing = [defect for defect, boxes in collected.items() if not boxes]
        if missing:
            raise ValueError(f"No real YOLO boxes found for: {missing}")
        return cls({key: tuple(value) for key, value in collected.items()})

    def targets(self, defect, count, seed):
        """Return a deterministic stratified sample of real physical sizes."""
        boxes = list(self.boxes_by_defect[defect])
        rng = random.Random(int(seed))
        rng.shuffle(boxes)
        if count <= len(boxes):
            # Evenly cover the shuffled real distribution without replacement.
            step = len(boxes) / float(count)
            return [boxes[min(int(index * step), len(boxes) - 1)] for index in range(count)]
        return [boxes[index % len(boxes)] for index in range(count)]

    def summary(self, defect):
        values = np.asarray(self.boxes_by_defect[defect], dtype=np.float32)
        return {
            "count": int(len(values)),
            "width_median": float(np.median(values[:, 0])),
            "height_median": float(np.median(values[:, 1])),
            "area_median": float(np.median(values[:, 0] * values[:, 1])),
            "width_range": [float(values[:, 0].min()), float(values[:, 0].max())],
            "height_range": [float(values[:, 1].min()), float(values[:, 1].max())],
        }


def support_box(alpha, threshold):
    ys, xs = np.where(alpha >= float(threshold))
    if not len(xs):
        ys, xs = np.where(alpha > 0)
    if not len(xs):
        raise ValueError("HGRR received an empty alpha support")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def recompose_geometry(
    residual,
    alpha,
    target_wh,
    core_threshold=0.50,
    context_strength=0.50,
):
    """Scale one residual to a real target area and separate core/context."""
    x0, y0, x1, y1 = support_box(alpha, core_threshold)
    core_width = max(1, x1 - x0)
    core_height = max(1, y1 - y0)
    target_width, target_height = map(float, target_wh)
    target_area = max(1.0, target_width * target_height)
    area_scale = np.sqrt(target_area / float(core_width * core_height))
    # Preserve generated aspect ratio but never exceed either physical
    # dimension of the paired real box. This prevents diffuse residuals from
    # becoming detector labels wider/taller than the real category support.
    width_scale = target_width / float(core_width)
    height_scale = target_height / float(core_height)
    scale = min(area_scale, width_scale, height_scale)
    new_width = max(1, int(round(residual.shape[1] * scale)))
    new_height = max(1, int(round(residual.shape[0] * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_residual = cv2.resize(
        residual, (new_width, new_height), interpolation=interpolation
    ).astype(np.float32)
    resized_alpha = cv2.resize(
        alpha.astype(np.float32), (new_width, new_height), interpolation=interpolation
    ).clip(0, 1)

    # Core remains fully expressive. The uncertain halo is attenuated for
    # blending, but is deliberately excluded from the detector annotation.
    core = np.clip(
        (resized_alpha - core_threshold) / max(1e-6, 1.0 - core_threshold), 0, 1
    )
    context = np.clip(resized_alpha - core, 0, 1)
    blend_alpha = np.clip(core + float(context_strength) * context, 0, 1)
    core_box = support_box(resized_alpha, core_threshold)
    metadata = {
        "target_wh": [target_width, target_height],
        "original_core_wh": [core_width, core_height],
        "scale": float(scale),
        "area_scale": float(area_scale),
        "resized_patch_wh": [new_width, new_height],
        "resized_core_box": list(core_box),
    }
    return resized_residual, blend_alpha, resized_alpha, metadata
