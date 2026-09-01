"""Training-free Material-aware Defect Appearance Prior (MDAP).

The module extracts a signed luminance residual from a real reference defect,
adapts it to a target background, and returns a prior image plus interpretable
physical attributes. It does not contain trainable parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PhysicalDescriptor:
    defect_mode: str
    brightness_change: float
    contrast_ratio: float
    area_ratio: float
    radius: float
    edge_softness: float
    halo_strength: float
    local_variance: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def infer_defect_mode(defect_name: str, mode_map: Optional[Dict[str, str]] = None) -> str:
    if mode_map:
        for key, value in mode_map.items():
            if key.lower() in defect_name.lower():
                return value
    name = defect_name.lower()
    if any(key in name for key in ("flash", "bright", "white", "light")):
        return "bright"
    if any(key in name for key in ("black", "dark", "spot")):
        return "dark"
    raise ValueError(
        f"Cannot infer MDAP defect mode from '{defect_name}'. "
        "Add a modes mapping in the mdap config."
    )


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Reference defect mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fit_background_plane(luminance: np.ndarray, defect_mask: np.ndarray, ring_width: int) -> np.ndarray:
    kernel_size = max(3, int(ring_width) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate((defect_mask > 0).astype(np.uint8), kernel, iterations=1)
    ring = (dilated > 0) & (defect_mask == 0)
    ys, xs = np.where(ring)
    if len(xs) < 12:
        value = float(np.median(luminance[defect_mask == 0]))
        return np.full_like(luminance, value, dtype=np.float32)

    # Robust two-pass least squares for B(x,y)=ax+by+c.
    h, w = luminance.shape
    design = np.stack([xs / max(w - 1, 1), ys / max(h - 1, 1), np.ones_like(xs)], axis=1)
    values = luminance[ys, xs].astype(np.float64)
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    error = np.abs(design @ coeff - values)
    keep = error <= np.percentile(error, 85)
    if keep.sum() >= 12:
        coeff, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
    yy, xx = np.mgrid[0:h, 0:w]
    full = np.stack(
        [xx.ravel() / max(w - 1, 1), yy.ravel() / max(h - 1, 1), np.ones(h * w)],
        axis=1,
    )
    return (full @ coeff).reshape(h, w).astype(np.float32)


def _soft_mask(binary: np.ndarray, blur_sigma: float) -> np.ndarray:
    mask = binary.astype(np.float32)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    maximum = float(mask.max())
    return mask / maximum if maximum > 0 else mask


def build_mdap_prior(
    reference_image: Image.Image,
    reference_mask: Image.Image,
    target_image: Image.Image,
    target_mask: Image.Image,
    defect_mode: str,
    *,
    ring_width: int = 12,
    threshold_sigma: float = 1.0,
    blur_sigma: float = 2.0,
    contrast_min: float = 0.5,
    contrast_max: float = 2.0,
    amplitude: float = 1.0,
) -> Tuple[Image.Image, Image.Image, PhysicalDescriptor]:
    """Create a target-adapted defect prior.

    Args:
        defect_mode: ``dark`` for black spots or ``bright`` for flash points.
    """
    if defect_mode not in {"dark", "bright"}:
        raise ValueError("defect_mode must be 'dark' or 'bright'")

    target_size = target_image.size
    ref_rgb = np.asarray(reference_image.convert("RGB"), dtype=np.uint8)
    ref_mask = np.asarray(reference_mask.convert("L"), dtype=np.uint8) > 127
    tgt_rgb = np.asarray(target_image.convert("RGB"), dtype=np.uint8)
    tgt_mask = np.asarray(target_mask.resize(target_size, Image.Resampling.NEAREST).convert("L")) > 127

    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_l = ref_lab[..., 0]
    tgt_l = tgt_lab[..., 0]
    background = _fit_background_plane(ref_l, ref_mask.astype(np.uint8), ring_width)
    residual = ref_l - background

    kernel = np.ones((max(3, ring_width * 2 + 1),) * 2, np.uint8)
    ring = (cv2.dilate(ref_mask.astype(np.uint8), kernel, iterations=1) > 0) & (~ref_mask)
    ring_std = float(np.std(residual[ring])) if ring.any() else float(np.std(residual))
    threshold = max(1.0, threshold_sigma * ring_std)
    signed = residual < -threshold if defect_mode == "dark" else residual > threshold
    candidate = signed & ref_mask
    if not candidate.any():
        candidate = ref_mask

    x0, y0, x1, y1 = _bbox(candidate)
    residual_crop = residual[y0:y1, x0:x1]
    support_crop = candidate[y0:y1, x0:x1]

    tx0, ty0, tx1, ty1 = _bbox(tgt_mask)
    target_w, target_h = tx1 - tx0, ty1 - ty0
    transferred = cv2.resize(residual_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    transferred_support = cv2.resize(
        support_crop.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    target_support = tgt_mask[ty0:ty1, tx0:tx1].astype(np.float32)
    transferred_support = _soft_mask((transferred_support > 0.25) * target_support, blur_sigma)

    ref_std = max(float(np.std(ref_l[ring])) if ring.any() else float(np.std(ref_l)), 1e-6)
    target_ring = (cv2.dilate(tgt_mask.astype(np.uint8), kernel, iterations=1) > 0) & (~tgt_mask)
    target_std = float(np.std(tgt_l[target_ring])) if target_ring.any() else float(np.std(tgt_l))
    contrast_ratio = float(np.clip(target_std / ref_std, contrast_min, contrast_max))
    transferred *= contrast_ratio * float(amplitude)

    local_target = tgt_l[ty0:ty1, tx0:tx1]
    if defect_mode == "dark":
        transferred = np.minimum(transferred, 0.0)
        transferred = np.maximum(transferred, -local_target)
    else:
        transferred = np.maximum(transferred, 0.0)
        transferred = np.minimum(transferred, 255.0 - local_target)
    transferred *= transferred_support

    prior_lab = tgt_lab.copy()
    prior_lab[ty0:ty1, tx0:tx1, 0] = np.clip(local_target + transferred, 0, 255)
    prior_rgb = cv2.cvtColor(prior_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    full_soft_mask = np.zeros_like(tgt_l, dtype=np.float32)
    full_soft_mask[ty0:ty1, tx0:tx1] = transferred_support
    values = transferred[transferred_support > 0.05]
    area = float((transferred_support > 0.05).sum())
    radius = float(np.sqrt(area / np.pi))
    grad_x = cv2.Sobel(transferred_support, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(transferred_support, cv2.CV_32F, 0, 1)
    edge_softness = float(1.0 / (np.mean(np.sqrt(grad_x**2 + grad_y**2)) + 1e-6))
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    halo = float(np.mean(np.abs(values)) / (peak + 1e-6)) if values.size else 0.0
    descriptor = PhysicalDescriptor(
        defect_mode=defect_mode,
        brightness_change=float(np.mean(values)) if values.size else 0.0,
        contrast_ratio=contrast_ratio,
        area_ratio=area / float(tgt_l.size),
        radius=radius,
        edge_softness=edge_softness,
        halo_strength=halo if defect_mode == "bright" else 0.0,
        local_variance=float(target_std**2),
    )
    mask_image = Image.fromarray(np.clip(full_soft_mask * 255, 0, 255).astype(np.uint8), mode="L")
    return Image.fromarray(prior_rgb, mode="RGB"), mask_image, descriptor


def find_reference_pair(
    dataset_root: str,
    category: str,
    defect_class: str,
    index: int = 0,
) -> Tuple[Path, Path]:
    """Find a deterministic real defect image/mask pair in MVTec layout."""
    root = Path(dataset_root)
    image_dir = root / category / "test" / defect_class
    mask_dir = root / category / "ground_truth" / defect_class
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise FileNotFoundError(f"No reference images found in {image_dir}")
    image_path = images[index % len(images)]
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.jpg",
        mask_dir / f"{image_path.stem}.jpg",
    ]
    mask_path = next((path for path in candidates if path.exists()), None)
    if mask_path is None:
        raise FileNotFoundError(f"No reference mask matching {image_path.name} in {mask_dir}")
    return image_path, mask_path
