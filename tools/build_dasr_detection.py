"""Build a detector-aware synthetic rebalancing (DASR) YOLO benchmark.

The builder never trusts the coarse diffusion mask as a detector annotation.
It extracts the actual generated residual, rejects missing/outlier candidates,
restores a physical size sampled from real YOLO boxes, and selects a fixed
class-balanced subset using realism, optional probe-detector hardness, a real
weakness profile, and feature-space diversity.  Synthetic images are added to
training only; real validation data is copied unchanged to every branch.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from .build_glrr_stage1 import (
        defect_residual,
        image_files,
        is_valid_background,
        matching_file,
        resolve_result_root,
        valid_locations,
        yolo_box,
    )
    from .hgrr_recomposition import RealBoxPrior, recompose_geometry
except ImportError:
    from build_glrr_stage1 import (
        defect_residual,
        image_files,
        is_valid_background,
        matching_file,
        resolve_result_root,
        valid_locations,
        yolo_box,
    )
    from hgrr_recomposition import RealBoxPrior, recompose_geometry


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFECTS = ("01_Flash_point", "02_Big_black_spots")
CLASS_IDS = {"01_Flash_point": 0, "02_Big_black_spots": 1}
ATTRIBUTE_NAMES = ("positive", "negative", "chroma")


@dataclass
class Candidate:
    defect: str
    stem: str
    image_path: Path
    normal_path: Path
    mask_path: Path
    residual: np.ndarray
    alpha: np.ndarray
    source_box: tuple[int, int, int, int]
    metrics: dict[str, float]
    feature: np.ndarray
    realism: float = 0.0
    weakness: float = 1.0
    probe_confidence: float | None = None
    probe_iou: float | None = None
    hardness: float = 1.0
    base_score: float = 0.0
    selection_score: float = 0.0
    reject_reason: str = ""
    composition: dict = field(default_factory=dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", type=Path, required=True,
                        help="Pure-real YOLO root with images/labels train and val.")
    parser.add_argument("--generated_root", type=Path, action="append", required=True,
                        help="Repeat for each MSDF seed result root.")
    parser.add_argument("--background_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="Plastic_Bomo")
    parser.add_argument("--flash_count", type=int, default=80)
    parser.add_argument("--black_count", type=int, default=80)
    parser.add_argument("--max_candidates_per_class", type=int, default=None,
                        help="Optional deterministic cap, mainly for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe_weights", type=Path, default=None,
                        help="Optional Real-only/OOF YOLO weights for candidate hardness.")
    parser.add_argument("--probe_device", default="0")
    parser.add_argument("--probe_imgsz", type=int, default=1536)
    parser.add_argument("--weakness_profile", type=Path, default=None,
                        help="Optional JSON written by profile_dasr_weakness.py.")
    parser.add_argument("--threshold_mad_scale", type=float, default=3.0)
    parser.add_argument("--min_support_pixels", type=int, default=6)
    parser.add_argument("--max_support_ratio", type=float, default=0.45)
    parser.add_argument("--max_outside_mae", type=float, default=4.0)
    parser.add_argument("--attribute_margin", type=float, default=1.0,
                        help="Expand real q05-q95 bounds by this many IQRs.")
    parser.add_argument("--feather_radius", type=float, default=2.0)
    parser.add_argument("--core_threshold", type=float, default=0.50)
    parser.add_argument("--context_strength", type=float, default=0.50)
    parser.add_argument("--hardness_target", type=float, default=0.30)
    parser.add_argument("--hardness_sigma", type=float, default=0.25)
    parser.add_argument("--min_background_mean", type=float, default=30.0)
    parser.add_argument("--min_nonblack_ratio", type=float, default=0.80)
    parser.add_argument("--allow_background_reuse", action="store_true")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gradient_mean(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def robust_attribute_bounds(rows: list[dict[str, float]], margin: float):
    output = {}
    for name in ATTRIBUTE_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=np.float32)
        q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        iqr = max(float(q75 - q25), 1.0)
        output[name] = {
            "lower": max(0.0, float(q05 - margin * iqr)),
            "upper": float(q95 + margin * iqr),
            "median": float(median),
            "scale": iqr,
        }
    return output


def real_attribute_priors(real_root: Path):
    by_defect = {defect: [] for defect in DEFECTS}
    image_dir = real_root / "images" / "train"
    label_dir = real_root / "labels" / "train"
    images = {path.stem: path for path in image_files(image_dir)}
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = images.get(label_path.stem)
        if image_path is None:
            continue
        image = load_rgb(image_path)
        height, width = image.shape[:2]
        gray = luminance(image)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5 or int(fields[0]) not in (0, 1):
                continue
            class_id = int(fields[0])
            defect = DEFECTS[class_id]
            cx, cy, bw, bh = map(float, fields[1:])
            x0 = max(0, int(round((cx - bw / 2) * width)))
            x1 = min(width, int(round((cx + bw / 2) * width)))
            y0 = max(0, int(round((cy - bh / 2) * height)))
            y1 = min(height, int(round((cy + bh / 2) * height)))
            if x1 <= x0 or y1 <= y0:
                continue
            pad_x, pad_y = max(2, x1 - x0), max(2, y1 - y0)
            rx0, rx1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
            ry0, ry1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
            ring_mask = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
            ring_mask[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False
            ring = image[ry0:ry1, rx0:rx1][ring_mask]
            if not len(ring):
                continue
            core = image[y0:y1, x0:x1]
            background_rgb = ring.mean(0)
            delta = core - background_rgb
            delta_luma = luminance(delta)
            chroma = np.sqrt(
                (delta[..., 0] - delta[..., 1]) ** 2
                + (delta[..., 2] - delta[..., 1]) ** 2
            )
            by_defect[defect].append({
                "positive": float(np.maximum(delta_luma, 0).mean()),
                "negative": float(np.maximum(-delta_luma, 0).mean()),
                "chroma": float(chroma.mean()),
                "area_fraction": float(bw * bh),
                "aspect": float(bw * width / max(bh * height, 1e-6)),
                "texture": gradient_mean(gray[ry0:ry1, rx0:rx1]),
            })
    missing = [defect for defect, rows in by_defect.items() if not rows]
    if missing:
        raise ValueError(f"No real training boxes found for: {missing}")
    return by_defect


def extract_candidate(base: Path, defect: str, image_path: Path, args) -> Candidate | None:
    normal_path = matching_file(base / "normal", image_path.stem)
    mask_path = matching_file(base / "masks", image_path.stem)
    result = defect_residual(
        image_path, normal_path, mask_path,
        args.threshold_mad_scale, args.min_support_pixels,
        args.max_support_ratio, args.feather_radius,
    )
    if result is None:
        return None
    residual, alpha, source_box = result
    generated = load_rgb(image_path)
    normal = load_rgb(normal_path)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if mask.shape != generated.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), generated.shape[1::-1],
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    full_residual = generated - normal
    outside_mae = float(np.abs(full_residual[~mask]).mean()) if (~mask).any() else 0.0
    support = alpha >= args.core_threshold
    if not support.any():
        support = alpha > 0.05
    luma_residual = luminance(residual)
    selected_luma = luma_residual[support]
    selected_residual = residual[support]
    chroma = np.sqrt(
        (selected_residual[:, 0] - selected_residual[:, 1]) ** 2
        + (selected_residual[:, 2] - selected_residual[:, 1]) ** 2
    )
    x0, y0, x1, y1 = source_box
    source_normal = normal[y0:y1, x0:x1]
    local_gray = luminance(source_normal)
    ys, xs = np.where(support)
    support_width = int(xs.max() - xs.min() + 1)
    support_height = int(ys.max() - ys.min() + 1)
    metrics = {
        "positive": float(np.maximum(selected_luma, 0).mean()),
        "negative": float(np.maximum(-selected_luma, 0).mean()),
        "chroma": float(chroma.mean()),
        "outside_mae": outside_mae,
        "support_pixels": int(support.sum()),
        "support_ratio": float(support.sum() / max(mask.sum(), 1)),
        "aspect": float(support_width / max(support_height, 1)),
        "normal_mean": float(local_gray.mean()),
        "normal_std": float(local_gray.std()),
        "normal_gradient": gradient_mean(local_gray),
    }
    feature = np.asarray([
        metrics["positive"], metrics["negative"], metrics["chroma"],
        math.log1p(metrics["support_pixels"]), math.log(max(metrics["aspect"], 1e-6)),
        metrics["normal_mean"], metrics["normal_std"], metrics["normal_gradient"],
    ], dtype=np.float32)
    return Candidate(
        defect=defect, stem=image_path.stem, image_path=image_path,
        normal_path=normal_path, mask_path=mask_path, residual=residual,
        alpha=alpha, source_box=source_box, metrics=metrics, feature=feature,
    )


def quality_gate(candidate: Candidate, prior_rows, bounds, args) -> str:
    metrics = candidate.metrics
    if metrics["support_pixels"] < args.min_support_pixels:
        return "missing_support"
    if metrics["support_ratio"] > args.max_support_ratio:
        return "support_too_large"
    if metrics["outside_mae"] > args.max_outside_mae:
        return "background_leakage"
    for name in ATTRIBUTE_NAMES:
        if not bounds[name]["lower"] <= metrics[name] <= bounds[name]["upper"]:
            return f"{name}_outlier"
    real_positive = np.median([row["positive"] for row in prior_rows])
    real_negative = np.median([row["negative"] for row in prior_rows])
    if real_positive > 1.25 * real_negative and metrics["positive"] <= metrics["negative"]:
        return "wrong_dark_polarity"
    if real_negative > 1.25 * real_positive and metrics["negative"] <= metrics["positive"]:
        return "wrong_bright_polarity"
    return ""


def realism_score(candidate: Candidate, bounds) -> float:
    squared = []
    for name in ATTRIBUTE_NAMES:
        item = bounds[name]
        squared.append(((candidate.metrics[name] - item["median"]) / item["scale"]) ** 2)
    return float(math.exp(-0.5 * min(float(np.mean(squared)), 25.0)))


def load_weakness(path: Path | None):
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", -1)) != 1 or not isinstance(data.get("bins"), list):
        raise ValueError(f"Unsupported DASR weakness profile: {path}")
    return data["bins"]


def weakness_weight(class_id: int, area_fraction: float, bins) -> float:
    for item in bins:
        if int(item["class_id"]) == class_id and float(item["area_low"]) <= area_fraction <= float(item["area_high"]):
            return float(item["weight"])
    return 1.0


def link_or_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_real(real_root: Path, destination: Path):
    for split in ("train", "val"):
        for image in image_files(real_root / "images" / split):
            link_or_copy(image, destination / "images" / split / image.name)
            label = real_root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing real label: {label}")
            link_or_copy(label, destination / "labels" / split / label.name)


def compose(candidate: Candidate, background_path: Path, target_wh, args):
    residual, alpha, label_support, geometry = recompose_geometry(
        candidate.residual, candidate.alpha, target_wh,
        core_threshold=args.core_threshold,
        context_strength=args.context_strength,
    )
    background = load_rgb(background_path)
    location = valid_locations(
        background, residual.shape[:2],
        random.Random(args.seed + int(hashlib.sha256(f"{candidate.defect}:{candidate.stem}:{background_path.name}".encode()).hexdigest()[:8], 16)),
    )
    if location is None:
        raise RuntimeError(f"No valid DASR placement on {background_path}")
    x, y = location
    height, width = residual.shape[:2]
    output = background.copy()
    target = output[y:y + height, x:x + width]
    target += residual * alpha[..., None]
    output[y:y + height, x:x + width] = np.clip(target, 0, 255)
    box = yolo_box(x, y, label_support, output.shape[1], output.shape[0], args.core_threshold)
    cx, cy, bw, bh = box
    xyxy = [
        (cx - bw / 2) * output.shape[1], (cy - bh / 2) * output.shape[0],
        (cx + bw / 2) * output.shape[1], (cy + bh / 2) * output.shape[0],
    ]
    return output.astype(np.uint8), box, xyxy, geometry


def box_iou(box, boxes):
    if not len(boxes):
        return np.zeros(0, dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(1)
    area_a = np.maximum(box[2:] - box[:2], 0).prod()
    area_b = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-6)


def score_with_probe(candidates, image_paths, args):
    from ultralytics import YOLO

    model = YOLO(str(args.probe_weights))
    results = model.predict(
        source=[str(path) for path in image_paths], imgsz=args.probe_imgsz,
        device=args.probe_device, conf=0.001, verbose=False, stream=False,
    )
    if len(results) != len(candidates):
        raise RuntimeError("Probe result count does not match DASR candidate count")
    for candidate, result in zip(candidates, results):
        class_id = CLASS_IDS[candidate.defect]
        if result.boxes is None or len(result.boxes) == 0:
            confidence, best_iou = 0.0, 0.0
        else:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            keep = classes == class_id
            boxes = result.boxes.xyxy.detach().cpu().numpy()[keep]
            confidences = result.boxes.conf.detach().cpu().numpy()[keep]
            ious = box_iou(candidate.composition["xyxy"], boxes)
            if len(ious):
                index = int(np.argmax(ious * np.maximum(confidences, 1e-6)))
                confidence, best_iou = float(confidences[index]), float(ious[index])
            else:
                confidence, best_iou = 0.0, 0.0
        candidate.probe_confidence = confidence
        candidate.probe_iou = best_iou
        candidate.hardness = float(math.exp(
            -0.5 * ((confidence - args.hardness_target) / max(args.hardness_sigma, 1e-6)) ** 2
        ))


def diverse_select(candidates: list[Candidate], count: int) -> list[Candidate]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} valid unique candidates, found {len(candidates)}")
    features = np.stack([candidate.feature for candidate in candidates])
    median = np.median(features, axis=0)
    scale = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
    features = (features - median) / np.maximum(scale, 1e-6)
    base = np.asarray([candidate.base_score for candidate in candidates], dtype=np.float32)
    selected = [int(np.argmax(base))]
    distances = np.linalg.norm(features - features[selected[0]], axis=1)
    while len(selected) < count:
        diversity = distances / max(float(distances.max()), 1e-6)
        score = base * (0.25 + 0.75 * diversity)
        score[selected] = -1
        index = int(np.argmax(score))
        candidates[index].selection_score = float(score[index])
        selected.append(index)
        distances = np.minimum(distances, np.linalg.norm(features - features[index], axis=1))
    candidates[selected[0]].selection_score = float(base[selected[0]])
    return [candidates[index] for index in selected]


def write_yaml(output: Path, group: str):
    (output / f"{group}.yaml").write_text(
        f"path: {(output.resolve() / group).as_posix()}\n"
        "train: images/train\nval: images/val\n\n"
        "names:\n  0: Flash point\n  1: Big black spots\n",
        encoding="utf-8",
    )


def hash_validation(group_root: Path):
    output = []
    for image in image_files(group_root / "images" / "val"):
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        label = group_root / "labels" / "val" / f"{image.stem}.txt"
        output.append((image.name, digest, hashlib.sha256(label.read_bytes()).hexdigest()))
    return output


def audit_row(candidate: Candidate, selected: bool):
    row = {
        "defect": candidate.defect, "stem": candidate.stem,
        "source_image": str(candidate.image_path),
        "selected": int(selected), "reject_reason": candidate.reject_reason,
        "realism": candidate.realism, "weakness": candidate.weakness,
        "hardness": candidate.hardness, "base_score": candidate.base_score,
        "selection_score": candidate.selection_score,
        "probe_confidence": candidate.probe_confidence,
        "probe_iou": candidate.probe_iou,
    }
    row.update(candidate.metrics)
    return row


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")
    if args.flash_count < 1 or args.black_count < 1:
        raise ValueError("DASR class counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    real_rows = real_attribute_priors(args.real_root)
    bounds = {
        defect: robust_attribute_bounds(real_rows[defect], args.attribute_margin)
        for defect in DEFECTS
    }
    weakness_bins = load_weakness(args.weakness_profile)
    result_roots = [
        resolve_result_root(root, args.category) for root in args.generated_root
    ]
    all_candidates: list[Candidate] = []
    rejected_rows = []
    for result_root in result_roots:
        for defect in DEFECTS:
            base = result_root / args.category / defect
            source_images = image_files(base / "image")
            if args.max_candidates_per_class is not None:
                source_images = source_images[:args.max_candidates_per_class]
            for image_path in source_images:
                candidate = extract_candidate(base, defect, image_path, args)
                if candidate is None:
                    rejected_rows.append({
                        "defect": defect, "stem": image_path.stem,
                        "source_image": str(image_path), "reject_reason": "missing_residual",
                    })
                    continue
                candidate.reject_reason = quality_gate(candidate, real_rows[defect], bounds[defect], args)
                candidate.realism = realism_score(candidate, bounds[defect])
                if candidate.reject_reason:
                    rejected_rows.append(audit_row(candidate, False))
                else:
                    all_candidates.append(candidate)

    requested = {DEFECTS[0]: args.flash_count, DEFECTS[1]: args.black_count}
    available = {
        defect: sum(candidate.defect == defect for candidate in all_candidates)
        for defect in DEFECTS
    }
    shortages = {
        defect: (available[defect], count)
        for defect, count in requested.items()
        if available[defect] < count
    }
    if shortages:
        reasons = {
            defect: dict(Counter(
                row.get("reject_reason", "unknown")
                for row in rejected_rows
                if row.get("defect") == defect
            ))
            for defect in shortages
        }
        details = "; ".join(
            f"{defect}: valid={valid}, requested={needed}, rejected={reasons[defect]}"
            for defect, (valid, needed) in shortages.items()
        )
        raise ValueError(
            "Insufficient quality-gated DASR candidates. " + details
            + ". Generate more independent inference seeds and repeat --generated_root; "
              "do not duplicate the same synthetic defect."
        )

    backgrounds = [
        path for path in image_files(args.background_dir)
        if is_valid_background(path, args.min_background_mean, args.min_nonblack_ratio)
    ]
    selected_total = args.flash_count + args.black_count
    if len(backgrounds) < selected_total and not args.allow_background_reuse:
        raise ValueError(
            f"Need {selected_total} unique valid backgrounds for the selected DASR set, "
            f"found {len(backgrounds)}; add backgrounds or pass --allow_background_reuse"
        )
    if not backgrounds:
        raise ValueError("No valid DASR background image")
    rng = random.Random(args.seed)
    rng.shuffle(backgrounds)

    geometry_prior = RealBoxPrior.from_yolo(
        args.real_root / "labels" / "train", (0, 1), DEFECTS, 4096, 1024
    )
    composed_images: list[Path | None] = [None] * len(all_candidates)
    candidate_indices = {id(candidate): index for index, candidate in enumerate(all_candidates)}
    with tempfile.TemporaryDirectory(prefix="dasr_pool_", dir=args.output) as temporary:
        temporary = Path(temporary)
        for defect in DEFECTS:
            subset = [candidate for candidate in all_candidates if candidate.defect == defect]
            targets = geometry_prior.targets(defect, len(subset), args.seed + CLASS_IDS[defect] * 100003)
            for local_index, (candidate, target_wh) in enumerate(zip(subset, targets)):
                global_index = candidate_indices[id(candidate)]
                background = backgrounds[global_index % len(backgrounds)]
                image, box, xyxy, geometry = compose(candidate, background, target_wh, args)
                area_fraction = float(box[2] * box[3])
                candidate.weakness = weakness_weight(CLASS_IDS[defect], area_fraction, weakness_bins)
                candidate.composition = {
                    "background": str(background), "target_wh": list(map(float, target_wh)),
                    "yolo_box": list(map(float, box)), "xyxy": list(map(float, xyxy)),
                    "geometry": geometry, "area_fraction": area_fraction,
                }
                candidate.feature = np.concatenate((
                    candidate.feature,
                    np.asarray([
                        math.log(max(area_fraction, 1e-12)),
                        math.log(max(float(target_wh[0]) / max(float(target_wh[1]), 1e-6), 1e-6)),
                    ], dtype=np.float32),
                ))
                path = temporary / f"candidate_{global_index:05d}.jpg"
                Image.fromarray(image).save(path, quality=95, subsampling=0)
                composed_images[global_index] = path

        if any(path is None for path in composed_images):
            raise RuntimeError("Some DASR candidates were not composed")
        composed_images = [path for path in composed_images if path is not None]
        if args.probe_weights is not None:
            score_with_probe(all_candidates, composed_images, args)
        for candidate in all_candidates:
            candidate.base_score = max(candidate.realism, 1e-6) * candidate.weakness * candidate.hardness

        selected = []
        for defect, count in ((DEFECTS[0], args.flash_count), (DEFECTS[1], args.black_count)):
            selected.extend(diverse_select(
                [candidate for candidate in all_candidates if candidate.defect == defect], count
            ))
        selected_ids = {id(candidate) for candidate in selected}

        for group in ("real_only", "real_dasr"):
            copy_real(args.real_root, args.output / group)
            write_yaml(args.output, group)
        for index, candidate in enumerate(selected):
            class_id = CLASS_IDS[candidate.defect]
            name = f"dasr_{candidate.defect}_{index:04d}"
            target_image = args.output / "real_dasr" / "images" / "train" / f"{name}.jpg"
            target_label = args.output / "real_dasr" / "labels" / "train" / f"{name}.txt"
            # Recompose the chosen subset on a unique background. Candidate-pool
            # scoring may reuse backgrounds, but the final training set does not.
            background = backgrounds[index % len(backgrounds)]
            target_wh = tuple(candidate.composition["target_wh"])
            image, box, xyxy, geometry = compose(candidate, background, target_wh, args)
            candidate.composition.update({
                "background": str(background), "yolo_box": list(map(float, box)),
                "xyxy": list(map(float, xyxy)), "geometry": geometry,
                "area_fraction": float(box[2] * box[3]),
            })
            Image.fromarray(image).save(target_image, quality=95, subsampling=0)
            box = candidate.composition["yolo_box"]
            target_label.write_text(
                f"{class_id} " + " ".join(f"{value:.8f}" for value in box) + "\n",
                encoding="utf-8",
            )

    if hash_validation(args.output / "real_only") != hash_validation(args.output / "real_dasr"):
        raise RuntimeError("DASR validation data differ from Real-only")

    audit = rejected_rows + [audit_row(candidate, id(candidate) in selected_ids) for candidate in all_candidates]
    fields = sorted({key for row in audit for key in row})
    with (args.output / "candidate_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    summary = {
        "version": 1, "seed": args.seed,
        "real_root": str(args.real_root.resolve()),
        "generated_roots": [str(root.resolve()) for root in result_roots],
        "background_dir": str(args.background_dir.resolve()),
        "requested": requested,
        "extracted": {defect: sum(candidate.defect == defect for candidate in all_candidates) for defect in DEFECTS},
        "selected": {defect: sum(candidate.defect == defect for candidate in selected) for defect in DEFECTS},
        "rejected": len(rejected_rows), "probe_weights": str(args.probe_weights) if args.probe_weights else None,
        "weakness_profile": str(args.weakness_profile) if args.weakness_profile else None,
        "validation_identical": True, "real_attribute_bounds": bounds,
    }
    (args.output / "dasr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DASR dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
