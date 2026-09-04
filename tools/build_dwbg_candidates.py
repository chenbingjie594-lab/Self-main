"""Build a valid, high-resolution DWBG synthetic candidate pool from MSDF outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from .build_dasr_detection import (
        CLASS_IDS, DEFECTS, extract_candidate, quality_gate, real_attribute_priors,
        robust_attribute_bounds,
    )
    from .build_glrr_stage1 import image_files, is_valid_background, resolve_result_root, valid_locations, yolo_box
    from .dwbg_utils import BIN_NAMES, assign_profile_bin, compute_bbox_geometry, compute_local_contrast, native
    from .hgrr_recomposition import RealBoxPrior, recompose_geometry
except ImportError:
    from build_dasr_detection import (
        CLASS_IDS, DEFECTS, extract_candidate, quality_gate, real_attribute_priors,
        robust_attribute_bounds,
    )
    from build_glrr_stage1 import image_files, is_valid_background, resolve_result_root, valid_locations, yolo_box
    from dwbg_utils import BIN_NAMES, assign_profile_bin, compute_bbox_geometry, compute_local_contrast, native
    from hgrr_recomposition import RealBoxPrior, recompose_geometry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--real_root", type=Path, required=True)
    parser.add_argument("--background_dir", type=Path, required=True)
    parser.add_argument("--generated_root", nargs=2, metavar=("ROOT", "SEED"), action="append", required=True,
                        help="Repeat once per independently generated MSDF seed.")
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="Plastic_Bomo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_mad_scale", type=float, default=3.0)
    parser.add_argument("--min_support_pixels", type=int, default=6)
    parser.add_argument("--max_support_ratio", type=float, default=0.45)
    parser.add_argument("--max_outside_mae", type=float, default=4.0)
    parser.add_argument("--attribute_margin", type=float, default=1.0)
    parser.add_argument("--feather_radius", type=float, default=2.0)
    parser.add_argument("--core_threshold", type=float, default=0.50)
    parser.add_argument("--context_strength", type=float, default=0.50)
    parser.add_argument("--ring_scale", type=float, default=None,
                        help="Defaults to the exact Stage-1 profile setting.")
    parser.add_argument("--min_background_mean", type=float, default=30.0)
    parser.add_argument("--min_nonblack_ratio", type=float, default=0.80)
    return parser.parse_args()


def rgb_gray(image):
    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def source_seed(root_and_seed):
    root, seed = root_and_seed
    return Path(root), int(seed)


def compose(candidate, background_path, target_wh, args):
    residual, alpha, support, geometry = recompose_geometry(
        candidate.residual, candidate.alpha, target_wh,
        core_threshold=args.core_threshold, context_strength=args.context_strength,
    )
    background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.float32)
    key = f"{candidate.defect}:{candidate.stem}:{background_path}:{args.seed}"
    rng = random.Random(args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    location = valid_locations(background, residual.shape[:2], rng)
    if location is None:
        raise RuntimeError(f"No valid placement on {background_path}")
    x, y = location
    height, width = residual.shape[:2]
    output = background.copy()
    output[y:y + height, x:x + width] = np.clip(
        output[y:y + height, x:x + width] + residual * alpha[..., None], 0, 255
    )
    box = yolo_box(x, y, support, output.shape[1], output.shape[0], args.core_threshold)
    cx, cy, bw, bh = box
    xyxy = [(cx - bw / 2) * output.shape[1], (cy - bh / 2) * output.shape[0],
            (cx + bw / 2) * output.shape[1], (cy + bh / 2) * output.shape[0]]
    return output.astype(np.uint8), box, xyxy, geometry


def profile_thresholds(profile, class_id):
    current = profile["class_profiles"][str(class_id)]
    return current["thresholds"]


def assign_bins(record, thresholds):
    record["scale_bin"] = assign_profile_bin(record["area_fraction"], thresholds["scale"], BIN_NAMES["scale"])
    record["contrast_bin"] = assign_profile_bin(record["local_contrast"], thresholds["contrast"], BIN_NAMES["contrast"])
    record["morphology_bin"] = assign_profile_bin(record["elongation"], thresholds["morphology"], BIN_NAMES["morphology"])
    return record


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")
    if len(args.generated_root) < args.num_seeds:
        raise ValueError(f"--num_seeds={args.num_seeds}, but only {len(args.generated_root)} generated roots were supplied")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if int(profile.get("version", 0)) < 2:
        raise ValueError("DWBG Stage-2 requires the version-2 Stage-1 weakness profile")
    ring_scale = args.ring_scale if args.ring_scale is not None else float(profile["settings"]["ring_scale"])
    args.output.mkdir(parents=True, exist_ok=True)
    image_out, label_out = args.output / "images", args.output / "labels"
    image_out.mkdir(), label_out.mkdir()

    real_rows = real_attribute_priors(args.real_root)
    bounds = {defect: robust_attribute_bounds(real_rows[defect], args.attribute_margin) for defect in DEFECTS}
    backgrounds = [p for p in image_files(args.background_dir)
                   if is_valid_background(p, args.min_background_mean, args.min_nonblack_ratio)]
    if not backgrounds:
        raise ValueError("No valid non-black high-resolution backgrounds")
    random.Random(args.seed).shuffle(backgrounds)
    geometry_prior = RealBoxPrior.from_yolo(args.real_root / "labels" / "train", (0, 1), DEFECTS, 4096, 1024)

    accepted, rejected, candidate_index = [], [], 0
    seed_roots = [(resolve_result_root(root, args.category), seed) for root, seed in map(source_seed, args.generated_root)]
    for result_root, source_seed_value in seed_roots:
        for defect in DEFECTS:
            base = result_root / args.category / defect
            for generated_path in image_files(base / "image"):
                candidate = extract_candidate(base, defect, generated_path, args)
                if candidate is None:
                    rejected.append({"source_image": str(generated_path), "defect": defect, "seed": source_seed_value,
                                     "reject_reason": "missing_residual"})
                    continue
                reason = quality_gate(candidate, real_rows[defect], bounds[defect], args)
                if reason:
                    rejected.append({"source_image": str(generated_path), "defect": defect, "seed": source_seed_value,
                                     "reject_reason": reason, **candidate.metrics})
                    continue
                target_wh = geometry_prior.targets(defect, 1, args.seed + source_seed_value * 1009 + candidate_index)[0]
                background = backgrounds[candidate_index % len(backgrounds)]
                image, box, xyxy, composition = compose(candidate, background, target_wh, args)
                class_id = CLASS_IDS[defect]
                candidate_id = f"dwbg_{class_id}_{source_seed_value}_{candidate.stem}_{candidate_index:06d}"
                image_path, label_path = image_out / f"{candidate_id}.jpg", label_out / f"{candidate_id}.txt"
                Image.fromarray(image).save(image_path, quality=95, subsampling=0)
                label_path.write_text(f"{class_id} " + " ".join(f"{value:.8f}" for value in box) + "\n", encoding="utf-8")
                record = {
                    "candidate_id": candidate_id, "class_id": class_id, "class_name": defect,
                    "source_image": str(generated_path), "reference_image": str(candidate.normal_path),
                    "seed": source_seed_value, "image_path": str(image_path.resolve()), "label_path": str(label_path.resolve()),
                    "bbox": [float(value) for value in box], "bbox_xyxy": [float(value) for value in xyxy],
                    "background": str(background), "composition": composition,
                    **compute_bbox_geometry(xyxy, image.shape[1], image.shape[0]),
                    **compute_local_contrast(rgb_gray(image), xyxy, ring_scale),
                    "source_metrics": candidate.metrics,
                }
                accepted.append(assign_bins(record, profile_thresholds(profile, class_id)))
                candidate_index += 1
    payload = native({
        "version": 1, "profile": str(args.profile.resolve()), "num_seeds": args.num_seeds,
        "ring_scale": ring_scale, "real_root": str(args.real_root.resolve()),
        "candidate_count": len(accepted), "rejected_count": len(rejected),
        "candidates": accepted, "rejected": rejected,
    })
    (args.output / "candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    by_class = {str(cid): sum(c["class_id"] == cid for c in accepted) for cid in (0, 1)}
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "by_class": by_class,
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
