"""Prepare fixed paired-normal datasets for coarse-mask robustness inference."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VARIANTS = ("original", "erode20", "dilate20", "dilate40", "shift8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_results", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--category", default="Plastic_Bomo")
    parser.add_argument("--samples_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift_pixels", type=int, default=8)
    return parser.parse_args()


def files(folder):
    return sorted(
        (p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )


def matching_file(folder, stem):
    for suffix in IMAGE_SUFFIXES:
        candidate = Path(folder) / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No file for {stem} under {folder}")


def morph_to_ratio(mask, target_ratio, erode):
    original_area = int(mask.sum())
    if original_area == 0:
        raise ValueError("Empty source mask")
    target_area = max(1, int(round(original_area * target_ratio)))
    best = mask.copy()
    best_error = abs(int(best.sum()) - target_area)
    for radius in range(1, max(mask.shape) // 2):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        candidate = cv2.erode(mask, kernel) if erode else cv2.dilate(mask, kernel)
        area = int(candidate.sum())
        if area == 0:
            break
        error = abs(area - target_area)
        if error < best_error:
            best, best_error = candidate, error
        if (erode and area <= target_area) or (not erode and area >= target_area):
            break
    return best


def shift_mask(mask, pixels, direction):
    dx, dy = {"left": (-pixels, 0), "right": (pixels, 0),
              "up": (0, -pixels), "down": (0, pixels)}[direction]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        mask, matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def variants(mask, shift_pixels, direction):
    return {
        "original": mask,
        "erode20": morph_to_ratio(mask, 0.8, True),
        "dilate20": morph_to_ratio(mask, 1.2, False),
        "dilate40": morph_to_ratio(mask, 1.4, False),
        "shift8": shift_mask(mask, shift_pixels, direction),
    }


def main():
    args = parse_args()
    source = Path(args.source_results).resolve() / args.category
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    defect_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    directions = ("left", "right", "up", "down")
    manifest = {
        "source_results": str(source), "category": args.category,
        "samples_per_class": args.samples_per_class, "seed": args.seed,
        "variants": list(VARIANTS), "samples": [],
    }

    for defect in defect_dirs:
        mask_paths = files(defect / "masks")[:args.samples_per_class]
        if not mask_paths:
            continue
        for index, mask_path in enumerate(mask_paths):
            stem = f"{index:04d}"
            normal_path = matching_file(defect / "normal", mask_path.stem)
            mask = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
            direction = directions[(args.seed + index) % len(directions)]
            generated = variants(mask, args.shift_pixels, direction)
            record = {
                "sample_id": stem, "defect": defect.name,
                "source_mask": mask_path.name, "source_normal": normal_path.name,
                "seed_index": index, "shift_direction": direction,
                "areas": {},
            }
            for variant, variant_mask in generated.items():
                category_root = output / variant / args.category
                mask_dir = category_root / "ground_truth" / defect.name
                paired_dir = category_root / "paired_normal" / defect.name
                good_dir = category_root / "train" / "good"
                mask_dir.mkdir(parents=True, exist_ok=True)
                paired_dir.mkdir(parents=True, exist_ok=True)
                good_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(variant_mask * 255).save(mask_dir / f"{stem}.png")
                shutil.copy2(normal_path, paired_dir / f"{stem}{normal_path.suffix.lower()}")
                # inference.py validates train/good even in paired-normal mode.
                good_path = good_dir / f"{defect.name}_{stem}{normal_path.suffix.lower()}"
                if not good_path.exists():
                    shutil.copy2(normal_path, good_path)
                record["areas"][variant] = int(variant_mask.sum())
            manifest["samples"].append(record)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Prepared {len(manifest['samples'])} samples x {len(VARIANTS)} variants")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
