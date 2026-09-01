"""GLRR stage 1: paste a generated local defect residual into full-resolution images.

The generated patch is never pasted directly. We estimate a soft defect support
inside its coarse mask, feather the support, and add the RGB residual
(generated - paired normal) to a 4096x1024 defect-free background at 1:1 scale.
All methods share a deterministic placement manifest for a fair comparison.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from .hgrr_recomposition import RealBoxPrior, recompose_geometry
    from .shdr_recomposition import detector_aware_target, recompose_scale_aware
except ImportError:  # Direct execution: python tools/build_glrr_stage1.py
    from hgrr_recomposition import RealBoxPrior, recompose_geometry
    from shdr_recomposition import detector_aware_target, recompose_scale_aware


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_IDS = {"01_Flash_point": 1, "02_Big_black_spots": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--generated", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RESULT_ROOT"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--category", default="Plastic_Bomo")
    parser.add_argument("--flash_count", type=int, default=38)
    parser.add_argument("--black_count", type=int, default=18)
    parser.add_argument(
        "--allow_source_reuse", action="store_true",
        help=(
            "Deterministically reuse a valid local defect on different backgrounds "
            "when the requested count exceeds the number of common valid sources."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_background_mean", type=float, default=30.0)
    parser.add_argument("--min_nonblack_ratio", type=float, default=0.80)
    parser.add_argument("--threshold_mad_scale", type=float, default=3.0)
    parser.add_argument("--min_support_pixels", type=int, default=6)
    parser.add_argument("--max_support_ratio", type=float, default=0.80)
    parser.add_argument("--feather_radius", type=float, default=2.0)
    parser.add_argument("--hgrr", action="store_true")
    parser.add_argument(
        "--shdr", action="store_true",
        help="Restore real training-set defect scale with decoupled core/context.",
    )
    parser.add_argument("--real_label_dir")
    parser.add_argument(
        "--real_class_ids", type=int, nargs=2, default=(1, 2),
        metavar=("FLASH_ID", "BLACK_ID"),
    )
    parser.add_argument("--real_image_width", type=int, default=4096)
    parser.add_argument("--real_image_height", type=int, default=1024)
    parser.add_argument("--hgrr_core_threshold", type=float, default=0.50)
    parser.add_argument("--hgrr_context_strength", type=float, default=0.50)
    parser.add_argument("--shdr_core_threshold", type=float, default=0.50)
    parser.add_argument("--shdr_context_scale", type=float, default=1.20)
    parser.add_argument("--shdr_context_strength", type=float, default=0.40)
    parser.add_argument("--shdr_detector_imgsz", type=int, default=1536)
    parser.add_argument("--shdr_min_visible_short_side", type=float, default=14.0)
    parser.add_argument("--shdr_scale_min", type=float, default=1.20)
    parser.add_argument("--shdr_scale_max", type=float, default=1.70)
    parser.add_argument("--shdr_max_scale", type=float, default=2.0)
    return parser.parse_args()


def image_files(folder):
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def matching_file(folder, stem):
    for suffix in SUFFIXES:
        candidate = Path(folder) / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No matching file for {stem} under {folder}")


def resolve_result_root(root, category):
    """Resolve a possibly over-broad result path to the parent of category.

    Result archives have historically used both ``root/Plastic_Bomo`` and
    ``root/<experiment>/Plastic_Bomo``. Accept either layout while rejecting an
    ambiguous archive containing multiple matching experiments.
    """
    root = Path(root).resolve()
    defects = tuple(CLASS_IDS)

    def is_match(category_dir):
        return category_dir.is_dir() and all(
            (category_dir / defect / "image").is_dir()
            and (category_dir / defect / "normal").is_dir()
            and (category_dir / defect / "masks").is_dir()
            for defect in defects
        )

    direct = root / category
    if is_match(direct):
        return root
    if root.name == category and is_match(root):
        return root.parent
    matches = sorted(
        candidate for candidate in root.rglob(category) if is_match(candidate)
    )
    if not matches:
        raise FileNotFoundError(
            f"Cannot locate {category}/<defect>/(image,normal,masks) under {root}"
        )
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple compatible result roots found under {root}; pass one of:\n  "
            f"{formatted}"
        )
    resolved = matches[0].parent
    print(f"Resolved result root: {root} -> {resolved}")
    return resolved


def is_valid_background(path, minimum_mean, minimum_ratio):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return bool(
        image is not None
        and float(image.mean()) >= minimum_mean
        and float((image > 20).mean()) >= minimum_ratio
    )


def defect_residual(image_path, normal_path, mask_path, mad_scale,
                    min_pixels, max_ratio, feather_radius):
    generated = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    normal = np.asarray(Image.open(normal_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if generated.shape != normal.shape:
        raise ValueError(f"Generated/normal shape mismatch: {image_path}")
    if mask.shape != generated.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), generated.shape[1::-1],
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if not mask.any():
        raise ValueError(f"Empty mask: {mask_path}")

    residual = generated - normal
    magnitude = np.mean(np.abs(residual), axis=2)
    outside = magnitude[~mask]
    median = float(np.median(outside)) if outside.size else 0.0
    mad = float(np.median(np.abs(outside - median))) if outside.size else 0.0
    threshold = median + mad_scale * max(1.4826 * mad, 1.0)
    support = (magnitude > threshold) & mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_pixels:
            components.append((area, index))
    if not components:
        return None
    components.sort(reverse=True)
    # Keep all meaningful components near the generated defect, including a
    # possible halo, while rejecting isolated JPEG/background noise.
    keep = np.zeros_like(support)
    largest = components[0][0]
    for area, index in components:
        if area >= max(min_pixels, int(round(largest * 0.08))):
            keep |= labels == index
    if keep.sum() / max(mask.sum(), 1) > max_ratio:
        return None

    ys, xs = np.where(keep)
    margin = max(4, int(np.ceil(feather_radius * 3)))
    x0, x1 = max(0, xs.min() - margin), min(keep.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(keep.shape[0], ys.max() + margin + 1)
    local_support = keep[y0:y1, x0:x1].astype(np.float32)
    alpha = cv2.GaussianBlur(local_support, (0, 0), feather_radius)
    if alpha.max() > 0:
        alpha /= alpha.max()
    return residual[y0:y1, x0:x1], alpha, (x0, y0, x1, y1)


def valid_locations(background, patch_shape, rng, attempts=200):
    height, width = background.shape[:2]
    ph, pw = patch_shape
    if ph >= height or pw >= width:
        return None
    gray = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    for _ in range(attempts):
        x = rng.randint(0, width - pw)
        y = rng.randint(0, height - ph)
        region = gray[y:y + ph, x:x + pw]
        if float(region.mean()) >= 30.0 and float((region > 20).mean()) >= 0.90:
            return x, y
    return None


def yolo_box(x, y, support, full_width, full_height, threshold=0.10):
    ys, xs = np.where(support >= float(threshold))
    left, right = x + int(xs.min()), x + int(xs.max()) + 1
    top, bottom = y + int(ys.min()), y + int(ys.max()) + 1
    return (
        (left + right) / 2 / full_width,
        (top + bottom) / 2 / full_height,
        (right - left) / full_width,
        (bottom - top) / full_height,
    )


def common_valid_sources(roots, category, defect, count, args):
    by_method = {}
    for label, root in roots:
        base = root / category / defect
        by_method[label] = {path.stem: path for path in image_files(base / "image")}
    common_stems = sorted(
        set.intersection(*(set(paths) for paths in by_method.values())),
        key=lambda stem: int(stem) if stem.isdigit() else stem,
    )
    selected = {label: [] for label, _ in roots}
    rejected = []
    for stem in common_stems:
        valid = True
        for label, root in roots:
            image_path = by_method[label][stem]
            base = root / category / defect
            result = defect_residual(
                image_path, matching_file(base / "normal", stem),
                matching_file(base / "masks", stem),
                args.threshold_mad_scale, args.min_support_pixels,
                args.max_support_ratio, args.feather_radius,
            )
            if result is None:
                valid = False
                break
        if not valid:
            rejected.append(stem)
            continue
        for label, _ in roots:
            selected[label].append(by_method[label][stem])
        if len(next(iter(selected.values()))) == count:
            break
    available = len(next(iter(selected.values())))
    if available < count and not args.allow_source_reuse:
        raise ValueError(
            f"{defect}: need {count} common valid sources, found "
            f"{available}; pass --allow_source_reuse to place valid residuals "
            "on multiple independent backgrounds"
        )
    if available == 0:
        raise ValueError(f"{defect}: no common valid source is available")
    if available < count:
        for label, _ in roots:
            unique = list(selected[label])
            selected[label] = [unique[index % available] for index in range(count)]
    print(
        f"{defect}: selected={count}, unique_sources={available}, "
        f"reused={max(0, count - available)}, rejected={len(rejected)}"
    )
    return selected


def main():
    args = parse_args()
    if args.hgrr and args.shdr:
        raise ValueError("--hgrr and --shdr are mutually exclusive")
    if args.shdr_scale_min > args.shdr_scale_max:
        raise ValueError("--shdr_scale_min must not exceed --shdr_scale_max")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    backgrounds = [
        p for p in image_files(args.background_dir)
        if is_valid_background(p, args.min_background_mean, args.min_nonblack_ratio)
    ]
    required = args.flash_count + args.black_count
    if len(backgrounds) < required:
        raise ValueError(f"Need {required} valid backgrounds, found {len(backgrounds)}")
    rng = random.Random(args.seed)
    rng.shuffle(backgrounds)
    roots = [
        (label, resolve_result_root(path, args.category))
        for label, path in args.generated
    ]
    geometry_prior = None
    geometry_targets = {}
    if args.hgrr or args.shdr:
        if not args.real_label_dir:
            raise ValueError("--hgrr/--shdr requires --real_label_dir")
        geometry_prior = RealBoxPrior.from_yolo(
            args.real_label_dir,
            args.real_class_ids,
            tuple(CLASS_IDS),
            args.real_image_width,
            args.real_image_height,
        )
        for defect, count in (("01_Flash_point", args.flash_count),
                              ("02_Big_black_spots", args.black_count)):
            geometry_targets[defect] = geometry_prior.targets(
                defect, count, args.seed + CLASS_IDS[defect] * 100003
            )
            mode = "SHDR" if args.shdr else "HGRR"
            print(f"{mode} real prior {defect}: {geometry_prior.summary(defect)}")

    source_lists = {}
    for defect, count in (("01_Flash_point", args.flash_count),
                          ("02_Big_black_spots", args.black_count)):
        selected = common_valid_sources(roots, args.category, defect, count, args)
        for label, _ in roots:
            source_lists[(label, defect)] = selected[label]

    manifest = {
        "version": 1, "seed": args.seed, "category": args.category,
        "background_dir": str(Path(args.background_dir).resolve()),
        "methods": [label for label, _ in roots], "samples": [],
    }
    background_index = 0
    for defect, count in (("01_Flash_point", args.flash_count),
                          ("02_Big_black_spots", args.black_count)):
        for index in range(count):
            background_path = backgrounds[background_index]
            background_index += 1
            background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.float32)
            sample_name = f"glrr_{defect}_{index:04d}"
            record = {
                "sample": sample_name, "defect": defect,
                "class_id": CLASS_IDS[defect], "background": background_path.name,
                "methods": {},
            }
            # Placement is derived from the first method, then reused exactly.
            shared_position = None
            for method_index, (label, root) in enumerate(roots):
                image_path = source_lists[(label, defect)][index]
                base = root / args.category / defect
                residual_data = defect_residual(
                    image_path, matching_file(base / "normal", image_path.stem),
                    matching_file(base / "masks", image_path.stem),
                    args.threshold_mad_scale, args.min_support_pixels,
                    args.max_support_ratio, args.feather_radius,
                )
                if residual_data is None:
                    raise RuntimeError(
                        f"Unable to extract reliable residual: {label}/{defect}/{image_path.name}"
                    )
                residual, alpha, source_box = residual_data
                geometry_metadata = None
                label_support = alpha
                label_threshold = 0.10
                if args.hgrr:
                    residual, alpha, label_support, hgrr_metadata = recompose_geometry(
                        residual,
                        alpha,
                        geometry_targets[defect][index],
                        core_threshold=args.hgrr_core_threshold,
                        context_strength=args.hgrr_context_strength,
                    )
                    geometry_metadata = {"mode": "hgrr", **hgrr_metadata}
                    label_threshold = args.hgrr_core_threshold
                elif args.shdr:
                    scale_rng = random.Random(
                        args.seed
                        + CLASS_IDS[defect] * 1000003
                        + index * 9176
                    )
                    sampled_multiplier = scale_rng.uniform(
                        args.shdr_scale_min, args.shdr_scale_max
                    )
                    calibrated_wh, detector_metadata = detector_aware_target(
                        geometry_targets[defect][index],
                        detector_imgsz=args.shdr_detector_imgsz,
                        image_wh=(args.real_image_width, args.real_image_height),
                        min_visible_short_side=args.shdr_min_visible_short_side,
                        scale_multiplier=sampled_multiplier,
                        max_scale_multiplier=args.shdr_max_scale,
                    )
                    residual, alpha, label_support, shdr_metadata = recompose_scale_aware(
                        residual,
                        alpha,
                        calibrated_wh,
                        core_threshold=args.shdr_core_threshold,
                        context_scale=args.shdr_context_scale,
                        context_strength=args.shdr_context_strength,
                    )
                    geometry_metadata = {
                        "mode": "shdr_detector_aware",
                        **detector_metadata,
                        **shdr_metadata,
                    }
                    label_threshold = args.shdr_core_threshold
                if shared_position is None:
                    shared_position = valid_locations(
                        background, residual.shape[:2], random.Random(args.seed + background_index * 1009)
                    )
                    if shared_position is None:
                        raise RuntimeError(f"No valid placement on {background_path}")
                x, y = shared_position
                # Different methods can yield slightly different residual crop
                # sizes. Align their centres to the same full-image coordinate.
                if method_index:
                    cx = shared_position[0] + first_shape[1] // 2
                    cy = shared_position[1] + first_shape[0] // 2
                    x = max(0, min(background.shape[1] - residual.shape[1], cx - residual.shape[1] // 2))
                    y = max(0, min(background.shape[0] - residual.shape[0], cy - residual.shape[0] // 2))
                else:
                    first_shape = residual.shape[:2]

                composed = background.copy()
                ph, pw = residual.shape[:2]
                target = composed[y:y + ph, x:x + pw]
                target += residual * alpha[..., None]
                composed[y:y + ph, x:x + pw] = np.clip(target, 0, 255)
                image_dir = output / label / "images"
                label_dir = output / label / "labels"
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(composed.astype(np.uint8)).save(
                    image_dir / f"{sample_name}.jpg", quality=95, subsampling=0
                )
                box = yolo_box(
                    x, y, label_support, background.shape[1], background.shape[0],
                    threshold=label_threshold,
                )
                (label_dir / f"{sample_name}.txt").write_text(
                    f"{CLASS_IDS[defect]} " + " ".join(f"{value:.8f}" for value in box) + "\n",
                    encoding="utf-8",
                )
                record["methods"][label] = {
                    "source": image_path.name, "source_box": list(map(int, source_box)),
                    "paste_xy": [int(x), int(y)], "patch_wh": [int(pw), int(ph)],
                    "yolo_box": list(map(float, box)),
                    "geometry": geometry_metadata,
                }
            manifest["samples"].append(record)

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"GLRR stage-1: {len(manifest['samples'])} samples per method")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
