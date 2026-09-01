"""Region-aware metrics for coarse-mask defect generation.

Reports:
  BPE-L1: mean absolute RGB change outside a dilated coarse mask.
  CCE: Wasserstein distance between real/generated signed local contrast.
  DAD: Wasserstein distance between real/generated salient defect area ratios.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def wasserstein_distance(first, second, samples=4096):
    """Deterministic one-dimensional W1 approximation on a dense quantile grid."""
    quantiles = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    return float(np.mean(np.abs(
        np.quantile(first, quantiles) - np.quantile(second, quantiles)
    )))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True)
    parser.add_argument("--generated", nargs=2, action="append", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--ring_radius", type=int, default=12)
    parser.add_argument("--bpe_margin", type=int, default=8)
    parser.add_argument("--saliency_mad_scale", type=float, default=3.0)
    return parser.parse_args()


def image_files(folder):
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def matching_file(folder, stem):
    for candidate_stem in (stem, f"{stem}_mask"):
        for suffix in SUFFIXES:
            candidate = Path(folder) / f"{candidate_stem}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No match for {stem} under {folder}")


def held_out_pairs(images, mask_dir, ratio, seed):
    pairs = [(p, matching_file(mask_dir, p.stem)) for p in images]
    if ratio is None:
        return pairs
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(pairs) - 1, int(round(len(pairs) * ratio))))
    return sorted(shuffled[count:], key=lambda pair: pair[0].name)


def generated_triplets(root, category, defect):
    base = root / category / defect
    return [
        (
            path,
            matching_file(base / "masks", path.stem),
            matching_file(base / "normal", path.stem),
        )
        for path in image_files(base / "image")
    ]


def binary_mask(path, size):
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def dilate(mask, radius):
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    expanded = cv2.dilate(
        mask.astype(np.uint8), np.ones((kernel, kernel), dtype=np.uint8)
    )
    return expanded.astype(bool)


def region_statistics(image_path, mask_path, ring_radius, mad_scale):
    image_pil = Image.open(image_path).convert("RGB")
    image = np.asarray(image_pil, dtype=np.float32) / 255.0
    mask = binary_mask(mask_path, image_pil.size)
    ring = dilate(mask, ring_radius) & ~mask
    if mask.sum() < 1 or ring.sum() < 3:
        raise ValueError(f"Invalid mask/ring for {image_path}")

    ring_rgb = image[ring]
    background_rgb = np.median(ring_rgb, axis=0)
    ring_distance = np.linalg.norm(ring_rgb - background_rgb, axis=1)
    median_distance = np.median(ring_distance)
    mad = np.median(np.abs(ring_distance - median_distance))
    threshold = median_distance + mad_scale * max(1.4826 * mad, 1.0 / 255.0)

    inside_rgb = image[mask]
    inside_distance = np.linalg.norm(inside_rgb - background_rgb, axis=1)
    support = inside_distance > threshold
    area_ratio = float(support.mean())

    # Keep contrast defined for very weak/empty predictions without changing
    # the area measurement: use the most salient 5% only as a contrast probe.
    if support.sum() < 3:
        count = min(len(inside_distance), max(3, int(np.ceil(len(inside_distance) * 0.05))))
        indices = np.argpartition(inside_distance, -count)[-count:]
        defect_rgb = inside_rgb[indices]
    else:
        defect_rgb = inside_rgb[support]

    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    defect_luma = defect_rgb @ weights
    ring_luma = ring_rgb @ weights
    contrast = float(
        (defect_luma.mean() - np.median(ring_luma))
        / (ring_luma.std() + 0.03)
    )
    return contrast, area_ratio


def bpe_l1(image_path, normal_path, mask_path, margin):
    image_pil = Image.open(image_path).convert("RGB")
    normal_pil = Image.open(normal_path).convert("RGB")
    if normal_pil.size != image_pil.size:
        normal_pil = normal_pil.resize(image_pil.size, Image.Resampling.BILINEAR)
    mask = dilate(binary_mask(mask_path, image_pil.size), margin)
    outside = ~mask
    image = np.asarray(image_pil, dtype=np.float32) / 255.0
    normal = np.asarray(normal_pil, dtype=np.float32) / 255.0
    return float(np.abs(image - normal)[outside].mean())


def main():
    args = parse_args()
    real_root = Path(args.real_path)
    roots = [(label, Path(path)) for label, path in args.generated]
    macro = {label: {"bpe": [], "cce": [], "dad": []} for label, _ in roots}

    for category_index, category in enumerate(args.categories):
        test_root = real_root / category / "test"
        defects = sorted(p.name for p in test_root.iterdir() if p.is_dir() and p.name != "good")
        for defect_index, defect in enumerate(defects):
            seed = args.split_seed + category_index * 100003 + defect_index * 1009
            real_pairs = held_out_pairs(
                image_files(test_root / defect),
                real_root / category / "ground_truth" / defect,
                args.train_ratio,
                seed,
            )
            real_stats = [
                region_statistics(image, mask, args.ring_radius, args.saliency_mad_scale)
                for image, mask in real_pairs
            ]
            real_contrast = np.asarray([value[0] for value in real_stats])
            real_area = np.asarray([value[1] for value in real_stats])
            print(f"{category}/{defect}: real={len(real_pairs)}")

            for label, root in roots:
                triplets = generated_triplets(root, category, defect)
                generated_stats = [
                    region_statistics(image, mask, args.ring_radius, args.saliency_mad_scale)
                    for image, mask, _ in triplets
                ]
                generated_contrast = np.asarray([value[0] for value in generated_stats])
                generated_area = np.asarray([value[1] for value in generated_stats])
                bpe = float(np.mean([
                    bpe_l1(image, normal, mask, args.bpe_margin)
                    for image, mask, normal in triplets
                ]))
                cce = float(wasserstein_distance(real_contrast, generated_contrast))
                dad = float(wasserstein_distance(real_area, generated_area))
                macro[label]["bpe"].append(bpe)
                macro[label]["cce"].append(cce)
                macro[label]["dad"].append(dad)
                print(
                    f"  {label}: BPE-L1x100={bpe * 100:.3f} | "
                    f"CCE={cce:.4f} | DADx100={dad * 100:.3f}"
                )

    print("Overall macro average:")
    for label, values in macro.items():
        print(
            f"  {label}: BPE-L1x100={np.mean(values['bpe']) * 100:.3f} | "
            f"CCE={np.mean(values['cce']):.4f} | "
            f"DADx100={np.mean(values['dad']) * 100:.3f}"
        )


if __name__ == "__main__":
    main()
