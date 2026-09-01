"""Evaluate paired coarse-mask perturbation robustness."""

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import scipy.ndimage  # noqa: F401
except ImportError:
    scipy_module = types.ModuleType("scipy")
    ndimage_module = types.ModuleType("scipy.ndimage")
    ndimage_module.zoom = lambda *args, **kwargs: None
    scipy_module.ndimage = ndimage_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.ndimage"] = ndimage_module

import lpips


VARIANTS = ("erode20", "dilate20", "dilate40", "shift8")
DEFECTS = ("01_Flash_point", "02_Big_black_spots")
SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/mask_robustness")
    parser.add_argument("--category", default="Plastic_Bomo")
    parser.add_argument("--method", nargs=2, action="append", required=True)
    parser.add_argument("--context_scale", type=float, default=2.0)
    parser.add_argument("--min_crop_size", type=int, default=96)
    parser.add_argument("--success_threshold", type=float, default=0.01)
    return parser.parse_args()


def files(folder):
    return sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in SUFFIXES),
        key=lambda p: int(p.stem),
    )


def match(folder, stem):
    for suffix in SUFFIXES:
        path = Path(folder) / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {stem} under {folder}")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path, shape):
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape != shape:
        mask = cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return mask > 127


def crop_box(mask, context_scale, minimum):
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    side = min(max(mask.shape), max(minimum, int(np.ceil(max(width, height) * context_scale))))
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    left = max(0, min(mask.shape[1] - side, int(round(cx - side / 2))))
    top = max(0, min(mask.shape[0] - side, int(round(cy - side / 2))))
    return left, top, left + side, top + side


def perceptual_pair(first_path, second_path, union_mask, scale, minimum):
    box = crop_box(union_mask, scale, minimum)
    first = PREPROCESS(Image.open(first_path).convert("RGB").crop(box))
    second = PREPROCESS(Image.open(second_path).convert("RGB").crop(box))
    return first, second


def perceptual_distances(model, pairs, device, batch_size=32):
    values = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            first = torch.stack([pair[0] for pair in chunk]).to(device)
            second = torch.stack([pair[1] for pair in chunk]).to(device)
            values.extend(model(first, second).reshape(-1).cpu().tolist())
    return values


def change_metrics(image_path, normal_path, mask_path):
    image, normal = load_rgb(image_path), load_rgb(normal_path)
    difference = np.abs(image - normal).mean(axis=2)
    mask = load_mask(mask_path, difference.shape)
    energy = float(difference.mean())
    leakage = float(difference[~mask].sum() / max(difference.sum(), 1e-8))
    inside = float(difference[mask].mean()) if mask.any() else 0.0
    return energy, leakage, inside, mask


def main():
    args = parse_args()
    root = Path(args.results_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    methods = [(label, folder) for label, folder in args.method]
    summary = {label: {variant: [] for variant in VARIANTS} for label, _ in methods}

    for label, method in methods:
        print(f"METHOD={label}")
        for defect in DEFECTS:
            original = root / "original" / method / args.category / defect
            original_images = files(original / "image")
            original_data = {}
            for image_path in original_images:
                stem = image_path.stem
                original_data[stem] = change_metrics(
                    image_path, match(original / "normal", stem), match(original / "masks", stem)
                )
            for variant in VARIANTS:
                base = root / variant / method / args.category / defect
                raw_values = []
                perceptual_pairs = []
                for image_path in files(base / "image"):
                    stem = image_path.stem
                    energy0, _, inside0, mask0 = original_data[stem]
                    energy, leakage, inside, mask = change_metrics(
                        image_path, match(base / "normal", stem), match(base / "masks", stem)
                    )
                    cve = abs(energy - energy0) / max(energy0, 1e-8)
                    success = float(inside >= args.success_threshold)
                    perceptual_pairs.append(perceptual_pair(
                        original / "image" / image_path.name, image_path,
                        mask0 | mask, args.context_scale, args.min_crop_size,
                    ))
                    raw_values.append(
                        (cve, leakage, success, energy, inside, inside0)
                    )
                cares = perceptual_distances(model, perceptual_pairs, device)
                values = [
                    (care, *raw) for care, raw in zip(cares, raw_values)
                ]
                array = np.asarray(values)
                means = array.mean(axis=0)
                summary[label][variant].extend(values)
                print(
                    f"  {defect}/{variant}: CARE={means[0]:.4f} | "
                    f"CVE={means[1]:.4f} | ELR={means[2]:.4f} | "
                    f"Success={means[3]*100:.1f}% | ChangeE={means[4]:.4f} | "
                    f"InsideE={means[5]:.4f}"
                )

    print("\nMACRO (40 paired samples per perturbation)")
    for label, _ in methods:
        all_values = []
        for variant in VARIANTS:
            array = np.asarray(summary[label][variant])
            means = array.mean(axis=0)
            all_values.extend(summary[label][variant])
            print(
                f"{label}/{variant}: CARE={means[0]:.4f} | CVE={means[1]:.4f} | "
                f"ELR={means[2]:.4f} | Success={means[3]*100:.1f}%"
            )
        means = np.asarray(all_values).mean(axis=0)
        print(
            f"{label}/AVERAGE: CARE={means[0]:.4f} | CVE={means[1]:.4f} | "
            f"ELR={means[2]:.4f} | Success={means[3]*100:.1f}% | "
            f"ChangeE={means[4]:.4f} | InsideE={means[5]:.4f}"
        )


if __name__ == "__main__":
    main()
