"""Compare paired LPIPS inside a mask-centred local crop."""

import argparse
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import scipy.ndimage  # noqa: F401
except ImportError:
    scipy_module = types.ModuleType("scipy")
    ndimage_module = types.ModuleType("scipy.ndimage")

    def _unused_zoom(*args, **kwargs):
        raise RuntimeError("scipy is required only for LPIPS trainer visualization")

    ndimage_module.zoom = _unused_zoom
    scipy_module.ndimage = ndimage_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.ndimage"] = ndimage_module

import lpips


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True)
    parser.add_argument("--generated", nargs=2, action="append", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--context_scale", type=float, default=4.0)
    parser.add_argument("--min_crop_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def inference_order(path):
    match = re.search(r"\d+", path.name)
    return int(match.group()) if match else float("inf")


def image_files(folder, key):
    return sorted(
        (
            path
            for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=key,
    )


def matching_image(image_dir, stem):
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image matching {stem} under {image_dir}")


def crop_box(mask, context_scale, min_crop_size):
    array = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(array)
    if len(xs) == 0:
        raise ValueError("Encountered an empty mask")
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    side = max(min_crop_size, int(np.ceil(max(width, height) * context_scale)))
    side = min(side, mask.width, mask.height)
    cx = (int(xs.min()) + int(xs.max())) / 2.0
    cy = (int(ys.min()) + int(ys.max())) / 2.0
    left = max(0, min(mask.width - side, int(round(cx - side / 2))))
    top = max(0, min(mask.height - side, int(round(cy - side / 2))))
    return left, top, left + side, top + side


def load_crops(pairs, context_scale, min_crop_size):
    real_tensors = []
    generated_tensors = []
    for real_path, generated_path, mask_path in pairs:
        mask = Image.open(mask_path)
        box = crop_box(mask, context_scale, min_crop_size)
        real_tensors.append(
            PREPROCESS(Image.open(real_path).convert("RGB").crop(box))
        )
        generated_tensors.append(
            PREPROCESS(Image.open(generated_path).convert("RGB").crop(box))
        )
    return torch.stack(real_tensors), torch.stack(generated_tensors)


def score_pairs(model, pairs, device, batch_size, context_scale, min_crop_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            real, generated = load_crops(
                pairs[start : start + batch_size], context_scale, min_crop_size
            )
            values.append(
                model(real.to(device), generated.to(device)).reshape(-1).cpu()
            )
    return float(torch.cat(values).mean())


def main():
    args = parse_args()
    if args.context_scale <= 0 or args.min_crop_size <= 0:
        raise ValueError("context_scale and min_crop_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    real_root = Path(args.real_path)
    generated_roots = [(label, Path(path)) for label, path in args.generated]
    overall = {label: [] for label, _ in generated_roots}

    for category in args.categories:
        test_root = real_root / category / "test"
        defects = sorted(
            path.name
            for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
        for defect in defects:
            mask_paths = image_files(
                real_root / category / "ground_truth" / defect,
                inference_order,
            )
            real_paths = [
                matching_image(test_root / defect, mask_path.stem)
                for mask_path in mask_paths
            ]
            print(f"{category}/{defect}: paired samples={len(real_paths)}")
            for label, generated_root in generated_roots:
                available_generated = image_files(
                    generated_root / category / defect / "image",
                    lambda path: path.name,
                )
                generated_by_stem = {path.stem: path for path in available_generated}
                if all(mask_path.stem in generated_by_stem for mask_path in mask_paths):
                    generated_paths = [
                        generated_by_stem[mask_path.stem] for mask_path in mask_paths
                    ]
                else:
                    generated_paths = sorted(
                        available_generated, key=lambda path: int(path.stem)
                    )
                if len(generated_paths) != len(real_paths):
                    raise ValueError(
                        f"Pair count mismatch for {label}/{category}/{defect}: "
                        f"real={len(real_paths)}, generated={len(generated_paths)}"
                    )
                pairs = list(zip(real_paths, generated_paths, mask_paths))
                score = score_pairs(
                    model,
                    pairs,
                    device,
                    args.batch_size,
                    args.context_scale,
                    args.min_crop_size,
                )
                overall[label].append(score)
                print(f"  {label}: Mask-Crop-LPIPSx100={score * 100:.1f}")

    print("Overall macro average:")
    for label, scores in overall.items():
        print(f"  {label}: Mask-Crop-LPIPSx100={np.mean(scores) * 100:.1f}")


if __name__ == "__main__":
    main()
