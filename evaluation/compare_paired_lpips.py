"""Class-wise paired LPIPS comparison for aligned generated/real datasets."""

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
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def inference_order(path):
    match = re.search(r"\d+", path.name)
    return int(match.group()) if match else float("inf")


def image_files(folder, key=None):
    paths = [
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths, key=key)


def load_batch(paths):
    return torch.stack(
        [PREPROCESS(Image.open(path).convert("RGB")) for path in paths]
    )


def paired_lpips(model, real_paths, generated_paths, device, batch_size):
    if len(real_paths) != len(generated_paths):
        raise ValueError(
            f"Pair count mismatch: real={len(real_paths)}, generated={len(generated_paths)}"
        )
    values = []
    with torch.no_grad():
        for start in range(0, len(real_paths), batch_size):
            real = load_batch(real_paths[start : start + batch_size]).to(device)
            generated = load_batch(
                generated_paths[start : start + batch_size]
            ).to(device)
            values.append(model(real, generated).reshape(-1).cpu())
    return float(torch.cat(values).mean())


def main():
    args = parse_args()
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
            real_paths = image_files(test_root / defect, key=inference_order)
            print(f"{category}/{defect}: paired real={len(real_paths)}")
            for label, generated_root in generated_roots:
                generated_paths = image_files(
                    generated_root / category / defect / "image",
                    key=lambda path: int(path.stem),
                )
                score = paired_lpips(
                    model, real_paths, generated_paths, device, args.batch_size
                )
                overall[label].append(score)
                print(f"  {label}: LPIPSx100={score * 100:.1f}")

    print("Overall macro average:")
    for label, scores in overall.items():
        print(f"  {label}: LPIPSx100={np.mean(scores) * 100:.1f}")


if __name__ == "__main__":
    main()
