"""LPIPS spatial distance averaged outside each generated coarse mask."""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", nargs=2, action="append", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
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
    raise FileNotFoundError(f"No file for stem={stem} under {folder}")


def load_batch(pairs):
    generated, normal, masks = [], [], []
    for image_path, normal_path, mask_path in pairs:
        generated.append(PREPROCESS(Image.open(image_path).convert("RGB")))
        normal.append(PREPROCESS(Image.open(normal_path).convert("RGB")))
        mask = Image.open(mask_path).convert("L").resize((256, 256), Image.Resampling.NEAREST)
        masks.append(torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None])
    return torch.stack(generated), torch.stack(normal), torch.stack(masks)


def score(model, pairs, device, batch_size):
    total_distance = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            generated, normal, mask = load_batch(pairs[start:start + batch_size])
            spatial = model(generated.to(device), normal.to(device))
            outside = 1.0 - F.interpolate(mask.to(device), spatial.shape[-2:], mode="nearest")
            total_distance += float((spatial * outside).sum().cpu())
            total_weight += float(outside.sum().cpu())
    return total_distance / max(total_weight, 1.0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
    roots = [(label, Path(path)) for label, path in args.generated]
    overall = {label: [] for label, _ in roots}

    for category in args.categories:
        defect_sets = []
        for _, root in roots:
            defect_sets.append({p.name for p in (root / category).iterdir() if p.is_dir()})
        defects = sorted(set.intersection(*defect_sets))
        for defect in defects:
            print(f"{category}/{defect}:")
            for label, root in roots:
                base = root / category / defect
                pairs = []
                for image_path in files(base / "image"):
                    pairs.append((
                        image_path,
                        matching_file(base / "normal", image_path.stem),
                        matching_file(base / "masks", image_path.stem),
                    ))
                value = score(model, pairs, device, args.batch_size)
                overall[label].append(value)
                print(f"  {label}: BG-LPIPSx100={value * 100:.3f} (n={len(pairs)})")
    print("Overall macro average:")
    for label, values in overall.items():
        print(f"  {label}: BG-LPIPSx100={np.mean(values) * 100:.3f}")


if __name__ == "__main__":
    main()
