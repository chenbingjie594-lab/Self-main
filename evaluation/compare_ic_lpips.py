"""Reproducible IC-LPIPS comparison with a held-out real split."""

import argparse
import itertools
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# The upstream lpips package imports its optional visualization trainer at
# package import time. Evaluation itself does not use scipy.ndimage.zoom, so a
# minimal stub keeps inference usable in lean environments without SciPy.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True)
    parser.add_argument("--generated", nargs=2, action="append", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--cluster_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def image_files(folder):
    return sorted(
        path for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def held_out_files(files, train_ratio, seed):
    if train_ratio is None:
        return files
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(files) - 1, int(round(len(files) * train_ratio))))
    return sorted(shuffled[train_count:])


PREPROCESS = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def load_images(paths, device):
    return torch.stack(
        [PREPROCESS(Image.open(path).convert("RGB")) for path in paths]
    ).to(device)


def distances(model, first, second, batch_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(first), batch_size):
            values.append(
                model(
                    first[start : start + batch_size],
                    second[start : start + batch_size],
                ).reshape(-1).cpu()
            )
    return torch.cat(values)


def ic_lpips(model, real_paths, generated_paths, device, batch_size, cluster_size, seed):
    real = load_images(real_paths, device)
    generated = load_images(generated_paths, device)
    clusters = [[] for _ in real_paths]
    with torch.no_grad():
        for generated_index in range(len(generated)):
            sample = generated[generated_index : generated_index + 1]
            candidate_distances = []
            for start in range(0, len(real), batch_size):
                real_batch = real[start : start + batch_size]
                repeated = sample.expand(len(real_batch), -1, -1, -1)
                candidate_distances.append(
                    model(repeated, real_batch).reshape(-1).cpu()
                )
            closest = int(torch.cat(candidate_distances).argmin())
            clusters[closest].append(generated_index)

    rng = random.Random(seed)
    cluster_scores = []
    for indices in clusters:
        indices = list(indices)
        rng.shuffle(indices)
        indices = indices[:cluster_size]
        pairs = list(itertools.combinations(indices, 2))
        if not pairs:
            continue
        first = generated[[left for left, _ in pairs]]
        second = generated[[right for _, right in pairs]]
        cluster_scores.append(float(distances(model, first, second, batch_size).mean()))
    if not cluster_scores:
        raise ValueError("No cluster contains at least two generated images")
    return float(np.mean(cluster_scores)), len(cluster_scores)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    real_root = Path(args.real_path)
    generated_roots = [(label, Path(path)) for label, path in args.generated]
    overall = {label: [] for label, _ in generated_roots}
    for category_index, category in enumerate(args.categories):
        test_root = real_root / category / "test"
        defects = sorted(
            path.name for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
        for defect_index, defect in enumerate(defects):
            seed = args.split_seed + category_index * 100003 + defect_index * 1009
            real_paths = held_out_files(
                image_files(test_root / defect), args.train_ratio, seed
            )
            print(f"{category}/{defect}: held-out real={len(real_paths)}")
            for label, generated_root in generated_roots:
                generated_paths = image_files(generated_root / category / defect / "image")
                score, valid_clusters = ic_lpips(
                    model,
                    real_paths,
                    generated_paths,
                    device,
                    args.batch_size,
                    args.cluster_size,
                    seed,
                )
                overall[label].append(score)
                print(
                    f"  {label}: IC-LPIPSx100={score * 100:.1f} "
                    f"(valid clusters={valid_clusters})"
                )
    print("Overall macro average:")
    for label, scores in overall.items():
        print(f"  {label}: IC-LPIPSx100={np.mean(scores) * 100:.1f}")


if __name__ == "__main__":
    main()
