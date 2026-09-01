"""Compare generated and real defect-change maps inside the supplied masks."""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True)
    parser.add_argument("--generated", nargs=2, action="append", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--change_threshold", type=float, default=20.0)
    return parser.parse_args()


def inference_order(path):
    match = re.search(r"\d+", path.name)
    return int(match.group()) if match else float("inf")


def files(folder, key):
    return sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=key,
    )


def matching_image(folder, stem):
    for suffix in IMAGE_SUFFIXES:
        path = folder / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image matching {stem} under {folder}")


def main():
    args = parse_args()
    real_root = Path(args.real_path)
    generated_roots = [(label, Path(path)) for label, path in args.generated]
    overall = {label: [] for label, _ in generated_roots}
    for category in args.categories:
        test_root = real_root / category / "test"
        defects = sorted(p.name for p in test_root.iterdir() if p.is_dir())
        for defect in defects:
            masks = files(
                real_root / category / "ground_truth" / defect, inference_order
            )
            real = [matching_image(test_root / defect, mask.stem) for mask in masks]
            for label, generated_root in generated_roots:
                available_images = files(
                    generated_root / category / defect / "image",
                    lambda path: path.name,
                )
                available_normals = files(
                    generated_root / category / defect / "normal",
                    lambda path: path.name,
                )
                images_by_stem = {path.stem: path for path in available_images}
                normals_by_stem = {path.stem: path for path in available_normals}
                if all(mask.stem in images_by_stem for mask in masks):
                    images = [images_by_stem[mask.stem] for mask in masks]
                    normals = [normals_by_stem[mask.stem] for mask in masks]
                else:
                    images = sorted(available_images, key=lambda path: int(path.stem))
                    normals = sorted(available_normals, key=lambda path: int(path.stem))
                if not (len(real) == len(images) == len(normals) == len(masks)):
                    raise ValueError(f"Pair count mismatch for {label}/{category}/{defect}")
                scores = []
                for real_path, image_path, normal_path, mask_path in zip(
                    real, images, normals, masks
                ):
                    target = np.asarray(Image.open(real_path).convert("L"), dtype=np.int16)
                    generated = np.asarray(Image.open(image_path).convert("L"), dtype=np.int16)
                    normal = np.asarray(Image.open(normal_path).convert("L"), dtype=np.int16)
                    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
                    real_change = (np.abs(target - normal) > args.change_threshold) & mask
                    generated_change = (
                        np.abs(generated - normal) > args.change_threshold
                    ) & mask
                    union = np.count_nonzero(real_change | generated_change)
                    if union:
                        scores.append(
                            np.count_nonzero(real_change & generated_change) / union
                        )
                score = float(np.mean(scores))
                overall[label].append(score)
                print(f"{category}/{defect}/{label}: Change-IoU={score:.4f}")
    print("Overall macro average:")
    for label, scores in overall.items():
        print(f"  {label}: Change-IoU={np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
