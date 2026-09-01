"""Create reproducible few-shot train/evaluation MVTec-style views.

The source dataset is never modified. Files are linked or copied into two
separate roots and a manifest records the exact split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--shots", type=int, default=None)
    split_group.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Per-class train ratio for a standard stratified split, e.g. 0.7",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="hardlink saves space and is recommended when roots share a filesystem",
    )
    parser.add_argument(
        "--mask_mode",
        choices=["original", "bbox"],
        default="original",
        help=(
            "original preserves pixel masks; bbox converts each GT region into "
            "a coarse rectangular spatial prior without modifying the source."
        ),
    )
    return parser.parse_args()


def image_files(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def matching_mask(mask_dir: Path, image_path: Path) -> Path:
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.jpg",
        mask_dir / f"{image_path.stem}.jpg",
        mask_dir / f"{image_path.stem}_mask.jpeg",
        mask_dir / f"{image_path.stem}.jpeg",
    ]
    match = next((path for path in candidates if path.is_file()), None)
    if match is None:
        raise FileNotFoundError(
            f"No ground-truth mask matches image {image_path.name} in {mask_dir}"
        )
    return match


def place_file(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(
                f"Hardlink failed for {source} -> {destination}. "
                "Use --mode symlink or --mode copy."
            ) from error


def place_mask(source: Path, destination: Path, mode: str, mask_mode: str):
    if mask_mode == "original":
        place_file(source, destination, mode)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    with Image.open(source) as image:
        binary = image.convert("L").point(lambda value: 255 if value > 0 else 0)
        bbox = binary.getbbox()
        if bbox is None:
            raise ValueError(f"Ground-truth mask contains no foreground: {source}")
        coarse = Image.new("L", binary.size, color=0)
        # PIL bbox uses an exclusive right/bottom edge.
        ImageDraw.Draw(coarse).rectangle(
            (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255
        )
        coarse.save(destination)


def link_normal_images(category_source: Path, destination_root: Path, category: str, mode: str):
    good_dir = category_source / "train" / "good"
    for source in image_files(good_dir):
        place_file(source, destination_root / category / "train" / "good" / source.name, mode)


def main():
    args = parse_args()
    if args.shots is not None and args.shots <= 0:
        raise ValueError("--shots must be greater than zero")
    if args.train_ratio is not None and not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between zero and one")

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root is not empty: {output_root}. "
            "Use a new directory to preserve previous splits."
        )

    train_root = output_root / "train"
    eval_root = output_root / "eval"
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split_type": "few_shot" if args.shots is not None else "train_ratio",
        "shots": args.shots,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "mode": args.mode,
        "mask_mode": args.mask_mode,
        "categories": {},
    }

    for category_index, category in enumerate(args.categories):
        category_source = source_root / category
        test_root = category_source / "test"
        gt_root = category_source / "ground_truth"
        if not test_root.is_dir() or not gt_root.is_dir():
            raise FileNotFoundError(f"Invalid MVTec layout for category: {category_source}")

        link_normal_images(category_source, train_root, category, args.mode)
        link_normal_images(category_source, eval_root, category, args.mode)
        defect_classes = sorted(
            path.name for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
        manifest["categories"][category] = {}

        for defect_index, defect_class in enumerate(defect_classes):
            images = image_files(test_root / defect_class)
            train_count = (
                args.shots
                if args.shots is not None
                else max(1, min(len(images) - 1, int(round(len(images) * args.train_ratio))))
            )
            if len(images) <= train_count:
                raise ValueError(
                    f"{category}/{defect_class} has {len(images)} images; "
                    f"cannot select {train_count} while retaining a held-out set"
                )
            rng = random.Random(args.seed + category_index * 100003 + defect_index * 1009)
            shuffled = images.copy()
            rng.shuffle(shuffled)
            train_images = sorted(shuffled[:train_count])
            eval_images = sorted(shuffled[train_count:])

            class_manifest = {"train": [], "eval": []}
            for split_name, selected, split_root in (
                ("train", train_images, train_root),
                ("eval", eval_images, eval_root),
            ):
                for image_path in selected:
                    mask_path = matching_mask(gt_root / defect_class, image_path)
                    place_file(
                        image_path,
                        split_root / category / "test" / defect_class / image_path.name,
                        args.mode,
                    )
                    place_mask(
                        mask_path,
                        split_root / category / "ground_truth" / defect_class / mask_path.name,
                        args.mode,
                        args.mask_mode,
                    )
                    class_manifest[split_name].append(
                        {"image": image_path.name, "mask": mask_path.name}
                    )
            manifest["categories"][category][defect_class] = class_manifest
            print(
                f"{category}/{defect_class}: "
                f"train={len(train_images)}, eval={len(eval_images)}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Train root: {train_root}")
    print(f"Eval root:  {eval_root}")
    print(f"Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
