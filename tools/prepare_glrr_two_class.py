"""Build the clean two-class GLRR YOLO detection benchmark.

Kept classes from the original Plastic_Bomo annotation scheme:
  1 (Flash point)      -> 0
  2 (Big black spots)  -> 1

An image containing any other labelled class is excluded completely so that no
visible defect is silently converted into background. Images and labels are
hard-linked where possible, with a copy fallback across filesystems.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_MAP = {1: 0, 2: 1}
CLASS_NAMES = ["Flash point", "Big black spots"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--groups", nargs="+", default=["real_only", "real_baseline", "real_carf"]
    )
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def load_rows(label_path: Path) -> list[list[str]]:
    if not label_path.exists():
        return []
    return [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def process_split(source: Path, output: Path, group: str, split: str) -> dict[str, object]:
    image_dir = source / group / "images" / split
    label_dir = source / group / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    stats: dict[str, object] = {
        "kept_images": 0,
        "excluded_images": 0,
        "background_images": 0,
        "boxes": [0, 0],
    }
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label_path = label_dir / f"{image_path.stem}.txt"
        rows = load_rows(label_path)
        original_ids = [int(row[0]) for row in rows]

        # A non-target defect must not remain visible without a label.
        if any(class_id not in CLASS_MAP for class_id in original_ids):
            stats["excluded_images"] = int(stats["excluded_images"]) + 1
            continue

        remapped = []
        boxes = stats["boxes"]
        assert isinstance(boxes, list)
        for row, old_id in zip(rows, original_ids):
            new_id = CLASS_MAP[old_id]
            row[0] = str(new_id)
            remapped.append(" ".join(row))
            boxes[new_id] += 1

        target_image = output / group / "images" / split / image_path.name
        target_label = output / group / "labels" / split / label_path.name
        link_or_copy(image_path, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text(("\n".join(remapped) + "\n") if remapped else "", encoding="utf-8")
        stats["kept_images"] = int(stats["kept_images"]) + 1
        if not remapped:
            stats["background_images"] = int(stats["background_images"]) + 1
    return stats


def write_yaml(output: Path, group: str) -> None:
    content = (
        f"path: {output / group}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: Flash point\n"
        "  1: Big black spots\n"
    )
    (output / f"{group}.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    for group in args.groups:
        for split in ("train", "val"):
            stats = process_split(args.source, args.output, group, split)
            print(
                f"{group}/{split}: kept={stats['kept_images']} "
                f"excluded={stats['excluded_images']} backgrounds={stats['background_images']} "
                f"Flash_point={stats['boxes'][0]} Big_black_spots={stats['boxes'][1]}"
            )
        write_yaml(args.output, group)

    print(f"Created clean two-class dataset: {args.output}")


if __name__ == "__main__":
    main()
