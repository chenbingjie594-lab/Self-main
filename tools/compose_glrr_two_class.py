"""Compose fair two-class YOLO groups from real data and paired GLRR addons."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", type=Path, required=True)
    parser.add_argument("--synthetic_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["Baseline", "CARF"])
    return parser.parse_args()


def link_or_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_real(real_root: Path, group_root: Path):
    for split in ("train", "val"):
        for image in sorted((real_root / "images" / split).iterdir()):
            if image.is_file() and image.suffix.lower() in SUFFIXES:
                link_or_copy(image, group_root / "images" / split / image.name)
                label = real_root / "labels" / split / f"{image.stem}.txt"
                link_or_copy(label, group_root / "labels" / split / label.name)


def add_synthetic(synthetic_root: Path, method: str, group_root: Path):
    image_root = synthetic_root / method / "images"
    label_root = synthetic_root / method / "labels"
    for image in sorted(p for p in image_root.iterdir() if p.suffix.lower() in SUFFIXES):
        target_image = group_root / "images" / "train" / image.name
        target_label = group_root / "labels" / "train" / f"{image.stem}.txt"
        if target_image.exists() or target_label.exists():
            raise FileExistsError(f"Synthetic/real filename collision: {image.name}")
        rows = []
        for line in (label_root / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            old_id = int(fields[0])
            if old_id not in (1, 2):
                raise ValueError(f"Unexpected synthetic class {old_id}: {image.name}")
            fields[0] = str(old_id - 1)
            rows.append(" ".join(fields))
        link_or_copy(image, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_yaml(output: Path, group: str):
    (output / f"{group}.yaml").write_text(
        f"path: {output / group}\n"
        "train: images/train\nval: images/val\n\n"
        "names:\n  0: Flash point\n  1: Big black spots\n",
        encoding="utf-8",
    )


def summarize(group_root: Path):
    for split in ("train", "val"):
        images = list((group_root / "images" / split).iterdir())
        counts = [0, 0]
        for label in (group_root / "labels" / split).glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
        print(
            f"{group_root.name}/{split}: images={len(images)} "
            f"Flash_point={counts[0]} Big_black_spots={counts[1]}"
        )


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    real_group = args.output / "real_only"
    copy_real(args.real_root, real_group)
    write_yaml(args.output, "real_only")
    summarize(real_group)

    for method in args.methods:
        group = f"real_{method.lower()}"
        group_root = args.output / group
        copy_real(args.real_root, group_root)
        add_synthetic(args.synthetic_root, method, group_root)
        write_yaml(args.output, group)
        summarize(group_root)

    print(f"Created balanced two-class benchmark: {args.output}")


if __name__ == "__main__":
    main()
