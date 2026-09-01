"""Deterministically split the reviewed Black-instance YOLO set."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = sorted((args.source / "images").glob("*.jpg"))
    if not 0 < args.val_count < len(images):
        raise ValueError(f"val-count must be between 1 and {len(images)-1}")
    missing = [p.name for p in images if not (args.source / "labels" / f"{p.stem}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing labels: {missing}")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    shuffled = images.copy()
    random.Random(args.seed).shuffle(shuffled)
    val_names = {p.name for p in shuffled[: args.val_count]}
    rows = []
    for image in images:
        split = "val" if image.name in val_names else "train"
        image_out = args.output / "images" / split
        label_out = args.output / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        label = args.source / "labels" / f"{image.stem}.txt"
        shutil.copy2(image, image_out / image.name)
        shutil.copy2(label, label_out / label.name)
        boxes = sum(bool(line.strip()) for line in label.read_text(encoding="utf-8").splitlines())
        rows.append({"filename": image.name, "split": split, "boxes": boxes})

    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "split", "boxes"))
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        print(f"{split}: images={len(selected)} boxes={sum(int(row['boxes']) for row in selected)}")


if __name__ == "__main__":
    main()
