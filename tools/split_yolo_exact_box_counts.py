"""Re-split a YOLO dataset with exact per-class validation box counts."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val_counts", type=int, nargs=2, default=[20, 20])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def label_counts(path: Path) -> tuple[int, int]:
    counts = [0, 0]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        if class_id not in (0, 1):
            raise ValueError(f"Expected only class 0/1, found {class_id}: {path}")
        counts[class_id] += 1
    return counts[0], counts[1]


def collect_samples(source: Path):
    samples = {}
    for split in ("train", "val"):
        for image in sorted((source / "images" / split).iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image.name in samples:
                raise ValueError(f"Duplicate image name across splits: {image.name}")
            label = source / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            samples[image.name] = (image, label, label_counts(label))
    return list(samples.values())


def exact_subset(samples, targets, seed):
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    # Dynamic programming over the two small target box counts.
    states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for index in order:
        flash, black = samples[index][2]
        updates = {}
        for (current_flash, current_black), chosen in states.items():
            next_state = (current_flash + flash, current_black + black)
            if next_state[0] > targets[0] or next_state[1] > targets[1]:
                continue
            if next_state not in states and next_state not in updates:
                updates[next_state] = chosen + (index,)
        states.update(updates)
        if targets in states:
            return set(states[targets])
    reachable = min(
        states,
        key=lambda state: abs(targets[0] - state[0]) + abs(targets[1] - state[1]),
    )
    raise ValueError(
        f"Cannot form exact validation counts {targets}; closest reachable={reachable}"
    )


def summarize(output: Path, split: str):
    counts = [0, 0]
    labels = list((output / "labels" / split).glob("*.txt"))
    for label in labels:
        current = label_counts(label)
        counts[0] += current[0]
        counts[1] += current[1]
    images = sum(
        1 for path in (output / "images" / split).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    print(
        f"{split}: images={images} labels={len(labels)} "
        f"Flash_point={counts[0]} Big_black_spots={counts[1]}"
    )


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")
    samples = collect_samples(args.source)
    targets = tuple(args.val_counts)
    validation = exact_subset(samples, targets, args.seed)

    for index, (image, label, _) in enumerate(samples):
        split = "val" if index in validation else "train"
        link_or_copy(image, args.output / "images" / split / image.name)
        link_or_copy(label, args.output / "labels" / split / label.name)

    (args.output / "dataset.yaml").write_text(
        f"path: {args.output}\n"
        "train: images/train\nval: images/val\n\n"
        "names:\n  0: Flash point\n  1: Big black spots\n",
        encoding="utf-8",
    )
    summarize(args.output, "train")
    summarize(args.output, "val")
    print(f"Created exact-count real split: {args.output}")


if __name__ == "__main__":
    main()
