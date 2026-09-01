"""Create deterministic image-level OOF folds from the real training split."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def link_or_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def class_signature(label: Path):
    classes = set()
    for line in label.read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            if class_id in (0, 1):
                classes.add(class_id)
    return tuple(sorted(classes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("At least two OOF folds are required")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")

    image_root = args.real_root / "images" / "train"
    label_root = args.real_root / "labels" / "train"
    images = sorted(path for path in image_root.iterdir() if path.suffix.lower() in SUFFIXES)
    groups = defaultdict(list)
    for image in images:
        label = label_root / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing real label: {label}")
        groups[class_signature(label)].append(image)

    validation = [[] for _ in range(args.folds)]
    for group_index, (signature, members) in enumerate(sorted(groups.items(), key=lambda item: item[0])):
        members = list(members)
        random.Random(args.seed + group_index * 1009).shuffle(members)
        for index, image in enumerate(members):
            validation[index % args.folds].append(image)

    manifest = {"version": 1, "seed": args.seed, "folds": []}
    all_names = {image.name for image in images}
    for fold_index, held_out in enumerate(validation):
        held_names = {image.name for image in held_out}
        train_names = all_names - held_names
        fold_root = args.output / f"fold_{fold_index}"
        for split, names in (("train", train_names), ("val", held_names)):
            for name in sorted(names):
                source_image = image_root / name
                source_label = label_root / f"{source_image.stem}.txt"
                link_or_copy(source_image, fold_root / "images" / split / name)
                link_or_copy(source_label, fold_root / "labels" / split / source_label.name)
        yaml = args.output / f"fold_{fold_index}.yaml"
        yaml.write_text(
            f"path: {fold_root.resolve().as_posix()}\n"
            "train: images/train\nval: images/val\n\n"
            "names:\n  0: Flash point\n  1: Big black spots\n",
            encoding="utf-8",
        )
        counts = {"0": 0, "1": 0}
        for image in held_out:
            for class_id in class_signature(label_root / f"{image.stem}.txt"):
                counts[str(class_id)] += 1
        manifest["folds"].append({
            "fold": fold_index, "train_images": len(train_names),
            "heldout_images": len(held_names), "heldout_image_class_presence": counts,
            "yaml": str(yaml.resolve()),
        })
    # Every real training image must be held out exactly once.
    flattened = [image.name for fold in validation for image in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != all_names:
        raise RuntimeError("OOF fold assignment is not an exact partition")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"DASR OOF folds: {args.output.resolve()}")


if __name__ == "__main__":
    main()
