"""Build a clean downstream benchmark for selected generators without HGRR.

The real-only dataset already uses the two-class IDs 0/1. Stage-1 synthetic
sets use the original Plastic Bomo IDs 1/2, which are remapped to 0/1 here.
Validation data is copied unchanged and identically to all three groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_MAP = {1: 0, 2: 1}
GROUP_METHOD = {
    "real_only": None,
    "real_baseline": "Baseline",
    "real_carf": "CARF",
    "real_msdf": "MSDF",
}


def copy_tree_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_synthetic_label(source: Path, target: Path) -> tuple[int, int]:
    counts = [0, 0]
    output = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{index}: expected 5 fields")
        old_id = int(fields[0])
        if old_id not in CLASS_MAP:
            raise ValueError(f"{source}:{index}: unsupported class {old_id}")
        new_id = CLASS_MAP[old_id]
        fields[0] = str(new_id)
        output.append(" ".join(fields))
        counts[new_id] += 1
    target.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return counts[0], counts[1]


def count_boxes(label_dir: Path) -> tuple[int, int]:
    counts = [0, 0]
    for label in label_dir.glob("*.txt"):
        if label.name == "classes.txt":
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            if class_id not in (0, 1):
                raise ValueError(f"Unexpected class {class_id} in {label}")
            counts[class_id] += 1
    return counts[0], counts[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(GROUP_METHOD),
        default=list(GROUP_METHOD),
        help="Build only the requested experiment branches.",
    )
    parser.add_argument(
        "--yaml_base",
        default=None,
        help="Optional dataset path prefix used in YAML (for example Y:/dataset).",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    rows = []
    selected_groups = {group: GROUP_METHOD[group] for group in args.groups}
    for group, method in selected_groups.items():
        group_root = args.output / group
        for split in ("train", "val"):
            copy_tree_files(args.real / "images" / split, group_root / "images" / split)
            copy_tree_files(args.real / "labels" / split, group_root / "labels" / split)

        if method is not None:
            source_images = args.synthetic / method / "images"
            source_labels = args.synthetic / method / "labels"
            prefix = f"synthetic_{method.lower()}_"
            for image in sorted(p for p in source_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                label = source_labels / f"{image.stem}.txt"
                if not label.exists():
                    raise FileNotFoundError(f"Missing label for {image}")
                image_name = prefix + image.name
                label_name = prefix + image.stem + ".txt"
                shutil.copy2(image, group_root / "images" / "train" / image_name)
                remap_synthetic_label(label, group_root / "labels" / "train" / label_name)

        yaml_root = (
            Path(args.yaml_base) / args.output.name / group
            if args.yaml_base
            else (args.output.resolve() / group)
        )
        yaml = (
            f"path: {yaml_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n"
            "  0: Flash point\n"
            "  1: Big black spots\n"
        )
        (args.output / f"{group}.yaml").write_text(yaml, encoding="utf-8")

        for split in ("train", "val"):
            images = [p for p in (group_root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
            labels = [p for p in (group_root / "labels" / split).glob("*.txt") if p.name != "classes.txt"]
            flash, black = count_boxes(group_root / "labels" / split)
            rows.append({"group": group, "split": split, "images": len(images), "labels": len(labels), "flash_boxes": flash, "black_boxes": black})

    # Exact validation image equality is required across all groups.
    validation_hashes = []
    for group in selected_groups:
        files = sorted((args.output / group / "images" / "val").iterdir())
        validation_hashes.append([(p.name, sha256(p)) for p in files if p.suffix.lower() in IMAGE_SUFFIXES])
    if not all(item == validation_hashes[0] for item in validation_hashes[1:]):
        raise RuntimeError("Validation images differ across groups")

    with (args.output / "dataset_stats.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("validation_identical=True")


if __name__ == "__main__":
    main()
