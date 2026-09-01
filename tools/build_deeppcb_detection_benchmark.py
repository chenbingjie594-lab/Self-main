"""Build leakage-free DeepPCB YOLO detection augmentation benchmarks.

Real training crops come from the converted official train split. Synthetic
training crops must also be generated from that split. The official eval split
is copied only to validation and is identical for every experiment group.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ("mousebite", "open", "pinhole", "short", "spur", "spurious_copper")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--baseline_generated", required=True)
    parser.add_argument("--carf_generated", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--category", default="DeepPCB")
    parser.add_argument("--link_mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--yaml_base", default=None)
    return parser.parse_args()


def image_files(folder):
    return sorted(
        (p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )


def matching_file(folder, stem):
    for candidate_stem in (stem, f"{stem}_mask"):
        for suffix in SUFFIXES:
            path = Path(folder) / f"{candidate_stem}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No mask for {stem} under {folder}")


def resolve_generated(root, category):
    root = Path(root).resolve()
    direct = root / category
    if direct.is_dir():
        return root
    matches = sorted(p.parent for p in root.rglob(category) if p.is_dir())
    valid = [
        p for p in matches
        if all((p / category / cls / "image").is_dir() for cls in CLASSES)
    ]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one generated experiment under {root}, found {valid}")
    return valid[0]


def mask_box(mask_path):
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError(f"Empty mask: {mask_path}")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (
        ((x0 + x1) / 2.0) / width,
        ((y0 + y1) / 2.0) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def place(source, target, mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def add_samples(image_root, mask_root, destination, split, prefix, mode, stats):
    for class_id, defect in enumerate(CLASSES):
        images = image_files(Path(image_root) / defect)
        for index, image_path in enumerate(images):
            mask_path = matching_file(Path(mask_root) / defect, image_path.stem)
            name = f"{prefix}_{defect}_{index:05d}"
            output_image = destination / "images" / split / f"{name}{image_path.suffix.lower()}"
            output_label = destination / "labels" / split / f"{name}.txt"
            place(image_path, output_image, mode)
            box = mask_box(mask_path)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(
                f"{class_id} " + " ".join(f"{value:.8f}" for value in box) + "\n",
                encoding="utf-8",
            )
            stats[defect] += 1


def add_real(root, destination, split, prefix, category, mode, stats):
    category_root = Path(root) / category
    add_samples(
        category_root / "test",
        category_root / "ground_truth",
        destination,
        split,
        prefix,
        mode,
        stats,
    )


def add_generated(root, destination, prefix, category, mode, stats):
    category_root = Path(root) / category
    for class_id, defect in enumerate(CLASSES):
        base = category_root / defect
        images = image_files(base / "image")
        for index, image_path in enumerate(images):
            mask_path = matching_file(base / "masks", image_path.stem)
            name = f"{prefix}_{defect}_{index:05d}"
            output_image = destination / "images/train" / f"{name}{image_path.suffix.lower()}"
            output_label = destination / "labels/train" / f"{name}.txt"
            place(image_path, output_image, mode)
            box = mask_box(mask_path)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(
                f"{class_id} " + " ".join(f"{value:.8f}" for value in box) + "\n",
                encoding="utf-8",
            )
            stats[defect] += 1


def write_yaml(output_root, group, yaml_base):
    path_root = (
        f"{yaml_base.rstrip('/')}/{group}" if yaml_base else (output_root / group).as_posix()
    )
    lines = [
        f"path: {path_root}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ] + [f"  {index}: {name}" for index, name in enumerate(CLASSES)]
    (output_root / f"{group}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output must be empty: {output_root}")
    baseline = resolve_generated(args.baseline_generated, args.category)
    carf = resolve_generated(args.carf_generated, args.category)
    groups = {
        "real_only": None,
        "real_baseline": baseline,
        "real_carf": carf,
    }
    for group, synthetic in groups.items():
        destination = output_root / group
        stats = {"train_real": {name: 0 for name in CLASSES},
                 "train_synthetic": {name: 0 for name in CLASSES},
                 "val_real": {name: 0 for name in CLASSES}}
        add_real(args.train_root, destination, "train", "real", args.category,
                 args.link_mode, stats["train_real"])
        if synthetic is not None:
            add_generated(synthetic, destination, synthetic.name, args.category,
                          args.link_mode, stats["train_synthetic"])
        add_real(args.eval_root, destination, "val", "eval", args.category,
                 args.link_mode, stats["val_real"])
        write_yaml(output_root, group, args.yaml_base)
        print(group, stats)

    # File-name identity is sufficient because validation is built once from
    # the same eval source in the deterministic loop above.
    reference = {p.name for p in (output_root / "real_only/images/val").iterdir()}
    for group in ("real_baseline", "real_carf"):
        current = {p.name for p in (output_root / group / "images/val").iterdir()}
        if current != reference:
            raise RuntimeError(f"Validation mismatch for {group}")
    print(f"VALIDATION IDENTICAL: images={len(reference)}")


if __name__ == "__main__":
    main()
