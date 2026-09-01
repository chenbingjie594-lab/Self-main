"""Create a YOLO dataset copy with selected classes removed safely.

Images containing a removed class are excluded completely.  Removing only the
corresponding label row would leave a visible but unlabeled object and teach the
detector to treat that object as background.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["real_only", "real_baseline", "real_carf"])
    parser.add_argument("--remove-class-id", type=int, required=True)
    parser.add_argument("--remove-class-name", required=True)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_label(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def filter_group(source: Path, output: Path, group: str, removed_id: int) -> dict[str, int]:
    stats = {"kept": 0, "excluded": 0, "boxes": 0}
    for split in ("train", "val"):
        image_dir = source / group / "images" / split
        label_dir = source / group / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_dir / f"{image_path.stem}.txt"
            rows = read_label(label_path)
            ids = [int(row.split(maxsplit=1)[0]) for row in rows]
            if removed_id in ids:
                stats["excluded"] += 1
                continue

            remapped = []
            for row, class_id in zip(rows, ids):
                fields = row.split()
                fields[0] = str(class_id - 1 if class_id > removed_id else class_id)
                remapped.append(" ".join(fields))

            target_image = output / group / "images" / split / image_path.name
            target_label = output / group / "labels" / split / label_path.name
            link_or_copy(image_path, target_image)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text(("\n".join(remapped) + "\n") if remapped else "", encoding="utf-8")
            stats["kept"] += 1
            stats["boxes"] += len(remapped)
    return stats


def write_yaml(output: Path, group: str, names: list[str]) -> None:
    lines = [f"path: {output / group}", "train: images/train", "val: images/val", "", "names:"]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / f"{group}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")

    # Plastic_Bomo's original YOLO class order.
    names = [
        "Fine black spots",
        "Flash point",
        "Big black spots",
        "Black lines",
        "Scar",
        "Oil stains",
        "Colour difference",
        "Insect",
        "Bubble",
    ]
    if not 0 <= args.remove_class_id < len(names):
        raise ValueError(f"remove-class-id must be in [0, {len(names) - 1}]")
    if names[args.remove_class_id].casefold() != args.remove_class_name.casefold():
        raise ValueError(
            f"Class {args.remove_class_id} is {names[args.remove_class_id]!r}, "
            f"not {args.remove_class_name!r}"
        )
    kept_names = names[: args.remove_class_id] + names[args.remove_class_id + 1 :]

    args.output.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        stats = filter_group(args.source, args.output, group, args.remove_class_id)
        write_yaml(args.output, group, kept_names)
        print(f"{group}: kept={stats['kept']} excluded={stats['excluded']} boxes={stats['boxes']}")

    print(f"Created filtered dataset: {args.output}")


if __name__ == "__main__":
    main()
