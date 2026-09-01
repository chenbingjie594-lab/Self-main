"""Verify that two generation runs used identical conditions.

The generated images are expected to differ. File names, paired normal images,
and masks must be byte-identical for a fair method comparison.
"""

import argparse
import hashlib
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--category", default="DeepPCB")
    return parser.parse_args()


def files(folder):
    return {
        path.name: path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    }


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.digest()


def main():
    args = parse_args()
    first = Path(args.first) / args.category
    second = Path(args.second) / args.category
    first_classes = {p.name for p in first.iterdir() if p.is_dir()}
    second_classes = {p.name for p in second.iterdir() if p.is_dir()}
    if first_classes != second_classes:
        raise RuntimeError(
            f"Class mismatch: first={sorted(first_classes)}, "
            f"second={sorted(second_classes)}"
        )

    total = 0
    for defect in sorted(first_classes):
        image_names = None
        for subfolder in ("image", "normal", "masks"):
            left = files(first / defect / subfolder)
            right = files(second / defect / subfolder)
            if set(left) != set(right):
                raise RuntimeError(
                    f"{defect}/{subfolder}: file-name sets differ "
                    f"({len(left)} vs {len(right)})"
                )
            if subfolder == "image":
                image_names = set(left)
            else:
                mismatched = [
                    name for name in left if digest(left[name]) != digest(right[name])
                ]
                if mismatched:
                    raise RuntimeError(
                        f"{defect}/{subfolder}: {len(mismatched)} condition files "
                        f"differ; examples={mismatched[:5]}"
                    )
        total += len(image_names)
        print(f"{defect}: matched={len(image_names)}")
    print(f"MATCHED CONDITIONS: classes={len(first_classes)}, samples={total}")


if __name__ == "__main__":
    main()
