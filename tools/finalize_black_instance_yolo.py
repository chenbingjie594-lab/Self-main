"""Build the reviewed Black-instance YOLO dataset from normalized images.

This intentionally excludes exact duplicates and images for which the
pre-label detector produced no reliable localization. It converts every kept
localization to class 1 (Big black spots) and strips prediction confidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", required=True, type=Path)
    parser.add_argument("--prelabels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-id", type=int, default=1)
    args = parser.parse_args()

    images_out = args.output / "images"
    labels_out = args.output / "labels"
    review_out = args.output / "needs_manual_review"
    for directory in (images_out, labels_out, review_out):
        directory.mkdir(parents=True, exist_ok=True)

    seen = {}
    rows = []
    for image in sorted(args.normalized.glob("*.jpg"), key=numeric_key):
        digest = sha256(image)
        label = args.prelabels / f"{image.stem}.txt"
        if digest in seen:
            rows.append({"image": image.name, "status": "excluded_duplicate", "detail": seen[digest], "boxes": 0})
            continue
        seen[digest] = image.name
        if not label.exists() or not label.read_text(encoding="utf-8").strip():
            shutil.copy2(image, review_out / image.name)
            rows.append({"image": image.name, "status": "needs_manual_review", "detail": "no reliable prelabel", "boxes": 0})
            continue

        output_lines = []
        for raw in label.read_text(encoding="utf-8").splitlines():
            fields = raw.split()
            if len(fields) < 5:
                continue
            # Prelabels may contain class 0/1 and an optional confidence value.
            output_lines.append(" ".join([str(args.class_id), *fields[1:5]]))
        if not output_lines:
            shutil.copy2(image, review_out / image.name)
            rows.append({"image": image.name, "status": "needs_manual_review", "detail": "invalid prelabel", "boxes": 0})
            continue
        shutil.copy2(image, images_out / image.name)
        (labels_out / f"{image.stem}.txt").write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        rows.append({"image": image.name, "status": "kept", "detail": "class normalized to 1", "boxes": len(output_lines)})

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "status", "detail", "boxes"))
        writer.writeheader()
        writer.writerows(rows)

    kept = [row for row in rows if row["status"] == "kept"]
    duplicate = [row for row in rows if row["status"] == "excluded_duplicate"]
    review = [row for row in rows if row["status"] == "needs_manual_review"]
    print(f"kept_images={len(kept)} boxes={sum(int(row['boxes']) for row in kept)}")
    print(f"excluded_duplicates={len(duplicate)} needs_manual_review={len(review)}")


if __name__ == "__main__":
    main()
