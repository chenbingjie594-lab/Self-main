"""Normalize Black instance images and create reviewable YOLO box candidates.

The source set contains 4096x1024 JPG files and screenshot-like PNG files with
white horizontal padding. PNG content is cropped to its non-white band before
resizing to 4096x1024. Candidate boxes are obtained from local dark contrast;
they must be visually reviewed before being treated as final annotations.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def numeric_key(path: Path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def crop_content(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rgb = np.asarray(image.convert("RGB"))
    # White screenshot padding has almost all pixels near 255 in a full row.
    white_fraction = (rgb.min(axis=2) >= 245).mean(axis=1)
    valid = np.flatnonzero(white_fraction < 0.98)
    if len(valid) == 0:
        return image.convert("RGB"), (0, 0, image.width, image.height)
    top, bottom = int(valid[0]), int(valid[-1] + 1)
    return image.convert("RGB").crop((0, top, image.width, bottom)), (0, top, image.width, bottom)


def components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    output = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        x0 = x1 = int(x)
        y0 = y1 = int(y)
        area = 0
        while queue:
            cy, cx = queue.popleft()
            area += 1
            x0, x1 = min(x0, cx), max(x1, cx)
            y0, y1 = min(y0, cy), max(y1, cy)
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        output.append((x0, y0, x1 + 1, y1 + 1, area))
    return output


def candidate_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    # Work at half resolution: real black points remain visible but film texture
    # and memory usage are reduced.
    small = image.resize((2048, 512), Image.Resampling.LANCZOS).convert("L")
    arr = np.asarray(small, dtype=np.int16)
    local = np.asarray(small.filter(ImageFilter.GaussianBlur(10)), dtype=np.int16)
    contrast = local - arr
    mask = (contrast >= 14) & (local >= 38)
    # Remove isolated noise while preserving compact dark spots.
    cleaned = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    comps = components(np.asarray(cleaned) > 0)
    boxes = []
    for x0, y0, x1, y1, area in comps:
        bw, bh = x1 - x0, y1 - y0
        if not (3 <= bw <= 45 and 3 <= bh <= 45 and 8 <= area <= 900):
            continue
        # Reject long scratch-like structures.
        if max(bw / max(bh, 1), bh / max(bw, 1)) > 3.0:
            continue
        pad = 5
        boxes.append((max(0, 2 * (x0 - pad)), max(0, 2 * (y0 - pad)), min(4096, 2 * (x1 + pad)), min(1024, 2 * (y1 + pad))))
    return boxes


def yolo_line(box, width=4096, height=1024, class_id=1):
    x0, y0, x1, y1 = box
    return f"{class_id} {(x0+x1)/(2*width):.6f} {(y0+y1)/(2*height):.6f} {(x1-x0)/width:.6f} {(y1-y0)/height:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-id", type=int, default=1)
    args = parser.parse_args()

    image_dir = args.output / "images"
    label_dir = args.output / "labels_candidates"
    review_dir = args.output / "review"
    for directory in (image_dir, label_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    files = sorted((p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=numeric_key)
    manifest = []
    thumbs = []
    font = ImageFont.load_default()
    for source in files:
        with Image.open(source) as raw:
            cropped, crop = crop_content(raw)
            normalized = cropped.resize((4096, 1024), Image.Resampling.LANCZOS)
        output_name = f"{source.stem}.jpg"
        normalized.save(image_dir / output_name, quality=95)
        boxes = candidate_boxes(normalized)
        (label_dir / f"{source.stem}.txt").write_text(
            "\n".join(yolo_line(b, class_id=args.class_id) for b in boxes) + ("\n" if boxes else ""),
            encoding="utf-8",
        )
        review = normalized.copy()
        draw = ImageDraw.Draw(review)
        for box in boxes:
            draw.rectangle(box, outline=(255, 0, 0), width=8)
        thumb = review.resize((512, 128), Image.Resampling.LANCZOS)
        ImageDraw.Draw(thumb).rectangle((0, 0, 90, 18), fill=(255, 255, 255))
        ImageDraw.Draw(thumb).text((4, 3), f"{source.name}: {len(boxes)}", fill=(0, 0, 0), font=font)
        thumbs.append(thumb)
        review.save(review_dir / output_name, quality=90)
        manifest.append({"source": source.name, "crop": str(crop), "output": output_name, "candidate_boxes": len(boxes)})

    sheet = Image.new("RGB", (1024, ((len(thumbs) + 1) // 2) * 128), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 512, (index // 2) * 128))
    sheet.save(args.output / "candidate_contact_sheet.jpg", quality=92)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
