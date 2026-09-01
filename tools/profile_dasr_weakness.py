"""Create a leakage-free DASR detector weakness profile from OOF folds.

Each ``--fold`` supplies held-out real images, their YOLO labels, and the YOLO
weights trained without those images.  Recall is measured in class-specific
relative-area tertiles and converted to bounded sampling weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .build_dasr_detection import box_iou, image_files
except ImportError:
    from build_dasr_detection import box_iou, image_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", nargs=3, action="append", required=True,
        metavar=("HELDOUT_IMAGES", "HELDOUT_LABELS", "WEIGHTS"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--match_confidence", type=float, default=0.25)
    parser.add_argument("--match_iou", type=float, default=0.50)
    return parser.parse_args()


def yolo_boxes(path: Path, width: int, height: int):
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id = int(fields[0])
        if class_id not in (0, 1):
            continue
        cx, cy, bw, bh = map(float, fields[1:])
        output.append({
            "class_id": class_id,
            "area_fraction": bw * bh,
            "xyxy": np.asarray([
                (cx - bw / 2) * width, (cy - bh / 2) * height,
                (cx + bw / 2) * width, (cy + bh / 2) * height,
            ], dtype=np.float32),
        })
    return output


def main():
    args = parse_args()
    from ultralytics import YOLO

    records = []
    fold_summaries = []
    for fold_index, (image_root, label_root, weights) in enumerate(args.fold):
        image_root, label_root, weights = Path(image_root), Path(label_root), Path(weights)
        images = image_files(image_root)
        if not images:
            raise ValueError(f"No held-out images under {image_root}")
        model = YOLO(str(weights))
        results = model.predict(
            source=[str(path) for path in images], imgsz=args.imgsz,
            device=args.device, conf=args.confidence, verbose=False, stream=False,
        )
        if len(results) != len(images):
            raise RuntimeError("OOF prediction count mismatch")
        fold_count = 0
        for image_path, result in zip(images, results):
            with Image.open(image_path) as image:
                width, height = image.size
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing held-out label: {label_path}")
            ground_truth = yolo_boxes(label_path, width, height)
            if result.boxes is None or len(result.boxes) == 0:
                predicted_boxes = np.zeros((0, 4), dtype=np.float32)
                predicted_classes = np.zeros(0, dtype=int)
                predicted_confidences = np.zeros(0, dtype=np.float32)
            else:
                predicted_boxes = result.boxes.xyxy.detach().cpu().numpy()
                predicted_classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                predicted_confidences = result.boxes.conf.detach().cpu().numpy()
            for item in ground_truth:
                keep = (predicted_classes == item["class_id"]) & (
                    predicted_confidences >= args.match_confidence
                )
                ious = box_iou(item["xyxy"], predicted_boxes[keep])
                best_iou = float(ious.max()) if len(ious) else 0.0
                records.append({
                    "class_id": item["class_id"],
                    "area_fraction": item["area_fraction"],
                    "detected": best_iou >= args.match_iou,
                    "best_iou": best_iou,
                    "fold": fold_index,
                })
                fold_count += 1
        fold_summaries.append({
            "fold": fold_index, "images": len(images), "boxes": fold_count,
            "weights": str(weights.resolve()),
        })

    bins = []
    for class_id in (0, 1):
        class_records = [record for record in records if record["class_id"] == class_id]
        if not class_records:
            raise ValueError(f"OOF folds contain no class {class_id} boxes")
        areas = np.asarray([record["area_fraction"] for record in class_records])
        edges = np.quantile(areas, [0.0, 1 / 3, 2 / 3, 1.0])
        edges[0] = max(0.0, edges[0] - 1e-12)
        edges[-1] = edges[-1] + 1e-12
        class_bins = []
        for index in range(3):
            selected = [
                record for record in class_records
                if edges[index] <= record["area_fraction"] <= edges[index + 1]
                and (index == 2 or record["area_fraction"] < edges[index + 1])
            ]
            recall = float(np.mean([record["detected"] for record in selected])) if selected else 0.0
            class_bins.append({
                "class_id": class_id,
                "area_low": float(edges[index]),
                "area_high": float(edges[index + 1]),
                "count": len(selected),
                "recall50": recall,
                "raw_weight": 1.1 - recall,
            })
        normalizer = np.average(
            [item["raw_weight"] for item in class_bins],
            weights=[max(item["count"], 1) for item in class_bins],
        )
        for item in class_bins:
            item["weight"] = float(np.clip(item.pop("raw_weight") / max(normalizer, 1e-6), 0.5, 2.0))
            bins.append(item)

    output = {
        "version": 1,
        "match_confidence": args.match_confidence,
        "match_iou": args.match_iou,
        "folds": fold_summaries,
        "bins": bins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"DASR weakness profile: {args.output.resolve()}")


if __name__ == "__main__":
    main()
