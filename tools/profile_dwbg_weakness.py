"""Leakage-free OOF detector weakness profiling for DWBG stage one.

The module intentionally keeps all statistics as pure functions so they can be
tested without Ultralytics or detector weights.  Model loading is confined to
``main``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .dwbg_utils import (
        BIN_NAMES, compute_bbox_geometry, compute_local_contrast,
        context_ring_masks, native, quantile_labels,
    )
except ImportError:
    from dwbg_utils import (
        BIN_NAMES, compute_bbox_geometry, compute_local_contrast,
        context_ring_masks, native, quantile_labels,
    )

try:
    from .build_dasr_detection import box_iou, image_files
except ImportError:
    from build_dasr_detection import box_iou, image_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--ring_scale", type=float, default=1.5)
    parser.add_argument("--min_joint_count", type=int, default=3)
    parser.add_argument("--top_k_modes", type=int, default=10)
    args = parser.parse_args()
    if args.ring_scale <= 1.0:
        parser.error("--ring_scale must be greater than 1")
    if args.min_joint_count < 1:
        parser.error("--min_joint_count must be positive")
    if args.top_k_modes < 1:
        parser.error("--top_k_modes must be positive")
    return args


def read_yolo_boxes(path: Path, width: int, height: int):
    """Return class-0/1 GT records with normalized and pixel geometry."""
    records = []
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
        if bw <= 0 or bh <= 0:
            raise ValueError(f"{path}:{line_number}: bbox width/height must be positive")
        x0 = float(np.clip((cx - bw / 2) * width, 0, width))
        y0 = float(np.clip((cy - bh / 2) * height, 0, height))
        x1 = float(np.clip((cx + bw / 2) * width, 0, width))
        y1 = float(np.clip((cy + bh / 2) * height, 0, height))
        records.append({
            "class_id": class_id,
            **compute_bbox_geometry([x0, y0, x1, y1], width, height),
            "xyxy": [x0, y0, x1, y1],
        })
    return records


def local_contrast(gray, xyxy, ring_scale=1.5, eps=1e-6):
    return compute_local_contrast(gray, xyxy, ring_scale, eps)


def match_prediction(gt_xyxy, boxes, classes, confidences, class_id,
                     match_iou=0.5, match_confidence=0.25):
    """Match the same-class prediction with maximum IoU, then read its conf."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    keep = classes == int(class_id)
    same_boxes, same_confidences = boxes[keep], confidences[keep]
    ious = box_iou(np.asarray(gt_xyxy, dtype=np.float32), same_boxes)
    if not len(ious):
        best_iou = matched_confidence = 0.0
    else:
        best = int(np.argmax(ious))
        best_iou = float(ious[best])
        matched_confidence = float(same_confidences[best])
    return {
        "best_iou": best_iou,
        "matched_confidence": matched_confidence,
        "detected": bool(best_iou >= match_iou and matched_confidence >= match_confidence),
    }


def bin_statistics(records):
    count = int(len(records))
    detected = int(sum(bool(record["detected"]) for record in records))
    recall = float(detected / count) if count else 0.0
    smoothed = float((detected + 1) / (count + 2))
    return {
        "count": count,
        "detected": detected,
        "recall": recall,
        "smoothed_recall": smoothed,
        "weakness": float(1.0 - smoothed),
        "mean_confidence": float(np.mean([r["matched_confidence"] for r in records])) if count else 0.0,
        "mean_iou": float(np.mean([r["best_iou"] for r in records])) if count else 0.0,
    }


def build_profiles(instances, min_joint_count=3, top_k_modes=10):
    class_profiles, failure_modes = {}, {}
    for class_id in (0, 1):
        records = [record for record in instances if int(record["class_id"]) == class_id]
        thresholds = {}
        dimensions = (
            ("scale", "area_fraction"),
            ("contrast", "local_contrast"),
            ("morphology", "elongation"),
        )
        for dimension, value_name in dimensions:
            labels, edge = quantile_labels(
                [record[value_name] for record in records], BIN_NAMES[dimension]
            )
            thresholds[dimension] = edge
            for record, label in zip(records, labels):
                record[f"{dimension}_bin"] = label

        profile = {"overall": bin_statistics(records), "thresholds": thresholds}
        for dimension, _ in dimensions:
            groups = defaultdict(list)
            for record in records:
                groups[record[f"{dimension}_bin"]].append(record)
            profile[dimension] = []
            for name in BIN_NAMES[dimension]:
                if name in groups:
                    profile[dimension].append({"name": name, **bin_statistics(groups[name])})

        joint_groups = defaultdict(list)
        for record in records:
            key = (record["scale_bin"], record["contrast_bin"], record["morphology_bin"])
            joint_groups[key].append(record)
        joint = []
        for (scale, contrast, morphology), members in joint_groups.items():
            item = {
                "class_id": class_id, "scale": scale, "contrast": contrast,
                "morphology": morphology, **bin_statistics(members),
            }
            item["eligible"] = bool(item["count"] >= min_joint_count)
            joint.append(item)
        joint.sort(key=lambda item: (-item["weakness"], -item["count"],
                                     item["scale"], item["contrast"], item["morphology"]))
        profile["joint_modes"] = joint
        class_profiles[str(class_id)] = profile
        failure_modes[str(class_id)] = [item for item in joint if item["eligible"]][:top_k_modes]
    return class_profiles, failure_modes


def print_summary(profile):
    print("=" * 60)
    print("DWBG Detector Weakness Profile")
    print("=" * 60)
    for class_id in ("0", "1"):
        item = profile["class_profiles"][class_id]
        print(f"\nClass {class_id}")
        print(f"Overall Recall: {item['overall']['recall']:.3f} "
              f"(n={item['overall']['count']})")
        for dimension in ("scale", "contrast", "morphology"):
            print(f"\n{dimension.title()}:")
            for row in item[dimension]:
                print(f"{row['name']:<12} Recall={row['recall']:.3f}  "
                      f"Weakness={row['weakness']:.3f}  n={row['count']}")
        print("\nTop failure modes:")
        modes = profile["failure_modes"][class_id]
        if not modes:
            print("  (none eligible)")
        for index, mode in enumerate(modes, 1):
            print(f"{index}. {mode['scale']} + {mode['contrast']}_contrast + "
                  f"{mode['morphology']}  Weakness={mode['weakness']:.3f} "
                  f"n={mode['count']}")
    print("\n" + "=" * 60)


def main():
    args = parse_args()
    from ultralytics import YOLO

    instances, fold_summaries = [], []
    for fold_index, (image_root, label_root, weights) in enumerate(args.fold):
        image_root, label_root, weights = Path(image_root), Path(label_root), Path(weights)
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"OOF fold {fold_index} image directory does not exist: {image_root}. "
                "Run tools/create_dasr_yolo_folds.py first and verify fold_0/fold_1/fold_2."
            )
        if not label_root.is_dir():
            raise FileNotFoundError(
                f"OOF fold {fold_index} label directory does not exist: {label_root}. "
                "The --fold image and label paths must come from the same fold."
            )
        if not weights.is_file():
            raise FileNotFoundError(
                f"OOF fold {fold_index} weights do not exist: {weights}. "
                "Train the corresponding fold detector before profiling."
            )
        images = image_files(image_root)
        if not images:
            raise ValueError(f"No held-out images under {image_root}")
        model = YOLO(str(weights))
        predictions = model.predict(
            source=[str(path) for path in images], imgsz=args.imgsz,
            device=args.device, conf=args.confidence, verbose=False, stream=False,
        )
        if len(predictions) != len(images):
            raise RuntimeError("OOF prediction count mismatch")
        fold_instance_count = 0
        for image_path, prediction in zip(images, predictions):
            image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            height, width = image.shape
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing held-out label: {label_path}")
            ground_truth = read_yolo_boxes(label_path, width, height)
            if prediction.boxes is None or len(prediction.boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros(0, dtype=np.int64)
                confidences = np.zeros(0, dtype=np.float32)
            else:
                boxes = prediction.boxes.xyxy.detach().cpu().numpy()
                classes = prediction.boxes.cls.detach().cpu().numpy().astype(np.int64)
                confidences = prediction.boxes.conf.detach().cpu().numpy()
            for gt_index, gt in enumerate(ground_truth):
                contrast = local_contrast(image, gt["xyxy"], args.ring_scale)
                response = match_prediction(
                    gt["xyxy"], boxes, classes, confidences, gt["class_id"],
                    args.match_iou, args.match_confidence,
                )
                instances.append({
                    "image": str(image_path), "label": str(label_path),
                    "fold": fold_index, "instance_index": gt_index,
                    **{key: value for key, value in gt.items() if key != "xyxy"},
                    "bbox_xyxy": gt["xyxy"], **contrast, **response,
                })
                fold_instance_count += 1
        fold_summaries.append({
            "fold": fold_index, "heldout_images": len(images),
            "instances": fold_instance_count, "images": str(image_root),
            "labels": str(label_root), "weights": str(weights),
        })

    class_profiles, failure_modes = build_profiles(
        instances, args.min_joint_count, args.top_k_modes
    )
    output = native({
        "version": 2,
        "settings": {
            "prediction_confidence": args.confidence,
            "match_confidence": args.match_confidence,
            "match_iou": args.match_iou,
            "ring_scale": args.ring_scale,
            "min_joint_count": args.min_joint_count,
            "top_k_modes": args.top_k_modes,
            "binning": "class-specific tertiles with duplicate-edge merging",
            "smoothing": "Beta(1,1) / Laplace",
        },
        "folds": fold_summaries,
        "instances": instances,
        "class_profiles": class_profiles,
        "failure_modes": failure_modes,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(output)
    print(f"DWBG weakness JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
