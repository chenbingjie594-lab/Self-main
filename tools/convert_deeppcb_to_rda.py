"""Convert official DeepPCB pairs into leakage-free RDA/MVTec-style data.

DeepPCB images contain several defects from different classes.  Each output
sample therefore keeps one real annotated defect patch on its aligned clean
template, producing a single-class target with a paired normal crop and a
coarse bounding-box mask.  The official trainval/test pair split is preserved
before instance samples are created.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",
    6: "pinhole",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        required=True,
        help="DeepPCB repository root or its PCBData directory.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--category", default="DeepPCB")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=sorted(CLASS_NAMES.values()),
        default=sorted(CLASS_NAMES.values()),
    )
    return parser.parse_args()


def resolve_data_root(source_root: Path) -> Path:
    source_root = source_root.expanduser().resolve()
    data_root = source_root / "PCBData"
    if data_root.is_dir():
        return data_root
    if (source_root / "trainval.txt").is_file() and (source_root / "test.txt").is_file():
        return source_root
    raise FileNotFoundError(
        f"Cannot find PCBData/trainval.txt/test.txt below {source_root}"
    )


def official_pairs(data_root: Path, list_name: str):
    list_path = data_root / list_name
    if not list_path.is_file():
        raise FileNotFoundError(list_path)
    pairs = []
    for line_number, raw_line in enumerate(
        list_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid {list_path}:{line_number}: {raw_line}")
        image_reference = Path(fields[0])
        annotation_path = data_root / Path(fields[1])
        image_directory = data_root / image_reference.parent
        pair_id = image_reference.stem
        template_path = image_directory / f"{pair_id}_temp.jpg"
        tested_path = image_directory / f"{pair_id}_test.jpg"
        for required in (template_path, tested_path, annotation_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        pairs.append((pair_id, template_path, tested_path, annotation_path))
    return pairs


def annotations(path: Path):
    values = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid {path}:{line_number}: {raw_line}")
        x1, y1, x2, y2, class_id = map(int, fields)
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unknown DeepPCB class id {class_id} in {path}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox in {path}:{line_number}")
        values.append((x1, y1, x2, y2, class_id))
    return values


def crop_box(image_size, target_box, crop_size):
    width, height = image_size
    if crop_size > width or crop_size > height:
        raise ValueError(
            f"crop_size={crop_size} exceeds DeepPCB image size {image_size}"
        )
    x1, y1, x2, y2 = target_box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    left = min(max(center_x - crop_size // 2, 0), width - crop_size)
    top = min(max(center_y - crop_size // 2, 0), height - crop_size)
    return left, top, left + crop_size, top + crop_size


def ensure_empty_output(output_root: Path):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root is not empty: {output_root}. Use a new directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def save_jpeg(image: Image.Image, path: Path, quality: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=quality, subsampling=0)


def convert_split(
    data_root,
    output_root,
    category,
    source_list,
    split_name,
    selected_classes,
    crop_size,
    jpeg_quality,
):
    category_root = output_root / split_name / category
    counts = Counter()
    manifest_entries = []
    seen_pair_ids = set()

    for pair_id, template_path, tested_path, annotation_path in official_pairs(
        data_root, source_list
    ):
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair id in {source_list}: {pair_id}")
        seen_pair_ids.add(pair_id)
        pair_annotations = annotations(annotation_path)
        with Image.open(template_path) as template_source, Image.open(
            tested_path
        ) as tested_source:
            template = template_source.convert("RGB")
            tested = tested_source.convert("RGB")
            if template.size != tested.size:
                raise ValueError(
                    f"Pair size mismatch for {pair_id}: {template.size} vs {tested.size}"
                )

            for instance_index, (x1, y1, x2, y2, class_id) in enumerate(
                pair_annotations
            ):
                class_name = CLASS_NAMES[class_id]
                if class_name not in selected_classes:
                    continue
                window = crop_box(template.size, (x1, y1, x2, y2), crop_size)
                left, top, _, _ = window
                normal_crop = template.crop(window)
                defect_crop = normal_crop.copy()

                # Retain only this real defect. All other annotated defects are
                # absent because the aligned clean template is the background.
                target_patch = tested.crop((x1, y1, x2, y2))
                local_box = (x1 - left, y1 - top, x2 - left, y2 - top)
                defect_crop.paste(target_patch, local_box[:2])

                mask = Image.new("L", (crop_size, crop_size), color=0)
                ImageDraw.Draw(mask).rectangle(
                    (
                        local_box[0],
                        local_box[1],
                        local_box[2] - 1,
                        local_box[3] - 1,
                    ),
                    fill=255,
                )

                sample_name = (
                    f"{pair_id}_{instance_index:02d}_{class_name}"
                )
                defect_path = (
                    category_root / "test" / class_name / f"{sample_name}.jpg"
                )
                mask_path = (
                    category_root
                    / "ground_truth"
                    / class_name
                    / f"{sample_name}.png"
                )
                normal_path = (
                    category_root / "train" / "good" / f"{sample_name}.jpg"
                )
                paired_normal_path = (
                    category_root
                    / "paired_normal"
                    / class_name
                    / f"{sample_name}.jpg"
                )
                save_jpeg(defect_crop, defect_path, jpeg_quality)
                save_jpeg(normal_crop, normal_path, jpeg_quality)
                save_jpeg(normal_crop, paired_normal_path, jpeg_quality)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask.save(mask_path)
                counts[class_name] += 1
                manifest_entries.append(
                    {
                        "pair_id": pair_id,
                        "instance_index": instance_index,
                        "class": class_name,
                        "source_template": str(template_path),
                        "source_tested": str(tested_path),
                        "source_annotation": str(annotation_path),
                        "source_bbox": [x1, y1, x2, y2],
                        "crop_window": list(window),
                        "image": str(defect_path.relative_to(output_root)),
                        "normal": str(normal_path.relative_to(output_root)),
                        "paired_normal": str(
                            paired_normal_path.relative_to(output_root)
                        ),
                        "mask": str(mask_path.relative_to(output_root)),
                    }
                )
    return {
        "source_list": source_list,
        "pair_count": len(seen_pair_ids),
        "sample_count": len(manifest_entries),
        "class_counts": dict(sorted(counts.items())),
        "samples": manifest_entries,
    }


def main():
    args = parse_args()
    if args.crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    data_root = resolve_data_root(Path(args.source_root))
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_empty_output(output_root)
    selected_classes = set(args.classes)

    manifest = {
        "source_root": str(data_root),
        "output_root": str(output_root),
        "category": args.category,
        "crop_size": args.crop_size,
        "jpeg_quality": args.jpeg_quality,
        "classes": args.classes,
        "class_id_mapping": CLASS_NAMES,
        "split_policy": "official DeepPCB trainval.txt/test.txt before instance conversion",
        "sample_policy": "aligned clean template plus one real defect bbox",
        "splits": {},
    }
    manifest["splits"]["train"] = convert_split(
        data_root,
        output_root,
        args.category,
        "trainval.txt",
        "train",
        selected_classes,
        args.crop_size,
        args.jpeg_quality,
    )
    manifest["splits"]["eval"] = convert_split(
        data_root,
        output_root,
        args.category,
        "test.txt",
        "eval",
        selected_classes,
        args.crop_size,
        args.jpeg_quality,
    )

    train_pairs = {
        sample["pair_id"] for sample in manifest["splits"]["train"]["samples"]
    }
    eval_pairs = {
        sample["pair_id"] for sample in manifest["splits"]["eval"]["samples"]
    }
    overlap = sorted(train_pairs & eval_pairs)
    if overlap:
        raise RuntimeError(f"Official split leakage detected: {overlap[:10]}")
    manifest["pair_overlap_count"] = 0
    manifest_path = output_root / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for split_name, values in manifest["splits"].items():
        print(
            f"{split_name}: pairs={values['pair_count']}, "
            f"samples={values['sample_count']}, classes={values['class_counts']}"
        )
    print(f"Output:   {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
