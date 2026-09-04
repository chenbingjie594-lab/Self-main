"""Build class-specific real TRAINING feature banks for each OOF detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .dwbg_feature_extraction import DetectInputExtractor
    from .dwbg_utils import native
    from .profile_dwbg_weakness import read_yolo_boxes
    from .dwbg_v2_utils import manifold_reference
except ImportError:
    from dwbg_feature_extraction import DetectInputExtractor
    from dwbg_utils import native
    from profile_dwbg_weakness import read_yolo_boxes
    from dwbg_v2_utils import manifold_reference


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", nargs=3, action="append", required=True,
                   metavar=("TRAIN_IMAGES", "TRAIN_LABELS", "WEIGHTS"))
    p.add_argument("--output_dir", type=Path, default=Path("results/dwbg/v2/feature_banks"))
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--quantile", type=float, default=.95)
    return p.parse_args()


def image_files(folder):
    return sorted(x for x in folder.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"version": 2, "source": "real_training_only", "folds": []}
    for fold, (image_dir, label_dir, weights) in enumerate(args.fold):
        image_dir, label_dir, weights = Path(image_dir), Path(label_dir), Path(weights)
        if not image_dir.is_dir() or not label_dir.is_dir() or not weights.is_file():
            raise FileNotFoundError(f"Invalid fold {fold}: {image_dir}, {label_dir}, {weights}")
        extractor = DetectInputExtractor(weights, args.device, args.imgsz)
        vectors, classes, records = [], [], []
        try:
            for image_path in image_files(image_dir):
                label = label_dir / f"{image_path.stem}.txt"
                with Image.open(image_path) as image:
                    width, height = image.size
                for index, gt in enumerate(read_yolo_boxes(label, width, height)):
                    cid = int(gt["class_id"])
                    vectors.append(extractor.encode(image_path, gt["xyxy"], width, height))
                    classes.append(cid)
                    records.append({"image": str(image_path.resolve()), "class_id": cid,
                                    "bbox": gt["xyxy"], "instance_index": index})
        finally:
            extractor.close()
        if not vectors: raise ValueError(f"No real instances in {image_dir}")
        features = np.stack(vectors).astype(np.float32); classes = np.asarray(classes, dtype=np.int64)
        refs = {str(cid): manifold_reference(features[classes == cid], args.k, args.quantile)
                for cid in (0, 1)}
        npz = args.output_dir / f"real_feature_bank_fold{fold}.npz"
        np.savez_compressed(npz, features=features, class_ids=classes)
        meta = args.output_dir / f"real_feature_bank_fold{fold}_metadata.json"
        meta.write_text(json.dumps(native({"fold": fold, "weights": str(weights.resolve()),
            "images": str(image_dir.resolve()), "records": records, "references": refs}), ensure_ascii=False, indent=2), encoding="utf-8")
        summary["folds"].append({"fold": fold, "bank": str(npz.resolve()), "metadata": str(meta.resolve()),
            "counts": {str(cid): int((classes == cid).sum()) for cid in (0,1)}})
    (args.output_dir / "feature_bank_manifest.json").write_text(json.dumps(native(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(native(summary), ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
