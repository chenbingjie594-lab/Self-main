"""Recover original DeepPCB sample stems for legacy index-named outputs."""

import argparse
import hashlib
import itertools
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--category", default="DeepPCB")
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--feature_size", type=int, default=64)
    return parser.parse_args()


def image_files(folder):
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def matching_file(folder, stem):
    for suffix in IMAGE_SUFFIXES:
        path = Path(folder) / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No file matching {stem} under {folder}")


def mask_signature(path):
    array = np.asarray(Image.open(path).convert("L")) > 127
    shape = np.asarray(array.shape, dtype=np.int32).tobytes()
    return hashlib.sha1(shape + np.packbits(array).tobytes()).hexdigest()


def normal_feature(path, size):
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def best_assignment(outputs, candidates, output_features, candidate_features):
    if len(outputs) != len(candidates):
        raise ValueError(
            f"Signature count mismatch: outputs={len(outputs)}, candidates={len(candidates)}"
        )
    scored = []
    for permutation in itertools.permutations(candidates):
        cost = sum(
            float(np.mean((output_features[out] - candidate_features[candidate]) ** 2))
            for out, candidate in zip(outputs, permutation)
        )
        scored.append((cost, permutation))
    scored.sort(key=lambda item: item[0])
    margin = None if len(scored) == 1 else scored[1][0] - scored[0][0]
    return dict(zip(outputs, scored[0][1])), scored[0][0], margin


def place(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        os.link(source, destination)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    generated_root = Path(args.generated_root)
    output_root = Path(args.output_root)
    category = args.category
    generated_category = generated_root / category
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")

    manifest = {
        "dataset_root": str(dataset_root.resolve()),
        "generated_root": str(generated_root.resolve()),
        "output_root": str(output_root.resolve()),
        "category": category,
        "classes": {},
    }
    for defect_dir in sorted(path for path in generated_category.iterdir() if path.is_dir()):
        defect = defect_dir.name
        output_masks = image_files(defect_dir / "masks")
        output_images = {path.stem: path for path in image_files(defect_dir / "image")}
        output_normals = {path.stem: path for path in image_files(defect_dir / "normal")}
        source_masks = image_files(dataset_root / category / "ground_truth" / defect)

        outputs_by_signature = defaultdict(list)
        candidates_by_signature = defaultdict(list)
        for path in output_masks:
            outputs_by_signature[mask_signature(path)].append(path)
        for path in source_masks:
            candidates_by_signature[mask_signature(path)].append(path)
        if set(outputs_by_signature) != set(candidates_by_signature):
            raise ValueError(f"Mask signatures differ for {category}/{defect}")

        output_features = {
            path: normal_feature(output_normals[path.stem], args.feature_size)
            for path in output_masks
        }
        candidate_features = {
            path: normal_feature(
                matching_file(
                    dataset_root / category / "paired_normal" / defect, path.stem
                ),
                args.feature_size,
            )
            for path in source_masks
        }
        recovered = {}
        ambiguous_groups = []
        for signature in sorted(outputs_by_signature):
            outputs = sorted(outputs_by_signature[signature], key=lambda path: int(path.stem))
            candidates = sorted(candidates_by_signature[signature])
            assignment, cost, margin = best_assignment(
                outputs, candidates, output_features, candidate_features
            )
            recovered.update(assignment)
            if len(outputs) > 1:
                ambiguous_groups.append(
                    {
                        "size": len(outputs),
                        "best_cost": cost,
                        "next_best_margin": margin,
                    }
                )

        if len(recovered) != len(source_masks):
            raise ValueError(f"Incomplete recovery for {category}/{defect}")
        if len({path.stem for path in recovered.values()}) != len(recovered):
            raise ValueError(f"Recovery is not one-to-one for {category}/{defect}")

        entries = []
        for output_mask, source_mask in sorted(
            recovered.items(), key=lambda item: int(item[0].stem)
        ):
            old_stem = output_mask.stem
            new_name = f"{source_mask.stem}.jpg"
            for subdir, source in (
                ("image", output_images[old_stem]),
                ("normal", output_normals[old_stem]),
                ("masks", output_mask),
            ):
                place(
                    source,
                    output_root / category / defect / subdir / new_name,
                    args.mode,
                )
            entries.append(
                {
                    "legacy_name": output_mask.name,
                    "source_stem": source_mask.stem,
                    "source_mask": source_mask.name,
                }
            )
        manifest["classes"][defect] = {
            "count": len(entries),
            "duplicate_signature_groups": ambiguous_groups,
            "entries": entries,
        }
        print(
            f"{category}/{defect}: recovered={len(entries)}, "
            f"duplicate_groups={len(ambiguous_groups)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Recovered view: {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
