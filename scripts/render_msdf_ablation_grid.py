#!/usr/bin/env python3
"""Render fixed-slot MSDF ablation comparisons from real PNG manifests."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path):
    value = load(path)
    rows = value.get("records", value.get("slots", value))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest has no records: {path}")
    return rows


def pick(row, *names):
    scopes = [row, row.get("frozen_slot", {}), row.get("runtime_slot", {})]
    for scope in scopes:
        for name in names:
            if scope.get(name) is not None:
                return scope[name]
    return None


def resolve(manifest, value):
    if value is None:
        raise ValueError(f"Missing required path in {manifest}")
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def generated_path(manifest, row):
    return resolve(manifest, pick(row, "image_path", "output_path", "generated_path"))


def normal_path(manifest, row):
    return resolve(manifest, pick(row, "normal_path", "conditioning_normal_path",
                                  "background_path", "normal_image_path"))


def reference_path(manifest, row):
    return resolve(manifest, pick(row, "reference_path", "source_path",
                                  "donor_path", "reference_image_path"))


def class_name(row):
    return str(pick(row, "class", "class_name", "category") or "unknown").lower()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def required(row, label, *names):
    value = pick(row, *names)
    if value is None:
        raise ValueError(
            f"Missing {label} for bank/class/slot="
            f"{pick(row, 'bank')}/{class_name(row)}/{row.get('slot_id')}"
        )
    return value


def geometry_value(row, *names):
    for scope in (row, row.get("frozen_slot", {}), row.get("runtime_slot", {})):
        geometry = scope.get("geometry") or {}
        for name in names:
            if geometry.get(name) is not None:
                return geometry[name]
    return None


def identity(row):
    """Fields that must be invariant when only the trained variant changes."""
    bbox = pick(row, "bbox", "bbox_xywh", "yolo_bbox")
    if bbox is None:
        bbox = geometry_value(row, "final_bbox_normalized_xywh",
                              "generator_bbox_xyxy", "original_bbox_xyxy")
    if bbox is None:
        raise ValueError(f"Missing BBox for {row.get('slot_id')}")
    mask_sha = pick(row, "mask_sha256", "decoded_mask_sha256")
    if mask_sha is None:
        mask_sha = geometry_value(row, "mask_decoded_sha256")
    if mask_sha is None:
        raise ValueError(f"Missing Mask identity for {row.get('slot_id')}")
    return {
        "bank": str(required(row, "bank", "bank")),
        "class": class_name(row),
        "slot_id": str(required(row, "slot_id", "slot_id")),
        "conditioning_normal_id": str(required(
            row, "conditioning normal", "conditioning_normal_id", "normal_id")),
        "normal_sha256": str(required(row, "normal SHA256", "normal_sha256")),
        "reference_id": str(required(
            row, "reference defect identity", "annotation_id", "reference_id",
            "donor_id", "canonical_image_id")),
        "reference_sha256": str(required(
            row, "reference defect SHA256", "source_sha256", "reference_sha256",
            "donor_sha256")),
        "mask_sha256": str(mask_sha),
        "bbox": bbox,
        "sample_seed": int(required(row, "generation seed", "sample_seed",
                                    "generation_seed", "seed")),
        "bank_seed": int(required(row, "bank seed", "bank_seed")),
        "prompt": str(required(row, "prompt", "prompt")),
        "generation_protocol_sha256": str(required(
            row, "generation protocol SHA256", "generation_protocol_sha256")),
        "inference_parameters": required(row, "inference parameters", "inference_parameters"),
        "injection_gamma": float(required(row, "injection gamma", "injection_gamma")),
        "quality_selection": bool(required(row, "quality selection flag", "quality_selection")),
        "reroll": bool(required(row, "reroll flag", "reroll")),
    }


def index_manifest(path):
    result = {}
    for row in records(path):
        key = (str(required(row, "bank", "bank")), class_name(row),
               str(required(row, "slot_id", "slot_id")))
        if key in result:
            raise ValueError(f"Duplicate bank/class/slot_id in {path}: {key}")
        result[key] = row
    return result


def verify_file(manifest, row, path_names, sha_names, label):
    value = pick(row, *path_names)
    expected = pick(row, *sha_names)
    if value is None or expected is None:
        raise ValueError(f"Missing {label} path/SHA256 for {row.get('slot_id')}")
    path = resolve(manifest, value)
    actual = sha(path)
    if actual != str(expected):
        raise ValueError(
            f"{label} SHA256 mismatch for {row.get('slot_id')}: "
            f"manifest={expected} actual={actual}"
        )
    return {"path": str(path), "sha256": actual}


def verify_row_artifacts(manifest, row):
    """Verify source bytes and actual inference inputs, not only manifest equality."""
    return {
        "generated": verify_file(
            manifest, row, ("image_path", "output_path", "generated_path"),
            ("image_sha256", "image_SHA256"), "generated PNG"),
        "normal_source": verify_file(
            manifest, row, ("normal_path", "conditioning_normal_path"),
            ("normal_sha256",), "normal source"),
        "reference_source": verify_file(
            manifest, row, ("reference_path", "source_path"),
            ("reference_sha256", "source_sha256"), "reference source"),
        "normal_input": verify_file(
            manifest, row, ("normal_input_path",), ("normal_input_sha256",),
            "normal crop input"),
        "reference_input": verify_file(
            manifest, row, ("reference_input_path",), ("reference_input_sha256",),
            "reference crop input"),
        "mask_input": verify_file(
            manifest, row, ("mask_input_path",), ("mask_input_sha256",),
            "mask input"),
    }


def audit_pairing(manifests, indexed):
    methods = list(indexed)
    expected = set(indexed[methods[0]])
    mismatches = []
    for method in methods[1:]:
        if set(indexed[method]) != expected:
            mismatches.append({
                "method": method,
                "missing_keys": sorted(expected - set(indexed[method])),
                "extra_keys": sorted(set(indexed[method]) - expected),
            })
    artifact_checks = {}
    if not mismatches:
        for key in sorted(expected):
            values = {method: identity(indexed[method][key]) for method in methods}
            baseline = canonical(values[methods[0]])
            if any(canonical(values[method]) != baseline for method in methods[1:]):
                mismatches.append({"key": key, "identities": values})
                continue
            artifact_checks["/".join(key)] = {
                method: verify_row_artifacts(manifests[method], indexed[method][key])
                for method in methods
            }
            for role in ("normal_source", "reference_source", "normal_input",
                         "reference_input", "mask_input"):
                hashes = {artifact_checks["/".join(key)][method][role]["sha256"]
                          for method in methods}
                if len(hashes) != 1:
                    mismatches.append({"key": key, "artifact_role": role,
                                       "hashes": sorted(hashes)})
    audit = {
        "status": "PASS" if not mismatches else "FAIL",
        "key": ["bank", "class", "slot_id"],
        "count_per_method": {method: len(rows) for method, rows in indexed.items()},
        "verified_equal_fields": [
            "conditioning_normal_id", "normal_sha256", "reference_id",
            "reference_sha256", "mask_sha256", "bbox", "sample_seed", "bank_seed",
            "prompt", "generation_protocol_sha256", "inference_parameters",
            "injection_gamma", "quality_selection", "reroll",
        ],
        "actual_file_sha256_verified": [
            "generated PNG", "normal source", "reference source", "normal crop input",
            "reference crop input", "mask input",
        ],
        "artifact_checks": artifact_checks,
        "source_manifests": {method: {"path": str(path), "sha256": sha(path)}
                             for method, path in manifests.items()},
        "mismatches": mismatches,
    }
    if mismatches:
        raise ValueError("A0/A1/A2 pairing audit failed: " + canonical(mismatches[:3]))
    return audit


def roi(row, width, height):
    value = pick(row, "roi_xyxy", "bbox_xyxy")
    if value is not None:
        x0, y0, x1, y1 = map(float, value)
    else:
        value = pick(row, "bbox", "bbox_xywh", "yolo_bbox")
        if value is None:
            geometry = pick(row, "geometry") or {}
            value = geometry.get("bbox_xyxy") or geometry.get("roi_xyxy")
            if value is None:
                raise ValueError(f"No ROI/bbox for slot {row.get('slot_id')}")
            x0, y0, x1, y1 = map(float, value)
        else:
            cx, cy, bw, bh = map(float, value)
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
                cx, bw = cx * width, bw * width
                cy, bh = cy * height, bh * height
            x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    x0 = max(0, min(width - 1, math.floor(x0)))
    y0 = max(0, min(height - 1, math.floor(y0)))
    x1 = max(x0 + 1, min(width, math.ceil(x1)))
    y1 = max(y0 + 1, min(height, math.ceil(y1)))
    return [x0, y0, x1, y1]


def image(path):
    with Image.open(path) as value:
        return value.convert("RGB")


def display_image(value, max_width):
    if value.width <= max_width:
        return value
    height = max(1, round(value.height * max_width / value.width))
    return value.resize((max_width, height), Image.Resampling.BILINEAR)


def select(common, per_class):
    chosen = []
    for label in sorted({item[1] for item in common}):
        keys = sorted(key for key in common if key[1] == label)
        if len(keys) < per_class:
            raise ValueError(f"Need {per_class} fixed samples for {label}; found {len(keys)}")
        indices = np.linspace(0, len(keys) - 1, per_class).round().astype(int)
        chosen.extend(keys[index] for index in indices)
    return chosen


def context_box(box, width, height, scale):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0, 16) * float(scale)
    nx0 = max(0, int(math.floor(cx - size / 2)))
    ny0 = max(0, int(math.floor(cy - size / 2)))
    nx1 = min(width, int(math.ceil(cx + size / 2)))
    ny1 = min(height, int(math.ceil(cy + size / 2)))
    return [nx0, ny0, max(nx0 + 1, nx1), max(ny0 + 1, ny1)]


def save_figure(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--no-latent-manifest", type=Path, required=True)
    parser.add_argument("--no-morph-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots-per-class", type=int, default=4)
    parser.add_argument("--zoom", type=int, default=4)
    parser.add_argument("--context-scale", type=float, default=4.0)
    parser.add_argument("--display-max-width", type=int, default=1024,
                        help="Consistent figure-only resize; source PNGs remain untouched.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate frozen pairing and write pairing_audit.json without rendering.")
    args = parser.parse_args()
    manifests = {"A0 Full": args.full_manifest.resolve(),
                 "A1 w/o Latent": args.no_latent_manifest.resolve(),
                 "A2 w/o Morph": args.no_morph_manifest.resolve()}
    indexed = {}
    for method, path in manifests.items():
        indexed[method] = index_manifest(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairing_audit = audit_pairing(manifests, indexed)
    (args.output_dir / "pairing_audit.json").write_text(
        json.dumps(pairing_audit, indent=2), encoding="utf-8")
    if args.audit_only:
        print(json.dumps({"status": "PASS", "pairing_count": len(indexed["A0 Full"]),
                          "output": str(args.output_dir / "pairing_audit.json")}))
        return
    common = set(indexed["A0 Full"])
    selected = select(common, args.slots_per_class)
    columns = ["Normal", "Reference", *indexed]
    prepared, ledger, heat_values = [], [], []
    for key in selected:
        base = indexed["A0 Full"][key]
        npath = normal_path(manifests["A0 Full"], base)
        rpath = reference_path(manifests["A0 Full"], base)
        normal = image(npath); reference = image(rpath)
        box = roi(base, *normal.size)
        generated = {}
        for method in indexed:
            row = indexed[method][key]
            path = generated_path(manifests[method], row)
            value = image(path)
            if value.size != normal.size:
                raise ValueError(f"Canvas mismatch for {key}: {method}")
            generated[method] = value
            heat_values.append(np.abs(np.asarray(value, dtype=np.float32) / 255.0
                                      - np.asarray(normal, dtype=np.float32) / 255.0).mean(2))
        prepared.append((key, normal, reference, generated, box))
        ledger.append({"bank": key[0], "class": key[1], "slot_id": key[2],
                       "identity": identity(base), "roi_xyxy": box,
                       "normal": {"path": str(npath), "sha256": sha(npath)},
                       "reference": {"path": str(rpath), "sha256": sha(rpath)},
                       "generated": {method: {"path": str(generated_path(manifests[method], indexed[method][key])),
                                                "sha256": sha(generated_path(manifests[method], indexed[method][key]))}
                                     for method in indexed}})
    vmax = float(max(value.max() for value in heat_values)) or 1.0
    fig, axes = plt.subplots(len(prepared), len(columns), figsize=(3 * len(columns), 3 * len(prepared)), squeeze=False)
    for row_index, (key, normal, reference, generated, box) in enumerate(prepared):
        values = [normal, reference, *generated.values()]
        for col_index, (title, value) in enumerate(zip(columns, values)):
            marked = value.copy(); ImageDraw.Draw(marked).rectangle(box, outline=(255, 40, 40), width=3)
            axes[row_index, col_index].imshow(display_image(marked, args.display_max_width)); axes[row_index, col_index].axis("off")
            if row_index == 0: axes[row_index, col_index].set_title(title)
            if col_index == 0: axes[row_index, col_index].set_ylabel(f"{key[0]}/{key[1]}\n{key[2]}")
    save_figure(fig, args.output_dir, "full_comparison")
    fig, axes = plt.subplots(len(prepared), len(indexed), figsize=(3 * len(indexed), 3 * len(prepared)), squeeze=False)
    for i, (key, _normal, _reference, generated, box) in enumerate(prepared):
        crop_box = context_box(box, generated[next(iter(generated))].width,
                               generated[next(iter(generated))].height,
                               args.context_scale)
        for j, (method, value) in enumerate(generated.items()):
            crop = value.crop(crop_box)
            relative_roi = [box[0] - crop_box[0], box[1] - crop_box[1],
                            box[2] - crop_box[0], box[3] - crop_box[1]]
            ImageDraw.Draw(crop).rectangle(relative_roi, outline=(255, 40, 40), width=2)
            crop = crop.resize((crop.width * args.zoom, crop.height * args.zoom),
                               Image.Resampling.NEAREST)
            axes[i, j].imshow(crop); axes[i, j].axis("off")
            if i == 0: axes[i, j].set_title(method)
            if j == 0: axes[i, j].set_ylabel(f"{key[0]}/{key[1]}\n{key[2]}")
    save_figure(fig, args.output_dir, "roi_zoom")
    fig, axes = plt.subplots(len(prepared), len(indexed), figsize=(3 * len(indexed), 3 * len(prepared)), squeeze=False)
    heat_artist = None
    for i, (key, normal, _reference, generated, _box) in enumerate(prepared):
        n = np.asarray(normal, dtype=np.float32) / 255.0
        for j, (method, value) in enumerate(generated.items()):
            heat = np.abs(np.asarray(value, dtype=np.float32) / 255.0 - n).mean(2)
            if heat.shape[1] > args.display_max_width:
                scale_h = max(1, round(heat.shape[0] * args.display_max_width / heat.shape[1]))
                heat = np.asarray(Image.fromarray(heat, mode="F").resize(
                    (args.display_max_width, scale_h), Image.Resampling.BILINEAR))
            heat_artist = axes[i, j].imshow(heat, cmap="inferno", vmin=0, vmax=vmax)
            axes[i, j].axis("off")
            if i == 0: axes[i, j].set_title(method)
            if j == 0: axes[i, j].set_ylabel(f"{key[0]}/{key[1]}\n{key[2]}")
    if heat_artist is not None:
        fig.colorbar(heat_artist, ax=axes.ravel().tolist(), shrink=0.8,
                     label="mean RGB absolute change [0, 1]")
    save_figure(fig, args.output_dir, "diff_heatmap")
    figure_manifest = {"status": "COMPLETE", "selection_rule":
        "sorted common class/slot IDs; evenly spaced fixed indices per class; no image-quality filtering",
        "slots_per_class": args.slots_per_class, "selected": ledger,
        "columns": columns, "reference_is_pixel_ground_truth": False,
        "zoom": args.zoom, "zoom_interpolation": "nearest",
        "context_scale": args.context_scale,
        "figure_display_max_width": args.display_max_width,
        "figure_resize_is_uniform_across_methods": True, "heatmap":
        {"definition": "mean_rgb(abs(generated-normal)) on [0,1]", "vmin": 0, "global_vmax": vmax,
         "independent_per_image_scaling": False},
        "pairing_audit": "pairing_audit.json",
        "pairing_status": pairing_audit["status"],
        "source_manifests": pairing_audit["source_manifests"]}
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "selected": len(selected), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
