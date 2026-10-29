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
    for label in sorted({item[0] for item in common}):
        keys = sorted(key for key in common if key[0] == label)
        if len(keys) <= per_class:
            chosen.extend(keys)
        else:
            indices = np.linspace(0, len(keys) - 1, per_class).round().astype(int)
            chosen.extend(keys[index] for index in indices)
    return chosen


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
    parser.add_argument("--display-max-width", type=int, default=1024,
                        help="Consistent figure-only resize; source PNGs remain untouched.")
    args = parser.parse_args()
    manifests = {"A0 Full": args.full_manifest.resolve(),
                 "A1 w/o Latent": args.no_latent_manifest.resolve(),
                 "A2 w/o Morph": args.no_morph_manifest.resolve()}
    indexed = {}
    for method, path in manifests.items():
        indexed[method] = {(class_name(row), str(row["slot_id"])): row
                           for row in records(path)}
    common = set.intersection(*(set(value) for value in indexed.values()))
    if not common:
        raise ValueError("No class/slot_id intersection across the three manifests")
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
        ledger.append({"class": key[0], "slot_id": key[1], "roi_xyxy": box,
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
            if col_index == 0: axes[row_index, col_index].set_ylabel(f"{key[0]}\n{key[1]}")
    save_figure(fig, args.output_dir, "full_comparison")
    fig, axes = plt.subplots(len(prepared), len(indexed), figsize=(3 * len(indexed), 3 * len(prepared)), squeeze=False)
    for i, (key, _normal, _reference, generated, box) in enumerate(prepared):
        for j, (method, value) in enumerate(generated.items()):
            crop = value.crop(box).resize((max(1, (box[2]-box[0]) * args.zoom),
                                           max(1, (box[3]-box[1]) * args.zoom)), Image.Resampling.NEAREST)
            axes[i, j].imshow(crop); axes[i, j].axis("off")
            if i == 0: axes[i, j].set_title(method)
            if j == 0: axes[i, j].set_ylabel(f"{key[0]}\n{key[1]}")
    save_figure(fig, args.output_dir, "roi_zoom")
    fig, axes = plt.subplots(len(prepared), len(indexed), figsize=(3 * len(indexed), 3 * len(prepared)), squeeze=False)
    for i, (key, normal, _reference, generated, _box) in enumerate(prepared):
        n = np.asarray(normal, dtype=np.float32) / 255.0
        for j, (method, value) in enumerate(generated.items()):
            heat = np.abs(np.asarray(value, dtype=np.float32) / 255.0 - n).mean(2)
            if heat.shape[1] > args.display_max_width:
                scale_h = max(1, round(heat.shape[0] * args.display_max_width / heat.shape[1]))
                heat = np.asarray(Image.fromarray(heat, mode="F").resize(
                    (args.display_max_width, scale_h), Image.Resampling.BILINEAR))
            axes[i, j].imshow(heat, cmap="inferno", vmin=0, vmax=vmax); axes[i, j].axis("off")
            if i == 0: axes[i, j].set_title(method)
            if j == 0: axes[i, j].set_ylabel(f"{key[0]}\n{key[1]}")
    save_figure(fig, args.output_dir, "diff_heatmap")
    figure_manifest = {"status": "COMPLETE", "selection_rule":
        "sorted common class/slot IDs; evenly spaced fixed indices per class; no image-quality filtering",
        "slots_per_class": args.slots_per_class, "selected": ledger,
        "columns": columns, "reference_is_pixel_ground_truth": False,
        "zoom": args.zoom, "zoom_interpolation": "nearest",
        "figure_display_max_width": args.display_max_width,
        "figure_resize_is_uniform_across_methods": True, "heatmap":
        {"definition": "mean_rgb(abs(generated-normal)) on [0,1]", "vmin": 0, "global_vmax": vmax,
         "independent_per_image_scaling": False},
        "source_manifests": {method: {"path": str(path), "sha256": sha(path)}
                             for method, path in manifests.items()}}
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "selected": len(selected), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
