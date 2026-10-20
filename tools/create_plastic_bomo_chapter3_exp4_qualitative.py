#!/usr/bin/env python3
"""Create Exp4 figures from frozen images only.

`prepare` deliberately has no imaging-library import.  It validates metadata and
file hashes, then freezes the matched population and the seed-2026 sample list.
`render` may open pixels only after those frozen files pass their SHA bindings.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/plastic_bomo_chapter3_exp4_qualitative.json"
METHOD_KEYS = ("raw", "msdf", "anomalydiffusion")
METHOD_LABELS = ("Raw SD2", "MSDF", "AnomalyDiffusion")
ZERO_ACTIONS = {
    "training_count": 0,
    "generation_count": 0,
    "detector_training_count": 0,
    "quality_selection_count": 0,
    "manual_selection_count": 0,
    "source_image_modification_count": 0,
    "new_metric_count": 0,
}


class Exp4Error(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {detail}")


def require(value, status, detail):
    if not value:
        raise Exp4Error(status, detail)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=False,
                         indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def portable(path):
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def checked_file(path, expected, status="FROZEN_INPUT_CHANGED"):
    path = Path(path)
    require(path.is_file(), "INPUT_ASSET_MISSING", str(path))
    actual = sha256(path)
    require(actual == expected, status,
            {"path": portable(path), "expected": expected, "actual": actual})
    return path


def crop_box(bbox, width, height, margin=0.25):
    require(len(bbox) == 4 and all(math.isfinite(x) for x in bbox),
            "BBOX_INVALID", bbox)
    cx, cy, bw, bh = bbox
    require(width > 0 and height > 0 and 0 <= cx <= 1 and 0 <= cy <= 1
            and 0 < bw <= 1 and 0 < bh <= 1 and margin == 0.25,
            "BBOX_INVALID", bbox)
    cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
    return [max(0, math.floor(cx - 0.75 * bw)),
            max(0, math.floor(cy - 0.75 * bh)),
            min(width, math.ceil(cx + 0.75 * bw)),
            min(height, math.ceil(cy + 0.75 * bh))]


def config_bindings(cfg):
    result = {}
    source_paths = {
        "exp1_visual_manifest": resolve(cfg["exp1_visual_manifest"]),
        "real_reference_manifest": resolve(cfg["real_reference_manifest"]),
    }
    for method, root_name in cfg["method_roots"].items():
        root = resolve(root_name)
        source_paths[f"{method}/formal_synthetic_manifest.json"] = root / "formal_synthetic_manifest.json"
        for bank in cfg["banks"]:
            key = f"{method}/formal_bank_{bank}_manifest.json"
            source_paths[key] = root / f"formal_bank_{bank}_manifest.json"
    require(set(source_paths) == set(cfg["input_manifest_sha256"]),
            "PROTOCOL_INVALID", "input manifest binding keys differ")
    for key, path in source_paths.items():
        checked_file(path, cfg["input_manifest_sha256"][key])
        result[key] = {"path": portable(path), "sha256": cfg["input_manifest_sha256"][key]}
    return result


def method_records(cfg, method):
    root = resolve(cfg["method_roots"][method])
    global_manifest = load(root / "formal_synthetic_manifest.json")
    require(global_manifest.get("status") == "FROZEN"
            and global_manifest.get("count") == 240
            and len(global_manifest.get("records", [])) == 240,
            "FROZEN_GENERATION_INCOMPLETE", method)
    quality = global_manifest.get("quality_selection",
                                  bool(global_manifest.get("selector_count", 0)))
    require(quality is False and global_manifest.get("reroll_count", 0) == 0,
            "QUALITY_SELECTION_DETECTED", method)
    global_map = {(row["bank"], row["slot_id"]): row
                  for row in global_manifest["records"]}
    require(len(global_map) == 240, "MATCH_KEY_DUPLICATE", method)
    bank_rows = []
    for bank in cfg["banks"]:
        manifest = load(root / f"formal_bank_{bank}_manifest.json")
        rows = manifest.get("records", [])
        require(manifest.get("status") == "COMPLETE"
                and manifest.get("count") == 80 and len(rows) == 80
                and manifest.get("quality_selection") is False
                and manifest.get("reroll_count") == 0,
                "BANK_MANIFEST_INVALID", f"{method}:{bank}")
        require(all(row["bank"] == bank for row in rows),
                "BANK_MANIFEST_INVALID", f"{method}:{bank}:row bank")
        require(Counter(row["class"] for row in rows) == {"flash": 40, "black": 40},
                "BANK_MANIFEST_INVALID", f"{method}:{bank}:class counts")
        for row in rows:
            key = (bank, row["slot_id"])
            require(key in global_map and row == global_map[key],
                    "GLOBAL_BANK_MANIFEST_MISMATCH", f"{method}:{bank}:{row['slot_id']}")
        bank_rows.extend(rows)
    require({(r["bank"], r["slot_id"]) for r in bank_rows} == set(global_map),
            "GLOBAL_BANK_MANIFEST_MISMATCH", method)
    return root, global_map


def build_matched(cfg):
    exp1 = load(resolve(cfg["exp1_visual_manifest"]))
    require(exp1.get("status") == "FROZEN" and exp1.get("count") == 240
            and exp1.get("no_filtering") is True and exp1.get("no_reroll") is True,
            "EXP1_MANIFEST_INVALID", "visual_comparison_manifest.json")
    exp1_map = {(r["bank"], r["slot_id"]): r for r in exp1["records"]}
    require(len(exp1_map) == 240, "MATCH_KEY_DUPLICATE", "Experiment 1")
    methods = {}
    for method in METHOD_KEYS:
        methods[method] = method_records(cfg, method)
    keysets = [set(exp1_map)] + [set(methods[m][1]) for m in METHOD_KEYS]
    require(all(keys == keysets[0] for keys in keysets[1:]),
            "THREE_METHOD_MATCH_INCOMPLETE", [len(keys) for keys in keysets])

    records = []
    source_images_checked = 0
    normal_paths_checked = set()
    for key in sorted(exp1_map):
        bank, slot_id = key
        exp = exp1_map[key]
        rows = {m: methods[m][1][key] for m in METHOD_KEYS}
        frozen = rows["raw"]["frozen_slot"]
        frozen_sha = canonical_sha(frozen)
        require(all(row["frozen_slot"] == frozen for row in rows.values()),
                "FROZEN_SLOT_MISMATCH", f"{bank}:{slot_id}")
        require(frozen_sha == exp["slot_metadata_sha256"]
                and frozen_sha == rows["anomalydiffusion"].get("frozen_slot_sha256"),
                "FROZEN_SLOT_SHA_MISMATCH", f"{bank}:{slot_id}")
        klass = frozen["class"]
        bbox = frozen["bbox"]
        width, height = frozen["geometry"]["final_canvas_wh"]
        roi = crop_box(bbox, width, height, cfg["roi_context_margin_each_side"])
        require(bank in cfg["banks"] and klass in cfg["classes"]
                and all(row["bank"] == bank and row["slot_id"] == slot_id
                        and row["class"] == klass for row in rows.values())
                and exp["class"] == klass and exp["bbox"] == bbox
                and exp["canvas_wh"] == [width, height] and exp["roi_xyxy"] == roi
                and exp["conditioning_normal_id"] == frozen["conditioning_normal_id"]
                and exp["conditioning_normal_sha256"] == frozen["normal_sha256"],
                "MATCH_CONDITION_MISMATCH", f"{bank}:{slot_id}")

        paths = {}
        for method in METHOD_KEYS:
            row = rows[method]
            image = checked_file(methods[method][0] / row["image_path"], row["image_sha256"])
            paths[method] = image
            source_images_checked += 1
        require(portable(paths["raw"]) == exp["raw_image_path"]
                and portable(paths["msdf"]) == exp["msdf_image_path"]
                and rows["raw"]["image_sha256"] == exp["raw_image_sha256"]
                and rows["msdf"]["image_sha256"] == exp["msdf_image_sha256"],
                "EXP1_IMAGE_BINDING_MISMATCH", f"{bank}:{slot_id}")
        normal_path = resolve(frozen["normal_path"])
        normal_key = (str(normal_path), frozen["normal_sha256"])
        if normal_key not in normal_paths_checked:
            checked_file(normal_path, frozen["normal_sha256"])
            normal_paths_checked.add(normal_key)
        records.append({
            "bank": bank, "slot_id": slot_id, "class": klass,
            "bbox": bbox, "bbox_format": "normalized_xywh",
            "canvas_wh": [width, height], "roi_xyxy": roi,
            "conditioning_normal_id": frozen["conditioning_normal_id"],
            "conditioning_normal_path": portable(normal_path),
            "conditioning_normal_sha256": frozen["normal_sha256"],
            "frozen_slot_sha256": frozen_sha,
            "raw_image_path": portable(paths["raw"]),
            "raw_image_sha256": rows["raw"]["image_sha256"],
            "msdf_image_path": portable(paths["msdf"]),
            "msdf_image_sha256": rows["msdf"]["image_sha256"],
            "anomalydiffusion_image_path": portable(paths["anomalydiffusion"]),
            "anomalydiffusion_image_sha256": rows["anomalydiffusion"]["image_sha256"],
        })
    require(Counter(r["class"] for r in records) == {"flash": 120, "black": 120}
            and Counter(r["bank"] for r in records) == {"A": 80, "B": 80, "C": 80},
            "MATCH_POPULATION_INVALID", "expected class and bank counts")
    return records, source_images_checked, len(normal_paths_checked)


def freeze_sample(cfg, records, real_reference):
    rng = random.Random(cfg["seed"])
    by_group = {}
    for klass in cfg["classes"]:
        for bank in cfg["banks"]:
            by_group[(klass, bank)] = sorted(
                (r for r in records if r["class"] == klass and r["bank"] == bank),
                key=lambda r: r["slot_id"])
    selected = []
    used_by_class = {klass: set() for klass in cfg["classes"]}
    for klass in cfg["classes"]:
        for bank in cfg["banks"]:
            candidates = [r for r in by_group[(klass, bank)]
                          if r["slot_id"] not in used_by_class[klass]]
            rng.shuffle(candidates)
            chosen = candidates[:cfg["samples_per_class_bank"]]
            require(len(chosen) == cfg["samples_per_class_bank"],
                    "SAMPLE_POPULATION_INSUFFICIENT", f"{klass}:{bank}")
            for index, row in enumerate(chosen, 1):
                used_by_class[klass].add(row["slot_id"])
                selected.append({
                    "class": klass, "bank": bank, "sample_index": index,
                    "slot_id": row["slot_id"],
                    "frozen_slot_sha256": row["frozen_slot_sha256"],
                    "output_name": f"{klass}_{bank}_{index:02d}.png",
                    "paper_selection": index == 1,
                })
    require(len(selected) == 24
            and all(len(used_by_class[c]) == 12 for c in cfg["classes"]),
            "SAMPLE_COUNT_INVALID", "expected 24 groups and no repeated slot per class")

    real = load(real_reference)
    instances = real.get("instances", [])
    real_selected = []
    for klass in cfg["classes"]:
        ranked = sorted((r for r in instances if r["class"] == klass),
            key=lambda r: hashlib.sha256(
                f"{cfg['seed']}|real-reference|{klass}|{r['annotation_id']}".encode()).hexdigest())
        used_nodes = set()
        for row in ranked:
            if row["node_id"] in used_nodes:
                continue
            real_selected.append({
                "class": klass, "node_id": row["node_id"],
                "annotation_id": row["annotation_id"], "image_path": row["image_path"],
                "image_sha256": row["image_sha256"], "canvas_wh": row["canvas_wh"],
                "roi_xyxy": row["roi_xyxy"],
                "selection_rank_sha256": hashlib.sha256(
                    f"{cfg['seed']}|real-reference|{klass}|{row['annotation_id']}".encode()).hexdigest(),
            })
            used_nodes.add(row["node_id"])
            if len([x for x in real_selected if x["class"] == klass]) == 3:
                break
        require(len([x for x in real_selected if x["class"] == klass]) == 3,
                "REAL_REFERENCE_INSUFFICIENT", klass)
    return selected, real_selected


def prepare(config_path):
    config_path = Path(config_path).resolve()
    cfg = load(config_path)
    output = resolve(cfg["output"])
    require(not output.exists() or not any(output.iterdir()),
            "OUTPUT_NOT_EMPTY", str(output))
    print("[Exp4] 验证冻结清单、bank 配对和 720 张源图 SHA256；不读取像素。", flush=True)
    bindings = config_bindings(cfg)
    records, source_checked, normal_checked = build_matched(cfg)
    real_reference = resolve(cfg["real_reference_manifest"])
    selected, real_selected = freeze_sample(cfg, records, real_reference)

    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "FROZEN_BEFORE_IMAGE_VIEW",
        "experiment": cfg["experiment"],
        "methods": cfg["methods"], "classes": cfg["classes"], "banks": cfg["banks"],
        "matching_key": ["bank", "slot_id"],
        "required_equal_fields": ["class", "conditioning_normal_id", "bbox", "frozen_slot", "bank"],
        "sampling_seed": cfg["seed"], "samples_per_class_bank": cfg["samples_per_class_bank"],
        "sampling_fields_used": ["bank", "slot_id", "class", "frozen_slot_sha256"],
        "sampling_fields_prohibited": ["FID", "KID", "LPIPS", "image pixels", "quality scores"],
        "sampling_algorithm": "Python random.Random(2026); class then bank order; sort by slot_id; exclude previously selected slot_ids within class; shuffle; take first four",
        "paper_selection_rule": "sample_index 01 from each class/bank, frozen before pixels were opened",
        "real_reference_selection_rule": "three distinct node_ids per class ranked by SHA256('2026|real-reference|class|annotation_id')",
        "roi_rule": "Experiment 1 normalized xywh bbox expanded by 25% on every side, floor lower bounds, ceil upper bounds, clipped to canvas",
        "roi_display_wh": cfg["roi_display_wh"], "full_display_wh": cfg["full_display_wh"],
        "display_resampling": cfg["display_resampling"], "png_dpi": cfg["png_dpi"],
        "residual_rule": "per-channel absolute uint8 pixel difference from conditioning normal; fixed 0..255; no per-image normalization",
        "image_content_read_count_before_sampling_freeze": 0,
        "quality_metric_read_count_for_sampling": 0,
        "manual_selection_count": 0,
        "source_manifest_bindings": bindings,
        **ZERO_ACTIONS,
    }
    matched = {
        "status": "FROZEN", "match_count": len(records),
        "class_counts": dict(Counter(r["class"] for r in records)),
        "bank_counts": dict(Counter(r["bank"] for r in records)),
        "source_image_sha256_checks": source_checked,
        "unique_conditioning_normal_sha256_checks": normal_checked,
        "records": records,
    }
    sampling = {
        "status": "FROZEN_BEFORE_IMAGE_VIEW", "seed": cfg["seed"],
        "selection_count": len(selected),
        "class_counts": dict(Counter(r["class"] for r in selected)),
        "class_bank_counts": {f"{c}_{b}": sum(r["class"] == c and r["bank"] == b for r in selected)
                              for c in cfg["classes"] for b in cfg["banks"]},
        "unique_slot_ids_by_class": {c: len({r["slot_id"] for r in selected if r["class"] == c})
                                     for c in cfg["classes"]},
        "paper_selection_count": sum(r["paper_selection"] for r in selected),
        "records": selected, "real_reference_records": real_selected,
        "image_content_read_count_before_freeze": 0,
        "quality_metric_read_count": 0, "manual_replacement_count": 0,
    }
    save(output / "qualitative_protocol.json", protocol)
    save(output / "three_method_matched_manifest.json", matched)
    save(output / "qualitative_sampling_manifest.json", sampling)
    binding = {
        "status": "FROZEN_BEFORE_IMAGE_VIEW",
        "sha256": {name: sha256(output / name) for name in (
            "qualitative_protocol.json", "three_method_matched_manifest.json",
            "qualitative_sampling_manifest.json")},
        "implementation_sha256": {
            portable(config_path): sha256(config_path), portable(Path(__file__)): sha256(Path(__file__))},
        "pixel_files_opened_before_freeze": 0,
    }
    save(output / "pre_render_binding.json", binding)
    save(output / "exp4_status.json", {
        "status": "PREPARED", "sampling_frozen": True, "render_complete": False,
        "matched_count": 240, "selected_count": 24, **ZERO_ACTIONS})
    print(f"[Exp4] 抽样已先行冻结：24 组；绑定 {output / 'pre_render_binding.json'}", flush=True)


def verify_prepared(cfg, config_path, output):
    binding = load(output / "pre_render_binding.json")
    require(binding["status"] == "FROZEN_BEFORE_IMAGE_VIEW"
            and binding["pixel_files_opened_before_freeze"] == 0,
            "PRE_RENDER_BINDING_INVALID", "freeze status")
    for name, digest in binding["sha256"].items():
        checked_file(output / name, digest, "PRE_RENDER_ARTIFACT_CHANGED")
    expected_impl = {portable(Path(config_path)): sha256(config_path),
                     portable(Path(__file__)): sha256(Path(__file__))}
    require(binding["implementation_sha256"] == expected_impl,
            "IMPLEMENTATION_CHANGED_AFTER_FREEZE", expected_impl)
    config_bindings(cfg)


def font(size, bold=False):
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def rgb_open(path, expected_wh):
    from PIL import Image
    image = Image.open(path)
    image.load()
    require(image.size == tuple(expected_wh), "INVALID_INPUT",
            {"path": portable(path), "expected_wh": expected_wh, "actual_wh": image.size})
    return image.convert("RGB")


def paste_center(canvas, image, box, border=True):
    from PIL import ImageDraw
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    canvas.paste(image, (x, y))
    if border:
        ImageDraw.Draw(canvas).rectangle((x - 1, y - 1, x + image.width, y + image.height),
                                         outline=(75, 75, 75), width=2)


def load_group(row):
    wh = row["canvas_wh"]
    return [
        rgb_open(resolve(row["conditioning_normal_path"]), wh),
        rgb_open(resolve(row["raw_image_path"]), wh),
        rgb_open(resolve(row["msdf_image_path"]), wh),
        rgb_open(resolve(row["anomalydiffusion_image_path"]), wh),
    ]


def save_audit_figure(path, sample, row, images, cfg):
    from PIL import Image, ImageDraw
    resample = Image.Resampling.BILINEAR
    full_w, full_h = cfg["full_display_wh"]
    roi_w, roi_h = cfg["roi_display_wh"]
    cell_w, gap, left, right = full_w + 24, 18, 100, 30
    top, title_h, row_gap, bottom = 30, 96, 58, 42
    width = left + 4 * cell_w + 3 * gap + right
    y_full = top + title_h
    y_roi = y_full + full_h + row_gap
    height = y_roi + roi_h + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{sample['class'].title()}  |  Bank {sample['bank']}  |  Slot ID: {sample['slot_id']}"
    draw.text((left, 12), title, fill="black", font=font(32, True))
    columns = ("Conditioning Normal",) + METHOD_LABELS
    for col, label in enumerate(columns):
        x0 = left + col * (cell_w + gap)
        draw.text((x0 + cell_w / 2, top + 50), label, fill="black",
                  font=font(24, True), anchor="mm")
        full = images[col].resize((full_w, full_h), resample)
        roi = images[col].crop(tuple(row["roi_xyxy"])).resize((roi_w, roi_h), resample)
        paste_center(canvas, full, (x0, y_full, x0 + cell_w, y_full + full_h))
        paste_center(canvas, roi, (x0, y_roi, x0 + cell_w, y_roi + roi_h))
    draw.text((12, y_full + full_h / 2), "Full", fill="black", font=font(24, True), anchor="lm")
    draw.text((12, y_roi + roi_h / 2), "ROI", fill="black", font=font(24, True), anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG", dpi=(cfg["png_dpi"], cfg["png_dpi"]), optimize=False)


def save_residual_figure(path, sample, row, images, cfg):
    from PIL import Image, ImageChops, ImageDraw
    resample = Image.Resampling.BILINEAR
    residuals = [ImageChops.difference(images[i], images[0]) for i in range(1, 4)]
    full_w, full_h = cfg["full_display_wh"]
    roi_w, roi_h = cfg["roi_display_wh"]
    cell_w, gap, left = full_w + 24, 18, 100
    y_full, row_gap = 116, 58
    y_roi = y_full + full_h + row_gap
    canvas = Image.new("RGB", (left + 3 * cell_w + 2 * gap + 30, y_roi + roi_h + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 15),
              f"Absolute RGB residual | {sample['class'].title()} | Bank {sample['bank']} | Slot ID: {sample['slot_id']}",
              fill="black", font=font(30, True))
    for col, (label, residual) in enumerate(zip(METHOD_LABELS, residuals)):
        x0 = left + col * (cell_w + gap)
        draw.text((x0 + cell_w / 2, 82), label, fill="black", font=font(24, True), anchor="mm")
        full = residual.resize((full_w, full_h), resample)
        roi = residual.crop(tuple(row["roi_xyxy"])).resize((roi_w, roi_h), resample)
        paste_center(canvas, full, (x0, y_full, x0 + cell_w, y_full + full_h))
        paste_center(canvas, roi, (x0, y_roi, x0 + cell_w, y_roi + roi_h))
    draw.text((12, y_full + full_h / 2), "Full", fill="black", font=font(24, True), anchor="lm")
    draw.text((12, y_roi + roi_h / 2), "ROI", fill="black", font=font(24, True), anchor="lm")
    canvas.save(path, "PNG", dpi=(cfg["png_dpi"], cfg["png_dpi"]), optimize=False)


def save_paper_figure(path, klass, groups, cfg):
    from PIL import Image, ImageDraw
    resample = Image.Resampling.BILINEAR
    full_w, full_h = cfg["full_display_wh"]
    roi_w, roi_h = cfg["roi_display_wh"]
    cell_w, gap, left, top = full_w + 24, 18, 105, 110
    pair_gap, group_gap = 28, 58
    group_h = full_h + pair_gap + roi_h
    width = left + 3 * cell_w + 2 * gap + 30
    height = top + 3 * group_h + 2 * group_gap + 40
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 18), f"Qualitative comparison: {klass.title()}",
              fill="black", font=font(32, True))
    for col, label in enumerate(METHOD_LABELS):
        x0 = left + col * (cell_w + gap)
        draw.text((x0 + cell_w / 2, 78), label, fill="black", font=font(25, True), anchor="mm")
    for idx, (sample, row, images) in enumerate(groups):
        y_full = top + idx * (group_h + group_gap)
        y_roi = y_full + full_h + pair_gap
        draw.text((12, y_full + full_h / 2), f"Bank {sample['bank']}\nFull",
                  fill="black", font=font(23, True), anchor="lm", spacing=5)
        draw.text((35, y_roi + roi_h / 2), "ROI", fill="black", font=font(23, True), anchor="lm")
        for col, image in enumerate(images[1:]):
            x0 = left + col * (cell_w + gap)
            paste_center(canvas, image.resize((full_w, full_h), resample),
                         (x0, y_full, x0 + cell_w, y_full + full_h))
            paste_center(canvas, image.crop(tuple(row["roi_xyxy"])).resize((roi_w, roi_h), resample),
                         (x0, y_roi, x0 + cell_w, y_roi + roi_h))
    canvas.save(path, "PNG", dpi=(cfg["png_dpi"], cfg["png_dpi"]), optimize=False)


def save_real_reference(path, selections, cfg):
    from PIL import Image, ImageDraw
    resample = Image.Resampling.BILINEAR
    roi_w, roi_h = cfg["roi_display_wh"]
    cell_w, gap, left, top = roi_w + 40, 35, 45, 110
    width = left * 2 + 3 * cell_w + 2 * gap
    height = top + 2 * (roi_h + 80) + 35
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width / 2, 24), "Representative real examples (not paired ground truth)",
              fill="black", font=font(28, True), anchor="ma")
    for row_index, klass in enumerate(cfg["classes"]):
        rows = [r for r in selections if r["class"] == klass]
        y = top + row_index * (roi_h + 80)
        draw.text((12, y + roi_h / 2), klass.title(), fill="black", font=font(22, True), anchor="lm")
        for col, row in enumerate(rows):
            image = rgb_open(resolve(row["image_path"]), row["canvas_wh"])
            roi = image.crop(tuple(row["roi_xyxy"])).resize((roi_w, roi_h), resample)
            x = left + col * (cell_w + gap)
            paste_center(canvas, roi, (x, y, x + cell_w, y + roi_h))
            draw.text((x + cell_w / 2, y + roi_h + 22), f"Example {col + 1}",
                      fill="black", font=font(18), anchor="ma")
    canvas.save(path, "PNG", dpi=(cfg["png_dpi"], cfg["png_dpi"]), optimize=False)


def png_record(path):
    from PIL import Image
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (None, None))
        return {"path": portable(path), "sha256": sha256(path),
                "pixel_wh": list(image.size),
                "dpi": [round(float(x), 3) if x is not None else None for x in dpi]}


def write_report(output, audit):
    report = """# Chapter 3 Experiment 4 — Three-Method Qualitative Comparison

## 1. 实验目的

在完全冻结的 Raw SD2、MSDF 与 AnomalyDiffusion 输出上，制作可审计的三方法定性对比材料。本实验未训练、未重新生成、未调整模型或生成参数，也未计算新指标。

## 2. 数据来源

输入来自 Stage20B Raw SD2、Stage25 MSDF、Experiment 2 AnomalyDiffusion，以及 Experiment 1 已冻结的视觉匹配清单和真实参考清单。服务器端内部清单保存路径、样本身份和逐文件 SHA256；公开摘要不包含这些字段。

## 3. 三方法匹配规则

以 `bank + slot_id` 为唯一匹配键。每组要求 class、conditioning normal、bbox、frozen slot 和 bank 完全一致。三方法共匹配 240 组；所有 720 张生成图在抽样前均按原始清单核验 SHA256。

## 4. 抽样规则

固定种子为 2026。按 Flash/Black 和 Bank A/B/C 分层，每层抽取 4 组，共 24 组；每类 12 个 slot_id 均不重复。抽样只使用 bank、slot_id、class 和 frozen metadata。抽样名单在任何图像像素被打开前冻结。论文图固定使用每个 class/bank 的第 01 组，共 6 组。

## 5. ROI 裁剪协议

完全复用 Experiment 1 的 ROI 坐标：以 normalized xywh bbox 为基础，每侧扩展 25% context margin，左上界向下取整、右下界向上取整并裁剪到画布。所有方法与 conditioning normal 使用相同 ROI；展示缩放统一采用 bilinear 插值，ROI 为 256×256。

## 6. 论文图片组成

`qualitative_full_set/` 含 24 张四列两行审计图；`paper_figures/` 含 Flash 与 Black 各一张论文组合图；`qualitative_residuals/` 含 24 张固定 0–255 显示范围的绝对 RGB 残差图；另含一张明确标注为非配对真值的真实缺陷参考图。所有图片均保存为 300 DPI PNG。

## 7. 审计结果

审计状态：**PASS**。24 组完整存在，Flash=12、Black=12，每类各 Bank=4；匹配、ROI、冻结前抽样、源文件 SHA256 复核、输出文件数量和 PNG 元数据检查均通过。训练、生成、质量筛选、人工换样、源图修改及新指标计数均为 0。

## 8. 观察记录模板

下表留给人工查看后填写。这里没有使用 FID/KID/LPIPS 自动生成定性结论。

| 类别 | Bank | 缺陷清晰度 | 形态是否符合类别 | 背景融合 | 纹理/结构伪影 | bbox 一致性 | 三方法可见差异 |
|---|---|---|---|---|---|---|---|
| Flash | A |  |  |  |  |  |  |
| Flash | B |  |  |  |  |  |  |
| Flash | C |  |  |  |  |  |  |
| Black | A |  |  |  |  |  |  |
| Black | B |  |  |  |  |  |  |
| Black | C |  |  |  |  |  |  |

## 9. 结果解释边界

这些图用于观察固定样本上的可见现象，不构成统计显著性、总体优越性或因果结论。残差图是诊断材料，不是新评价指标。真实参考样本只展示类别外观，与 synthetic slot 不是一一配对的 ground truth。具体视觉结论应由研究者查看全部 24 组后记录。
"""
    (output / "EXP4_QUALITATIVE_REPORT.md").write_text(report, encoding="utf-8", newline="\n")


def render(config_path):
    config_path = Path(config_path).resolve()
    cfg = load(config_path)
    output = resolve(cfg["output"])
    require(output.is_dir(), "DATA_REQUIRED", "run prepare first")
    verify_prepared(cfg, config_path, output)
    matched = load(output / "three_method_matched_manifest.json")
    sampling = load(output / "qualitative_sampling_manifest.json")
    records = {(r["bank"], r["slot_id"]): r for r in matched["records"]}
    require(sampling["status"] == "FROZEN_BEFORE_IMAGE_VIEW"
            and sampling["image_content_read_count_before_freeze"] == 0,
            "SAMPLING_NOT_FROZEN", "sampling manifest")
    selected = sampling["records"]
    require(len(selected) == 24, "SAMPLE_COUNT_INVALID", len(selected))
    full_dir = output / "qualitative_full_set"
    residual_dir = output / "qualitative_residuals"
    paper_dir = output / "paper_figures"
    full_dir.mkdir(exist_ok=True); residual_dir.mkdir(exist_ok=True); paper_dir.mkdir(exist_ok=True)

    print("[Exp4] 抽样绑定验证通过，开始只读取已冻结的 24 组图像像素。", flush=True)
    cache = {}
    input_before = {}
    for i, sample in enumerate(selected, 1):
        key = (sample["bank"], sample["slot_id"])
        require(key in records and records[key]["frozen_slot_sha256"] == sample["frozen_slot_sha256"],
                "SAMPLED_RECORD_BINDING_INVALID", key)
        row = records[key]
        source_pairs = [(row["conditioning_normal_path"], row["conditioning_normal_sha256"])] + [
            (row[f"{m}_image_path"], row[f"{m}_image_sha256"]) for m in METHOD_KEYS]
        for name, digest in source_pairs:
            input_before[name] = digest
            checked_file(resolve(name), digest)
        images = load_group(row)
        cache[key] = images
        save_audit_figure(full_dir / sample["output_name"], sample, row, images, cfg)
        save_residual_figure(residual_dir / sample["output_name"], sample, row, images, cfg)
        print(f"[Exp4] {i:02d}/24 {sample['output_name']}", flush=True)

    for klass in cfg["classes"]:
        groups = []
        for sample in selected:
            if sample["class"] == klass and sample["paper_selection"]:
                key = (sample["bank"], sample["slot_id"])
                groups.append((sample, records[key], cache[key]))
        require([g[0]["bank"] for g in groups] == cfg["banks"],
                "PAPER_SELECTION_INVALID", klass)
        save_paper_figure(paper_dir / f"Figure_Qualitative_{klass.title()}.png", klass, groups, cfg)
    for row in sampling["real_reference_records"]:
        checked_file(resolve(row["image_path"]), row["image_sha256"])
        input_before[row["image_path"]] = row["image_sha256"]
    save_real_reference(output / "real_reference_examples.png",
                        sampling["real_reference_records"], cfg)

    # Recheck every source opened during rendering; outputs were written elsewhere.
    for name, digest in input_before.items():
        checked_file(resolve(name), digest, "SOURCE_IMAGE_CHANGED_DURING_RENDER")
    full_files = sorted(full_dir.glob("*.png"))
    residual_files = sorted(residual_dir.glob("*.png"))
    paper_files = sorted(paper_dir.glob("*.png"))
    require(len(full_files) == 24 and len(residual_files) == 24 and len(paper_files) == 2,
            "OUTPUT_COUNT_INVALID", [len(full_files), len(residual_files), len(paper_files)])
    expected_names = sorted(r["output_name"] for r in selected)
    require([p.name for p in full_files] == expected_names
            and [p.name for p in residual_files] == expected_names,
            "OUTPUT_NAME_INVALID", "full/residual names")
    output_pngs = [png_record(p) for p in full_files + residual_files + paper_files
                   + [output / "real_reference_examples.png"]]
    require(all(x["dpi"] and abs(x["dpi"][0] - 300) < 0.1
                and abs(x["dpi"][1] - 300) < 0.1 for x in output_pngs),
            "PNG_DPI_INVALID", "expected 300 DPI")

    audit = {
        "status": "PASS", "matched_population_count": 240,
        "selected_group_count": 24,
        "class_counts": dict(Counter(r["class"] for r in selected)),
        "class_bank_counts": {f"{c}_{b}": sum(r["class"] == c and r["bank"] == b for r in selected)
                              for c in cfg["classes"] for b in cfg["banks"]},
        "unique_slot_ids_by_class": {c: len({r["slot_id"] for r in selected if r["class"] == c})
                                     for c in cfg["classes"]},
        "three_method_match_checks": 24,
        "identical_roi_checks_per_group": 24,
        "all_720_generation_sha256_verified_before_sampling": True,
        "selected_source_files_reverified_after_render": len(input_before),
        "sampling_frozen_before_pixel_read": True,
        "sampling_used_quality_scores": False,
        "manual_sample_replacement": False,
        "source_pixels_modified": False,
        "residual_display_range": [0, 255],
        "per_image_residual_normalization": False,
        "full_set_png_count": 24, "residual_png_count": 24,
        "paper_figure_png_count": 2, "real_reference_png_count": 1,
        "output_png_records": output_pngs,
        **ZERO_ACTIONS,
    }
    save(output / "qualitative_integrity_audit.json", audit)
    write_report(output, audit)
    public_protocol = {
        "experiment": cfg["experiment"], "status": "COMPLETE",
        "methods": cfg["methods"], "sampling_seed": cfg["seed"],
        "sampling": "4 metadata-only random samples per class/bank; no repeated slot_id within class",
        "sampling_frozen_before_pixel_read": True,
        "paper_selection": "sample 01 in every class/bank, frozen before image view",
        "roi": "Experiment 1 bbox plus 25% context on every side; identical crop and bilinear resize for all methods",
        "residual": "absolute RGB difference, fixed 0..255, no per-image normalization",
        "publication_images_uploaded": False,
        "publication_permission_status": "not confirmed; figures retained on server",
        **ZERO_ACTIONS,
    }
    public_audit = {k: audit[k] for k in (
        "status", "matched_population_count", "selected_group_count", "class_counts",
        "class_bank_counts", "unique_slot_ids_by_class", "three_method_match_checks",
        "identical_roi_checks_per_group", "all_720_generation_sha256_verified_before_sampling",
        "sampling_frozen_before_pixel_read", "sampling_used_quality_scores",
        "manual_sample_replacement", "source_pixels_modified", "full_set_png_count",
        "residual_png_count", "paper_figure_png_count", "real_reference_png_count")}
    public_audit.update(ZERO_ACTIONS)
    paper_hashes = {p.name: sha256(p) for p in paper_files}
    public_summary = {
        "status": "COMPLETE", "selected_groups": 24, "paper_examples": 6,
        "paper_figure_sha256": paper_hashes,
        "real_reference_figure_sha256": sha256(output / "real_reference_examples.png"),
        "figures_retained_on_server": True, "figures_uploaded": False,
        "reason": "data publication permission has not been confirmed",
    }
    save(output / "EXP4_PUBLIC_PROTOCOL.json", public_protocol)
    save(output / "EXP4_PUBLIC_INTEGRITY_AUDIT.json", public_audit)
    save(output / "EXP4_PUBLIC_SUMMARY.json", public_summary)
    artifact_hashes = {p.relative_to(output).as_posix(): sha256(p)
                       for p in sorted(output.rglob("*")) if p.is_file()
                       and p.name != "exp4_status.json"}
    save(output / "exp4_status.json", {
        "status": "COMPLETE", "sampling_frozen": True, "render_complete": True,
        "matched_count": 240, "selected_count": 24,
        "full_set_count": 24, "residual_count": 24, "paper_figure_count": 2,
        "real_reference_count": 1, "artifact_sha256": artifact_hashes,
        **ZERO_ACTIONS,
    })
    print(f"[Exp4] COMPLETE：24 组审计图、24 组残差、2 张论文图、1 张真实参考图。", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "render", "all"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    try:
        if args.mode in ("prepare", "all"):
            prepare(args.config)
        if args.mode in ("render", "all"):
            render(args.config)
    except Exp4Error as exc:
        print(f"[{exc.status}] {exc.detail}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
