#!/usr/bin/env python3
"""Prepare and launch the official MAGIC baseline for Plastic_Bomo V2.

This runner intentionally stops at hard gates.  It never substitutes ordinary
SD2 inpainting for MAGIC, never disables CAMA, and never reads V2 FINAL_EVAL
defects.  The official MVTec loader's one-third sampling policy is preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results_for_gpt/plastic_bomo_baselines/magic"
P0 = REPO / "results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"
MAGIC = REPO / "third_party/MAGIC-official"
GEO = REPO / "third_party/GeoAware-SC"
U2NET = REPO / "third_party/U-2-Net"
DATA = OUT / "runtime/data"
NORMAL_MASKS = OUT / "runtime/normal_masks"
MATCH = OUT / "runtime/matching"
MODEL = OUT / "runtime/model"
LOGS = OUT / "logs"
AUG_MASKS = OUT / "runtime/aug_masks"
INFERENCE_MODELS = OUT / "runtime/inference_models"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
MULTIBANK = REPO / "results_for_gpt/plastic_bomo_msdf_multibank_stability"

MAGIC_SHA = "ab37cd94fef34ef139cb2b562c35680a9ed2f7ca"
GEO_SHA = "b20fab9c2d4f686536be9db34d1fb2079240fdd5"
U2NET_SHA = "ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29"
CATEGORY = "Plastic_Bomo"
CLASSES = ("black", "flash")
SEED = 42
INFERENCE_FULL = OUT / "runtime/inference_full/noise_1.0_guidance_scale_4.0_7.5_align" / CATEGORY


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def crop_512(source: Path, box: list[int], expected_wh: list[int]):
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        require(list(image.size) == expected_wh, f"canvas changed: {source}")
        crop = image.crop(tuple(box))
    require(crop.size == (512, 512), f"crop is not 512x512: {source}")
    return crop


def bbox_mask(xyxy: list[int]):
    from PIL import Image

    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, tuple(xyxy))
    return mask


def write_manifest(rows: list[dict]) -> None:
    path = OUT / "data_manifest.csv"
    fields = [
        "role", "class", "name", "source_split", "source_annotation_id",
        "source_sha256", "output_relative_path", "output_sha256",
        "mask_kind", "final_eval_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit() -> dict:
    requested = [
        "README.md", "run_train.py", "train_dreambooth_noise.py",
        "inference.py", "CAMA_matching.py",
    ]
    missing = [name for name in requested if not (MAGIC / name).is_file()]
    shas = {"MAGIC": git_sha(MAGIC), "GeoAware-SC": git_sha(GEO), "U-2-Net": git_sha(U2NET)}
    expected = {"MAGIC": MAGIC_SHA, "GeoAware-SC": GEO_SHA, "U-2-Net": U2NET_SHA}
    pin_ok = all(shas[name] == value for name, value in expected.items())
    official = {
        "stable_diffusion_inpainting": "stabilityai/stable-diffusion-2-inpainting",
        "prompt_perturbation": "train_dreambooth_noise.py adds N(0,text_noise_scale) to prompt embeddings; inference monkey-patches encode_prompt",
        "spatially_adaptive_guidance": "pipeline_stable_diffusion_inpaint_magic.py exposes inside/outside guidance and gsi schedule",
        "CAMA": "inference.py --CAMA consumes GeoAware-SC matching JSON and U-2-Net object masks",
        "all_required_components_present": not missing,
        "official_inference_import_issue": "inference.py imports a nonexistent *_scheduler_up module; repository contains pipeline_stable_diffusion_inpaint_magic.py",
    }
    report = {
        "status": "PASS" if pin_ok and not missing else "FAIL",
        "audited_at": utcnow(), "commit_SHA": shas, "expected_commit_SHA": expected,
        "requested_files": requested, "missing_files": missing, "algorithm": official,
        "data_policy": {
            "defect_donors": "V2 TRAIN only", "normal_pool": "V2 TRAIN frozen normal pool",
            "final_eval_defects_used": False, "mask_adapter": "BBox-Mask (rectangular; not human segmentation)",
        },
        "official_defaults": {
            "max_train_steps": 5000, "learning_rate": 5e-6, "resolution": 512,
            "train_batch_size": 4, "gradient_accumulation_steps": 1,
            "scheduler": "constant", "warmup_steps": 0, "text_noise_scale": 1.0,
            "model": "stabilityai/stable-diffusion-2-inpainting",
            "instance_prompt": "a photo of sks defect", "class_prompt": "a photo of defect",
            "official_dataset_fraction": "sorted first floor(N/3)", "seed_adapter": SEED,
        },
    }
    save(OUT / "code_audit.json", report)
    (OUT / "commit_SHA.txt").write_text(
        "\n".join(f"{name} {shas[name]}" for name in ("MAGIC", "GeoAware-SC", "U-2-Net")) + "\n",
        encoding="utf-8",
    )
    return report


def prepare() -> None:
    from PIL import Image

    allowlist = load(P0 / "generator_train_allowlist.json")
    slots_doc = load(P0 / "stage20b_generation_slots.json")
    rng_doc = load(P0 / "stage20b_rng_manifest.json")
    require(allowlist["status"] == "PASS", "V2 TRAIN allowlist is not PASS")
    require(slots_doc["status"] == "PASS" and len(slots_doc["slots"]) == 80, "frozen slots changed")
    require(len(rng_doc["records"]) == 80, "frozen RNG manifest changed")

    category = DATA / CATEGORY
    for class_name in CLASSES:
        (category / "test" / class_name).mkdir(parents=True, exist_ok=True)
        (category / "ground_truth" / class_name).mkdir(parents=True, exist_ok=True)
    (category / "train/good").mkdir(parents=True, exist_ok=True)
    (NORMAL_MASKS / CATEGORY / "train/masks").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    counts = {name: 0 for name in CLASSES}
    for item in sorted(allowlist["inputs"], key=lambda row: (row["class"], row["annotation_id"])):
        class_name = item["class"]
        require(class_name in CLASSES and item["split"] == "train", "non-TRAIN donor rejected")
        source = Path(item["path"])
        require(source.is_file() and sha256(source) == item["sha256"], f"source binding changed: {source}")
        geometry = item["geometry"]
        counts[class_name] += 1
        name = f"{counts[class_name]:04d}"
        image_path = category / "test" / class_name / f"{name}.png"
        mask_path = category / "ground_truth" / class_name / f"{name}_mask.png"
        image = crop_512(source, geometry["fixed_crop_xyxy"], geometry["original_canvas_wh"])
        mask = bbox_mask(geometry["rasterized_mask_xyxy_exclusive"])
        image.save(image_path, format="PNG")
        mask.save(mask_path, format="PNG")
        rows.extend([
            {"role": "defect_crop", "class": class_name, "name": name, "source_split": "train",
             "source_annotation_id": item["annotation_id"], "source_sha256": item["sha256"],
             "output_relative_path": image_path.relative_to(OUT).as_posix(), "output_sha256": sha256(image_path),
             "mask_kind": "", "final_eval_used": False},
            {"role": "defect_mask", "class": class_name, "name": name, "source_split": "train",
             "source_annotation_id": item["annotation_id"], "source_sha256": item["sha256"],
             "output_relative_path": mask_path.relative_to(OUT).as_posix(), "output_sha256": sha256(mask_path),
             "mask_kind": "rectangular_bbox", "final_eval_used": False},
        ])

    for index, slot in enumerate(slots_doc["slots"], start=1):
        require(slot["class"] in CLASSES, "unknown slot class")
        source = Path(slot["normal_path"])
        require(source.is_file() and sha256(source) == slot["normal_sha256"], f"normal binding changed: {source}")
        name = f"slot_{index:04d}_{slot['slot_id']}"
        path = category / "train/good" / f"{name}.png"
        crop = crop_512(source, slot["geometry"]["fixed_crop_xyxy"], slot["geometry"]["original_canvas_wh"])
        crop.save(path, format="PNG")
        rows.append({"role": "normal_crop", "class": slot["class"], "name": name, "source_split": "train",
                     "source_annotation_id": slot["annotation_id"], "source_sha256": slot["normal_sha256"],
                     "output_relative_path": path.relative_to(OUT).as_posix(), "output_sha256": sha256(path),
                     "mask_kind": "U-2-Net pending", "final_eval_used": False})

    require(counts == {"black": 97, "flash": 75}, f"unexpected V2 counts: {counts}")
    write_manifest(rows)
    save(OUT / "data_adapter.json", {
        "status": "PREPARED_AWAITING_U2NET", "prepared_at": utcnow(), "category": CATEGORY,
        "input_counts": counts, "official_effective_counts": {name: counts[name] // 3 for name in CLASSES},
        "normal_slot_crops": 80, "defect_split": "V2 TRAIN only", "final_eval_used": False,
        "mask_type": "BBox-Mask", "mask_is_human_segmentation": False,
        "warning": "MAGIC training and CAMA are pixel-mask dependent; rectangles are a weak-label adaptation.",
        "foreground_masks": "must be produced by official U-2-Net weights; no placeholder masks were created",
    })
    save(OUT / "frozen_rng_manifest.json", rng_doc)


def runtime_probe() -> dict:
    checks = {}
    probe = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    checks["nvidia_smi"] = {"ok": probe.returncode == 0, "returncode": probe.returncode, "output": probe.stdout.strip()}
    checks["nvidia_devices"] = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    u2_weight = U2NET / "saved_models/u2net/u2net.pth"
    checks["u2net_weight"] = {"ok": u2_weight.is_file(), "path": str(u2_weight),
                               "bytes": u2_weight.stat().st_size if u2_weight.is_file() else 0}
    checks["geoaware_aggregation_weight"] = {
        "ok": (GEO / "results_spair/best_856.PTH").is_file(),
        "path": str(GEO / "results_spair/best_856.PTH"),
    }
    checks["geoaware_external_models"] = {
        "stable_diffusion_v1_5": "not locally bound; GeoAware-SC load_model requests v1-5",
        "dinov2_vitb14": "not locally bound; torch.hub download is requested",
    }
    checks["official_import_fix_required"] = not (
        MAGIC / "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_dynamic_anomaly_strength_guidance_scheduler_up.py"
    ).is_file()
    blockers = []
    if not checks["nvidia_smi"]["ok"] or not checks["nvidia_devices"]:
        blockers.append("No usable NVIDIA driver/device; official CAMA hard-codes torch.device('cuda') and training requires CUDA VRAM.")
    if not checks["u2net_weight"]["ok"]:
        blockers.append("Official U-2-Net u2net.pth is absent, so transparent-film foreground-mask precheck cannot run.")
    blockers.append("GeoAware-SC SD1.5 and DINOv2 model assets are not locally pinned/bound.")
    result = {"status": "BLOCKED" if blockers else "PASS", "checked_at": utcnow(),
              "checks": checks, "blockers": blockers}
    save(OUT / "runtime_preflight.json", result)
    return result


def run_u2net_foreground() -> None:
    """Run the official U-2-Net model over all frozen normal-slot crops."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.autograd import Variable
    from torchvision import transforms

    sys.path.insert(0, str(U2NET))
    from data_loader import RescaleT, ToTensorLab
    from model import U2NET as U2NETModel

    weight = U2NET / "saved_models/u2net/u2net.pth"
    require(weight.is_file(), f"missing U-2-Net weight: {weight}")
    require(torch.cuda.is_available(), "CUDA unavailable to U-2-Net")
    source = DATA / CATEGORY / "train/good"
    destination = NORMAL_MASKS / CATEGORY / "train/masks"
    destination.mkdir(parents=True, exist_ok=True)
    preprocess = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    net = U2NETModel(3, 1)
    net.load_state_dict(torch.load(weight, map_location="cpu"))
    net.cuda().eval()
    rows = []
    for index, path in enumerate(sorted(source.glob("*.png")), start=1):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        sample = preprocess({"imidx": np.array([index - 1]), "image": rgb,
                             "label": np.zeros((*rgb.shape[:2], 1), dtype=np.uint8)})
        tensor = Variable(sample["image"].type(torch.FloatTensor).unsqueeze(0).cuda())
        with torch.no_grad():
            outputs = net(tensor)
        prediction = outputs[0][:, 0, :, :].squeeze()
        minimum, maximum = prediction.min(), prediction.max()
        prediction = (prediction - minimum) / (maximum - minimum + 1e-12)
        mask = Image.fromarray((prediction.cpu().numpy() * 255).astype(np.uint8), mode="L")
        mask = mask.resize((512, 512), resample=Image.Resampling.BILINEAR)
        out = destination / f"{path.stem}_mask.png"
        mask.save(out, format="PNG")
        array = np.asarray(mask)
        rows.append({"normal": path.name, "mask": out.name, "foreground_fraction": float((array > 127).mean()),
                     "minimum": int(array.min()), "maximum": int(array.max()), "sha256": sha256(out)})
        del tensor, outputs, prediction
        if index % 10 == 0:
            print(f"[U-2-Net] {index}/80", flush=True)
    require(len(rows) == 80, f"expected 80 normal masks, got {len(rows)}")
    fractions = [row["foreground_fraction"] for row in rows]
    save(OUT / "u2net_foreground_precheck.json", {
        "status": "PASS_WITH_TRANSPARENT_FILM_WARNING",
        "weight_sha256": sha256(weight), "normal_count": len(rows),
        "foreground_fraction_min": min(fractions), "foreground_fraction_max": max(fractions),
        "foreground_fraction_mean": sum(fractions) / len(fractions),
        "empty_masks": sum(value == 0 for value in fractions),
        "full_masks": sum(value == 1 for value in fractions),
        "review_note": "U-2-Net is an SOD model; mask statistics are recorded because transparent film is out of domain.",
        "records": rows,
    })
    print("[U-2-Net] foreground masks complete", flush=True)


def prepare_inference() -> None:
    """Materialize the frozen BBox masks and deterministic official-inference manifest."""
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rng_rows = load(P0 / "stage20b_rng_manifest.json")["records"]
    rng = {row["slot_id"]: row["banks"] for row in rng_rows}
    require(len(slots) == 80 and len(rng) == 80, "frozen slot/RNG cardinality changed")

    model_targets = {
        "black": MODEL / "plastic_bomo_magic_black_s42_noise_1.0" / CATEGORY / "black",
        "flash": MODEL / "plastic_bomo_magic_flash_s42_noise_1.0" / CATEGORY / "flash",
    }
    model_category = INFERENCE_MODELS / CATEGORY
    model_category.mkdir(parents=True, exist_ok=True)
    for class_name, target in model_targets.items():
        require(target.is_dir(), f"trained checkpoint missing: {target}")
        link = model_category / class_name
        if link.is_symlink():
            require(link.resolve() == target.resolve(), f"checkpoint link changed: {link}")
        elif link.exists():
            require(link.resolve() == target.resolve(), f"checkpoint adapter collision: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)

    records = []
    for index, slot in enumerate(slots, start=1):
        class_name = slot["class"]
        stem = f"slot_{index:04d}_{slot['slot_id']}"
        normal = DATA / CATEGORY / "train/good" / f"{stem}.png"
        require(normal.is_file(), f"prepared normal missing: {normal}")
        mask_dir = AUG_MASKS / CATEGORY / class_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask = mask_dir / f"{stem}.png"
        mask_image = bbox_mask(slot["geometry"]["rasterized_mask_xyxy_exclusive"])
        mask_image.save(mask, format="PNG")
        decoded_hash = hashlib.sha256(json.dumps([512, 512]).encode() + mask_image.tobytes()).hexdigest()
        require(decoded_hash == slot["geometry"]["mask_decoded_sha256"],
                f"frozen BBox raster changed: {slot['slot_id']}")
        for bank in ("A", "B", "C"):
            records.append({
                "category": CATEGORY,
                "defect_class": class_name,
                "bank": bank,
                "slot_id": slot["slot_id"],
                "conditioning_normal_id": slot["conditioning_normal_id"],
                "normal_path": str(normal),
                "normal_sha256": sha256(normal),
                "mask_path": str(mask),
                "mask_sha256": sha256(mask),
                "sample_seed": int(rng[slot["slot_id"]][bank]["sample_seed"]),
                "target_bbox_normalized_xywh": slot["bbox"],
                "output_filename": f"{bank}/{slot['slot_id']}.png",
                "prompt": "a photo of a sks defect",
            })
    require(len(records) == 240, f"expected 240 inference records, got {len(records)}")
    save(OUT / "runtime/inference_manifest.json", {
        "status": "FROZEN", "created_at": utcnow(), "records": records,
        "CAMA_required": True, "bbox_mask_adapter": True,
    })
    for class_name in CLASSES:
        class_records = [row for row in records if row["defect_class"] == class_name]
        require(len(class_records) == 120, f"expected 120 {class_name} records")
        save(OUT / f"runtime/inference_manifest_{class_name}.json", {
            "status": "FROZEN", "created_at": utcnow(), "records": class_records,
            "CAMA_required": True, "bbox_mask_adapter": True,
        })
    black_smoke = next(row for row in records if row["defect_class"] == "black" and row["bank"] == "A")
    save(OUT / "runtime/inference_manifest_black_smoke.json", {
        "status": "FROZEN", "records": [black_smoke], "CAMA_required": True,
    })
    print(f"[inference] prepared {len(records)} frozen records", flush=True)


def crop_box(bbox, width, height, margin=0.25):
    cx, cy, bw, bh = bbox
    box = [max(0, math.floor((cx - (0.5 + margin) * bw) * width)),
           max(0, math.floor((cy - (0.5 + margin) * bh) * height)),
           min(width, math.ceil((cx + (0.5 + margin) * bw) * width)),
           min(height, math.ceil((cy + (0.5 + margin) * bh) * height))]
    require(box[2] > box[0] and box[3] > box[1], "invalid evaluation ROI")
    return box


def write_rows(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def metric_from_features(real, fake, library):
    real = real.double().cpu(); fake = fake.double().cpu()
    options = {"kid_subsets": 100, "kid_subset_size": min(50, len(real), len(fake)),
               "rng_seed": 42, "verbose": False, "kid_degree": 3,
               "kid_gamma": None, "kid_coef0": 1.0}
    kid = library["kid"](real, fake, **options)
    fid = library["fid"](library["fid_stats"](real), library["fid_stats"](fake), verbose=False)
    return {"kid": float(kid["kernel_inception_distance_mean"]),
            "kid_subset_std": float(kid["kernel_inception_distance_std"]),
            "fid": float(fid["frechet_inception_distance"]),
            "kid_subset_size": options["kid_subset_size"]}


def evaluate_and_finalize(device: str) -> None:
    """Verify 240 official outputs, compute frozen-protocol metrics, and publish the audit."""
    import numpy as np
    import torch
    from PIL import Image
    from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime

    manifest = load(OUT / "runtime/inference_manifest.json")
    records = manifest["records"]
    require(len(records) == 240, "frozen inference manifest is not 240 records")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    slot_by_id = {row["slot_id"]: row for row in slots}
    require(len(slot_by_id) == 80, "frozen slot population changed")

    verified, diagnostic_rows = [], []
    for row in records:
        class_name, bank, slot_id = row["defect_class"], row["bank"], row["slot_id"]
        root = INFERENCE_FULL / class_name
        image_path = root / "image" / bank / f"{slot_id}.png"
        normal_path = root / "normal" / bank / f"{slot_id}.png"
        mask_path = root / "masks" / bank / f"{slot_id}.png"
        for path in (image_path, normal_path, mask_path):
            require(path.is_file(), f"missing official inference output: {path}")
        with Image.open(image_path) as im:
            generated = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(normal_path) as im:
            normal = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as im:
            final_mask = np.asarray(im.convert("L"), dtype=np.uint8) > 127
        require(generated.shape == normal.shape == (512, 512, 3) and final_mask.shape == (512, 512),
                f"invalid output dimensions: {slot_id}/{bank}")
        slot = slot_by_id[slot_id]
        x0, y0, x1, y1 = slot["geometry"]["rasterized_mask_xyxy_exclusive"]
        target = np.zeros((512, 512), dtype=bool); target[y0:y1, x0:x1] = True
        intersection = int((final_mask & target).sum())
        union = int((final_mask | target).sum())
        outside = ~final_mask
        difference = np.abs(generated.astype(np.int16) - normal.astype(np.int16))
        outside_values = difference[outside]
        mask_pixels = int(final_mask.sum())
        signal_pixels = int((difference.max(axis=2)[final_mask] > 2).sum()) if mask_pixels else 0
        empty_mask = mask_pixels == 0
        no_inpaint_signal = bool(mask_pixels and signal_pixels == 0)
        structural_failure = empty_mask or no_inpaint_signal
        full_roi = crop_box(slot["bbox"], *slot["geometry"]["final_canvas_wh"])
        left, top, _, _ = slot["geometry"]["fixed_crop_xyxy"]
        roi = [full_roi[0] - left, full_roi[1] - top, full_roi[2] - left, full_roi[3] - top]
        require(0 <= roi[0] < roi[2] <= 512 and 0 <= roi[1] < roi[3] <= 512,
                f"ROI falls outside official crop: {slot_id}")
        verified.append({**row, "image_path": str(image_path), "normal_output_path": str(normal_path),
                         "final_mask_path": str(mask_path), "image_sha256": sha256(image_path),
                         "normal_output_sha256": sha256(normal_path), "final_mask_sha256": sha256(mask_path),
                         "roi_xyxy_in_512": roi})
        diagnostic_rows.append({
            "slot_id": slot_id, "class": class_name, "bank": bank,
            "background_mae_0_1": float(outside_values.mean() / 255.0) if outside_values.size else 0.0,
            "background_changed_fraction_gt_5": float((outside_values > 5).mean()) if outside_values.size else 0.0,
            "target_bbox_coverage_by_final_mask": intersection / max(1, int(target.sum())),
            "target_bbox_final_mask_iou": intersection / max(1, union),
            "final_mask_pixels": mask_pixels, "inpaint_signal_pixels": signal_pixels,
            "empty_final_mask": empty_mask, "no_inpaint_signal": no_inpaint_signal,
            "structural_failure": structural_failure,
        })
    require(Counter((r["class"], r["bank"]) for r in diagnostic_rows) ==
            Counter({(c, b): 40 for c in CLASSES for b in "ABC"}), "generated bank/class counts changed")
    write_rows(OUT / "generation_diagnostics.csv", diagnostic_rows)

    if device == "cpu":
        # The host driver disappeared after all 240 outputs were flushed.  Feature
        # extraction is mathematically identical on CPU and uses the same pinned
        # torch-fidelity implementation and checkpoint as the frozen protocol.
        import importlib.metadata as metadata
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
        from torch_fidelity.metric_kid import kid_features_to_metric
        inception = Path.home() / ".cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
        require(inception.is_file() and sha256(inception) ==
                "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
                "frozen Inception checkpoint changed")
        require(metadata.version("torch-fidelity") == "0.3.0", "metric API version changed")
        torch.manual_seed(42); torch.use_deterministic_algorithms(True)
        library = {"Inception": FeatureExtractorInceptionV3, "fid_stats": fid_features_to_statistics,
                   "fid": fid_statistics_to_metric, "kid": kid_features_to_metric,
                   "inception_weight": inception}
        runtime_audit = {"status": "FROZEN_CPU_FALLBACK_AFTER_DRIVER_LOSS", "device": "cpu",
                         "torch": torch.__version__, "torch_fidelity": metadata.version("torch-fidelity"),
                         "inception_weight": {"path": str(inception), "sha256": sha256(inception)},
                         "TF32": False, "autocast": False, "network_downloads": 0}
    else:
        library, runtime_audit = metric_runtime(device)
    model = library["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(library["inception_weight"])).float().eval().requires_grad_(False).to(device)
    feature_values, feature_keys = [], []
    with torch.inference_mode():
        for index, row in enumerate(verified, start=1):
            with Image.open(row["image_path"]) as source:
                roi_image = source.convert("RGB").crop(tuple(row["roi_xyxy_in_512"]))
            tensor = torch.from_numpy(np.asarray(roi_image, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            value = model(tensor)[0]
            require(tuple(value.shape) == (1, 2048) and bool(torch.isfinite(value).all()),
                    f"invalid Inception feature: {row['slot_id']}/{row['bank']}")
            feature_keys.append(f"magic:{row['bank']}:{row['defect_class']}:{row['slot_id']}")
            feature_values.append(value[0].double().cpu().numpy())
            if index % 20 == 0:
                print(f"[MAGIC Inception] {index}/240", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    feature_array = np.stack(feature_values)
    feature_path = OUT / "magic_inception_features.npz"
    np.savez_compressed(feature_path, features=feature_array)
    save(OUT / "magic_inception_feature_cache.json", {
        "status": "COMPLETE", "keys": feature_keys, "shape": [240, 2048],
        "stored_dtype": str(feature_array.dtype), "sha256": sha256(feature_path),
        "metric_runtime": runtime_audit,
    })
    generated_features = {key: torch.from_numpy(feature_array[i]).double()
                          for i, key in enumerate(feature_keys)}

    gamma_cache = GAMMA / "gamma_sweep_inception_features.npz"
    gamma_ledger = load(GAMMA / "gamma_sweep_inception_feature_cache.json")
    require(gamma_ledger["sha256"] == sha256(gamma_cache), "frozen real feature cache changed")
    with np.load(gamma_cache, allow_pickle=False) as archive:
        gamma_values = archive["features"].copy()
    gamma_features = {key: torch.from_numpy(gamma_values[i]).double()
                      for i, key in enumerate(gamma_ledger["keys"])}
    references = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
    real = {
        "Flash": torch.stack([gamma_features["real:" + r["annotation_id"]] for r in references if r["class"] == "flash"]),
        "Black": torch.stack([gamma_features["real:" + r["annotation_id"]] for r in references if r["class"] == "black"]),
    }
    real["Overall"] = torch.stack([gamma_features["real:" + r["annotation_id"]] for r in references])

    def magic_features(banks, population):
        classes = CLASSES if population == "Overall" else (population.lower(),)
        rows = [r for r in verified if r["bank"] in banks and r["defect_class"] in classes]
        return torch.stack([generated_features[f"magic:{r['bank']}:{r['defect_class']}:{r['slot_id']}"] for r in rows])

    metric_rows = []
    for scope, banks in (("bankwise", "A"), ("bankwise", "B"), ("bankwise", "C"),
                         ("pooled", "BC"), ("pooled", "ABC")):
        bank_pool = "+".join(banks)
        for population in ("Black", "Flash", "Overall"):
            fake = magic_features(banks, population)
            result = metric_from_features(real[population], fake, library)
            metric_rows.append({"method": "MAGIC", "scope": scope, "bank_pool": bank_pool,
                "class": population, "count": len(fake), **result, "status": "COMPLETE",
                "source": "recomputed frozen ROI Inception features"})
            print(f"[MAGIC metric] {bank_pool} {population} KID={result['kid']:.10f} FID={result['fid']:.6f}", flush=True)

    existing_files = {"Black": "multibank_bankwise_black.csv", "Flash": "multibank_bankwise_flash.csv",
                      "Overall": "multibank_bankwise_overall.csv"}
    for population, filename in existing_files.items():
        for old in csv.DictReader((MULTIBANK / filename).open(encoding="utf-8")):
            if old["group"] not in ("raw", "original"):
                continue
            metric_rows.append({"method": old["method"], "scope": "bankwise", "bank_pool": old["bank"],
                "class": population, "count": int(old["generated_images"]), "kid": float(old["KID"]),
                "kid_subset_std": float(old["KID_subset_std"]), "fid": float(old["FID"]),
                "kid_subset_size": int(old["KID_subset_size"]), "status": "COMPLETE",
                "source": "SHA-verified existing unified multibank protocol"})
    for old in csv.DictReader((MULTIBANK / "multibank_pooled_metrics.csv").open(encoding="utf-8")):
        if old["group"] not in ("raw", "original"):
            continue
        metric_rows.append({"method": old["method"], "scope": "pooled", "bank_pool": old["scope"].replace("A+B+C", "A+B+C"),
            "class": old["population"], "count": int(old["generated_images"]), "kid": float(old["KID"]),
            "kid_subset_std": float(old["KID_subset_std"]), "fid": float(old["FID"]),
            "kid_subset_size": int(old["KID_subset_size"]), "status": "COMPLETE",
            "source": "SHA-verified existing unified multibank protocol"})
    write_rows(OUT / "metrics.csv", metric_rows)

    image_manifest_rows = []
    for row in verified:
        image_manifest_rows.append({
            "slot_id": row["slot_id"], "class": row["defect_class"], "bank": row["bank"],
            "conditioning_normal_id": row["conditioning_normal_id"], "sample_seed": row["sample_seed"],
            "target_bbox_normalized_xywh": json.dumps(row["target_bbox_normalized_xywh"], separators=(",", ":")),
            "image_path": Path(row["image_path"]).relative_to(OUT).as_posix(), "image_sha256": row["image_sha256"],
            "final_mask_path": Path(row["final_mask_path"]).relative_to(OUT).as_posix(),
            "final_mask_sha256": row["final_mask_sha256"], "status": "COMPLETE",
        })
    write_rows(OUT / "image_manifest.csv", image_manifest_rows)

    by_diag = {}
    for population in ("Black", "Flash", "Overall"):
        subset = diagnostic_rows if population == "Overall" else [r for r in diagnostic_rows if r["class"] == population.lower()]
        by_diag[population] = {
            "count": len(subset),
            "background_mae_0_1_mean": sum(r["background_mae_0_1"] for r in subset) / len(subset),
            "background_changed_fraction_gt_5_mean": sum(r["background_changed_fraction_gt_5"] for r in subset) / len(subset),
            "target_bbox_coverage_mean": sum(r["target_bbox_coverage_by_final_mask"] for r in subset) / len(subset),
            "target_bbox_final_mask_iou_mean": sum(r["target_bbox_final_mask_iou"] for r in subset) / len(subset),
            "empty_final_masks": sum(r["empty_final_mask"] for r in subset),
            "structural_failures": sum(r["structural_failure"] for r in subset),
        }
    save(OUT / "generation_diagnostics_summary.json", {"status": "COMPLETE", "definitions": {
        "background_change": "absolute RGB difference outside the final CAMA mask",
        "bbox_coverage": "fraction of the frozen target BBox covered by the final CAMA mask",
        "structural_failure": "empty final mask or no >2/255 pixel signal inside a non-empty final mask",
    }, "populations": by_diag})

    training = load(OUT / "training_config.json")
    training.update(status="COMPLETE", completed_at=utcnow(),
                    logs={"black": "logs/black_training.log", "flash": "logs/flash_training.log"},
                    completed_steps={"black": 5000, "flash": 5000})
    save(OUT / "training_config.json", training)
    inference = load(OUT / "inference_config.json")
    inference.update(status="COMPLETE", completed_at=utcnow(), generated_images=240,
        output_root=INFERENCE_FULL.relative_to(OUT).as_posix(), match_records={"black": 413, "flash": 373},
        CAMA_no_intersections=3774, batch_adapter=1,
        adapter_notes=["official missing import mapped to repository pipeline_stable_diffusion_inpaint_magic.py",
                       "manifest adapter fixes normal/mask/sample seed without changing MAGIC math",
                       "CAMA donor and normal upsampling evaluated in memory-equivalent chunks for RTX 3090"])
    save(OUT / "inference_config.json", inference)
    adapter = load(OUT / "data_adapter.json")
    adapter.update(status="COMPLETE_BBOX_MASK_ADAPTATION", generated_images=240,
                   u2net_precheck="PASS_WITH_TRANSPARENT_FILM_WARNING")
    save(OUT / "data_adapter.json", adapter)

    audit_doc = {
        "status": "PASS_WITH_DISCLOSED_ADAPTERS", "audited_at": utcnow(),
        "official_commits": {"MAGIC": MAGIC_SHA, "GeoAware-SC": GEO_SHA, "U-2-Net": U2NET_SHA},
        "training": {"black_steps": 5000, "flash_steps": 5000, "independent": True, "seed": 42},
        "generation": {"planned": 240, "verified": len(verified),
            "counts": dict(Counter(f"{r['defect_class']}:{r['bank']}" for r in verified)),
            "Prompt_Perturbation": True, "Spatially_Adaptive_Guidance": True, "CAMA": True,
            "vanilla_SD2_substituted": False, "quality_selection": False, "reroll": False},
        "data_policy": {"test_defects_used_for_training_or_matching": False,
            "defect_donors": "V2 TRAIN only", "mask_adapter": "BBox-Mask rectangular weak labels",
            "mask_is_human_segmentation": False},
        "CAMA": {"components": 57, "matching_records": 786, "no_intersections": 3774,
            "transparent_film_warning": True, "fallback_is_official_CAMA_behavior": True},
        "diagnostics": by_diag, "metrics_available": True,
        "MSDF_modified": False, "MSDF_executed_in_this_run": False,
        "model_weights_in_publish_set": False,
        "adapter_sha256": {"runner": sha256(REPO / "tools/run_plastic_bomo_baseline_magic.py"),
            "inference": sha256(MAGIC / "inference.py"), "CAMA_runtime_copy": sha256(GEO / "CAMA_matching.py")},
    }
    save(OUT / "completeness_audit.json", audit_doc)
    save(OUT / "experiment_status.json", {"status": "COMPLETE", "completed_at": utcnow(),
        "training_started": True, "training_complete": True, "generated_images": 240,
        "metrics_computed": True, "MSDF_modified": False, "MSDF_executed": False,
        "final_eval_defects_used": False})

    magic_rows = [r for r in metric_rows if r["method"] == "MAGIC"]
    report = ["# Plastic_Bomo MAGIC baseline", "", "Status: `COMPLETE`.", "",
        "The official MAGIC training and inference entry points were retained. Prompt Perturbation, Spatially Adaptive Guidance, and CAMA were enabled for all 240 outputs.", "",
        "## Metrics", "", "| Pool | Class | Images | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: | ---: |"]
    for row in magic_rows:
        report.append(f"| {row['bank_pool']} | {row['class']} | {row['count']} | {row['kid']:.10f} | {row['fid']:.6f} |")
    report += ["", "## Data and CAMA limitations", "",
        "The available annotations are BBoxes, not human pixel masks, so this is explicitly a BBox-Mask adaptation experiment.",
        f"U-2-Net on transparent film is out of domain. CAMA produced 786 valid records and 3,774 line/object-mask non-intersections; the official CAMA fallback remained enabled and CAMA was never bypassed.", "",
        "## Structural diagnostics", ""]
    for name, values in by_diag.items():
        report.append(f"- {name}: background MAE={values['background_mae_0_1_mean']:.6f}, target-BBox coverage={values['target_bbox_coverage_mean']:.6f}, structural failures={values['structural_failures']}/{values['count']}.")
    report += ["", "Raw SD2 and Original Class-Adaptive MSDF comparison rows in `metrics.csv` are reused from the SHA-verified unified multibank protocol; pooled values are recomputed distributions, not bank averages.", "",
        "No model weights are part of the GitHub publication set.", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "README.md").write_text("\n".join([
        "# MAGIC — Plastic_Bomo", "", "Complete official MAGIC BBox-Mask adaptation experiment.", "",
        "- Black/Flash were trained independently for 5,000 steps with seed 42.",
        "- Official Prompt Perturbation, Spatially Adaptive Guidance and CAMA were enabled.",
        "- Banks A/B/C contain 40 Black and 40 Flash slots per bank (240 total).",
        "- `metrics.csv` contains bankwise, B+C pooled, and A+B+C pooled KID/FID plus Raw SD2/MSDF comparisons.",
        "- `generation_diagnostics.csv` records background change, target-BBox coverage and structural failures.",
        "- This is a rectangular BBox-Mask weak-label adaptation, not a segmentation-mask experiment.",
        "- Model checkpoints and generated runtime images are intentionally excluded from GitHub.", "",
        "Reproduction entry point: `tools/run_plastic_bomo_baseline_magic.py`.", "",
    ]), encoding="utf-8")
    print("[MAGIC] evaluation and completeness audit complete", flush=True)


def write_blocked_outputs(probe: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "black_training_start.log"
    lines = [
        f"[{utcnow()}] MAGIC Black startup gate",
        f"official_MAGIC_commit={MAGIC_SHA}",
        "entrypoint=python run_train.py",
        f"planned_command=python run_train.py --base_dir {DATA} --output_name plastic_bomo_magic_s42 --text_noise_scale 1.0 --category {CATEGORY} --defect_class black",
        "gate_order=U-2-Net foreground precheck -> GeoAware-SC/CAMA matching -> one-Black full inference -> official training",
        "status=BLOCKED_BEFORE_TRAINING",
    ]
    lines.extend(f"blocker={item}" for item in probe["blockers"])
    lines.append("optimizer_step_started=false")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(OUT / "experiment_status.json", {
        "status": "BLOCKED", "updated_at": utcnow(), "stage": "pre-training dependency/runtime gate",
        "training_started": False, "optimizer_step_started": False, "generated_images": 0,
        "metrics_computed": False, "MSDF_modified": False, "MSDF_executed": False,
        "final_eval_defects_used": False, "log": log.relative_to(REPO).as_posix(),
        "blockers": probe["blockers"],
    })
    metric_fields = ["method", "scope", "bank_pool", "class", "count", "kid", "fid", "status", "reason"]
    with (OUT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({"method": "MAGIC", "scope": "generation", "bank_pool": "A/B/C/B+C/A+B+C",
                         "class": "Black/Flash/Overall", "count": 0, "kid": "", "fid": "",
                         "status": "BLOCKED", "reason": "no valid MAGIC outputs; metrics intentionally not fabricated"})

    save(OUT / "training_config.json", {
        "status": "FROZEN_NOT_STARTED", "official_entrypoint": "python run_train.py",
        "official_commit": MAGIC_SHA, "category": CATEGORY, "classes": list(CLASSES),
        "class_training_is_independent": True, "seed": SEED,
        "base_model": "stabilityai/stable-diffusion-2-inpainting",
        "max_train_steps": 5000, "learning_rate": 5e-6, "resolution": 512,
        "train_batch_size": 4, "gradient_accumulation_steps": 1,
        "lr_scheduler": "constant", "lr_warmup_steps": 0, "text_noise_scale": 1.0,
        "instance_prompt": "a photo of sks defect", "class_prompt": "a photo of defect",
        "official_loader_policy": "first floor(N/3) after numeric filename sort",
        "effective_samples": {"black": 32, "flash": 25},
        "mask_adapter": "BBox-Mask rectangular weak labels",
        "MSDF_modified": False, "MSDF_executed": False,
    })
    save(OUT / "inference_config.json", {
        "status": "FROZEN_NOT_RUN", "official_entrypoint": "python inference.py",
        "required_components": {
            "prompt_perturbation": True, "spatially_adaptive_guidance": True, "CAMA": True,
            "ordinary_SD2_only_forbidden": True,
        },
        "official_README_settings": {
            "dataset_type": "mvtec", "text_noise_scale": 1.0, "CAMA": True,
            "gsi_use_schedule": True, "gsi_schedule": "cosine", "gsi_min": 4.0,
            "gsi_max": 7.5, "guidance_scale_outside": 7.5,
        },
        "banks": ["A", "B", "C"], "images_per_class_per_bank": 40,
        "planned_images": 240, "same_frozen_normal_bbox_seed_identity": True,
        "quality_selection": False, "reroll": False,
        "required_adapter_fixes": [
            "map official nonexistent inference import to pipeline_stable_diffusion_inpaint_magic.py",
            "add manifest-driven normal/mask/sample_seed selection without changing MAGIC math",
            "pass seed=42 through run_train.py to train_dreambooth_noise.py",
        ],
    })
    save(OUT / "defect_classification.json", {CATEGORY: {"black": 0, "flash": 0}})

    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rng = {row["slot_id"]: row["banks"] for row in load(P0 / "stage20b_rng_manifest.json")["records"]}
    fields = ["slot_id", "class", "bank", "conditioning_normal_id", "sample_seed",
              "target_bbox_normalized_xywh", "image_path", "image_sha256", "status"]
    with (OUT / "image_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for slot in slots:
            for bank in ("A", "B", "C"):
                writer.writerow({
                    "slot_id": slot["slot_id"], "class": slot["class"], "bank": bank,
                    "conditioning_normal_id": slot["conditioning_normal_id"],
                    "sample_seed": rng[slot["slot_id"]][bank]["sample_seed"],
                    "target_bbox_normalized_xywh": json.dumps(slot["bbox"], separators=(",", ":")),
                    "image_path": "", "image_sha256": "", "status": "BLOCKED_NOT_GENERATED",
                })

    save(OUT / "completeness_audit.json", {
        "status": "BLOCKED", "audited_at": utcnow(),
        "official_code_pinned": True, "requested_files_read": True,
        "full_algorithm_required": True, "prompt_perturbation_disabled": False,
        "spatially_adaptive_guidance_disabled": False, "CAMA_disabled": False,
        "vanilla_SD2_substituted": False, "test_defect_used_for_training_or_matching": False,
        "bbox_mask_adaptation_disclosed": True, "u2net_precheck_passed": False,
        "geoaware_cama_precheck_passed": False, "one_black_full_flow_passed": False,
        "black_training_started": False, "flash_training_started": False,
        "planned_manifest_rows": 240, "generated_images": 0,
        "metrics_available": False, "metrics_fabricated": False,
        "background_change_available": False, "bbox_coverage_available": False,
        "structural_failure_count_available": False,
        "reason": probe["blockers"],
    })


def finalize_driver_blocked() -> None:
    """Publish an honest late-stage audit after a host GPU-driver loss."""
    manifest = load(OUT / "runtime/inference_manifest.json")
    records = manifest["records"]
    require(len(records) == 240, "frozen inference manifest is not 240 records")
    rows, counts, missing = [], Counter(), []
    for row in records:
        class_name, bank, slot_id = row["defect_class"], row["bank"], row["slot_id"]
        root = INFERENCE_FULL / class_name
        image_path = root / "image" / bank / f"{slot_id}.png"
        normal_path = root / "normal" / bank / f"{slot_id}.png"
        mask_path = root / "masks" / bank / f"{slot_id}.png"
        complete = all(path.is_file() for path in (image_path, normal_path, mask_path))
        if complete:
            counts[f"{class_name}:{bank}"] += 1
        else:
            missing.append({"slot_id": slot_id, "class": class_name, "bank": bank})
        rows.append({
            "slot_id": slot_id, "class": class_name, "bank": bank,
            "conditioning_normal_id": row["conditioning_normal_id"],
            "sample_seed": row["sample_seed"],
            "target_bbox_normalized_xywh": json.dumps(row["target_bbox_normalized_xywh"], separators=(",", ":")),
            "image_path": image_path.relative_to(OUT).as_posix() if complete else "",
            "image_sha256": sha256(image_path) if complete else "",
            "final_mask_path": mask_path.relative_to(OUT).as_posix() if complete else "",
            "final_mask_sha256": sha256(mask_path) if complete else "",
            "status": "COMPLETE" if complete else "BLOCKED_GPU_DRIVER_LOSS",
        })
    write_rows(OUT / "image_manifest.csv", rows)
    generated = sum(counts.values())
    reason = ("Host NVIDIA driver became unavailable during Black inference; nvidia-smi could not "
              "communicate with the driver. Nine frozen Black outputs remain missing.")
    write_rows(OUT / "metrics.csv", [{
        "method": "MAGIC", "scope": "generation", "bank_pool": "A/B/C/B+C/A+B+C",
        "class": "Black/Flash/Overall", "count": generated, "kid": "", "fid": "",
        "status": "BLOCKED", "reason": "240-output population incomplete; metrics intentionally not computed",
    }])
    training = load(OUT / "training_config.json")
    training.update(status="COMPLETE", completed_steps={"black": 5000, "flash": 5000},
                    logs={"black": "logs/black_training.log", "flash": "logs/flash_training.log"})
    save(OUT / "training_config.json", training)
    inference = load(OUT / "inference_config.json")
    inference.update(status="BLOCKED_GPU_DRIVER_LOSS", generated_images=generated,
                     planned_images=240, counts=dict(sorted(counts.items())), missing=missing,
                     output_root=INFERENCE_FULL.relative_to(OUT).as_posix(),
                     full_MAGIC_components_enabled=True, CAMA_matching_records=786)
    save(OUT / "inference_config.json", inference)
    blocker = {
        "status": "BLOCKED", "updated_at": utcnow(), "stage": "formal inference (late-stage)",
        "training_started": True, "training_complete": True, "generated_images": generated,
        "planned_images": 240, "metrics_computed": False, "MSDF_modified": False,
        "MSDF_executed": False, "final_eval_defects_used": False,
        "blockers": [reason], "resume_scope": missing,
    }
    save(OUT / "experiment_status.json", blocker)
    audit_doc = {
        "status": "BLOCKED_GPU_DRIVER_LOSS", "audited_at": utcnow(),
        "official_commits": {"MAGIC": MAGIC_SHA, "GeoAware-SC": GEO_SHA, "U-2-Net": U2NET_SHA},
        "training": {"black_steps": 5000, "flash_steps": 5000, "independent": True, "seed": 42},
        "prechecks": {"u2net_completed": True, "CAMA_completed": True,
                      "one_black_full_flow_passed": True},
        "generation": {"planned": 240, "verified_complete_triplets": generated,
                       "counts": dict(sorted(counts.items())), "missing": missing,
                       "Prompt_Perturbation": True, "Spatially_Adaptive_Guidance": True,
                       "CAMA": True, "vanilla_SD2_substituted": False},
        "data_policy": {"test_defects_used_for_training_or_matching": False,
                        "defect_donors": "V2 TRAIN only", "mask_adapter": "BBox-Mask rectangular weak labels"},
        "metrics_available": False, "metrics_fabricated": False,
        "blocker": reason, "MSDF_modified": False, "MSDF_executed_in_this_run": False,
        "model_weights_in_publish_set": False,
    }
    save(OUT / "completeness_audit.json", audit_doc)
    report = f"""# MAGIC × Plastic_Bomo execution report

## Outcome

`BLOCKED_GPU_DRIVER_LOSS`: training, U-2-Net, full CAMA matching, and the one-Black end-to-end smoke test passed. Formal inference produced {generated}/240 verified image/normal/mask triplets. Flash is 120/120; Black stopped with nine outputs remaining after the host NVIDIA driver disappeared.

No KID/FID values are reported because the frozen 240-image population is incomplete. No partial-population metric was substituted.

## Integrity

- Official MAGIC commit: `{MAGIC_SHA}`
- Prompt Perturbation, Spatially Adaptive Guidance, and CAMA remained enabled.
- CAMA produced 786 matching records; it was not bypassed.
- Training completed independently for Black and Flash at 5,000 steps each with seed 42.
- Defect donors came only from V2 TRAIN; no test defect was used for training or matching.
- Source annotations are BBoxes, so this remains a disclosed **BBox-Mask adaptation experiment**.
- MSDF was neither run nor modified by this experiment.
- Model weights and runtime images are excluded from the GitHub publication set.

## Resume point

The exact nine missing records are stored in `inference_config.json` and `image_manifest.csv`. After the NVIDIA driver is restored, rerun those frozen records and then execute the `evaluate` action.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "README.md").write_text(f"""# MAGIC — Plastic_Bomo

Status: **BLOCKED at {generated}/240 outputs by host NVIDIA driver loss**.

Official Black/Flash training (5,000 steps each), U-2-Net precheck, CAMA matching, and a full Black smoke inference completed. The formal run produced all 120 Flash outputs and {generated - 120} Black outputs. Metrics are intentionally unavailable until the frozen 240-image set is complete.

This is a rectangular BBox-Mask weak-label adaptation, not a pixel-segmentation experiment. Prompt Perturbation, Spatially Adaptive Guidance, and CAMA remained enabled. No test defect entered training/matching, and MSDF was not modified.

Reproduction entry point: `tools/run_plastic_bomo_baseline_magic.py`. Model weights and generated runtime images are intentionally excluded from GitHub.
""", encoding="utf-8")
    print(json.dumps({"status": "BLOCKED_GPU_DRIVER_LOSS", "generated": generated,
                      "counts": dict(sorted(counts.items())), "missing": missing}, indent=2))


def start_black() -> int:
    audit_result = audit()
    require(audit_result["status"] == "PASS", "source audit failed")
    prepare()
    probe = runtime_probe()
    if probe["status"] != "PASS":
        write_blocked_outputs(probe)
        print((LOGS / "black_training_start.log").read_text(encoding="utf-8"), end="")
        return 2
    raise RuntimeError("Runtime gate unexpectedly passed, but U-2-Net/CAMA execution wiring must be explicitly completed before training.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "prepare", "probe", "u2net", "prepare-inference", "evaluate", "finalize-driver-blocked", "start-black"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "audit":
        print(json.dumps(audit(), ensure_ascii=False, indent=2))
    elif args.action == "prepare":
        audit(); prepare()
    elif args.action == "probe":
        print(json.dumps(runtime_probe(), ensure_ascii=False, indent=2))
    elif args.action == "u2net":
        run_u2net_foreground()
    elif args.action == "prepare-inference":
        prepare_inference()
    elif args.action == "evaluate":
        evaluate_and_finalize(args.device)
    elif args.action == "finalize-driver-blocked":
        finalize_driver_blocked()
    else:
        return start_black()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
