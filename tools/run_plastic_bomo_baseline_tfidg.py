#!/usr/bin/env python3
"""Run the frozen Plastic_Bomo TF-IDG baseline without training or selection.

The script deliberately separates audit, two-image smoke testing, and formal
generation.  It adapts only data/paths/seeds around the public TF-IDG
inference implementation; no model parameter is updated.
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO.parent / "third_party/TF-IDG-main"
DEFAULT_PREPARED = (REPO / "results_for_gpt/plastic_bomo_chapter3_exp2_baseline_generation/"
                    "prepared_v2_20261017_073039")
DEFAULT_P0 = (REPO / "results_for_gpt/plastic_bomo_stage20b_p0_preflight/"
              "server_preflight_20261008_084115")
DEFAULT_OUTPUT = REPO / "results_for_gpt/plastic_bomo_baselines/tfidg"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
BANKS = ("A", "B", "C")
CLASSES = ("black", "flash")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def relative_or_absolute(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "smoke", "generate", "evaluate", "all"),
                        default="audit")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--p0", type=Path, default=DEFAULT_P0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def expected_paths(args):
    source = args.source.resolve()
    return {
        "anydoor": source / "checkPoint/epoch=1-step=8687.ckpt",
        "dinov2": source / "checkPoint/dinov2_vitg14_pretrain.pth",
        "inference_config": source / "configs/inference.yaml",
        "model_config": source / "configs/anydoor.yaml",
        "tfidg_entry": source / "run_inference.py",
        "training_manifest": args.prepared.resolve() / "training_manifest.json",
        "generation_slots": args.prepared.resolve() / "generation_slots.json",
        "prepared_binding": args.prepared.resolve() / "prepared_binding.json",
        "stage20b_slots": args.p0.resolve() / "stage20b_generation_slots.json",
        "stage20b_rng": args.p0.resolve() / "stage20b_rng_manifest.json",
    }


def validate_inputs(args):
    paths = expected_paths(args)
    for name, path in paths.items():
        require(path.is_file(), f"MISSING_INPUT:{name}:{path}")
    binding = load(paths["prepared_binding"])
    require(binding["status"] == "FROZEN", "PREPARED_BINDING_NOT_FROZEN")
    require(sha256(paths["training_manifest"]) == binding["training_manifest_sha256"],
            "TRAINING_MANIFEST_CHANGED")
    require(sha256(paths["generation_slots"]) == binding["generation_slots_sha256"],
            "GENERATION_SLOTS_CHANGED")
    return paths


def build_records(args, paths):
    training = load(paths["training_manifest"])
    require(training["status"] == "TRAIN_ONLY_ASSETS_PREPARED" and training["count"] == 172,
            "TRAINING_MANIFEST_INVALID")
    train_by_id = {row["annotation_id"]: row for row in training["records"]}
    require(len(train_by_id) == 172 and Counter(r["class"] for r in train_by_id.values()) ==
            Counter({"black": 97, "flash": 75}), "TRAINING_REFERENCE_POOL_CHANGED")

    stage_slots = load(paths["stage20b_slots"])["slots"]
    stage_by_id = {row["slot_id"]: row for row in stage_slots}
    rng = {row["slot_id"]: row["banks"] for row in load(paths["stage20b_rng"])["records"]}
    require(len(stage_by_id) == len(rng) == 80, "STAGE20B_SLOT_OR_RNG_COUNT_CHANGED")

    prepared = load(paths["generation_slots"])
    source_rows = prepared["methods"]["anomalydiffusion"]
    require(len(source_rows) == 240, "PREPARED_SLOT_COUNT_CHANGED")
    by_key = {(row["bank"], row["slot_id"]): row for row in source_rows}
    require(len(by_key) == 240, "PREPARED_SLOT_DUPLICATE")

    records = []
    for bank in BANKS:
        for slot_id in sorted(stage_by_id):
            slot = stage_by_id[slot_id]
            row = by_key[(bank, slot_id)]
            require(row["frozen_slot"] == slot, f"SLOT_BINDING_CHANGED:{bank}:{slot_id}")
            digest = hashlib.sha256(canonical(slot).encode("utf-8")).hexdigest()
            frozen_rng = rng[slot_id][bank]
            require(frozen_rng["non_rng_metadata_sha256"] == digest,
                    f"SLOT_DIGEST_CHANGED:{bank}:{slot_id}")
            reference = train_by_id[slot["annotation_id"]]
            require(reference["class"] == slot["class"], f"REFERENCE_CLASS_MISMATCH:{slot_id}")
            records.append({
                "bank": bank,
                "class": slot["class"],
                "slot_id": slot_id,
                "bank_seed": int(frozen_rng["bank_seed"]),
                "sample_seed": int(frozen_rng["sample_seed"]),
                "tfidg_effective_seed": int(frozen_rng["sample_seed"] % (2 ** 32 - 1)),
                "conditioning_normal_id": slot["conditioning_normal_id"],
                "target_image_path": str((args.prepared / row["image_path"]).resolve()),
                "target_image_sha256": row["image_sha256"],
                "target_mask_path": str((args.prepared / row["mask_path"]).resolve()),
                "target_mask_sha256": row["mask_sha256"],
                "reference_annotation_id": reference["annotation_id"],
                "reference_source_group_id": reference["source_group_id"],
                "reference_image_path": str((args.prepared / reference["image_path"]).resolve()),
                "reference_image_sha256": reference["image_sha256"],
                "reference_mask_path": str((args.prepared / reference["mask_path"]).resolve()),
                "reference_mask_sha256": reference["mask_sha256"],
                "reference_mask_type": "COARSE_BBOX_MASK",
                "target_mask_type": "COARSE_BBOX_MASK",
                "frozen_slot": slot,
                "frozen_slot_sha256": digest,
            })
    require(Counter((r["bank"], r["class"]) for r in records) ==
            Counter({(b, c): 40 for b in BANKS for c in CLASSES}), "FORMAL_COVERAGE_CHANGED")
    return records, train_by_id


def audit_environment(args, paths, records, train_by_id):
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    environment = {
        "status": "PASS" if torch.cuda.is_available() else "CUDA_UNAVAILABLE",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_request": args.device,
        "packages": {name: package_version(name) for name in (
            "torch", "torchvision", "xformers", "numpy", "opencv-python-headless",
            "albumentations", "omegaconf", "pytorch-lightning", "timm", "transformers",
            "scikit-image", "scikit-learn", "geomloss", "kornia")},
        "official_environment_install_deviations": {
            "cmake==3.28.0": "omitted; unavailable exact release and not an inference runtime dependency",
            "share==1.0.4": "omitted; absent from public PyPI and not imported by run_inference.py",
            "opencv_triplicate": "resolved to opencv-python-headless==4.7.0.72 to avoid three cv2 wheels",
            "CUDA_wheel": "cu118 wheel; driver compatibility checked at runtime",
        },
        "training_performed": False,
        "optimizer_step_count": 0,
        "parameters_updated": False,
    }
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        environment["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "free_mib_at_audit": free // 2 ** 20,
            "total_mib": total // 2 ** 20,
        }
    save(args.output / "environment_audit.json", environment)

    weight_rows = {}
    for name in ("anydoor", "dinov2"):
        path = paths[name]
        print(f"[运行中] SHA256 {name}: {path} ({path.stat().st_size} bytes)", flush=True)
        weight_rows[name] = {
            "path_server_only": str(path), "size": path.stat().st_size,
            "sha256": sha256(path), "format": "PyTorch ZIP archive",
            "trained_on_plastic_bomo": False, "parameter_updates_this_experiment": 0,
        }
    save(args.output / "pretrained_weights.json", {
        "status": "FROZEN", "method": "TF-IDG", "training_free": True,
        "weights": weight_rows,
    })

    pools = {}
    for class_name in CLASSES:
        rows = [row for row in train_by_id.values() if row["class"] == class_name]
        pools[class_name] = [{
            "annotation_id": r["annotation_id"], "source_group_id": r["source_group_id"],
            "image_path_server_only": str((args.prepared / r["image_path"]).resolve()),
            "image_sha256": r["image_sha256"],
            "mask_path_server_only": str((args.prepared / r["mask_path"]).resolve()),
            "mask_sha256": r["mask_sha256"], "mask_type": "COARSE_BBOX_MASK",
            "split": "V2_TRAIN", "geometry": r["geometry"],
        } for r in sorted(rows, key=lambda x: x["annotation_id"])]
    selection = [{
        "bank": r["bank"], "slot_id": r["slot_id"], "class": r["class"],
        "reference_annotation_id": r["reference_annotation_id"],
        "policy": "same frozen TRAIN annotation identity carried by Stage20B slot",
    } for r in records]
    save(args.output / "reference_manifest.json", {
        "status": "FROZEN", "source_split": "Plastic_Bomo V2 TRAIN only",
        "test_images_read": 0, "test_GT_masks_read": 0,
        "pool_counts": {c: len(pools[c]) for c in CLASSES},
        "mask_semantics": "COARSE_BBOX_MASK; no dense human segmentation was available in the frozen V2 TRAIN assets",
        "pools": pools, "formal_selection": selection,
    })

    save(args.output / "generation_manifest.json", {
        "status": "AUDITED_NOT_GENERATED", "method": "TF-IDG", "planned_count": 240,
        "completed_count": 0, "records": [], "quality_selection": False,
        "best_of_n": False, "reroll_count": 0,
        "native_inference": {"DDIM_steps": 50, "CFG": 3.5, "eta": 0.0,
            "strength": 1.0, "start_step": 25, "adain_weight": 0.5,
            "energy_score": True, "adaptive_mask": True, "batch_size": 1,
            "seed_note": "Stage20B sample_seed identities retained; effective seed is modulo 2^32-1. TF-IDG latent trajectory is method-native and is not claimed identical to SD2."},
    })
    for name, header in (
        ("bankwise_metrics.csv", ["method", "bank", "cohort", "generated_count", "real_count", "KID", "KID_subset_std", "FID"]),
        ("pooled_metrics.csv", ["method", "banks", "cohort", "generated_count", "real_count", "KID", "KID_subset_std", "FID"]),
    ):
        if not (args.output / name).exists():
            with (args.output / name).open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream).writerow(header)
    if not (args.output / "REPORT.md").exists():
        (args.output / "REPORT.md").write_text(
            "# Plastic_Bomo TF-IDG baseline\n\nStatus: `AUDITED_NOT_GENERATED`. No model was trained or fine-tuned.\n",
            encoding="utf-8")
    save(args.output / "experiment_status.json", {
        "status": "AUDITED_NOT_GENERATED", "full_experiment_complete": False,
        "training_free": True, "optimizer_step_count": 0, "generated_count": 0,
        "smoke_passed": False, "metrics_computed": False,
    })
    (args.output / "MODEL_README.md").write_text(
        "# TF-IDG model card for this comparison\n\n"
        "TF-IDG is used as a training-free reference-guided diffusion baseline. The public AnyDoor and "
        "DINOv2 ViT-g weights are frozen. Plastic_Bomo contributes only V2 TRAIN reference defects and "
        "frozen normal/BBox slots. No MSDF component, detector, selection, fine-tuning, or test GT is used.\n\n"
        "Reference and target masks are frozen rectangular BBox masks (`COARSE_BBOX_MASK`), not dense "
        "human segmentations. Images and weights remain server-only.\n",
        encoding="utf-8")
    print("[完成] 环境、权重、Reference Pool 与240个冻结计划已审计。", flush=True)


def check_asset(path, expected):
    path = Path(path)
    require(path.is_file(), f"ASSET_MISSING:{path}")
    require(sha256(path) == expected, f"ASSET_HASH_CHANGED:{path}")


def load_runtime(args):
    os.environ["TFIDG_SAVE_MEMORY"] = "1"
    os.environ["TFIDG_DDIM_STEPS"] = "50"
    os.environ["TFIDG_ENERGY_SCORE"] = "1"
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    print("[运行中] 加载 TF-IDG AnyDoor + DINOv2（冻结参数）。", flush=True)
    import run_inference as runtime
    runtime.model.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    return runtime


def mask_bounds(mask):
    import numpy as np
    yy, xx = np.where(mask > 0)
    require(len(xx) > 0, "EMPTY_MASK")
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def diagnostics(reference_image, reference_mask, target_image, target_mask, generated):
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity
    diff = np.abs(generated.astype(np.int16) - target_image.astype(np.int16)).max(axis=2)
    target = target_mask > 0
    changed = diff > 2
    outside = ~target
    yy, xx = np.where(changed)
    centroid = [float(xx.mean()), float(yy.mean())] if len(xx) else None
    bbox = mask_bounds(target_mask)
    x0, y0, x1, y1 = bbox
    generated_roi = generated[y0:y1, x0:x1]
    ref_box = mask_bounds(reference_mask)
    rx0, ry0, rx1, ry1 = ref_box
    ref_roi = reference_image[ry0:ry1, rx0:rx1]
    resized = cv2.resize(ref_roi, (max(1, x1-x0), max(1, y1-y0)), interpolation=cv2.INTER_AREA)
    copy_ssim = float(structural_similarity(generated_roi, resized, channel_axis=2, data_range=255))
    bbox_center = [(x0+x1-1)/2.0, (y0+y1-1)/2.0]
    shift = (math.dist(centroid, bbox_center) if centroid else None)
    return {
        "target_mask_bounds_xyxy": bbox,
        "changed_pixel_count": int(changed.sum()),
        "outside_bbox_changed_pixel_count": int((changed & outside).sum()),
        "outside_bbox_changed_fraction": float((changed & outside).sum() / max(1, outside.sum())),
        "outside_bbox_mean_abs_rgb": float(np.abs(generated.astype(np.float32)-target_image).mean(axis=2)[outside].mean()),
        "change_centroid_xy": centroid,
        "change_centroid_shift_from_bbox_center_px": shift,
        "reference_copy_similarity_ssim_resized_bbox": copy_ssim,
        "background_pollution_flag": bool((changed & outside).sum() / max(1, outside.sum()) > 0.01),
    }


def generate_one(args, runtime, row, scope):
    import gc
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    for key in ("target_image", "target_mask", "reference_image", "reference_mask"):
        check_asset(row[key + "_path"], row[key + "_sha256"])
    with Image.open(row["reference_image_path"]) as image:
        reference_image = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(row["reference_mask_path"]) as image:
        reference_mask = (np.asarray(image.convert("L"), dtype=np.uint8) >= 128).astype(np.uint8)
    with Image.open(row["target_image_path"]) as image:
        target_image = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(row["target_mask_path"]) as image:
        target_mask = (np.asarray(image.convert("L"), dtype=np.uint8) >= 128).astype(np.uint8)
    require(reference_image.shape == target_image.shape == (512, 512, 3), "CROP_SIZE_CHANGED")
    require(reference_mask.shape == target_mask.shape == (512, 512), "MASK_SIZE_CHANGED")

    seed = row["tfidg_effective_seed"]
    os.environ["TFIDG_SAMPLE_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    # TF-IDG is parameter-training-free but its energy guidance differentiates
    # with respect to the current latent. Do not wrap this call in no_grad or
    # inference_mode; all model parameters remain requires_grad=False.
    generated, _ = runtime.inference_single_image(
        reference_image, reference_mask, target_image.copy(), target_mask)
    elapsed = time.monotonic() - started
    generated = np.clip(generated, 0, 255).round().astype(np.uint8)
    require(generated.shape == (512, 512, 3), "TFIDG_OUTPUT_SHAPE_INVALID")
    require(np.isfinite(generated).all(), "TFIDG_OUTPUT_NONFINITE")

    slot = row["frozen_slot"]
    with Image.open(slot["normal_path"]) as original:
        full = original.convert("RGB")
    require(list(full.size) == slot["geometry"]["final_canvas_wh"], "NORMAL_CANVAS_CHANGED")
    crop_x, crop_y, _, _ = slot["geometry"]["fixed_crop_xyxy"]
    left, top, right, bottom = slot["geometry"]["paste_valid_crop_region_xyxy"]
    patch = Image.fromarray(generated, "RGB")
    full.paste(patch.crop((left-crop_x, top-crop_y, right-crop_x, bottom-crop_y)), (left, top))

    destination = args.output / scope / row["bank"] / row["class"]
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / (row["slot_id"] + ".png")
    mask_path = destination / (row["slot_id"] + "_mask.png")
    full.save(image_path, format="PNG")
    full_mask = Image.new("L", full.size, 0)
    bbox = slot["geometry"]["original_bbox_xyxy"]
    x0, y0, x1, y1 = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1])), min(full.width, math.ceil(bbox[2])), min(full.height, math.ceil(bbox[3]))
    full_mask.paste(255, (x0, y0, x1, y1))
    full_mask.save(mask_path, format="PNG")

    diag = diagnostics(reference_image, reference_mask, target_image, target_mask, generated)
    result = {
        **row,
        "status": "COMPLETE",
        "output_image_path_server_only": str(image_path.resolve()),
        "output_image_sha256": sha256(image_path),
        "output_mask_path_server_only": str(mask_path.resolve()),
        "output_mask_sha256": sha256(mask_path),
        "output_mode": "RGB", "output_canvas_wh": list(full.size),
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() // 2 ** 20),
        "peak_cuda_reserved_mib": int(torch.cuda.max_memory_reserved() // 2 ** 20),
        "diagnostics_crop512": diag,
        "quality_selection": False, "best_of_n": 1, "reroll_count": 0,
        "latent_identity_claim": "METHOD_NATIVE_SEEDED_RUN; not identical to Raw SD2 latent noise",
    }
    # Long 240-image jobs otherwise retain allocator fragments near the former
    # 60% cap.  Collection is an algorithm-neutral batch=1 memory optimization.
    del generated, reference_image, reference_mask, target_image, target_mask, full, full_mask, patch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return result


def run_smoke(args, records):
    status = load(args.output / "experiment_status.json")
    require(status["status"] in ("AUDITED_NOT_GENERATED", "SMOKE_FAILED", "SMOKE_PASS"),
            "AUDIT_REQUIRED_BEFORE_SMOKE")
    chosen = [next(r for r in records if r["bank"] == "A" and r["class"] == class_name)
              for class_name in CLASSES]
    runtime = load_runtime(args)
    outputs = []
    try:
        for index, row in enumerate(chosen, 1):
            print(f"[运行中] smoke {index}/2 class={row['class']} slot={row['slot_id']}", flush=True)
            outputs.append(generate_one(args, runtime, row, "smoke"))
        passed = all(not row["diagnostics_crop512"]["background_pollution_flag"] for row in outputs)
        passed = passed and all(row["diagnostics_crop512"]["target_mask_bounds_xyxy"] for row in outputs)
        save(args.output / "smoke_manifest.json", {
            "status": "PASS" if passed else "FAILED", "count": len(outputs), "records": outputs,
            "automatic_checks": "decode/RGB/canvas/mask-coordinate/outside-BBox change only; no quality ranking",
        })
        status.update(status="SMOKE_PASS" if passed else "SMOKE_FAILED", smoke_passed=passed,
                      smoke_count=len(outputs), smoke_records=[r["slot_id"] for r in outputs])
        save(args.output / "experiment_status.json", status)
        require(passed, "SMOKE_STRUCTURAL_CHECK_FAILED")
        print("[完成] Black/Flash 各1张 smoke 已生成并通过自动结构检查。", flush=True)
    finally:
        del runtime


def formal_existing(args):
    path = args.output / "generation_manifest.json"
    if not path.exists():
        return []
    value = load(path)
    return value.get("records", []) if value.get("status") in ("RUNNING", "COMPLETE") else []


def run_generation(args, records):
    status = load(args.output / "experiment_status.json")
    require(status.get("smoke_passed"), "SMOKE_PASS_REQUIRED_BEFORE_FORMAL_GENERATION")
    existing = formal_existing(args) if args.resume else []
    require(args.resume or not existing, "FORMAL_OUTPUT_EXISTS_USE_RESUME")
    completed = {(r["bank"], r["slot_id"]): r for r in existing}
    runtime = load_runtime(args)
    try:
        for index, row in enumerate(records, 1):
            key = (row["bank"], row["slot_id"])
            if key in completed:
                check_asset(completed[key]["output_image_path_server_only"], completed[key]["output_image_sha256"])
                continue
            print(f"[运行中] formal {index}/240 bank={row['bank']} class={row['class']} slot={row['slot_id']}", flush=True)
            result = generate_one(args, runtime, row, "images")
            completed[key] = result
            ordered = [completed[(r["bank"], r["slot_id"])] for r in records
                       if (r["bank"], r["slot_id"]) in completed]
            save(args.output / "generation_manifest.json", {
                "status": "COMPLETE" if len(ordered) == 240 else "RUNNING",
                "method": "TF-IDG", "planned_count": 240, "completed_count": len(ordered),
                "records": ordered, "quality_selection": False, "best_of_n": False,
                "reroll_count": 0, "DDIM_steps": 50, "CFG": 3.5,
            })
            status.update(status="GENERATING", generated_count=len(ordered), current_slot=row["slot_id"])
            save(args.output / "experiment_status.json", status)
        require(len(completed) == 240, "FORMAL_GENERATION_INCOMPLETE")
        status.update(status="GENERATION_COMPLETE", generated_count=240, current_slot=None)
        save(args.output / "experiment_status.json", status)
        print("[完成] TF-IDG Bank A/B/C 240/240 生成完成。", flush=True)
    finally:
        del runtime


def extract_tfidg_features(args, device, torch, lib, manifest):
    import numpy as np
    from PIL import Image

    cache = args.output / "tfidg_inception_features.npz"
    ledger_path = args.output / "tfidg_inception_feature_cache.json"
    keys = [f"tfidg:{row['bank']}:{row['class']}:{row['slot_id']}" for row in manifest["records"]]
    if cache.is_file() and ledger_path.is_file():
        ledger = load(ledger_path)
        require(ledger["keys"] == keys and ledger["sha256"] == sha256(cache),
                "TFIDG_FEATURE_CACHE_BINDING_CHANGED")
        with np.load(cache, allow_pickle=False) as archive:
            values = archive["features"].copy()
    else:
        model = lib["Inception"](
            "inception-v3-compat", ["2048"],
            feature_extractor_weights_path=str(lib["inception_weight"])
        ).float().eval().requires_grad_(False).to(device)
        values = []
        with torch.inference_mode():
            for index, row in enumerate(manifest["records"]):
                image_path = Path(row["output_image_path_server_only"])
                check_asset(image_path, row["output_image_sha256"])
                bbox = row["frozen_slot"]["bbox"]
                width, height = row["output_canvas_wh"]
                cx, cy, bw, bh = bbox
                crop = (
                    max(0, math.floor((cx - .75 * bw) * width)),
                    max(0, math.floor((cy - .75 * bh) * height)),
                    min(width, math.ceil((cx + .75 * bw) * width)),
                    min(height, math.ceil((cy + .75 * bh) * height)),
                )
                require(crop[2] > crop[0] and crop[3] > crop[1], "INVALID_ROI")
                with Image.open(image_path) as source:
                    roi = source.convert("RGB").crop(crop)
                tensor = torch.from_numpy(np.asarray(roi, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
                feature = model(tensor)[0]
                require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()),
                        row["slot_id"])
                values.append(feature[0].double().cpu().numpy())
                if (index + 1) % 20 == 0:
                    print(f"[Inception] {index + 1}/240", flush=True)
        values = np.stack(values)
        np.savez_compressed(cache, features=values)
        save(ledger_path, {"status": "COMPLETE", "keys": keys, "shape": [240, 2048],
            "stored_dtype": str(values.dtype), "extraction_dtype": "float32",
            "sha256": sha256(cache), "image_sha256_checks": 240,
            "ROI": "normalized bbox expanded 25% per side; floor/ceil; clipped"})
        del model
        torch.cuda.empty_cache()
    require(values.shape == (240, 2048), "TFIDG_FEATURE_MATRIX_INVALID")
    return {key: torch.from_numpy(values[index]).double() for index, key in enumerate(keys)}


def write_report(args, bankwise, pooled, comparison, manifest):
    diagnostics = [row["diagnostics_crop512"] for row in manifest["records"]]
    pollution = sum(bool(row["background_pollution_flag"]) for row in diagnostics)
    outside_mean = sum(row["outside_bbox_changed_fraction"] for row in diagnostics) / len(diagnostics)
    shift_mean = sum(row["change_centroid_shift_from_bbox_center_px"] for row in diagnostics) / len(diagnostics)
    copy_mean = sum(row["reference_copy_similarity_ssim_resized_bbox"] for row in diagnostics) / len(diagnostics)
    lines = [
        "# Plastic_Bomo TF-IDG baseline", "", "Status: `COMPLETE`", "",
        "TF-IDG was evaluated as a training-free method. AnyDoor and DINOv2 remained frozen; no Plastic_Bomo model, MSDF module, detector, or selector was trained.",
        "Reference and target masks are coarse frozen BBox masks. All 240 outputs were retained without Best-of-N, reroll, or quality filtering.",
        "", "## Bankwise ROI metrics", "",
        "| Bank | Cohort | KID ↓ | KID subset std | FID ↓ | Generated |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in bankwise:
        lines.append(f"| {row['bank']} | {row['cohort']} | {row['KID']:.10f} | {row['KID_subset_std']:.10f} | {row['FID']:.6f} | {row['generated_count']} |")
    lines += ["", "## Pooled ROI metrics (recomputed from concatenated features)", "",
              "| Banks | Cohort | KID ↓ | KID subset std | FID ↓ | Generated |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in pooled:
        lines.append(f"| {row['banks']} | {row['cohort']} | {row['KID']:.10f} | {row['KID_subset_std']:.10f} | {row['FID']:.6f} | {row['generated_count']} |")
    lines += [
        "", "## Diagnostics", "",
        f"- Mean changed-pixel fraction outside target BBox: `{outside_mean:.8f}`.",
        f"- Mean defect-change centroid shift: `{shift_mean:.4f}` crop pixels.",
        f"- Mean resized-BBox reference-copy SSIM: `{copy_mean:.6f}`.",
        f"- Background-pollution flags: `{pollution}/240`.",
        "", "## Comparison and protocol", "",
        "`comparison.csv` compares TF-IDG with frozen same-protocol Raw SD2, MSDF Full, and MSDF Max-Support metrics. Raw/MSDF are full-UNet fine-tuning routes; TF-IDG is training-free.",
        "B+C and A+B+C metrics were recomputed from concatenated Inception-2048 features and are not averages of bank metrics.",
        "KID uses 100 fixed-seed subsets (seed 42), subset size min(50, n_real, n_generated). Real references are the same 172 V2 TRAIN ROIs (Black 97, Flash 75).",
        "Images, pretrained weights, and feature caches remain server-only.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args):
    import torch
    import run_plastic_bomo_msdf_multibank_stability as x12
    import run_plastic_bomo_baseline_lora as lora_baseline
    from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime

    manifest = load(args.output / "generation_manifest.json")
    require(manifest.get("status") == "COMPLETE" and manifest.get("completed_count") == 240 and
            len(manifest.get("records", [])) == 240, "GENERATION_NOT_COMPLETE")
    require(Counter((row["bank"], row["class"]) for row in manifest["records"]) ==
            Counter({(bank, class_name): 40 for bank in BANKS for class_name in CLASSES}),
            "BANK_CLASS_COUNTS_CHANGED")
    status = load(args.output / "experiment_status.json")
    status.update(status="EVALUATING", metrics_computed=False)
    save(args.output / "experiment_status.json", status)

    lib, metric_audit = metric_runtime(args.device)
    features = extract_tfidg_features(args, args.device, torch, lib, manifest)
    gamma = x12.array_features(GAMMA / "gamma_sweep_inception_features.npz",
                               GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
    references = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
    real = {
        "Black": torch.stack([gamma["real:" + row["annotation_id"]] for row in references if row["class"] == "black"]),
        "Flash": torch.stack([gamma["real:" + row["annotation_id"]] for row in references if row["class"] == "flash"]),
        "Overall": torch.stack([gamma["real:" + row["annotation_id"]] for row in references]),
    }
    require({key: len(value) for key, value in real.items()} == {"Black": 97, "Flash": 75, "Overall": 172},
            "V2_TRAIN_REFERENCE_COUNTS_CHANGED")

    def fake(bank, cohort):
        selected = [row for row in manifest["records"] if row["bank"] == bank and
                    (cohort == "Overall" or row["class"] == cohort.lower())]
        return torch.stack([features[f"tfidg:{row['bank']}:{row['class']}:{row['slot_id']}"] for row in selected])

    bankwise = []
    for bank in BANKS:
        for cohort in ("Black", "Flash", "Overall"):
            result = x12.metric(real[cohort], fake(bank, cohort), lib)
            row = {"method": "TF-IDG", "bank": bank, "cohort": cohort,
                   "generated_count": result["generated_images"], "real_count": result["reference_ROIs"],
                   "KID": result["KID"], "KID_subset_std": result["KID_subset_std"], "FID": result["FID"],
                   "KID_subset_size": result["KID_subset_size"], "runtime_seconds": result["runtime_seconds"],
                   "metric_source": "new TF-IDG same-protocol V2 TRAIN ROI features"}
            bankwise.append(row)
            print(f"[指标] Bank {bank} {cohort} KID={row['KID']:.10f} FID={row['FID']:.6f}", flush=True)
    pooled = []
    for scope, banks in (("B+C", "BC"), ("A+B+C", "ABC")):
        for cohort in ("Black", "Flash", "Overall"):
            result = x12.metric(real[cohort], torch.cat([fake(bank, cohort) for bank in banks]), lib)
            row = {"method": "TF-IDG", "banks": scope, "cohort": cohort,
                   "generated_count": result["generated_images"], "real_count": result["reference_ROIs"],
                   "KID": result["KID"], "KID_subset_std": result["KID_subset_std"], "FID": result["FID"],
                   "KID_subset_size": result["KID_subset_size"], "runtime_seconds": result["runtime_seconds"],
                   "metric_source": "concatenated TF-IDG features; not bank average"}
            pooled.append(row)
            print(f"[Pooled] {scope} {cohort} KID={row['KID']:.10f} FID={row['FID']:.6f}", flush=True)
    write_csv(args.output / "bankwise_metrics.csv", bankwise)
    write_csv(args.output / "pooled_metrics.csv", pooled)

    comparison = lora_baseline.historical_comparison_rows()
    for row in bankwise:
        comparison.append({"scope": row["bank"], "population": row["cohort"], "method": "TF-IDG",
            "fine_tuning_type": "training-free; frozen AnyDoor and DINOv2",
            "KID": row["KID"], "KID_subset_std": row["KID_subset_std"], "FID": row["FID"],
            "reference_ROIs": row["real_count"], "generated_images": row["generated_count"],
            "KID_subset_size": row["KID_subset_size"], "metric_source": row["metric_source"]})
    for row in pooled:
        comparison.append({"scope": row["banks"], "population": row["cohort"], "method": "TF-IDG",
            "fine_tuning_type": "training-free; frozen AnyDoor and DINOv2",
            "KID": row["KID"], "KID_subset_std": row["KID_subset_std"], "FID": row["FID"],
            "reference_ROIs": row["real_count"], "generated_images": row["generated_count"],
            "KID_subset_size": row["KID_subset_size"], "metric_source": row["metric_source"]})
    scope_order = {"A": 0, "B": 1, "C": 2, "B+C": 3, "A+B+C": 4}
    population_order = {"Black": 0, "Flash": 1, "Overall": 2}
    method_order = {"Raw SD2": 0, "MSDF Full (Full Mean-Support)": 1, "MSDF Max-Support": 2, "TF-IDG": 3}
    comparison.sort(key=lambda row: (scope_order[row["scope"]], population_order[row["population"]],
                                     method_order[row["method"]]))
    require(len(comparison) == 60, f"COMPARISON_ROW_COUNT_CHANGED:{len(comparison)}")
    write_csv(args.output / "comparison.csv", comparison)
    write_report(args, bankwise, pooled, comparison, manifest)
    save(args.output / "metric_runtime.json", metric_audit)
    status.update(status="COMPLETE", full_experiment_complete=True, metrics_computed=True,
                  bankwise_metric_rows=9, pooled_metric_rows=6, comparison_rows=60,
                  pooled_recomputed_not_bank_average=True, generated_count=240,
                  completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save(args.output / "experiment_status.json", status)
    print("[完成] TF-IDG 240张生成、统一ROI KID/FID与历史对比全部完成。", flush=True)


def main():
    args = parse_args()
    args.source, args.prepared, args.p0, args.output = (p.resolve() for p in
                                                        (args.source, args.prepared, args.p0, args.output))
    print(f"[启动] TF-IDG phase={args.phase} output={args.output}", flush=True)
    try:
        paths = validate_inputs(args)
        records, train_by_id = build_records(args, paths)
        if args.phase in ("audit", "all"):
            audit_environment(args, paths, records, train_by_id)
        if args.phase in ("smoke", "all"):
            run_smoke(args, records)
        if args.phase in ("generate", "all"):
            run_generation(args, records)
        if args.phase in ("evaluate", "all"):
            run_evaluation(args)
        print("[已退出] 退出码=0", flush=True)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        save(args.output / "last_error.json", {
            "status": "FAILED", "phase": args.phase,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"[已退出] {type(exc).__name__}: {exc}；退出码=1", flush=True)
        raise


if __name__ == "__main__":
    main()
