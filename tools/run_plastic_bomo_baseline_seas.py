#!/usr/bin/env python3
"""Reproduce official SeaS on Plastic_Bomo with an auditable coarse-mask adapter.

SeaS remains an SD1.4 Separation-and-Sharing model.  The program refuses to
train before audit, data preparation, dataset preflight, and CUDA checks pass.
Images, weights, checkpoints, and prepared private data stay server-side.
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
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SEAS = ROOT.parent / "third_party/SeaS-main"
PREPARED = (ROOT / "results_for_gpt/plastic_bomo_chapter3_exp2_baseline_generation"
            / "prepared_v2_20261017_073039")
P0 = (ROOT / "results_for_gpt/plastic_bomo_stage20b_p0_preflight"
      / "server_preflight_20261008_084115")
OUT = ROOT / "results_for_gpt/plastic_bomo_baselines/seas"
RUNTIME = SEAS / "runtime/plastic_bomo"
OFFICIAL_COMMIT = "4645d3bda982855388bfc46fd1ceebc66b48c3f7"
CLASSES = ("black", "flash")
BANKS = ("A", "B", "C")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    require(rows, f"NO_ROWS:{path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def status_or_default(a):
    path = a.output / "experiment_status.json"
    return load(path) if path.is_file() else {
        "status": "AUDIT_PENDING", "full_experiment_complete": False,
        "formal_training_started": False, "generated_count": 0, "blockers": []}


def package(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "prepare", "preflight", "train-generation",
                        "train-mask", "infer", "evaluate", "all"), default="audit")
    parser.add_argument("--seas", type=Path, default=SEAS)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--p0", type=Path, default=P0)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="Physical GPU exposed to the one-process official scripts")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def inputs(a):
    return {
        "requirements": a.seas / "requirements.txt",
        "config": a.seas / "configs/seas.yaml",
        "generation_script": a.seas / "scripts/train_generation.sh",
        "mask_script": a.seas / "scripts/train_mask.sh",
        "infer_script": a.seas / "scripts/infer.sh",
        "training_manifest": a.prepared / "training_manifest.json",
        "generation_slots": a.prepared / "generation_slots.json",
        "prepared_binding": a.prepared / "prepared_binding.json",
        "stage20b_slots": a.p0 / "stage20b_generation_slots.json",
        "stage20b_rng": a.p0 / "stage20b_rng_manifest.json",
        "sd14": a.seas / "CompVis/stable-diffusion-v1-4",
    }


def validate(a):
    p = inputs(a)
    for name, path in p.items():
        require(path.is_file() if name != "sd14" else path.is_dir(), f"MISSING:{name}:{path}")
    binding = load(p["prepared_binding"])
    require(binding["status"] == "FROZEN", "PREPARED_NOT_FROZEN")
    require(sha256(p["training_manifest"]) == binding["training_manifest_sha256"],
            "TRAINING_MANIFEST_CHANGED")
    require(sha256(p["generation_slots"]) == binding["generation_slots_sha256"],
            "GENERATION_SLOTS_CHANGED")
    return p


def audit(a):
    p = validate(a)
    a.output.mkdir(parents=True, exist_ok=True)
    import torch
    requirements = p["requirements"].read_text(encoding="utf-8").splitlines()
    hashes = {}
    for path in sorted(p["sd14"].rglob("*")):
        if path.is_file():
            hashes[path.relative_to(p["sd14"]).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    environment = {
        "status": "PASS" if torch.cuda.is_available() else "BLOCKED_CUDA_UNAVAILABLE",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "requested_device": a.device,
        "official_python": "3.9", "official_cuda": "11.8", "official_torch": "2.2.1",
        "packages": {n: package(n) for n in ("torch", "torchvision", "accelerate",
                    "bitsandbytes", "transformers", "safetensors", "numpy", "Pillow")},
        "requirements_sha256": sha256(p["requirements"]),
        "requirements_exact_lines": requirements,
    }
    if torch.cuda.is_available():
        device = torch.device(a.device)
        free, total = torch.cuda.mem_get_info(device)
        environment["gpu"] = {"name": torch.cuda.get_device_name(device),
                              "free_mib": free // 2**20, "total_mib": total // 2**20}
    save(a.output / "environment_audit.json", environment)
    save(a.output / "pretrained_weights.json", {
        "status": "FROZEN", "backbone": "CompVis stable-diffusion-v1-4 (SD1.4)",
        "backbone_replaced_with_sd2": False, "files": hashes,
    })
    source_hashes = {name: sha256(path) for name, path in p.items()
                     if path.is_file() and name in ("config", "generation_script", "mask_script", "infer_script")}
    save(a.output / "source_audit.json", {
        "official_repository": "HUST-SLOW/SeaS",
        "official_commit_pinned": OFFICIAL_COMMIT,
        "local_checkout_has_git_metadata": (a.seas / ".git").is_dir(),
        "local_files_match_policy": "critical files recorded by SHA256; archive has no .git metadata",
        "critical_file_sha256": source_hashes,
        "mechanism": {"preserved": True, "normal_tokens": 1, "anomaly_tokens_per_class": 4,
                      "DA_loss": True, "NA_loss": True, "RMP_mask_stage": True,
                      "joint_category_sharing": ["black", "flash"]},
    })
    status = {
        "status": environment["status"], "full_experiment_complete": False,
        "phase": "AUDIT", "formal_training_started": False, "generated_count": 0,
        "blockers": [] if torch.cuda.is_available() else ["CUDA_UNAVAILABLE_ON_CURRENT_NODE"],
    }
    save(a.output / "experiment_status.json", status)
    print(f"[audit] {environment['status']} SD1.4 files={len(hashes)}", flush=True)


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(a):
    p = validate(a)
    training = load(p["training_manifest"])
    rows = training["records"]
    require(training["status"] == "TRAIN_ONLY_ASSETS_PREPARED" and len(rows) == 172,
            "TRAIN_ASSET_COUNT_CHANGED")
    require(Counter(r["class"] for r in rows) == Counter(black=97, flash=75),
            "TRAIN_CLASS_COUNTS_CHANGED")
    data = a.runtime / "data"
    anomalies, masks, normals = data / "anomaly", data / "coarse_bbox_mask", data / "normal"
    manifest = []
    for class_index, class_name in enumerate(CLASSES):
        selected = sorted((r for r in rows if r["class"] == class_name),
                          key=lambda r: r["annotation_id"])
        # The official loader divides each category directory count by three and then reads
        # 000..N-1. Keep every unique TRAIN record in that prefix and add two audit-marked
        # duplicate tails so its unmodified indexing sees all N records.
        for index, row in enumerate(selected):
            source_image = a.prepared / row["image_path"]
            source_mask = a.prepared / row["mask_path"]
            require(not row.get("mask_is_human_segmentation", row["geometry"].get("mask_is_human_segmentation")),
                    "UNEXPECTED_PIXEL_GT_CLAIM")
            image_name, mask_name = f"{index:03}.png", f"{index:03}_mask.png"
            link_or_copy(source_image, anomalies / class_name / image_name)
            link_or_copy(source_mask, masks / class_name / mask_name)
            for duplicate in (1, 2):
                tail = index + duplicate * len(selected)
                link_or_copy(source_image, anomalies / class_name / f"{tail:03}.png")
                link_or_copy(source_mask, masks / class_name / f"{tail:03}_mask.png")
            manifest.append({"class": class_name, "class_index": class_index,
                "dataset_index": index, "annotation_id": row["annotation_id"],
                "source_split": "V2_TRAIN", "mask_semantics": "COARSE_BBOX_MASK",
                "mask_is_human_segmentation": False, "image_sha256": row["image_sha256"],
                "mask_sha256": row["mask_sha256"]})
    slots = load(p["generation_slots"])["methods"]["anomalydiffusion"]
    unique = {}
    for row in slots:
        unique.setdefault(row["conditioning_normal_id"], a.prepared / row["image_path"])
    require(len(unique) >= 60, "INSUFFICIENT_TRAIN_NORMALS")
    normal_manifest = []
    for index, (identity, path) in enumerate(sorted(unique.items())[:60]):
        target = normals / f"{index:03}.png"
        link_or_copy(path, target)
        normal_manifest.append({"index": index, "conditioning_normal_id": identity,
                                "sha256": sha256(path), "source_split": "NORMAL_POOL"})
    model_link = a.seas / "model_hub/stable-diffusion-v1-4"
    model_link.parent.mkdir(parents=True, exist_ok=True)
    if not model_link.exists():
        model_link.symlink_to(p["sd14"], target_is_directory=True)
    save(a.output / "data_adapter_manifest.json", {
        "status": "PREPARED", "protocol": "COARSE_BBOX_MASK_ADAPTER_NOT_PAPER_PIXEL_GT",
        "test_defects_used": 0, "test_gt_masks_used": 0,
        "unique_anomaly_count": len(manifest), "loader_duplicate_tail_count": len(manifest) * 2,
        "class_counts": dict(Counter(r["class"] for r in manifest)),
        "normal_count": len(normal_manifest), "records": manifest, "normals": normal_manifest,
        "server_data_root": str(data.resolve()),
    })
    status = status_or_default(a)
    status.update(status="DATA_PREPARED", phase="PREPARE", data_prepared=True)
    save(a.output / "experiment_status.json", status)
    print("[prepare] unique anomalies=172 normals=60 masks=COARSE_BBOX_MASK", flush=True)


def preflight(a):
    import torch
    from PIL import Image
    data = a.runtime / "data"
    adapter = load(a.output / "data_adapter_manifest.json")
    require(adapter["status"] == "PREPARED", "RUN_PREPARE_FIRST")
    checks = []
    for class_name, expected in (("black", 97), ("flash", 75)):
        image_dir, mask_dir = data / "anomaly" / class_name, data / "coarse_bbox_mask" / class_name
        require(len(list(image_dir.glob("*.png"))) == expected * 3, f"{class_name}:IMAGE_LAYOUT")
        require(len(list(mask_dir.glob("*.png"))) == expected * 3, f"{class_name}:MASK_LAYOUT")
        with Image.open(image_dir / "000.png") as image, Image.open(mask_dir / "000_mask.png") as mask:
            require(image.convert("RGB").size == mask.convert("L").size == (512, 512), "SIZE")
            require(mask.convert("L").getbbox() is not None, "EMPTY_MASK")
            checks.append({"class": class_name, "image_mode": image.mode, "mask_mode": mask.mode,
                           "size": list(image.size), "mask_bbox": list(mask.convert("L").getbbox())})
    # Import the official dataset and fetch both category indices without loading a model.
    sys.path.insert(0, str(a.seas))
    from transformers import AutoTokenizer
    from datasets.seas_dataset import SeaSTrainDataset
    tokenizer = AutoTokenizer.from_pretrained(a.seas / "model_hub/stable-diffusion-v1-4/tokenizer",
                                              use_fast=False)
    dataset = SeaSTrainDataset(data / "anomaly", data / "coarse_bbox_mask", tokenizer,
        normal_data_root=data / "normal", normal_num=60, size=512, rotation=True,
        normal_token_num=1, anomaly_token_num=4)
    require(dataset.num_instance_images_list == {"black": 97, "flash": 75},
            f"OFFICIAL_LOADER_COUNTS:{dataset.num_instance_images_list}")
    black, flash = dataset[0], dataset[97]
    require(black["label"] == 0 and flash["label"] == 1, "CLASS_TOKEN_MAPPING")
    result = {"status": "PASS" if torch.cuda.is_available() else "DATA_PASS_CUDA_BLOCKED",
        "official_dataset_length": len(dataset), "category_counts": dataset.num_instance_images_list,
        "class_token_mapping": {"black": ["sks1", "sks2", "sks3", "sks4"],
                                "flash": ["sks5", "sks6", "sks7", "sks8"]},
        "normal_shared_token": "ob1", "checks": checks,
        "cuda_available": torch.cuda.is_available(),
        "formal_training_authorized": bool(torch.cuda.is_available())}
    save(a.output / "preflight.json", result)
    status = status_or_default(a)
    status.update(status=result["status"], phase="PREFLIGHT", preflight=result["status"])
    save(a.output / "experiment_status.json", status)
    print(f"[preflight] {result['status']} loader={dataset.num_instance_images_list}", flush=True)


def ensure_cuda(a):
    import torch
    require((a.output / "preflight.json").is_file(), "PREFLIGHT_REQUIRED")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE_REFUSING_FORMAL_STAGE")
    require(load(a.output / "preflight.json")["status"] == "PASS", "PREFLIGHT_NOT_PASS")


def run_logged(command, cwd, log_path, environment=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(map(str, command))}\n")
        log.flush()
        process = subprocess.run(command, cwd=cwd, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    return process.returncode, time.time() - started


def train(a, stage):
    ensure_cuda(a)
    data, checkpoint = a.runtime / "data", a.runtime / "checkpoints/plastic_bomo"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    common = ["--config", str(a.seas / "configs/seas.yaml"), "--output_dir", str(checkpoint),
              "--instance_data_dir", str(data / "anomaly"), "--mask_dir", str(data / "coarse_bbox_mask"),
              "--normal_data_dir", str(data / "normal"), "--checkpointing_steps", "1600"]
    if stage == "generation":
        entry = "examples/SeaS_main.py"; extra = ["--gen_train_steps", "1600"]
    else:
        require((checkpoint / "generation-checkpoint").is_dir(), "GENERATION_CHECKPOINT_REQUIRED")
        entry = "examples/SeaS_mask_main.py"; extra = ["--mask_train_steps", "1600",
                "--seas_trained_model_path", str(checkpoint / "generation-checkpoint")]
    command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
               "configs/accelerater_config.yaml", "--gpu_ids", str(a.gpu_index), entry] + common + extra
    status = load(a.output / "experiment_status.json")
    status.update(status=f"TRAINING_{stage.upper()}", phase=f"TRAIN_{stage.upper()}",
                  formal_training_started=True)
    save(a.output / "experiment_status.json", status)
    code, elapsed = run_logged(command, a.seas, a.output / f"logs/train_{stage}.log", env)
    record = {"stage": stage, "official_recipe_steps": 1600, "actual_exit_code": code,
              "elapsed_seconds": elapsed, "batch_size": 2, "gradient_accumulation": 2,
              "mixed_precision": "no", "optimizer": "AdamW8bit",
              "algorithm_neutral_oom_changes": [], "command": command}
    save(a.output / f"training_{stage}.json", record)
    require(code == 0, f"{stage.upper()}_FAILED_SEE_LOG")
    status.update(status=f"TRAIN_{stage.upper()}_COMPLETE")
    save(a.output / "experiment_status.json", status)


def infer(a):
    ensure_cuda(a)
    checkpoint = a.runtime / "checkpoints/plastic_bomo"
    rmp = checkpoint / "mask-checkpoint/rmp"
    require(rmp.is_file(), "MASK_CHECKPOINT_REQUIRED")
    source_rows = load(inputs(a)["generation_slots"])["methods"]["anomalydiffusion"]
    by_group = {(bank, cls): sorted((r for r in source_rows if r["bank"] == bank and
                r["class_name"] == cls), key=lambda r: r["slot_id"])
                for bank in BANKS for cls in CLASSES}
    require(all(len(v) == 40 for v in by_group.values()), "FROZEN_BANK_COVERAGE_CHANGED")
    import yaml
    from PIL import Image
    import numpy as np
    env = os.environ.copy(); env.pop("CUDA_VISIBLE_DEVICES", None)
    records = []
    manifest_path = a.output / "generation_manifest.json"
    if manifest_path.is_file() and a.resume:
        records = load(manifest_path).get("records", [])
    complete = {(r["bank"], r["class"], r["slot_id"]): r for r in records}
    base_cfg = yaml.safe_load((a.seas / "configs/seas.yaml").read_text(encoding="utf-8"))
    generation_root = a.runtime / "generated"
    infer_log = a.output / "logs/infer.log"
    for class_index, cls in enumerate(CLASSES):
        prompt_tokens = range(1 + class_index * 4, 5 + class_index * 4)
        prompt = "a ob1 with " + " ".join(f"sks{i}" for i in prompt_tokens)
        for bank_index, bank in enumerate(BANKS):
            rows = by_group[(bank, cls)]
            for chunk_start in range(0, 40, 10):
                chunk = rows[chunk_start:chunk_start + 10]
                if all((bank, cls, row["slot_id"]) in complete for row in chunk):
                    continue
                tag = f"{cls}_{bank}_{chunk_start:02d}"
                ref_dir = generation_root / "refs" / tag
                out_dir = generation_root / "chunks" / tag
                if out_dir.exists(): shutil.rmtree(out_dir)
                ref_dir.mkdir(parents=True, exist_ok=True)
                for old in ref_dir.glob("*.png"): old.unlink()
                for i, row in enumerate(chunk):
                    link_or_copy(a.prepared / row["image_path"], ref_dir / f"{i:03}.png")
                cfg = json.loads(json.dumps(base_cfg))
                cfg["inference"]["datasets"]["batch_size"] = 10
                native_seed_start = class_index * 120 + bank_index * 40 + chunk_start
                cfg["inference"]["seed_start"] = native_seed_start
                cfg["inference"]["device"] = "cuda:0"
                cfg_path = generation_root / "configs" / f"{tag}.yaml"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                command = [sys.executable, "examples/SeaS_infer.py", "--config", str(cfg_path),
                    "--output_dir", str(out_dir), "--ref_data_dir", str(ref_dir),
                    "--gen_model_path", str(checkpoint / "generation-checkpoint"),
                    "--rmp_model_path", str(rmp), "--prompt", prompt, "--total_infer_num", "10"]
                code, elapsed = run_logged(command, a.seas, infer_log, env | {"CUDA_VISIBLE_DEVICES": str(a.gpu_index)})
                require(code == 0, f"INFER_CHUNK_FAILED:{tag}:SEE_LOG")
                for i, row in enumerate(chunk):
                    source_image, source_mask = out_dir / "image" / f"{i:03}.png", out_dir / "mask" / f"{i:03}.png"
                    require(source_image.is_file() and source_mask.is_file(), f"MISSING_OUTPUT:{tag}:{i}")
                    final_dir = generation_root / "final" / bank / cls
                    final_dir.mkdir(parents=True, exist_ok=True)
                    image_path, mask_path = final_dir / f"{row['slot_id']}.png", final_dir / f"{row['slot_id']}_mask.png"
                    with Image.open(source_image) as image:
                        rgb = image.convert("RGB"); require(rgb.size == (512, 512), "OUTPUT_SIZE"); rgb.save(image_path)
                    with Image.open(source_mask) as mask:
                        gray = mask.convert("L"); require(gray.size == (512, 512), "MASK_SIZE"); gray.save(mask_path)
                        binary = np.asarray(gray) >= 128
                    target = np.zeros((512, 512), dtype=bool)
                    x0, y0, x1, y1 = row["frozen_slot"]["geometry"]["rasterized_mask_xyxy_exclusive"]
                    target[y0:y1, x0:x1] = True
                    inter, union = int((binary & target).sum()), int((binary | target).sum())
                    yy, xx = np.where(binary)
                    centroid = [float(xx.mean()), float(yy.mean())] if len(xx) else None
                    target_center = [(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2]
                    key = (bank, cls, row["slot_id"])
                    complete[key] = {"bank": bank, "class": cls, "slot_id": row["slot_id"],
                        "status": "COMPLETE", "output_mode": "RGB", "output_canvas_wh": [512, 512],
                        "output_image_path_server_only": str(image_path), "output_image_sha256": sha256(image_path),
                        "output_mask_path_server_only": str(mask_path), "output_mask_sha256": sha256(mask_path),
                        "native_seed": native_seed_start + i, "frozen_sample_seed_identity": row["seed"],
                        "latent_identity_claim": "SEAS_NATIVE_SEED; NOT IDENTICAL TO SD2",
                        "conditioning_normal_id": row["conditioning_normal_id"],
                        "conditioning_crop_path": str((a.prepared / row["image_path"]).resolve()),
                        "frozen_slot": row["frozen_slot"], "prompt": prompt,
                        "generation_mode": "NATIVE_GENERATION_NO_BBOX_CONTROL",
                        "mask_target_iou": inter / max(1, union), "generated_mask_pixels": int(binary.sum()),
                        "generated_mask_centroid_xy": centroid,
                        "defect_location_shift_px": math.dist(centroid, target_center) if centroid else None,
                        "quality_selection": False, "reroll_count": 0, "chunk_runtime_seconds": elapsed}
                ordered = [complete[(b, c, r["slot_id"])] for c in CLASSES for b in BANKS
                           for r in by_group[(b, c)] if (b, c, r["slot_id"]) in complete]
                save(manifest_path, {"status": "COMPLETE" if len(ordered) == 240 else "RUNNING",
                    "method": "SeaS", "mode": "NATIVE_GENERATION_NO_BBOX_CONTROL",
                    "planned_count": 240, "completed_count": len(ordered), "records": ordered,
                    "quality_selection": False, "best_of_n": False, "reroll_count": 0,
                    "native_parameters": {"backbone": "SD1.4", "steps": 25, "guidance_scale": 8,
                        "add_noise_step": 1500, "threshold": 0.2, "batch_size": 10}})
    require(len(complete) == 240, "GENERATION_INCOMPLETE")
    status = status_or_default(a); status.update(status="GENERATION_COMPLETE", phase="INFER", generated_count=240)
    save(a.output / "experiment_status.json", status)


def evaluate(a):
    import numpy as np
    import torch
    from PIL import Image
    import run_plastic_bomo_msdf_multibank_stability as x12
    import run_plastic_bomo_baseline_lora as lora
    from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
    manifest = load(a.output / "generation_manifest.json")
    require(manifest.get("status") == "COMPLETE" and manifest.get("completed_count") == 240,
            "GENERATION_INCOMPLETE")
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    lib, runtime_audit = metric_runtime(device)
    cache = a.runtime / "seas_inception_features.npz"
    audit_path = a.output / "feature_cache_audit.json"
    if cache.is_file() and audit_path.is_file():
        audit = load(audit_path); require(audit["cache_sha256"] == sha256(cache), "FEATURE_CACHE_CHANGED")
        matrix = np.load(cache, allow_pickle=False)["features"]
        keys = audit["keys"]
        require(matrix.shape == (240, 2048) and len(keys) == 240, "FEATURE_CACHE_SHAPE")
        print("[Inception] reuse verified 240x2048 cache", flush=True)
    else:
        inception = lib["Inception"]("inception-v3-compat", ["2048"],
            feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
        keys, arrays = [], []
        with torch.inference_mode():
          for index, row in enumerate(manifest["records"]):
            path = Path(row["output_image_path_server_only"])
            require(path.is_file() and sha256(path) == row["output_image_sha256"], "IMAGE_CHANGED")
            x0, y0, x1, y1 = row["frozen_slot"]["geometry"]["generator_bbox_xyxy"]
            width = max(1.0, x1 - x0); height = max(1.0, y1 - y0)
            crop = (max(0, math.floor(x0 - .25 * width)), max(0, math.floor(y0 - .25 * height)),
                    min(512, math.ceil(x1 + .25 * width)), min(512, math.ceil(y1 + .25 * height)))
            with Image.open(path) as image: roi = image.convert("RGB").crop(crop)
            tensor = torch.from_numpy(np.asarray(roi, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = inception(tensor)[0]
            require(tuple(feature.shape) == (1, 2048), "FEATURE_SHAPE")
            keys.append(f"seas:{row['bank']}:{row['class']}:{row['slot_id']}")
            arrays.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0: print(f"[Inception] {index + 1}/240", flush=True)
        matrix = np.stack(arrays); np.savez_compressed(cache, features=matrix)
        save(audit_path, {"status": "COMPLETE", "shape": [240, 2048],
            "cache_server_only": str(cache), "cache_sha256": sha256(cache), "keys": keys,
            "ROI": "frozen target generator BBox expanded 25 percent per side"})
        del inception; torch.cuda.empty_cache()
    features = {key: torch.from_numpy(matrix[i]).double() for i, key in enumerate(keys)}
    gamma_root = ROOT / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
    gamma = x12.array_features(gamma_root / "gamma_sweep_inception_features.npz",
        gamma_root / "gamma_sweep_inception_feature_cache.json", torch)
    references = load(gamma_root / "gamma_sweep_protocol.json")["development_reference"]["records"]
    real = {"Black": torch.stack([gamma["real:" + r["annotation_id"]] for r in references if r["class"] == "black"]),
            "Flash": torch.stack([gamma["real:" + r["annotation_id"]] for r in references if r["class"] == "flash"]),
            "Overall": torch.stack([gamma["real:" + r["annotation_id"]] for r in references])}
    def fake(banks, population):
        chosen = [r for r in manifest["records"] if r["bank"] in banks and
                  (population == "Overall" or r["class"] == population.lower())]
        return torch.stack([features[f"seas:{r['bank']}:{r['class']}:{r['slot_id']}"] for r in chosen])
    bankwise = []
    for bank in BANKS:
        for population in ("Black", "Flash", "Overall"):
            result = x12.metric(real[population], fake(bank, population), lib)
            bankwise.append({"method": "SeaS SD1.4", "bank": bank, "population": population, **result,
                "metric_source": "new SeaS native-generation frozen-target-ROI features"})
            print(f"[metric] {bank} {population} KID={result['KID']:.8f} FID={result['FID']:.4f}", flush=True)
    pooled = []
    for scope, banks in (("B+C", "BC"), ("A+B+C", "ABC")):
        for population in ("Black", "Flash", "Overall"):
            result = x12.metric(real[population], fake(banks, population), lib)
            pooled.append({"method": "SeaS SD1.4", "scope": scope, "population": population, **result,
                "metric_source": "concatenated SeaS features; not bank average"})
    write_csv(a.output / "bankwise_metrics.csv", bankwise)
    write_csv(a.output / "pooled_metrics.csv", pooled)
    comparison = lora.historical_comparison_rows()
    for row in bankwise:
        comparison.append({"scope": row["bank"], "population": row["population"], "method": "SeaS SD1.4",
            "fine_tuning_type": "SeaS Separation-and-Sharing full fine-tuning (SD1.4)",
            **{k: row[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
            "metric_source": row["metric_source"]})
    for row in pooled:
        comparison.append({"scope": row["scope"], "population": row["population"], "method": "SeaS SD1.4",
            "fine_tuning_type": "SeaS Separation-and-Sharing full fine-tuning (SD1.4)",
            **{k: row[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
            "metric_source": row["metric_source"]})
    write_csv(a.output / "comparison.csv", comparison)
    diagnostics = manifest["records"]
    summary = {"mean_mask_target_iou": sum(r["mask_target_iou"] for r in diagnostics) / 240,
        "mean_location_shift_px": sum(r["defect_location_shift_px"] for r in diagnostics if r["defect_location_shift_px"] is not None) /
                                  sum(r["defect_location_shift_px"] is not None for r in diagnostics),
        "empty_mask_count": sum(r["generated_mask_pixels"] == 0 for r in diagnostics)}
    save(a.output / "location_consistency.json", summary)
    status = status_or_default(a); status.update(status="COMPLETE", phase="EVALUATE",
        full_experiment_complete=True, metrics_computed=True, bankwise_rows=9, pooled_rows=6)
    save(a.output / "experiment_status.json", status)
    save(a.output / "metric_runtime_audit.json", json.loads(json.dumps(runtime_audit, default=str)))


def placeholders(a):
    for name, columns in {
        "bankwise_metrics.csv": ["method", "bank", "cohort", "generated_count", "real_count", "KID", "FID"],
        "pooled_metrics.csv": ["method", "banks", "cohort", "generated_count", "real_count", "KID", "FID"],
    }.items():
        path = a.output / name
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                csv.writer(stream).writerow(columns)
    report = a.output / "REPORT.md"
    if not report.exists():
        report.write_text("# Plastic_Bomo SeaS baseline\n\n"
            "Status: `AUDIT/PREFLIGHT IN PROGRESS`. SeaS uses its official SD1.4 backbone; "
            "MSDF uses SD2, so this is a method-level comparison, not a same-backbone ablation.\n\n"
            "Plastic_Bomo V2 provides bounding boxes rather than human pixel segmentation. "
            "This run is therefore explicitly a `COARSE_BBOX_MASK_ADAPTER`, not the paper's pixel-GT protocol.\n",
            encoding="utf-8")


def main():
    a = args_parser()
    a.seas, a.prepared, a.p0, a.output, a.runtime = map(Path.resolve,
        (a.seas, a.prepared, a.p0, a.output, a.runtime))
    a.output.mkdir(parents=True, exist_ok=True); placeholders(a)
    phases = [a.phase] if a.phase != "all" else ["audit", "prepare", "preflight",
              "train-generation", "train-mask", "infer", "evaluate"]
    for phase in phases:
        if phase == "audit": audit(a)
        elif phase == "prepare": prepare(a)
        elif phase == "preflight": preflight(a)
        elif phase == "train-generation": train(a, "generation")
        elif phase == "train-mask": train(a, "mask")
        elif phase == "infer": infer(a)
        elif phase == "evaluate": evaluate(a)


if __name__ == "__main__":
    main()
