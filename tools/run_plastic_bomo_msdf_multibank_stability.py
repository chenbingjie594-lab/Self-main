"""Experiment 12: frozen MSDF BalancedRegion multi-bank stability.

This runner has no training or calibration entry point.  It verifies the
frozen checkpoints and Bank-A artifacts, generates only the missing B/C
images, and evaluates bankwise and pooled ROI Inception metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

BASE_COMMIT = "59a7b70436a5cd03fe7939fa8be758c57942d33b"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_multibank_stability"
RAW = REPO / "results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"
STAGE25 = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214"
RUNTIME = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/server_runtime_20261013_080546_552698"
P0 = RUNTIME / "p0_view"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
EXP7 = REPO / "results_for_gpt/plastic_bomo_msdf_raw_matched_comparison"
EXP10 = REPO / "results_for_gpt/plastic_bomo_msdf_black_gradient_background"
EXP11 = REPO / "results_for_gpt/plastic_bomo_msdf_black_balanced_region"
BANK_SEEDS = {"A": 42, "B": 3407, "C": 2026}
GROUPS = {
    "raw": "Raw SD2",
    "original": "Original Class-Adaptive MSDF",
    "background": "BackgroundOnly Hybrid",
    "balanced": "BalancedRegion Hybrid",
}
NEW_GROUPS = {
    "flash_gamma0": {"class": "flash", "root": STAGE25,
                     "binding": "msdf_checkpoint_binding.json", "gamma": 0.0},
    "background_only": {"class": "black", "root": EXP10,
                        "binding": "background_only_checkpoint_binding.json", "gamma": 1.0},
    "balanced_region": {"class": "black", "root": EXP11,
                        "binding": "balanced_region_checkpoint_binding.json", "gamma": 1.0},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows):
    require(rows, "cannot write empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def crop_box(bbox, width, height, margin=0.25):
    cx, cy, bw, bh = bbox
    box = [max(0, math.floor((cx - (0.5 + margin) * bw) * width)),
           max(0, math.floor((cy - (0.5 + margin) * bh) * height)),
           min(width, math.ceil((cx + (0.5 + margin) * bw) * width)),
           min(height, math.ceil((cy + (0.5 + margin) * bh) * height))]
    require(box[2] > box[0] and box[3] > box[1], "invalid ROI")
    return box


def source_hashes():
    names = [
        "tools/run_plastic_bomo_msdf_multibank_stability.py",
        "tools/run_plastic_bomo_msdf_black_balanced_region.py",
        "tools/run_plastic_bomo_msdf_black_gradient_background.py",
        "tools/stage25_msdf_generator.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py",
    ]
    return {name: sha(REPO / name) for name in names}


def update_status(**values):
    path = OUTPUT / "multibank_status.json"
    row = load(path); row.update(values); save(path, row)


@contextmanager
def gpu_runtime(device):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden, set_gpu_runtime
    with network_forbidden():
        import torch
        set_gpu_runtime(torch, device)
        require("RTX 3090" in torch.cuda.get_device_name(device), "an RTX 3090 is required")
        require(torch.__version__ == "2.5.1+cu124" and torch.version.cuda == "12.4",
                "Torch/CUDA runtime changed")
        yield torch


def checkpoint_entry(root, binding, class_name):
    if "classes" in binding:
        row = binding["classes"][class_name]
    else:
        require(binding["class"] == class_name, "checkpoint class changed")
        row = binding
    require(binding["status"] == "FROZEN" and row["successful_updates"] == 2000,
            "checkpoint is not frozen at 2000 updates")
    checkpoint = root / row["checkpoint"]["path"]
    expected = row["checkpoint"]["sha256"]
    for name, digest in expected.items():
        require((checkpoint / name).is_file() and sha(checkpoint / name) == digest,
                f"checkpoint file changed: {checkpoint / name}")
    # Match stage20b_formal_common.tree_binding exactly.  The frozen ledger
    # hashes canonical JSON, rather than an ad-hoc textual file list.
    tree = hashlib.sha256(canonical(expected).encode()).hexdigest()
    require(tree == row["checkpoint"]["tree_sha256"], "checkpoint tree SHA changed")
    return checkpoint, row


def manifest_records(root, name, class_name=None):
    value = load(root / name)
    require(value["status"] == "COMPLETE", f"incomplete manifest: {name}")
    rows = value["records"]
    if class_name:
        rows = [row for row in rows if row["class"] == class_name]
    return rows


def verify_images(root, rows):
    for row in rows:
        path = root / row["image_path"]
        require(path.is_file() and sha(path) == row["image_sha256"], f"image changed: {path}")
    return len(rows)


def bank_a_regression():
    black = {row["Method"]: row for row in csv.DictReader(
        (EXP11 / "balanced_region_black_metrics.csv").open(encoding="utf-8"))}
    overall = {row["Method"]: row for row in csv.DictReader(
        (EXP11 / "balanced_region_overall_metrics.csv").open(encoding="utf-8"))}
    expected = {
        "Raw_Black_KID": 0.03797344467731272,
        "Background_Black_KID": 0.06100557551585336,
        "Balanced_Black_KID": 0.059381087604135144,
        "Raw_Overall_KID": 0.03522257098656576,
        "Raw_Overall_FID": 149.42750331396738,
        "Background_Overall_KID": 0.04294524729498305,
        "Background_Overall_FID": 154.90536831861988,
        "Balanced_Overall_KID": 0.04164533873626484,
        "Balanced_Overall_FID": 152.7822999488124,
    }
    actual = {
        "Raw_Black_KID": float(black["Raw SD2"]["Black_KID"]),
        "Background_Black_KID": float(black["MSDF-BackgroundOnly"]["Black_KID"]),
        "Balanced_Black_KID": float(black["MSDF-BalancedRegion"]["Black_KID"]),
        "Raw_Overall_KID": float(overall["Raw SD2"]["Overall_KID"]),
        "Raw_Overall_FID": float(overall["Raw SD2"]["Overall_FID"]),
        "Background_Overall_KID": float(overall["BackgroundOnly-Black Hybrid MSDF"]["Overall_KID"]),
        "Background_Overall_FID": float(overall["BackgroundOnly-Black Hybrid MSDF"]["Overall_FID"]),
        "Balanced_Overall_KID": float(overall["BalancedRegion-Black Hybrid MSDF"]["Overall_KID"]),
        "Balanced_Overall_FID": float(overall["BalancedRegion-Black Hybrid MSDF"]["Overall_FID"]),
    }
    require(actual == expected, "Bank A frozen metric regression failed")
    caches = [
        (GAMMA / "gamma_sweep_inception_features.npz", GAMMA / "gamma_sweep_inception_feature_cache.json"),
        (EXP7 / "raw_bank_A_inception_features.npz", EXP7 / "raw_bank_A_inception_feature_cache.json"),
        (EXP10 / "background_only_inception_features.npz", EXP10 / "background_only_inception_feature_cache.json"),
        (EXP11 / "balanced_region_inception_features.npz", EXP11 / "balanced_region_inception_feature_cache.json"),
    ]
    cache_audit = {}
    for data, ledger in caches:
        record = load(ledger); require(record["sha256"] == sha(data), f"cache changed: {data}")
        cache_audit[data.name] = {"sha256": record["sha256"], "shape": record.get("shape")}
    return {"status": "PASS", "reuse_mode": "SHA-bound Experiment 7/10/11 features and metrics",
            "expected": expected, "actual": actual, "feature_caches": cache_audit}


def audit():
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    (OUTPUT / "logs").mkdir(parents=True)
    for group in NEW_GROUPS:
        for bank in "BC":
            (OUTPUT / "images" / group / bank).mkdir(parents=True)
    save(OUTPUT / "multibank_status.json", {
        "status": "AUDITING", "created_at": now(), "base_commit": BASE_COMMIT,
        "training_count": 0, "calibration_count": 0, "generation_count": 0,
        "reroll_count": 0, "selection_count": 0, "final_eval_read_or_used": False,
        "BootstrapGuard": False, "detector_training_started": False,
    })
    raw_binding = load(RAW / "generator_checkpoint_binding.json")
    stage_binding = load(STAGE25 / "msdf_checkpoint_binding.json")
    bg_binding = load(EXP10 / "background_only_checkpoint_binding.json")
    br_binding = load(EXP11 / "balanced_region_checkpoint_binding.json")
    checkpoints = {}
    for key, root, binding, class_name in [
        ("raw_flash", RAW, raw_binding, "flash"), ("raw_black", RAW, raw_binding, "black"),
        ("stage25_flash", STAGE25, stage_binding, "flash"), ("stage25_black", STAGE25, stage_binding, "black"),
        ("background_black", EXP10, bg_binding, "black"), ("balanced_black", EXP11, br_binding, "black")]:
        _path, entry = checkpoint_entry(root, binding, class_name)
        checkpoints[key] = {"tree_sha256": entry["checkpoint"]["tree_sha256"],
                            "unet_sha256": entry["checkpoint"]["sha256"]["unet/diffusion_pytorch_model.safetensors"],
                            "msdf_sha256": entry["checkpoint"]["sha256"].get("msdf.pt"),
                            "successful_updates": entry["successful_updates"]}
    require(bg_binding["morphology_alignment_enabled"] is True and bg_binding["injection_gamma"] == 1.0,
            "BackgroundOnly inference controls changed")
    require(br_binding["morphology_alignment_enabled"] is True and br_binding["injection_gamma"] == 1.0,
            "BalancedRegion inference controls changed")
    require(br_binding["auxiliary_loss_weights"] == {
        "region": 0.0012344430664000732, "reconstruction": 0.0, "structure": 0.0,
        "background": 0.2, "scale": 0.0, "support": 0.0},
        "BalancedRegion training weights changed")
    require(load(EXP11 / "experiment_status.json")["status"] == "BALANCED_REGION_EXPLORATORY_IMPROVED",
            "Experiment 11 is incomplete")
    rng = load(P0 / "stage20b_rng_manifest.json")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    require(Counter(row["class"] for row in slots) == {"flash": 40, "black": 40}, "slot counts changed")
    rng_by_id = {row["slot_id"]: row["banks"] for row in rng["records"]}
    require(set(rng_by_id) == {row["slot_id"] for row in slots}, "RNG slot population changed")
    for bank, seed in BANK_SEEDS.items():
        require(all(rng_by_id[row["slot_id"]][bank]["bank_seed"] == seed for row in slots),
                f"Bank {bank} seed changed")
    protocol = load(P0 / "stage20b_generation_protocol.json")
    require(protocol["scheduler"] == "DDIMScheduler" and protocol["num_inference_steps"] == 50 and
            protocol["guidance_scale"] == 7.5 and protocol["eta"] == 0 and protocol["strength"] == 1 and
            protocol["height"] == protocol["width"] == 512 and protocol["precision"].startswith("fp16"),
            "generation protocol changed")
    existing = {}
    for bank in "ABC":
        raw_rows = manifest_records(RAW, f"formal_bank_{bank}_manifest.json")
        msdf_black = manifest_records(STAGE25, f"msdf_bank_{bank}_manifest.json", "black")
        require(len(raw_rows) == 80 and len(msdf_black) == 40, f"historical Bank {bank} incomplete")
        existing[f"raw_{bank}"] = verify_images(RAW, raw_rows)
        existing[f"original_black_{bank}"] = verify_images(STAGE25, msdf_black)
    gamma_a = [row for row in manifest_records(GAMMA, "gamma_0p00_manifest.json") if row["class"] == "flash"]
    bg_a = manifest_records(EXP10, "background_only_generation_manifest.json", "black")
    br_a = manifest_records(EXP11, "balanced_region_generation_manifest.json", "black")
    require(len(gamma_a) == len(bg_a) == len(br_a) == 40, "Bank A frozen groups incomplete")
    existing["flash_gamma0_A"] = verify_images(GAMMA, gamma_a)
    existing["background_A"] = verify_images(EXP10, bg_a)
    existing["balanced_A"] = verify_images(EXP11, br_a)
    regression = bank_a_regression()
    checkpoint_audit = {
        "status": "PASS", "base_commit": BASE_COMMIT, "checkpoints": checkpoints,
        "balanced_training_weights": br_binding["auxiliary_loss_weights"],
        "morphology_alignment_enabled": True,
        "inference_gamma": {"flash_gamma0": 0.0, "original_black": 1.0,
                            "background_black": 1.0, "balanced_black": 1.0},
        "historical_image_SHA256_checks": sum(existing.values()), "historical_groups": existing,
        "source_code_SHA256": source_hashes(), "completed_at": now(),
    }
    save(OUTPUT / "multibank_checkpoint_audit.json", checkpoint_audit)
    save(OUTPUT / "multibank_protocol.json", {
        "status": "FROZEN_BEFORE_NEW_GENERATION", "experiment": "Experiment 12 — MSDF BalancedRegion Multi-Bank Stability",
        "base_commit": BASE_COMMIT, "banks": BANK_SEEDS, "slots_per_bank": {"flash": 40, "black": 40},
        "groups": GROUPS, "shared_flash_for_G1_G2_G3": "Stage25 Flash checkpoint with gamma=0",
        "new_generation": {"banks": ["B", "C"], "groups": list(NEW_GROUPS), "maximum_images": 240},
        "generation_protocol": protocol, "evaluation": {
            "references": {"flash": 75, "black": 97, "overall": 172},
            "ROI": "bbox plus 25 percent context; frozen floor/ceil clipping",
            "Inception": "torch-fidelity inception-v3-compat 2048D; float32 extraction; float64 CPU metrics",
            "KID": {"subsets": 100, "seed": 42, "degree": 3,
                    "subset_size": "min(50,nreal,nfake)"},
        }, "Bank_A_regression": regression, "training_forbidden": True,
        "calibration_forbidden": True, "final_eval_read_or_used": False,
    })
    save(OUTPUT / "multibank_generation_manifest.json", {
        "status": "PENDING", "planned_count": 240, "records": [], "reroll_count": 0, "selection_count": 0})
    update_status(status="AUDIT_PASS", audit_completed_at=now(), Bank_A_regression="PASS")
    print("[审计通过] 冻结checkpoint、A/B/C历史图片、Bank seed、协议与Bank A指标全部通过。", flush=True)


def group_checkpoint(group):
    spec = NEW_GROUPS[group]
    binding = load(spec["root"] / spec["binding"])
    checkpoint, entry = checkpoint_entry(spec["root"], binding, spec["class"])
    return checkpoint, entry, spec


def generate(device):
    require(load(OUTPUT / "multibank_status.json")["status"] == "AUDIT_PASS", "generation requires audit PASS")
    require(not (OUTPUT / "logs/generate_claim.json").exists(), "generation already claimed")
    save(OUTPUT / "logs/generate_claim.json", {"started_at": now(), "planned_count": 240})
    update_status(status="GENERATING", generation_started_at=now())
    manifest_path = OUTPUT / "multibank_generation_manifest.json"
    records = []; started = time.monotonic()
    try:
        with gpu_runtime(device) as torch:
            import numpy as np
            from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
            from generate_plastic_bomo_stage20b_banks import sample
            from stage20b_formal_common import synthetic_label
            protocol = load(P0 / "stage20b_generation_protocol.json")
            slots = load(P0 / "stage20b_generation_slots.json")["slots"]
            originals = {row["slot_id"]: row for row in load(
                Path(load(STAGE25 / "stage25_run_binding.json")["p0"]) / "stage20b_generation_slots.json")["slots"]}
            rngs = {row["slot_id"]: row["banks"] for row in load(P0 / "stage20b_rng_manifest.json")["records"]}
            for group in ("flash_gamma0", "background_only", "balanced_region"):
                checkpoint, entry, spec = group_checkpoint(group)
                pipe = StableDiffusionInpaintPipeline.from_pretrained(str(checkpoint), local_files_only=True,
                    use_safetensors=True, torch_dtype=torch.float16, safety_checker=None,
                    feature_extractor=None, requires_safety_checker=False).to(device)
                pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
                pipe.disable_xformers_memory_efficient_attention()
                pipe.scheduler = DDIMScheduler.from_config(protocol["scheduler_config"])
                msdf = load_msdf_adapter(checkpoint / "msdf.pt", pipe.unet, device)
                require(msdf.ablation["morphology_alignment"] is True, f"{group} morphology disabled")
                msdf.set_injection_gamma(spec["gamma"])
                require(msdf.injection_gamma == spec["gamma"], f"{group} gamma restore failed")
                selected = [row for row in slots if row["class"] == spec["class"]]
                require(len(selected) == 40, f"{group} slot count changed")
                for bank in "BC":
                    for index, slot in enumerate(selected):
                        frozen = originals[slot["slot_id"]]; rng = rngs[slot["slot_id"]][bank]
                        require(rng["bank_seed"] == BANK_SEEDS[bank] and
                                rng["non_rng_metadata_sha256"] == hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                                f"frozen RNG/slot mismatch: {group}/{bank}/{slot['slot_id']}")
                        donor, mask = crop_and_mask({**slot, "path": slot["source_path"], "sha256": slot["source_sha256"]})
                        pixels = torch.from_numpy(np.asarray(donor).copy()).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
                        mask_t = torch.from_numpy(np.asarray(mask).copy())[None, None].to(device).float() / 255
                        with torch.inference_mode():
                            reference = pipe.vae.encode(pixels.half()).latent_dist.mode() * pipe.vae.config.scaling_factor

                        def prepare(_module, args, kwargs):
                            inp = args[0] if args else kwargs["sample"]
                            timestep = args[1] if len(args) > 1 else kwargs["timestep"]
                            require(list(inp.shape)[1:] == [9, 64, 64] and inp.shape[0] == 2, "invalid inference input")
                            msdf.prepare(reference.float(), mask_t, inp[:1, 4:5].float(), timestep,
                                         pipe.scheduler.config.num_train_timesteps, True, pixels)

                        handle = pipe.unet.register_forward_pre_hook(prepare, with_kwargs=True)
                        item_started = time.monotonic()
                        try:
                            image, trace = sample(pipe, slot, rng, protocol, device)
                        finally:
                            handle.remove()
                        image_path = OUTPUT / "images" / group / bank / (slot["slot_id"] + ".png")
                        label_path = image_path.with_suffix(".txt")
                        require(not image_path.exists() and not label_path.exists(), "new output would overwrite a file")
                        image.save(image_path, format="PNG")
                        label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                        width, height = frozen["geometry"]["final_canvas_wh"]
                        row = {
                            "group": group, "bank": bank, "class": spec["class"], "slot_id": slot["slot_id"],
                            "frozen_slot": frozen, "runtime_slot": slot,
                            "sample_seed": rng["sample_seed"], "bank_seed": rng["bank_seed"],
                            "checkpoint_tree_SHA256": entry["checkpoint"]["tree_sha256"],
                            "msdf_checkpoint_SHA256": entry["checkpoint"]["sha256"]["msdf.pt"],
                            "morphology_alignment_enabled": True, "injection_gamma": spec["gamma"],
                            "image_path": image_path.relative_to(OUTPUT).as_posix(), "image_SHA256": sha(image_path),
                            "label_path": label_path.relative_to(OUTPUT).as_posix(), "label_SHA256": sha(label_path),
                            "roi_xyxy": crop_box(frozen["bbox"], width, height),
                            "quality_selection": False, "reroll": False,
                            "runtime_seconds": time.monotonic() - item_started, **trace,
                        }
                        records.append(row)
                        save(manifest_path, {"status": "RUNNING", "planned_count": 240,
                            "records": records, "reroll_count": 0, "selection_count": 0})
                        print(f"[生成] {group} Bank {bank} {index + 1}/40 total={len(records)}/240", flush=True)
                msdf.detach(); del msdf, pipe; gc.collect(); torch.cuda.empty_cache()
        require(len(records) == 240 and Counter((r["group"], r["bank"]) for r in records) ==
                Counter({(group, bank): 40 for group in NEW_GROUPS for bank in "BC"}),
                "new generation population incomplete")
        save(manifest_path, {"status": "COMPLETE", "count": 240, "records": records,
            "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
            "runtime_seconds": time.monotonic() - started})
        update_status(status="GENERATION_COMPLETE", generation_count=240,
                      generation_completed_at=now(), reroll_count=0, selection_count=0)
        print("[生成完成] Bank B/C 新增240/240；无重抽、筛选或替换。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "records": records,
                            "error": f"{type(error).__name__}: {error}", "reroll_count": 0})
        update_status(status="GENERATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise


def metric(real, fake, lib):
    real = real.double().cpu(); fake = fake.double().cpu()
    options = {"kid_subsets": 100, "kid_subset_size": min(50, len(real), len(fake)),
               "rng_seed": 42, "verbose": False, "kid_degree": 3,
               "kid_gamma": None, "kid_coef0": 1.0}
    started = time.monotonic()
    kid = lib["kid"](real, fake, **options)
    fid = lib["fid"](lib["fid_stats"](real), lib["fid_stats"](fake), verbose=False)
    return {"KID": float(kid["kernel_inception_distance_mean"]),
            "KID_subset_std": float(kid["kernel_inception_distance_std"]),
            "FID": float(fid["frechet_inception_distance"]),
            "reference_ROIs": len(real), "generated_images": len(fake),
            "KID_subset_size": options["kid_subset_size"], "runtime_seconds": time.monotonic() - started}


def array_features(path, ledger_path, torch):
    import numpy as np
    ledger = load(ledger_path); require(ledger["sha256"] == sha(path), f"feature cache changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["features"].copy()
    require(len(values) == len(ledger["keys"]), f"feature ledger mismatch: {path}")
    return {key: torch.from_numpy(values[i]).double() for i, key in enumerate(ledger["keys"])}


def historical_path(group, bank, row):
    if group == "raw":
        return RAW / row["image_path"]
    if group == "original_black":
        return STAGE25 / row["image_path"]
    raise KeyError(group)


def extract_new_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    model = lib["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
    items = []
    for bank in "BC":
        for row in manifest_records(RAW, f"formal_bank_{bank}_manifest.json"):
            width, height = row["frozen_slot"]["geometry"]["final_canvas_wh"]
            items.append((f"raw:{bank}:{row['slot_id']}", RAW / row["image_path"],
                          crop_box(row["frozen_slot"]["bbox"], width, height), row["image_sha256"]))
        for row in manifest_records(STAGE25, f"msdf_bank_{bank}_manifest.json", "black"):
            width, height = row["frozen_slot"]["geometry"]["final_canvas_wh"]
            items.append((f"original_black:{bank}:{row['slot_id']}", STAGE25 / row["image_path"],
                          crop_box(row["frozen_slot"]["bbox"], width, height), row["image_sha256"]))
    for row in manifest["records"]:
        items.append((f"{row['group']}:{row['bank']}:{row['slot_id']}", OUTPUT / row["image_path"],
                      row["roi_xyxy"], row["image_SHA256"]))
    require(len(items) == 480 and len({x[0] for x in items}) == 480, "feature input population changed")
    values, keys = [], []
    with torch.inference_mode():
        for index, (key, path, roi, digest) in enumerate(items):
            require(path.is_file() and sha(path) == digest, f"feature input changed: {path}")
            with Image.open(path) as source:
                crop = source.convert("RGB").crop(tuple(roi))
            tensor = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            value = model(tensor)[0]
            require(tuple(value.shape) == (1, 2048) and bool(torch.isfinite(value).all()), key)
            keys.append(key); values.append(value[0].double().cpu().numpy())
            if (index + 1) % 20 == 0:
                print(f"[Inception] {index + 1}/480", flush=True)
    data = np.stack(values); path = OUTPUT / "multibank_inception_features.npz"
    np.savez_compressed(path, features=data)
    save(OUTPUT / "multibank_inception_feature_cache.json", {
        "status": "COMPLETE", "keys": keys, "shape": [480, 2048],
        "stored_dtype": str(data.dtype), "extraction_dtype": "float32",
        "sha256": sha(path), "image_SHA256_checks": 480,
    })
    del model; torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[i]).double() for i, key in enumerate(keys)}


def bank_a_rows():
    black_src = {row["Method"]: row for row in csv.DictReader(
        (EXP11 / "balanced_region_black_metrics.csv").open(encoding="utf-8"))}
    overall_src = {row["Method"]: row for row in csv.DictReader(
        (EXP11 / "balanced_region_overall_metrics.csv").open(encoding="utf-8"))}
    classwise = list(csv.DictReader((EXP7 / "raw_matched_classwise.csv").open(encoding="utf-8")))
    raw_flash = next(row for row in classwise if row["method"] == "Raw SD2" and row["class"] == "Flash")
    shared_flash = next(row for row in classwise if row["method"] == "MSDF Class-Adaptive gamma" and row["class"] == "Flash")
    black_map = {"raw": "Raw SD2", "original": "Original MSDF",
                 "background": "MSDF-BackgroundOnly", "balanced": "MSDF-BalancedRegion"}
    overall_map = {"raw": "Raw SD2", "original": "Original Class-Adaptive MSDF",
                   "background": "BackgroundOnly-Black Hybrid MSDF",
                   "balanced": "BalancedRegion-Black Hybrid MSDF"}
    black, flash, overall = [], [], []
    for key, label in GROUPS.items():
        b = black_src[black_map[key]]
        black.append({"bank": "A", "group": key, "method": label,
                      "KID": float(b["Black_KID"]), "KID_subset_std": float(b["Black_KID_subset_std"]),
                      "FID": float(b["Black_FID"]), "generated_images": 40, "reference_ROIs": 97,
                      "KID_subset_size": 40, "runtime_seconds": float(b["metric_runtime_seconds"]),
                      "metric_source": "frozen SHA-verified Experiment 11 cache"})
        f = raw_flash if key == "raw" else shared_flash
        flash.append({"bank": "A", "group": key, "method": label,
                      "KID": float(f["KID"]), "KID_subset_std": float(f["KID_subset_std"]),
                      "FID": float(f["FID"]), "generated_images": 40, "reference_ROIs": 75,
                      "KID_subset_size": 40, "runtime_seconds": float(f["metric_runtime_seconds"]),
                      "metric_source": "frozen SHA-verified Experiment 7 cache"})
        o = overall_src[overall_map[key]]
        overall.append({"bank": "A", "group": key, "method": label,
                        "KID": float(o["Overall_KID"]), "KID_subset_std": float(o["Overall_KID_subset_std"]),
                        "FID": float(o["Overall_FID"]), "generated_images": 80, "reference_ROIs": 172,
                        "KID_subset_size": 50, "runtime_seconds": float(o["metric_runtime_seconds"]),
                        "metric_source": "frozen SHA-verified Experiment 11 cache"})
    return black, flash, overall


def evaluate(device):
    require(load(OUTPUT / "multibank_status.json")["status"] == "GENERATION_COMPLETE",
            "evaluation requires complete B/C generation")
    update_status(status="EVALUATING", evaluation_started_at=now())
    started = time.monotonic()
    try:
        with gpu_runtime(device) as torch:
            from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
            lib, runtime_audit = metric_runtime(device)
            manifest = load(OUTPUT / "multibank_generation_manifest.json")
            require(manifest["status"] == "COMPLETE" and manifest["count"] == 240, "manifest incomplete")
            new = extract_new_features(device, torch, lib, manifest)
            gamma = array_features(GAMMA / "gamma_sweep_inception_features.npz",
                                   GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
            raw_a = array_features(EXP7 / "raw_bank_A_inception_features.npz",
                                   EXP7 / "raw_bank_A_inception_feature_cache.json", torch)
            bg_a = array_features(EXP10 / "background_only_inception_features.npz",
                                  EXP10 / "background_only_inception_feature_cache.json", torch)
            br_a = array_features(EXP11 / "balanced_region_inception_features.npz",
                                  EXP11 / "balanced_region_inception_feature_cache.json", torch)
            references = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
            real_flash = torch.stack([gamma["real:" + r["annotation_id"]] for r in references if r["class"] == "flash"])
            real_black = torch.stack([gamma["real:" + r["annotation_id"]] for r in references if r["class"] == "black"])
            real_all = torch.stack([gamma["real:" + r["annotation_id"]] for r in references])
            slots = load(P0 / "stage20b_generation_slots.json")["slots"]
            by_class = {name: [r for r in slots if r["class"] == name] for name in ("flash", "black")}

            def features(group, bank, class_name):
                selected = by_class[class_name]
                if bank == "A":
                    if group == "raw":
                        return torch.stack([raw_a["raw:roi:A:" + r["slot_id"]] for r in selected])
                    if class_name == "flash":
                        return torch.stack([gamma["fake:0.00:" + r["slot_id"]] for r in selected])
                    if group == "original":
                        return torch.stack([gamma["fake:1.00:" + r["slot_id"]] for r in selected])
                    cache, prefix = (bg_a, "background_only:") if group == "background" else (br_a, "balanced_region:")
                    return torch.stack([cache[prefix + r["slot_id"]] for r in selected])
                if group == "raw":
                    return torch.stack([new[f"raw:{bank}:" + r["slot_id"]] for r in selected])
                if class_name == "flash":
                    return torch.stack([new[f"flash_gamma0:{bank}:" + r["slot_id"]] for r in selected])
                prefix = {"original": "original_black", "background": "background_only",
                          "balanced": "balanced_region"}[group]
                return torch.stack([new[f"{prefix}:{bank}:" + r["slot_id"]] for r in selected])

            black_rows, flash_rows, overall_rows = bank_a_rows()
            for bank in "BC":
                for group, label in GROUPS.items():
                    black_result = metric(real_black, features(group, bank, "black"), lib)
                    black_rows.append({"bank": bank, "group": group, "method": label, **black_result,
                                       "metric_source": "new same-protocol features"})
                    print(f"[指标] Bank {bank} {label} Black KID={black_result['KID']:.10f} FID={black_result['FID']:.6f}", flush=True)
                raw_flash_result = metric(real_flash, features("raw", bank, "flash"), lib)
                shared_flash_result = metric(real_flash, features("original", bank, "flash"), lib)
                for group, label in GROUPS.items():
                    result = raw_flash_result if group == "raw" else shared_flash_result
                    flash_rows.append({"bank": bank, "group": group, "method": label, **result,
                                       "metric_source": "new same-protocol features; G1/G2/G3 share Flash"})
                print(f"[指标] Bank {bank} Flash Raw/shared KID={raw_flash_result['KID']:.10f}/{shared_flash_result['KID']:.10f}", flush=True)
                for group, label in GROUPS.items():
                    combined = torch.cat([features(group, bank, "flash"), features(group, bank, "black")])
                    result = metric(real_all, combined, lib)
                    overall_rows.append({"bank": bank, "group": group, "method": label, **result,
                                         "metric_source": "new same-protocol features"})
                    print(f"[指标] Bank {bank} {label} Overall KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
            write_csv(OUTPUT / "multibank_bankwise_black.csv", black_rows)
            write_csv(OUTPUT / "multibank_bankwise_flash.csv", flash_rows)
            write_csv(OUTPUT / "multibank_bankwise_overall.csv", overall_rows)

            pooled_rows = []
            for scope, banks in (("A+B+C", "ABC"), ("B+C", "BC")):
                for group, label in GROUPS.items():
                    for class_name, real in (("Black", real_black), ("Flash", real_flash)):
                        fake = torch.cat([features(group, bank, class_name.lower()) for bank in banks])
                        result = metric(real, fake, lib)
                        pooled_rows.append({"scope": scope, "group": group, "method": label,
                                            "population": class_name, **result})
                        print(f"[Pooled] {scope} {label} {class_name} KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
                    fake_all = torch.cat([torch.cat([features(group, bank, "flash"),
                                                    features(group, bank, "black")]) for bank in banks])
                    result = metric(real_all, fake_all, lib)
                    pooled_rows.append({"scope": scope, "group": group, "method": label,
                                        "population": "Overall", **result})
                    print(f"[Pooled] {scope} {label} Overall KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
            write_csv(OUTPUT / "multibank_pooled_metrics.csv", pooled_rows)

        deltas = []
        def add_delta(scope, population, current, baseline):
            for metric_name in ("KID", "FID"):
                value = current[metric_name] - baseline[metric_name]
                deltas.append({"scope": scope, "population": population,
                    "comparison": current["group"] + " - " + baseline["group"], "metric": metric_name,
                    "current": current[metric_name], "baseline": baseline[metric_name],
                    "absolute_delta": value, "relative_percent": 100 * value / baseline[metric_name],
                    "improved": value < 0})
        for bank in "ABC":
            for population, rows in (("Black", black_rows), ("Flash", flash_rows), ("Overall", overall_rows)):
                data = {r["group"]: r for r in rows if r["bank"] == bank}
                add_delta(bank, population, data["balanced"], data["background"])
                add_delta(bank, population, data["balanced"], data["raw"])
                add_delta(bank, population, data["background"], data["raw"])
        for scope in ("A+B+C", "B+C"):
            for population in ("Black", "Flash", "Overall"):
                data = {r["group"]: r for r in pooled_rows if r["scope"] == scope and r["population"] == population}
                add_delta(scope, population, data["balanced"], data["background"])
                add_delta(scope, population, data["balanced"], data["raw"])
                add_delta(scope, population, data["background"], data["raw"])
        write_csv(OUTPUT / "multibank_pairwise_deltas.csv", deltas)

        def wins(rows, population, metric_name, banks="BC"):
            count = 0
            details = {}
            for bank in banks:
                data = {r["group"]: r for r in rows if r["bank"] == bank}
                delta = data["balanced"][metric_name] - data["background"][metric_name]
                details[bank] = delta; count += int(delta < 0)
            return {"wins": count, "banks": len(banks), "deltas": details}
        black_kid = wins(black_rows, "Black", "KID"); black_fid = wins(black_rows, "Black", "FID")
        overall_kid = wins(overall_rows, "Overall", "KID"); overall_fid = wins(overall_rows, "Overall", "FID")
        if all(row["wins"] == 2 for row in (black_kid, black_fid, overall_kid, overall_fid)):
            conclusion = "MULTIBANK_CONSISTENT_IMPROVEMENT"
        elif black_kid["wins"] == black_fid["wins"] == 0:
            conclusion = "MULTIBANK_NOT_CONFIRMED"
        else:
            conclusion = "MULTIBANK_MIXED"
        raw_wins = {}
        for population, rows in (("Black", black_rows), ("Overall", overall_rows)):
            raw_wins[population] = {}
            for metric_name in ("KID", "FID"):
                raw_wins[population][metric_name] = {bank: next(r for r in rows if r["bank"] == bank and r["group"] == "balanced")[metric_name] <
                    next(r for r in rows if r["bank"] == bank and r["group"] == "raw")[metric_name] for bank in "ABC"}
        summary = {
            "status": conclusion, "primary_comparison": "BalancedRegion minus BackgroundOnly",
            "B_C_wins": {"Black_KID": black_kid, "Black_FID": black_fid,
                         "Overall_KID": overall_kid, "Overall_FID": overall_fid},
            "BalancedRegion_better_than_Raw": raw_wins,
            "Bank_A_regression": load(OUTPUT / "multibank_protocol.json")["Bank_A_regression"],
            "pooled_is_recomputed_not_bank_average": True,
            "KID_subset_std_interpretation": "internal 100-subset variation; not across-bank uncertainty or a 95% CI",
            "research_boundary": "Three generation seeds over shared slots with V2 TRAIN references; not cross-dataset or independent-test validation.",
        }
        summary.update(pooled_interpretation(pooled_rows))
        save(OUTPUT / "multibank_stability_summary.json", summary)
        integrity = {
            "status": "PASS", "new_generation_records": 240, "new_image_SHA256_checks": 240,
            "bank_class_group_counts": dict(Counter(f"{r['group']}:{r['bank']}:{r['class']}" for r in manifest["records"])),
            "same_slot_normal_bbox_prompt_seed_scheduler_steps_CFG_crop_paste": True,
            "reroll_count": 0, "selection_count": 0, "feature_input_checks": 480,
            "feature_cache_SHA256": sha(OUTPUT / "multibank_inception_features.npz"),
            "Bank_A_regression": "PASS", "final_eval_read_or_used": False,
        }
        save(OUTPUT / "multibank_generation_integrity.json", integrity)
        protocol = load(OUTPUT / "multibank_protocol.json")
        protocol.update(status="COMPLETE", new_generation_count=240,
                        feature_extraction_count=480, conclusion=conclusion,
                        source_code_SHA256=source_hashes())
        save(OUTPUT / "multibank_protocol.json", protocol)
        create_report(conclusion, black_rows, flash_rows, overall_rows, pooled_rows, summary)
        update_status(status=conclusion, evaluation_completed=True, completed_at=now(),
                      new_generation_count=240, new_feature_count=480,
                      evaluation_runtime_seconds=time.monotonic() - started,
                      followup_started=False)
        print(f"[完成] Experiment 12 status={conclusion}；停止，不启动后续实验。", flush=True)
    except Exception as error:
        update_status(status="EVALUATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise


def create_report(conclusion, black_rows, flash_rows, overall_rows, pooled_rows, summary):
    lines = ["# Experiment 12 — MSDF BalancedRegion Multi-Bank Stability", "",
             f"Status: `{conclusion}`.", "",
             "No model was trained, continued, recalibrated or structurally modified. Bank A was reused; only missing Bank B/C Flash gamma=0, BackgroundOnly Black and BalancedRegion Black images were generated from frozen slots and RNG manifests.", "",
             "## Bankwise Black", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in black_rows:
        lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Bankwise Flash", "",
              "Original, BackgroundOnly and BalancedRegion hybrids share the exact same frozen Stage25 Flash gamma=0 images within each bank.", "",
              "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in flash_rows:
        lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Bankwise Overall", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in overall_rows:
        lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Pooled metrics", "",
              "Pooled metrics were recomputed from all generated features and are not averages of bankwise KID/FID.", "",
              "| Scope | Population | Method | Images | KID ↓ | FID ↓ |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in pooled_rows:
        lines.append(f"| {row['scope']} | {row['population']} | {row['method']} | {row['generated_images']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    w = summary["B_C_wins"]
    abc = summary["pooled_comparisons"]["A+B+C"]
    bc = summary["pooled_comparisons"]["B+C"]
    lines += ["", "## Stability finding", "",
              f"Across held-out generation Banks B/C, BalancedRegion beat BackgroundOnly in Black KID on {w['Black_KID']['wins']}/2 banks, Black FID on {w['Black_FID']['wins']}/2, Overall KID on {w['Overall_KID']['wins']}/2, and Overall FID on {w['Overall_FID']['wins']}/2.", "",
              f"On A+B+C pooled features, BalancedRegion versus BackgroundOnly changed Black KID by {abc['Black']['balanced_minus_background']['KID']['absolute_delta']:+.10f} ({abc['Black']['balanced_minus_background']['KID']['relative_percent']:+.2f}%) and Black FID by {abc['Black']['balanced_minus_background']['FID']['absolute_delta']:+.6f} ({abc['Black']['balanced_minus_background']['FID']['relative_percent']:+.2f}%). Overall KID changed by {abc['Overall']['balanced_minus_background']['KID']['absolute_delta']:+.10f} ({abc['Overall']['balanced_minus_background']['KID']['relative_percent']:+.2f}%) and Overall FID by {abc['Overall']['balanced_minus_background']['FID']['absolute_delta']:+.6f} ({abc['Overall']['balanced_minus_background']['FID']['relative_percent']:+.2f}%).", "",
              f"When Bank A is excluded, B+C pooled BalancedRegion is worse than BackgroundOnly by {bc['Black']['balanced_minus_background']['KID']['absolute_delta']:+.10f} Black KID, {bc['Black']['balanced_minus_background']['FID']['absolute_delta']:+.6f} Black FID, {bc['Overall']['balanced_minus_background']['KID']['absolute_delta']:+.10f} Overall KID and {bc['Overall']['balanced_minus_background']['FID']['absolute_delta']:+.6f} Overall FID.", "",
              f"Against Raw SD2 on A+B+C pooled features, BalancedRegion remains higher by {abc['Black']['balanced_minus_raw']['KID']['absolute_delta']:+.10f} Black KID ({abc['Black']['balanced_minus_raw']['KID']['relative_percent']:+.2f}%), {abc['Black']['balanced_minus_raw']['FID']['absolute_delta']:+.6f} Black FID ({abc['Black']['balanced_minus_raw']['FID']['relative_percent']:+.2f}%), {abc['Overall']['balanced_minus_raw']['KID']['absolute_delta']:+.10f} Overall KID ({abc['Overall']['balanced_minus_raw']['KID']['relative_percent']:+.2f}%) and {abc['Overall']['balanced_minus_raw']['FID']['absolute_delta']:+.6f} Overall FID ({abc['Overall']['balanced_minus_raw']['FID']['relative_percent']:+.2f}%).", "",
              "All absolute and relative deltas, including comparisons with Raw SD2, are recorded in `multibank_pairwise_deltas.csv`. A negative delta means the first method is lower and better.", "",
              "## Recommendation", "", summary["recommendation"], "",
              "## Research boundary", "",
              "A/B/C change only the generation seed and share the same slots. The real reference remains V2 TRAIN and may overlap generator training. Multi-bank stability is not cross-dataset generalization or independent-test validation.", "",
              "KID subset standard deviation describes the internal 100-subset computation. It is not uncertainty across three banks and is not a 95% confidence interval.", "",
              "The experiment stops here. No final_eval, loss search, recalibration, retraining, Gamma change, architecture change, BootstrapGuard or detector training was started.", ""]
    (OUTPUT / "MULTIBANK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def pooled_interpretation(pooled_rows):
    comparisons = {}
    for scope in ("A+B+C", "B+C"):
        comparisons[scope] = {}
        for population in ("Black", "Overall"):
            rows = {row["group"]: row for row in pooled_rows
                    if row["scope"] == scope and row["population"] == population}
            comparisons[scope][population] = {}
            for name, baseline in (("balanced_minus_background", "background"),
                                   ("balanced_minus_raw", "raw")):
                comparisons[scope][population][name] = {}
                for metric_name in ("KID", "FID"):
                    current = float(rows["balanced"][metric_name])
                    reference = float(rows[baseline][metric_name])
                    delta = current - reference
                    comparisons[scope][population][name][metric_name] = {
                        "absolute_delta": delta,
                        "relative_percent": 100.0 * delta / reference,
                        "improved": delta < 0,
                    }
    return {
        "pooled_comparisons": comparisons,
        "recommendation": (
            "Do not promote BalancedRegion as a stable replacement or advance it alone to independent validation. "
            "Its Bank-A improvement did not reproduce in B/C: KID lost on both held-out banks, and B+C pooled "
            "Black/Overall KID and FID are all worse than BackgroundOnly. Preserve the frozen evidence and stop "
            "this Region-weight direction; a future authorized study should return to a training-level module "
            "ablation such as morphology guidance or dual-branch fusion before independent validation."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 12 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "audit": audit()
    elif args.phase == "generate": generate(args.device)
    else: evaluate(args.device)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
