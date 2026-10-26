"""Plastic_Bomo V2 SD2-Inpainting Attention-LoRA baseline.

This is a deliberately narrow runner.  It trains one standard PEFT LoRA for
each V2 TRAIN defect class, generates the frozen Stage20B A/B/C slots, and
evaluates the same expanded-ROI Inception features used by the historical
multi-bank experiments.  It has no MSDF code path and no hyperparameter search.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

OUTPUT = REPO / "results_for_gpt/plastic_bomo_baselines/lora"
P0 = REPO / "results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"
BASE_MODEL = REPO / "sd2-community/stable-diffusion-2-inpainting"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
MULTI = REPO / "results_for_gpt/plastic_bomo_msdf_multibank_stability"
MAX_SUPPORT = REPO / "results_for_gpt/plastic_bomo_msdf_max_support"

CLASSES = ("black", "flash")
BANKS = ("A", "B", "C")
CLASS_COUNTS = {"black": 97, "flash": 75}
PROMPTS = {"black": "a photo of a sks black spot defect", "flash": "a photo of a sks flash defect"}
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def htext(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows, fieldnames=None):
    require(rows, f"refusing to write empty CSV: {path}")
    fieldnames = fieldnames or list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def offline():
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )


def set_runtime(torch, device_index):
    require(torch.cuda.is_available(), "CUDA is unavailable")
    device = f"cuda:{device_index}"
    torch.cuda.set_device(device_index)
    require("RTX 3090" in torch.cuda.get_device_name(device_index), "frozen runtime requires RTX 3090")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return device


def status_update(**values):
    path = OUTPUT / "experiment_status.json"
    state = load(path) if path.is_file() else {}
    state.update(values)
    save(path, state)


def source_hashes():
    names = (
        "tools/run_plastic_bomo_baseline_lora.py",
        "tools/train_plastic_bomo_stage20b_generator.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "tools/run_plastic_bomo_msdf_multibank_stability.py",
        "tools/run_plastic_bomo_msdf_max_support.py",
        "tools/qualify_plastic_bomo_stage20b_r0.py",
        "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py",
        "diffusers/loaders/lora_pipeline.py",
    )
    return {name: sha(REPO / name) for name in names}


def protocol_document():
    raw_train = load(P0 / "stage20b_generator_training_protocol.json")
    generation = load(P0 / "stage20b_generation_protocol.json")
    return {
        "status": "FROZEN_BEFORE_TRAINING",
        "experiment": "Plastic_Bomo V2 SD2 Inpainting standard Attention-LoRA baseline",
        "scope": "LoRA only; no MSDF modification or execution",
        "base_model": "sd2-community/stable-diffusion-2-inpainting",
        "base_model_local_role": "offline immutable public checkpoint",
        "training": {
            "classes": list(CLASSES),
            "separate_class_adapters": True,
            "class_order": list(CLASSES),
            "dataset": "V2 TRAIN corresponding class only",
            "class_instance_counts": CLASS_COUNTS,
            "resolution": 512,
            "mask_processing": raw_train["mask_crop_policy"],
            "prompts": PROMPTS,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "fp16 autocast with FP32 trainable masters",
            "seed": 42,
            "successful_optimizer_updates_per_class": 2000,
            "attempted_updates_and_AMP_skips_recorded_separately": True,
            "learning_rate": 1e-4,
            "lr_scheduler": "constant; no warmup",
            "optimizer": {
                "name": "AdamW", "beta1": 0.9, "beta2": 0.999,
                "weight_decay": 0.01, "epsilon": 1e-8,
            },
            "max_gradient_norm": 1.0,
            "loss": "original unweighted diffusion denoising MSE only",
            "noise_offset": 0.0,
            "input_perturbation": 0.0,
            "snr_gamma": None,
            "augmentation": "none",
            "gradient_checkpointing": True,
            "xformers": False,
            "TF32": False,
            "automatic_resume": False,
            "hyperparameter_search": False,
            "runs_per_class": 1,
            "epoch_order": "annotation_id sort followed by deterministic permutation seed=42+epoch",
            "tail_accumulation": "historical generator divisor remains 4 for partial epoch groups",
        },
        "lora": {
            "implementation": "PEFT standard LoRA injected by diffusers UNet.add_adapter",
            "scope": "UNet Attention projections only",
            "target_modules": list(LORA_TARGETS),
            "rank": 16,
            "alpha": 16,
            "dropout": 0.0,
            "bias": "none",
            "init_lora_weights": "gaussian",
            "use_rslora": False,
            "use_dora": False,
            "adapter_name": "default",
            "base_UNet_frozen": True,
            "text_encoder_frozen": True,
            "VAE_frozen": True,
            "expected_UNet_in_channels": 9,
            "forbidden": ["MSDF", "extra Attention", "custom loss", "reference-image feature fusion"],
        },
        "generation": {
            "source_protocol_sha256": sha(P0 / "stage20b_generation_protocol.json"),
            "frozen_slots_sha256": sha(P0 / "stage20b_generation_slots.json"),
            "frozen_rng_manifest_sha256": sha(P0 / "stage20b_rng_manifest.json"),
            "banks": list(BANKS), "images_per_class_per_bank": 40, "total_images": 240,
            "scheduler": "DDIMScheduler", "steps": 50, "CFG": 7.5, "eta": 0.0,
            "strength": 1.0, "height": 512, "width": 512,
            "quality_selection": False, "best_of_n": False, "reroll": False,
            "all_successful_outputs_retained": True,
            "frozen_historical_protocol": generation,
        },
        "evaluation": {
            "reference": "same 172 V2 TRAIN expanded ROI features (Black=97, Flash=75)",
            "roi_margin_fraction_each_side": 0.25,
            "feature_extractor": "torch-fidelity inception-v3-compat 2048",
            "KID": {"subsets": 100, "subset_size": "min(50, real, generated)", "seed": 42,
                    "degree": 3, "gamma": None, "coef0": 1.0},
            "FID": "torch-fidelity feature statistics",
            "pooled": "concatenate image features then recompute; never average bank metrics",
            "historical_comparators": ["Raw SD2", "MSDF Full (Full Mean-Support)", "MSDF Max-Support"],
        },
        "privacy": {
            "weights_images_private_server_only": True,
            "GitHub_allowlist": ["tools/run_plastic_bomo_baseline_lora.py", "REPORT.md",
                "training_protocol.json", "checkpoint_audit.json", "generation_manifest.json",
                "bankwise_metrics.csv", "pooled_metrics.csv", "comparison.csv", "experiment_status.json"],
            "manifests_do_not_store_private_absolute_paths": True,
        },
        "source_code_SHA256": source_hashes(),
    }


def audit():
    require(not OUTPUT.exists(), f"output already exists; refusing a second run: {OUTPUT}")
    (OUTPUT / "logs").mkdir(parents=True)
    (OUTPUT / "checkpoints").mkdir()
    (OUTPUT / "images").mkdir()
    save(OUTPUT / "experiment_status.json", {
        "status": "AUDITING", "created_at": now(), "device": "cuda:0",
        "training_runs_started": 0, "training_runs_completed": 0,
        "successful_updates": {"black": 0, "flash": 0},
        "generated_images": 0, "quality_selection_count": 0, "reroll_count": 0,
        "MSDF_modified": False, "MSDF_executed": False,
        "final_eval_content_read_or_used": False, "github_submission": "PENDING",
    })
    training_protocol = protocol_document()
    save(OUTPUT / "training_protocol.json", training_protocol)

    base_binding = load(P0 / "sd2_base_weight_binding.json")
    require(base_binding["status"] == "PASS" and not base_binding["errors"], "historical base binding failed")
    core_names = (
        "model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors",
    )
    base_rows = {}
    for name in core_names:
        path = BASE_MODEL / name
        require(path.is_file(), f"missing base checkpoint file: {name}")
        actual = sha(path)
        expected = base_binding["files"][name]["sha256"]
        require(actual == expected, f"base checkpoint SHA256 mismatch: {name}")
        base_rows[name] = {"sha256": actual, "bytes": path.stat().st_size, "historical_binding_match": True}

    allowlist = load(P0 / "generator_train_allowlist.json")
    require(allowlist["status"] == "PASS", "training allowlist is not PASS")
    counts = Counter(row["class"] for row in allowlist["inputs"])
    require(counts == Counter(CLASS_COUNTS), f"V2 TRAIN class counts changed: {counts}")
    for row in allowlist["inputs"]:
        require(row["split"] == "train", "non-TRAIN sample in generator allowlist")
        require(Path(row["path"]).is_file() and Path(row["full_label_path"]).is_file(), "V2 TRAIN asset missing")

    slots_doc = load(P0 / "stage20b_generation_slots.json")
    rng_doc = load(P0 / "stage20b_rng_manifest.json")
    slots = slots_doc["slots"]
    require(slots_doc["status"] == "PASS" and slots_doc["frozen"] and len(slots) == 80, "frozen slots changed")
    require(Counter(x["class"] for x in slots) == Counter({"black": 40, "flash": 40}), "slot class counts changed")
    rngs = {row["slot_id"]: row["banks"] for row in rng_doc["records"]}
    require(set(rngs) == {x["slot_id"] for x in slots}, "slot/RNG identity mismatch")
    normal_checks = 0
    for slot in slots:
        require(Path(slot["normal_path"]).is_file(), "frozen normal missing")
        require(slot["prompt"] == PROMPTS[slot["class"]], "frozen prompt changed")
        for bank in BANKS:
            require(rngs[slot["slot_id"]][bank]["non_rng_metadata_sha256"] == htext(canonical(slot)),
                    "frozen slot metadata SHA mismatch")
        normal_checks += 1

    historical = {
        "Raw_and_MSDF_Full_bankwise_black": MULTI / "multibank_bankwise_black.csv",
        "Raw_and_MSDF_Full_bankwise_flash": MULTI / "multibank_bankwise_flash.csv",
        "Raw_and_MSDF_Full_bankwise_overall": MULTI / "multibank_bankwise_overall.csv",
        "Raw_and_MSDF_Full_pooled": MULTI / "multibank_pooled_metrics.csv",
        "MSDF_Max_bankwise_black": MAX_SUPPORT / "max_support_bankwise_black.csv",
        "MSDF_Max_bankwise_overall": MAX_SUPPORT / "max_support_bankwise_overall.csv",
        "MSDF_Max_pooled": MAX_SUPPORT / "max_support_pooled_metrics.csv",
        "V2_TRAIN_reference_features": GAMMA / "gamma_sweep_inception_features.npz",
        "V2_TRAIN_reference_feature_ledger": GAMMA / "gamma_sweep_inception_feature_cache.json",
    }
    historical_rows = {}
    for role, path in historical.items():
        require(path.is_file(), f"missing historical metric evidence: {path}")
        historical_rows[role] = {"path": path.relative_to(REPO).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}

    checkpoint_audit = {
        "status": "BASE_CHECKPOINT_PASS_LORA_PENDING",
        "audited_at": now(),
        "base_model_id": "sd2-community/stable-diffusion-2-inpainting",
        "base_model_public_revision": base_binding["public_reference"].get("revision"),
        "base_checkpoint_files": base_rows,
        "base_UNet_SHA256": base_rows["unet/diffusion_pytorch_model.safetensors"]["sha256"],
        "expected_UNet_in_channels": 9,
        "lora_checkpoints": {},
        "historical_evidence": historical_rows,
    }
    save(OUTPUT / "checkpoint_audit.json", checkpoint_audit)
    status_update(status="AUDIT_PASS", audit_completed_at=now(), train_allowlist_count=172,
                  frozen_slot_count=80, frozen_normal_existence_checks=normal_checks,
                  base_checkpoint_SHA256_recomputed=True)
    print("[审计] PASS：基础权重、172 个 TRAIN 条目、80 个冻结 slot/RNG 和历史指标证据已绑定。", flush=True)


def load_training_rows(class_name):
    rows = [row for row in load(P0 / "generator_train_allowlist.json")["inputs"] if row["class"] == class_name]
    rows.sort(key=lambda row: row["annotation_id"])
    require(len(rows) == CLASS_COUNTS[class_name] and all(row["split"] == "train" for row in rows),
            f"invalid {class_name} V2 TRAIN population")
    return rows


def train_one(class_name, device_index):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask

    state = load(OUTPUT / "experiment_status.json")
    expected_completed = CLASSES.index(class_name)
    require(state["status"] in ("AUDIT_PASS", "TRAINING") and state["training_runs_completed"] == expected_completed,
            f"training order/state invalid for {class_name}")
    dest = OUTPUT / "checkpoints" / class_name
    require(not dest.exists(), f"refusing checkpoint overwrite: {dest}")
    rows = load_training_rows(class_name)
    audit_path = OUTPUT / f"training_{class_name}.json"
    audit = {
        "status": "RUNNING", "class": class_name, "started_at": now(), "attempted_updates": 0,
        "successful_updates": 0, "AMP_skips": 0, "nonfinite_losses": 0,
        "gradient_nonfinite_count": 0, "microbatch_draws": 0, "epochs_completed": 0,
        "checkpoint_saved": False, "automatic_resume": False, "TRAIN_instances": len(rows),
        "input_annotation_ids": [row["annotation_id"] for row in rows],
        "training_protocol_sha256": sha(OUTPUT / "training_protocol.json"),
    }
    save(audit_path, audit)
    status_update(status="TRAINING", active_class=class_name,
                  training_runs_started=state["training_runs_started"] + 1)
    started = time.monotonic()
    device = set_runtime(torch, device_index)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(BASE_MODEL), local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        ).to(device)
        require(pipe.unet.config.in_channels == 9, "ordinary 4-channel text-to-image UNet is forbidden")
        pipe.vae.requires_grad_(False).eval().to(dtype=torch.float16)
        pipe.text_encoder.requires_grad_(False).eval().to(dtype=torch.float16)
        pipe.unet.requires_grad_(False).train().to(dtype=torch.float32)
        pipe.unet.add_adapter(LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0, bias="none", init_lora_weights="gaussian",
            target_modules=list(LORA_TARGETS), use_rslora=False, use_dora=False,
        ))
        pipe.unet.enable_gradient_checkpointing()
        pipe.disable_xformers_memory_efficient_attention()
        trainable = [(name, parameter) for name, parameter in pipe.unet.named_parameters() if parameter.requires_grad]
        require(trainable and all("lora_" in name for name, _ in trainable), "non-LoRA UNet parameter became trainable")
        require(all(parameter.dtype == torch.float32 for _, parameter in trainable), "LoRA masters must be FP32")
        target_counts = Counter()
        for name, _ in trainable:
            for target in LORA_TARGETS:
                if f".{target}." in name:
                    target_counts[target] += 1
        require(set(target_counts) == set(LORA_TARGETS), f"LoRA target coverage incomplete: {target_counts}")
        trainable_count = sum(parameter.numel() for _, parameter in trainable)
        total_count = sum(parameter.numel() for parameter in pipe.unet.parameters())

        scheduler = DDPMScheduler.from_pretrained(str(BASE_MODEL), subfolder="scheduler", local_files_only=True)
        require(scheduler.config.prediction_type in ("epsilon", "v_prediction"), "unsupported prediction target")
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable], lr=1e-4, betas=(0.9, 0.999),
            weight_decay=0.01, eps=1e-8,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        actual_steps = [0]
        hook = optimizer.register_step_post_hook(lambda *args, **kwargs: actual_steps.__setitem__(0, actual_steps[0] + 1))
        input_ids = pipe.tokenizer(
            PROMPTS[class_name], padding=False, truncation=True,
            max_length=pipe.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            text_embeddings = pipe.text_encoder(input_ids)[0]

        history = OUTPUT / "logs" / f"training_{class_name}_updates.jsonl"
        epoch = 0
        with history.open("x", encoding="utf-8") as log:
            while audit["successful_updates"] < 2000:
                order = torch.randperm(len(rows), generator=torch.Generator(device="cpu").manual_seed(42 + epoch)).tolist()
                audit.setdefault("epoch_order_sha256", []).append(htext(canonical([rows[index]["annotation_id"] for index in order])))
                completed_epoch = True
                for offset in range(0, len(order), 4):
                    optimizer.zero_grad(set_to_none=True)
                    group = order[offset:offset + 4]
                    losses = []
                    for index in group:
                        image, mask = crop_and_mask(rows[index])
                        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
                        mask_tensor = torch.from_numpy(np.asarray(mask).copy())[None, None].to(device).float() / 255
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                            latents = pipe.vae.encode(pixels.half()).latent_dist.sample() * pipe.vae.config.scaling_factor
                            masked_latents = pipe.vae.encode((pixels * (mask_tensor < 0.5)).half()).latent_dist.sample() * pipe.vae.config.scaling_factor
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                        latent_mask = F.interpolate(mask_tensor, size=latents.shape[-2:], mode="nearest")
                        model_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)
                        require(tuple(model_input.shape) == (1, 9, 64, 64), f"invalid inpainting input: {model_input.shape}")
                        with torch.autocast("cuda", dtype=torch.float16):
                            prediction = pipe.unet(model_input, timesteps, encoder_hidden_states=text_embeddings).sample
                        target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
                        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                        finite = bool(torch.isfinite(loss))
                        audit["microbatch_draws"] += 1
                        audit["nonfinite_losses"] += int(not finite)
                        require(finite, "nonfinite denoising loss; no rescue or hyperparameter change permitted")
                        losses.append(float(loss.detach()))
                        scaler.scale(loss / 4).backward()
                        del pixels, mask_tensor, latents, masked_latents, noise, timesteps
                        del noisy_latents, latent_mask, model_input, prediction, target, loss

                    audit["attempted_updates"] += 1
                    scaler.unscale_(optimizer)
                    grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _, p in trainable)
                    audit["gradient_nonfinite_count"] += int(not grad_finite)
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                    require(not grad_finite or bool(torch.isfinite(norm)), "invalid finite-gradient norm")
                    before = actual_steps[0]
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    success = actual_steps[0] == before + 1
                    audit["successful_updates"] += int(success)
                    audit["AMP_skips"] += int(not success)
                    require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0, "invalid GradScaler state")
                    log.write(canonical({
                        "epoch": epoch, "draws": len(group), "loss_normalizer": 4,
                        "attempt": audit["attempted_updates"], "successful_updates": audit["successful_updates"],
                        "AMP_skip": not success, "loss_mean": sum(losses) / len(losses),
                        "gradient_norm": float(norm) if bool(torch.isfinite(norm)) else None,
                        "scale_before": scale_before, "scale_after": scaler.get_scale(),
                    }) + "\n")
                    log.flush()
                    current = load(OUTPUT / "experiment_status.json")["successful_updates"]
                    current[class_name] = audit["successful_updates"]
                    status_update(successful_updates=current)
                    save(audit_path, audit)
                    if audit["attempted_updates"] % 20 == 0 or audit["successful_updates"] == 2000:
                        print(f"[训练] {class_name}: successful={audit['successful_updates']}/2000 "
                              f"attempts={audit['attempted_updates']} skips={audit['AMP_skips']}", flush=True)
                    if audit["successful_updates"] == 2000:
                        completed_epoch = False
                        break
                if completed_epoch:
                    audit["epochs_completed"] += 1
                epoch += 1
        hook.remove()
        require(audit["successful_updates"] == 2000, "successful update budget not reached")
        require(all(bool(torch.isfinite(p).all()) for _, p in trainable), "nonfinite final LoRA parameter")
        dest.mkdir(parents=True)
        lora_state = get_peft_model_state_dict(pipe.unet)
        require(lora_state and all("lora_" in key for key in lora_state), "unexpected LoRA state dict")
        StableDiffusionInpaintPipeline.save_lora_weights(
            str(dest), unet_lora_layers=lora_state, weight_name=WEIGHT_NAME, safe_serialization=True,
        )
        weight = dest / WEIGHT_NAME
        require(weight.is_file(), "LoRA checkpoint was not saved")
        audit.update({
            "status": "COMPLETE", "completed_at": now(), "checkpoint_saved": True,
            "checkpoint_path": weight.relative_to(OUTPUT).as_posix(), "checkpoint_sha256": sha(weight),
            "checkpoint_bytes": weight.stat().st_size, "successful_updates": 2000,
            "trainable_parameter_count": trainable_count, "UNet_parameter_count_including_LoRA": total_count,
            "trainable_fraction": trainable_count / total_count, "target_parameter_tensor_counts": dict(target_counts),
            "UNet_in_channels": pipe.unet.config.in_channels, "final_grad_scaler_scale": scaler.get_scale(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device), "wall_seconds": time.monotonic() - started,
            "only_UNet_attention_LoRA_trainable": True, "base_UNet_text_encoder_VAE_frozen": True,
        })
        save(audit_path, audit)
        checkpoint = load(OUTPUT / "checkpoint_audit.json")
        checkpoint["lora_checkpoints"][class_name] = {
            key: audit[key] for key in ("checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
                                        "successful_updates", "trainable_parameter_count", "target_parameter_tensor_counts",
                                        "UNet_in_channels", "only_UNet_attention_LoRA_trainable")
        }
        checkpoint["status"] = "BASE_CHECKPOINT_PASS_ONE_LORA_COMPLETE" if expected_completed == 0 else "PASS"
        checkpoint["audited_at"] = now()
        save(OUTPUT / "checkpoint_audit.json", checkpoint)
        status_update(status="TRAINING", active_class=None, training_runs_completed=expected_completed + 1)
        del optimizer, pipe, text_embeddings, input_ids
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        audit.update(status="FAILED", failed_at=now(), failure=traceback.format_exc(), wall_seconds=time.monotonic() - started)
        save(audit_path, audit)
        raise


def train(device_index):
    for class_name in CLASSES:
        train_one(class_name, device_index)
    checkpoint = load(OUTPUT / "checkpoint_audit.json")
    require(checkpoint["status"] == "PASS" and set(checkpoint["lora_checkpoints"]) == set(CLASSES),
            "two-class LoRA checkpoint barrier failed")
    status_update(status="TRAINING_COMPLETE", training_completed_at=now())
    print("[训练完成] Black/Flash 均达到 2000 次 successful optimizer updates。", flush=True)


def retry_flash_all(device_index):
    """Audited recovery from an externally terminated, non-checkpointed Flash attempt.

    The interrupted attempt is retained verbatim as evidence.  Because neither
    LoRA nor optimizer state existed at interruption, the official Flash run is
    restarted from seed 42 and step zero; Black is never rerun.
    """
    state_path = OUTPUT / "experiment_status.json"
    old_audit_path = OUTPUT / "training_flash.json"
    old_log_path = OUTPUT / "logs/training_flash_updates.jsonl"
    require(state_path.is_file() and old_audit_path.is_file() and old_log_path.is_file(),
            "interrupted Flash evidence is missing")
    state = load(state_path)
    old_audit = load(old_audit_path)
    require(state["status"] == "TRAINING" and state["training_runs_completed"] == 1,
            "Flash retry requires one completed Black run and interrupted Flash state")
    require(old_audit["status"] == "RUNNING" and old_audit["successful_updates"] == 453 and
            old_audit["attempted_updates"] == 453 and old_audit["AMP_skips"] == 0 and
            not old_audit["checkpoint_saved"], "unexpected interrupted Flash audit")
    require(not (OUTPUT / "checkpoints/flash").exists(), "partial Flash checkpoint unexpectedly exists")
    black = load(OUTPUT / "training_black.json")
    require(black["status"] == "COMPLETE" and black["successful_updates"] == 2000 and
            sha(OUTPUT / black["checkpoint_path"]) == black["checkpoint_sha256"],
            "completed Black checkpoint changed")

    archive = OUTPUT / "interrupted_flash_attempt_20261025_135555"
    require(not archive.exists(), f"retry archive already exists: {archive}")
    archive.mkdir()
    old_audit.update({
        "status": "INTERRUPTED_EXTERNAL_PROCESS_LOSS",
        "interruption_detected_at": now(),
        "last_successful_update": 453,
        "checkpoint_available": False,
        "exact_state_resume_available": False,
        "hyperparameters_changed_for_retry": False,
        "disposition": "excluded incomplete attempt; preserve audit and restart same seed from step zero",
    })
    save(archive / "training_flash.json", old_audit)
    shutil.move(str(old_log_path), str(archive / old_log_path.name))
    old_audit_path.unlink()
    save(archive / "interruption_audit.json", {
        "status": "PRESERVED",
        "reason": "managed execution session disappeared without Python traceback or catchable exception",
        "successful_updates_before_interruption": 453,
        "optimizer_or_LoRA_checkpoint_available": False,
        "training_audit_sha256": sha(archive / "training_flash.json"),
        "update_log_sha256": sha(archive / old_log_path.name),
        "update_log_rows": 453,
        "original_launcher_log_sha256": sha(OUTPUT.parent / "lora_gpu0_launcher.log"),
        "retry_policy": "same seed/config/data/order; restart from step zero; no rescue tuning",
        "black_checkpoint_reused_not_retrained": True,
        "detected_at": now(),
    })

    protocol_path = OUTPUT / "training_protocol.json"
    protocol = load(protocol_path)
    original_protocol_sha = sha(protocol_path)
    protocol["interruption_recovery"] = {
        "original_protocol_sha256": original_protocol_sha,
        "interrupted_class": "flash", "last_successful_update": 453,
        "partial_checkpoint_available": False, "optimizer_state_available": False,
        "action": "restart Flash from seed 42 and step zero",
        "Black_retrained": False, "hyperparameters_changed": False,
        "reason": "external managed-session loss; no training traceback",
        "incomplete_attempt_evidence": "interrupted_flash_attempt_20261025_135555",
        "authorized_by_user": True,
    }
    protocol["source_code_SHA256"] = source_hashes()
    save(protocol_path, protocol)

    successful = dict(state["successful_updates"])
    successful["flash"] = 0
    status_update(
        status="TRAINING", active_class=None, successful_updates=successful,
        flash_retry_count=1, flash_retry_from_step_zero=True,
        flash_interrupted_attempt_updates=453, flash_interrupted_attempt_preserved=True,
        hyperparameters_changed_for_retry=False, retry_started_at=now(),
    )
    print("[恢复] 已保留 Flash 453-step 中断证据；Black 不重跑；Flash 同协议从 step 0 重跑。", flush=True)
    train_one("flash", device_index)
    checkpoint = load(OUTPUT / "checkpoint_audit.json")
    require(checkpoint["status"] == "PASS" and set(checkpoint["lora_checkpoints"]) == set(CLASSES),
            "two-class LoRA checkpoint barrier failed after Flash retry")
    status_update(status="TRAINING_COMPLETE", training_completed_at=now(),
                  flash_retry_completed=True)
    print("[训练完成] Black/Flash 最终 LoRA 均为 2000 successful updates；开始冻结生成。", flush=True)
    generate(device_index)
    evaluate(device_index)


def crop_box(bbox, width, height, margin=0.25):
    cx, cy, bw, bh = bbox
    result = [
        max(0, math.floor((cx - (0.5 + margin) * bw) * width)),
        max(0, math.floor((cy - (0.5 + margin) * bh) * height)),
        min(width, math.ceil((cx + (0.5 + margin) * bw) * width)),
        min(height, math.ceil((cy + (0.5 + margin) * bh) * height)),
    ]
    require(result[2] > result[0] and result[3] > result[1], "invalid ROI")
    return result


def generate(device_index):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from generate_plastic_bomo_stage20b_banks import sample

    state = load(OUTPUT / "experiment_status.json")
    require(state["status"] == "TRAINING_COMPLETE", "generation requires both final LoRA checkpoints")
    checkpoint = load(OUTPUT / "checkpoint_audit.json")
    require(checkpoint["status"] == "PASS", "checkpoint audit is not PASS")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rngs = {row["slot_id"]: row["banks"] for row in load(P0 / "stage20b_rng_manifest.json")["records"]}
    cfg = load(P0 / "stage20b_generation_protocol.json")
    require(cfg["num_inference_steps"] == 50 and cfg["guidance_scale"] == 7.5, "generation protocol changed")
    manifest_path = OUTPUT / "generation_manifest.json"
    require(not manifest_path.exists(), "refusing to rerun or overwrite generation")
    records = []
    save(manifest_path, {"status": "RUNNING", "planned_count": 240, "records": records,
                         "reroll_count": 0, "quality_selection_count": 0})
    status_update(status="GENERATING", generation_started_at=now())
    device = set_runtime(torch, device_index)
    started = time.monotonic()
    try:
        for class_name in CLASSES:
            weight = OUTPUT / checkpoint["lora_checkpoints"][class_name]["checkpoint_path"]
            require(sha(weight) == checkpoint["lora_checkpoints"][class_name]["checkpoint_sha256"],
                    f"LoRA checkpoint changed: {class_name}")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(BASE_MODEL), local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
                safety_checker=None, feature_extractor=None, requires_safety_checker=False,
            ).to(device)
            require(pipe.unet.config.in_channels == 9, "generation UNet is not 9-channel inpainting")
            pipe.load_lora_weights(str(weight.parent), weight_name=weight.name, adapter_name="default")
            pipe.set_adapters("default", adapter_weights=1.0)
            pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
            pipe.disable_xformers_memory_efficient_attention()
            pipe.scheduler = DDIMScheduler.from_config(cfg["scheduler_config"])
            require(all(getattr(pipe.scheduler.config, key) == value for key, value in cfg["scheduler_config"].items()),
                    "DDIM scheduler config changed")
            selected = [slot for slot in slots if slot["class"] == class_name]
            require(len(selected) == 40, f"frozen slot count changed for {class_name}")
            for bank in BANKS:
                folder = OUTPUT / "images" / bank / class_name
                folder.mkdir(parents=True, exist_ok=False)
                for index, slot in enumerate(selected):
                    rng = rngs[slot["slot_id"]][bank]
                    require(rng["non_rng_metadata_sha256"] == htext(canonical(slot)), "frozen slot changed")
                    item_started = time.monotonic()
                    image, trace = sample(pipe, slot, rng, cfg, device)
                    path = folder / f"{slot['slot_id']}.png"
                    image.save(path, format="PNG")
                    width, height = slot["geometry"]["final_canvas_wh"]
                    record = {
                        "slot_id": slot["slot_id"], "annotation_id": slot["annotation_id"],
                        "class": class_name, "bank": bank, "prompt": slot["prompt"],
                        "bbox_normalized_xywh": slot["bbox"],
                        "mask_decoded_sha256": slot["geometry"]["mask_decoded_sha256"],
                        "normal_id": slot["conditioning_normal_id"], "normal_sha256": slot["normal_sha256"],
                        "non_rng_metadata_sha256": rng["non_rng_metadata_sha256"],
                        "bank_seed": rng["bank_seed"], "sample_seed": rng["sample_seed"],
                        "lora_checkpoint_sha256": checkpoint["lora_checkpoints"][class_name]["checkpoint_sha256"],
                        "image_path": path.relative_to(OUTPUT).as_posix(), "image_SHA256": sha(path),
                        "roi_xyxy": crop_box(slot["bbox"], width, height),
                        "quality_selection": False, "reroll": False,
                        "runtime_seconds": time.monotonic() - item_started,
                        **{key: value for key, value in trace.items() if key not in ("image_path", "image_sha256")},
                    }
                    records.append(record)
                    save(manifest_path, {"status": "RUNNING", "planned_count": 240, "records": records,
                                         "reroll_count": 0, "quality_selection_count": 0})
                    status_update(generated_images=len(records))
                    print(f"[生成] {class_name} Bank {bank} {index + 1}/40 total={len(records)}/240", flush=True)
            del pipe
            gc.collect(); torch.cuda.empty_cache()
        require(len(records) == 240, "generation count is not 240")
        require(Counter((r["bank"], r["class"]) for r in records) ==
                Counter({(bank, class_name): 40 for bank in BANKS for class_name in CLASSES}),
                "bank/class generation counts changed")
        save(manifest_path, {
            "status": "COMPLETE", "count": 240, "records": records,
            "source_frozen_slots_sha256": sha(P0 / "stage20b_generation_slots.json"),
            "source_frozen_rng_manifest_sha256": sha(P0 / "stage20b_rng_manifest.json"),
            "all_outputs_retained": True, "quality_selection_count": 0, "best_of_n": False,
            "reroll_count": 0, "private_paths_omitted": True,
            "runtime_seconds": time.monotonic() - started, "completed_at": now(),
        })
        status_update(status="GENERATION_COMPLETE", generation_completed_at=now(), generated_images=240)
        print("[生成完成] A/B/C × Black/Flash = 240；无筛选、Best-of-N 或重抽。", flush=True)
    except Exception:
        save(manifest_path, {"status": "FAILED", "planned_count": 240, "records": records,
                             "reroll_count": 0, "quality_selection_count": 0,
                             "failure": traceback.format_exc(), "failed_at": now()})
        raise


def extract_lora_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    cache = OUTPUT / "lora_inception_features.npz"
    ledger_path = OUTPUT / "lora_inception_feature_cache.json"
    model = lib["Inception"](
        "inception-v3-compat", ["2048"], feature_extractor_weights_path=str(lib["inception_weight"])
    ).float().eval().requires_grad_(False).to(device)
    keys, values = [], []
    with torch.inference_mode():
        for index, row in enumerate(manifest["records"]):
            path = OUTPUT / row["image_path"]
            require(path.is_file() and sha(path) == row["image_SHA256"], f"generated image changed: {path}")
            with Image.open(path) as source:
                roi = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
            tensor = torch.from_numpy(np.asarray(roi, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = model(tensor)[0]
            require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()), row["slot_id"])
            keys.append(f"lora:{row['bank']}:{row['class']}:{row['slot_id']}")
            values.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0:
                print(f"[Inception] {index + 1}/240", flush=True)
    data = np.stack(values)
    np.savez_compressed(cache, features=data)
    save(ledger_path, {"status": "COMPLETE", "keys": keys, "shape": [240, 2048],
                       "stored_dtype": str(data.dtype), "extraction_dtype": "float32",
                       "sha256": sha(cache), "image_SHA256_checks": 240})
    del model
    torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[index]).double() for index, key in enumerate(keys)}


def numeric(row):
    result = dict(row)
    for key in ("KID", "KID_subset_std", "FID", "runtime_seconds"):
        if key in result:
            result[key] = float(result[key])
    for key in ("reference_ROIs", "generated_images", "KID_subset_size"):
        if key in result:
            result[key] = int(float(result[key]))
    return result


def historical_comparison_rows():
    rows = []
    multi_files = {
        "Black": MULTI / "multibank_bankwise_black.csv",
        "Flash": MULTI / "multibank_bankwise_flash.csv",
        "Overall": MULTI / "multibank_bankwise_overall.csv",
    }
    for population, path in multi_files.items():
        source = read_csv(path)
        for group, method, tuning in (
            ("raw", "Raw SD2", "full UNet fine-tuning"),
            ("background", "MSDF Full (Full Mean-Support)", "full UNet fine-tuning + MSDF"),
        ):
            for item in source:
                if item["group"] == group:
                    item = numeric(item)
                    rows.append({"scope": item["bank"], "population": population, "method": method,
                                 "fine_tuning_type": tuning, **{k: item[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                                 "metric_source": item["metric_source"]})
        if population == "Flash":
            for item in source:
                if item["group"] == "background":
                    item = numeric(item)
                    rows.append({"scope": item["bank"], "population": population,
                                 "method": "MSDF Max-Support", "fine_tuning_type": "full UNet fine-tuning + MSDF",
                                 **{k: item[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                                 "metric_source": "same frozen Flash outputs as MSDF Full; Max-Support modifies Black only"})

    max_files = {
        "Black": MAX_SUPPORT / "max_support_bankwise_black.csv",
        "Overall": MAX_SUPPORT / "max_support_bankwise_overall.csv",
    }
    for population, path in max_files.items():
        for item in read_csv(path):
            if item["group"] == "max":
                item = numeric(item)
                rows.append({"scope": item["bank"], "population": population,
                             "method": "MSDF Max-Support", "fine_tuning_type": "full UNet fine-tuning + MSDF",
                             **{k: item[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                             "metric_source": item["metric_source"]})

    multi_pooled = [numeric(item) for item in read_csv(MULTI / "multibank_pooled_metrics.csv")]
    for population in ("Black", "Flash", "Overall"):
        for scope in ("B+C", "A+B+C"):
            for group, method, tuning in (
                ("raw", "Raw SD2", "full UNet fine-tuning"),
                ("background", "MSDF Full (Full Mean-Support)", "full UNet fine-tuning + MSDF"),
            ):
                item = next(row for row in multi_pooled if row["scope"] == scope and row["population"] == population and row["group"] == group)
                rows.append({"scope": scope, "population": population, "method": method,
                             "fine_tuning_type": tuning,
                             **{k: item[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                             "metric_source": "reused SHA-bound historical concatenated-feature metric"})
            if population == "Flash":
                item = next(row for row in multi_pooled if row["scope"] == scope and row["population"] == population and row["group"] == "background")
                rows.append({"scope": scope, "population": population, "method": "MSDF Max-Support",
                             "fine_tuning_type": "full UNet fine-tuning + MSDF",
                             **{k: item[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                             "metric_source": "same frozen Flash outputs as MSDF Full; Max-Support modifies Black only"})

    max_pooled = [numeric(item) for item in read_csv(MAX_SUPPORT / "max_support_pooled_metrics.csv")]
    for population in ("Black", "Overall"):
        for scope in ("B+C", "A+B+C"):
            item = next(row for row in max_pooled if row["scope"] == scope and row["population"] == population and row["group"] == "max")
            rows.append({"scope": scope, "population": population, "method": "MSDF Max-Support",
                         "fine_tuning_type": "full UNet fine-tuning + MSDF",
                         **{k: item[k] for k in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                         "metric_source": item["metric_source"]})
    return rows


def report(bankwise, pooled, comparison):
    lines = [
        "# Plastic_Bomo SD2 Inpainting + LoRA 对比实验", "",
        "Status: `COMPLETE`", "",
        "本轮只执行标准 UNet Attention LoRA；未修改或训练 MSDF。Black 与 Flash 各自独立训练一次，均完成 2000 次 successful optimizer updates。",
        "基础网络为冻结的 `sd2-community/stable-diffusion-2-inpainting` 9 通道 Inpainting UNet；LoRA rank/alpha 均为 16。",
        "", "## LoRA 逐 Bank 指标", "",
        "| Bank | Population | KID ↓ | KID subset std | FID ↓ | Generated |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in bankwise:
        lines.append(f"| {row['bank']} | {row['population']} | {row['KID']:.10f} | {row['KID_subset_std']:.10f} | {row['FID']:.6f} | {row['generated_images']} |")
    lines += ["", "## LoRA Pooled（从拼接特征重算）", "",
              "| Scope | Population | KID ↓ | KID subset std | FID ↓ | Generated |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in pooled:
        lines.append(f"| {row['scope']} | {row['population']} | {row['KID']:.10f} | {row['KID_subset_std']:.10f} | {row['FID']:.6f} | {row['generated_images']} |")
    lines += [
        "", "## 对比口径", "",
        "`comparison.csv` 并列 Raw SD2、MSDF Full（Full Mean-Support）、MSDF Max-Support 与本次 LoRA。Raw/MSDF 使用历史 SHA 绑定的同协议指标；LoRA 使用本轮 240 张图重新提取的特征。Raw/MSDF 是基础 UNet 全量微调路线，LoRA 是仅 UNet Attention 的低秩微调。Max-Support 只改变 Black 分支，因此其 Flash 行与 MSDF Full 使用同一冻结 Flash 输出，并已在 `metric_source` 中标明。",
        "", "Pooled B+C 与 A+B+C 均将对应图片的 Inception-2048 特征拼接后重新计算 KID/FID，绝非逐 Bank 指标平均。",
        "", "## 完整性", "",
        "- 训练数据：V2 TRAIN，Black=97、Flash=75；未读取 official final-eval 内容。",
        "- 生成：A/B/C 每 Bank 每类 40 张，共 240 张；DDIM 50、CFG 7.5；无筛选、Best-of-N 或 reroll。",
        "- 私有图片、LoRA 权重和 Inception 特征缓存只保留在服务器，不应提交 GitHub。",
        "- KID subset std 是 100 个内部子集的波动，不是跨 Bank 置信区间。",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(device_index):
    import torch
    import run_plastic_bomo_msdf_multibank_stability as x12
    from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime

    require(load(OUTPUT / "experiment_status.json")["status"] == "GENERATION_COMPLETE",
            "evaluation requires complete 240-image generation")
    manifest = load(OUTPUT / "generation_manifest.json")
    require(manifest["status"] == "COMPLETE" and manifest["count"] == 240, "generation manifest incomplete")
    status_update(status="EVALUATING", evaluation_started_at=now())
    device = set_runtime(torch, device_index)
    started = time.monotonic()
    lib, runtime_audit = metric_runtime(device)
    features = extract_lora_features(device, torch, lib, manifest)
    gamma = x12.array_features(GAMMA / "gamma_sweep_inception_features.npz",
                               GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
    references = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
    real = {
        "Black": torch.stack([gamma["real:" + row["annotation_id"]] for row in references if row["class"] == "black"]),
        "Flash": torch.stack([gamma["real:" + row["annotation_id"]] for row in references if row["class"] == "flash"]),
        "Overall": torch.stack([gamma["real:" + row["annotation_id"]] for row in references]),
    }
    require({key: len(value) for key, value in real.items()} == {"Black": 97, "Flash": 75, "Overall": 172},
            "V2 TRAIN reference counts changed")

    def fake(bank, population):
        selected = [row for row in manifest["records"] if row["bank"] == bank and
                    (population == "Overall" or row["class"] == population.lower())]
        return torch.stack([features[f"lora:{row['bank']}:{row['class']}:{row['slot_id']}"] for row in selected])

    bankwise = []
    for bank in BANKS:
        for population in ("Black", "Flash", "Overall"):
            result = x12.metric(real[population], fake(bank, population), lib)
            bankwise.append({"bank": bank, "population": population, **result,
                             "metric_source": "new LoRA same-protocol ROI features"})
            print(f"[指标] Bank {bank} {population} KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
    pooled = []
    for scope, banks in (("B+C", "BC"), ("A+B+C", "ABC")):
        for population in ("Black", "Flash", "Overall"):
            combined = torch.cat([fake(bank, population) for bank in banks])
            result = x12.metric(real[population], combined, lib)
            pooled.append({"scope": scope, "population": population, **result,
                           "metric_source": "new concatenated LoRA features; not bank average"})
            print(f"[Pooled] {scope} {population} KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)

    write_csv(OUTPUT / "bankwise_metrics.csv", bankwise)
    write_csv(OUTPUT / "pooled_metrics.csv", pooled)
    comparison = historical_comparison_rows()
    for row in bankwise:
        comparison.append({"scope": row["bank"], "population": row["population"],
                           "method": "SD2 Inpainting + LoRA r16", "fine_tuning_type": "UNet Attention LoRA fine-tuning",
                           **{key: row[key] for key in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                           "metric_source": row["metric_source"]})
    for row in pooled:
        comparison.append({"scope": row["scope"], "population": row["population"],
                           "method": "SD2 Inpainting + LoRA r16", "fine_tuning_type": "UNet Attention LoRA fine-tuning",
                           **{key: row[key] for key in ("KID", "KID_subset_std", "FID", "reference_ROIs", "generated_images", "KID_subset_size")},
                           "metric_source": row["metric_source"]})
    scope_order = {"A": 0, "B": 1, "C": 2, "B+C": 3, "A+B+C": 4}
    population_order = {"Black": 0, "Flash": 1, "Overall": 2}
    method_order = {"Raw SD2": 0, "MSDF Full (Full Mean-Support)": 1,
                    "MSDF Max-Support": 2, "SD2 Inpainting + LoRA r16": 3}
    comparison.sort(key=lambda row: (scope_order[row["scope"]], population_order[row["population"]], method_order[row["method"]]))
    require(len(comparison) == 60, f"comparison row count changed: {len(comparison)}")
    write_csv(OUTPUT / "comparison.csv", comparison)
    report(bankwise, pooled, comparison)
    protocol = load(OUTPUT / "training_protocol.json")
    protocol.update(status="COMPLETE", completed_at=now(), source_code_SHA256=source_hashes(),
                    runtime_metric_audit=runtime_audit)
    save(OUTPUT / "training_protocol.json", protocol)
    status_update(status="COMPLETE", completed_at=now(), evaluation_completed=True,
                  evaluation_runtime_seconds=time.monotonic() - started,
                  bankwise_metric_rows=len(bankwise), pooled_metric_rows=len(pooled),
                  comparison_rows=len(comparison), pooled_recomputed_not_bank_average=True,
                  quality_selection_count=0, reroll_count=0, followup_model_started=False)
    print("[完成] LoRA 训练、240 张生成、ROI KID/FID 与历史对比均完成；不启动其他模型。", flush=True)


def mark_failed(stage, error):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failure = f"{type(error).__name__}: {error}"
    status_update(status=f"{stage.upper()}_FAILED", failed_stage=stage, failure=failure,
                  traceback=traceback.format_exc(), failed_at=now(),
                  no_hyperparameter_rescue=True, followup_model_started=False)
    report_path = OUTPUT / "REPORT.md"
    if not report_path.is_file():
        report_path.write_text(
            "# Plastic_Bomo SD2 Inpainting + LoRA 对比实验\n\n"
            f"Status: `{stage.upper()}_FAILED`\n\n"
            f"失败原因：`{failure}`\n\n"
            "未私自更改超参数，未启动其他模型。详见 `experiment_status.json`。\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "train", "generate", "evaluate", "all", "retry-flash-all"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    offline()
    if args.stage == "retry-flash-all":
        try:
            retry_flash_all(args.device)
        except Exception as error:
            mark_failed("retry_flash_all", error)
            raise
        return
    stages = ("audit", "train", "generate", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        try:
            if stage == "audit":
                audit()
            elif stage == "train":
                train(args.device)
            elif stage == "generate":
                generate(args.device)
            else:
                evaluate(args.device)
        except Exception as error:
            mark_failed(stage, error)
            raise


if __name__ == "__main__":
    main()
