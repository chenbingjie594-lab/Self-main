"""Experiment 11: calibrate and train gradient-balanced Region supervision.

The runner deliberately reuses the frozen Stage25 construction, microbatch,
loss, sampling, and data bindings.  It creates a separate experiment tree and
never edits the historical Stage25 run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_black_balanced_region"
EXPERIMENT_CONFIG = REPO / "configs/plastic_bomo_msdf_black_balanced_region.json"
SOURCE_RUN = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214"
SOURCE_RUNTIME = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/server_runtime_20261013_080546_552698"
RAW = REPO / "results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
EXP7 = REPO / "results_for_gpt/plastic_bomo_msdf_raw_matched_comparison"
NOMORPH = REPO / "results_for_gpt/plastic_bomo_msdf_black_no_morph_alignment"
NOAUX = REPO / "results_for_gpt/plastic_bomo_msdf_black_no_aux"
EXP10 = REPO / "results_for_gpt/plastic_bomo_msdf_black_gradient_background"
BASE_COMMIT = "c580c24d179ca3bc46f6cb31ccc44a8189c45cdd"
AUXILIARY_NAMES = ("region", "reconstruction", "structure", "background", "scale", "support")
AUXILIARY_WEIGHT_KEYS = {name: name + "_weight" for name in AUXILIARY_NAMES}
HISTORICAL_WEIGHTS = {"region": 0.75, "reconstruction": 0.50, "structure": 0.25,
                      "background": 0.20, "scale": 0.25, "support": 0.25}
CALIBRATION_WEIGHTS = {"region": 0.0, "reconstruction": 0.0, "structure": 0.0,
                       "background": 0.20, "scale": 0.0, "support": 0.0}
REGION_WEIGHT_CAP = 0.02
MIN_REGION_WEIGHT = 1e-6


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hvalue(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("cannot write an empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def calibrated_weights(require_calibration=True):
    weights = dict(CALIBRATION_WEIGHTS)
    path = OUTPUT / "gradient_balance_calibration.json"
    if require_calibration:
        require(path.is_file(), "gradient balance calibration is missing")
        calibration = load(path)
        require(calibration["status"] == "PASS", "gradient balance calibration did not pass")
        value = calibration["lambda_region"]
        require(math.isfinite(value) and value >= MIN_REGION_WEIGHT and value <= REGION_WEIGHT_CAP,
                "frozen lambda_region is invalid")
        weights["region"] = value
    return weights


def binding(require_calibration=True):
    bound = load(SOURCE_RUN / "stage25_run_binding.json")
    cfg = load(REPO / "configs/plastic_bomo_stage25_server.json")
    experiment = load(EXPERIMENT_CONFIG)
    require(experiment["base_commit"] == BASE_COMMIT and experiment["class"] == "black",
            "Experiment 11 config identity changed")
    phase_b = experiment["phase_b"]
    require(phase_b["morphology_alignment_enabled"] is True and phase_b["injection_gamma"] == 1.0,
            "Experiment 11 inference controls changed")
    fixed = phase_b["fixed_auxiliary_loss_weights"]
    require(fixed == {"reconstruction": 0.0, "structure": 0.0, "background": 0.20,
                      "scale": 0.0, "support": 0.0}, "Experiment 11 fixed auxiliary controls changed")
    weights = calibrated_weights(require_calibration=require_calibration)
    for name, key in AUXILIARY_WEIGHT_KEYS.items():
        cfg["MSDF_preserved_defaults"][key] = weights[name]
    return bound, cfg


def source_hashes():
    names = (
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py",
        "train_rda.py",
        "tools/stage25_msdf_generator.py",
        "tools/stage25_msdf_contract.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "tools/qualify_plastic_bomo_stage20b_r0.py",
        "tools/run_plastic_bomo_msdf_black_balanced_region.py",
        "configs/plastic_bomo_msdf_black_balanced_region.json",
    )
    return {name: sha(REPO / name) for name in names}


def protocol_record(cfg):
    train = cfg["common_generator_training_target"]
    defaults = cfg["MSDF_preserved_defaults"]
    return {
        "status": "FROZEN_BEFORE_TRAINING",
        "experiment": "Experiment 11 — Gradient-Balanced Region Supervision",
        "base_commit": BASE_COMMIT,
        "class": "black",
        "groups": {
            "G0": "Frozen Raw SD2 Black",
            "G1": "Frozen MSDF-NoAux Black",
            "G2": "Frozen MSDF-BackgroundOnly Black",
            "G3": "New MSDF-BalancedRegion Black",
        },
        "single_planned_training": "G3 only",
        "training": {**train, "objective": "conditional_loss + 0.20 * msdf_background_loss + lambda_region * msdf_region_loss"},
        "msdf": {**defaults, "morphology_alignment_enabled": True, "injection_gamma": 1.0},
        "auxiliary_loss_weights": {name: defaults[AUXILIARY_WEIGHT_KEYS[name]] for name in AUXILIARY_NAMES},
        "only_planned_factor": "Add the once-calibrated lambda_region * Region Loss to frozen BackgroundOnly",
        "semantic_effect": "Background weight remains 0.20; Region uses the frozen calibration value; the other four auxiliary weights are zero.",
        "bank": "A",
        "generation_count": 40,
        "evaluation": {
            "reference": "V2 TRAIN Black 97 ROI; possible checkpoint-training overlap",
            "independent_validation": False,
            "final_eval_read_or_used": False,
            "ROI": "bbox plus 25 percent context with frozen floor/ceil clipping",
            "feature_extractor": "torch-fidelity Inception-v3-compat 2048D float32",
            "metric_dtype": "float64 CPU",
            "KID": {"subsets": 100, "subset_size": 40, "seed": 42, "degree": 3},
        },
        "source_code_SHA256": source_hashes(),
        "historical_stage25_protocol_SHA256": sha(REPO / "configs/plastic_bomo_stage25_server.json"),
        "historical_results_modified": False,
        "prohibited_followup_started": False,
    }


def validate_protocol(cfg):
    train = cfg["common_generator_training_target"]
    defaults = cfg["MSDF_preserved_defaults"]
    historical = load(REPO / "configs/plastic_bomo_stage25_server.json")["MSDF_preserved_defaults"]
    expected_train = {
        "seed": 42, "resolution": 512, "train_batch_size": 1,
        "gradient_accumulation_steps": 4, "learning_rate": 1e-6,
        "max_train_steps": 2000, "optimizer": "AdamW", "mixed_precision": "fp16",
        "gradient_checkpointing": True, "xformers": False, "max_grad_norm": 1,
    }
    expected_msdf = {
        "hidden_dim": 64, "context_scale": 1.75, "max_injection": 0.75,
        "max_residual_ratio": 0.25, "branch_dropout": 0.2,
        "pixel_support_weight": 0.5, "region_weight": 0.75,
        "reconstruction_weight": 0.5, "structure_weight": 0.25,
        "background_weight": 0.2, "scale_weight": 0.25,
        "support_weight": 0.25, "reference_cross_probability": 0.3,
        "mask_jitter_radius": 24, "mask_blur_radius": 2,
        "MSDF_learning_rate": 1e-4,
    }
    require(all(train.get(k) == v for k, v in expected_train.items()), "Stage25 training protocol changed")
    require(all(historical.get(k) == v for k, v in expected_msdf.items()), "Stage25 MSDF defaults changed")
    require(historical["morphology_alignment_enabled"] is True, "historical G1 morphology flag changed")
    structural_keys = set(expected_msdf) - set(AUXILIARY_WEIGHT_KEYS.values())
    require(all(defaults.get(k) == historical.get(k) for k in structural_keys), "BalancedRegion changed an MSDF structural control")
    require({name: defaults[AUXILIARY_WEIGHT_KEYS[name]] for name in AUXILIARY_NAMES} == calibrated_weights(),
            "BalancedRegion auxiliary weights changed")
    require(defaults["morphology_alignment_enabled"] is True, "BalancedRegion morphology alignment is not enabled")
    require(train["prompts"]["black"] == "a photo of a sks black spot defect", "Black prompt changed")
    return {"training": expected_train, "historical_msdf": expected_msdf,
            "balanced_region_weights": {name: defaults[AUXILIARY_WEIGHT_KEYS[name]] for name in AUXILIARY_NAMES}}


def initialize_output(cfg):
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    (OUTPUT / "logs").mkdir(parents=True)
    (OUTPUT / "images/bank_A").mkdir(parents=True)
    save(OUTPUT / "balanced_region_protocol.json", protocol_record(cfg))
    save(OUTPUT / "experiment_status.json", {
        "status": "CALIBRATION_PENDING", "created_at": now(), "base_commit": BASE_COMMIT,
        "training_count": 0, "generation_count": 0, "reroll_count": 0,
        "selection_count": 0, "final_eval_read_or_used": False,
        "BootstrapGuard": False, "detector_training_started": False,
        "other_structure_search_started": False,
    })


def update_status(**values):
    path = OUTPUT / "experiment_status.json"
    row = load(path)
    row.update(values)
    save(path, row)


def cpu_branch_test():
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter
    torch.manual_seed(42)
    adapter = MSDFAdapter(4, 8, [8], 1.75, 0.75, 0.25, 0.2, 0.5).eval()
    adapter.set_ablation({"morphology_alignment": True})
    adapter.set_injection_gamma(1.0)
    called = {"align": 0}
    original_align = adapter._align_reference

    def observed(*args, **kwargs):
        called["align"] += 1
        return original_align(*args, **kwargs)

    adapter._align_reference = observed
    reference = torch.randn(1, 4, 8, 8)
    reference_mask = torch.zeros(1, 1, 64, 64)
    reference_mask[:, :, 20:44, 23:41] = 1
    prior = torch.zeros(1, 1, 8, 8)
    prior[:, :, 2:7, 2:6] = 1
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    with torch.no_grad():
        gates = adapter.prepare(reference, reference_mask, prior, torch.tensor([500]), 1000, False, pixels)
    require(called["align"] == 1, "CPU morphology-alignment branch was not called exactly once")
    require(bool(torch.isfinite(adapter.last_aligned_support).all()), "aligned support is non-finite")
    require(adapter.ablation == {"pixel_branch": True, "latent_branch": True,
            "morphology_alignment": True, "up_block_mask": [True], "residual_scale": 1.0},
            "unexpected ablation state")
    require(tuple(gates.shape) == (1, 1) and bool(torch.isfinite(gates).all()), "invalid CPU gates")
    return {
        "status": "PASS", "morphology_alignment": True,
        "align_reference_call_count": 1,
        "aligned_support_finite": True,
        "pixel_branch_enabled": True, "latent_branch_enabled": True,
        "support_head_present": hasattr(adapter, "support_head"),
        "injection_gamma": adapter.injection_gamma,
    }


@contextmanager
def gpu_context(device, evaluation=False, calibration=False):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    bound, cfg = binding(require_calibration=not calibration)
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden, final_eval_content_forbidden, set_gpu_runtime
    counts = {"forbidden_final_eval_open_attempts": 0}
    with ExitStack() as stack:
        stack.enter_context(network_forbidden())
        if not evaluation:
            stack.enter_context(final_eval_content_forbidden(Path(bound["stage20a"]), Path(bound["p0_view"]), counts))
        import torch
        set_gpu_runtime(torch, device)
        require("RTX 3090" in torch.cuda.get_device_name(device), "an RTX 3090 is required")
        require(torch.__version__ == "2.5.1+cu124" and torch.version.cuda == "12.4", "Torch/CUDA runtime changed")
        yield torch, bound, cfg
    require(counts["forbidden_final_eval_open_attempts"] == 0, "final_eval access attempted")


def verify_assets(bound):
    p0 = Path(bound["p0_view"])
    asset_audit = load(SOURCE_RUNTIME / "server_asset_hash_audit.json")
    checks = asset_audit["TRAIN_checks"]
    require(len(checks) == 666, "TRAIN asset audit population changed")
    for row in checks:
        require(Path(row["path"]).is_file() and sha(row["path"]) == row["expected_sha256"],
                f"TRAIN asset changed: {row['path']}")
    rows = load(p0 / "generator_train_allowlist.json")["inputs"]
    slots = load(p0 / "stage20b_generation_slots.json")["slots"]
    require(Counter(r["class"] for r in rows) == {"flash": 75, "black": 97}, "TRAIN population changed")
    require(Counter(r["class"] for r in slots) == {"flash": 40, "black": 40}, "slot population changed")
    weights = load(p0 / "sd2_base_weight_binding.json")["files"]
    for name, row in weights.items():
        require(sha(Path(bound["base_model"]) / name) == row["sha256"], f"base model changed: {name}")
    return {"TRAIN_file_checks": len(checks), "Black_TRAIN_instances": 97,
            "Black_Bank_A_slots": 40, "base_model_file_checks": len(weights)}


def state_sha(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(msdf):
    return {
        "format_version": 3, "latent_channels": msdf.latent_channels,
        "hidden_dim": msdf.hidden_dim, "block_channels": msdf.block_channels,
        "context_scale": msdf.context_scale, "max_injection": msdf.max_injection,
        "max_residual_ratio": msdf.max_residual_ratio, "branch_dropout": msdf.branch_dropout,
        "pixel_support_weight": msdf.pixel_support_weight,
        "morphology_alignment_enabled": True,
        "injection_gamma": 1.0,
        "auxiliary_loss_weights": calibrated_weights(),
        "state_dict": {name: value.detach().cpu() for name, value in msdf.state_dict().items()},
    }


LOSS_LABELS = {
    "denoising": "Denoising", "region": "Region", "reconstruction": "Reconstruction",
    "structure": "Structure", "background": "Background", "scale": "Multiscale", "support": "Support",
}


def freeze_gradient_samples(bound):
    """Reuse the exact twelve samples and references frozen by Experiment 10."""
    rows = sorted((r for r in load(Path(bound["p0_view"]) / "generator_train_allowlist.json")["inputs"]
                   if r["class"] == "black"), key=lambda r: r["annotation_id"])
    require(len(rows) == 97, "Black TRAIN population changed")
    previous = load(EXP10 / "gradient_probe_protocol.json")
    require(previous["status"] == "COMPLETE" and previous["sample_count"] == 12,
            "Experiment 10 gradient sample protocol is not frozen")
    records = previous["records"]
    by_id = {row["annotation_id"]: row for row in rows}
    require(all(r["annotation_id"] in by_id and r["reference_annotation_id"] in by_id for r in records),
            "Experiment 10 frozen gradient sample identity is missing")
    selected = [by_id[r["annotation_id"]] for r in records]
    require(len({r["source_group_id"] for r in selected}) == 12 and
            sum(r["reference_mode"] == "self" for r in records) == 8 and
            sum(r["reference_mode"] == "cross" for r in records) == 4,
            "Experiment 10 frozen sample distribution changed")
    protocol = {
        "status": "FROZEN_BEFORE_GRADIENTS", "created_at": now(), "base_commit": BASE_COMMIT,
        "checkpoint": "Experiment 10 MSDF-BackgroundOnly Black final_step2000",
        "checkpoint_binding_SHA256": sha(EXP10 / "background_only_checkpoint_binding.json"),
        "source_gradient_protocol_SHA256": sha(EXP10 / "gradient_probe_protocol.json"),
        "selection_seed": 42, "selection_rule": previous["selection_rule"],
        "reference_rule": previous["reference_rule"],
        "sample_count": 12, "self_reference_count": 8, "cross_reference_count": 4,
        "distinct_source_group_count": 12, "records": records,
        "losses": {"base": "Denoising + 0.20 * Background", "region": "Raw Region"},
        "parameter_groups": ["UNet", "MSDF"],
        "calibration_formula": "min(0.02, median_norm_base_unet/median_norm_region_unet, median_norm_base_msdf/median_norm_region_msdf)",
        "minimum_allowed_lambda": MIN_REGION_WEIGHT,
        "optimizer_step_count": 0, "final_eval_read_or_used": False,
    }
    return rows, selected, protocol


def rng_snapshot(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state, torch, np):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"]); torch.cuda.set_rng_state_all(state["torch_cuda"])


def differentiable_microbatch(objects, row, reference_row, bound, cfg, torch):
    """Run the literal Stage25 loss body while retaining its seven loss tensors."""
    import numpy as np
    import torch.nn.functional as F
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    from stage25_msdf_generator import LOSSES
    pipe, msdf, scheduler, text, env, code, args, _identity = objects

    def tensors(record):
        image, mask = crop_and_mask(record)
        return (torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].to(bound["device"]).float() / 127.5 - 1,
                torch.from_numpy(np.asarray(mask).copy())[None, None].to(bound["device"]).float() / 255)

    pixels, original_mask = tensors(row)
    reference, reference_mask = tensors(reference_row)
    defaults = cfg["MSDF_preserved_defaults"]
    mask = env["augment_coarse_mask"](original_mask, defaults["mask_jitter_radius"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        latents = pipe.vae.encode(pixels.half()).latent_dist.sample() * pipe.vae.config.scaling_factor
        masked = pipe.vae.encode((pixels * (mask < .5)).half()).latent_dist.sample() * pipe.vae.config.scaling_factor
        reference_latents = pipe.vae.encode(reference.half()).latent_dist.mode() * pipe.vae.config.scaling_factor
    soft_prior = env["make_soft_prior"](mask, latents.shape[-2:], defaults["mask_blur_radius"])
    saliency = env["build_defect_saliency"](pixels, original_mask, latents.shape[-2:],
        defaults["saliency_low_quantile"], defaults["saliency_high_quantile"]).detach()
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=bound["device"]).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    model_input = torch.cat([noisy_latents, soft_prior.to(noisy_latents.dtype), masked], 1)
    require(list(model_input.shape) == [1, 9, 64, 64], "gradient probe input shape changed")
    msdf.prepare(reference_latents.float(), reference_mask, soft_prior, timesteps,
        scheduler.config.num_train_timesteps, classifier_free_guidance=True, reference_pixels=reference)
    with torch.autocast("cuda", dtype=torch.float16):
        prediction = pipe.unet(torch.cat([model_input, model_input]), torch.cat([timesteps, timesteps]),
                               encoder_hidden_states=text).sample
        prediction_cond = prediction.chunk(2)[1]
        target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
        context = dict(env, args=args, msdf=msdf, scheduler=scheduler, device=torch.device(bound["device"]),
            pixel_error=(prediction_cond.float() - target.float()).square(),
            conditional_loss=F.mse_loss(prediction_cond.float(), target.float()), defect_saliency=saliency,
            reference_is_self=torch.tensor([row["annotation_id"] == reference_row["annotation_id"]], device=bound["device"]),
            timesteps=timesteps, noisy_latents=noisy_latents, latents=latents, soft_prior=soft_prior,
            prediction_cond=prediction_cond,
            **{"msdf_" + name + "_loss": torch.zeros((), device=bound["device"]) for name in LOSSES})
        exec(code, context)
    losses = {"denoising": context["conditional_loss"],
              **{name: context["msdf_" + name + "_loss"] for name in LOSSES}}
    require(all(value.ndim == 0 and bool(torch.isfinite(value)) for value in losses.values()),
            "gradient probe loss is non-finite")
    return losses, {name: float(value.detach()) for name, value in losses.items()}, int(timesteps.item())


def load_background_probe_model(torch, bound, cfg):
    from stage25_msdf_generator import construct
    from stage20b_formal_common import verify_tree
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    from safetensors.torch import load_file
    checkpoint_binding = load(EXP10 / "background_only_checkpoint_binding.json")
    require(checkpoint_binding["status"] == "FROZEN" and checkpoint_binding["successful_updates"] == 2000,
            "Experiment 10 BackgroundOnly checkpoint is not frozen")
    checkpoint = verify_tree(EXP10, checkpoint_binding["checkpoint"])
    values = list(construct(torch, bound, cfg, "black"))
    pipe, fresh = values[:2]
    fresh.detach(); del fresh
    trained_unet = load_file(str(checkpoint / "unet/diffusion_pytorch_model.safetensors"), device="cpu")
    pipe.unet.load_state_dict(trained_unet, strict=True); del trained_unet
    trained = load_msdf_adapter(checkpoint / "msdf.pt", pipe.unet, bound["device"])
    trained.train(); trained.set_ablation({"morphology_alignment": True}); trained.set_injection_gamma(1.0)
    values[1] = trained
    return tuple(values), checkpoint_binding


def capture_gradient_baseline(module, scale, torch):
    saved, norm2, none_count, zero_count = [], 0.0, 0, 0
    for _name, parameter in module.named_parameters():
        if parameter.grad is None:
            saved.append(None); none_count += 1; continue
        value = (parameter.grad.detach().float() / scale).cpu()
        require(bool(torch.isfinite(value).all()), "non-finite base gradient")
        saved.append(value)
        square = float(torch.sum(value.double() * value.double()))
        norm2 += square; zero_count += int(square == 0.0)
    return {"values": saved, "norm2": norm2, "norm": math.sqrt(norm2),
            "none_parameter_tensors": none_count, "zero_parameter_tensors": zero_count,
            "parameter_tensors": len(saved)}


def compare_gradient(module, baseline, scale, torch):
    norm2 = dot = 0.0; none_count = zero_count = 0
    for (_name, parameter), reference in zip(module.named_parameters(), baseline["values"]):
        if parameter.grad is None:
            none_count += 1; continue
        current = (parameter.grad.detach().float() / scale).cpu()
        require(bool(torch.isfinite(current).all()), "non-finite Region gradient")
        square = float(torch.sum(current.double() * current.double()))
        norm2 += square; zero_count += int(square == 0.0)
        if reference is not None:
            dot += float(torch.sum(current.double() * reference.double()))
    norm = math.sqrt(norm2); base_norm = baseline["norm"]
    cosine = None if norm == 0.0 or base_norm == 0.0 else dot / (norm * base_norm)
    return {"norm2": norm2, "norm": norm, "dot_with_base": dot, "cosine": cosine,
            "base_norm": base_norm, "none_parameter_tensors": none_count,
            "zero_parameter_tensors": zero_count, "parameter_tensors": baseline["parameter_tensors"]}


def aggregate_calibration(rows):
    import statistics
    summary = []
    for group in ("UNet", "MSDF"):
        selected = [r for r in rows if r["parameter_group"] == group]
        valid = [r for r in selected if r["status"] == "VALID"]
        base = [r["base_gradient_norm"] for r in valid]
        region = [r["region_gradient_norm"] for r in valid]
        cosines = [r["cosine_base_region"] for r in valid]
        require(base and region and all(v > 0 and math.isfinite(v) for v in base + region),
                f"{group} has no valid nonzero calibration gradients")
        base_median = statistics.median(base); region_median = statistics.median(region)
        require(region_median > 0 and math.isfinite(base_median / region_median),
                f"{group} Region median denominator is invalid")
        summary.append({
            "parameter_group": group,
            "mean_base_gradient_norm": statistics.fmean(base),
            "median_base_gradient_norm": base_median,
            "mean_region_gradient_norm": statistics.fmean(region),
            "median_region_gradient_norm": region_median,
            "mean_cosine": statistics.fmean(cosines),
            "median_cosine": statistics.median(cosines),
            "negative_cosine_fraction": sum(v < 0 for v in cosines) / len(cosines),
            "positive_cosine_fraction": sum(v > 0 for v in cosines) / len(cosines),
            "valid_samples": len(valid),
            "zero_gradient_samples": sum(r["status"] == "ZERO_GRADIENT" for r in selected),
            "nonfinite_gradient_samples": 0,
            "ratio_base_over_region": base_median / region_median,
        })
    return summary


def gradient_probe(device):
    bound, cfg = binding(require_calibration=False)
    initialize_output(cfg)
    rows, _selected, protocol = freeze_gradient_samples(bound)
    save(OUTPUT / "gradient_balance_protocol.json", protocol)
    update_status(status="CALIBRATION_RUNNING", calibration_started_at=now())
    started = time.monotonic(); per_sample = []
    try:
        with gpu_context(device, calibration=True) as (torch, bound, cfg):
            import numpy as np
            torch.cuda.reset_peak_memory_stats()
            objects, checkpoint_binding = load_background_probe_model(torch, bound, cfg)
            pipe, msdf = objects[:2]
            initial_unet = state_sha(pipe.unet); initial_msdf = state_sha(msdf)
            by_id = {row["annotation_id"]: row for row in rows}
            gradient_scale = 1.0
            for item in protocol["records"]:
                row = by_id[item["annotation_id"]]; reference = by_id[item["reference_annotation_id"]]
                random.seed(item["probe_seed"]); np.random.seed(item["probe_seed"] % (2 ** 32))
                torch.manual_seed(item["probe_seed"]); torch.cuda.manual_seed_all(item["probe_seed"])
                state = rng_snapshot(torch, np); reference_values = None; reference_timestep = None
                baselines = None; base_loss_value = region_loss_value = None
                for objective in ("base", "region"):
                    restore_rng(state, torch, np)
                    pipe.unet.zero_grad(set_to_none=True); msdf.zero_grad(set_to_none=True)
                    loss_tensors, scalar_values, timestep = differentiable_microbatch(
                        objects, row, reference, bound, cfg, torch)
                    if reference_values is None:
                        reference_values, reference_timestep = scalar_values, timestep
                    else:
                        require(timestep == reference_timestep and all(
                            abs(scalar_values[name] - reference_values[name]) <= 1e-7 * max(1.0, abs(reference_values[name]))
                            for name in scalar_values), "repeated calibration forward changed loss/timestep")
                    target_loss = (loss_tensors["denoising"] + 0.20 * loss_tensors["background"]
                                   if objective == "base" else loss_tensors["region"])
                    require(bool(torch.isfinite(target_loss)), f"{objective} calibration loss is non-finite")
                    target_loss.backward()
                    if objective == "base":
                        base_loss_value = float(target_loss.detach())
                        baselines = {"UNet": capture_gradient_baseline(pipe.unet, gradient_scale, torch),
                                     "MSDF": capture_gradient_baseline(msdf, gradient_scale, torch)}
                    else:
                        region_loss_value = float(target_loss.detach())
                        stats = {"UNet": compare_gradient(pipe.unet, baselines["UNet"], gradient_scale, torch),
                                 "MSDF": compare_gradient(msdf, baselines["MSDF"], gradient_scale, torch)}
                        for group, result in stats.items():
                            base_norm = result["base_norm"]; region_norm = result["norm"]
                            valid = base_norm > 0 and region_norm > 0 and result["cosine"] is not None
                            per_sample.append({
                                "sample_index": item["sample_index"],
                                "sample_identity_SHA256": item["sample_image_SHA256"],
                                "source_group_SHA256": hashlib.sha256(item["source_group_id"].encode()).hexdigest(),
                                "reference_mode": item["reference_mode"], "parameter_group": group,
                                "base_loss": base_loss_value, "region_loss": region_loss_value,
                                "base_gradient_norm": base_norm, "region_gradient_norm": region_norm,
                                "cosine_base_region": result["cosine"],
                                "none_parameter_tensors": result["none_parameter_tensors"],
                                "zero_parameter_tensors": result["zero_parameter_tensors"],
                                "status": "VALID" if valid else "ZERO_GRADIENT",
                            })
                    del loss_tensors, target_loss
                    gc.collect(); torch.cuda.empty_cache()
                del baselines
                print(f"[Region校准] {item['sample_index'] + 1}/12 mode={item['reference_mode']}", flush=True)
            require(state_sha(pipe.unet) == initial_unet and state_sha(msdf) == initial_msdf,
                    "calibration modified model parameters")
            peak = torch.cuda.max_memory_allocated()
            msdf.detach(); del objects, pipe, msdf; gc.collect(); torch.cuda.empty_cache()
        summary = aggregate_calibration(per_sample)
        groups = {row["parameter_group"]: row for row in summary}
        r_unet = groups["UNet"]["ratio_base_over_region"]
        r_msdf = groups["MSDF"]["ratio_base_over_region"]
        lambda_region = min(REGION_WEIGHT_CAP, r_unet, r_msdf)
        require(math.isfinite(lambda_region) and lambda_region > 0, "lambda_region is non-finite or non-positive")
        require(lambda_region >= MIN_REGION_WEIGHT, "lambda_region is below the frozen 1e-6 stop threshold")
        write_csv(OUTPUT / "gradient_balance_per_sample.csv", per_sample)
        calibration = {
            "status": "PASS", "formula": "min(0.02, r_unet, r_msdf)",
            "region_weight_cap": REGION_WEIGHT_CAP, "minimum_allowed_lambda": MIN_REGION_WEIGHT,
            "groups": groups, "r_unet": r_unet, "r_msdf": r_msdf,
            "lambda_region": lambda_region, "full_precision_training_value": repr(lambda_region),
            "sample_count": 12, "self_reference_count": 8, "cross_reference_count": 4,
            "optimizer_steps": 0, "repeated_forward_consistency": "PASS",
            "model_parameter_identity_unchanged": True,
            "checkpoint_tree_SHA256": checkpoint_binding["checkpoint"]["tree_sha256"],
            "source_code_SHA256": source_hashes(), "peak_GPU_memory_bytes": peak,
            "runtime_seconds": time.monotonic() - started, "completed_at": now(),
        }
        save(OUTPUT / "gradient_balance_calibration.json", calibration)
        phase_b_protocol = load(OUTPUT / "balanced_region_protocol.json")
        phase_b_protocol["auxiliary_loss_weights"] = calibrated_weights()
        phase_b_protocol["lambda_region"] = lambda_region
        phase_b_protocol["calibration_SHA256"] = sha(OUTPUT / "gradient_balance_calibration.json")
        phase_b_protocol["training"]["objective"] = (
            "conditional_loss + 0.20 * msdf_background_loss + " +
            repr(lambda_region) + " * msdf_region_loss")
        save(OUTPUT / "balanced_region_protocol.json", phase_b_protocol)
        protocol.update(status="COMPLETE", repeated_forward_consistency="PASS",
                        model_parameter_identity_unchanged=True, optimizer_step_count=0,
                        runtime_seconds=time.monotonic() - started, lambda_region=lambda_region,
                        calibration_SHA256=sha(OUTPUT / "gradient_balance_calibration.json"))
        save(OUTPUT / "gradient_balance_protocol.json", protocol)
        update_status(status="CALIBRATION_PASS", calibration_completed_at=now(),
                      gradient_calibration_optimizer_steps=0, lambda_region=lambda_region)
        print(f"[校准完成] r_unet={r_unet:.12g} r_msdf={r_msdf:.12g} lambda_region={lambda_region:.17g}", flush=True)
    except Exception as error:
        update_status(status="CALIBRATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise

def preflight(device):
    bound, cfg = binding()
    status = load(OUTPUT / "experiment_status.json")
    require(status["status"] == "CALIBRATION_PASS",
            "BalancedRegion preflight requires a passing frozen Region calibration")
    calibration = load(OUTPUT / "gradient_balance_calibration.json")
    lambda_region = calibration["lambda_region"]
    calibration_groups = calibration["groups"]
    require(set(calibration_groups) == {"UNet", "MSDF"} and all(
        calibration_groups[name]["valid_samples"] > 0 and
        calibration_groups[name]["zero_gradient_samples"] == 0 and
        calibration_groups[name]["nonfinite_gradient_samples"] == 0
        for name in ("UNet", "MSDF")),
        "frozen Region calibration does not prove finite nonzero propagation")
    started = time.monotonic()
    audit = {"status": "RUNNING", "started_at": now(), "optimizer_steps": 0,
             "sampling_count": 0, "final_eval_content_read": 0}
    save(OUTPUT / "balanced_region_preflight.json", audit)
    try:
        audit["protocol"] = validate_protocol(cfg)
        audit["CPU_morphology_alignment_branch"] = cpu_branch_test()
        audit["assets"] = verify_assets(bound)
        with gpu_context(device) as (torch, bound, cfg):
            from stage25_msdf_generator import construct, microbatch, frozen_hashes, gradient_record, LOSSES
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            torch.cuda.reset_peak_memory_stats()
            objects = construct(torch, bound, cfg, "black")
            pipe, msdf = objects[:2]
            initial_adapter_sha = state_sha(msdf)
            require(initial_adapter_sha == load(EXP10 / "background_only_training_audit.json")["initial_adapter_state_SHA256"],
                    "preflight adapter initialization differs from Experiment 10")
            state = msdf.set_ablation({"morphology_alignment": True})
            msdf.set_injection_gamma(1.0)
            require(state["morphology_alignment"] is True and state["pixel_branch"] and state["latent_branch"],
                    "BalancedRegion state invalid")
            require(state_sha(msdf) == initial_adapter_sha, "set_ablation changed adapter parameters")
            before = frozen_hashes(pipe)
            rows = sorted((r for r in load(Path(bound["p0_view"]) / "generator_train_allowlist.json")["inputs"]
                           if r["class"] == "black"), key=lambda r: r["annotation_id"])
            loss, losses, shapes = microbatch(objects, rows[0], rows[0], bound, cfg, torch)
            expected_objective = losses["denoising"] + 0.20 * losses["background"] + lambda_region * losses["region"]
            objective_delta = abs(float(loss.detach()) - expected_objective)
            weights = {name: cfg["MSDF_preserved_defaults"][AUXILIARY_WEIGHT_KEYS[name]]
                       for name in AUXILIARY_NAMES}
            require(weights == calibrated_weights(), "BalancedRegion weights changed")
            require(objective_delta <= 1e-7, "BalancedRegion diffusion objective differs from the frozen formula")
            target = cfg["common_generator_training_target"]
            lambda_region = cfg["MSDF_preserved_defaults"]["region_weight"]
            # Match the formal FP16 training backward exactly.  An unscaled
            # diagnostic backward can underflow through the zero-initialized
            # FP16 projection path even though GradScaler makes that same
            # conditional-loss gradient effective during training.
            probe_optimizer = torch.optim.AdamW([
                {"params": list(pipe.unet.parameters()), "lr": target["learning_rate"]},
                {"params": list(msdf.parameters()), "lr": cfg["MSDF_preserved_defaults"]["MSDF_learning_rate"]},
            ], betas=(target["adam_beta1"], target["adam_beta2"]),
                weight_decay=target["adam_weight_decay"], eps=target["adam_epsilon"])
            probe_scaler = torch.amp.GradScaler("cuda", enabled=True)
            probe_scaler.scale(loss / target["gradient_accumulation_steps"]).backward()
            probe_scaler.unscale_(probe_optimizer)
            gradients = {"unet": gradient_record(pipe.unet, torch), "msdf": gradient_record(msdf, torch)}
            frozen_ok = before == frozen_hashes(pipe) and all(
                not p.requires_grad and p.grad is None for module in (pipe.vae, pipe.text_encoder) for p in module.parameters())
            audit["single_batch_gradient_probe"] = gradients
            save(OUTPUT / "balanced_region_preflight.json", audit)
            require(all(row["all_finite"] and row["nonzero_parameter_names"] for row in gradients.values()),
                    "UNet or MSDF has no finite nonzero denoising gradient")
            nonzero = gradients["msdf"]["nonzero_parameter_names"]
            disconnected = set(gradients["msdf"]["disconnected"])
            latent_parameters = [name for name, _value in msdf.named_parameters() if name.startswith("encoder.")]
            pixel_parameters = [name for name, _value in msdf.named_parameters() if name.startswith("pixel_encoder.")]
            require(latent_parameters and pixel_parameters and
                    not disconnected.intersection(latent_parameters + pixel_parameters),
                    "pixel/latent branch gradient graph is disconnected")
            require(msdf.branch_dropout == cfg["MSDF_preserved_defaults"]["branch_dropout"] == 0.2,
                    "branch dropout changed")
            require(set(losses) == {*LOSSES, "denoising"} and all(math.isfinite(v) for v in losses.values()),
                    "six-loss path changed")
            require(msdf.last_gates is not None and bool(torch.isfinite(msdf.last_gates).all()) and
                    bool(torch.count_nonzero(msdf.last_gates)), "dynamic gate is inactive or non-finite")
            require(frozen_ok, "VAE/Text Encoder freeze check failed")
            temporary = OUTPUT / "logs/preflight_balanced_region_msdf.pt"
            torch.save(checkpoint_payload(msdf), temporary)
            temporary_payload = torch.load(temporary, map_location="cpu", weights_only=False)
            require(temporary_payload["auxiliary_loss_weights"] == calibrated_weights(),
                    "checkpoint did not preserve the frozen training weights")
            msdf.detach()
            reloaded = load_msdf_adapter(temporary, pipe.unet, device)
            require(reloaded.ablation["morphology_alignment"] is True and not reloaded.training,
                    "checkpoint reload did not restore morphology_alignment=True")
            require(reloaded.injection_gamma == 1.0, "inference gamma is not 1")
            reloaded.detach()
            temporary.unlink()
            audit["GPU_single_batch"] = {
                "status": "PASS", "device": device, "device_name": torch.cuda.get_device_name(device),
                "diffusion_objective": float(loss.detach()), "conditional_loss": losses["denoising"],
                "expected_objective": float(expected_objective),
                "objective_minus_expected_abs": objective_delta,
                "backward_matches_formal_training": True,
                "gradient_scaler_scale": probe_scaler.get_scale(),
                "optimizer_step_called": False,
                "raw_auxiliary_losses_diagnostic_only": {name: losses[name] for name in AUXILIARY_NAMES},
                "auxiliary_loss_weights": weights,
                "weighted_background_contribution": 0.20 * losses["background"],
                "lambda_region": lambda_region,
                "weighted_region_contribution": lambda_region * losses["region"],
                "other_four_weighted_contribution": 0.0,
                "Region_gradient_propagation": {
                    name: {
                        "valid_samples": calibration_groups[name]["valid_samples"],
                        "zero_gradient_samples": calibration_groups[name]["zero_gradient_samples"],
                        "nonfinite_gradient_samples": calibration_groups[name]["nonfinite_gradient_samples"],
                    } for name in ("UNet", "MSDF")
                },
                "tensor_shapes": shapes,
                "gradients": gradients, "dual_branch_gradient_graph_connected": True,
                "all_MSDF_parameters_connected": gradients["msdf"]["all_connected"],
                "single_batch_latent_branch_nonzero": any(name.startswith("encoder.") for name in nonzero),
                "single_batch_pixel_branch_nonzero": any(name.startswith("pixel_encoder.") for name in nonzero),
                "single_batch_zero_gradient_note": "A connected branch may be zero in one batch by the frozen branch_dropout=0.2 design.",
                "VAE_text_frozen_unchanged": frozen_ok,
                "morphology_alignment_train_forward": True,
                "morphology_alignment_reload": True,
                "dynamic_gate_active": True,
                "checkpoint_reload_eval_mode": True,
                "six_auxiliary_losses_computed_for_diagnostics": list(LOSSES),
                "weighted_auxiliary_losses_in_objective": ["background", "region"],
                "initial_adapter_state_SHA256": initial_adapter_sha,
                "set_ablation_parameter_identity_unchanged": True,
                "peak_GPU_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            msdf_values = cfg["MSDF_preserved_defaults"]
            audit["controlled_variables"] = {
                "all_expected_values_match": True,
                "G1_G3_public_SD2_start": bound["base_model"],
                "adapter_initialization_seed": cfg["common_generator_training_target"]["seed"],
                "reference_sampling_rule": msdf_values["reference_cross_probability"],
                "random_trajectory_limit": (
                    "Experiment 10 records the initial adapter/UNet identities and RNG summaries. Experiment 11 reuses the "
                    "same seed, sorted input order, epoch permutation and reference rule; different objective code execution "
                    "means complete byte-for-byte RNG trajectory equivalence is audited but not assumed."
                ),
            }
            del loss, reloaded, msdf, pipe, objects
            gc.collect(); torch.cuda.empty_cache()
        audit.update(status="PASS", completed_at=now(), wall_seconds=time.monotonic() - started,
                     formal_training_authorized=True)
        save(OUTPUT / "balanced_region_preflight.json", audit)
        update_status(status="PREFLIGHT_PASS", preflight_completed_at=now())
        print("[检查通过] BalancedRegion CPU、单batch、双分支梯度、六项Loss、冻结组件和checkpoint恢复全部通过。", flush=True)
    except Exception as error:
        audit.update(status="FAIL", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
        save(OUTPUT / "balanced_region_preflight.json", audit)
        update_status(status="PREFLIGHT_FAILED", failure=audit["error"])
        raise


def train(device):
    status = load(OUTPUT / "experiment_status.json")
    require(status["status"] == "PREFLIGHT_PASS", "formal training requires a PASS preflight")
    require(not (OUTPUT / "logs/train_claim.json").exists(), "training already claimed; automatic retry is forbidden")
    save(OUTPUT / "logs/train_claim.json", {"started_at": now(), "class": "black", "planned_successful_updates": 2000})
    update_status(status="TRAINING", training_count=1, training_started_at=now())
    started = time.monotonic()
    audit = {
        "status": "RUNNING", "class": "black", "morphology_alignment_enabled": True,
        "injection_gamma": 1.0, "successful_updates": 0, "scheduled_attempts": 0,
        "AMP_skips": 0, "microbatch_draws": 0, "checkpoint_saved": False,
        "automatic_resume": False, "final_eval_content_read": 0,
        "auxiliary_loss_weights": calibrated_weights(),
        "lambda_region": calibrated_weights()["region"],
        "training_objective": "conditional_loss + 0.20 * msdf_background_loss + lambda_region * msdf_region_loss",
    }
    path = OUTPUT / "balanced_region_training_audit.json"
    save(path, audit)
    try:
        with gpu_context(device) as (torch, bound, cfg):
            from stage25_msdf_generator import construct, microbatch, frozen_hashes
            from stage25_msdf_contract import tree_binding, verified_copy
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            torch.cuda.reset_peak_memory_stats()
            objects = construct(torch, bound, cfg, "black")
            pipe, msdf = objects[:2]
            initial_adapter_sha = state_sha(msdf)
            msdf.set_ablation({"morphology_alignment": True})
            msdf.set_injection_gamma(1.0)
            require(state_sha(msdf) == initial_adapter_sha, "BalancedRegion changed initialization")
            background_training = load(EXP10 / "background_only_training_audit.json")
            require(initial_adapter_sha == background_training["initial_adapter_state_SHA256"],
                    "BalancedRegion adapter initialization differs from Experiment 10 BackgroundOnly")
            before = frozen_hashes(pipe)
            rows = sorted((r for r in load(Path(bound["p0_view"]) / "generator_train_allowlist.json")["inputs"]
                           if r["class"] == "black"), key=lambda r: r["annotation_id"])
            require(len(rows) == 97, "Black TRAIN population changed")
            target = cfg["common_generator_training_target"]
            lambda_region = cfg["MSDF_preserved_defaults"]["region_weight"]
            opt = torch.optim.AdamW([
                {"params": list(pipe.unet.parameters()), "lr": target["learning_rate"]},
                {"params": list(msdf.parameters()), "lr": cfg["MSDF_preserved_defaults"]["MSDF_learning_rate"]},
            ], betas=(target["adam_beta1"], target["adam_beta2"]),
                weight_decay=target["adam_weight_decay"], eps=target["adam_epsilon"])
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            actual = [0]
            hook = opt.register_step_post_hook(lambda *_args, **_kw: actual.__setitem__(0, actual[0] + 1))
            params = [p for group in opt.param_groups for p in group["params"]]
            sums = {"training_loss": 0.0, "gradient_norm": 0.0,
                    **{"auxiliary_" + name: 0.0 for name in ("region", "reconstruction", "structure", "background", "scale", "support", "denoising")}}
            value_count = 0
            audit.update(
                precision="fp16 autocast, FP32 full UNet/MSDF masters",
                optimizer_groups=[{k: v for k, v in group.items() if k != "params"} for group in opt.param_groups],
                epoch_order_sha256=[], epoch_tail_divisor=4,
                input_population_SHA256=hvalue([r["annotation_id"] for r in rows]),
                initial_adapter_state_SHA256=initial_adapter_sha,
                initial_UNet_weight_file_SHA256=sha(Path(bound["base_model"]) / "unet/diffusion_pytorch_model.safetensors"),
                initialization_matches_Experiment10_BackgroundOnly=True,
                initial_RNG_state_summary={
                    "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    "torch_CPU": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    "torch_CUDA": hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
                },
                source_code_SHA256=source_hashes(),
                random_trajectory_limit=load(OUTPUT / "balanced_region_preflight.json")["controlled_variables"]["random_trajectory_limit"],
                MSDF_nonzero_gradient_attempts=0,
            )
            epoch, consecutive_skips = 0, 0
            log_path = OUTPUT / "logs/balanced_region_black_updates.jsonl"
            with log_path.open("x", encoding="utf-8") as log:
                while actual[0] < target["max_train_steps"]:
                    order = torch.randperm(len(rows), generator=torch.Generator(device="cpu").manual_seed(target["seed"] + epoch)).tolist()
                    audit["epoch_order_sha256"].append(hvalue([rows[i]["annotation_id"] for i in order]))
                    for offset in range(0, len(order), target["gradient_accumulation_steps"]):
                        opt.zero_grad(set_to_none=True)
                        records = []
                        for index in order[offset:offset + target["gradient_accumulation_steps"]]:
                            reference_index = index
                            if random.random() < cfg["MSDF_preserved_defaults"]["reference_cross_probability"]:
                                reference_index = (index + random.randint(1, len(rows) - 1)) % len(rows)
                            loss, losses, shapes = microbatch(objects, rows[index], rows[reference_index], bound, cfg, torch)
                            expected_objective = (losses["denoising"] + 0.20 * losses["background"] +
                                                  lambda_region * losses["region"])
                            objective_delta = abs(float(loss.detach()) - expected_objective)
                            require(objective_delta <= 1e-7,
                                    "training objective differs from the frozen BalancedRegion formula")
                            scaler.scale(loss / target["gradient_accumulation_steps"]).backward()
                            loss_value = float(loss.detach())
                            sums["training_loss"] += loss_value
                            for name, value in losses.items():
                                sums["auxiliary_" + name] += value
                            value_count += 1
                            records.append({**shapes, "loss": loss_value, "conditional_loss": losses["denoising"],
                                            "background_loss": losses["background"],
                                            "region_loss": losses["region"],
                                            "weighted_region_contribution": lambda_region * losses["region"],
                                            "objective_minus_expected_abs": objective_delta,
                                            "raw_auxiliary_losses": {name: losses[name] for name in AUXILIARY_NAMES}})
                            audit["microbatch_draws"] += 1
                            del loss
                        audit["scheduled_attempts"] += 1
                        scaler.unscale_(opt)
                        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                        msdf_gradients = [p.grad for p in msdf.parameters() if p.grad is not None]
                        msdf_finite = bool(msdf_gradients) and all(bool(torch.isfinite(g).all()) for g in msdf_gradients)
                        msdf_nonzero = sum(bool(torch.count_nonzero(g)) for g in msdf_gradients if bool(torch.isfinite(g).all()))
                        audit["MSDF_gradient_status"] = {
                            "connected_parameter_tensors": len(msdf_gradients),
                            "all_connected": len(msdf_gradients) == sum(1 for _ in msdf.parameters()),
                            "all_finite": msdf_finite,
                            "nonzero_parameter_tensors": msdf_nonzero,
                        }
                        audit["MSDF_nonzero_gradient_attempts"] += int(msdf_nonzero > 0)
                        unet_gradients = [p.grad for p in pipe.unet.parameters() if p.grad is not None]
                        unet_finite = bool(unet_gradients) and all(bool(torch.isfinite(g).all()) for g in unet_gradients)
                        unet_nonzero = sum(bool(torch.count_nonzero(g)) for g in unet_gradients if bool(torch.isfinite(g).all()))
                        audit["UNet_gradient_status"] = {
                            "connected_parameter_tensors": len(unet_gradients),
                            "all_connected": len(unet_gradients) == sum(1 for _ in pipe.unet.parameters()),
                            "all_finite": unet_finite, "nonzero_parameter_tensors": unet_nonzero,
                        }
                        norm = torch.nn.utils.clip_grad_norm_(params, target["max_grad_norm"])
                        require(not finite or bool(torch.isfinite(norm)), "invalid gradient norm")
                        before_step, before_scale = actual[0], scaler.get_scale()
                        scaler.step(opt); scaler.update()
                        success = actual[0] == before_step + 1
                        require(not success or (msdf_finite and msdf_nonzero > 0 and unet_finite and unet_nonzero > 0),
                                "successful update had no effective finite MSDF/UNet gradient")
                        audit["successful_updates"] = actual[0]
                        audit["AMP_skips"] += int(not success)
                        consecutive_skips = 0 if success else consecutive_skips + 1
                        require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0 and consecutive_skips < 20,
                                "invalid scaler or 20 consecutive AMP skips")
                        norm_value = float(norm) if bool(torch.isfinite(norm)) else None
                        if norm_value is not None:
                            sums["gradient_norm"] += norm_value
                        log.write(canonical({"epoch": epoch, "offset": offset, "attempt": audit["scheduled_attempts"],
                            "successful_updates": actual[0], "AMP_skip": not success,
                            "gradients_finite": finite, "gradient_norm": norm_value,
                            "scale_before": before_scale, "scale_after": scaler.get_scale(),
                            "microbatches": records}) + "\n")
                        log.flush()
                        audit["running_means"] = {name: value / max(value_count, 1) for name, value in sums.items()}
                        audit["peak_GPU_memory_bytes"] = torch.cuda.max_memory_allocated()
                        save(path, audit)
                        if actual[0] % 25 == 0 or not success:
                            print(f"[训练] Black BalancedRegion successful={actual[0]} attempts={audit['scheduled_attempts']} skips={audit['AMP_skips']}", flush=True)
                        if actual[0] == target["max_train_steps"]:
                            require(success and finite and norm_value is not None, "last update invalid")
                            break
                    epoch += 1
            hook.remove()
            require(actual[0] == 2000 and all(bool(torch.isfinite(p).all()) for p in params), "2000/finiteness barrier failed")
            require(before == frozen_hashes(pipe), "VAE/Text Encoder changed")
            destination = OUTPUT / "checkpoints/black/final_step2000"
            require(not destination.exists(), "final checkpoint already exists")
            pipe.unet.save_pretrained(str(destination / "unet"), safe_serialization=True)
            for name, row in load(Path(bound["p0_view"]) / "sd2_base_weight_binding.json")["files"].items():
                if name == "unet/diffusion_pytorch_model.safetensors":
                    continue
                target_path = destination / name
                if target_path.exists():
                    require(name == "unet/config.json", f"unexpected existing checkpoint component: {name}")
                    target_path.write_bytes((Path(bound["base_model"]) / name).read_bytes())
                else:
                    verified_copy(Path(bound["base_model"]) / name, target_path, row["sha256"])
                require(sha(target_path) == row["sha256"], f"checkpoint component changed: {name}")
            torch.save(checkpoint_payload(msdf), destination / "msdf.pt")
            checkpoint = tree_binding(destination, OUTPUT)
            payload = torch.load(destination / "msdf.pt", map_location="cpu", weights_only=False)
            require(payload["morphology_alignment_enabled"] is True, "checkpoint metadata flag is not true")
            require(payload["auxiliary_loss_weights"] == calibrated_weights(),
                    "checkpoint auxiliary weights changed")
            msdf.detach()
            reloaded = load_msdf_adapter(destination / "msdf.pt", pipe.unet, device)
            require(reloaded.ablation["morphology_alignment"] is True and reloaded.injection_gamma == 1.0,
                    "saved checkpoint reload failed")
            reloaded.detach()
            checkpoint_binding = {
                "status": "FROZEN", "class": "black", "successful_updates": 2000,
                "fresh_public_SD2_start": True, "continued_from_full_MSDF": False,
                "morphology_alignment_enabled": True, "injection_gamma": 1.0,
                "auxiliary_loss_weights": calibrated_weights(),
                "lambda_region": calibrated_weights()["region"],
                "checkpoint": checkpoint, "msdf_checkpoint_SHA256": sha(destination / "msdf.pt"),
                "base_model_binding_SHA256": sha(Path(bound["p0_view"]) / "sd2_base_weight_binding.json"),
            }
            save(OUTPUT / "balanced_region_checkpoint_binding.json", checkpoint_binding)
            audit.update(status="COMPLETE", checkpoint_saved=True, checkpoint=checkpoint,
                         VAE_text_frozen_unchanged=True, final_grad_scaler_scale=scaler.get_scale(),
                         peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),
                         update_log_SHA256=sha(log_path), completed_at=now())
            del reloaded, msdf, pipe, objects, opt, scaler
            gc.collect(); torch.cuda.empty_cache()
        audit["wall_seconds"] = time.monotonic() - started
        save(path, audit)
        update_status(status="TRAINING_COMPLETE", training_completed_at=now(), successful_updates=2000,
                      scheduled_attempts=audit["scheduled_attempts"], AMP_skips=audit["AMP_skips"])
        print("[训练完成] Black BalancedRegion final_step2000 已保存并通过冻结lambda_region重载检查。", flush=True)
    except Exception as error:
        audit.update(status="FAILED", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
        save(path, audit)
        update_status(status="TRAINING_FAILED", failure=audit["error"])
        raise


def crop_box(bbox, width, height, margin=0.25):
    cx, cy, bw, bh = bbox
    result = [max(0, math.floor((cx - (0.5 + margin) * bw) * width)),
              max(0, math.floor((cy - (0.5 + margin) * bh) * height)),
              min(width, math.ceil((cx + (0.5 + margin) * bw) * width)),
              min(height, math.ceil((cy + (0.5 + margin) * bh) * height))]
    require(result[2] > result[0] and result[3] > result[1], "invalid ROI")
    return result


def generate(device):
    status = load(OUTPUT / "experiment_status.json")
    require(status["status"] == "TRAINING_COMPLETE", "generation requires completed training")
    require(not (OUTPUT / "logs/generate_claim.json").exists(), "generation already claimed; reroll is forbidden")
    save(OUTPUT / "logs/generate_claim.json", {"started_at": now(), "bank": "A", "planned_count": 40})
    update_status(status="GENERATING", generation_started_at=now())
    started = time.monotonic()
    records = []
    manifest_path = OUTPUT / "balanced_region_generation_manifest.json"
    save(manifest_path, {"status": "RUNNING", "bank": "A", "records": []})
    try:
        with gpu_context(device) as (torch, bound, cfg):
            import numpy as np
            from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
            from generate_plastic_bomo_stage20b_banks import sample
            from stage20b_formal_common import synthetic_label, verify_tree
            checkpoint_binding = load(OUTPUT / "balanced_region_checkpoint_binding.json")
            checkpoint = verify_tree(OUTPUT, checkpoint_binding["checkpoint"])
            pipe = StableDiffusionInpaintPipeline.from_pretrained(str(checkpoint), local_files_only=True,
                use_safetensors=True, torch_dtype=torch.float16, safety_checker=None,
                feature_extractor=None, requires_safety_checker=False).to(device)
            pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
            pipe.disable_xformers_memory_efficient_attention()
            protocol = load(Path(bound["p0_view"]) / "stage20b_generation_protocol.json")
            pipe.scheduler = DDIMScheduler.from_config(protocol["scheduler_config"])
            msdf = load_msdf_adapter(checkpoint / "msdf.pt", pipe.unet, device)
            require(msdf.ablation["morphology_alignment"] is True, "generation did not restore morphology alignment")
            msdf.set_injection_gamma(1.0)
            p0 = Path(bound["p0_view"])
            slots = load(p0 / "stage20b_generation_slots.json")["slots"]
            original = {row["slot_id"]: row for row in load(Path(bound["p0"]) / "stage20b_generation_slots.json")["slots"]}
            rngs = {row["slot_id"]: row["banks"]["A"] for row in load(p0 / "stage20b_rng_manifest.json")["records"]}
            black_slots = [row for row in slots if row["class"] == "black"]
            require(len(black_slots) == 40, "Black Bank-A slots changed")
            for index, slot in enumerate(black_slots):
                item_started = time.monotonic()
                frozen = original[slot["slot_id"]]
                rng = rngs[slot["slot_id"]]
                require(rng["non_rng_metadata_sha256"] == hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                        "frozen slot changed")
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
                try:
                    image, trace = sample(pipe, slot, rng, protocol, device)
                finally:
                    handle.remove()
                image_path = OUTPUT / "images/bank_A" / (slot["slot_id"] + ".png")
                label_path = OUTPUT / "images/bank_A" / (slot["slot_id"] + ".txt")
                image.save(image_path, format="PNG")
                label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                width, height = frozen["geometry"]["final_canvas_wh"]
                record = {
                    "slot_id": slot["slot_id"], "class": "black", "bank": "A",
                    "frozen_slot": frozen, "runtime_slot": slot,
                    "image_path": image_path.relative_to(OUTPUT).as_posix(), "image_sha256": sha(image_path),
                    "label_path": label_path.relative_to(OUTPUT).as_posix(), "label_sha256": sha(label_path),
                    "roi_xyxy": crop_box(frozen["bbox"], width, height),
                    "sample_seed": rng["sample_seed"], "bank_seed": rng["bank_seed"],
                    "checkpoint_tree_SHA256": checkpoint_binding["checkpoint"]["tree_sha256"],
                    "morphology_alignment_enabled": True, "injection_gamma": 1.0,
                    "training_auxiliary_loss_weights": calibrated_weights(),
                    "quality_selection": False, "reroll": False,
                    "runtime_seconds": time.monotonic() - item_started, **trace,
                }
                records.append(record)
                save(manifest_path, {"status": "RUNNING", "bank": "A", "records": records})
                print(f"[生成] Black BalancedRegion Bank A {index + 1}/40 {slot['slot_id']}", flush=True)
            msdf.detach()
            del msdf, pipe
            gc.collect(); torch.cuda.empty_cache()
        require(len(records) == 40 and len({r["slot_id"] for r in records}) == 40, "generation population incomplete")
        save(manifest_path, {"status": "COMPLETE", "bank": "A", "class": "black", "count": 40,
            "records": records, "all_outputs_retained": True, "reroll_count": 0,
            "selection_count": 0, "runtime_seconds": time.monotonic() - started,
            "generation_protocol_SHA256": sha(Path(bound["p0_view"]) / "stage20b_generation_protocol.json")})
        update_status(status="GENERATION_COMPLETE", generation_count=40, generation_completed_at=now(),
                      reroll_count=0, selection_count=0)
        print("[生成完成] Black BalancedRegion Bank A 40/40；无重抽、替换或筛选。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "bank": "A", "records": records,
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
            "real_roi_count": len(real), "generated_roi_count": len(fake),
            "KID_subsets": 100, "KID_subset_size": options["kid_subset_size"],
            "runtime_seconds": time.monotonic() - started}


def percent(delta, baseline):
    return 100.0 * delta / baseline


def evaluate(device):
    status = load(OUTPUT / "experiment_status.json")
    require(status["status"] == "GENERATION_COMPLETE", "evaluation requires complete generation")
    update_status(status="EVALUATING", evaluation_started_at=now())
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
        lib, runtime_audit = metric_runtime(device)
        manifest = load(OUTPUT / "balanced_region_generation_manifest.json")
        require(manifest["status"] == "COMPLETE" and manifest["count"] == 40, "generation manifest incomplete")
        original = load(SOURCE_RUN / "formal_bank_A_manifest.json")
        original_black = {row["slot_id"]: row for row in original["records"] if row["class"] == "black"}
        gamma1 = load(GAMMA / "gamma_1p00_manifest.json")
        gamma1_black = {row["slot_id"]: row for row in gamma1["records"] if row["class"] == "black"}
        require(set(original_black) == set(gamma1_black) == {row["slot_id"] for row in manifest["records"]},
                "Black slot population mismatch")
        for row in manifest["records"]:
            frozen = original_black[row["slot_id"]]
            require(row["sample_seed"] == frozen["sample_seed"] == gamma1_black[row["slot_id"]]["sample_seed"] and
                    row["bank_seed"] == frozen["bank_seed"] == 42 and
                    row["frozen_slot"] == frozen["frozen_slot"] and row["roi_xyxy"] == gamma1_black[row["slot_id"]]["roi_xyxy"],
                    f"frozen slot mismatch: {row['slot_id']}")
            require(sha(OUTPUT / row["image_path"]) == row["image_sha256"], f"generated image changed: {row['slot_id']}")

        model = lib["Inception"]("inception-v3-compat", ["2048"],
            feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
        values, keys = [], []
        with torch.inference_mode():
            for index, row in enumerate(manifest["records"]):
                with Image.open(OUTPUT / row["image_path"]) as source:
                    image = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
                tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
                value = model(tensor)[0]
                require(tuple(value.shape) == (1, 2048) and bool(torch.isfinite(value).all()), row["slot_id"])
                values.append(value[0].double().cpu().numpy()); keys.append("balanced_region:" + row["slot_id"])
                if (index + 1) % 10 == 0:
                    print(f"[Inception] BalancedRegion Black {index + 1}/40", flush=True)
        new_array = np.stack(values)
        cache = OUTPUT / "balanced_region_inception_features.npz"
        np.savez_compressed(cache, features=new_array)
        save(OUTPUT / "balanced_region_inception_feature_cache.json", {
            "status": "COMPLETE", "keys": keys, "shape": [40, 2048], "stored_dtype": str(new_array.dtype),
            "extraction_dtype": "float32", "sha256": sha(cache), "image_SHA256_checks": 40,
            "runtime": runtime_audit,
        })
        del model; torch.cuda.empty_cache()

        gamma_ledger = load(GAMMA / "gamma_sweep_inception_feature_cache.json")
        gamma_cache = GAMMA / "gamma_sweep_inception_features.npz"
        require(gamma_ledger["sha256"] == sha(gamma_cache) and gamma_ledger["shape"] == [572, 2048],
                "gamma feature cache changed")
        with np.load(gamma_cache, allow_pickle=False) as archive:
            gamma_array = archive["features"].copy()
        gamma_features = {key: torch.from_numpy(gamma_array[i]).double() for i, key in enumerate(gamma_ledger["keys"])}
        raw_ledger = load(EXP7 / "raw_bank_A_inception_feature_cache.json")
        raw_cache = EXP7 / "raw_bank_A_inception_features.npz"
        require(raw_ledger["sha256"] == sha(raw_cache) and raw_ledger["feature_count"] == 80, "Raw feature cache changed")
        with np.load(raw_cache, allow_pickle=False) as archive:
            raw_array = archive["features"].copy()
        raw_features = {key: torch.from_numpy(raw_array[i]).double() for i, key in enumerate(raw_ledger["keys"])}
        noaux_ledger = load(NOAUX / "noaux_inception_feature_cache.json")
        noaux_cache = NOAUX / "noaux_inception_features.npz"
        require(noaux_ledger["sha256"] == sha(noaux_cache) and noaux_ledger["shape"] == [40, 2048],
                "NoAux feature cache changed")
        with np.load(noaux_cache, allow_pickle=False) as archive:
            noaux_array = archive["features"].copy()
        noaux_features = {key: torch.from_numpy(noaux_array[i]).double()
                          for i, key in enumerate(noaux_ledger["keys"])}
        background_ledger = load(EXP10 / "background_only_inception_feature_cache.json")
        background_cache = EXP10 / "background_only_inception_features.npz"
        require(background_ledger["sha256"] == sha(background_cache) and
                background_ledger["shape"] == [40, 2048],
                "BackgroundOnly feature cache changed")
        with np.load(background_cache, allow_pickle=False) as archive:
            background_array = archive["features"].copy()
        background_features = {key: torch.from_numpy(background_array[i]).double()
                               for i, key in enumerate(background_ledger["keys"])}
        new_features = {key: torch.from_numpy(new_array[i]).double() for i, key in enumerate(keys)}

        gamma_protocol = load(GAMMA / "gamma_sweep_protocol.json")
        references = gamma_protocol["development_reference"]["records"]
        real_black = torch.stack([gamma_features["real:" + row["annotation_id"]] for row in references if row["class"] == "black"])
        real_all = torch.stack([gamma_features["real:" + row["annotation_id"]] for row in references])
        slots = load(SOURCE_RUNTIME / "p0_view/stage20b_generation_slots.json")["slots"]
        black_slots = [row for row in slots if row["class"] == "black"]
        flash_slots = [row for row in slots if row["class"] == "flash"]
        black_sets = {
            "Raw SD2": torch.stack([raw_features["raw:roi:A:" + row["slot_id"]] for row in black_slots]),
            "Original MSDF": torch.stack([gamma_features["fake:1.00:" + row["slot_id"]] for row in black_slots]),
            "MSDF-NoAux": torch.stack([noaux_features["noaux:" + row["slot_id"]] for row in black_slots]),
            "MSDF-BackgroundOnly": torch.stack(
                [background_features["background_only:" + row["slot_id"]] for row in black_slots]),
            "MSDF-BalancedRegion": torch.stack([new_features["balanced_region:" + row["slot_id"]] for row in black_slots]),
        }
        overall_sets = {
            "Raw SD2": torch.stack([raw_features["raw:roi:A:" + row["slot_id"]] for row in slots]),
            "Original Class-Adaptive MSDF": torch.stack([
                gamma_features[f"fake:{0.0 if row['class'] == 'flash' else 1.0:.2f}:" + row["slot_id"]] for row in slots]),
            "NoAux-Black Hybrid MSDF": torch.stack(
                [gamma_features["fake:0.00:" + row["slot_id"]] for row in flash_slots] +
                [noaux_features["noaux:" + row["slot_id"]] for row in black_slots]),
            "BackgroundOnly-Black Hybrid MSDF": torch.stack(
                [gamma_features["fake:0.00:" + row["slot_id"]] for row in flash_slots] +
                [background_features["background_only:" + row["slot_id"]] for row in black_slots]),
            "BalancedRegion-Black Hybrid MSDF": torch.stack(
                [gamma_features["fake:0.00:" + row["slot_id"]] for row in flash_slots] +
                [new_features["balanced_region:" + row["slot_id"]] for row in black_slots]),
        }
        black_rows, overall_rows = [], []
        for method, features in black_sets.items():
            result = metric(real_black, features, lib)
            black_rows.append({"Method": method, "Black_KID": result["KID"],
                               "Black_KID_subset_std": result["KID_subset_std"], "Black_FID": result["FID"],
                               "generated_images": 40, "reference_ROIs": 97, "metric_runtime_seconds": result["runtime_seconds"]})
            print(f"[指标] {method} Black KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
        for method, features in overall_sets.items():
            result = metric(real_all, features, lib)
            overall_rows.append({"Method": method, "Overall_KID": result["KID"],
                                 "Overall_KID_subset_std": result["KID_subset_std"], "Overall_FID": result["FID"],
                                 "generated_images": 80, "reference_ROIs": 172, "metric_runtime_seconds": result["runtime_seconds"]})
            print(f"[指标] {method} Overall KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
        write_csv(OUTPUT / "balanced_region_black_metrics.csv", black_rows)
        write_csv(OUTPUT / "balanced_region_overall_metrics.csv", overall_rows)

        def black(method):
            return next(row for row in black_rows if row["Method"] == method)
        def overall(method):
            return next(row for row in overall_rows if row["Method"] == method)
        expected = {
            "Raw_Black_KID": 0.0379734447, "Raw_Black_FID": 161.766924,
            "Full_Black_KID": 0.0800462518, "Full_Black_FID": 208.478914,
            "NoAux_Black_KID": 0.06726729600075611, "NoAux_Black_FID": 199.14360997525102,
            "Background_Black_KID": 0.06100557551585336, "Background_Black_FID": 191.9669145866637,
            "Raw_Overall_KID": 0.03522257098656576, "Raw_Overall_FID": 149.42750331396738,
            "Adaptive_Overall_KID": 0.04989160653660615, "Adaptive_Overall_FID": 165.1311363969408,
            "NoAux_Overall_KID": 0.045893492852877825, "NoAux_Overall_FID": 160.52035112576777,
            "Background_Overall_KID": 0.04294524729498305, "Background_Overall_FID": 154.90536831861988,
        }
        actual = {
            "Raw_Black_KID": black("Raw SD2")["Black_KID"], "Raw_Black_FID": black("Raw SD2")["Black_FID"],
            "Full_Black_KID": black("Original MSDF")["Black_KID"], "Full_Black_FID": black("Original MSDF")["Black_FID"],
            "NoAux_Black_KID": black("MSDF-NoAux")["Black_KID"],
            "NoAux_Black_FID": black("MSDF-NoAux")["Black_FID"],
            "Background_Black_KID": black("MSDF-BackgroundOnly")["Black_KID"],
            "Background_Black_FID": black("MSDF-BackgroundOnly")["Black_FID"],
            "Raw_Overall_KID": overall("Raw SD2")["Overall_KID"], "Raw_Overall_FID": overall("Raw SD2")["Overall_FID"],
            "Adaptive_Overall_KID": overall("Original Class-Adaptive MSDF")["Overall_KID"],
            "Adaptive_Overall_FID": overall("Original Class-Adaptive MSDF")["Overall_FID"],
            "NoAux_Overall_KID": overall("NoAux-Black Hybrid MSDF")["Overall_KID"],
            "NoAux_Overall_FID": overall("NoAux-Black Hybrid MSDF")["Overall_FID"],
            "Background_Overall_KID": overall("BackgroundOnly-Black Hybrid MSDF")["Overall_KID"],
            "Background_Overall_FID": overall("BackgroundOnly-Black Hybrid MSDF")["Overall_FID"],
        }
        tolerances = {name: (5e-11 if name.endswith("KID") else 5e-7) for name in expected}
        errors = {name: abs(actual[name] - expected[name]) for name in expected}
        require(all(errors[name] <= tolerances[name] for name in errors), f"frozen metric regression failed: {errors}")
        regression = {"status": "PASS", "expected": expected, "actual": actual,
                      "absolute_errors": errors, "tolerances": tolerances,
                      "same_reference_feature_population": True,
                      "new_Inception_forward_count": 40}
        previous = black("Original MSDF"); noaux = black("MSDF-NoAux")
        background = black("MSDF-BackgroundOnly")
        new = black("MSDF-BalancedRegion"); raw = black("Raw SD2")
        hybrid = overall("BalancedRegion-Black Hybrid MSDF"); adaptive = overall("Original Class-Adaptive MSDF")
        noaux_hybrid = overall("NoAux-Black Hybrid MSDF")
        background_hybrid = overall("BackgroundOnly-Black Hybrid MSDF")
        raw_overall = overall("Raw SD2")

        def difference(current, baseline, kid_key, fid_key):
            kd = current[kid_key] - baseline[kid_key]; fd = current[fid_key] - baseline[fid_key]
            return {"KID": {"absolute": kd, "relative_percent": percent(kd, baseline[kid_key])},
                    "FID": {"absolute": fd, "relative_percent": percent(fd, baseline[fid_key])}}
        comparisons = {
            "G3_BalancedRegion_minus_G1_Full_Black": difference(new, previous, "Black_KID", "Black_FID"),
            "G3_BalancedRegion_minus_G2_NoAux_Black": difference(new, noaux, "Black_KID", "Black_FID"),
            "G3_BalancedRegion_minus_BackgroundOnly_Black": difference(new, background, "Black_KID", "Black_FID"),
            "G3_BalancedRegion_minus_G0_Raw_Black": difference(new, raw, "Black_KID", "Black_FID"),
            "BalancedRegion_Hybrid_minus_Original_Adaptive_Overall": difference(hybrid, adaptive, "Overall_KID", "Overall_FID"),
            "BalancedRegion_Hybrid_minus_NoAux_Hybrid_Overall": difference(hybrid, noaux_hybrid, "Overall_KID", "Overall_FID"),
            "BalancedRegion_Hybrid_minus_BackgroundOnly_Hybrid_Overall": difference(
                hybrid, background_hybrid, "Overall_KID", "Overall_FID"),
            "BalancedRegion_Hybrid_minus_Raw_Overall": difference(hybrid, raw_overall, "Overall_KID", "Overall_FID"),
        }
        kid_improved = new["Black_KID"] < background["Black_KID"]
        fid_improved = new["Black_FID"] < background["Black_FID"]
        conclusion = ("BALANCED_REGION_EXPLORATORY_IMPROVED" if kid_improved and fid_improved else
                      "BALANCED_REGION_EXPLORATORY_MIXED" if kid_improved or fid_improved else
                      "BALANCED_REGION_EXPLORATORY_NOT_IMPROVED")
        comparison = {
            "status": conclusion, "differences": comparisons,
            "answers": {
                "BalancedRegion_better_than_BackgroundOnly_KID": kid_improved,
                "BalancedRegion_better_than_BackgroundOnly_FID": fid_improved,
                "BalancedRegion_better_than_Full_KID": new["Black_KID"] < previous["Black_KID"],
                "BalancedRegion_better_than_Full_FID": new["Black_FID"] < previous["Black_FID"],
                "Raw_Black_gap_change": {
                    "KID_gap_before": background["Black_KID"] - raw["Black_KID"],
                    "KID_gap_after": new["Black_KID"] - raw["Black_KID"],
                    "FID_gap_before": background["Black_FID"] - raw["Black_FID"],
                    "FID_gap_after": new["Black_FID"] - raw["Black_FID"],
                },
                "hybrid_Overall_KID_improved_vs_original_adaptive": hybrid["Overall_KID"] < adaptive["Overall_KID"],
                "hybrid_Overall_FID_improved_vs_original_adaptive": hybrid["Overall_FID"] < adaptive["Overall_FID"],
                "gradient_conflict_mitigation_worth_studying": kid_improved or fid_improved,
            },
            "interpretation_limit": "Bank A with V2 TRAIN references; possible checkpoint-training overlap; not independent validation.",
            "random_trajectory_limit": load(OUTPUT / "balanced_region_training_audit.json")["random_trajectory_limit"],
        }
        save(OUTPUT / "balanced_region_comparison.json", comparison)
        checks = load(OUTPUT / "balanced_region_preflight.json")
        checks["metric_regression"] = regression
        checks["generation_integrity"] = {"status": "PASS", "slot_checks": 40, "image_SHA256_checks": 40,
            "same_normal_bbox_prompt_seed_scheduler_steps_CFG_crop_paste": True, "reroll_count": 0, "selection_count": 0}
        save(OUTPUT / "balanced_region_integrity_audit.json", checks)
        protocol = load(OUTPUT / "balanced_region_protocol.json")
        protocol.update(status="COMPLETE", checkpoint_sources={
            "Raw_SD2": "Frozen independent Raw checkpoints",
            "Original_MSDF": "Frozen Stage25 Full MSDF Black final_step2000",
            "NoAux_Black": "Frozen Experiment 9 independent Black final_step2000",
            "BackgroundOnly_Black": "Frozen Experiment 10 independent Black final_step2000",
            "BalancedRegion_Black": "Experiment 11 independent Black final_step2000 from public SD2 base",
            "Hybrid_Flash": "Frozen MSDF gamma=0 Flash Bank A",
            "Hybrid_Black": "Experiment 11 BalancedRegion Black Bank A",
        }, feature_reuse={"real_and_original_MSDF": "Gamma Sweep SHA-bound cache",
                          "Raw": "Experiment 7 SHA-bound cache",
                          "NoAux": "Experiment 9 SHA-bound cache",
                          "BackgroundOnly": "Experiment 10 SHA-bound cache",
                          "new_BalancedRegion_Black_forwards": 40})
        save(OUTPUT / "balanced_region_protocol.json", protocol)
        save(OUTPUT / "experiment_status.json", {**load(OUTPUT / "experiment_status.json"),
            "status": conclusion, "completed_at": now(), "evaluation_completed": True,
            "numerical_regression": "PASS", "new_Inception_forward_count": 40,
            "evaluation_runtime_seconds": time.monotonic() - started,
            "followup_started": False})
        create_report(conclusion, black_rows, overall_rows, comparisons, kid_improved, fid_improved)
        print(f"[完成] Experiment 11 status={conclusion}；停止，不启动后续实验。", flush=True)
    except Exception as error:
        update_status(status="EVALUATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise


def create_report(conclusion, black_rows, overall_rows, comparisons, kid_improved, fid_improved):
    b = {row["Method"]: row for row in black_rows}
    o = {row["Method"]: row for row in overall_rows}
    full, noaux = b["Original MSDF"], b["MSDF-NoAux"]
    background = b["MSDF-BackgroundOnly"]
    new, raw = b["MSDF-BalancedRegion"], b["Raw SD2"]
    adaptive = o["Original Class-Adaptive MSDF"]
    noaux_hybrid = o["NoAux-Black Hybrid MSDF"]
    background_hybrid = o["BackgroundOnly-Black Hybrid MSDF"]
    hybrid = o["BalancedRegion-Black Hybrid MSDF"]
    gap_before_kid = background["Black_KID"] - raw["Black_KID"]
    gap_after_kid = new["Black_KID"] - raw["Black_KID"]
    gap_before_fid = background["Black_FID"] - raw["Black_FID"]
    gap_after_fid = new["Black_FID"] - raw["Black_FID"]
    calibration = load(OUTPUT / "gradient_balance_calibration.json")
    unet, msdf = calibration["groups"]["UNet"], calibration["groups"]["MSDF"]
    lambda_region = calibration["lambda_region"]
    lines = [
        "# Experiment 11 — Gradient-Balanced Region Supervision", "",
        f"Status: `{conclusion}`.", "",
        "## Phase A — frozen Region calibration", "",
        "Calibration used the exact 12 Black sample/reference pairs frozen by Experiment 10 (8 self-reference and 4 cross-reference) and the frozen BackgroundOnly step2000 checkpoint. It restored the same RNG state for the base and Region forwards and performed zero optimizer updates.", "",
        f"The calibration passed. `lambda_region = {lambda_region:.17g}` from `min(0.02, r_unet, r_msdf)`, with `r_unet = {calibration['r_unet']:.17g}` and `r_msdf = {calibration['r_msdf']:.17g}`. The full-precision value was frozen before training.", "",
        "| Parameter group | Median base norm | Median Region norm | Base/Region ratio | Mean cosine | Valid | Zero | Nonfinite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| UNet | {unet['median_base_gradient_norm']:.10g} | {unet['median_region_gradient_norm']:.10g} | {unet['ratio_base_over_region']:.10g} | {unet['mean_cosine']:.6f} | {unet['valid_samples']} | {unet['zero_gradient_samples']} | {unet['nonfinite_gradient_samples']} |",
        f"| MSDF | {msdf['median_base_gradient_norm']:.10g} | {msdf['median_region_gradient_norm']:.10g} | {msdf['ratio_base_over_region']:.10g} | {msdf['mean_cosine']:.6f} | {msdf['valid_samples']} | {msdf['zero_gradient_samples']} | {msdf['nonfinite_gradient_samples']} |", "",
        "## Phase B — BalancedRegion training", "",
        f"G3 was trained independently from the same public SD2 base for exactly 2000 successful optimizer updates. The frozen objective was `L_denoising + 0.20 * L_background + {lambda_region:.17g} * L_region`; Reconstruction, Structure, Multiscale and Support weights were zero. Morphology alignment, both branches, dynamic gate and up-block injection remained enabled, with gamma 1.0.", "",
        "## Black results", "",
        "| Method | Black KID ↓ | Black FID ↓ |", "| --- | ---: | ---: |",
    ]
    for name in ("Raw SD2", "Original MSDF", "MSDF-NoAux", "MSDF-BackgroundOnly", "MSDF-BalancedRegion"):
        lines.append(f"| {name} | {b[name]['Black_KID']:.10f} | {b[name]['Black_FID']:.6f} |")
    lines += ["", "## Overall results", "",
              "The hybrid combines the frozen gamma=0 Flash checkpoint/images with the independently trained BalancedRegion Black checkpoint/images; it is not one jointly trained model.", "",
              "| Method | Overall KID ↓ | Overall FID ↓ |", "| --- | ---: | ---: |"]
    for name in ("Raw SD2", "Original Class-Adaptive MSDF", "NoAux-Black Hybrid MSDF", "BackgroundOnly-Black Hybrid MSDF", "BalancedRegion-Black Hybrid MSDF"):
        lines.append(f"| {name} | {o[name]['Overall_KID']:.10f} | {o[name]['Overall_FID']:.6f} |")
    lines += ["", "## Comparisons", "",
              "Negative differences mean BalancedRegion/Hybrid is lower and better.", "",
              "| Comparison | Δ KID | Relative KID | Δ FID | Relative FID |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in comparisons.items():
        lines.append(f"| {name.replace('_', ' ')} | {row['KID']['absolute']:+.10f} | {row['KID']['relative_percent']:+.3f}% | {row['FID']['absolute']:+.6f} | {row['FID']['relative_percent']:+.3f}% |")
    lines += ["", "## Required findings", "",
              f"1. The actual frozen `lambda_region` was `{lambda_region:.17g}`.",
              "2. Region gradient calibration passed with finite, nonzero gradients in both UNet and MSDF groups.",
              f"3. Black KID {'was' if new['Black_KID'] < 0.06100558 else 'was not'} below 0.06100558 ({new['Black_KID']:.10f}).",
              f"4. Black FID {'was' if new['Black_FID'] < 191.966915 else 'was not'} below 191.966915 ({new['Black_FID']:.6f}).",
              f"5. Overall KID {'was' if hybrid['Overall_KID'] < 0.04294525 else 'was not'} below 0.04294525 ({hybrid['Overall_KID']:.10f}).",
              f"6. Overall FID {'was' if hybrid['Overall_FID'] < 154.905368 else 'was not'} below 154.905368 ({hybrid['Overall_FID']:.6f}).",
              f"7. Relative to Raw SD2, the Black KID gap changed from {gap_before_kid:+.10f} to {gap_after_kid:+.10f}, and the Black FID gap changed from {gap_before_fid:+.6f} to {gap_after_fid:+.6f}.",
              f"8. Adding calibrated Region supervision {'improved both Black distribution metrics' if kid_improved and fid_improved else 'improved one Black distribution metric' if kid_improved or fid_improved else 'did not improve either Black distribution metric'} relative to BackgroundOnly.",
              f"9. Further work on gradient-balanced auxiliary supervision is {'supported for independent validation' if kid_improved or fid_improved else 'not supported by this screening result'}.", "",
              "## Interpretation boundary", "",
              "This is a Bank-A development ablation evaluated against V2 TRAIN references that may overlap generator training. It is not an independent test result and cannot establish final-paper superiority.", "",
              "Experiment 10 recorded initialization and RNG summaries. Experiment 11 reused the same initialization rule, seed, sample-order rule and reference rule, but complete byte-for-byte trajectory equivalence is not assumed.", "",
              "The experiment stops here. No other loss restoration, weight search, new Flash model, architecture change, extra seed, Bank B/C, BootstrapGuard or detector training was started.", ""]
    (OUTPUT / "BALANCED_REGION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "preflight", "train", "generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 11 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "calibrate":
        gradient_probe(args.device)
    elif args.phase == "preflight":
        preflight(args.device)
    elif args.phase == "train":
        train(args.device)
    elif args.phase == "generate":
        generate(args.device)
    else:
        evaluate(args.device)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
