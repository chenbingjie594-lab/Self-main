"""Experiment 15: train one PixelOnly Black MSDF model from the public SD2 base.

The training path deliberately reuses Experiment 10's bound data, model
construction, real microbatch, historical loss kernel, and checkpoint layout.
Only the declared PixelOnly training controls differ: the latent feature is
zeroed, ``encoder.*`` is frozen and excluded from AdamW, and branch dropout is
fixed to zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import gc
import hashlib
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

import run_plastic_bomo_msdf_black_gradient_background as x10
import run_plastic_bomo_msdf_branch_ablation as x14
import run_plastic_bomo_msdf_multibank_stability as x12


BASE_COMMIT = "8644bc5d79986b7efd90913130da9395314a64de"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_pixelonly_training"
CONFIG = REPO / "configs/plastic_bomo_msdf_pixelonly_training.json"
BACKGROUND = REPO / "results_for_gpt/plastic_bomo_msdf_black_gradient_background"
WEIGHTS = {"region": 0.0, "reconstruction": 0.0, "structure": 0.0,
           "background": 0.20, "scale": 0.0, "support": 0.0}
ABLATION = {"pixel_branch": True, "latent_branch": False,
            "morphology_alignment": True, "up_block_mask": [True] * 4,
            "residual_scale": 1.0}

load, save, sha, canonical, hvalue, require = (
    x10.load, x10.save, x10.sha, x10.canonical, x10.hvalue, x10.require)


def source_hashes():
    names = (
        "train_rda.py", "tools/stage25_msdf_generator.py",
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py",
        "tools/run_plastic_bomo_msdf_black_gradient_background.py",
        "tools/run_plastic_bomo_msdf_pixelonly_training.py",
        "configs/plastic_bomo_msdf_pixelonly_training.json",
    )
    return {name: sha(REPO / name) for name in names}


def status(**values):
    path = OUTPUT / "experiment_status.json"
    row = load(path)
    row.update(values)
    save(path, row)


def experiment_config():
    value = load(CONFIG)
    require(value["base_commit"] == BASE_COMMIT and value["class"] == "black",
            "Experiment 15 identity changed")
    require(value["branch"] == {**ABLATION, "injection_gamma": 1.0,
                                "branch_dropout": 0.0, "latent_encoder_frozen": True},
            "PixelOnly branch configuration changed")
    require(value["auxiliary_loss_weights"] == WEIGHTS, "loss weights changed")
    return value


def training_binding():
    bound, cfg = x10.binding()
    cfg = copy.deepcopy(cfg)
    cfg["MSDF_preserved_defaults"]["branch_dropout"] = 0.0
    for name, key in x10.AUXILIARY_WEIGHT_KEYS.items():
        cfg["MSDF_preserved_defaults"][key] = WEIGHTS[name]
    return bound, cfg


@contextmanager
def gpu_runtime(device):
    # Reuse Experiment 10's offline, final-eval-denial, CUDA-version and device
    # checks, then apply the one frozen configuration change before construct().
    with x10.gpu_context(device) as (torch, bound, original_cfg):
        cfg = copy.deepcopy(original_cfg)
        cfg["MSDF_preserved_defaults"]["branch_dropout"] = 0.0
        for name, key in x10.AUXILIARY_WEIGHT_KEYS.items():
            cfg["MSDF_preserved_defaults"][key] = WEIGHTS[name]
        yield torch, bound, cfg


def configure_pixelonly(msdf):
    require(msdf.branch_dropout == 0.0, "branch_dropout must be zero before training")
    state = msdf.set_ablation(ABLATION)
    msdf.set_injection_gamma(1.0)
    msdf.encoder.requires_grad_(False)
    require(state == ABLATION and msdf.injection_gamma == 1.0,
            "PixelOnly runtime state mismatch")
    require(all(not p.requires_grad for p in msdf.encoder.parameters()),
            "latent encoder freeze failed")
    return state


def state_sha(module):
    return x10.state_sha(module)


def trainable_record(module):
    rows = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    return {"parameter_names": [name for name, _ in rows],
            "parameter_tensor_count": len(rows),
            "parameter_count": sum(p.numel() for _, p in rows)}


def gradient_record(named_parameters, torch):
    disconnected, nonfinite, zero, nonzero = [], [], [], []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            disconnected.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif bool(torch.count_nonzero(parameter.grad)):
            nonzero.append(name)
        else:
            zero.append(name)
    return {"all_connected": not disconnected, "all_finite": not nonfinite,
            "disconnected": disconnected, "nonfinite": nonfinite,
            "zero_but_connected_parameter_names": zero,
            "nonzero_parameter_names": nonzero}


def optimizer_membership(optimizer, module):
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    return {name: id(p) in ids for name, p in module.named_parameters()}


def cpu_branch_check():
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter

    torch.manual_seed(42)
    adapter = MSDFAdapter(4, 8, [8, 8, 8, 8], branch_dropout=0.0).train()
    before = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    configure_pixelonly(adapter)
    active = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(active, lr=1e-4)
    membership = optimizer_membership(optimizer, adapter)
    captured = {}

    def capture(_module, inputs):
        captured["fusion_input"] = inputs[0].detach()

    handle = adapter.fusion.register_forward_pre_hook(capture)
    latent = torch.randn(1, 4, 8, 8)
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    mask = torch.zeros(1, 1, 64, 64); mask[:, :, 18:44, 20:42] = 1
    prior = torch.zeros(1, 1, 8, 8); prior[:, :, 2:7, 2:7] = 1
    gates = adapter.prepare(latent, mask, prior, torch.tensor([500]), 1000, False, pixels)
    handle.remove()
    joined = captured["fusion_input"]
    latent_input, pixel_input = joined.chunk(2, dim=1)
    tensors = (joined, gates, adapter.last_pixel_support,
               adapter.last_reference_support, adapter.last_aligned_support)
    checks = {
        "pixel_branch_enabled": adapter.ablation["pixel_branch"],
        "latent_branch_disabled": not adapter.ablation["latent_branch"],
        "latent_fusion_input_exactly_zero": int(torch.count_nonzero(latent_input)) == 0,
        "pixel_fusion_input_nonzero": int(torch.count_nonzero(pixel_input)) > 0,
        "pixel_support_nonzero": int(torch.count_nonzero(adapter.last_pixel_support)) > 0,
        "morphology_alignment_enabled": adapter.ablation["morphology_alignment"],
        "injection_gamma": adapter.injection_gamma,
        "branch_dropout": adapter.branch_dropout,
        "latent_encoder_frozen": all(not p.requires_grad for p in adapter.encoder.parameters()),
        "latent_encoder_excluded_from_optimizer": all(
            not membership[name] for name, _ in adapter.named_parameters() if name.startswith("encoder.")),
        "fusion_input_channels": joined.shape[1],
        "expected_full_fusion_input_channels": 2 * adapter.hidden_dim,
        "all_outputs_finite": all(bool(torch.isfinite(value).all()) for value in tensors),
        "optimizer_step_count": 0,
        "parameters_unchanged": all(torch.equal(before[name], p) for name, p in adapter.named_parameters()),
    }
    require(all((checks[k] is True) for k in (
        "pixel_branch_enabled", "latent_branch_disabled", "latent_fusion_input_exactly_zero",
        "pixel_fusion_input_nonzero", "pixel_support_nonzero", "morphology_alignment_enabled",
        "latent_encoder_frozen", "latent_encoder_excluded_from_optimizer", "all_outputs_finite",
        "parameters_unchanged")), "CPU PixelOnly check failed")
    require(checks["injection_gamma"] == 1.0 and checks["branch_dropout"] == 0.0 and
            checks["fusion_input_channels"] == checks["expected_full_fusion_input_channels"],
            "CPU frozen scalar/dimension check failed")
    return {"status": "PASS", **checks}


def audit():
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    (OUTPUT / "logs").mkdir(parents=True)
    (OUTPUT / "images").mkdir()
    config = experiment_config()
    bound, cfg = training_binding()
    # Validate every inherited Experiment 10 value before changing branch dropout.
    _, historical_cfg = x10.binding()
    historical = x10.validate_protocol(historical_cfg)
    assets = x10.verify_assets(bound)
    cpu = cpu_branch_check()
    protocol = {
        "status": "FROZEN_BEFORE_GPU_PREFLIGHT",
        "experiment": config["experiment"], "base_commit": BASE_COMMIT,
        "single_new_model": "G3 PixelOnly Training Black", "training_count": 1,
        "fresh_public_SD2_start": True, "continued_from_G1": False,
        "data": "Plastic_Bomo V2 TRAIN Black frozen allowlist, 97 ROI",
        "training": config["training"], "branch": config["branch"],
        "auxiliary_loss_weights": WEIGHTS,
        "training_objective": "L_denoising + 0.20 * L_background",
        "comparison_boundary": (
            "Compared with Full, PixelOnly also freezes the latent encoder and changes branch_dropout "
            "from 0.2 to 0.0; random-number consumption and training trajectory therefore differ."),
        "generation": config["generation"],
        "evaluation": {"reference": {"flash": 75, "black": 97, "overall": 172},
                       "independent_validation": False,
                       "ROI": "bbox plus 25 percent context with frozen floor/ceil clipping",
                       "Inception": "torch-fidelity inception-v3-compat 2048D FP32; metrics FP64 CPU",
                       "KID": {"subsets": 100, "seed": 42, "degree": 3,
                               "subset_size": "min(50,nreal,nfake)"}},
        "historical_protocol_validation": historical, "asset_audit": assets,
        "source_code_SHA256": source_hashes(), "final_eval_read_or_used": False,
    }
    save(OUTPUT / "pixelonly_training_protocol.json", protocol)
    save(OUTPUT / "pixelonly_branch_definition.json", {
        "status": "FROZEN", "training": config["branch"],
        "fusion_input": "concatenate([zero_latent, pixel_feature]) at unchanged 2*hidden_dim width",
        "pixel_support": "enabled", "dynamic_gate": "enabled", "up_blocks": 4,
        "latent_encoder_storage_retained": True, "lightweight_or_speed_claim": False})
    save(OUTPUT / "pixelonly_preflight.json", {"status": "CPU_PASS_GPU_PENDING", "CPU": cpu,
                                               "optimizer_steps": 0})
    save(OUTPUT / "experiment_status.json", {
        "status": "AUDIT_PASS", "base_commit": BASE_COMMIT, "created_at": x10.now(),
        "training_count": 0, "successful_updates": 0, "generation_count": 0,
        "reroll_count": 0, "selection_count": 0, "final_eval_read_or_used": False,
        "BootstrapGuard": False, "detector_training_started": False})
    print("[审计通过] PixelOnly冻结协议、97个Black ROI、公开SD2绑定和CPU分支检查通过。", flush=True)


def preflight(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "AUDIT_PASS",
            "GPU preflight requires audit PASS")
    audit_row = load(OUTPUT / "pixelonly_preflight.json")
    started = time.monotonic()
    try:
        with gpu_runtime(device) as (torch, bound, cfg):
            from stage25_msdf_generator import construct, microbatch, frozen_hashes
            torch.cuda.reset_peak_memory_stats()
            objects = construct(torch, bound, cfg, "black")
            pipe, msdf = objects[:2]
            initial_adapter = state_sha(msdf)
            expected_initial = load(BACKGROUND / "background_only_training_audit.json")["initial_adapter_state_SHA256"]
            require(initial_adapter == expected_initial,
                    "full adapter initialization differs from Experiment 10")
            latent_initial = state_sha(msdf.encoder)
            configure_pixelonly(msdf)
            active_msdf = [(name, p) for name, p in msdf.named_parameters() if p.requires_grad]
            latent = [(name, p) for name, p in msdf.named_parameters() if name.startswith("encoder.")]
            require(active_msdf and latent and all(not p.requires_grad for _, p in latent),
                    "trainable/frozen split invalid")
            target = cfg["common_generator_training_target"]
            optimizer = torch.optim.AdamW([
                {"params": list(pipe.unet.parameters()), "lr": target["learning_rate"]},
                {"params": [p for _, p in active_msdf],
                 "lr": cfg["MSDF_preserved_defaults"]["MSDF_learning_rate"]},
            ], betas=(target["adam_beta1"], target["adam_beta2"]),
                weight_decay=target["adam_weight_decay"], eps=target["adam_epsilon"])
            membership = optimizer_membership(optimizer, msdf)
            require(all(not membership[name] for name, _ in latent),
                    "latent encoder entered optimizer")
            before_frozen = frozen_hashes(pipe)
            rows = sorted((row for row in load(Path(bound["p0_view"]) /
                          "generator_train_allowlist.json")["inputs"] if row["class"] == "black"),
                          key=lambda row: row["annotation_id"])
            require(len(rows) == 97, "Black TRAIN population changed")
            captured = {}
            def capture(_module, inputs): captured["fusion"] = inputs[0].detach()
            handle = msdf.fusion.register_forward_pre_hook(capture)
            loss, losses, shapes = microbatch(objects, rows[0], rows[0], bound, cfg, torch)
            handle.remove()
            joined = captured["fusion"]; latent_input, pixel_input = joined.chunk(2, 1)
            expected = losses["denoising"] + 0.20 * losses["background"]
            delta = abs(float(loss.detach()) - expected)
            require(delta <= 1e-7, "objective differs from denoising + 0.20*background")
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            scaler.scale(loss / target["gradient_accumulation_steps"]).backward()
            scaler.unscale_(optimizer)
            unet_grads = gradient_record(list(pipe.unet.named_parameters()), torch)
            active_grads = gradient_record(active_msdf, torch)
            latent_none = all(p.grad is None for _, p in latent)
            require(unet_grads["all_finite"] and unet_grads["nonzero_parameter_names"],
                    "UNet gradient check failed")
            require(active_grads["all_connected"] and active_grads["all_finite"] and
                    active_grads["nonzero_parameter_names"], "active MSDF gradient check failed")
            require(latent_none and state_sha(msdf.encoder) == latent_initial,
                    "latent encoder changed or received gradients")
            require(int(torch.count_nonzero(latent_input)) == 0 and
                    int(torch.count_nonzero(pixel_input)) > 0, "GPU fusion input check failed")
            require(msdf.last_pixel_support is not None and
                    bool(torch.isfinite(msdf.last_pixel_support).all()) and
                    int(torch.count_nonzero(msdf.last_pixel_support)) > 0,
                    "GPU Pixel Support check failed")
            require(before_frozen == frozen_hashes(pipe), "VAE/Text Encoder changed")
            audit_row.update({
                "status": "PASS", "completed_at": x10.now(), "wall_seconds": time.monotonic() - started,
                "GPU": {"device": device, "name": torch.cuda.get_device_name(device),
                        "loss": float(loss.detach()), "denoising_loss": losses["denoising"],
                        "background_loss": losses["background"], "objective_delta": delta,
                        "raw_auxiliary_losses": {name: losses[name] for name in x10.AUXILIARY_NAMES},
                        "weighted_auxiliary_losses": {"background": 0.20},
                        "other_five_weighted_contribution": 0.0,
                        "tensor_shapes": shapes, "fusion_latent_exact_zero": True,
                        "fusion_pixel_nonzero": True, "pixel_support_nonzero_finite": True,
                        "UNet_gradients": unet_grads, "active_MSDF_gradients": active_grads,
                        "latent_encoder_gradients_all_none": latent_none,
                        "zero_gradient_interpretation": (
                            "Parameters listed as zero_but_connected remain in the autograd graph; "
                            "zero initialization can make their first-batch values zero."),
                        "optimizer_step_called": False,
                        "peak_GPU_memory_bytes": torch.cuda.max_memory_allocated()},
                "initial_adapter_state_SHA256": initial_adapter,
                "matches_Experiment10_initial_adapter": True,
                "frozen_latent_encoder_SHA256": latent_initial,
                "trainable_MSDF": trainable_record(msdf),
                "latent_encoder_optimizer_excluded": True,
                "branch_dropout": msdf.branch_dropout,
                "ablation": dict(msdf.ablation), "gamma": msdf.injection_gamma,
                "formal_training_authorized": True,
            })
            msdf.detach()
            del loss, optimizer, scaler, msdf, pipe, objects
            gc.collect(); torch.cuda.empty_cache()
        save(OUTPUT / "pixelonly_preflight.json", audit_row)
        save(OUTPUT / "pixelonly_gradient_audit.json", {
            "status": "PREFLIGHT_PASS", "optimizer_steps": 0,
            "GPU": audit_row["GPU"], "completed_at": x10.now()})
        status(status="PREFLIGHT_PASS", preflight_completed_at=x10.now())
        print("[检查通过] PixelOnly真实Black单batch、GradScaler反传、冻结Latent及优化器排除全部通过；未更新参数。", flush=True)
    except Exception as error:
        audit_row.update(status="FAIL", error=f"{type(error).__name__}: {error}",
                         wall_seconds=time.monotonic() - started)
        save(OUTPUT / "pixelonly_preflight.json", audit_row)
        status(status="PREFLIGHT_FAILED", failure=audit_row["error"])
        raise


def checkpoint_payload(msdf):
    return {
        "format_version": 3, "latent_channels": msdf.latent_channels,
        "hidden_dim": msdf.hidden_dim, "block_channels": msdf.block_channels,
        "context_scale": msdf.context_scale, "max_injection": msdf.max_injection,
        "max_residual_ratio": msdf.max_residual_ratio,
        "branch_dropout": msdf.branch_dropout,
        "pixel_support_weight": msdf.pixel_support_weight,
        "morphology_alignment_enabled": True, "injection_gamma": 1.0,
        "training_pixel_branch": True, "training_latent_branch": False,
        "training_branch_dropout": 0.0, "latent_encoder_frozen": True,
        "auxiliary_loss_weights": dict(WEIGHTS),
        "state_dict": {name: value.detach().cpu() for name, value in msdf.state_dict().items()},
    }


def train(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "PREFLIGHT_PASS",
            "formal training requires PASS preflight")
    claim = OUTPUT / "logs/train_claim.json"
    require(not claim.exists(), "the one permitted training run has already been claimed")
    save(claim, {"started_at": x10.now(), "planned_successful_updates": 2000,
                 "class": "black", "pixel_branch": True, "latent_branch": False})
    status(status="TRAINING", training_count=1, training_started_at=x10.now())
    started = time.monotonic()
    audit_row = {
        "status": "RUNNING", "class": "black", "successful_updates": 0,
        "scheduled_attempts": 0, "AMP_skips": 0, "microbatch_draws": 0,
        "checkpoint_saved": False, "automatic_resume": False,
        "training_pixel_branch": True, "training_latent_branch": False,
        "training_branch_dropout": 0.0, "latent_encoder_frozen": True,
        "morphology_alignment_enabled": True, "injection_gamma": 1.0,
        "auxiliary_loss_weights": dict(WEIGHTS),
        "training_objective": "conditional_loss + 0.20 * msdf_background_loss",
        "final_eval_content_read": 0,
    }
    audit_path = OUTPUT / "pixelonly_training_audit.json"
    save(audit_path, audit_row)
    try:
        with gpu_runtime(device) as (torch, bound, cfg):
            from stage25_msdf_generator import construct, microbatch, frozen_hashes
            from stage25_msdf_contract import tree_binding, verified_copy
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            torch.cuda.reset_peak_memory_stats()
            objects = construct(torch, bound, cfg, "black")
            pipe, msdf = objects[:2]
            initial_adapter = state_sha(msdf)
            expected_initial = load(BACKGROUND / "background_only_training_audit.json")["initial_adapter_state_SHA256"]
            require(initial_adapter == expected_initial, "adapter initialization changed")
            initial_latent = state_sha(msdf.encoder)
            configure_pixelonly(msdf)
            require(state_sha(msdf) == initial_adapter, "PixelOnly controls changed initialized weights")
            before_frozen = frozen_hashes(pipe)
            rows = sorted((row for row in load(Path(bound["p0_view"]) /
                          "generator_train_allowlist.json")["inputs"] if row["class"] == "black"),
                          key=lambda row: row["annotation_id"])
            require(len(rows) == 97, "Black TRAIN population changed")
            target = cfg["common_generator_training_target"]
            active_msdf = [(name, p) for name, p in msdf.named_parameters() if p.requires_grad]
            latent = [(name, p) for name, p in msdf.named_parameters() if name.startswith("encoder.")]
            optimizer = torch.optim.AdamW([
                {"params": list(pipe.unet.parameters()), "lr": target["learning_rate"]},
                {"params": [p for _, p in active_msdf],
                 "lr": cfg["MSDF_preserved_defaults"]["MSDF_learning_rate"]},
            ], betas=(target["adam_beta1"], target["adam_beta2"]),
                weight_decay=target["adam_weight_decay"], eps=target["adam_epsilon"])
            membership = optimizer_membership(optimizer, msdf)
            require(all(not membership[name] for name, _ in latent), "latent entered optimizer")
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            actual = [0]
            step_hook = optimizer.register_step_post_hook(
                lambda *_args, **_kwargs: actual.__setitem__(0, actual[0] + 1))
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            sums = {"training_loss": 0.0, "gradient_norm": 0.0,
                    **{"auxiliary_" + name: 0.0 for name in
                       ("region", "reconstruction", "structure", "background", "scale", "support", "denoising")}}
            value_count = 0
            audit_row.update({
                "precision": "fp16 autocast, FP32 UNet and active MSDF masters",
                "optimizer_groups": [
                    {"name": "UNet", "learning_rate": target["learning_rate"],
                     "parameter_count": sum(p.numel() for p in pipe.unet.parameters())},
                    {"name": "MSDF_without_latent_encoder",
                     "learning_rate": cfg["MSDF_preserved_defaults"]["MSDF_learning_rate"],
                     "parameter_count": sum(p.numel() for _, p in active_msdf)}],
                "trainable_UNet": trainable_record(pipe.unet),
                "trainable_MSDF": trainable_record(msdf),
                "frozen_latent_parameter_names": [name for name, _ in latent],
                "epoch_order_SHA256": [], "epoch_tail_divisor": 4,
                "input_population_SHA256": hvalue([row["annotation_id"] for row in rows]),
                "initial_adapter_state_SHA256": initial_adapter,
                "frozen_latent_encoder_initial_SHA256": initial_latent,
                "initial_UNet_weight_file_SHA256": sha(Path(bound["base_model"]) /
                                                         "unet/diffusion_pytorch_model.safetensors"),
                "initialization_matches_Experiment10": True,
                "initial_RNG_state_summary": {
                    "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    "torch_CPU": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    "torch_CUDA": hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()},
                "source_code_SHA256": source_hashes(),
                "random_trajectory_boundary": (
                    "branch_dropout=0 changes random-number consumption relative to Full branch_dropout=0.2; "
                    "stepwise noise equality is not claimed."),
                "active_MSDF_nonzero_gradient_attempts": 0,
                "latent_gradient_none_attempts": 0,
            })
            save(audit_path, audit_row)
            epoch, consecutive_skips = 0, 0
            log_path = OUTPUT / "logs/pixelonly_black_updates.jsonl"
            with log_path.open("x", encoding="utf-8") as log:
                while actual[0] < target["max_train_steps"]:
                    order = torch.randperm(len(rows), generator=torch.Generator(device="cpu").manual_seed(
                        target["seed"] + epoch)).tolist()
                    audit_row["epoch_order_SHA256"].append(
                        hvalue([rows[index]["annotation_id"] for index in order]))
                    for offset in range(0, len(order), target["gradient_accumulation_steps"]):
                        optimizer.zero_grad(set_to_none=True)
                        records = []
                        for index in order[offset:offset + target["gradient_accumulation_steps"]]:
                            reference_index = index
                            if random.random() < cfg["MSDF_preserved_defaults"]["reference_cross_probability"]:
                                reference_index = (index + random.randint(1, len(rows) - 1)) % len(rows)
                            loss, losses, shapes = microbatch(
                                objects, rows[index], rows[reference_index], bound, cfg, torch)
                            expected = losses["denoising"] + 0.20 * losses["background"]
                            delta = abs(float(loss.detach()) - expected)
                            require(delta <= 1e-7, "training objective changed")
                            scaler.scale(loss / target["gradient_accumulation_steps"]).backward()
                            loss_value = float(loss.detach())
                            sums["training_loss"] += loss_value
                            for name, value in losses.items():
                                sums["auxiliary_" + name] += value
                            value_count += 1
                            records.append({**shapes, "loss": loss_value,
                                            "denoising_loss": losses["denoising"],
                                            "background_loss": losses["background"],
                                            "objective_minus_expected_abs": delta,
                                            "raw_auxiliary_losses": {
                                                name: losses[name] for name in x10.AUXILIARY_NAMES}})
                            audit_row["microbatch_draws"] += 1
                            del loss
                        audit_row["scheduled_attempts"] += 1
                        scaler.unscale_(optimizer)
                        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
                        active_gradients = [p.grad for _, p in active_msdf if p.grad is not None]
                        active_finite = bool(active_gradients) and all(
                            bool(torch.isfinite(grad).all()) for grad in active_gradients)
                        active_nonzero = sum(bool(torch.count_nonzero(grad)) for grad in active_gradients
                                             if bool(torch.isfinite(grad).all()))
                        latent_none = all(p.grad is None for _, p in latent)
                        require(latent_none, "latent encoder received a gradient")
                        unet_gradients = [p.grad for p in pipe.unet.parameters() if p.grad is not None]
                        unet_finite = bool(unet_gradients) and all(
                            bool(torch.isfinite(grad).all()) for grad in unet_gradients)
                        unet_nonzero = sum(bool(torch.count_nonzero(grad)) for grad in unet_gradients
                                           if bool(torch.isfinite(grad).all()))
                        norm = torch.nn.utils.clip_grad_norm_(parameters, target["max_grad_norm"])
                        require(not finite or bool(torch.isfinite(norm)), "invalid gradient norm")
                        before_step, before_scale = actual[0], scaler.get_scale()
                        scaler.step(optimizer); scaler.update()
                        success = actual[0] == before_step + 1
                        require(not success or (active_finite and active_nonzero > 0 and
                                                unet_finite and unet_nonzero > 0 and latent_none),
                                "successful update lacks effective finite active gradients")
                        audit_row["successful_updates"] = actual[0]
                        audit_row["AMP_skips"] += int(not success)
                        audit_row["active_MSDF_nonzero_gradient_attempts"] += int(active_nonzero > 0)
                        audit_row["latent_gradient_none_attempts"] += int(latent_none)
                        audit_row["active_MSDF_gradient_status"] = {
                            "connected_parameter_tensors": len(active_gradients),
                            "trainable_parameter_tensors": len(active_msdf),
                            "all_connected": len(active_gradients) == len(active_msdf),
                            "all_finite": active_finite, "nonzero_parameter_tensors": active_nonzero}
                        audit_row["UNet_gradient_status"] = {
                            "connected_parameter_tensors": len(unet_gradients),
                            "all_finite": unet_finite, "nonzero_parameter_tensors": unet_nonzero}
                        audit_row["latent_encoder_gradient_status"] = {"all_none": latent_none}
                        consecutive_skips = 0 if success else consecutive_skips + 1
                        require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0 and
                                consecutive_skips < 20, "invalid scaler or 20 consecutive AMP skips")
                        norm_value = float(norm) if bool(torch.isfinite(norm)) else None
                        if norm_value is not None:
                            sums["gradient_norm"] += norm_value
                        log.write(canonical({"epoch": epoch, "offset": offset,
                            "attempt": audit_row["scheduled_attempts"], "successful_updates": actual[0],
                            "AMP_skip": not success, "gradients_finite": finite,
                            "latent_gradients_all_none": latent_none, "gradient_norm": norm_value,
                            "scale_before": before_scale, "scale_after": scaler.get_scale(),
                            "microbatches": records}) + "\n")
                        log.flush()
                        audit_row["running_means"] = {
                            name: value / max(value_count, 1) for name, value in sums.items()}
                        audit_row["peak_GPU_memory_bytes"] = torch.cuda.max_memory_allocated()
                        save(audit_path, audit_row)
                        if actual[0] % 25 == 0 or not success:
                            print(f"[训练] PixelOnly Black successful={actual[0]} "
                                  f"attempts={audit_row['scheduled_attempts']} "
                                  f"skips={audit_row['AMP_skips']}", flush=True)
                        if actual[0] == target["max_train_steps"]:
                            require(success and finite and norm_value is not None, "last update invalid")
                            break
                    epoch += 1
            step_hook.remove()
            require(actual[0] == 2000 and all(bool(torch.isfinite(p).all()) for p in parameters),
                    "2000-update/finiteness barrier failed")
            require(before_frozen == frozen_hashes(pipe), "VAE/Text Encoder changed")
            final_latent = state_sha(msdf.encoder)
            require(final_latent == initial_latent, "frozen latent encoder weights changed")
            destination = OUTPUT / "checkpoints/black/final_step2000"
            require(not destination.exists(), "final checkpoint already exists")
            pipe.unet.save_pretrained(str(destination / "unet"), safe_serialization=True)
            for name, row in load(Path(bound["p0_view"]) / "sd2_base_weight_binding.json")["files"].items():
                if name == "unet/diffusion_pytorch_model.safetensors":
                    continue
                target_path = destination / name
                if target_path.exists():
                    require(name == "unet/config.json", f"unexpected checkpoint component: {name}")
                    target_path.write_bytes((Path(bound["base_model"]) / name).read_bytes())
                else:
                    verified_copy(Path(bound["base_model"]) / name, target_path, row["sha256"])
                require(sha(target_path) == row["sha256"], f"checkpoint component changed: {name}")
            torch.save(checkpoint_payload(msdf), destination / "msdf.pt")
            checkpoint = tree_binding(destination, OUTPUT)
            msdf.detach()
            reloaded = load_msdf_adapter(destination / "msdf.pt", pipe.unet, device)
            # The loader defaults to both branches ON.  G3 must always set its
            # inference state explicitly after loading.
            reloaded.set_ablation(ABLATION)
            reloaded.set_injection_gamma(1.0)
            require(reloaded.ablation == ABLATION and reloaded.injection_gamma == 1.0 and
                    reloaded.branch_dropout == 0.0, "PixelOnly reload state failed")
            require(state_sha(reloaded.encoder) == initial_latent,
                    "reloaded frozen latent encoder identity changed")
            reloaded.detach()
            binding = {
                "status": "FROZEN", "class": "black", "successful_updates": 2000,
                "fresh_public_SD2_start": True, "continued_from_G1": False,
                "training_pixel_branch": True, "training_latent_branch": False,
                "training_branch_dropout": 0.0, "latent_encoder_frozen": True,
                "frozen_latent_encoder_initial_SHA256": initial_latent,
                "frozen_latent_encoder_final_SHA256": final_latent,
                "morphology_alignment_enabled": True, "inference_gamma": 1.0,
                "auxiliary_loss_weights": dict(WEIGHTS), "checkpoint": checkpoint,
                "msdf_checkpoint_SHA256": sha(destination / "msdf.pt"),
                "unet_checkpoint_SHA256": sha(destination / "unet/diffusion_pytorch_model.safetensors"),
                "base_model_binding_SHA256": sha(Path(bound["p0_view"]) / "sd2_base_weight_binding.json"),
                "reload_ablation_explicitly_applied": True,
            }
            save(OUTPUT / "pixelonly_checkpoint_binding.json", binding)
            audit_row.update(status="COMPLETE", checkpoint_saved=True, checkpoint=checkpoint,
                frozen_latent_encoder_final_SHA256=final_latent,
                frozen_latent_encoder_strictly_unchanged=True,
                VAE_text_frozen_unchanged=True, final_grad_scaler_scale=scaler.get_scale(),
                peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),
                update_log_SHA256=sha(log_path), completed_at=x10.now())
            del reloaded, msdf, pipe, objects, optimizer, scaler
            gc.collect(); torch.cuda.empty_cache()
        audit_row["wall_seconds"] = time.monotonic() - started
        save(audit_path, audit_row)
        gradient_audit = load(OUTPUT / "pixelonly_gradient_audit.json")
        gradient_audit.update(status="TRAINING_COMPLETE", final_training_gradients={
            "active_MSDF": audit_row["active_MSDF_gradient_status"],
            "UNet": audit_row["UNet_gradient_status"],
            "latent_encoder": audit_row["latent_encoder_gradient_status"]},
            frozen_latent_encoder_strictly_unchanged=True, completed_at=x10.now())
        save(OUTPUT / "pixelonly_gradient_audit.json", gradient_audit)
        status(status="TRAINING_COMPLETE", successful_updates=2000,
               scheduled_attempts=audit_row["scheduled_attempts"], AMP_skips=audit_row["AMP_skips"],
               training_completed_at=x10.now())
        print("[训练完成] PixelOnly Black final_step2000 已保存；Latent Encoder初始化与最终SHA严格一致。", flush=True)
    except Exception as error:
        audit_row.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                         wall_seconds=time.monotonic() - started)
        save(audit_path, audit_row)
        status(status="TRAINING_FAILED", failure=audit_row["error"])
        raise


def write_csv(path, rows):
    require(rows, "cannot write an empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def verified_checkpoint():
    binding = load(OUTPUT / "pixelonly_checkpoint_binding.json")
    require(binding["status"] == "FROZEN" and binding["successful_updates"] == 2000 and
            binding["training_pixel_branch"] is True and binding["training_latent_branch"] is False and
            binding["training_branch_dropout"] == 0.0 and binding["latent_encoder_frozen"] is True,
            "PixelOnly checkpoint binding changed")
    root = OUTPUT / binding["checkpoint"]["path"]
    expected = binding["checkpoint"]["sha256"]
    require(all((root / name).is_file() for name in expected), "checkpoint file missing")
    require(hashlib.sha256(canonical(expected).encode()).hexdigest() ==
            binding["checkpoint"]["tree_sha256"], "checkpoint tree digest changed")
    require(expected["msdf.pt"] == binding["msdf_checkpoint_SHA256"] and
            expected["unet/diffusion_pytorch_model.safetensors"] == binding["unet_checkpoint_SHA256"],
            "checkpoint component binding changed")
    return root, binding


def generate(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "TRAINING_COMPLETE",
            "generation requires complete training")
    claim = OUTPUT / "logs/generate_claim.json"
    require(not claim.exists(), "the one permitted generation run has already been claimed")
    save(claim, {"started_at": x10.now(), "planned_count": 120,
                 "banks": {"A": 42, "B": 3407, "C": 2026}})
    for bank in "ABC":
        (OUTPUT / "images" / ("bank_" + bank)).mkdir(parents=True, exist_ok=True)
    status(status="GENERATING", generation_started_at=x10.now())
    manifest_path = OUTPUT / "pixelonly_generation_manifest.json"
    records, started = [], time.monotonic()
    save(manifest_path, {"status": "RUNNING", "planned_count": 120, "records": [],
                         "reroll_count": 0, "selection_count": 0})
    try:
        checkpoint, checkpoint_binding = verified_checkpoint()
        with gpu_runtime(device) as (torch, bound, _cfg):
            from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            from stage20b_formal_common import synthetic_label
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(checkpoint), local_files_only=True, use_safetensors=True,
                torch_dtype=torch.float16, safety_checker=None, feature_extractor=None,
                requires_safety_checker=False).to(device)
            pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
            pipe.disable_xformers_memory_efficient_attention()
            generation = load(Path(bound["p0_view"]) / "stage20b_generation_protocol.json")
            require(generation["scheduler"] == "DDIMScheduler" and
                    generation["num_inference_steps"] == 50 and
                    generation["guidance_scale"] == 7.5 and generation["eta"] == 0 and
                    generation["strength"] == 1 and generation["height"] == 512 and
                    generation["width"] == 512, "generation protocol changed")
            pipe.scheduler = DDIMScheduler.from_config(generation["scheduler_config"])
            adapter = load_msdf_adapter(checkpoint / "msdf.pt", pipe.unet, device).eval()
            adapter.set_ablation(ABLATION); adapter.set_injection_gamma(1.0)
            require(adapter.ablation == ABLATION and adapter.injection_gamma == 1.0 and
                    adapter.branch_dropout == 0.0 and not adapter.training,
                    "G3 explicit inference state mismatch")
            adapter_before = state_sha(adapter)
            slots = [row for row in load(Path(bound["p0_view"]) /
                     "stage20b_generation_slots.json")["slots"] if row["class"] == "black"]
            originals = {row["slot_id"]: row for row in load(Path(bound["p0"]) /
                         "stage20b_generation_slots.json")["slots"]}
            rngs = {row["slot_id"]: row["banks"] for row in load(Path(bound["p0_view"]) /
                    "stage20b_rng_manifest.json")["records"]}
            require(len(slots) == 40, "Black slot population changed")
            for bank, bank_seed in {"A": 42, "B": 3407, "C": 2026}.items():
                for index, slot in enumerate(slots):
                    item_started = time.monotonic()
                    frozen, rng = originals[slot["slot_id"]], rngs[slot["slot_id"]][bank]
                    require(rng["bank_seed"] == bank_seed and
                            rng["non_rng_metadata_sha256"] ==
                            hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                            f"frozen RNG/slot mismatch: {bank}/{slot['slot_id']}")
                    image, trace = x14.generate_one(pipe, adapter, slot, rng, generation, device)
                    image_path = OUTPUT / "images" / ("bank_" + bank) / (slot["slot_id"] + ".png")
                    label_path = image_path.with_suffix(".txt")
                    require(not image_path.exists() and not label_path.exists(),
                            "generation would overwrite an output")
                    image.save(image_path, format="PNG")
                    label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                    width, height = frozen["geometry"]["final_canvas_wh"]
                    records.append({
                        "group": "pixel_training", "method": "PixelOnly Training", "bank": bank,
                        "class": "black", "slot_id": slot["slot_id"], "frozen_slot": frozen,
                        "runtime_slot": slot, "normal_SHA256": frozen["source_sha256"],
                        "bbox": frozen["bbox"], "prompt": frozen["prompt"],
                        "sample_seed": rng["sample_seed"], "bank_seed": rng["bank_seed"],
                        "checkpoint_tree_SHA256": checkpoint_binding["checkpoint"]["tree_sha256"],
                        "unet_checkpoint_SHA256": checkpoint_binding["unet_checkpoint_SHA256"],
                        "msdf_checkpoint_SHA256": checkpoint_binding["msdf_checkpoint_SHA256"],
                        "branch_config": dict(ABLATION), "injection_gamma": 1.0,
                        "training_branch_dropout": 0.0, "latent_encoder_frozen": True,
                        "image_path": image_path.relative_to(OUTPUT).as_posix(),
                        "image_SHA256": sha(image_path),
                        "label_path": label_path.relative_to(OUTPUT).as_posix(),
                        "label_SHA256": sha(label_path),
                        "roi_xyxy": x10.crop_box(frozen["bbox"], width, height),
                        "quality_selection": False, "reroll": False,
                        "runtime_seconds": time.monotonic() - item_started, **trace})
                    save(manifest_path, {"status": "RUNNING", "planned_count": 120,
                        "records": records, "reroll_count": 0, "selection_count": 0})
                    print(f"[生成] PixelOnly Training Bank {bank} {index + 1}/40 "
                          f"total={len(records)}/120", flush=True)
            require(adapter_before == state_sha(adapter), "adapter parameters changed during generation")
            adapter.detach(); del adapter, pipe
            gc.collect(); torch.cuda.empty_cache()
        counts = Counter(row["bank"] for row in records)
        require(len(records) == 120 and counts == {"A": 40, "B": 40, "C": 40} and
                len({(row["bank"], row["slot_id"]) for row in records}) == 120,
                "generation population incomplete")
        # Match every G3 item to Raw, Full and PixelOnly-Inference metadata.
        historical = load(x14.OUTPUT / "branch_generation_manifest.json")
        require(historical["status"] == "COMPLETE" and historical["count"] == 240,
                "Experiment 14 branch manifest changed")
        pixel_inference = {(row["bank"], row["slot_id"]): row for row in historical["records"]
                           if row["group"] == "pixel_only"}
        comparisons = 0
        for row in records:
            key = (row["bank"], row["slot_id"]); prior = pixel_inference[key]
            for field in ("frozen_slot", "runtime_slot", "sample_seed", "bank_seed"):
                require(row[field] == prior[field], f"matched protocol mismatch: {key}/{field}")
            require(sha(OUTPUT / row["image_path"]) == row["image_SHA256"] and
                    sha(OUTPUT / row["label_path"]) == row["label_SHA256"],
                    f"generated asset changed: {key}")
            comparisons += 1
        manifest = {"status": "COMPLETE", "count": 120, "records": records,
                    "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
                    "runtime_seconds": time.monotonic() - started,
                    "generation_protocol_SHA256": sha(Path(bound["p0_view"]) /
                                                        "stage20b_generation_protocol.json")}
        save(manifest_path, manifest)
        save(OUTPUT / "pixelonly_generation_integrity.json", {
            "status": "PASS", "count": 120, "bank_counts": dict(counts),
            "matched_to_PixelOnly_Inference_records": comparisons,
            "slot_normal_bbox_prompt_sample_seed_bank_seed_match": True,
            "image_and_label_SHA256_checks": 240,
            "checkpoint_tree_SHA256": checkpoint_binding["checkpoint"]["tree_sha256"],
            "unet_SHA256": checkpoint_binding["unet_checkpoint_SHA256"],
            "msdf_SHA256": checkpoint_binding["msdf_checkpoint_SHA256"],
            "actual_branch_config": ABLATION, "gamma": 1.0,
            "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
            "completed_at": x10.now()})
        status(status="GENERATION_COMPLETE", generation_count=120,
               generation_completed_at=x10.now(), reroll_count=0, selection_count=0)
        print("[生成完成] PixelOnly Training A/B/C 共120/120；无重抽、筛选或替换。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "records": records,
                             "error": f"{type(error).__name__}: {error}",
                             "reroll_count": 0, "selection_count": 0})
        status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def extract_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    cache_path = OUTPUT / "pixelonly_inception_features.npz"
    ledger_path = OUTPUT / "pixelonly_inception_feature_cache.json"
    keys = [f"pixel_training:{row['bank']}:{row['slot_id']}" for row in manifest["records"]]
    if cache_path.is_file() and ledger_path.is_file():
        ledger = load(ledger_path)
        require(ledger["status"] == "COMPLETE" and ledger["keys"] == keys and
                ledger["sha256"] == sha(cache_path) and ledger["image_SHA256_checks"] == 120,
                "existing G3 feature cache binding changed")
        return x12.array_features(cache_path, ledger_path, torch)
    model = lib["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
    values = []
    with torch.inference_mode():
        for index, row in enumerate(manifest["records"]):
            path = OUTPUT / row["image_path"]
            require(path.is_file() and sha(path) == row["image_SHA256"],
                    f"generated image changed: {path}")
            with Image.open(path) as source:
                crop = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
            tensor = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = model(tensor)[0]
            require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()),
                    row["slot_id"])
            values.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0:
                print(f"[Inception] {index + 1}/120", flush=True)
    data = np.stack(values)
    np.savez_compressed(cache_path, features=data)
    save(ledger_path, {"status": "COMPLETE", "keys": keys, "shape": [120, 2048],
        "stored_dtype": str(data.dtype), "extraction_dtype": "float32",
        "sha256": sha(cache_path), "image_SHA256_checks": 120})
    del model; torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[index]).double() for index, key in enumerate(keys)}


def create_report(conclusion, black, overall, pooled, summary):
    lines = ["# Experiment 15 — Training-Level PixelOnly Ablation", "",
        f"Status: `{conclusion}`.", "",
        "One PixelOnly Black model was trained from the same public SD2 base for exactly 2000 successful updates. "
        "Its latent encoder remained in the checkpoint structure but was frozen, excluded from AdamW, and its feature "
        "contribution was zeroed. Pixel support, fusion, morphology alignment, dynamic gating and all four up-block "
        "injections remained enabled. Branch dropout was fixed to 0.0.", "",
        "Compared with the historical Full model, this changes both the branch configuration and the necessary "
        "single-branch dropout setting. It also changes random-number consumption. The comparison is therefore not a "
        "strict one-variable causal experiment.", "", "## Bankwise Black", "",
        "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in black:
        lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Bankwise Overall", "",
        "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in overall:
        lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Pooled metrics", "",
        "Pooled metrics were recomputed from concatenated features; bank and class scores were not averaged.", "",
        "| Scope | Population | Method | Images | KID ↓ | FID ↓ |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in pooled:
        lines.append(f"| {row['scope']} | {row['population']} | {row['method']} | "
                     f"{row['generated_images']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    bc = summary["BC_pooled_Black"]
    abc = summary["ABC_pooled_Black"]
    lines += ["", "## Required findings", "",
        f"1. G3 B+C pooled Black KID/FID are {bc['pixel_training']['KID']:.10f} / "
        f"{bc['pixel_training']['FID']:.6f}; A+B+C pooled values are "
        f"{abc['pixel_training']['KID']:.10f} / {abc['pixel_training']['FID']:.6f}.",
        f"2. Versus PixelOnly Inference on B+C, retraining changed KID by "
        f"{summary['G3_vs_G2_BC']['KID']['absolute_delta']:+.10f} and FID by "
        f"{summary['G3_vs_G2_BC']['FID']['absolute_delta']:+.6f}.",
        f"3. Versus Full on B+C, PixelOnly Training changed KID by "
        f"{summary['G3_vs_G1_BC']['KID']['absolute_delta']:+.10f} and FID by "
        f"{summary['G3_vs_G1_BC']['FID']['absolute_delta']:+.6f}.",
        f"4. B/C bankwise direction is recorded as `{summary['B_C_consistency']}`.",
        f"5. On A+B+C pooled Black, the best KID method is `{summary['best_ABC_Black_KID']}` and "
        f"the best FID method is `{summary['best_ABC_Black_FID']}`.",
        f"6. Versus Raw on B+C, G3 differs by KID "
        f"{summary['G3_vs_G0_BC']['KID']['absolute_delta']:+.10f} "
        f"({summary['G3_vs_G0_BC']['KID']['relative_percent']:+.2f}%) and FID "
        f"{summary['G3_vs_G0_BC']['FID']['absolute_delta']:+.6f} "
        f"({summary['G3_vs_G0_BC']['FID']['relative_percent']:+.2f}%).",
        "7. Pixel/Fusion and UNet gradients were finite and effective during training; latent encoder gradients remained None and its final SHA equals initialization.",
        "8. Pixel Support was nonzero and finite in preflight and remained enabled in training and inference.",
        f"9. Support for retaining the existing dual-branch training configuration follows the declared status: `{conclusion}`.",
        f"10. The next-step recommendation is: {summary['next_step_recommendation']}", "",
        "## Research boundary", "",
        "A/B/C share slots, V2 TRAIN references may overlap generator training, B/C have been observed in prior work, "
        "and only one training seed was run. These are development results, not independent validation or evidence of "
        "cross-seed statistical significance.", "",
        "No LatentOnly training, second seed, Full retraining, fusion change, loss search, gamma search, BootstrapGuard "
        "or detector training was started.", ""]
    (OUTPUT / "PIXELONLY_TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(device):
    require(load(OUTPUT / "experiment_status.json")["status"] in
            ("GENERATION_COMPLETE", "EVALUATING", "INCOMPLETE"),
            "evaluation requires complete generation")
    status(status="EVALUATING", evaluation_started_at=x10.now())
    started = time.monotonic()
    try:
        with x10.gpu_context(device, evaluation=True) as (torch, _bound, _cfg):
            from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
            lib, runtime_audit = metric_runtime(device)
            manifest = load(OUTPUT / "pixelonly_generation_manifest.json")
            require(manifest["status"] == "COMPLETE" and manifest["count"] == 120,
                    "generation manifest incomplete")
            new = extract_features(device, torch, lib, manifest)
            gamma = x12.array_features(x14.GAMMA / "gamma_sweep_inception_features.npz",
                                       x14.GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
            multibank = x12.array_features(x14.X12 / "multibank_inception_features.npz",
                                           x14.X12 / "multibank_inception_feature_cache.json", torch)
            references = load(x14.GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
            real_black = torch.stack([gamma["real:" + row["annotation_id"]]
                                      for row in references if row["class"] == "black"])
            real_all = torch.stack([gamma["real:" + row["annotation_id"]] for row in references])
            slots = load(x14.P0 / "stage20b_generation_slots.json")["slots"]
            black_slots = [row for row in slots if row["class"] == "black"]
            flash_slots = [row for row in slots if row["class"] == "flash"]

            def black_features(bank):
                return torch.stack([new[f"pixel_training:{bank}:{row['slot_id']}"]
                                    for row in black_slots])

            def shared_flash(bank):
                if bank == "A":
                    return torch.stack([gamma[f"fake:0.00:{row['slot_id']}"] for row in flash_slots])
                return torch.stack([multibank[f"flash_gamma0:{bank}:{row['slot_id']}"]
                                    for row in flash_slots])

            # Experiment 13 contains Raw/Gamma0/Full.  PixelOnly Inference was
            # newly evaluated in Experiment 14, so read Experiment 14's frozen
            # complete tables rather than its Experiment-13-only helper.
            def frozen_rows(path):
                numeric = {"KID", "KID_subset_std", "FID", "runtime_seconds"}
                integer = {"reference_ROIs", "generated_images", "KID_subset_size"}
                rows = []
                with Path(path).open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        rows.append({key: (float(value) if key in numeric else
                                           int(float(value)) if key in integer else value)
                                     for key, value in row.items()})
                return rows
            old_black = frozen_rows(x14.OUTPUT / "branch_bankwise_black.csv")
            old_overall = frozen_rows(x14.OUTPUT / "branch_bankwise_overall.csv")
            old_pooled = frozen_rows(x14.OUTPUT / "branch_pooled_metrics.csv")
            keep = {"raw", "gamma0", "full", "pixel_only"}
            old_black = [dict(row, method=("PixelOnly Inference" if row["group"] == "pixel_only"
                                           else row["method"])) for row in old_black if row["group"] in keep]
            old_overall = [dict(row, method=("PixelOnly Inference" if row["group"] == "pixel_only"
                                             else row["method"])) for row in old_overall if row["group"] in keep]
            old_pooled = [dict(row, method=("PixelOnly Inference" if row["group"] == "pixel_only"
                                            else row["method"])) for row in old_pooled if row["group"] in keep]
            black_rows, overall_rows = [], []
            for bank in "ABC":
                black_rows.extend(row for row in old_black if row["bank"] == bank)
                overall_rows.extend(row for row in old_overall if row["bank"] == bank)
                fake_black = black_features(bank)
                black_metric = x12.metric(real_black, fake_black, lib)
                overall_metric = x12.metric(real_all, torch.cat([shared_flash(bank), fake_black]), lib)
                black_rows.append({"bank": bank, "group": "pixel_training",
                    "method": "PixelOnly Training", **black_metric,
                    "metric_source": "new same-protocol G3 features"})
                overall_rows.append({"bank": bank, "group": "pixel_training",
                    "method": "PixelOnly Training", **overall_metric,
                    "metric_source": "new G3 Black plus shared Flash Gamma0 features"})
                print(f"[指标] {bank} G3 Black KID={black_metric['KID']:.10f} "
                      f"FID={black_metric['FID']:.6f}; Overall KID={overall_metric['KID']:.10f} "
                      f"FID={overall_metric['FID']:.6f}", flush=True)
            pooled_rows = []
            for scope, banks in (("A+B+C", "ABC"), ("B+C", "BC")):
                pooled_rows.extend(row for row in old_pooled if row["scope"] == scope)
                for population, real in (("Black", real_black), ("Overall", real_all)):
                    if population == "Black":
                        fake = torch.cat([black_features(bank) for bank in banks])
                    else:
                        fake = torch.cat([torch.cat([shared_flash(bank), black_features(bank)])
                                          for bank in banks])
                    result = x12.metric(real, fake, lib)
                    pooled_rows.append({"scope": scope, "group": "pixel_training",
                        "method": "PixelOnly Training", **result,
                        "metric_source": "new concatenated G3 features", "population": population})
                    print(f"[Pooled] {scope} {population} G3 KID={result['KID']:.10f} "
                          f"FID={result['FID']:.6f}", flush=True)

        order = {name: index for index, name in enumerate(
                 ("raw", "gamma0", "full", "pixel_only", "pixel_training"))}
        black_rows.sort(key=lambda row: (row["bank"], order[row["group"]]))
        overall_rows.sort(key=lambda row: (row["bank"], order[row["group"]]))
        pooled_rows.sort(key=lambda row: (("A+B+C", "B+C").index(row["scope"]),
                         ("Black", "Overall").index(row["population"]), order[row["group"]]))
        write_csv(OUTPUT / "pixelonly_bankwise_black.csv", black_rows)
        write_csv(OUTPUT / "pixelonly_bankwise_overall.csv", overall_rows)
        write_csv(OUTPUT / "pixelonly_pooled_metrics.csv", pooled_rows)

        comparisons = (("pixel_training", "full"), ("pixel_training", "pixel_only"),
                       ("pixel_training", "raw"), ("pixel_training", "gamma0"))
        deltas = []
        def add(scope, population, first, second):
            for metric_name in ("KID", "FID"):
                delta = first[metric_name] - second[metric_name]
                deltas.append({"scope": scope, "population": population,
                    "comparison": first["group"] + " - " + second["group"],
                    "metric": metric_name, "first": first[metric_name],
                    "second": second[metric_name], "absolute_delta": delta,
                    "relative_percent": 100 * delta / second[metric_name],
                    "first_is_better": delta < 0})
        for bank in "ABC":
            for population, rows in (("Black", black_rows), ("Overall", overall_rows)):
                by = {row["group"]: row for row in rows if row["bank"] == bank}
                for first, second in comparisons: add(bank, population, by[first], by[second])
        for scope in ("A+B+C", "B+C"):
            for population in ("Black", "Overall"):
                by = {row["group"]: row for row in pooled_rows
                      if row["scope"] == scope and row["population"] == population}
                for first, second in comparisons: add(scope, population, by[first], by[second])
        write_csv(OUTPUT / "pixelonly_pairwise_deltas.csv", deltas)

        def bank(bank):
            return {row["group"]: row for row in black_rows if row["bank"] == bank}
        def pool(scope, population="Black"):
            return {row["group"]: row for row in pooled_rows
                    if row["scope"] == scope and row["population"] == population}
        def comparison(first, second):
            return {name: {"absolute_delta": first[name] - second[name],
                           "relative_percent": 100 * (first[name] - second[name]) / second[name],
                           "first_is_better": first[name] < second[name]}
                    for name in ("KID", "FID")}
        full_wins = [bank(value)["full"][metric] < bank(value)["pixel_training"][metric]
                     for value in "BC" for metric in ("KID", "FID")]
        pixel_wins = [bank(value)["pixel_training"][metric] < bank(value)["full"][metric]
                      for value in "BC" for metric in ("KID", "FID")]
        bc = pool("B+C")
        full_pooled = [bc["full"][metric] < bc["pixel_training"][metric] for metric in ("KID", "FID")]
        pixel_pooled = [bc["pixel_training"][metric] < bc["full"][metric] for metric in ("KID", "FID")]
        if all(full_wins) and all(full_pooled):
            conclusion = "TRAINED_DUAL_BRANCH_CONSISTENT_BENEFIT"
        elif all(pixel_wins) and all(pixel_pooled):
            conclusion = "TRAINED_PIXEL_ONLY_CONSISTENT_BENEFIT"
        else:
            conclusion = "TRAINING_BRANCH_MIXED"
        abc = pool("A+B+C")
        bc_directions = [{"bank": value, "metric": metric,
                          "winner": "pixel_training" if bank(value)["pixel_training"][metric] <
                          bank(value)["full"][metric] else "full"}
                         for value in "BC" for metric in ("KID", "FID")]
        summary = {
            "status": conclusion, "primary_scope": "B+C pooled Black",
            "BC_pooled_Black": bc, "ABC_pooled_Black": abc,
            "G3_vs_G1_BC": comparison(bc["pixel_training"], bc["full"]),
            "G3_vs_G2_BC": comparison(bc["pixel_training"], bc["pixel_only"]),
            "G3_vs_G0_BC": comparison(bc["pixel_training"], bc["raw"]),
            "B_C_bankwise_checks": bc_directions,
            "B_C_consistency": ("Full wins every KID/FID check" if all(full_wins)
                                else "PixelOnly Training wins every KID/FID check" if all(pixel_wins)
                                else "mixed directions"),
            "best_ABC_Black_KID": min(abc.values(), key=lambda row: row["KID"])["group"],
            "best_ABC_Black_FID": min(abc.values(), key=lambda row: row["FID"])["group"],
            "gradient_and_parameter_update_normal": True,
            "pixel_support_working": True,
            "latent_encoder_initial_final_SHA_equal": True,
            "next_step_recommendation": (
                "retain the current Full configuration; a LatentOnly training run is unnecessary unless a future "
                "mechanistic study is explicitly authorized" if conclusion == "TRAINED_DUAL_BRANCH_CONSISTENT_BENEFIT"
                else "do not infer latent uselessness from one seed; inspect fusion behavior before authorizing any new training"),
            "research_boundary": {"single_training_seed": True, "independent_validation": False,
                                  "V2_TRAIN_reference_overlap_possible": True,
                                  "B_C_previously_observed": True,
                                  "strict_single_variable_causal_comparison": False},
        }
        save(OUTPUT / "pixelonly_stability_summary.json", summary)
        protocol = load(OUTPUT / "pixelonly_training_protocol.json")
        protocol.update(status="COMPLETE", conclusion=conclusion,
                        source_code_SHA256=source_hashes(), metric_runtime=runtime_audit,
                        completed_at=x10.now())
        save(OUTPUT / "pixelonly_training_protocol.json", protocol)
        create_report(conclusion, black_rows, overall_rows, pooled_rows, summary)
        final_status_path = OUTPUT / "experiment_status.json"
        final_status = load(final_status_path)
        final_status.pop("failure", None)
        final_status.update(status=conclusion, evaluation_completed=True, completed_at=x10.now(),
                            new_feature_count=120,
                            evaluation_runtime_seconds=time.monotonic() - started,
                            followup_started=False)
        save(final_status_path, final_status)
        print(f"[完成] Experiment 15 status={conclusion}；停止，不启动后续实验。", flush=True)
    except Exception as error:
        status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "preflight", "train", "generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 15 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "audit":
        audit()
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
