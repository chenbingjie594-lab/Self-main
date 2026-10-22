"""Experiment 9: train and evaluate one Black MSDF checkpoint without auxiliary losses.

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

OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_black_no_aux"
EXPERIMENT_CONFIG = REPO / "configs/plastic_bomo_msdf_black_no_aux.json"
SOURCE_RUN = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214"
SOURCE_RUNTIME = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/server_runtime_20261013_080546_552698"
RAW = REPO / "results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
EXP7 = REPO / "results_for_gpt/plastic_bomo_msdf_raw_matched_comparison"
NOMORPH = REPO / "results_for_gpt/plastic_bomo_msdf_black_no_morph_alignment"
BASE_COMMIT = "ca328eba82af1e5efc2debb314cb81413d2c92b9"
AUXILIARY_NAMES = ("region", "reconstruction", "structure", "background", "scale", "support")
AUXILIARY_WEIGHT_KEYS = {name: name + "_weight" for name in AUXILIARY_NAMES}


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


def binding():
    bound = load(SOURCE_RUN / "stage25_run_binding.json")
    cfg = load(REPO / "configs/plastic_bomo_stage25_server.json")
    experiment = load(EXPERIMENT_CONFIG)
    require(experiment["base_commit"] == BASE_COMMIT and experiment["class"] == "black",
            "Experiment 9 config identity changed")
    require(experiment["morphology_alignment_enabled"] is True and experiment["injection_gamma"] == 1.0,
            "Experiment 9 inference controls changed")
    require(set(experiment["auxiliary_loss_weights"]) == set(AUXILIARY_NAMES) and
            all(experiment["auxiliary_loss_weights"][name] == 0.0 for name in AUXILIARY_NAMES),
            "Experiment 9 auxiliary controls changed")
    for name, key in AUXILIARY_WEIGHT_KEYS.items():
        cfg["MSDF_preserved_defaults"][key] = experiment["auxiliary_loss_weights"][name]
    return bound, cfg


def source_hashes():
    names = (
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py",
        "train_rda.py",
        "tools/stage25_msdf_generator.py",
        "tools/stage25_msdf_contract.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "tools/qualify_plastic_bomo_stage20b_r0.py",
        "tools/run_plastic_bomo_msdf_black_no_aux.py",
        "configs/plastic_bomo_msdf_black_no_aux.json",
    )
    return {name: sha(REPO / name) for name in names}


def protocol_record(cfg):
    train = cfg["common_generator_training_target"]
    defaults = cfg["MSDF_preserved_defaults"]
    return {
        "status": "FROZEN_BEFORE_TRAINING",
        "experiment": "Experiment 9 — MSDF Black No Auxiliary Losses",
        "base_commit": BASE_COMMIT,
        "class": "black",
        "groups": {
            "G0": "Frozen Raw SD2 Black",
            "G1": "Frozen original MSDF Black; morphology alignment ON; gamma=1",
            "G2": "Frozen MSDF-NoMorph Black; morphology alignment OFF; six auxiliary losses ON; gamma=1",
            "G3": "New MSDF-NoAux Black; morphology alignment ON; six auxiliary loss weights zero; gamma=1",
        },
        "single_planned_training": "G3 only",
        "training": {**train, "objective": "conditional_loss only"},
        "msdf": {**defaults, "morphology_alignment_enabled": True, "injection_gamma": 1.0},
        "auxiliary_loss_weights": {name: defaults[AUXILIARY_WEIGHT_KEYS[name]] for name in AUXILIARY_NAMES},
        "only_planned_factor": "Six MSDF auxiliary training loss weights: historical values versus all zero",
        "semantic_effect": "The six raw diagnostic losses are still computed, but their weighted contribution is exactly zero; diffusion_objective equals conditional_loss.",
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
    require(all(defaults.get(k) == historical.get(k) for k in structural_keys), "NoAux changed an MSDF structural control")
    require(all(defaults.get(key) == 0.0 for key in AUXILIARY_WEIGHT_KEYS.values()),
            "NoAux auxiliary weights are not all zero")
    require(defaults["morphology_alignment_enabled"] is True, "NoAux morphology alignment is not enabled")
    require(train["prompts"]["black"] == "a photo of a sks black spot defect", "Black prompt changed")
    return {"training": expected_train, "historical_msdf": expected_msdf,
            "noaux_weights": {name: defaults[AUXILIARY_WEIGHT_KEYS[name]] for name in AUXILIARY_NAMES}}


def initialize_output(cfg):
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    (OUTPUT / "logs").mkdir(parents=True)
    (OUTPUT / "images/bank_A").mkdir(parents=True)
    save(OUTPUT / "noaux_protocol.json", protocol_record(cfg))
    save(OUTPUT / "noaux_status.json", {
        "status": "PREFLIGHT_PENDING", "created_at": now(), "base_commit": BASE_COMMIT,
        "training_count": 0, "generation_count": 0, "reroll_count": 0,
        "selection_count": 0, "final_eval_read_or_used": False,
        "BootstrapGuard": False, "detector_training_started": False,
        "other_structure_search_started": False,
    })


def update_status(**values):
    path = OUTPUT / "noaux_status.json"
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
def gpu_context(device, evaluation=False):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    bound, cfg = binding()
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
        "auxiliary_loss_weights": {name: 0.0 for name in AUXILIARY_NAMES},
        "state_dict": {name: value.detach().cpu() for name, value in msdf.state_dict().items()},
    }


def preflight(device):
    bound, cfg = binding()
    initialize_output(cfg)
    started = time.monotonic()
    audit = {"status": "RUNNING", "started_at": now(), "optimizer_steps": 0,
             "sampling_count": 0, "final_eval_content_read": 0}
    save(OUTPUT / "noaux_preflight_audit.json", audit)
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
            state = msdf.set_ablation({"morphology_alignment": True})
            msdf.set_injection_gamma(1.0)
            require(state["morphology_alignment"] is True and state["pixel_branch"] and state["latent_branch"],
                    "NoAux state invalid")
            require(state_sha(msdf) == initial_adapter_sha, "set_ablation changed adapter parameters")
            before = frozen_hashes(pipe)
            rows = sorted((r for r in load(Path(bound["p0_view"]) / "generator_train_allowlist.json")["inputs"]
                           if r["class"] == "black"), key=lambda r: r["annotation_id"])
            loss, losses, shapes = microbatch(objects, rows[0], rows[0], bound, cfg, torch)
            objective_delta = abs(float(loss.detach()) - losses["denoising"])
            weights = {name: cfg["MSDF_preserved_defaults"][AUXILIARY_WEIGHT_KEYS[name]]
                       for name in AUXILIARY_NAMES}
            require(all(value == 0.0 for value in weights.values()), "auxiliary weight is nonzero")
            require(objective_delta <= 1e-7, "diffusion_objective differs from conditional_loss")
            target = cfg["common_generator_training_target"]
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
            save(OUTPUT / "noaux_preflight_audit.json", audit)
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
            temporary = OUTPUT / "logs/preflight_noaux_msdf.pt"
            torch.save(checkpoint_payload(msdf), temporary)
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
                "objective_minus_conditional_abs": objective_delta,
                "backward_matches_formal_training": True,
                "gradient_scaler_scale": probe_scaler.get_scale(),
                "optimizer_step_called": False,
                "raw_auxiliary_losses_diagnostic_only": {name: losses[name] for name in AUXILIARY_NAMES},
                "auxiliary_loss_weights": weights, "weighted_auxiliary_contribution": 0.0,
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
                "six_auxiliary_losses_weighted_into_objective": False,
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
                    "Historical G1 did not persist a step-0 adapter SHA or complete RNG-state ledger. "
                    "G3 reuses seed, sorted input order, epoch permutation, reference rule, runtime and initialization code, "
                    "but exact byte-for-byte G1/G3 random trajectory cannot be independently proven."
                ),
            }
            del loss, reloaded, msdf, pipe, objects
            gc.collect(); torch.cuda.empty_cache()
        audit.update(status="PASS", completed_at=now(), wall_seconds=time.monotonic() - started,
                     formal_training_authorized=True)
        save(OUTPUT / "noaux_preflight_audit.json", audit)
        update_status(status="PREFLIGHT_PASS", preflight_completed_at=now())
        print("[检查通过] NoAux CPU、单batch、双分支梯度、六项Loss、冻结组件和checkpoint恢复全部通过。", flush=True)
    except Exception as error:
        audit.update(status="FAIL", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
        save(OUTPUT / "noaux_preflight_audit.json", audit)
        update_status(status="PREFLIGHT_FAILED", failure=audit["error"])
        raise


def train(device):
    status = load(OUTPUT / "noaux_status.json")
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
        "auxiliary_loss_weights": {name: 0.0 for name in AUXILIARY_NAMES},
        "training_objective": "conditional_loss only",
    }
    path = OUTPUT / "noaux_training_audit.json"
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
            require(state_sha(msdf) == initial_adapter_sha, "NoAux changed initialization")
            before = frozen_hashes(pipe)
            rows = sorted((r for r in load(Path(bound["p0_view"]) / "generator_train_allowlist.json")["inputs"]
                           if r["class"] == "black"), key=lambda r: r["annotation_id"])
            require(len(rows) == 97, "Black TRAIN population changed")
            target = cfg["common_generator_training_target"]
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
                initial_RNG_state_summary={
                    "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    "torch_CPU": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    "torch_CUDA": hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
                },
                source_code_SHA256=source_hashes(),
                random_trajectory_limit=load(OUTPUT / "noaux_preflight_audit.json")["controlled_variables"]["random_trajectory_limit"],
                MSDF_nonzero_gradient_attempts=0,
            )
            epoch, consecutive_skips = 0, 0
            log_path = OUTPUT / "logs/noaux_black_updates.jsonl"
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
                            objective_delta = abs(float(loss.detach()) - losses["denoising"])
                            require(objective_delta <= 1e-7, "training objective differs from conditional_loss")
                            scaler.scale(loss / target["gradient_accumulation_steps"]).backward()
                            loss_value = float(loss.detach())
                            sums["training_loss"] += loss_value
                            for name, value in losses.items():
                                sums["auxiliary_" + name] += value
                            value_count += 1
                            records.append({**shapes, "loss": loss_value, "conditional_loss": losses["denoising"],
                                            "objective_minus_conditional_abs": objective_delta,
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
                        norm = torch.nn.utils.clip_grad_norm_(params, target["max_grad_norm"])
                        require(not finite or bool(torch.isfinite(norm)), "invalid gradient norm")
                        before_step, before_scale = actual[0], scaler.get_scale()
                        scaler.step(opt); scaler.update()
                        success = actual[0] == before_step + 1
                        require(not success or (msdf_finite and msdf_nonzero > 0),
                                "successful update had no effective finite MSDF gradient")
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
                            print(f"[训练] Black NoAux successful={actual[0]} attempts={audit['scheduled_attempts']} skips={audit['AMP_skips']}", flush=True)
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
            require(payload["auxiliary_loss_weights"] == {name: 0.0 for name in AUXILIARY_NAMES},
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
                "auxiliary_loss_weights": {name: 0.0 for name in AUXILIARY_NAMES},
                "checkpoint": checkpoint, "msdf_checkpoint_SHA256": sha(destination / "msdf.pt"),
                "base_model_binding_SHA256": sha(Path(bound["p0_view"]) / "sd2_base_weight_binding.json"),
            }
            save(OUTPUT / "noaux_checkpoint_binding.json", checkpoint_binding)
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
        print("[训练完成] Black NoAux final_step2000 已保存并通过Morphology=True、AuxWeights=0重载检查。", flush=True)
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
    status = load(OUTPUT / "noaux_status.json")
    require(status["status"] == "TRAINING_COMPLETE", "generation requires completed training")
    require(not (OUTPUT / "logs/generate_claim.json").exists(), "generation already claimed; reroll is forbidden")
    save(OUTPUT / "logs/generate_claim.json", {"started_at": now(), "bank": "A", "planned_count": 40})
    update_status(status="GENERATING", generation_started_at=now())
    started = time.monotonic()
    records = []
    manifest_path = OUTPUT / "noaux_generation_manifest.json"
    save(manifest_path, {"status": "RUNNING", "bank": "A", "records": []})
    try:
        with gpu_context(device) as (torch, bound, cfg):
            import numpy as np
            from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
            from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
            from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
            from generate_plastic_bomo_stage20b_banks import sample
            from stage20b_formal_common import synthetic_label, verify_tree
            checkpoint_binding = load(OUTPUT / "noaux_checkpoint_binding.json")
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
                    "training_auxiliary_loss_weights": {name: 0.0 for name in AUXILIARY_NAMES},
                    "quality_selection": False, "reroll": False,
                    "runtime_seconds": time.monotonic() - item_started, **trace,
                }
                records.append(record)
                save(manifest_path, {"status": "RUNNING", "bank": "A", "records": records})
                print(f"[生成] Black NoAux Bank A {index + 1}/40 {slot['slot_id']}", flush=True)
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
        print("[生成完成] Black NoAux Bank A 40/40；无重抽、替换或筛选。", flush=True)
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
    status = load(OUTPUT / "noaux_status.json")
    require(status["status"] == "GENERATION_COMPLETE", "evaluation requires complete generation")
    update_status(status="EVALUATING", evaluation_started_at=now())
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
        lib, runtime_audit = metric_runtime(device)
        manifest = load(OUTPUT / "noaux_generation_manifest.json")
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
                values.append(value[0].double().cpu().numpy()); keys.append("noaux:" + row["slot_id"])
                if (index + 1) % 10 == 0:
                    print(f"[Inception] NoAux Black {index + 1}/40", flush=True)
        new_array = np.stack(values)
        cache = OUTPUT / "noaux_inception_features.npz"
        np.savez_compressed(cache, features=new_array)
        save(OUTPUT / "noaux_inception_feature_cache.json", {
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
        nomorph_ledger = load(NOMORPH / "nomorph_inception_feature_cache.json")
        nomorph_cache = NOMORPH / "nomorph_inception_features.npz"
        require(nomorph_ledger["sha256"] == sha(nomorph_cache) and nomorph_ledger["shape"] == [40, 2048],
                "NoMorph feature cache changed")
        with np.load(nomorph_cache, allow_pickle=False) as archive:
            nomorph_array = archive["features"].copy()
        nomorph_features = {key: torch.from_numpy(nomorph_array[i]).double()
                            for i, key in enumerate(nomorph_ledger["keys"])}
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
            "MSDF-NoMorph": torch.stack([nomorph_features["nomorph:" + row["slot_id"]] for row in black_slots]),
            "MSDF-NoAux": torch.stack([new_features["noaux:" + row["slot_id"]] for row in black_slots]),
        }
        overall_sets = {
            "Raw SD2": torch.stack([raw_features["raw:roi:A:" + row["slot_id"]] for row in slots]),
            "Original Class-Adaptive MSDF": torch.stack([
                gamma_features[f"fake:{0.0 if row['class'] == 'flash' else 1.0:.2f}:" + row["slot_id"]] for row in slots]),
            "NoMorph-Black Hybrid MSDF": torch.stack(
                [gamma_features["fake:0.00:" + row["slot_id"]] for row in flash_slots] +
                [nomorph_features["nomorph:" + row["slot_id"]] for row in black_slots]),
            "NoAux-Black Hybrid MSDF": torch.stack(
                [gamma_features["fake:0.00:" + row["slot_id"]] for row in flash_slots] +
                [new_features["noaux:" + row["slot_id"]] for row in black_slots]),
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
        write_csv(OUTPUT / "noaux_black_metrics.csv", black_rows)
        write_csv(OUTPUT / "noaux_overall_metrics.csv", overall_rows)

        def black(method):
            return next(row for row in black_rows if row["Method"] == method)
        def overall(method):
            return next(row for row in overall_rows if row["Method"] == method)
        expected = {
            "Raw_Black_KID": 0.0379734447, "Raw_Black_FID": 161.766924,
            "Full_Black_KID": 0.0800462518, "Full_Black_FID": 208.478914,
            "NoMorph_Black_KID": 0.0810946195563176, "NoMorph_Black_FID": 203.5375899268211,
            "Raw_Overall_KID": 0.03522257098656576, "Raw_Overall_FID": 149.42750331396738,
            "Adaptive_Overall_KID": 0.04989160653660615, "Adaptive_Overall_FID": 165.1311363969408,
            "NoMorph_Overall_KID": 0.04866829543849569, "NoMorph_Overall_FID": 162.4736380524388,
        }
        actual = {
            "Raw_Black_KID": black("Raw SD2")["Black_KID"], "Raw_Black_FID": black("Raw SD2")["Black_FID"],
            "Full_Black_KID": black("Original MSDF")["Black_KID"], "Full_Black_FID": black("Original MSDF")["Black_FID"],
            "NoMorph_Black_KID": black("MSDF-NoMorph")["Black_KID"],
            "NoMorph_Black_FID": black("MSDF-NoMorph")["Black_FID"],
            "Raw_Overall_KID": overall("Raw SD2")["Overall_KID"], "Raw_Overall_FID": overall("Raw SD2")["Overall_FID"],
            "Adaptive_Overall_KID": overall("Original Class-Adaptive MSDF")["Overall_KID"],
            "Adaptive_Overall_FID": overall("Original Class-Adaptive MSDF")["Overall_FID"],
            "NoMorph_Overall_KID": overall("NoMorph-Black Hybrid MSDF")["Overall_KID"],
            "NoMorph_Overall_FID": overall("NoMorph-Black Hybrid MSDF")["Overall_FID"],
        }
        tolerances = {name: (5e-11 if name.endswith("KID") else 5e-7) for name in expected}
        errors = {name: abs(actual[name] - expected[name]) for name in expected}
        require(all(errors[name] <= tolerances[name] for name in errors), f"frozen metric regression failed: {errors}")
        regression = {"status": "PASS", "expected": expected, "actual": actual,
                      "absolute_errors": errors, "tolerances": tolerances,
                      "same_reference_feature_population": True,
                      "new_Inception_forward_count": 40}
        previous = black("Original MSDF"); nomorph = black("MSDF-NoMorph")
        new = black("MSDF-NoAux"); raw = black("Raw SD2")
        hybrid = overall("NoAux-Black Hybrid MSDF"); adaptive = overall("Original Class-Adaptive MSDF")
        nomorph_hybrid = overall("NoMorph-Black Hybrid MSDF")
        raw_overall = overall("Raw SD2")

        def difference(current, baseline, kid_key, fid_key):
            kd = current[kid_key] - baseline[kid_key]; fd = current[fid_key] - baseline[fid_key]
            return {"KID": {"absolute": kd, "relative_percent": percent(kd, baseline[kid_key])},
                    "FID": {"absolute": fd, "relative_percent": percent(fd, baseline[fid_key])}}
        comparisons = {
            "G3_NoAux_minus_G1_Full_Black": difference(new, previous, "Black_KID", "Black_FID"),
            "G3_NoAux_minus_G2_NoMorph_Black": difference(new, nomorph, "Black_KID", "Black_FID"),
            "G3_NoAux_minus_G0_Raw_Black": difference(new, raw, "Black_KID", "Black_FID"),
            "NoAux_Hybrid_minus_Original_Adaptive_Overall": difference(hybrid, adaptive, "Overall_KID", "Overall_FID"),
            "NoAux_Hybrid_minus_NoMorph_Hybrid_Overall": difference(hybrid, nomorph_hybrid, "Overall_KID", "Overall_FID"),
            "NoAux_Hybrid_minus_Raw_Overall": difference(hybrid, raw_overall, "Overall_KID", "Overall_FID"),
        }
        kid_improved = new["Black_KID"] < previous["Black_KID"]
        fid_improved = new["Black_FID"] < previous["Black_FID"]
        conclusion = ("NOAUX_EXPLORATORY_IMPROVED" if kid_improved and fid_improved else
                      "NOAUX_EXPLORATORY_MIXED" if kid_improved or fid_improved else
                      "NOAUX_EXPLORATORY_NOT_IMPROVED")
        comparison = {
            "status": conclusion, "differences": comparisons,
            "answers": {
                "removing_auxiliary_losses_lowered_Black_KID": kid_improved,
                "removing_auxiliary_losses_lowered_Black_FID": fid_improved,
                "NoAux_better_than_NoMorph_KID": new["Black_KID"] < nomorph["Black_KID"],
                "NoAux_better_than_NoMorph_FID": new["Black_FID"] < nomorph["Black_FID"],
                "Raw_Black_gap_change": {
                    "KID_gap_before": previous["Black_KID"] - raw["Black_KID"],
                    "KID_gap_after": new["Black_KID"] - raw["Black_KID"],
                    "FID_gap_before": previous["Black_FID"] - raw["Black_FID"],
                    "FID_gap_after": new["Black_FID"] - raw["Black_FID"],
                },
                "hybrid_Overall_KID_improved_vs_original_adaptive": hybrid["Overall_KID"] < adaptive["Overall_KID"],
                "hybrid_Overall_FID_improved_vs_original_adaptive": hybrid["Overall_FID"] < adaptive["Overall_FID"],
                "auxiliary_losses_worth_redesigning": kid_improved or fid_improved,
            },
            "interpretation_limit": "Bank A with V2 TRAIN references; possible checkpoint-training overlap; not independent validation.",
            "random_trajectory_limit": load(OUTPUT / "noaux_training_audit.json")["random_trajectory_limit"],
        }
        save(OUTPUT / "noaux_comparison.json", comparison)
        checks = load(OUTPUT / "noaux_preflight_audit.json")
        checks["metric_regression"] = regression
        checks["generation_integrity"] = {"status": "PASS", "slot_checks": 40, "image_SHA256_checks": 40,
            "same_normal_bbox_prompt_seed_scheduler_steps_CFG_crop_paste": True, "reroll_count": 0, "selection_count": 0}
        save(OUTPUT / "noaux_integrity_audit.json", checks)
        protocol = load(OUTPUT / "noaux_protocol.json")
        protocol.update(status="COMPLETE", checkpoint_sources={
            "Raw_SD2": "Frozen independent Raw checkpoints",
            "Original_MSDF": "Frozen Stage25 Full MSDF Black final_step2000",
            "NoMorph_Black": "Frozen Experiment 8 independent Black final_step2000",
            "NoAux_Black": "Experiment 9 independent Black final_step2000 from public SD2 base",
            "Hybrid_Flash": "Frozen MSDF gamma=0 Flash Bank A",
            "Hybrid_Black": "Experiment 9 NoAux Black Bank A",
        }, feature_reuse={"real_and_original_MSDF": "Gamma Sweep SHA-bound cache",
                          "Raw": "Experiment 7 SHA-bound cache",
                          "NoMorph": "Experiment 8 SHA-bound cache",
                          "new_NoAux_Black_forwards": 40})
        save(OUTPUT / "noaux_protocol.json", protocol)
        save(OUTPUT / "noaux_status.json", {**load(OUTPUT / "noaux_status.json"),
            "status": conclusion, "completed_at": now(), "evaluation_completed": True,
            "numerical_regression": "PASS", "new_Inception_forward_count": 40,
            "evaluation_runtime_seconds": time.monotonic() - started,
            "followup_started": False})
        create_report(conclusion, black_rows, overall_rows, comparisons, kid_improved, fid_improved)
        print(f"[完成] Experiment 9 status={conclusion}；停止，不启动后续实验。", flush=True)
    except Exception as error:
        update_status(status="EVALUATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise


def create_report(conclusion, black_rows, overall_rows, comparisons, kid_improved, fid_improved):
    b = {row["Method"]: row for row in black_rows}
    o = {row["Method"]: row for row in overall_rows}
    full, nomorph = b["Original MSDF"], b["MSDF-NoMorph"]
    new, raw = b["MSDF-NoAux"], b["Raw SD2"]
    adaptive = o["Original Class-Adaptive MSDF"]
    nomorph_hybrid, hybrid = o["NoMorph-Black Hybrid MSDF"], o["NoAux-Black Hybrid MSDF"]
    gap_before_kid = full["Black_KID"] - raw["Black_KID"]
    gap_after_kid = new["Black_KID"] - raw["Black_KID"]
    gap_before_fid = full["Black_FID"] - raw["Black_FID"]
    gap_after_fid = new["Black_FID"] - raw["Black_FID"]
    lines = [
        "# Experiment 9 — MSDF Black No Auxiliary Losses", "",
        f"Status: `{conclusion}`.", "",
        "G3 was trained independently from the same public SD2 base for exactly 2000 successful optimizer updates. Morphology alignment, both adapter branches, support head, dynamic gate and up-block injection remained enabled; gamma remained 1.0. Only the six auxiliary training-loss weights were set to zero, so the optimized scalar objective equaled conditional_loss.", "",
        "## Black results", "",
        "| Method | Black KID ↓ | Black FID ↓ |", "| --- | ---: | ---: |",
    ]
    for name in ("Raw SD2", "Original MSDF", "MSDF-NoMorph", "MSDF-NoAux"):
        lines.append(f"| {name} | {b[name]['Black_KID']:.10f} | {b[name]['Black_FID']:.6f} |")
    lines += ["", "## Overall results", "",
              "The hybrid combines the frozen gamma=0 Flash checkpoint/images with the independently trained NoAux Black checkpoint/images; it is not one jointly trained model.", "",
              "| Method | Overall KID ↓ | Overall FID ↓ |", "| --- | ---: | ---: |"]
    for name in ("Raw SD2", "Original Class-Adaptive MSDF", "NoMorph-Black Hybrid MSDF", "NoAux-Black Hybrid MSDF"):
        lines.append(f"| {name} | {o[name]['Overall_KID']:.10f} | {o[name]['Overall_FID']:.6f} |")
    lines += ["", "## Comparisons", "",
              "Negative differences mean NoAux/Hybrid is lower and better.", "",
              "| Comparison | Δ KID | Relative KID | Δ FID | Relative FID |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in comparisons.items():
        lines.append(f"| {name.replace('_', ' ')} | {row['KID']['absolute']:+.10f} | {row['KID']['relative_percent']:+.3f}% | {row['FID']['absolute']:+.6f} | {row['FID']['relative_percent']:+.3f}% |")
    lines += ["", "## Required findings", "",
              f"1. Removing the six auxiliary losses {'lowered' if kid_improved else 'did not lower'} Black KID ({new['Black_KID']:.10f} vs {full['Black_KID']:.10f}).",
              f"2. It {'lowered' if fid_improved else 'did not lower'} Black FID ({new['Black_FID']:.6f} vs {full['Black_FID']:.6f}).",
              f"3. Versus NoMorph, NoAux changed Black KID by {new['Black_KID'] - nomorph['Black_KID']:+.10f} and Black FID by {new['Black_FID'] - nomorph['Black_FID']:+.6f}.",
              f"4. The Black gap to Raw changed from KID {gap_before_kid:+.10f} to {gap_after_kid:+.10f}, and FID {gap_before_fid:+.6f} to {gap_after_fid:+.6f}.",
              f"5. The NoAux hybrid changed Overall KID by {hybrid['Overall_KID'] - adaptive['Overall_KID']:+.10f} versus original class-adaptive MSDF and {hybrid['Overall_KID'] - nomorph_hybrid['Overall_KID']:+.10f} versus NoMorph hybrid.",
              f"6. The NoAux hybrid changed Overall FID by {hybrid['Overall_FID'] - adaptive['Overall_FID']:+.6f} versus original class-adaptive MSDF and {hybrid['Overall_FID'] - nomorph_hybrid['Overall_FID']:+.6f} versus NoMorph hybrid.",
              f"7. Auxiliary-loss redesign is {'supported for independent validation' if kid_improved or fid_improved else 'not supported by this screening result'}.", "",
              "## Interpretation boundary", "",
              "This is a Bank-A development ablation evaluated against V2 TRAIN references that may overlap generator training. It is not an independent test result and cannot establish final-paper superiority.", "",
              "The historical G1 run did not preserve a step-0 adapter SHA or full RNG-state ledger. The same seed, sorted sample order, epoch permutation, reference rule, runtime and initialization code were reused, but exact byte-for-byte G1/G3 random-trajectory identity cannot be independently proven. The result therefore remains an exploratory controlled ablation rather than a strict causal estimate.", "",
              "The experiment stops here. No per-loss restoration, loss-weight search, new Flash model, architecture change, extra seed, Bank B/C, BootstrapGuard, detector training or other generator experiment was started.", ""]
    (OUTPUT / "NOAUX_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "train", "generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 9 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "preflight":
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
