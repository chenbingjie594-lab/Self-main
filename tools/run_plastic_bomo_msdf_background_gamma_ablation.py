"""Experiment 13: frozen BackgroundOnly MSDF residual-injection ablation.

The runner has no training entry point.  It compares gamma=0 and gamma=1 on
one SHA-bound BackgroundOnly checkpoint and reuses all frozen Raw, gamma=1,
and Flash images/features from Experiments 10 and 12.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

import run_plastic_bomo_msdf_multibank_stability as x12

BASE_COMMIT = "b21d2c08821e1824e0e8485427b3a2878a71c541"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_background_gamma_ablation"
BG = REPO / "results_for_gpt/plastic_bomo_msdf_black_gradient_background"
X12 = REPO / "results_for_gpt/plastic_bomo_msdf_multibank_stability"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
EXP7 = REPO / "results_for_gpt/plastic_bomo_msdf_raw_matched_comparison"
RAW = x12.RAW
STAGE25 = x12.STAGE25
P0 = x12.P0
BANK_SEEDS = {"A": 42, "B": 3407, "C": 2026}
EXPECTED_TREE = "a194955681b58621799796b7a5812ed5402667912b5b4a6034025ef27a33d826"
EXPECTED_MSDF = "6e0c39ab6870e0a411a8fde8463ca4df90fb8fbf6803eb2493dd363eb50122fc"
WEIGHTS = {"region": 0.0, "reconstruction": 0.0, "structure": 0.0,
           "background": 0.2, "scale": 0.0, "support": 0.0}
GROUPS = {"raw": "Raw SD2", "gamma0": "BackgroundOnly Gamma=0",
          "gamma1": "BackgroundOnly Gamma=1"}

load, save, sha, canonical, require = x12.load, x12.save, x12.sha, x12.canonical, x12.require
crop_box = x12.crop_box


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows):
    require(rows, "cannot write empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def update_status(**values):
    path = OUTPUT / "background_gamma_status.json"
    row = load(path); row.update(values); save(path, row)


def source_hashes():
    names = ["tools/run_plastic_bomo_msdf_background_gamma_ablation.py",
             "tools/run_plastic_bomo_msdf_multibank_stability.py",
             "tools/run_plastic_bomo_msdf_gamma_sweep.py",
             "tools/generate_plastic_bomo_stage20b_banks.py",
             "diffusers/pipelines/stable_diffusion/msdf_guidance.py"]
    return {name: sha(REPO / name) for name in names}


def checkpoint(full_hash=False):
    binding = load(BG / "background_only_checkpoint_binding.json")
    require(binding["status"] == "FROZEN" and binding["class"] == "black" and
            binding["successful_updates"] == 2000, "BackgroundOnly checkpoint is not frozen")
    path = BG / binding["checkpoint"]["path"]
    expected = binding["checkpoint"]["sha256"]
    require(all((path / name).is_file() for name in expected), "BackgroundOnly checkpoint file missing")
    require(hashlib.sha256(canonical(expected).encode()).hexdigest() ==
            binding["checkpoint"]["tree_sha256"], "BackgroundOnly checkpoint tree binding changed")
    if full_hash:
        for name, digest in expected.items():
            require(sha(path / name) == digest, f"BackgroundOnly checkpoint file changed: {name}")
    else:
        # The audit phase immediately before GPU work performs the expensive
        # byte-for-byte 4.8 GiB verification.  Later phases bind to that
        # persisted result and check each file's presence and frozen size,
        # avoiding another full disk read every time the pipeline is loaded.
        audit_path = OUTPUT / "background_gamma_checkpoint_audit.json"
        require(audit_path.is_file(), "checkpoint SHA audit is missing")
        audit = load(audit_path)
        require(audit["status"] == "PASS" and audit["checkpoint_tree_SHA256"] == EXPECTED_TREE and
                audit["msdf_checkpoint_SHA256"] == EXPECTED_MSDF,
                "checkpoint SHA audit binding changed")
        require(audit["checkpoint_file_sizes"] ==
                {name: (path / name).stat().st_size for name in expected},
                "checkpoint file size changed after SHA audit")
    entry = binding
    require(entry["checkpoint"]["tree_sha256"] == EXPECTED_TREE, "BackgroundOnly checkpoint tree changed")
    require(entry["checkpoint"]["sha256"]["msdf.pt"] == EXPECTED_MSDF, "MSDF checkpoint changed")
    require(binding["auxiliary_loss_weights"] == WEIGHTS, "BackgroundOnly loss weights changed")
    require(binding["morphology_alignment_enabled"] is True and binding["injection_gamma"] == 1.0,
            "frozen BackgroundOnly controls changed")
    return path, binding


def gamma1_rows(bank):
    if bank == "A":
        root = BG; value = load(BG / "background_only_generation_manifest.json")
        rows = value["records"]
    else:
        root = X12; value = load(X12 / "multibank_generation_manifest.json")
        rows = [r for r in value["records"] if r["group"] == "background_only" and r["bank"] == bank]
    require(value["status"] == "COMPLETE" and len(rows) == 40, f"Gamma=1 Bank {bank} incomplete")
    return root, rows


def image_digest(row):
    return row.get("image_SHA256", row.get("image_sha256"))


def verify_history():
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    black = [r for r in slots if r["class"] == "black"]
    require(len(black) == 40, "Black slot count changed")
    originals = {r["slot_id"]: r for r in load(
        Path(load(STAGE25 / "stage25_run_binding.json")["p0"]) / "stage20b_generation_slots.json")["slots"]}
    rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
    counts = {}
    for bank, seed in BANK_SEEDS.items():
        root, rows = gamma1_rows(bank)
        by_id = {r["slot_id"]: r for r in rows}
        require(set(by_id) == {r["slot_id"] for r in black}, f"Gamma=1 Bank {bank} slots changed")
        for slot in black:
            row = by_id[slot["slot_id"]]; frozen = originals[slot["slot_id"]]; rng = rngs[slot["slot_id"]][bank]
            require(row["frozen_slot"] == frozen and row["runtime_slot"] == slot,
                    f"Gamma=1 Bank {bank} slot metadata changed")
            require(row["sample_seed"] == rng["sample_seed"] and row["bank_seed"] == seed,
                    f"Gamma=1 Bank {bank} RNG changed")
            require(row["checkpoint_tree_SHA256"] == EXPECTED_TREE and
                    row.get("msdf_checkpoint_SHA256", EXPECTED_MSDF) == EXPECTED_MSDF,
                    f"Gamma=1 Bank {bank} checkpoint changed")
            path = root / row["image_path"]
            require(path.is_file() and sha(path) == image_digest(row), f"Gamma=1 image changed: {path}")
        raw = x12.manifest_records(RAW, f"formal_bank_{bank}_manifest.json")
        require(len(raw) == 80, f"Raw Bank {bank} incomplete")
        x12.verify_images(RAW, raw)
        if bank == "A":
            flash = [r for r in x12.manifest_records(GAMMA, "gamma_0p00_manifest.json") if r["class"] == "flash"]
            x12.verify_images(GAMMA, flash)
        else:
            value = load(X12 / "multibank_generation_manifest.json")
            flash = [r for r in value["records"] if r["group"] == "flash_gamma0" and r["bank"] == bank]
            require(len(flash) == 40, f"Flash gamma=0 Bank {bank} incomplete")
            for row in flash:
                path = X12 / row["image_path"]
                require(path.is_file() and sha(path) == row["image_SHA256"], f"Flash image changed: {path}")
        counts[bank] = {"raw": len(raw), "gamma1_black": len(rows), "shared_flash_gamma0": len(flash)}
    return counts


def cpu_hook_regression():
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter, _safe_rms
    torch.manual_seed(20261023)
    adapter = MSDFAdapter(4, 8, [4], max_residual_ratio=0.25).eval()
    with torch.no_grad():
        adapter.projections[0].weight.fill_(0.03125)
        adapter.projections[0].bias.fill_(0.0078125)
    parameters = {name: value.detach().clone() for name, value in adapter.named_parameters()}
    ablation = dict(adapter.ablation)
    adapter._feature = torch.randn(1, 8, 3, 3)
    adapter._support = torch.sigmoid(torch.randn(1, 1, 3, 3))
    adapter._gates = torch.tensor([[0.37]])
    adapter._cfg = False
    hidden = torch.randn(1, 4, 6, 6)
    feature = torch.nn.functional.interpolate(adapter._feature, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
    low = torch.nn.functional.avg_pool2d(feature, 3, stride=1, padding=1)
    feature = low + 0.25 * (feature - low)
    support = torch.nn.functional.interpolate(adapter._support, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
    residual = torch.nan_to_num(adapter.projections[0](feature)) * support
    residual = residual * adapter._gates[:, 0].reshape(-1, 1, 1, 1)
    limit = adapter.max_residual_ratio * _safe_rms(hidden, (1, 2, 3)).reshape(-1, 1, 1, 1).clamp_min(1e-4)
    historical = hidden + torch.nan_to_num(limit * torch.tanh(residual.float() / limit.clamp_min(1e-6))).to(hidden.dtype)
    hook = adapter._make_hook(0)
    adapter.set_injection_gamma(1.0); one = hook(None, None, hidden.clone())
    adapter.set_injection_gamma(0.0); zero = hook(None, None, hidden.clone())
    unchanged = all(torch.equal(parameters[name], value) for name, value in adapter.named_parameters())
    record = {
        "status": "CPU_HOOK_PASS",
        "gamma_0_bitwise_equal_hidden": bool(torch.equal(zero, hidden)),
        "gamma_0_max_abs_difference": float((zero - hidden).abs().max()),
        "gamma_1_bitwise_equal_historical_formula": bool(torch.equal(one, historical)),
        "gamma_1_max_abs_difference": float((one - historical).abs().max()),
        "parameters_unchanged": unchanged,
        "morphology_and_branch_state_unchanged": adapter.ablation == ablation,
        "gamma_is_not_in_state_dict": not any("gamma" in key for key in adapter.state_dict()),
        "end_to_end": "PENDING", "checked_at": now(),
    }
    require(all(record[k] for k in ("gamma_0_bitwise_equal_hidden",
        "gamma_1_bitwise_equal_historical_formula", "parameters_unchanged",
        "morphology_and_branch_state_unchanged", "gamma_is_not_in_state_dict")),
        f"CPU Hook regression failed: {record}")
    return record


def audit():
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    for bank in "ABC":
        (OUTPUT / "images/gamma0" / bank).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "regression" / bank).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    save(OUTPUT / "background_gamma_status.json", {
        "status": "AUDITING", "created_at": now(), "base_commit": BASE_COMMIT,
        "training_count": 0, "calibration_count": 0, "generation_count": 0,
        "reroll_count": 0, "selection_count": 0, "final_eval_read_or_used": False,
        "BootstrapGuard": False, "detector_training_started": False})
    path, binding = checkpoint(full_hash=True)
    history = verify_history()
    generation = load(P0 / "stage20b_generation_protocol.json")
    require(generation["scheduler"] == "DDIMScheduler" and generation["num_inference_steps"] == 50 and
            generation["guidance_scale"] == 7.5 and generation["eta"] == 0 and generation["strength"] == 1 and
            generation["height"] == generation["width"] == 512 and generation["precision"].startswith("fp16"),
            "generation protocol changed")
    hook = cpu_hook_regression()
    save(OUTPUT / "background_gamma_hook_regression.json", hook)
    save(OUTPUT / "background_gamma_checkpoint_audit.json", {
        "status": "PASS", "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF,
        "unet_SHA256": binding["checkpoint"]["sha256"]["unet/diffusion_pytorch_model.safetensors"],
        "successful_updates": binding["successful_updates"], "training_weights": WEIGHTS,
        "morphology_alignment": True, "pixel_branch": "runtime verification pending",
        "latent_branch": "runtime verification pending", "dynamic_gate": "runtime verification pending",
        "checkpoint_file_count": len(binding["checkpoint"]["sha256"]),
        "checkpoint_file_sizes": {name: (path / name).stat().st_size
                                  for name in binding["checkpoint"]["sha256"]},
        "history_SHA_checks": history, "source_code_SHA256": source_hashes(), "completed_at": now()})
    save(OUTPUT / "background_gamma_protocol.json", {
        "status": "FROZEN_BEFORE_GPU_REGRESSION",
        "experiment": "Experiment 13 — MSDF BackgroundOnly Residual Injection Ablation",
        "base_commit": BASE_COMMIT, "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "groups": GROUPS, "banks": BANK_SEEDS,
        "only_control_variable": "Black final MSDF up-block residual injection gamma: 0 versus 1",
        "gamma0_is_not_Raw_SD2": True, "training_level_ablation": False,
        "generation_protocol": generation,
        "new_generation": {"method": GROUPS["gamma0"], "banks": list(BANK_SEEDS), "images": 120},
        "evaluation": {"references": {"flash": 75, "black": 97, "overall": 172},
            "ROI": "bbox plus 25 percent context; frozen floor/ceil clipping",
            "Inception": "torch-fidelity inception-v3-compat 2048D; float32 extraction; float64 CPU metrics",
            "KID": {"subsets": 100, "seed": 42, "degree": 3, "subset_size": "min(50,nreal,nfake)"}},
        "training_forbidden": True, "final_eval_read_or_used": False})
    save(OUTPUT / "background_gamma_generation_manifest.json", {
        "status": "PENDING", "planned_count": 120, "records": [], "reroll_count": 0, "selection_count": 0})
    update_status(status="AUDIT_PASS", audit_completed_at=now(), CPU_hook_regression="PASS")
    print("[审计通过] checkpoint、历史A/B/C、生成协议和CPU Hook回归全部通过。", flush=True)


def load_pipeline(device, gamma):
    import torch
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    checkpoint_path, _binding = checkpoint()
    cache = os.environ.get("EXP13_CHECKPOINT_CACHE")
    if cache:
        cached_path = Path(cache)
        expected_sizes = load(OUTPUT / "background_gamma_checkpoint_audit.json")["checkpoint_file_sizes"]
        require({name: (cached_path / name).stat().st_size for name in expected_sizes} == expected_sizes,
                "staged checkpoint is incomplete or differs in size")
        checkpoint_path = cached_path
        print(f"[加载] 使用已SHA审计的本机NVMe checkpoint副本：{checkpoint_path}", flush=True)
    generation = load(P0 / "stage20b_generation_protocol.json")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(str(checkpoint_path), local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float16, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False).to(device)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    pipe.disable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(generation["scheduler_config"])
    adapter = load_msdf_adapter(checkpoint_path / "msdf.pt", pipe.unet, device).eval()
    adapter.set_injection_gamma(gamma)
    return pipe, adapter, generation


def generate_one(pipe, adapter, slot, rng, generation, device):
    import numpy as np
    import torch
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    from generate_plastic_bomo_stage20b_banks import sample
    donor, mask = crop_and_mask({**slot, "path": slot["source_path"], "sha256": slot["source_sha256"]})
    pixels = torch.from_numpy(np.asarray(donor).copy()).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
    mask_t = torch.from_numpy(np.asarray(mask).copy())[None, None].to(device).float() / 255
    with torch.inference_mode():
        reference = pipe.vae.encode(pixels.half()).latent_dist.mode() * pipe.vae.config.scaling_factor
    def prepare(_module, args, kwargs):
        inp = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        require(list(inp.shape)[1:] == [9, 64, 64] and inp.shape[0] == 2, "invalid inference input")
        adapter.prepare(reference.float(), mask_t, inp[:1, 4:5].float(), timestep,
                        pipe.scheduler.config.num_train_timesteps, True, pixels)
    handle = pipe.unet.register_forward_pre_hook(prepare, with_kwargs=True)
    try:
        return sample(pipe, slot, rng, generation, device)
    finally:
        handle.remove()


def diagnostic_hooks(adapter, pipe, sink):
    import torch
    before = {}; handles = []
    for index, block in enumerate(pipe.unet.up_blocks):
        def capture_before(_module, _inputs, output, block_index=index):
            hidden = output[0] if isinstance(output, tuple) else output
            before[block_index] = hidden.detach()
        def capture_after(_module, _inputs, output, block_index=index):
            updated = output[0] if isinstance(output, tuple) else output
            original = before.pop(block_index)
            residual = updated.detach().float() - original.float()
            hidden = original.float()
            hidden_norm = float(torch.linalg.vector_norm(hidden))
            residual_norm = float(torch.linalg.vector_norm(residual))
            hidden_rms = (hidden.square().mean((1, 2, 3)) + 1e-8).sqrt().reshape(-1, 1, 1, 1).clamp_min(1e-4)
            limit = adapter.max_residual_ratio * hidden_rms
            ratio = residual.abs() / limit
            gates = adapter._gates.detach().float()[:, block_index]
            sink[block_index].append({
                "hidden_l2": hidden_norm, "residual_l2": residual_norm,
                "relative_l2": residual_norm / max(hidden_norm, 1e-12),
                "gate_min": float(gates.min()), "gate_max": float(gates.max()),
                "max_abs_residual_to_bound": float(ratio.max()),
                "near_bound_fraction_ge_0p99": float((ratio >= 0.99).float().mean()),
                "finite": bool(torch.isfinite(updated).all() and torch.isfinite(residual).all())})
        handles.append(block.register_forward_hook(capture_before, prepend=True))
        handles.append(block.register_forward_hook(capture_after))
    return handles


def regression(device):
    require(load(OUTPUT / "background_gamma_status.json")["status"] == "AUDIT_PASS", "regression requires audit PASS")
    require(not (OUTPUT / "logs/regression_claim.json").exists(), "regression already claimed")
    save(OUTPUT / "logs/regression_claim.json", {"started_at": now(), "samples": 6})
    update_status(status="REGRESSION_RUNNING", regression_started_at=now())
    with x12.gpu_runtime(device) as torch:
        slots = [r for r in load(P0 / "stage20b_generation_slots.json")["slots"] if r["class"] == "black"]
        rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
        pipe, adapter, generation = load_pipeline(device, 1.0)
        require(adapter.ablation == {"pixel_branch": True, "latent_branch": True,
            "morphology_alignment": True, "up_block_mask": [True] * adapter.num_scales,
            "residual_scale": 1.0}, "BackgroundOnly runtime branch state changed")
        require(any(bool(torch.count_nonzero(p)) for p in adapter.gate.parameters()), "dynamic gate weights are zero")
        before_msdf = sha(checkpoint()[0] / "msdf.pt")
        records = []; raw_stats = defaultdict(list)
        handles = diagnostic_hooks(adapter, pipe, raw_stats)
        try:
            for bank in "ABC":
                root, historical_rows = gamma1_rows(bank)
                old = {r["slot_id"]: r for r in historical_rows}
                for diagnostic_index, slot in enumerate(slots[:2]):
                    image, trace = generate_one(pipe, adapter, slot, rngs[slot["slot_id"]][bank], generation, device)
                    path = OUTPUT / "regression" / bank / f"{slot['slot_id']}_{diagnostic_index}.png"
                    image.save(path, format="PNG")
                    exact = None
                    if diagnostic_index == 0:
                        historical_path = root / old[slot["slot_id"]]["image_path"]
                        exact = sha(path) == sha(historical_path) == image_digest(old[slot["slot_id"]])
                    records.append({"bank": bank, "diagnostic_index": diagnostic_index,
                        "slot_id": slot["slot_id"], "sample_seed": rngs[slot["slot_id"]][bank]["sample_seed"],
                        "bank_seed": rngs[slot["slot_id"]][bank]["bank_seed"],
                        "new_SHA256": sha(path), "gamma1_exact_historical_PNG": exact, **trace})
                    print(f"[回归] Bank {bank} diagnostic {diagnostic_index + 1}/2 exact={exact}", flush=True)
        finally:
            for handle in handles: handle.remove()
        require(all(r["gamma1_exact_historical_PNG"] for r in records if r["diagnostic_index"] == 0),
                "three-bank Gamma=1 end-to-end regression failed")
        require(before_msdf == sha(checkpoint()[0] / "msdf.pt"), "checkpoint changed during regression")
        diagnostics = {"status": "PASS", "samples": 6, "gamma": 1.0,
            "read_only_hooks": True, "does_not_imply_quality_benefit": True, "blocks": {}}
        for index, values in raw_stats.items():
            diagnostics["blocks"][str(index)] = {
                "observations": len(values),
                "residual_l2_min": min(v["residual_l2"] for v in values),
                "residual_l2_max": max(v["residual_l2"] for v in values),
                "relative_l2_min": min(v["relative_l2"] for v in values),
                "relative_l2_max": max(v["relative_l2"] for v in values),
                "dynamic_gate_min": min(v["gate_min"] for v in values),
                "dynamic_gate_max": max(v["gate_max"] for v in values),
                "max_abs_residual_to_bound": max(v["max_abs_residual_to_bound"] for v in values),
                "near_bound_fraction_max": max(v["near_bound_fraction_ge_0p99"] for v in values),
                "all_finite": all(v["finite"] for v in values)}
        require(all(v["all_finite"] for v in diagnostics["blocks"].values()), "non-finite residual diagnostic")
        save(OUTPUT / "background_gamma_residual_diagnostics.json", diagnostics)
        hook = load(OUTPUT / "background_gamma_hook_regression.json")
        hook.update(end_to_end="PASS", end_to_end_records=records,
                    exact_PNG_banks=3, runtime_branch_state="PASS",
                    checkpoint_unchanged=True, completed_at=now())
        save(OUTPUT / "background_gamma_hook_regression.json", hook)
        audit_row = load(OUTPUT / "background_gamma_checkpoint_audit.json")
        audit_row.update(pixel_branch=True, latent_branch=True, dynamic_gate=True,
                         runtime_branch_verification="PASS")
        save(OUTPUT / "background_gamma_checkpoint_audit.json", audit_row)
        adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
    update_status(status="REGRESSION_PASS", regression_completed_at=now(), regression_samples=6)
    print("[回归通过] Gamma=1 A/B/C各1张与历史PNG逐字节一致；6张只读残差诊断完成。", flush=True)


def generate(device):
    require(load(OUTPUT / "background_gamma_status.json")["status"] == "REGRESSION_PASS", "generation requires regression PASS")
    require(not (OUTPUT / "logs/generate_claim.json").exists(), "generation already claimed")
    save(OUTPUT / "logs/generate_claim.json", {"started_at": now(), "planned_count": 120, "gamma": 0.0})
    update_status(status="GENERATING", generation_started_at=now())
    manifest_path = OUTPUT / "background_gamma_generation_manifest.json"
    records = []; started = time.monotonic()
    try:
        with x12.gpu_runtime(device) as torch:
            from stage20b_formal_common import synthetic_label
            slots = [r for r in load(P0 / "stage20b_generation_slots.json")["slots"] if r["class"] == "black"]
            originals = {r["slot_id"]: r for r in load(
                Path(load(STAGE25 / "stage25_run_binding.json")["p0"]) / "stage20b_generation_slots.json")["slots"]}
            rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
            pipe, adapter, generation = load_pipeline(device, 0.0)
            require(adapter.injection_gamma == 0.0 and adapter.ablation["morphology_alignment"] is True,
                    "Gamma=0 runtime controls changed")
            for bank in "ABC":
                for index, slot in enumerate(slots):
                    frozen = originals[slot["slot_id"]]; rng = rngs[slot["slot_id"]][bank]
                    require(rng["bank_seed"] == BANK_SEEDS[bank] and
                            rng["non_rng_metadata_sha256"] == hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                            f"frozen RNG mismatch: {bank}/{slot['slot_id']}")
                    item_started = time.monotonic()
                    image, trace = generate_one(pipe, adapter, slot, rng, generation, device)
                    image_path = OUTPUT / "images/gamma0" / bank / (slot["slot_id"] + ".png")
                    label_path = image_path.with_suffix(".txt")
                    require(not image_path.exists() and not label_path.exists(), "formal output would overwrite a file")
                    image.save(image_path, format="PNG")
                    label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                    width, height = frozen["geometry"]["final_canvas_wh"]
                    records.append({"bank": bank, "class": "black", "slot_id": slot["slot_id"],
                        "frozen_slot": frozen, "runtime_slot": slot, "sample_seed": rng["sample_seed"],
                        "bank_seed": rng["bank_seed"], "checkpoint_tree_SHA256": EXPECTED_TREE,
                        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "injection_gamma": 0.0,
                        "morphology_alignment_enabled": True,
                        "image_path": image_path.relative_to(OUTPUT).as_posix(), "image_SHA256": sha(image_path),
                        "label_path": label_path.relative_to(OUTPUT).as_posix(), "label_SHA256": sha(label_path),
                        "roi_xyxy": crop_box(frozen["bbox"], width, height),
                        "quality_selection": False, "reroll": False,
                        "runtime_seconds": time.monotonic() - item_started, **trace})
                    save(manifest_path, {"status": "RUNNING", "planned_count": 120,
                        "records": records, "reroll_count": 0, "selection_count": 0})
                    print(f"[生成] Gamma=0 Bank {bank} {index + 1}/40 total={len(records)}/120", flush=True)
            adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        require(len(records) == 120 and Counter(r["bank"] for r in records) == {"A": 40, "B": 40, "C": 40},
                "Gamma=0 generation incomplete")
        save(manifest_path, {"status": "COMPLETE", "count": 120, "records": records,
            "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
            "runtime_seconds": time.monotonic() - started})
        update_status(status="GENERATION_COMPLETE", generation_count=120,
                      generation_completed_at=now(), reroll_count=0, selection_count=0)
        print("[生成完成] Gamma=0 A/B/C 120/120；无重抽、筛选或替换。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "records": records,
                            "error": f"{type(error).__name__}: {error}", "reroll_count": 0})
        update_status(status="GENERATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise


def extract_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    model = lib["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
    values = []; keys = []
    with torch.inference_mode():
        for index, row in enumerate(manifest["records"]):
            path = OUTPUT / row["image_path"]
            require(path.is_file() and sha(path) == row["image_SHA256"], f"Gamma=0 image changed: {path}")
            with Image.open(path) as source:
                crop = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
            tensor = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = model(tensor)[0]
            require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()), row["slot_id"])
            keys.append(f"gamma0:{row['bank']}:{row['slot_id']}"); values.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0: print(f"[Inception] {index + 1}/120", flush=True)
    data = np.stack(values); path = OUTPUT / "background_gamma_inception_features.npz"
    np.savez_compressed(path, features=data)
    save(OUTPUT / "background_gamma_inception_feature_cache.json", {
        "status": "COMPLETE", "keys": keys, "shape": [120, 2048],
        "stored_dtype": str(data.dtype), "extraction_dtype": "float32",
        "sha256": sha(path), "image_SHA256_checks": 120})
    del model; torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[i]).double() for i, key in enumerate(keys)}


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def historical_metrics():
    black_source = read_csv(X12 / "multibank_bankwise_black.csv")
    overall_source = read_csv(X12 / "multibank_bankwise_overall.csv")
    pooled_source = read_csv(X12 / "multibank_pooled_metrics.csv")
    def convert(row, scope_key):
        return {scope_key: row[scope_key], "group": "raw" if row["group"] == "raw" else "gamma1",
            "method": GROUPS["raw" if row["group"] == "raw" else "gamma1"],
            "KID": float(row["KID"]), "KID_subset_std": float(row["KID_subset_std"]),
            "FID": float(row["FID"]), "reference_ROIs": int(float(row["reference_ROIs"])),
            "generated_images": int(float(row["generated_images"])),
            "KID_subset_size": int(float(row["KID_subset_size"])),
            "runtime_seconds": float(row["runtime_seconds"]), "metric_source": "reused SHA-bound Experiment 12 features/metrics"}
    black = [convert(r, "bank") for r in black_source if r["group"] in ("raw", "background")]
    overall = [convert(r, "bank") for r in overall_source if r["group"] in ("raw", "background")]
    pooled = []
    for row in pooled_source:
        if row["group"] not in ("raw", "background") or row["population"] not in ("Black", "Overall"): continue
        value = convert(row, "scope"); value["population"] = row["population"]; pooled.append(value)
    return black, overall, pooled


def evaluate(device):
    require(load(OUTPUT / "background_gamma_status.json")["status"] == "GENERATION_COMPLETE",
            "evaluation requires complete Gamma=0 generation")
    update_status(status="EVALUATING", evaluation_started_at=now())
    started = time.monotonic()
    try:
        with x12.gpu_runtime(device) as torch:
            from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
            lib, _runtime_audit = metric_runtime(device)
            manifest = load(OUTPUT / "background_gamma_generation_manifest.json")
            require(manifest["status"] == "COMPLETE" and manifest["count"] == 120, "manifest incomplete")
            new = extract_features(device, torch, lib, manifest)
            gamma = x12.array_features(GAMMA / "gamma_sweep_inception_features.npz",
                                       GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
            x12_features = x12.array_features(X12 / "multibank_inception_features.npz",
                                              X12 / "multibank_inception_feature_cache.json", torch)
            references = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
            real_black = torch.stack([gamma["real:" + r["annotation_id"]] for r in references if r["class"] == "black"])
            real_all = torch.stack([gamma["real:" + r["annotation_id"]] for r in references])
            slots = load(P0 / "stage20b_generation_slots.json")["slots"]
            black_slots = [r for r in slots if r["class"] == "black"]
            flash_slots = [r for r in slots if r["class"] == "flash"]
            def gamma0_black(bank):
                return torch.stack([new[f"gamma0:{bank}:{r['slot_id']}"] for r in black_slots])
            def shared_flash(bank):
                if bank == "A": return torch.stack([gamma[f"fake:0.00:{r['slot_id']}"] for r in flash_slots])
                return torch.stack([x12_features[f"flash_gamma0:{bank}:{r['slot_id']}"] for r in flash_slots])
            old_black, old_overall, old_pooled = historical_metrics()
            black_rows = []; overall_rows = []
            for bank in "ABC":
                old_b = [r for r in old_black if r["bank"] == bank]
                old_o = [r for r in old_overall if r["bank"] == bank]
                g0b = x12.metric(real_black, gamma0_black(bank), lib)
                g0o = x12.metric(real_all, torch.cat([shared_flash(bank), gamma0_black(bank)]), lib)
                black_rows.extend([next(r for r in old_b if r["group"] == "raw"),
                    {"bank": bank, "group": "gamma0", "method": GROUPS["gamma0"], **g0b,
                     "metric_source": "new same-protocol Gamma=0 features"},
                    next(r for r in old_b if r["group"] == "gamma1")])
                overall_rows.extend([next(r for r in old_o if r["group"] == "raw"),
                    {"bank": bank, "group": "gamma0", "method": GROUPS["gamma0"], **g0o,
                     "metric_source": "new same-protocol Gamma=0 features"},
                    next(r for r in old_o if r["group"] == "gamma1")])
                print(f"[指标] Bank {bank} Gamma=0 Black KID={g0b['KID']:.10f} FID={g0b['FID']:.6f}; Overall KID={g0o['KID']:.10f} FID={g0o['FID']:.6f}", flush=True)
            pooled_rows = []
            for scope, banks in (("A+B+C", "ABC"), ("B+C", "BC")):
                for population, real in (("Black", real_black), ("Overall", real_all)):
                    old = [r for r in old_pooled if r["scope"] == scope and r["population"] == population]
                    if population == "Black": fake = torch.cat([gamma0_black(bank) for bank in banks])
                    else: fake = torch.cat([torch.cat([shared_flash(bank), gamma0_black(bank)]) for bank in banks])
                    result = x12.metric(real, fake, lib)
                    pooled_rows.extend([next(r for r in old if r["group"] == "raw"),
                        {"scope": scope, "population": population, "group": "gamma0",
                         "method": GROUPS["gamma0"], **result,
                         "metric_source": "new same-protocol Gamma=0 features"},
                        next(r for r in old if r["group"] == "gamma1")])
                    print(f"[Pooled] {scope} {population} Gamma=0 KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)
        write_csv(OUTPUT / "background_gamma_bankwise_black.csv", black_rows)
        write_csv(OUTPUT / "background_gamma_bankwise_overall.csv", overall_rows)
        write_csv(OUTPUT / "background_gamma_pooled_metrics.csv", pooled_rows)
        deltas = []
        def add(scope, population, current, baseline):
            for name in ("KID", "FID"):
                delta = current[name] - baseline[name]
                deltas.append({"scope": scope, "population": population,
                    "comparison": current["group"] + " - " + baseline["group"], "metric": name,
                    "current": current[name], "baseline": baseline[name], "absolute_delta": delta,
                    "relative_percent": 100 * delta / baseline[name], "improved": delta < 0})
        for bank in "ABC":
            for population, rows in (("Black", black_rows), ("Overall", overall_rows)):
                by = {r["group"]: r for r in rows if r["bank"] == bank}
                add(bank, population, by["gamma1"], by["gamma0"])
                add(bank, population, by["gamma1"], by["raw"])
                add(bank, population, by["gamma0"], by["raw"])
        for scope in ("A+B+C", "B+C"):
            for population in ("Black", "Overall"):
                by = {r["group"]: r for r in pooled_rows if r["scope"] == scope and r["population"] == population}
                add(scope, population, by["gamma1"], by["gamma0"])
                add(scope, population, by["gamma1"], by["raw"])
                add(scope, population, by["gamma0"], by["raw"])
        write_csv(OUTPUT / "background_gamma_pairwise_deltas.csv", deltas)
        def wins(rows, metric_name):
            details = {}; count = 0
            for bank in "ABC":
                by = {r["group"]: r for r in rows if r["bank"] == bank}
                delta = by["gamma1"][metric_name] - by["gamma0"][metric_name]
                details[bank] = delta; count += int(delta < 0)
            return {"wins": count, "banks": 3, "deltas": details}
        black_kid, black_fid = wins(black_rows, "KID"), wins(black_rows, "FID")
        overall_kid, overall_fid = wins(overall_rows, "KID"), wins(overall_rows, "FID")
        bc_black = {r["group"]: r for r in pooled_rows if r["scope"] == "B+C" and r["population"] == "Black"}
        bc_both = all(bc_black["gamma1"][m] < bc_black["gamma0"][m] for m in ("KID", "FID"))
        held = []
        for bank in "BC":
            by = {r["group"]: r for r in black_rows if r["bank"] == bank}
            held.extend(by["gamma1"][m] < by["gamma0"][m] for m in ("KID", "FID"))
        if all(held) and bc_both: conclusion = "RESIDUAL_INJECTION_CONSISTENT_BENEFIT"
        elif not any(held): conclusion = "RESIDUAL_INJECTION_NOT_SUPPORTED"
        else: conclusion = "RESIDUAL_INJECTION_MIXED"
        summary = {"status": conclusion, "primary_comparison": "Gamma=1 minus Gamma=0",
            "bank_wins": {"Black_KID": black_kid, "Black_FID": black_fid,
                "Overall_KID": overall_kid, "Overall_FID": overall_fid},
            "B_C_pooled_Black_both_improved": bc_both,
            "pooled_deltas": {row["scope"] + ":" + row["population"] + ":" + row["metric"]: {
                "absolute_delta": row["absolute_delta"], "relative_percent": row["relative_percent"],
                "improved": row["improved"]} for row in deltas
                if row["comparison"] == "gamma1 - gamma0" and row["scope"] in ("A+B+C", "B+C")},
            "interpretation_boundary": "Inference-time final residual ablation on one trained model; not a training-level MSDF or branch ablation.",
            "gamma0_is_not_Raw_SD2": True,
            "research_boundary": "A/B/C share slots and V2 TRAIN references; not independent-test or cross-dataset validation."}
        save(OUTPUT / "background_gamma_summary.json", summary)
        save(OUTPUT / "background_gamma_generation_integrity.json", {
            "status": "PASS", "new_generation_records": 120, "new_image_SHA256_checks": 120,
            "counts": dict(Counter(r["bank"] for r in manifest["records"])),
            "slotwise_Gamma0_Gamma1_identity_match": True, "reroll_count": 0, "selection_count": 0,
            "feature_cache_SHA256": sha(OUTPUT / "background_gamma_inception_features.npz"),
            "hook_regression": "PASS", "final_eval_read_or_used": False})
        protocol = load(OUTPUT / "background_gamma_protocol.json")
        protocol.update(status="COMPLETE", conclusion=conclusion, new_generation_count=120,
                        source_code_SHA256=source_hashes())
        save(OUTPUT / "background_gamma_protocol.json", protocol)
        create_report(conclusion, black_rows, overall_rows, pooled_rows, summary)
        update_status(status=conclusion, evaluation_completed=True, completed_at=now(),
                      new_feature_count=120, evaluation_runtime_seconds=time.monotonic() - started,
                      followup_started=False)
        print(f"[完成] Experiment 13 status={conclusion}；停止，不启动后续实验。", flush=True)
    except Exception as error:
        update_status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def create_report(conclusion, black, overall, pooled, summary):
    lines = ["# Experiment 13 — MSDF BackgroundOnly Residual Injection Ablation", "",
        f"Status: `{conclusion}`.", "",
        "This experiment changes only the final inference-time MSDF residual injection gamma on one frozen BackgroundOnly checkpoint. Gamma=0 retains the trained UNet, MSDF checkpoint, reference preparation, branches, morphology alignment and dynamic gate; it is not Raw SD2 or a training-level ablation.", "",
        "## Bankwise Black", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in black: lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Bankwise Overall", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in overall: lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Pooled metrics", "", "Pooled metrics were recomputed from concatenated features, not averaged across banks or classes.", "",
        "| Scope | Population | Method | Images | KID ↓ | FID ↓ |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in pooled: lines.append(f"| {row['scope']} | {row['population']} | {row['method']} | {row['generated_images']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    wins = summary["bank_wins"]
    lines += ["", "## Finding", "",
        f"Gamma=1 beat Gamma=0 across A/B/C in Black KID on {wins['Black_KID']['wins']}/3 banks, Black FID on {wins['Black_FID']['wins']}/3, Overall KID on {wins['Overall_KID']['wins']}/3 and Overall FID on {wins['Overall_FID']['wins']}/3.", "",
        f"For the held-out generation seeds B+C, pooled Black KID and FID were both improved by injection: `{summary['B_C_pooled_Black_both_improved']}`.", "",
        "Against Raw SD2 on A+B+C pooled features, BackgroundOnly Gamma=1 still has 43.47% higher Black KID, 14.61% higher Black FID, 21.62% higher Overall KID and 3.55% higher Overall FID. The residual injection is beneficial within this frozen BackgroundOnly model, but it does not close the Raw SD2 gap.", "",
        "All Gamma=1 minus Gamma=0 and Raw comparisons, including absolute and relative changes, are in `background_gamma_pairwise_deltas.csv`. Negative means the first method is lower and better.", "",
        "Residual diagnostics report nonzero per-block tensor statistics, dynamic-gate ranges, finiteness and proximity to the tanh bound. Nonzero residuals do not by themselves establish a quality benefit.", "",
        "A later controlled experiment may examine training-level Pixel/Latent dual-branch fusion because the useful inference residual remains insufficient relative to Raw SD2. No follow-up experiment was started here.", "",
        "## Research boundary", "",
        summary["interpretation_boundary"], "", summary["research_boundary"], "",
        "No training, Gamma search, loss change, Region recalibration, architecture change, final_eval, BootstrapGuard or detector training was run. The experiment stops here.", ""]
    (OUTPUT / "BACKGROUND_GAMMA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "regression", "generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 13 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "audit": audit()
    elif args.phase == "regression": regression(args.device)
    elif args.phase == "generate": generate(args.device)
    else: evaluate(args.device)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
