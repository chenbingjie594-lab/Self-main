"""Experiment 14: frozen MSDF Pixel/Latent inference branch ablation.

This runner has no training entry point.  It keeps one SHA-bound
BackgroundOnly checkpoint and gamma=1, reuses Raw/Gamma0/Full/Flash evidence,
and generates only PixelOnly and LatentOnly Black samples for Banks A/B/C.
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
import run_plastic_bomo_msdf_background_gamma_ablation as x13

BASE_COMMIT = "52653e4c28c2f92d5c6ba1f697df846a78425eec"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_branch_ablation"
BG = REPO / "results_for_gpt/plastic_bomo_msdf_black_gradient_background"
X12 = REPO / "results_for_gpt/plastic_bomo_msdf_multibank_stability"
X13 = REPO / "results_for_gpt/plastic_bomo_msdf_background_gamma_ablation"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
RAW = x12.RAW
STAGE25 = x12.STAGE25
P0 = x12.P0
BANK_SEEDS = {"A": 42, "B": 3407, "C": 2026}
EXPECTED_TREE = "a194955681b58621799796b7a5812ed5402667912b5b4a6034025ef27a33d826"
EXPECTED_MSDF = "6e0c39ab6870e0a411a8fde8463ca4df90fb8fbf6803eb2493dd363eb50122fc"
EXPECTED_UNET = "559a6caaa45ab298f93a8ec105894389d27c4574a11359aae5221c3f593df6a1"
WEIGHTS = {"region": 0.0, "reconstruction": 0.0, "structure": 0.0,
           "background": 0.2, "scale": 0.0, "support": 0.0}
GROUPS = {"raw": "Raw SD2", "gamma0": "BackgroundOnly Gamma=0",
          "full": "Full Dual-Branch", "pixel_only": "PixelOnly",
          "latent_only": "LatentOnly"}
NEW_GROUPS = ("pixel_only", "latent_only")
ABLATIONS = {
    "full": {"pixel_branch": True, "latent_branch": True, "morphology_alignment": True},
    "pixel_only": {"pixel_branch": True, "latent_branch": False, "morphology_alignment": True},
    "latent_only": {"pixel_branch": False, "latent_branch": True, "morphology_alignment": True},
}

load, save, sha, canonical, require = x12.load, x12.save, x12.sha, x12.canonical, x12.require
crop_box = x12.crop_box


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows):
    require(rows, "cannot write empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def update_status(**values):
    path = OUTPUT / "branch_experiment_status.json"
    row = load(path); row.update(values); save(path, row)


def source_hashes():
    names = ["tools/run_plastic_bomo_msdf_branch_ablation.py",
             "tools/run_plastic_bomo_msdf_background_gamma_ablation.py",
             "tools/run_plastic_bomo_msdf_multibank_stability.py",
             "tools/generate_plastic_bomo_stage20b_banks.py",
             "diffusers/pipelines/stable_diffusion/msdf_guidance.py"]
    return {name: sha(REPO / name) for name in names}


def checkpoint(full_hash=False):
    binding = load(BG / "background_only_checkpoint_binding.json")
    require(binding["status"] == "FROZEN" and binding["class"] == "black" and
            binding["successful_updates"] == 2000, "BackgroundOnly checkpoint is not frozen")
    path = BG / binding["checkpoint"]["path"]
    expected = binding["checkpoint"]["sha256"]
    require(all((path / name).is_file() for name in expected), "checkpoint file missing")
    require(hashlib.sha256(canonical(expected).encode()).hexdigest() == EXPECTED_TREE,
            "checkpoint tree binding changed")
    if full_hash:
        for name, digest in expected.items():
            require(sha(path / name) == digest, f"checkpoint file changed: {name}")
    else:
        audit = load(OUTPUT / "branch_checkpoint_audit.json")
        require(audit["status"] == "PASS" and audit["checkpoint_tree_SHA256"] == EXPECTED_TREE and
                audit["msdf_checkpoint_SHA256"] == EXPECTED_MSDF and
                audit["unet_SHA256"] == EXPECTED_UNET, "checkpoint audit binding changed")
        require(audit["checkpoint_file_sizes"] ==
                {name: (path / name).stat().st_size for name in expected},
                "checkpoint file size changed after SHA audit")
    require(binding["checkpoint"]["tree_sha256"] == EXPECTED_TREE and
            expected["msdf.pt"] == EXPECTED_MSDF and
            expected["unet/diffusion_pytorch_model.safetensors"] == EXPECTED_UNET,
            "checkpoint identity changed")
    require(binding["auxiliary_loss_weights"] == WEIGHTS and
            binding["morphology_alignment_enabled"] is True and binding["injection_gamma"] == 1.0,
            "frozen BackgroundOnly controls changed")
    return path, binding


def full_rows(bank):
    if bank == "A":
        root = BG; value = load(BG / "background_only_generation_manifest.json")
        rows = value["records"]
    else:
        root = X12; value = load(X12 / "multibank_generation_manifest.json")
        rows = [r for r in value["records"] if r["group"] == "background_only" and r["bank"] == bank]
    require(value["status"] == "COMPLETE" and len(rows) == 40, f"Full Bank {bank} incomplete")
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
    gamma0_value = load(X13 / "background_gamma_generation_manifest.json")
    require(gamma0_value["status"] == "COMPLETE" and len(gamma0_value["records"]) == 120,
            "Experiment 13 Gamma=0 manifest incomplete")
    gamma0 = {(r["bank"], r["slot_id"]): r for r in gamma0_value["records"]}
    counts = {}
    for bank, bank_seed in BANK_SEEDS.items():
        root, full = full_rows(bank); full_by = {r["slot_id"]: r for r in full}
        raw = x12.manifest_records(RAW, f"formal_bank_{bank}_manifest.json")
        require(len(raw) == 80, f"Raw Bank {bank} incomplete"); x12.verify_images(RAW, raw)
        if bank == "A":
            flash = [r for r in x12.manifest_records(GAMMA, "gamma_0p00_manifest.json") if r["class"] == "flash"]
            x12.verify_images(GAMMA, flash)
        else:
            value = load(X12 / "multibank_generation_manifest.json")
            flash = [r for r in value["records"] if r["group"] == "flash_gamma0" and r["bank"] == bank]
            require(len(flash) == 40, f"Flash Bank {bank} incomplete")
            for row in flash:
                require(sha(X12 / row["image_path"]) == row["image_SHA256"], "Flash image changed")
        for slot in black:
            sid = slot["slot_id"]; frozen = originals[sid]; rng = rngs[sid][bank]
            frow = full_by[sid]; grow = gamma0[(bank, sid)]
            require(frow["frozen_slot"] == frozen and frow["runtime_slot"] == slot,
                    f"Full Bank {bank} slot metadata changed")
            require(grow["frozen_slot"] == frozen and grow["runtime_slot"] == slot,
                    f"Gamma0 Bank {bank} slot metadata changed")
            for row in (frow, grow):
                require(row["sample_seed"] == rng["sample_seed"] and row["bank_seed"] == bank_seed,
                        f"Bank {bank} RNG changed")
                require(row["checkpoint_tree_SHA256"] == EXPECTED_TREE,
                        f"Bank {bank} checkpoint tree changed")
                if "msdf_checkpoint_SHA256" in row:
                    require(row["msdf_checkpoint_SHA256"] == EXPECTED_MSDF,
                            f"Bank {bank} MSDF checkpoint changed")
            require(sha(root / frow["image_path"]) == image_digest(frow), "Full image changed")
            require(sha(X13 / grow["image_path"]) == image_digest(grow), "Gamma0 image changed")
        counts[bank] = {"raw": 80, "gamma0_black": 40, "full_black": 40, "shared_flash": 40}
    return counts


def expected_state(adapter, group):
    value = dict(ABLATIONS[group])
    value.update(up_block_mask=[True] * adapter.num_scales, residual_scale=1.0)
    return value


def cpu_branch_preflight():
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter
    torch.manual_seed(20261024)
    adapter = MSDFAdapter(4, 8, [4]).eval()
    parameters = {name: value.detach().clone() for name, value in adapter.named_parameters()}
    latents = torch.randn(1, 4, 8, 8)
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    mask = torch.zeros(1, 1, 64, 64); mask[:, :, 18:44, 20:42] = 1
    prior = torch.zeros(1, 1, 64, 64); prior[:, :, 16:48, 16:48] = 1
    rows = []
    for group in ("full", "pixel_only", "latent_only"):
        adapter.set_ablation(ABLATIONS[group]); adapter.set_injection_gamma(1.0)
        captured = {}
        def pre(_module, inputs): captured["fusion_input"] = inputs[0].detach()
        def post(_module, _inputs, output): captured["fusion_output"] = output.detach()
        handles = [adapter.fusion.register_forward_pre_hook(pre), adapter.fusion.register_forward_hook(post)]
        with torch.inference_mode():
            adapter.prepare(latents, mask, prior, torch.tensor([500]), 1000, False, pixels)
        for handle in handles: handle.remove()
        joined = captured["fusion_input"]; latent, pixel = joined.chunk(2, dim=1)
        row = {"group": group, "ablation": dict(adapter.ablation), "gamma": adapter.injection_gamma,
               "adapter_training": adapter.training,
               "latent_input_l2": float(torch.linalg.vector_norm(latent)),
               "pixel_input_l2": float(torch.linalg.vector_norm(pixel)),
               "fusion_output_l2": float(torch.linalg.vector_norm(captured["fusion_output"])),
               "pixel_support_l2": float(torch.linalg.vector_norm(adapter.last_pixel_support)),
               "all_finite": bool(all(torch.isfinite(v).all() for v in
                   (joined, captured["fusion_output"], adapter.last_pixel_support,
                    adapter.last_reference_support, adapter.last_aligned_support, adapter.last_gates)))}
        require(row["ablation"] == expected_state(adapter, group) and row["gamma"] == 1.0 and
                not row["adapter_training"] and row["all_finite"], f"CPU state failed: {group}")
        if group == "full":
            require(row["latent_input_l2"] > 0 and row["pixel_input_l2"] > 0, "Full inputs invalid")
        elif group == "pixel_only":
            require(row["latent_input_l2"] == 0 and row["pixel_input_l2"] > 0, "PixelOnly input invalid")
        else:
            require(row["latent_input_l2"] > 0 and row["pixel_input_l2"] == 0 and
                    row["pixel_support_l2"] == 0, "LatentOnly input/support invalid")
        rows.append(row)
    unchanged = all(torch.equal(parameters[name], value) for name, value in adapter.named_parameters())
    require(unchanged, "CPU preflight changed parameters")
    return {"status": "PASS", "groups": rows, "parameters_unchanged": True,
            "branch_dropout_inactive_in_eval": True,
            "fusion_output_not_required_to_be_zero": True,
            "semantic_boundary": "LatentOnly includes Pixel Support and downstream support/gate coupling changes."}


def audit():
    if OUTPUT.exists():
        status_path = OUTPUT / "branch_experiment_status.json"
        require(status_path.is_file() and load(status_path).get("status") == "AUDITING",
                f"output already exists in a non-recoverable state: {OUTPUT}")
        allowed_recovery_files = {status_path, OUTPUT / "logs/audit.log"}
        existing = [p for p in OUTPUT.rglob("*") if p.is_file() and p not in allowed_recovery_files]
        require(not existing, f"interrupted audit contains unexpected files: {existing}")
    for group in NEW_GROUPS:
        for bank in "ABC": (OUTPUT / "images" / group / bank).mkdir(parents=True, exist_ok=True)
    for group in ("full",) + NEW_GROUPS:
        for bank in "ABC": (OUTPUT / "regression" / group / bank).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    save(OUTPUT / "branch_experiment_status.json", {
        "status": "AUDITING", "created_at": now(), "base_commit": BASE_COMMIT,
        "training_count": 0, "parameter_update_count": 0, "generation_count": 0,
        "reroll_count": 0, "selection_count": 0, "final_eval_read_or_used": False,
        "BootstrapGuard": False, "detector_training_started": False})
    path, binding = checkpoint(full_hash=True)
    history = verify_history()
    generation = load(P0 / "stage20b_generation_protocol.json")
    require(generation["scheduler"] == "DDIMScheduler" and generation["num_inference_steps"] == 50 and
            generation["guidance_scale"] == 7.5 and generation["eta"] == 0 and generation["strength"] == 1 and
            generation["height"] == generation["width"] == 512 and generation["precision"].startswith("fp16"),
            "generation protocol changed")
    cpu = cpu_branch_preflight()
    cache = os.environ.get("EXP14_CHECKPOINT_CACHE")
    cache_status = "NOT_REQUESTED"
    if cache:
        cached = Path(cache); expected = binding["checkpoint"]["sha256"]
        require(all((cached / name).is_file() and sha(cached / name) == digest
                    for name, digest in expected.items()), "staged checkpoint SHA check failed")
        cache_status = "PASS"
    save(OUTPUT / "branch_checkpoint_audit.json", {
        "status": "PASS", "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "unet_SHA256": EXPECTED_UNET,
        "successful_updates": binding["successful_updates"], "training_weights": WEIGHTS,
        "checkpoint_file_count": len(binding["checkpoint"]["sha256"]),
        "checkpoint_file_sizes": {name: (path / name).stat().st_size
                                  for name in binding["checkpoint"]["sha256"]},
        "actual_server_file_SHA256_recomputed": True,
        "staged_checkpoint_SHA256_verification": cache_status,
        "history_SHA_checks": history, "source_code_SHA256": source_hashes(), "completed_at": now()})
    save(OUTPUT / "branch_preflight.json", {"status": "CPU_PASS_GPU_PENDING", "cpu": cpu,
        "required_configs": ABLATIONS, "gamma": 1.0, "up_block_mask": "all true",
        "residual_scale": 1.0, "training_forbidden": True})
    save(OUTPUT / "branch_ablation_protocol.json", {
        "status": "FROZEN_BEFORE_GPU_PREFLIGHT",
        "experiment": "Experiment 14 — MSDF Pixel/Latent Dual-Branch Ablation",
        "base_commit": BASE_COMMIT, "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "unet_SHA256": EXPECTED_UNET,
        "groups": GROUPS, "new_groups": list(NEW_GROUPS), "ablation_configs": ABLATIONS,
        "gamma": 1.0, "banks": BANK_SEEDS,
        "only_control_variable": "inference branch state using existing set_ablation controls",
        "asymmetry": "LatentOnly zeros pixel_encoded and pixel_support, thereby coupling morphology/support/gate behavior; it is not an isolated encoder-feature effect.",
        "generation_protocol": generation,
        "new_generation": {"pixel_only": 120, "latent_only": 120, "total": 240},
        "evaluation": {"references": {"flash": 75, "black": 97, "overall": 172},
            "ROI": "bbox plus 25 percent context; frozen floor/ceil clipping",
            "Inception": "torch-fidelity inception-v3-compat 2048D; float32 extraction; float64 CPU metrics",
            "KID": {"subsets": 100, "seed": 42, "degree": 3,
                    "subset_size": "min(50,nreal,nfake)"}},
        "training_forbidden": True, "final_eval_read_or_used": False})
    save(OUTPUT / "branch_generation_manifest.json", {
        "status": "PENDING", "planned_count": 240, "records": [],
        "reroll_count": 0, "selection_count": 0})
    update_status(status="AUDIT_PASS", audit_completed_at=now(), CPU_preflight="PASS")
    print("[审计通过] checkpoint、历史五组绑定、冻结协议及CPU分支测试通过。", flush=True)


def load_pipeline(device, group):
    """Load a fresh, identically weighted model for one branch state."""
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter

    require(group in ABLATIONS, f"unknown ablation group: {group}")
    checkpoint_path, binding = checkpoint()
    cache = os.environ.get("EXP14_CHECKPOINT_CACHE")
    if cache:
        checkpoint_path = Path(cache)
        expected = binding["checkpoint"]["sha256"]
        audited = load(OUTPUT / "branch_checkpoint_audit.json")
        require(audited["staged_checkpoint_SHA256_verification"] == "PASS", "checkpoint cache was not SHA audited")
        require({name: (checkpoint_path / name).stat().st_size for name in expected} ==
                audited["checkpoint_file_sizes"], "checkpoint cache differs after audit")
        print(f"[加载] 使用已SHA审计的NVMe checkpoint：{checkpoint_path}", flush=True)
    generation = load(P0 / "stage20b_generation_protocol.json")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(checkpoint_path), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float16, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False).to(device)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    pipe.disable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(generation["scheduler_config"])
    adapter = load_msdf_adapter(checkpoint_path / "msdf.pt", pipe.unet, device).eval()
    adapter.set_ablation(ABLATIONS[group])
    adapter.set_injection_gamma(1.0)
    require(adapter.ablation == expected_state(adapter, group), f"runtime ablation mismatch: {group}")
    require(adapter.injection_gamma == 1.0 and not adapter.training, f"runtime state mismatch: {group}")
    return pipe, adapter, generation


def adapter_state_digest(adapter):
    digest = hashlib.sha256()
    for name, value in sorted(adapter.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate_one(pipe, adapter, slot, rng, generation, device, prepared_callback=None):
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
        if prepared_callback is not None:
            prepared_callback(adapter)
    handle = pipe.unet.register_forward_pre_hook(prepare, with_kwargs=True)
    try:
        return sample(pipe, slot, rng, generation, device)
    finally:
        handle.remove()


def diagnostic_hooks(adapter, pipe, sink):
    """Read-only hooks for fusion inputs/output and final injected residuals."""
    import torch
    handles = []; before = {}

    def fusion_pre(_module, inputs):
        joined = inputs[0].detach().float(); latent, pixel = joined.chunk(2, dim=1)
        sink["fusion"].append({"latent_encoder_l2": float(torch.linalg.vector_norm(latent)),
            "pixel_encoder_l2": float(torch.linalg.vector_norm(pixel)),
            "input_finite": bool(torch.isfinite(joined).all())})

    def fusion_post(_module, _inputs, output):
        value = output.detach().float()
        sink["fusion"][-1].update(fusion_output_l2=float(torch.linalg.vector_norm(value)),
                                   output_finite=bool(torch.isfinite(value).all()))
    handles += [adapter.fusion.register_forward_pre_hook(fusion_pre),
                adapter.fusion.register_forward_hook(fusion_post)]
    for index, block in enumerate(pipe.unet.up_blocks):
        def capture_before(_module, _inputs, output, block_index=index):
            before[block_index] = (output[0] if isinstance(output, tuple) else output).detach()
        def capture_after(_module, _inputs, output, block_index=index):
            updated = (output[0] if isinstance(output, tuple) else output).detach().float()
            hidden = before.pop(block_index).float(); residual = updated - hidden
            hidden_l2 = float(torch.linalg.vector_norm(hidden)); residual_l2 = float(torch.linalg.vector_norm(residual))
            sink[f"up_block_{block_index}"].append({"hidden_l2": hidden_l2,
                "residual_l2": residual_l2, "relative_l2": residual_l2 / max(hidden_l2, 1e-12),
                "finite": bool(torch.isfinite(updated).all() and torch.isfinite(residual).all())})
        handles += [block.register_forward_hook(capture_before, prepend=True),
                    block.register_forward_hook(capture_after)]
    return handles


def support_callback(sink):
    def callback(adapter):
        import torch
        pixel = adapter.last_pixel_support.detach().float()
        reference = adapter.last_reference_support.detach().float()
        fused = (2 * reference - pixel).clamp(0, 1)
        aligned = adapter.last_aligned_support.detach().float()
        gates = adapter.last_gates.detach().float()
        sink["support"].append({
            "pixel_support_mean": float(pixel.mean()), "pixel_support_area_gt_0p05": float((pixel > .05).float().mean()),
            "fused_support_mean": float(fused.mean()), "fused_support_area_gt_0p05": float((fused > .05).float().mean()),
            "aligned_support_mean": float(aligned.mean()), "aligned_support_area_gt_0p05": float((aligned > .05).float().mean()),
            "dynamic_gate_min": float(gates.min()), "dynamic_gate_mean": float(gates.mean()),
            "dynamic_gate_max": float(gates.max()),
            "finite": bool(all(torch.isfinite(v).all() for v in (pixel, fused, aligned, gates)))})
    return callback


def summarize_diagnostics(group, bank, slot_id, sink):
    row = {"group": group, "bank": bank, "slot_id": slot_id, "diffusion_observations": len(sink["fusion"])}
    for key in ("latent_encoder_l2", "pixel_encoder_l2", "fusion_output_l2"):
        values = [v[key] for v in sink["fusion"]]
        row[key + "_min"] = min(values); row[key + "_max"] = max(values); row[key + "_mean"] = sum(values) / len(values)
    for key in ("pixel_support_mean", "pixel_support_area_gt_0p05", "fused_support_mean",
                "fused_support_area_gt_0p05", "aligned_support_mean", "aligned_support_area_gt_0p05",
                "dynamic_gate_min", "dynamic_gate_mean", "dynamic_gate_max"):
        values = [v[key] for v in sink["support"]]
        row[key + "_min"] = min(values); row[key + "_max"] = max(values); row[key + "_mean"] = sum(values) / len(values)
    row["up_blocks"] = {}
    for index in range(4):
        values = sink[f"up_block_{index}"]
        row["up_blocks"][str(index)] = {
            "residual_l2_min": min(v["residual_l2"] for v in values),
            "residual_l2_max": max(v["residual_l2"] for v in values),
            "relative_l2_min": min(v["relative_l2"] for v in values),
            "relative_l2_max": max(v["relative_l2"] for v in values),
            "all_finite": all(v["finite"] for v in values)}
    row["all_finite"] = (all(v["input_finite"] and v["output_finite"] for v in sink["fusion"]) and
                          all(v["finite"] for v in sink["support"]) and
                          all(v["all_finite"] for v in row["up_blocks"].values()))
    return row


def regression(device):
    require(load(OUTPUT / "branch_experiment_status.json")["status"] == "AUDIT_PASS", "regression requires audit PASS")
    claim = OUTPUT / "logs/regression_claim.json"
    require(not claim.exists(), "regression already claimed")
    save(claim, {"started_at": now(), "groups": ["full", "pixel_only", "latent_only"], "samples_per_group": 6})
    update_status(status="REGRESSION_RUNNING", regression_started_at=now())
    records = []; diagnostics = []; state_digests = {}
    with x12.gpu_runtime(device) as torch:
        slots = [r for r in load(P0 / "stage20b_generation_slots.json")["slots"] if r["class"] == "black"]
        rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
        for group in ("full", "pixel_only", "latent_only"):
            pipe, adapter, generation = load_pipeline(device, group)
            before = adapter_state_digest(adapter); state_digests[group] = before
            require(any(bool(torch.count_nonzero(p)) for p in adapter.gate.parameters()), "dynamic gate weights are zero")
            for bank in "ABC":
                historical = None
                if group == "full":
                    root, rows = full_rows(bank); historical = {r["slot_id"]: r for r in rows}
                for diagnostic_index, slot in enumerate(slots[:2]):
                    sink = defaultdict(list); handles = diagnostic_hooks(adapter, pipe, sink)
                    try:
                        image, trace = generate_one(pipe, adapter, slot, rngs[slot["slot_id"]][bank], generation,
                                                    device, support_callback(sink))
                    finally:
                        for handle in handles: handle.remove()
                    path = OUTPUT / "regression" / group / bank / f"{slot['slot_id']}_{diagnostic_index}.png"
                    image.save(path, format="PNG")
                    exact = None
                    if group == "full" and diagnostic_index == 0:
                        old = historical[slot["slot_id"]]
                        exact = sha(path) == sha(root / old["image_path"]) == image_digest(old)
                    records.append({"group": group, "bank": bank, "diagnostic_index": diagnostic_index,
                        "slot_id": slot["slot_id"], "sample_seed": rngs[slot["slot_id"]][bank]["sample_seed"],
                        "image_SHA256": sha(path), "full_exact_historical_PNG": exact, **trace})
                    summary = summarize_diagnostics(group, bank, slot["slot_id"], sink)
                    require(summary["all_finite"], f"non-finite diagnostic: {group}/{bank}/{slot['slot_id']}")
                    diagnostics.append(summary)
                    print(f"[GPU测试] {group} Bank {bank} {diagnostic_index + 1}/2 exact={exact}", flush=True)
            require(before == adapter_state_digest(adapter), f"adapter parameters changed: {group}")
            adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
    require(len(set(state_digests.values())) == 1, "groups did not use identical adapter weights")
    require(all(r["full_exact_historical_PNG"] for r in records
                if r["group"] == "full" and r["diagnostic_index"] == 0), "Full PNG regression failed")
    for row in diagnostics:
        if row["group"] == "full":
            require(row["latent_encoder_l2_max"] > 0 and row["pixel_encoder_l2_max"] > 0, "Full fusion input invalid")
        elif row["group"] == "pixel_only":
            require(row["latent_encoder_l2_max"] == 0 and row["pixel_encoder_l2_max"] > 0, "PixelOnly fusion input invalid")
        else:
            require(row["latent_encoder_l2_max"] > 0 and row["pixel_encoder_l2_max"] == 0 and
                    row["pixel_support_mean_max"] == 0, "LatentOnly fusion/support input invalid")
    save(OUTPUT / "branch_full_regression.json", {"status": "PASS", "records": records,
        "full_exact_PNG_banks": 3, "same_adapter_state_digest_across_groups": True,
        "adapter_state_SHA256": next(iter(state_digests.values())), "parameters_unchanged": True,
        "diagnostic_images_excluded_from_formal_pool": True, "completed_at": now()})
    save(OUTPUT / "branch_feature_diagnostics.json", {"status": "PASS", "records": diagnostics,
        "slots_per_bank_per_group": 2, "read_only_hooks": True,
        "interpretation": "LatentOnly includes the coupled removal of pixel support and its downstream morphology/support/gate effects."})
    preflight = load(OUTPUT / "branch_preflight.json")
    preflight.update(status="PASS", GPU="PASS", full_exact_regression="PASS",
                     same_checkpoint_weights="PASS", no_parameter_updates=True, completed_at=now())
    save(OUTPUT / "branch_preflight.json", preflight)
    update_status(status="REGRESSION_PASS", regression_completed_at=now(), regression_samples=18)
    print("[回归通过] Full A/B/C各1张PNG完全一致；三组18张GPU只读诊断通过。", flush=True)


def generate(device):
    require(load(OUTPUT / "branch_experiment_status.json")["status"] == "REGRESSION_PASS", "generation requires regression PASS")
    claim = OUTPUT / "logs/generate_claim.json"; require(not claim.exists(), "generation already claimed")
    save(claim, {"started_at": now(), "planned_count": 240, "groups": list(NEW_GROUPS)})
    update_status(status="GENERATING", generation_started_at=now())
    manifest_path = OUTPUT / "branch_generation_manifest.json"; records = []; started = time.monotonic()
    try:
        with x12.gpu_runtime(device) as torch:
            from stage20b_formal_common import synthetic_label
            slots = [r for r in load(P0 / "stage20b_generation_slots.json")["slots"] if r["class"] == "black"]
            originals = {r["slot_id"]: r for r in load(
                Path(load(STAGE25 / "stage25_run_binding.json")["p0"]) / "stage20b_generation_slots.json")["slots"]}
            rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
            for group in NEW_GROUPS:
                pipe, adapter, generation = load_pipeline(device, group); before = adapter_state_digest(adapter)
                for bank in "ABC":
                    for index, slot in enumerate(slots):
                        frozen = originals[slot["slot_id"]]; rng = rngs[slot["slot_id"]][bank]
                        require(rng["bank_seed"] == BANK_SEEDS[bank] and
                                rng["non_rng_metadata_sha256"] == hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                                f"frozen RNG mismatch: {group}/{bank}/{slot['slot_id']}")
                        item_started = time.monotonic()
                        image, trace = generate_one(pipe, adapter, slot, rng, generation, device)
                        image_path = OUTPUT / "images" / group / bank / (slot["slot_id"] + ".png")
                        label_path = image_path.with_suffix(".txt")
                        require(not image_path.exists() and not label_path.exists(), "formal output would overwrite a file")
                        image.save(image_path, format="PNG")
                        label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                        width, height = frozen["geometry"]["final_canvas_wh"]
                        records.append({"group": group, "method": GROUPS[group], "bank": bank, "class": "black",
                            "slot_id": slot["slot_id"], "frozen_slot": frozen, "runtime_slot": slot,
                            "normal_SHA256": frozen["source_sha256"], "bbox": frozen["bbox"],
                            "prompt": frozen["prompt"], "sample_seed": rng["sample_seed"],
                            "bank_seed": rng["bank_seed"], "checkpoint_tree_SHA256": EXPECTED_TREE,
                            "msdf_checkpoint_SHA256": EXPECTED_MSDF, "branch_config": ABLATIONS[group],
                            "injection_gamma": 1.0, "image_path": image_path.relative_to(OUTPUT).as_posix(),
                            "image_SHA256": sha(image_path), "label_path": label_path.relative_to(OUTPUT).as_posix(),
                            "label_SHA256": sha(label_path), "roi_xyxy": crop_box(frozen["bbox"], width, height),
                            "quality_selection": False, "reroll": False,
                            "runtime_seconds": time.monotonic() - item_started, **trace})
                        save(manifest_path, {"status": "RUNNING", "planned_count": 240, "records": records,
                            "reroll_count": 0, "selection_count": 0})
                        print(f"[生成] {group} Bank {bank} {index + 1}/40 total={len(records)}/240", flush=True)
                require(before == adapter_state_digest(adapter), f"parameters changed during generation: {group}")
                adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        counts = Counter((r["group"], r["bank"]) for r in records)
        require(len(records) == 240 and all(counts[(g, b)] == 40 for g in NEW_GROUPS for b in "ABC"),
                "branch generation incomplete")
        save(manifest_path, {"status": "COMPLETE", "count": 240, "records": records,
            "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
            "parameter_update_count": 0, "runtime_seconds": time.monotonic() - started})
        update_status(status="GENERATION_COMPLETE", generation_count=240, generation_completed_at=now(),
                      reroll_count=0, selection_count=0, parameter_update_count=0)
        print("[生成完成] PixelOnly/LatentOnly A/B/C 共240/240；无重抽、筛选或替换。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "records": records,
            "error": f"{type(error).__name__}: {error}", "reroll_count": 0, "selection_count": 0})
        update_status(status="GENERATION_FAILED", failure=f"{type(error).__name__}: {error}")
        raise


def extract_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    cache_path = OUTPUT / "branch_inception_features.npz"
    ledger_path = OUTPUT / "branch_inception_feature_cache.json"
    expected_keys = [f"{row['group']}:{row['bank']}:{row['slot_id']}" for row in manifest["records"]]
    if cache_path.is_file() and ledger_path.is_file():
        ledger = load(ledger_path)
        require(ledger["status"] == "COMPLETE" and ledger["keys"] == expected_keys and
                ledger["sha256"] == sha(cache_path) and ledger["image_SHA256_checks"] == 240,
                "existing branch feature cache binding changed")
        print("[恢复] 复用已完成且SHA绑定的240张Inception特征，不重复提取。", flush=True)
        return x12.array_features(cache_path, ledger_path, torch)
    model = lib["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
    values = []; keys = []
    with torch.inference_mode():
        for index, row in enumerate(manifest["records"]):
            path = OUTPUT / row["image_path"]
            require(path.is_file() and sha(path) == row["image_SHA256"], f"generated image changed: {path}")
            with Image.open(path) as source:
                crop = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
            tensor = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = model(tensor)[0]
            require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()), row["slot_id"])
            keys.append(f"{row['group']}:{row['bank']}:{row['slot_id']}")
            values.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0: print(f"[Inception] {index + 1}/240", flush=True)
    data = np.stack(values); path = OUTPUT / "branch_inception_features.npz"
    np.savez_compressed(path, features=data)
    save(OUTPUT / "branch_inception_feature_cache.json", {"status": "COMPLETE", "keys": keys,
        "shape": [240, 2048], "stored_dtype": str(data.dtype), "extraction_dtype": "float32",
        "sha256": sha(path), "image_SHA256_checks": 240})
    del model; torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[i]).double() for i, key in enumerate(keys)}


def historical_metrics():
    def convert(row, scope_key):
        group = "full" if row["group"] == "gamma1" else row["group"]
        return {scope_key: row[scope_key], "group": group, "method": GROUPS[group],
            "KID": float(row["KID"]), "KID_subset_std": float(row["KID_subset_std"]),
            "FID": float(row["FID"]), "reference_ROIs": int(float(row["reference_ROIs"])),
            "generated_images": int(float(row["generated_images"])),
            "KID_subset_size": int(float(row["KID_subset_size"])),
            "runtime_seconds": float(row["runtime_seconds"]),
            "metric_source": "reused SHA-bound Experiment 13 same-protocol features/metrics"}
    black = [convert(r, "bank") for r in read_csv(X13 / "background_gamma_bankwise_black.csv")]
    overall = [convert(r, "bank") for r in read_csv(X13 / "background_gamma_bankwise_overall.csv")]
    pooled = []
    for row in read_csv(X13 / "background_gamma_pooled_metrics.csv"):
        value = convert(row, "scope"); value["population"] = row["population"]; pooled.append(value)
    require(len(black) == len(overall) == 9 and len(pooled) == 12, "Experiment 13 metric population changed")
    return black, overall, pooled


def evaluate(device):
    require(load(OUTPUT / "branch_experiment_status.json")["status"] in ("GENERATION_COMPLETE", "EVALUATING"),
            "evaluation requires complete generation or a recoverable post-feature state")
    update_status(status="EVALUATING", evaluation_started_at=now()); started = time.monotonic()
    try:
        with x12.gpu_runtime(device) as torch:
            from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
            lib, runtime_audit = metric_runtime(device)
            manifest = load(OUTPUT / "branch_generation_manifest.json")
            require(manifest["status"] == "COMPLETE" and manifest["count"] == 240, "manifest incomplete")
            new = extract_features(device, torch, lib, manifest)
            gamma = x12.array_features(GAMMA / "gamma_sweep_inception_features.npz",
                                       GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
            multibank = x12.array_features(X12 / "multibank_inception_features.npz",
                                           X12 / "multibank_inception_feature_cache.json", torch)
            references = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
            real_black = torch.stack([gamma["real:" + r["annotation_id"]] for r in references if r["class"] == "black"])
            real_all = torch.stack([gamma["real:" + r["annotation_id"]] for r in references])
            slots = load(P0 / "stage20b_generation_slots.json")["slots"]
            black_slots = [r for r in slots if r["class"] == "black"]
            flash_slots = [r for r in slots if r["class"] == "flash"]

            def black_features(group, bank):
                return torch.stack([new[f"{group}:{bank}:{r['slot_id']}"] for r in black_slots])

            def shared_flash(bank):
                if bank == "A": return torch.stack([gamma[f"fake:0.00:{r['slot_id']}"] for r in flash_slots])
                return torch.stack([multibank[f"flash_gamma0:{bank}:{r['slot_id']}"] for r in flash_slots])

            old_black, old_overall, old_pooled = historical_metrics()
            black_rows = []; overall_rows = []
            for bank in "ABC":
                black_rows.extend([r for r in old_black if r["bank"] == bank])
                overall_rows.extend([r for r in old_overall if r["bank"] == bank])
                for group in NEW_GROUPS:
                    fake_black = black_features(group, bank)
                    black_metric = x12.metric(real_black, fake_black, lib)
                    overall_metric = x12.metric(real_all, torch.cat([shared_flash(bank), fake_black]), lib)
                    black_rows.append({"bank": bank, "group": group, "method": GROUPS[group], **black_metric,
                        "metric_source": "new same-protocol branch features"})
                    overall_rows.append({"bank": bank, "group": group, "method": GROUPS[group], **overall_metric,
                        "metric_source": "new same-protocol branch plus shared Flash features"})
                    print(f"[指标] {bank} {group} Black KID={black_metric['KID']:.10f} FID={black_metric['FID']:.6f}; "
                          f"Overall KID={overall_metric['KID']:.10f} FID={overall_metric['FID']:.6f}", flush=True)
            pooled_rows = []
            for scope, banks in (("A+B+C", "ABC"), ("B+C", "BC")):
                pooled_rows.extend([r for r in old_pooled if r["scope"] == scope])
                for population, real in (("Black", real_black), ("Overall", real_all)):
                    for group in NEW_GROUPS:
                        if population == "Black":
                            fake = torch.cat([black_features(group, bank) for bank in banks])
                        else:
                            fake = torch.cat([torch.cat([shared_flash(bank), black_features(group, bank)]) for bank in banks])
                        result = x12.metric(real, fake, lib)
                        pooled_rows.append({"scope": scope, "population": population, "group": group,
                            "method": GROUPS[group], **result, "metric_source": "new concatenated features"})
                        print(f"[Pooled] {scope} {population} {group} KID={result['KID']:.10f} FID={result['FID']:.6f}", flush=True)

        group_order = {name: i for i, name in enumerate(("raw", "gamma0", "full", "pixel_only", "latent_only"))}
        black_rows.sort(key=lambda r: (r["bank"], group_order[r["group"]]))
        overall_rows.sort(key=lambda r: (r["bank"], group_order[r["group"]]))
        pooled_rows.sort(key=lambda r: (("A+B+C", "B+C").index(r["scope"]),
                                        ("Black", "Overall").index(r["population"]), group_order[r["group"]]))
        write_csv(OUTPUT / "branch_bankwise_black.csv", black_rows)
        write_csv(OUTPUT / "branch_bankwise_overall.csv", overall_rows)
        write_csv(OUTPUT / "branch_pooled_metrics.csv", pooled_rows)

        comparisons = (("full", "pixel_only"), ("full", "latent_only"), ("pixel_only", "latent_only"),
                       ("full", "gamma0"), ("pixel_only", "gamma0"), ("latent_only", "gamma0"),
                       ("full", "raw"), ("pixel_only", "raw"), ("latent_only", "raw"))
        deltas = []
        def add(scope, population, first, second):
            for metric_name in ("KID", "FID"):
                delta = first[metric_name] - second[metric_name]
                deltas.append({"scope": scope, "population": population,
                    "comparison": first["group"] + " - " + second["group"], "metric": metric_name,
                    "first": first[metric_name], "second": second[metric_name], "absolute_delta": delta,
                    "relative_percent": 100 * delta / second[metric_name], "first_is_better": delta < 0})
        for bank in "ABC":
            for population, rows in (("Black", black_rows), ("Overall", overall_rows)):
                by = {r["group"]: r for r in rows if r["bank"] == bank}
                for first, second in comparisons: add(bank, population, by[first], by[second])
        for scope in ("A+B+C", "B+C"):
            for population in ("Black", "Overall"):
                by = {r["group"]: r for r in pooled_rows if r["scope"] == scope and r["population"] == population}
                for first, second in comparisons: add(scope, population, by[first], by[second])
        write_csv(OUTPUT / "branch_pairwise_deltas.csv", deltas)

        def by_bank(bank, population="Black"):
            rows = black_rows if population == "Black" else overall_rows
            return {r["group"]: r for r in rows if r["bank"] == bank}
        def by_pool(scope, population):
            return {r["group"]: r for r in pooled_rows if r["scope"] == scope and r["population"] == population}
        held = []
        for bank in "BC":
            by = by_bank(bank)
            for metric_name in ("KID", "FID"):
                held.extend([by["full"][metric_name] < by["pixel_only"][metric_name],
                             by["full"][metric_name] < by["latent_only"][metric_name]])
        bc = by_pool("B+C", "Black")
        pooled_wins = [bc["full"][m] < bc[g][m] for m in ("KID", "FID") for g in NEW_GROUPS]
        held_none = []
        for bank in "BC":
            by = by_bank(bank)
            for metric_name in ("KID", "FID"):
                held_none.append(by["full"][metric_name] >= by["pixel_only"][metric_name] and
                                 by["full"][metric_name] >= by["latent_only"][metric_name])
        if all(held) and all(pooled_wins): conclusion = "DUAL_BRANCH_CONSISTENT_BENEFIT"
        elif all(held_none): conclusion = "DUAL_BRANCH_NOT_SUPPORTED"
        else: conclusion = "DUAL_BRANCH_MIXED"

        abc_black = by_pool("A+B+C", "Black"); abc_overall = by_pool("A+B+C", "Overall")
        bc_overall = by_pool("B+C", "Overall")
        def best(rows, metric_name): return min(rows.values(), key=lambda r: r[metric_name])["group"]
        def comparison(scope, population, first, second):
            rows = by_pool(scope, population)
            return {m: {"absolute_delta": rows[first][m] - rows[second][m],
                        "relative_percent": 100 * (rows[first][m] - rows[second][m]) / rows[second][m],
                        "first_is_better": rows[first][m] < rows[second][m]} for m in ("KID", "FID")}
        full_pixel_bc = comparison("B+C", "Black", "full", "pixel_only")
        full_latent_bc = comparison("B+C", "Black", "full", "latent_only")
        pixel_removal_impact = abs(full_latent_bc["KID"]["absolute_delta"])
        latent_removal_impact = abs(full_pixel_bc["KID"]["absolute_delta"])
        larger = "pixel_branch_and_coupled_support" if pixel_removal_impact > latent_removal_impact else "latent_branch"
        summary = {"status": conclusion, "primary_scope": "B+C pooled Black",
            "full_vs_pixel_only_BC": full_pixel_bc, "full_vs_latent_only_BC": full_latent_bc,
            "full_vs_raw_ABC": comparison("A+B+C", "Black", "full", "raw"),
            "full_vs_raw_ABC_overall": comparison("A+B+C", "Overall", "full", "raw"),
            "B_C_bankwise_full_better_than_both_single_branches": {"all_KID_and_FID": all(held), "checks": held},
            "B_C_pooled_full_better_than_both_single_branches": {"all_KID_and_FID": all(pooled_wins), "checks": pooled_wins},
            "larger_closure_effect_by_BC_Black_KID": larger,
            "best_methods": {"ABC_Black_KID": best(abc_black, "KID"), "ABC_Black_FID": best(abc_black, "FID"),
                "BC_Black_KID": best(bc, "KID"), "BC_Black_FID": best(bc, "FID"),
                "ABC_Overall_KID": best(abc_overall, "KID"), "ABC_Overall_FID": best(abc_overall, "FID"),
                "BC_Overall_KID": best(bc_overall, "KID"), "BC_Overall_FID": best(bc_overall, "FID")},
            "interpretation_boundary": "Inference-time coupled branch ablation only. Pixel disablement also removes pixel support and changes downstream morphology/support/gating; results are not pure independent encoder contributions.",
            "research_boundary": "Banks share slots and V2 TRAIN references may overlap training; no independent-test or statistical-significance claim."}
        save(OUTPUT / "branch_stability_summary.json", summary)
        counts = Counter((r["group"], r["bank"]) for r in manifest["records"])
        require(all(counts[(g, b)] == 40 for g in NEW_GROUPS for b in "ABC"), "integrity count mismatch")
        identity = all(r["frozen_slot"] == next(x for x in manifest["records"]
            if x["group"] != r["group"] and x["bank"] == r["bank"] and x["slot_id"] == r["slot_id"])["frozen_slot"]
            for r in manifest["records"])
        require(identity, "cross-group slot metadata mismatch")
        save(OUTPUT / "branch_generation_integrity.json", {"status": "PASS", "counts": {
            g: {b: counts[(g, b)] for b in "ABC"} for g in NEW_GROUPS}, "total": 240,
            "image_SHA256_checks": 240, "cross_group_metadata_identical": True,
            "reroll_count": 0, "selection_count": 0, "parameter_update_count": 0,
            "feature_cache_SHA256": sha(OUTPUT / "branch_inception_features.npz"),
            "full_formal_images_reused": True, "raw_and_gamma0_reused": True,
            "final_eval_read_or_used": False})
        protocol = load(OUTPUT / "branch_ablation_protocol.json")
        protocol.update(status="COMPLETE", conclusion=conclusion, new_generation_count=240,
                        source_code_SHA256=source_hashes(), metric_runtime=runtime_audit)
        save(OUTPUT / "branch_ablation_protocol.json", protocol)
        create_report(conclusion, black_rows, overall_rows, pooled_rows, summary)
        update_status(status=conclusion, evaluation_completed=True, completed_at=now(),
                      new_feature_count=240, evaluation_runtime_seconds=time.monotonic() - started,
                      followup_started=False)
        print(f"[完成] Experiment 14 status={conclusion}；停止，不启动后续实验。", flush=True)
    except Exception as error:
        update_status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def create_report(conclusion, black, overall, pooled, summary):
    lines = ["# Experiment 14 — MSDF Pixel/Latent Dual-Branch Ablation", "", f"Status: `{conclusion}`.", "",
        "This is a frozen-checkpoint inference ablation. Full, PixelOnly and LatentOnly use identical weights and gamma=1. PixelOnly disables the latent encoder contribution. LatentOnly disables both the pixel encoder contribution and pixel support, which also changes downstream morphology, support and dynamic gating. The results therefore measure coupled branch-path effects, not pure independent encoder contributions.", "",
        "## Bankwise Black", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in black: lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Bankwise Overall", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in overall: lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Pooled metrics", "", "These metrics were recomputed from concatenated Inception features; no class or bank metric was averaged.", "",
        "| Scope | Population | Method | Images | KID ↓ | FID ↓ |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in pooled: lines.append(f"| {row['scope']} | {row['population']} | {row['method']} | {row['generated_images']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    fp = summary["full_vs_pixel_only_BC"]; fl = summary["full_vs_latent_only_BC"]
    best = summary["best_methods"]
    lines += ["", "## Findings", "",
        f"On B+C pooled Black, Full minus PixelOnly is {fp['KID']['absolute_delta']:+.10f} KID and {fp['FID']['absolute_delta']:+.6f} FID; Full is better when these values are negative.", "",
        f"On B+C pooled Black, Full minus LatentOnly is {fl['KID']['absolute_delta']:+.10f} KID and {fl['FID']['absolute_delta']:+.6f} FID.", "",
        f"The larger B+C Black KID closure effect came from `{summary['larger_closure_effect_by_BC_Black_KID']}`. For Pixel closure this includes the coupled pixel-support path.", "",
        f"B/C bankwise Full beat both single branches on every Black KID and FID check: `{summary['B_C_bankwise_full_better_than_both_single_branches']['all_KID_and_FID']}`. B+C pooled Full beat both on both metrics: `{summary['B_C_pooled_full_better_than_both_single_branches']['all_KID_and_FID']}`.", "",
        f"Best B+C pooled Black methods: KID `{best['BC_Black_KID']}`, FID `{best['BC_Black_FID']}`. Best A+B+C Overall methods: KID `{best['ABC_Overall_KID']}`, FID `{best['ABC_Overall_FID']}`.", "",
        f"Full versus Raw on A+B+C pooled Black: KID {summary['full_vs_raw_ABC']['KID']['absolute_delta']:+.10f} ({summary['full_vs_raw_ABC']['KID']['relative_percent']:+.2f}%), FID {summary['full_vs_raw_ABC']['FID']['absolute_delta']:+.6f} ({summary['full_vs_raw_ABC']['FID']['relative_percent']:+.2f}%).", "",
        "Whether the two branches should be retained follows the declared status above. A training-level single-branch ablation would be a separate experiment and is warranted only to test training behavior; this inference experiment cannot establish it.", "",
        "All requested pairwise absolute and relative changes are in `branch_pairwise_deltas.csv`. Feature, support, gate and up-block residual diagnostics are in `branch_feature_diagnostics.json`.", "",
        "## Research boundary", "", summary["interpretation_boundary"], "", summary["research_boundary"], "",
        "No training, architecture or fusion change, loss change, gamma search, seed search, final_eval use, BootstrapGuard or detector training occurred. The experiment stops here.", ""]
    (OUTPUT / "BRANCH_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "regression", "generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 14 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "audit": audit()
    elif args.phase == "regression": regression(args.device)
    elif args.phase == "generate": generate(args.device)
    else: evaluate(args.device)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
