"""Experiment 16: frozen MSDF pixel-feature / pixel-support factorial decoupling.

This runner never trains.  It binds the Experiment 10 BackgroundOnly
checkpoint, applies two research-only switches inside a local copy of
MSDFAdapter.prepare, strictly regresses F1S1 and F0S0 against historical PNGs,
diagnoses fixed slots, generates only F0S1/F1S0, and reuses the Experiment 14
metric protocol and historical feature evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

import run_plastic_bomo_msdf_multibank_stability as x12
import run_plastic_bomo_msdf_background_gamma_ablation as x13
import run_plastic_bomo_msdf_branch_ablation as x14

BASE_COMMIT = "9f11ebfea5902cf48281184d49d26cbeff431053"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_fusion_support_decoupling"
BG, X12, X13 = x14.BG, x14.X12, x14.X13
GAMMA, RAW, STAGE25, P0 = x14.GAMMA, x14.RAW, x14.STAGE25, x14.P0
BANK_SEEDS = dict(x14.BANK_SEEDS)
EXPECTED_TREE, EXPECTED_MSDF, EXPECTED_UNET = x14.EXPECTED_TREE, x14.EXPECTED_MSDF, x14.EXPECTED_UNET
WEIGHTS = dict(x14.WEIGHTS)
X14 = REPO / "results_for_gpt/plastic_bomo_msdf_branch_ablation"
GROUPS = {
    "raw": "Raw SD2",
    "gamma0": "BackgroundOnly Gamma=0",
    "f1s1": "F1S1 Full Dual-Branch",
    "f0s0": "F0S0 LatentOnly",
    "f0s1": "F0S1 (Pixel Feature OFF; Support ON)",
    "f1s0": "F1S0 (Pixel Feature ON; Support OFF)",
}
FACTORS = {
    "f1s1": {"pixel_feature_enabled": True, "pixel_support_enabled": True},
    "f0s0": {"pixel_feature_enabled": False, "pixel_support_enabled": False},
    "f0s1": {"pixel_feature_enabled": False, "pixel_support_enabled": True},
    "f1s0": {"pixel_feature_enabled": True, "pixel_support_enabled": False},
}
NEW_GROUPS = ("f0s1", "f1s0")
FACTOR_GROUPS = ("f1s1", "f0s0", "f0s1", "f1s0")

load, save, sha, canonical, require = x12.load, x12.save, x12.sha, x12.canonical, x12.require
crop_box = x12.crop_box


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows):
    require(rows, f"cannot write empty CSV: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def status(**values):
    path = OUTPUT / "experiment_status.json"
    row = load(path) if path.is_file() else {}
    row.update(values); save(path, row)


def checkpoint(full_hash=False):
    binding = load(BG / "background_only_checkpoint_binding.json")
    require(binding["status"] == "FROZEN" and binding["class"] == "black" and
            binding["successful_updates"] == 2000, "BackgroundOnly checkpoint is not frozen")
    path = BG / binding["checkpoint"]["path"]
    expected = binding["checkpoint"]["sha256"]
    require(hashlib.sha256(canonical(expected).encode()).hexdigest() == EXPECTED_TREE,
            "checkpoint tree binding changed")
    require(binding["checkpoint"]["tree_sha256"] == EXPECTED_TREE and
            expected["msdf.pt"] == EXPECTED_MSDF and
            expected["unet/diffusion_pytorch_model.safetensors"] == EXPECTED_UNET,
            "checkpoint identity changed")
    require(binding["auxiliary_loss_weights"] == WEIGHTS and
            binding["morphology_alignment_enabled"] is True and
            binding["injection_gamma"] == 1.0, "frozen checkpoint controls changed")
    require(all((path / name).is_file() for name in expected), "checkpoint file missing")
    if full_hash:
        for name, digest in expected.items():
            require(sha(path / name) == digest, f"checkpoint file changed: {name}")
    else:
        audit = load(OUTPUT / "factorial_checkpoint_audit.json")
        require(audit["status"] == "PASS" and audit["checkpoint_tree_SHA256"] == EXPECTED_TREE,
                "checkpoint audit changed")
        require(audit["checkpoint_file_sizes"] ==
                {name: (path / name).stat().st_size for name in expected},
                "checkpoint size changed after audit")
    return path, binding


def source_hashes():
    names = [
        "tools/run_plastic_bomo_msdf_fusion_support_decoupling.py",
        "tools/run_plastic_bomo_msdf_branch_ablation.py",
        "tools/run_plastic_bomo_msdf_multibank_stability.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py",
    ]
    return {name: sha(REPO / name) for name in names}


def factorial_prepare(self, reference_latents, reference_mask, target_prior, timesteps,
                      num_train_timesteps, classifier_free_guidance,
                      reference_pixels=None):
    """Research-only prepare with only F and S controls added to the frozen formula."""
    import torch
    import torch.nn.functional as F
    from diffusers.pipelines.stable_diffusion.msdf_guidance import _safe_rms

    controls = self.factorial_controls
    latent_size = reference_latents.shape[-2:]
    reference_mask_pixels = reference_mask.float().clamp(0, 1)
    reference_mask = F.interpolate(reference_mask_pixels, size=latent_size, mode="nearest").clamp(0, 1)
    target_prior = F.interpolate(target_prior.float(), size=latent_size,
                                 mode="bilinear", align_corners=False).clamp(0, 1)
    reference_latents = torch.nan_to_num(reference_latents.float(), nan=0.0,
                                         posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    outside = 1.0 - reference_mask
    kernel = 9
    outside_count = F.avg_pool2d(outside, kernel, stride=1, padding=kernel // 2)
    local_background = F.avg_pool2d(reference_latents * outside, kernel, stride=1,
                                    padding=kernel // 2) / outside_count.clamp_min(0.05)
    fallback_background = F.avg_pool2d(reference_latents, kernel, stride=1, padding=kernel // 2)
    reliable = (outside_count > 0.05).to(reference_latents.dtype)
    local_background = (reliable * local_background +
                        (1.0 - reliable) * fallback_background).clamp(-10.0, 10.0)
    defect_residual = ((reference_latents - local_background) * reference_mask).clamp(-20.0, 20.0)
    latent_encoded = self.encoder(torch.cat([defect_residual, reference_mask], dim=1))
    if not self.ablation["latent_branch"]:
        latent_encoded = torch.zeros_like(latent_encoded)
    if reference_pixels is None:
        raise ValueError("MSDF v2 requires reference_pixels before VAE compression")
    reference_pixels = torch.nan_to_num(reference_pixels.float(), nan=0.0,
                                        posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    pixel_mask = F.interpolate(reference_mask_pixels, size=reference_pixels.shape[-2:], mode="nearest")
    pixel_kernel = 31
    pixel_outside = 1.0 - pixel_mask
    pixel_count = F.avg_pool2d(pixel_outside, pixel_kernel, stride=1, padding=pixel_kernel // 2)
    pixel_background = F.avg_pool2d(reference_pixels * pixel_outside, pixel_kernel,
                                    stride=1, padding=pixel_kernel // 2) / pixel_count.clamp_min(0.02)
    pixel_fallback = F.avg_pool2d(reference_pixels, pixel_kernel, stride=1,
                                 padding=pixel_kernel // 2)
    pixel_reliable = (pixel_count > 0.02).to(reference_pixels.dtype)
    pixel_background = (pixel_reliable * pixel_background +
                        (1.0 - pixel_reliable) * pixel_fallback).clamp(-1.0, 1.0)
    pixel_residual = ((reference_pixels - pixel_background) * pixel_mask).clamp(-2.0, 2.0)
    pixel_encoded = self.pixel_encoder(torch.cat([pixel_residual, pixel_mask], dim=1))
    pixel_encoded = F.interpolate(pixel_encoded, size=latent_size,
                                  mode="bilinear", align_corners=False)
    pixel_support = (torch.sigmoid(torch.nan_to_num(
        self.pixel_support_head(pixel_encoded).float(), nan=-2.0,
        posinf=-2.0, neginf=-2.0).clamp(-12.0, 12.0)) * reference_mask)

    # The only two research interventions.
    pixel_feature_for_fusion = (pixel_encoded if controls["pixel_feature_enabled"]
                                else torch.zeros_like(pixel_encoded))
    pixel_support_for_reference = (pixel_support if controls["pixel_support_enabled"]
                                   else torch.zeros_like(pixel_support))
    self.last_pixel_support = pixel_support_for_reference

    if self.training and self.branch_dropout > 0:
        choice = torch.rand(latent_encoded.shape[0], 1, 1, 1,
                            device=latent_encoded.device)
        latent_encoded = latent_encoded * (choice >= self.branch_dropout)
        pixel_feature_for_fusion = pixel_feature_for_fusion * (choice < 1.0 - self.branch_dropout)

    fusion_input = torch.cat([latent_encoded, pixel_feature_for_fusion], dim=1)
    encoded = self.fusion(fusion_input)
    encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
    support_logits = torch.nan_to_num(self.support_head(encoded).float(), nan=-2.0,
                                      posinf=-2.0, neginf=-2.0).clamp(-12.0, 12.0)
    fused_support = torch.sigmoid(support_logits) * reference_mask
    reference_support = 0.5 * (fused_support + pixel_support_for_reference)
    if self.ablation["morphology_alignment"]:
        aligned, aligned_support = self._align_reference(encoded, reference_support, target_prior)
    else:
        aligned, aligned_support = encoded, reference_support
    context = F.max_pool2d(aligned_support, 3, stride=1, padding=1)
    support = (aligned_support + 0.25 * (context - aligned_support)).clamp(0, 1)
    support = support * F.max_pool2d(target_prior, 3, stride=1, padding=1)
    aligned = aligned * support
    area = target_prior.flatten(1).mean(1).clamp_min(1e-6)
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], device=area.device)
    timesteps = timesteps.to(device=area.device, dtype=torch.float32).flatten()
    if timesteps.numel() == 1 and area.numel() > 1:
        timesteps = timesteps.expand(area.numel())
    time_value = timesteps / max(float(num_train_timesteps - 1), 1.0)
    response = torch.nan_to_num(_safe_rms(aligned, dims=(1, 2, 3)), nan=0.0,
                                posinf=0.0, neginf=0.0).clamp(0.0, 20.0)
    gate_input = torch.stack([area.log(), time_value, response], dim=1)
    gate_logits = torch.nan_to_num(self.gate(gate_input).float(), nan=0.0,
                                   posinf=12.0, neginf=-12.0).clamp(-12.0, 12.0)
    gates = torch.sigmoid(gate_logits) * self.max_injection
    smallness = (-area.log() / 9.0).clamp(0, 1).unsqueeze(1)
    resolution_bias = torch.linspace(0.65, 1.35, self.num_scales,
                                     device=gates.device).unsqueeze(0)
    gates = gates * (1.0 + smallness * (resolution_bias - 1.0))
    gates = gates.clamp(0.0, self.max_injection)

    self._feature, self._support, self._gates = aligned, support, gates
    self._cfg = bool(classifier_free_guidance)
    self.last_gates = gates.detach()
    self.last_reference_support = reference_support
    self.last_aligned_support = aligned_support
    # Read-only diagnostic snapshots; none enters a computation above.
    self.factorial_snapshot = {
        "timestep": timesteps.detach(), "latent_encoded": latent_encoded.detach(),
        "pixel_encoded_raw": pixel_encoded.detach(),
        "pixel_feature_for_fusion": pixel_feature_for_fusion.detach(),
        "fusion_input": fusion_input.detach(), "fusion_output": encoded.detach(),
        "pixel_support_raw": pixel_support.detach(),
        "pixel_support_for_reference": pixel_support_for_reference.detach(),
        "fused_support": fused_support.detach(), "reference_support": reference_support.detach(),
        "aligned_support": aligned_support.detach(), "final_support": support.detach(),
        "gates": gates.detach(), "response": response.detach(), "target_area": area.detach(),
    }
    return gates


def bind_factorial(adapter, group):
    require(group in FACTORS, f"unknown factorial group: {group}")
    adapter.set_ablation({"pixel_branch": True, "latent_branch": True,
                          "morphology_alignment": True,
                          "up_block_mask": [True] * adapter.num_scales,
                          "residual_scale": 1.0})
    adapter.set_injection_gamma(1.0)
    adapter.factorial_controls = dict(FACTORS[group])
    adapter.prepare = MethodType(factorial_prepare, adapter)
    return adapter


def tensor_stats(value):
    import torch
    v = value.detach().float()
    return {"l2": float(torch.linalg.vector_norm(v)), "mean": float(v.mean()),
            "max": float(v.max()), "area_gt_0p05": float((v > 0.05).float().mean()),
            "finite": bool(torch.isfinite(v).all())}


def cpu_preflight():
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter
    torch.manual_seed(20261025)
    adapter = MSDFAdapter(4, 8, [4, 4, 4, 4]).eval()
    initial = {k: v.detach().clone() for k, v in adapter.named_parameters()}
    latents = torch.randn(1, 4, 8, 8)
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    mask = torch.zeros(1, 1, 64, 64); mask[:, :, 18:44, 20:42] = 1
    prior = torch.zeros(1, 1, 64, 64); prior[:, :, 16:48, 16:48] = 1
    rows = []
    for group in FACTOR_GROUPS:
        bind_factorial(adapter, group)
        with torch.inference_mode():
            adapter.prepare(latents, mask, prior, torch.tensor([500]), 1000, False, pixels)
        snap = adapter.factorial_snapshot
        row = {"group": group, **FACTORS[group], "gamma": adapter.injection_gamma,
               "latent_l2": tensor_stats(snap["latent_encoded"])["l2"],
               "pixel_raw_l2": tensor_stats(snap["pixel_encoded_raw"])["l2"],
               "pixel_fusion_l2": tensor_stats(snap["pixel_feature_for_fusion"])["l2"],
               "pixel_support_raw_l2": tensor_stats(snap["pixel_support_raw"])["l2"],
               "pixel_support_reference_l2": tensor_stats(snap["pixel_support_for_reference"])["l2"],
               "fusion_channels": int(snap["fusion_input"].shape[1]),
               "all_finite": all(tensor_stats(snap[k])["finite"] for k in
                  ("latent_encoded", "pixel_encoded_raw", "fusion_input", "fusion_output",
                   "pixel_support_raw", "pixel_support_for_reference", "reference_support",
                   "aligned_support", "final_support", "gates"))}
        require(row["latent_l2"] > 0 and row["pixel_raw_l2"] > 0 and
                row["pixel_support_raw_l2"] > 0 and row["fusion_channels"] == 16 and
                row["gamma"] == 1.0 and row["all_finite"], f"CPU common state failed: {group}")
        require((row["pixel_fusion_l2"] > 0) == FACTORS[group]["pixel_feature_enabled"],
                f"CPU F switch failed: {group}")
        require((row["pixel_support_reference_l2"] > 0) == FACTORS[group]["pixel_support_enabled"],
                f"CPU S switch failed: {group}")
        rows.append(row)
    require(all(torch.equal(initial[k], v) for k, v in adapter.named_parameters()),
            "CPU preflight changed parameters")
    return {"status": "PASS", "groups": rows, "parameters_unchanged": True,
            "optimizer_steps": 0, "pixel_support_recomputed_from_raw_encoder": True,
            "fusion_width_unchanged": True, "morphology_alignment": True,
            "dynamic_gate": True, "gamma": 1.0}


def select_diagnostic_slots(slots):
    black = [r for r in slots if r["class"] == "black"]
    require(len(black) == 40, "Black slot count changed")
    def area(row):
        b = row["bbox"]; return (b[2] - b[0]) * (b[3] - b[1])
    ordered = sorted(black, key=lambda r: (area(r), r["source_sha256"], r["slot_id"]))
    chosen = [ordered[i] for i in (0, 13, 26, 39)]
    return chosen


def audit():
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    for group in FACTOR_GROUPS:
        for bank in "ABC":
            (OUTPUT / "diagnostics" / group / bank).mkdir(parents=True, exist_ok=True)
    for group in NEW_GROUPS:
        for bank in "ABC": (OUTPUT / "images" / group / bank).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    status(status="AUDITING", created_at=now(), base_commit=BASE_COMMIT, training_count=0,
           parameter_update_count=0, generation_count=0, reroll_count=0,
           selection_count=0, final_eval_read_or_used=False, BootstrapGuard=False,
           detector_training_started=False)
    path, binding = checkpoint(full_hash=True)
    history = x14.verify_history()
    generation = load(P0 / "stage20b_generation_protocol.json")
    require(generation["scheduler"] == "DDIMScheduler" and generation["num_inference_steps"] == 50 and
            generation["guidance_scale"] == 7.5 and generation["eta"] == 0 and
            generation["strength"] == 1 and generation["height"] == generation["width"] == 512 and
            generation["precision"].startswith("fp16"), "generation protocol changed")
    cpu = cpu_preflight()
    core = REPO / "diffusers/pipelines/stable_diffusion/msdf_guidance.py"
    save(OUTPUT / "factorial_checkpoint_audit.json", {
        "status": "PASS", "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "unet_SHA256": EXPECTED_UNET,
        "successful_updates": binding["successful_updates"], "training_weights": WEIGHTS,
        "checkpoint_file_count": len(binding["checkpoint"]["sha256"]),
        "checkpoint_file_sizes": {name: (path / name).stat().st_size
                                  for name in binding["checkpoint"]["sha256"]},
        "actual_server_file_SHA256_recomputed": True, "history_SHA_checks": history,
        "source_code_SHA256": source_hashes(), "completed_at": now()})
    save(OUTPUT / "factorial_implementation_diff.json", {
        "status": "PASS", "core_source_modified": False, "core_source_SHA256": sha(core),
        "research_only_override": "factorial_prepare bound per loaded adapter instance",
        "controls": FACTORS,
        "only_formula_differences": [
            "pixel_feature_for_fusion = pixel_encoded or zeros_like(pixel_encoded)",
            "pixel_support_for_reference = pixel_support or zeros_like(pixel_support)"],
        "reference_support_formula": "0.5 * (fused_support + pixel_support_for_reference)",
        "support_weight_renormalized": False, "new_trainable_layers": 0,
        "checkpoint_format_modified": False})
    save(OUTPUT / "factorial_cpu_preflight.json", cpu)
    slots = select_diagnostic_slots(load(P0 / "stage20b_generation_slots.json")["slots"])
    records = []
    for bank in "ABC":
        for index, row in enumerate(slots):
            records.append({"bank": bank, "diagnostic_index": index,
                            "slot_id": row["slot_id"], "source_SHA256": row["source_sha256"],
                            "bbox": row["bbox"], "selection_basis": "fixed bbox-area quartile before generation"})
    save(OUTPUT / "factorial_diagnostic_sample_manifest.json", {
        "status": "FROZEN_BEFORE_GPU", "records": records, "count": 12,
        "groups": list(FACTOR_GROUPS), "images_planned": 48,
        "selection_used_generated_images": False})
    save(OUTPUT / "factorial_protocol.json", {
        "status": "FROZEN_BEFORE_GPU", "experiment": "Experiment 16 — MSDF Pixel Feature / Pixel Support Factorial Decoupling",
        "base_commit": BASE_COMMIT, "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "unet_SHA256": EXPECTED_UNET,
        "groups": GROUPS, "factor_controls": FACTORS, "new_groups": list(NEW_GROUPS),
        "gamma": 1.0, "morphology_alignment": True, "latent_encoder": True,
        "dynamic_gate": True, "up_block_injection": [True] * 4, "residual_scale": 1.0,
        "banks": BANK_SEEDS, "generation_protocol": generation,
        "new_generation": {"f0s1": 120, "f1s0": 120, "total": 240},
        "diagnostics": {"slots_per_bank": 4, "groups": 4, "steps": 50},
        "evaluation": {"references": {"flash": 75, "black": 97, "overall": 172},
            "ROI": "bbox plus 25 percent context; frozen floor/ceil clipping",
            "Inception": "torch-fidelity inception-v3-compat 2048D; float32 extraction; float64 CPU metrics",
            "KID": {"subsets": 100, "seed": 42, "degree": 3,
                    "subset_size": "min(50,nreal,nfake)"}},
        "training_forbidden": True, "final_eval_read_or_used": False})
    save(OUTPUT / "factorial_generation_manifest.json", {"status": "PENDING", "planned_count": 240,
        "records": [], "reroll_count": 0, "selection_count": 0})
    status(status="AUDIT_PASS", audit_completed_at=now(), CPU_preflight="PASS")
    print("[审计通过] 冻结checkpoint、历史数据、生成协议和四组CPU因子测试通过。", flush=True)


def load_pipeline(device, group):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    require(group in FACTORS, f"unknown group: {group}")
    checkpoint_path, binding = checkpoint()
    cache = os.environ.get("EXP16_CHECKPOINT_CACHE")
    if cache:
        cached = Path(cache)
        require(all((cached / name).is_file() and (cached / name).stat().st_size ==
                    (checkpoint_path / name).stat().st_size for name in binding["checkpoint"]["sha256"]),
                "checkpoint cache size audit failed")
        checkpoint_path = cached
    generation = load(P0 / "stage20b_generation_protocol.json")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(checkpoint_path), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float16, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False).to(device)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    pipe.disable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(generation["scheduler_config"])
    adapter = load_msdf_adapter(checkpoint_path / "msdf.pt", pipe.unet, device).eval()
    bind_factorial(adapter, group)
    require(adapter.injection_gamma == 1.0 and not adapter.training and
            adapter.ablation["morphology_alignment"] and adapter.ablation["latent_branch"],
            f"runtime controls invalid: {group}")
    return pipe, adapter, generation


def adapter_digest(adapter):
    digest = hashlib.sha256()
    for name, value in sorted(adapter.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate_one(pipe, adapter, slot, rng, generation, device, callback=None):
    return x14.generate_one(pipe, adapter, slot, rng, generation, device, callback)


def diagnostic_hooks(adapter, pipe, group, bank, slot_id, sink):
    import torch
    handles, before = [], {}

    def capture_before(_module, _inputs, output, block_index):
        before[block_index] = (output[0] if isinstance(output, tuple) else output).detach()

    def capture_after(_module, _inputs, output, block_index):
        updated = (output[0] if isinstance(output, tuple) else output).detach().float()
        hidden = before.pop(block_index).float(); residual = updated - hidden
        hidden_l2 = float(torch.linalg.vector_norm(hidden)); residual_l2 = float(torch.linalg.vector_norm(residual))
        hidden_rms = (hidden.square().mean(dim=(1, 2, 3)) + 1e-8).sqrt().reshape(-1, 1, 1, 1).clamp_min(1e-4)
        limit = adapter.max_residual_ratio * hidden_rms
        bounded_ratio = residual.abs() / limit.clamp_min(1e-6)
        current = dict(sink["current"])
        sink["gate_residual"].append({"group": group, "bank": bank, "slot_id": slot_id,
            "step_index": current["step_index"], "timestep": current["timestep"],
            "up_block": block_index, "dynamic_gate": current["gates"][block_index],
            "hidden_l2": hidden_l2, "residual_l2": residual_l2,
            "residual_hidden_relative_l2": residual_l2 / max(hidden_l2, 1e-12),
            "tanh_bound_ratio_mean": float(bounded_ratio.mean()),
            "tanh_bound_ratio_max": float(bounded_ratio.max()),
            "fraction_near_tanh_limit_gt_0p95": float((bounded_ratio > .95).float().mean()),
            "finite": bool(torch.isfinite(updated).all() and torch.isfinite(residual).all())})

    for index, block in enumerate(pipe.unet.up_blocks):
        handles.append(block.register_forward_hook(
            lambda m, i, o, idx=index: capture_before(m, i, o, idx), prepend=True))
        handles.append(block.register_forward_hook(
            lambda m, i, o, idx=index: capture_after(m, i, o, idx)))
    return handles


def diagnostic_callback(group, bank, slot_id, sink):
    def callback(adapter):
        snap = adapter.factorial_snapshot
        step = len(sink["feature"])
        timestep = float(snap["timestep"].flatten()[0])
        gates = [float(v) for v in snap["gates"].flatten()]
        sink["current"] = {"step_index": step, "timestep": timestep, "gates": gates}
        latent = tensor_stats(snap["latent_encoded"])
        pixel_raw = tensor_stats(snap["pixel_encoded_raw"])
        pixel_fusion = tensor_stats(snap["pixel_feature_for_fusion"])
        fusion_in = tensor_stats(snap["fusion_input"])
        fusion_out = tensor_stats(snap["fusion_output"])
        sink["feature"].append({"group": group, "bank": bank, "slot_id": slot_id,
            "step_index": step, "timestep": timestep, "pixel_feature_enabled": FACTORS[group]["pixel_feature_enabled"],
            "pixel_support_enabled": FACTORS[group]["pixel_support_enabled"],
            "pixel_encoder_l2": pixel_raw["l2"], "latent_encoder_l2": latent["l2"],
            "pixel_feature_for_fusion_l2": pixel_fusion["l2"],
            "fusion_input_l2": fusion_in["l2"], "fusion_output_l2": fusion_out["l2"],
            "all_finite": all(x["finite"] for x in (latent, pixel_raw, pixel_fusion, fusion_in, fusion_out))})
        raw = tensor_stats(snap["pixel_support_raw"])
        explicit = tensor_stats(snap["pixel_support_for_reference"])
        fused = tensor_stats(snap["fused_support"])
        reference = tensor_stats(snap["reference_support"])
        aligned = tensor_stats(snap["aligned_support"])
        final = tensor_stats(snap["final_support"])
        sink["support"].append({"group": group, "bank": bank, "slot_id": slot_id,
            "step_index": step, "timestep": timestep,
            "pixel_support_raw_mean": raw["mean"], "pixel_support_raw_max": raw["max"],
            "pixel_support_raw_area_gt_0p05": raw["area_gt_0p05"],
            "explicit_pixel_support_mean": explicit["mean"], "explicit_pixel_support_max": explicit["max"],
            "explicit_pixel_support_area_gt_0p05": explicit["area_gt_0p05"],
            "fused_support_mean": fused["mean"], "reference_support_mean": reference["mean"],
            "reference_support_max": reference["max"],
            "reference_support_area_gt_0p05": reference["area_gt_0p05"],
            "aligned_support_mean": aligned["mean"], "aligned_support_area_gt_0p05": aligned["area_gt_0p05"],
            "final_support_mean": final["mean"], "final_support_area_gt_0p05": final["area_gt_0p05"],
            "abnormally_small_support": final["area_gt_0p05"] < 1e-4,
            "response_rms": float(snap["response"].flatten()[0]),
            "all_finite": all(x["finite"] for x in (raw, explicit, fused, reference, aligned, final))})
    return callback


def historical_image(group, bank, slot_id):
    if group == "f1s1":
        root, rows = x14.full_rows(bank)
        row = next(r for r in rows if r["slot_id"] == slot_id)
        return root / row["image_path"], x14.image_digest(row)
    manifest = load(X14 / "branch_generation_manifest.json")
    row = next(r for r in manifest["records"] if r["group"] == "latent_only" and
               r["bank"] == bank and r["slot_id"] == slot_id)
    return X14 / row["image_path"], row["image_SHA256"]


def diagnose(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "AUDIT_PASS",
            "diagnostics require audit PASS")
    status(status="DIAGNOSTICS_RUNNING", diagnostics_started_at=now())
    sample_manifest = load(OUTPUT / "factorial_diagnostic_sample_manifest.json")
    selected = defaultdict(list)
    all_slots = {r["slot_id"]: r for r in load(P0 / "stage20b_generation_slots.json")["slots"]}
    for row in sample_manifest["records"]: selected[row["bank"]].append(all_slots[row["slot_id"]])
    rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
    feature_rows, support_rows, gate_rows, regression_rows = [], [], [], []
    state_digests = {}
    try:
        with x12.gpu_runtime(device) as torch:
            for group in FACTOR_GROUPS:
                pipe, adapter, generation = load_pipeline(device, group)
                before = adapter_digest(adapter); state_digests[group] = before
                for bank in "ABC":
                    for index, slot in enumerate(selected[bank]):
                        sink = {"feature": [], "support": [], "gate_residual": [], "current": {}}
                        handles = diagnostic_hooks(adapter, pipe, group, bank, slot["slot_id"], sink)
                        try:
                            image, trace = generate_one(pipe, adapter, slot, rngs[slot["slot_id"]][bank],
                                generation, device, diagnostic_callback(group, bank, slot["slot_id"], sink))
                        finally:
                            for handle in handles: handle.remove()
                        require(len(sink["feature"]) == len(sink["support"]) == 50 and
                                len(sink["gate_residual"]) == 200, "diagnostic step count mismatch")
                        require(all(r["all_finite"] for r in sink["feature"] + sink["support"]) and
                                all(r["finite"] for r in sink["gate_residual"]), "non-finite diagnostic")
                        path = OUTPUT / "diagnostics" / group / bank / f"{slot['slot_id']}_{index}.png"
                        image.save(path, format="PNG")
                        exact = None
                        if index == 0 and group in ("f1s1", "f0s0"):
                            old_path, old_digest = historical_image(group, bank, slot["slot_id"])
                            require(old_path.is_file() and sha(old_path) == old_digest, "historical regression image changed")
                            exact = sha(path) == old_digest
                        regression_rows.append({"group": group, "bank": bank, "diagnostic_index": index,
                            "slot_id": slot["slot_id"], "sample_seed": rngs[slot["slot_id"]][bank]["sample_seed"],
                            "image_SHA256": sha(path), "historical_exact_SHA256": exact, **trace})
                        feature_rows.extend(sink["feature"]); support_rows.extend(sink["support"])
                        gate_rows.extend(sink["gate_residual"])
                        print(f"[诊断] {group} Bank {bank} {index + 1}/4 exact={exact}", flush=True)
                require(before == adapter_digest(adapter), f"adapter parameters changed: {group}")
                adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        require(len(set(state_digests.values())) == 1, "factor groups used different adapter weights")
        exact_rows = [r for r in regression_rows if r["historical_exact_SHA256"] is not None]
        require(len(exact_rows) == 6 and all(r["historical_exact_SHA256"] for r in exact_rows),
                "Full/LatentOnly strict PNG regression failed")
        for group in FACTOR_GROUPS:
            fr = [r for r in feature_rows if r["group"] == group]
            sr = [r for r in support_rows if r["group"] == group]
            require(all(r["latent_encoder_l2"] > 0 and r["pixel_encoder_l2"] > 0 for r in fr),
                    f"raw encoders invalid: {group}")
            require(all((r["pixel_feature_for_fusion_l2"] > 0) == FACTORS[group]["pixel_feature_enabled"] for r in fr),
                    f"F runtime switch invalid: {group}")
            require(all((r["explicit_pixel_support_max"] > 0) == FACTORS[group]["pixel_support_enabled"] for r in sr),
                    f"S runtime switch invalid: {group}")
        write_csv(OUTPUT / "factorial_feature_diagnostics.csv", feature_rows)
        write_csv(OUTPUT / "factorial_support_statistics.csv", support_rows)
        write_csv(OUTPUT / "factorial_gate_residual_statistics.csv", gate_rows)
        save(OUTPUT / "factorial_full_latent_regression.json", {"status": "PASS",
            "records": regression_rows, "strict_checks": 6, "strict_passes": 6,
            "same_adapter_state_SHA256": next(iter(state_digests.values())),
            "parameters_unchanged": True, "diagnostic_images_excluded_from_formal_pool": True})
        sample_manifest.update(status="COMPLETE", completed_at=now(), generated_images=48)
        save(OUTPUT / "factorial_diagnostic_sample_manifest.json", sample_manifest)
        status(status="DIAGNOSTICS_PASS", diagnostics_completed_at=now(),
               full_latent_strict_regression="PASS", diagnostic_images=48)
        print("[诊断通过] F1S1/F0S0 六项PNG严格回归及四组48样本机制诊断通过。", flush=True)
    except Exception as error:
        status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def generate(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "DIAGNOSTICS_PASS",
            "formal generation requires diagnostics PASS")
    status(status="GENERATING", generation_started_at=now())
    records, started = [], time.monotonic()
    manifest_path = OUTPUT / "factorial_generation_manifest.json"
    try:
        with x12.gpu_runtime(device) as torch:
            from stage20b_formal_common import synthetic_label
            slots = [r for r in load(P0 / "stage20b_generation_slots.json")["slots"] if r["class"] == "black"]
            originals = {r["slot_id"]: r for r in load(
                Path(load(STAGE25 / "stage25_run_binding.json")["p0"]) / "stage20b_generation_slots.json")["slots"]}
            rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
            for group in NEW_GROUPS:
                pipe, adapter, protocol = load_pipeline(device, group); before = adapter_digest(adapter)
                for bank in "ABC":
                    for index, slot in enumerate(slots):
                        frozen, rng = originals[slot["slot_id"]], rngs[slot["slot_id"]][bank]
                        require(rng["bank_seed"] == BANK_SEEDS[bank] and
                                rng["non_rng_metadata_sha256"] == hashlib.sha256(canonical(frozen).encode()).hexdigest(),
                                "frozen RNG metadata mismatch")
                        item_started = time.monotonic()
                        image, trace = generate_one(pipe, adapter, slot, rng, protocol, device)
                        image_path = OUTPUT / "images" / group / bank / f"{slot['slot_id']}.png"
                        label_path = image_path.with_suffix(".txt")
                        require(not image_path.exists() and not label_path.exists(), "formal output overwrite blocked")
                        image.save(image_path, format="PNG")
                        label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                        width, height = frozen["geometry"]["final_canvas_wh"]
                        records.append({"group": group, "method": GROUPS[group], "bank": bank, "class": "black",
                            "slot_id": slot["slot_id"], "frozen_slot": frozen, "runtime_slot": slot,
                            "normal_SHA256": frozen["source_sha256"], "bbox": frozen["bbox"],
                            "prompt": frozen["prompt"], "sample_seed": rng["sample_seed"],
                            "bank_seed": rng["bank_seed"], "checkpoint_tree_SHA256": EXPECTED_TREE,
                            "msdf_checkpoint_SHA256": EXPECTED_MSDF, **FACTORS[group],
                            "injection_gamma": 1.0, "image_path": image_path.relative_to(OUTPUT).as_posix(),
                            "image_SHA256": sha(image_path), "label_path": label_path.relative_to(OUTPUT).as_posix(),
                            "label_SHA256": sha(label_path), "roi_xyxy": crop_box(frozen["bbox"], width, height),
                            "quality_selection": False, "reroll": False,
                            "runtime_seconds": time.monotonic() - item_started, **trace})
                        save(manifest_path, {"status": "RUNNING", "planned_count": 240, "records": records,
                            "reroll_count": 0, "selection_count": 0})
                        print(f"[生成] {group} Bank {bank} {index + 1}/40 total={len(records)}/240", flush=True)
                require(before == adapter_digest(adapter), f"parameters changed: {group}")
                adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        counts = Counter((r["group"], r["bank"]) for r in records)
        require(len(records) == 240 and all(counts[(g, b)] == 40 for g in NEW_GROUPS for b in "ABC"),
                "formal generation incomplete")
        save(manifest_path, {"status": "COMPLETE", "count": 240, "records": records,
            "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
            "parameter_update_count": 0, "runtime_seconds": time.monotonic() - started})
        status(status="GENERATION_COMPLETE", generation_count=240, generation_completed_at=now(),
               reroll_count=0, selection_count=0, parameter_update_count=0)
        print("[生成完成] F0S1/F1S0 A/B/C 共240/240。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "records": records,
            "error": f"{type(error).__name__}: {error}", "reroll_count": 0, "selection_count": 0})
        status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def extract_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    cache_path = OUTPUT / "factorial_inception_features.npz"
    ledger_path = OUTPUT / "factorial_inception_feature_cache.json"
    expected_keys = [f"{r['group']}:{r['bank']}:{r['slot_id']}" for r in manifest["records"]]
    if cache_path.is_file() and ledger_path.is_file():
        ledger = load(ledger_path)
        require(ledger["status"] == "COMPLETE" and ledger["keys"] == expected_keys and
                ledger["sha256"] == sha(cache_path), "feature cache binding changed")
        return x12.array_features(cache_path, ledger_path, torch)
    model = lib["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
    values, keys = [], []
    with torch.inference_mode():
        for index, row in enumerate(manifest["records"]):
            path = OUTPUT / row["image_path"]
            require(path.is_file() and sha(path) == row["image_SHA256"], "generated image changed")
            with Image.open(path) as source: crop = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
            tensor = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = model(tensor)[0]
            require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()), row["slot_id"])
            keys.append(f"{row['group']}:{row['bank']}:{row['slot_id']}")
            values.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0: print(f"[Inception] {index + 1}/240", flush=True)
    data = np.stack(values); np.savez_compressed(cache_path, features=data)
    save(ledger_path, {"status": "COMPLETE", "keys": keys, "shape": [240, 2048],
        "stored_dtype": str(data.dtype), "extraction_dtype": "float32", "sha256": sha(cache_path),
        "image_SHA256_checks": 240})
    del model; torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[i]).double() for i, key in enumerate(keys)}


def historical_metrics():
    def convert(row, scope_key):
        mapping = {"full": "f1s1", "latent_only": "f0s0"}
        group = mapping.get(row["group"], row["group"])
        return {scope_key: row[scope_key], "group": group, "method": GROUPS[group],
            "KID": float(row["KID"]), "KID_subset_std": float(row["KID_subset_std"]),
            "FID": float(row["FID"]), "reference_ROIs": int(float(row["reference_ROIs"])),
            "generated_images": int(float(row["generated_images"])),
            "KID_subset_size": int(float(row["KID_subset_size"])),
            "runtime_seconds": float(row["runtime_seconds"]),
            "metric_source": "reused SHA-bound Experiment 14 same-protocol metrics"}
    keep = {"raw", "gamma0", "full", "latent_only"}
    black = [convert(r, "bank") for r in read_csv(X14 / "branch_bankwise_black.csv") if r["group"] in keep]
    overall = [convert(r, "bank") for r in read_csv(X14 / "branch_bankwise_overall.csv") if r["group"] in keep]
    pooled = []
    for row in read_csv(X14 / "branch_pooled_metrics.csv"):
        if row["group"] in keep:
            value = convert(row, "scope"); value["population"] = row["population"]; pooled.append(value)
    require(len(black) == len(overall) == 12 and len(pooled) == 16, "historical metric set changed")
    return black, overall, pooled


def metric_delta(first, second, metric):
    delta = first[metric] - second[metric]
    return {"first": first[metric], "second": second[metric], "absolute_delta": delta,
            "relative_percent": 100 * delta / second[metric], "first_is_better": delta < 0}


def evaluate(device):
    require(load(OUTPUT / "experiment_status.json")["status"] in ("GENERATION_COMPLETE", "EVALUATING"),
            "evaluation requires complete generation")
    status(status="EVALUATING", evaluation_started_at=now()); started = time.monotonic()
    try:
        with x12.gpu_runtime(device) as torch:
            from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
            lib, runtime_audit = metric_runtime(device)
            manifest = load(OUTPUT / "factorial_generation_manifest.json")
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
            black_rows, overall_rows = [], []
            for bank in "ABC":
                black_rows.extend(r for r in old_black if r["bank"] == bank)
                overall_rows.extend(r for r in old_overall if r["bank"] == bank)
                for group in NEW_GROUPS:
                    fake = black_features(group, bank)
                    bm = x12.metric(real_black, fake, lib)
                    om = x12.metric(real_all, torch.cat([shared_flash(bank), fake]), lib)
                    black_rows.append({"bank": bank, "group": group, "method": GROUPS[group], **bm,
                                       "metric_source": "new same-protocol factorial features"})
                    overall_rows.append({"bank": bank, "group": group, "method": GROUPS[group], **om,
                                         "metric_source": "new factorial plus shared Flash features"})
                    print(f"[指标] {bank} {group} Black KID={bm['KID']:.10f} FID={bm['FID']:.6f}", flush=True)
            pooled_rows = []
            for scope, banks in (("A+B+C", "ABC"), ("B+C", "BC")):
                pooled_rows.extend(r for r in old_pooled if r["scope"] == scope)
                for population, real in (("Black", real_black), ("Overall", real_all)):
                    for group in NEW_GROUPS:
                        fake = (torch.cat([black_features(group, b) for b in banks]) if population == "Black" else
                                torch.cat([torch.cat([shared_flash(b), black_features(group, b)]) for b in banks]))
                        result = x12.metric(real, fake, lib)
                        pooled_rows.append({"scope": scope, "population": population, "group": group,
                            "method": GROUPS[group], **result, "metric_source": "new concatenated features"})
        order = {g: i for i, g in enumerate(("raw", "gamma0", "f1s1", "f0s0", "f0s1", "f1s0"))}
        black_rows.sort(key=lambda r: (r["bank"], order[r["group"]]))
        overall_rows.sort(key=lambda r: (r["bank"], order[r["group"]]))
        pooled_rows.sort(key=lambda r: (("A+B+C", "B+C").index(r["scope"]),
            ("Black", "Overall").index(r["population"]), order[r["group"]]))
        write_csv(OUTPUT / "factorial_bankwise_black.csv", black_rows)
        write_csv(OUTPUT / "factorial_bankwise_overall.csv", overall_rows)
        write_csv(OUTPUT / "factorial_pooled_metrics.csv", pooled_rows)

        effects, interactions = [], []
        scopes = [(b, "Black", {r["group"]: r for r in black_rows if r["bank"] == b}) for b in "ABC"]
        scopes += [(b, "Overall", {r["group"]: r for r in overall_rows if r["bank"] == b}) for b in "ABC"]
        scopes += [(s, p, {r["group"]: r for r in pooled_rows if r["scope"] == s and r["population"] == p})
                   for s in ("A+B+C", "B+C") for p in ("Black", "Overall")]
        for scope, population, by in scopes:
            for metric in ("KID", "FID"):
                values = {g: by[g][metric] for g in FACTOR_GROUPS}
                entries = {
                    "Delta_F_given_S1": values["f0s1"] - values["f1s1"],
                    "Delta_F_given_S0": values["f0s0"] - values["f1s0"],
                    "Delta_S_given_F1": values["f1s0"] - values["f1s1"],
                    "Delta_S_given_F0": values["f0s0"] - values["f0s1"],
                }
                for name, value in entries.items():
                    effects.append({"scope": scope, "population": population, "metric": metric,
                                    "effect": name, "value": value, "positive_means_retained_path_better": value > 0})
                interactions.append({"scope": scope, "population": population, "metric": metric,
                    "M11": values["f1s1"], "M01": values["f0s1"], "M10": values["f1s0"],
                    "M00": values["f0s0"],
                    "interaction": values["f1s1"] - values["f1s0"] - values["f0s1"] + values["f0s0"]})
        write_csv(OUTPUT / "factorial_conditional_effects.csv", effects)
        save(OUTPUT / "factorial_interaction_summary.json", {"status": "DESCRIPTIVE_ONLY",
            "records": interactions,
            "boundary": "KID/FID are set-level metrics; interactions are descriptive non-additivity, not significance or individual causal effects."})

        by_bank = lambda b: {r["group"]: r for r in black_rows if r["bank"] == b}
        bc = {r["group"]: r for r in pooled_rows if r["scope"] == "B+C" and r["population"] == "Black"}
        full_consistent = all(by_bank(b)["f1s1"][m] < by_bank(b)[g][m]
                              for b in "BC" for m in ("KID", "FID") for g in NEW_GROUPS)
        full_pooled = all(bc["f1s1"][m] < bc[g][m] for m in ("KID", "FID") for g in NEW_GROUPS)
        support_promising = (bc["f0s1"]["KID"] < bc["f0s0"]["KID"] and
                             bc["f0s1"]["FID"] < bc["f0s0"]["FID"])
        fusion_promising = (bc["f1s0"]["KID"] < bc["f0s0"]["KID"] and
                            bc["f1s0"]["FID"] < bc["f0s0"]["FID"])
        if full_consistent and full_pooled: conclusion = "FACTORIAL_FULL_CONSISTENT_BENEFIT"
        elif support_promising: conclusion = "PIXEL_SUPPORT_PATH_PROMISING"
        elif fusion_promising: conclusion = "PIXEL_FUSION_PATH_PROMISING"
        else: conclusion = "FACTORIAL_MIXED"

        diagnostics_support = read_csv(OUTPUT / "factorial_support_statistics.csv")
        diagnostics_gate = read_csv(OUTPUT / "factorial_gate_residual_statistics.csv")
        def avg(rows, field): return sum(float(r[field]) for r in rows) / len(rows)
        path_diag = {}
        for group in FACTOR_GROUPS:
            sr = [r for r in diagnostics_support if r["group"] == group]
            gr = [r for r in diagnostics_gate if r["group"] == group]
            path_diag[group] = {"aligned_support_area_mean": avg(sr, "aligned_support_area_gt_0p05"),
                "final_support_area_mean": avg(sr, "final_support_area_gt_0p05"),
                "residual_l2_mean": avg(gr, "residual_l2"),
                "relative_residual_l2_mean": avg(gr, "residual_hidden_relative_l2"),
                "near_tanh_limit_fraction_mean": avg(gr, "fraction_near_tanh_limit_gt_0p95"),
                "abnormally_small_support_steps": sum(r["abnormally_small_support"] == "True" for r in sr)}
        comparisons = {}
        for first, second in (("f0s1", "f1s1"), ("f1s0", "f1s1"),
                              ("f0s0", "f0s1"), ("f0s0", "f1s0"),
                              ("f0s1", "raw"), ("f1s0", "raw")):
            comparisons[f"{first}-{second}"] = {m: metric_delta(bc[first], bc[second], m) for m in ("KID", "FID")}
        save(OUTPUT / "factorial_conclusion.json", {"status": conclusion,
            "primary_scope": "B+C pooled Black", "primary_metrics": {g: {m: bc[g][m] for m in ("KID", "FID")}
                for g in ("raw", "gamma0") + FACTOR_GROUPS}, "comparisons": comparisons,
            "full_BC_bankwise_and_pooled_better_than_both_new_groups": full_consistent and full_pooled,
            "support_path_promising_vs_f0s0": support_promising,
            "fusion_path_promising_vs_f0s0": fusion_promising,
            "diagnostic_group_means": path_diag,
            "interpretation_boundary": "F controls pixel features entering Fusion directly, but can still alter fused support and downstream morphology/gates. S controls only the explicit pixel-support term. Frozen-model inference intervention only."})
        counts = Counter((r["group"], r["bank"]) for r in manifest["records"])
        require(all(counts[(g, b)] == 40 for g in NEW_GROUPS for b in "ABC"), "integrity count mismatch")
        paired = {(r["group"], r["bank"], r["slot_id"]): r for r in manifest["records"]}
        identity_fields = ("frozen_slot", "runtime_slot", "normal_SHA256", "bbox", "prompt",
                           "sample_seed", "bank_seed", "checkpoint_tree_SHA256",
                           "msdf_checkpoint_SHA256", "injection_gamma")
        paired_checks = 0
        for bank in "ABC":
            slot_ids = sorted(r["slot_id"] for r in manifest["records"]
                              if r["group"] == "f0s1" and r["bank"] == bank)
            require(len(slot_ids) == 40, f"paired slot count mismatch: {bank}")
            for slot_id in slot_ids:
                left, right = paired[("f0s1", bank, slot_id)], paired[("f1s0", bank, slot_id)]
                require(all(left[name] == right[name] for name in identity_fields),
                        f"cross-group metadata mismatch: {bank}/{slot_id}")
                paired_checks += 1
        save(OUTPUT / "factorial_generation_integrity.json", {"status": "PASS",
            "counts": {g: {b: counts[(g, b)] for b in "ABC"} for g in NEW_GROUPS},
            "total": 240, "image_SHA256_checks": 240, "cross_group_metadata_identical": True,
            "paired_slots_checked": paired_checks,
            "paired_identity_field_checks": paired_checks * len(identity_fields),
            "reroll_count": 0, "selection_count": 0, "parameter_update_count": 0,
            "feature_cache_SHA256": sha(OUTPUT / "factorial_inception_features.npz"),
            "historical_full_latent_reused": True, "final_eval_read_or_used": False})
        protocol = load(OUTPUT / "factorial_protocol.json")
        protocol.update(status="COMPLETE", conclusion=conclusion, source_code_SHA256=source_hashes(),
                        metric_runtime=runtime_audit, completed_at=now())
        save(OUTPUT / "factorial_protocol.json", protocol)
        create_report(conclusion, black_rows, overall_rows, pooled_rows, bc, comparisons, path_diag)
        status(status=conclusion, completed_at=now(), evaluation_runtime_seconds=time.monotonic() - started,
               generation_count=240, followup_started=False)
        print(f"[完成] Experiment 16 status={conclusion}；停止。", flush=True)
    except Exception as error:
        status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}")
        raise


def create_report(conclusion, black, overall, pooled, bc, comparisons, path_diag):
    lines = ["# Experiment 16 — MSDF Pixel Feature / Pixel Support Factorial Decoupling", "",
        f"Status: `{conclusion}`.", "",
        "This is a frozen-checkpoint 2×2 inference intervention. F controls the direct pixel-feature input to Fusion; S controls the explicit pixel-support term. F can still alter fused support and downstream morphology/gating, so the paths are not fully independent causal channels.", "",
        "## Bankwise Black", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in black: lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Bankwise Overall", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for row in overall: lines.append(f"| {row['bank']} | {row['method']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## Pooled metrics", "", "All pooled and Overall metrics were recomputed from concatenated features.", "",
        "| Scope | Population | Method | Images | KID ↓ | FID ↓ |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in pooled: lines.append(f"| {row['scope']} | {row['population']} | {row['method']} | {row['generated_images']} | {row['KID']:.10f} | {row['FID']:.6f} |")
    lines += ["", "## B+C pooled Black findings", ""]
    for group in ("raw", "gamma0") + FACTOR_GROUPS:
        lines.append(f"- {GROUPS[group]}: KID `{bc[group]['KID']:.10f}`, FID `{bc[group]['FID']:.6f}`.")
    lines += ["", "Requested signed comparisons (first minus second; negative is better):", ""]
    for name, values in comparisons.items():
        lines.append(f"- {name}: KID `{values['KID']['absolute_delta']:+.10f}`, FID `{values['FID']['absolute_delta']:+.6f}`.")
    aligned_f = abs(path_diag["f1s1"]["aligned_support_area_mean"] - path_diag["f0s1"]["aligned_support_area_mean"])
    aligned_s = abs(path_diag["f1s1"]["aligned_support_area_mean"] - path_diag["f1s0"]["aligned_support_area_mean"])
    residual_f = abs(path_diag["f1s1"]["residual_l2_mean"] - path_diag["f0s1"]["residual_l2_mean"])
    residual_s = abs(path_diag["f1s1"]["residual_l2_mean"] - path_diag["f1s0"]["residual_l2_mean"])
    lines += ["", "## Mechanism diagnostics", "",
        f"The larger mean Aligned Support area change relative to Full came from `{'F' if aligned_f > aligned_s else 'S'}` (|ΔF|={aligned_f:.8f}, |ΔS|={aligned_s:.8f}).", "",
        f"The larger mean up-block residual L2 change relative to Full came from `{'F' if residual_f > residual_s else 'S'}` (|ΔF|={residual_f:.8f}, |ΔS|={residual_s:.8f}).", "",
        "Step/sample/block diagnostics are preserved in the three factorial diagnostic CSV files. Support means alone are not treated as quality evidence; conclusions use KID/FID together with bankwise consistency.", "",
        "## Research boundary", "",
        "No model was retrained and no parameter, gamma, scheduler, seed, morphology algorithm, dynamic-gate formula, or residual injection formula changed. Banks share frozen slots; B/C were observed previously; V2 TRAIN references may overlap model training. This is not an independent test or a cross-dataset generalization claim.", "",
        "The experiment stops here without BootstrapGuard, detector training, or another optimization round.", ""]
    (OUTPUT / "FUSION_SUPPORT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(device):
    diagnose(device)
    generate(device)
    evaluate(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "diagnose", "generate", "evaluate", "run"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"[启动] Experiment 16 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "audit": audit()
    elif args.phase == "diagnose": diagnose(args.device)
    elif args.phase == "generate": generate(args.device)
    elif args.phase == "evaluate": evaluate(args.device)
    else: run_all(args.device)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
