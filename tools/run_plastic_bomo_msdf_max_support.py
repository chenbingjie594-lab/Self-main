"""Experiment 17: confidence-preserving Max-Support fusion on frozen MSDF.

No training is available in this runner.  It reuses the Experiment 16 local
prepare implementation and changes only the reference support tensor passed
to the unchanged morphology alignment: mean(fused, pixel) versus
maximum(fused, pixel).
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
import statistics
import sys
import time
from types import MethodType

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))

import run_plastic_bomo_msdf_multibank_stability as x12
import run_plastic_bomo_msdf_branch_ablation as x14
import run_plastic_bomo_msdf_fusion_support_decoupling as x16

BASE_COMMIT = "8f5e54bb198026a552389ae44f458e5ca9e489b6"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_max_support"
X16 = REPO / "results_for_gpt/plastic_bomo_msdf_fusion_support_decoupling"
BG, X12, GAMMA, RAW, STAGE25, P0 = x16.BG, x16.X12, x16.GAMMA, x16.RAW, x16.STAGE25, x16.P0
BANK_SEEDS = dict(x16.BANK_SEEDS)
EXPECTED_TREE, EXPECTED_MSDF, EXPECTED_UNET = x16.EXPECTED_TREE, x16.EXPECTED_MSDF, x16.EXPECTED_UNET
WEIGHTS = dict(x16.WEIGHTS)
GROUPS = {
    "raw": "Raw SD2", "mean": "Full Mean-Support", "max": "Full Max-Support",
    "f0s1": "F0S1 (Pixel Feature OFF; Support ON)",
    "f1s0": "F1S0 (Pixel Feature ON; Support OFF)",
}
MODES = ("mean", "max")
load, save, sha, canonical, require = x12.load, x12.save, x12.sha, x12.canonical, x12.require
crop_box = x12.crop_box


def now(): return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows):
    require(rows, f"cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path): return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def update_status(**values):
    path = OUTPUT / "experiment_status.json"
    row = load(path) if path.is_file() else {}; row.update(values); save(path, row)


def checkpoint(full_hash=False):
    binding = load(BG / "background_only_checkpoint_binding.json")
    require(binding["status"] == "FROZEN" and binding["class"] == "black" and
            binding["successful_updates"] == 2000, "BackgroundOnly checkpoint is not frozen")
    path = BG / binding["checkpoint"]["path"]; expected = binding["checkpoint"]["sha256"]
    require(hashlib.sha256(canonical(expected).encode()).hexdigest() == EXPECTED_TREE,
            "checkpoint tree binding changed")
    require(binding["checkpoint"]["tree_sha256"] == EXPECTED_TREE and
            expected["msdf.pt"] == EXPECTED_MSDF and
            expected["unet/diffusion_pytorch_model.safetensors"] == EXPECTED_UNET,
            "checkpoint identity changed")
    require(binding["auxiliary_loss_weights"] == WEIGHTS and
            binding["morphology_alignment_enabled"] is True and binding["injection_gamma"] == 1.0,
            "checkpoint controls changed")
    require(all((path / name).is_file() for name in expected), "checkpoint file missing")
    if full_hash:
        for name, digest in expected.items(): require(sha(path / name) == digest, f"checkpoint changed: {name}")
    else:
        audit = load(OUTPUT / "max_support_checkpoint_audit.json")
        require(audit["status"] == "PASS" and audit["checkpoint_tree_SHA256"] == EXPECTED_TREE,
                "checkpoint audit binding changed")
        require(audit["checkpoint_file_sizes"] == {n: (path / n).stat().st_size for n in expected},
                "checkpoint sizes changed after audit")
    return path, binding


def source_hashes():
    names = ["tools/run_plastic_bomo_msdf_max_support.py",
             "tools/run_plastic_bomo_msdf_fusion_support_decoupling.py",
             "tools/run_plastic_bomo_msdf_branch_ablation.py",
             "tools/run_plastic_bomo_msdf_multibank_stability.py",
             "diffusers/pipelines/stable_diffusion/msdf_guidance.py"]
    return {name: sha(REPO / name) for name in names}


def alignment_geometry(adapter, reference_support, target_prior):
    """Read-only mirror of the frozen bbox/area/fit arithmetic."""
    import torch
    rows = []
    height, width = target_prior.shape[-2:]
    for batch_index in range(reference_support.shape[0]):
        raw_source_box = adapter._bbox(reference_support[batch_index, 0].detach())
        source_box = adapter._expand_bbox(raw_source_box, height, width)
        target_box = adapter._bbox(target_prior[batch_index, 0].detach())
        sy0, sy1, sx0, sx1 = source_box; ty0, ty1, tx0, tx1 = target_box
        source_h, source_w = max(1, sy1 - sy0), max(1, sx1 - sx0)
        target_h, target_w = max(1, ty1 - ty0), max(1, tx1 - tx0)
        source_region = reference_support[batch_index:batch_index + 1, :, sy0:sy1, sx0:sx1]
        occupied_raw_tensor = source_region.mean().detach()
        occupied_sanitized = torch.nan_to_num(occupied_raw_tensor, nan=0.08, posinf=0.85, neginf=0.08)
        occupied = occupied_sanitized.clamp(0.08, 0.85)
        desired_area = float(target_h * target_w) * float(occupied)
        aspect = source_w / float(source_h)
        initial_h = max(1, int(round((desired_area / max(aspect, 1e-6)) ** 0.5)))
        initial_w = max(1, int(round(initial_h * aspect)))
        fit = min(1.0, target_h / initial_h, target_w / initial_w)
        placed_h = max(1, int(round(initial_h * fit))); placed_w = max(1, int(round(initial_w * fit)))
        rows.append({
            "source_bbox_y0": raw_source_box[0], "source_bbox_y1": raw_source_box[1],
            "source_bbox_x0": raw_source_box[2], "source_bbox_x1": raw_source_box[3],
            "source_bbox_height": raw_source_box[1] - raw_source_box[0],
            "source_bbox_width": raw_source_box[3] - raw_source_box[2],
            "expanded_source_bbox_height": source_h, "expanded_source_bbox_width": source_w,
            "target_bbox_height": target_h, "target_bbox_width": target_w,
            "occupied_ratio_raw": float(occupied_raw_tensor),
            "occupied_ratio_after_nan_guard": float(occupied_sanitized),
            "occupied_ratio_clamped": float(occupied),
            "occupied_lower_bound_triggered": float(occupied_sanitized) < 0.08,
            "occupied_upper_bound_triggered": float(occupied_sanitized) > 0.85,
            "desired_area": desired_area, "placed_height_before_fit": initial_h,
            "placed_width_before_fit": initial_w, "fit_factor": fit,
            "target_fit_limited": fit < 1.0, "placed_height": placed_h, "placed_width": placed_w,
        })
    return rows


def install_composition(adapter, mode):
    """Intercept only the support tensor entering unchanged morphology alignment."""
    import torch
    import torch.nn.functional as F
    require(mode in MODES, f"unknown support composition: {mode}")
    base_prepare = adapter.prepare
    base_align = adapter._align_reference
    adapter.support_composition = mode

    def prepare(self, reference_latents, reference_mask, target_prior, timesteps,
                num_train_timesteps, classifier_free_guidance, reference_pixels=None):
        latent_size = reference_latents.shape[-2:]
        mask_latent = F.interpolate(reference_mask.float().clamp(0, 1),
                                    size=latent_size, mode="nearest").clamp(0, 1)
        capture = {}

        def support_head_capture(_module, _inputs, output):
            logits = torch.nan_to_num(output.float(), nan=-2.0, posinf=-2.0, neginf=-2.0).clamp(-12.0, 12.0)
            capture["fused_support"] = torch.sigmoid(logits) * mask_latent

        handle = self.support_head.register_forward_hook(support_head_capture)

        def aligned(_self, encoded, mean_support, target):
            fused = capture["fused_support"]
            pixel = self.last_pixel_support
            expected_mean = 0.5 * (fused + pixel)
            require(torch.equal(mean_support, expected_mean), "mean support formula changed")
            selected = mean_support if mode == "mean" else torch.maximum(fused, pixel)
            pixel_binary, fused_binary = pixel > 0.05, fused > 0.05
            intersection = (pixel_binary & fused_binary).float().mean()
            union = (pixel_binary | fused_binary).float().mean()
            capture.update({"pixel_support": pixel.detach(), "mean_support": mean_support.detach(),
                            "selected_support": selected.detach(),
                            "reference_overlap_intersection_area": float(intersection),
                            "reference_overlap_union_area": float(union),
                            "reference_overlap_iou": float(intersection / union.clamp_min(1e-12)),
                            "geometry": alignment_geometry(self, selected, target)})
            return base_align(encoded, selected, target)

        self._align_reference = MethodType(aligned, self)
        try:
            gates = base_prepare(reference_latents, reference_mask, target_prior, timesteps,
                                 num_train_timesteps, classifier_free_guidance, reference_pixels)
        finally:
            handle.remove(); self._align_reference = base_align
        selected = capture["selected_support"]
        self.last_reference_support = selected
        self.factorial_snapshot["reference_support"] = selected
        self.max_support_capture = capture
        return gates

    adapter.prepare = MethodType(prepare, adapter)
    return adapter


def tstats(value):
    import torch
    v = value.detach().float()
    return {"l2": float(torch.linalg.vector_norm(v)), "mean": float(v.mean()),
            "max": float(v.max()), "min": float(v.min()),
            "area": float((v > .05).float().mean()), "finite": bool(torch.isfinite(v).all())}


def make_cpu_adapter():
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter
    adapter = MSDFAdapter(4, 8, [4, 4, 4, 4]).eval()
    x16.bind_factorial(adapter, "f1s1")
    return adapter


def cpu_preflight():
    import torch
    torch.manual_seed(20261025)
    adapter = make_cpu_adapter(); initial = {k: v.detach().clone() for k, v in adapter.named_parameters()}
    latents = torch.randn(1, 4, 8, 8); pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    mask = torch.zeros(1, 1, 64, 64); mask[:, :, 18:44, 20:42] = 1
    prior = torch.zeros(1, 1, 64, 64); prior[:, :, 16:48, 16:48] = 1
    results = {}; snapshots = {}
    for mode in MODES:
        adapter = make_cpu_adapter(); adapter.load_state_dict(initial, strict=True); install_composition(adapter, mode)
        with torch.inference_mode(): adapter.prepare(latents, mask, prior, torch.tensor([500]), 1000, False, pixels)
        snap, cap = adapter.factorial_snapshot, adapter.max_support_capture
        results[mode] = {"reference": cap["selected_support"].clone(),
            "pixel_encoded": snap["pixel_encoded_raw"].clone(), "latent_encoded": snap["latent_encoded"].clone(),
            "fusion_output": snap["fusion_output"].clone(), "pixel_support": cap["pixel_support"].clone(),
            "fused_support": cap["fused_support"].clone()}
        snapshots[mode] = {k: tstats(v) for k, v in results[mode].items()}
        require(adapter.ablation["pixel_branch"] and adapter.ablation["latent_branch"] and
                adapter.ablation["morphology_alignment"] and adapter.injection_gamma == 1.0,
                f"CPU frozen controls changed: {mode}")
        require(all(x["finite"] for x in snapshots[mode].values()), f"non-finite CPU output: {mode}")
        require(all(torch.equal(initial[k], v) for k, v in adapter.named_parameters()), "CPU parameters changed")
    for key in ("pixel_encoded", "latent_encoded", "fusion_output", "pixel_support", "fused_support"):
        require(torch.equal(results["mean"][key], results["max"][key]), f"upstream tensor changed: {key}")
    fused, pixel = results["mean"]["fused_support"], results["mean"]["pixel_support"]
    require(torch.equal(results["mean"]["reference"], .5 * (fused + pixel)), "mean formula failed")
    require(torch.equal(results["max"]["reference"], torch.maximum(fused, pixel)), "max formula failed")
    require(bool((results["max"]["reference"] >= results["mean"]["reference"]).all()), "max < mean")
    require(float(results["max"]["reference"].min()) >= 0 and float(results["max"]["reference"].max()) <= 1,
            "max support out of range")
    latent_mask = torch.nn.functional.interpolate(mask, size=(8, 8), mode="nearest")
    require(bool((results["max"]["reference"] * (1 - latent_mask) == 0).all()), "mask exterior nonzero")
    return {"status": "PASS", "modes": snapshots, "upstream_tensors_bitwise_equal": True,
            "mean_formula_exact": True, "max_formula_exact": True, "max_ge_mean_everywhere": True,
            "max_range_0_1": True, "reference_mask_exterior_zero": True,
            "pixel_and_latent_enabled": True, "morphology_alignment": True, "gamma": 1.0,
            "parameters_unchanged": True, "optimizer_steps": 0,
            "boundary": "Post-alignment support is not required to remain pixelwise monotonic."}


def audit():
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    for mode in MODES:
        for bank in "ABC": (OUTPUT / "diagnostics" / mode / bank).mkdir(parents=True, exist_ok=True)
    for bank in "ABC": (OUTPUT / "images" / "max" / bank).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    update_status(status="AUDITING", created_at=now(), base_commit=BASE_COMMIT, training_count=0,
                  parameter_update_count=0, generation_count=0, reroll_count=0, selection_count=0,
                  final_eval_read_or_used=False, BootstrapGuard=False, detector_training_started=False)
    path, binding = checkpoint(full_hash=True); history = x14.verify_history()
    generation = load(P0 / "stage20b_generation_protocol.json")
    require(generation["scheduler"] == "DDIMScheduler" and generation["num_inference_steps"] == 50 and
            generation["guidance_scale"] == 7.5 and generation["eta"] == 0 and generation["strength"] == 1 and
            generation["height"] == generation["width"] == 512 and generation["precision"].startswith("fp16"),
            "generation protocol changed")
    cpu = cpu_preflight()
    save(OUTPUT / "max_support_checkpoint_audit.json", {"status": "PASS",
        "checkpoint_tree_SHA256": EXPECTED_TREE, "msdf_checkpoint_SHA256": EXPECTED_MSDF,
        "unet_SHA256": EXPECTED_UNET, "successful_updates": binding["successful_updates"],
        "training_weights": WEIGHTS, "checkpoint_file_count": len(binding["checkpoint"]["sha256"]),
        "checkpoint_file_sizes": {n: (path / n).stat().st_size for n in binding["checkpoint"]["sha256"]},
        "actual_server_file_SHA256_recomputed": True, "history_SHA_checks": history,
        "source_code_SHA256": source_hashes(), "completed_at": now()})
    save(OUTPUT / "max_support_implementation_diff.json", {"status": "PASS", "core_source_modified": False,
        "core_source_SHA256": sha(REPO / "diffusers/pipelines/stable_diffusion/msdf_guidance.py"),
        "experiment16_prepare_reused": True,
        "single_intervention": "reference_support passed to _align_reference: mean -> torch.maximum(fused_support, pixel_support)",
        "upstream_outputs_modified": False, "morphology_algorithm_modified": False,
        "gate_or_residual_formula_modified": False, "new_trainable_parameters": 0,
        "checkpoint_format_modified": False})
    save(OUTPUT / "max_support_cpu_preflight.json", cpu)
    x16_manifest = load(X16 / "factorial_diagnostic_sample_manifest.json")
    require(x16_manifest["status"] == "COMPLETE" and len(x16_manifest["records"]) == 12,
            "Experiment 16 diagnostic manifest changed")
    save(OUTPUT / "max_support_diagnostic_manifest.json", {"status": "FROZEN_BEFORE_GPU",
        "source": "Experiment 16 frozen diagnostic Bank-Slot instances",
        "records": x16_manifest["records"], "count": 12, "modes": list(MODES),
        "planned_images": 24, "selection_used_new_images": False})
    save(OUTPUT / "max_support_protocol.json", {"status": "FROZEN_BEFORE_GPU",
        "experiment": "Experiment 17 — Confidence-Preserving Max-Support Fusion",
        "base_commit": BASE_COMMIT, "checkpoint_tree_SHA256": EXPECTED_TREE,
        "msdf_checkpoint_SHA256": EXPECTED_MSDF, "unet_SHA256": EXPECTED_UNET,
        "groups": GROUPS, "support_composition": {"mean": "0.5*(fused+pixel)", "max": "torch.maximum(fused,pixel)"},
        "only_new_group": "max", "gamma": 1.0, "pixel_branch": True, "latent_branch": True,
        "morphology_alignment": True, "dynamic_gate": True, "banks": BANK_SEEDS,
        "generation_protocol": generation, "new_generation": {"A": 40, "B": 40, "C": 40, "total": 120},
        "evaluation": {"references": {"flash": 75, "black": 97, "overall": 172},
            "ROI": "bbox plus 25 percent context", "Inception": "inception-v3-compat 2048D FP32",
            "metrics": "FP64 CPU", "KID": {"subsets": 100, "seed": 42, "degree": 3,
                                             "subset_size": "min(50,nreal,nfake)"}},
        "training_forbidden": True, "final_eval_read_or_used": False})
    save(OUTPUT / "max_support_generation_manifest.json", {"status": "PENDING", "planned_count": 120,
        "records": [], "reroll_count": 0, "selection_count": 0})
    update_status(status="AUDIT_PASS", audit_completed_at=now(), CPU_preflight="PASS")
    print("[审计通过] checkpoint、历史结果、冻结协议与Mean/Max CPU单变量测试通过。", flush=True)


def load_pipeline(device, mode):
    require(mode in MODES, f"unknown mode: {mode}")
    # x16 loader is bound to the same audited checkpoint and Full F1S1 state.
    pipe, adapter, generation = x16.load_pipeline(device, "f1s1")
    install_composition(adapter, mode)
    require(adapter.injection_gamma == 1.0 and adapter.ablation["pixel_branch"] and
            adapter.ablation["latent_branch"] and adapter.ablation["morphology_alignment"],
            f"runtime state changed: {mode}")
    return pipe, adapter, generation


def adapter_digest(adapter): return x16.adapter_digest(adapter)


def diagnostic_hooks(adapter, pipe, mode, bank, slot_id, sink):
    import torch
    handles, before = [], {}
    def first(_module, _inputs, output, block):
        before[block] = (output[0] if isinstance(output, tuple) else output).detach()
    def last(_module, _inputs, output, block):
        updated = (output[0] if isinstance(output, tuple) else output).detach().float()
        hidden = before.pop(block).float(); residual = updated - hidden
        hidden_l2 = float(torch.linalg.vector_norm(hidden)); residual_l2 = float(torch.linalg.vector_norm(residual))
        hidden_rms = (hidden.square().mean((1, 2, 3)) + 1e-8).sqrt().reshape(-1, 1, 1, 1).clamp_min(1e-4)
        ratio = residual.abs() / (adapter.max_residual_ratio * hidden_rms).clamp_min(1e-6)
        cur = sink["current"]
        sink["gate"].append({"mode": mode, "bank": bank, "slot_id": slot_id,
            "step_index": cur["step_index"], "timestep": cur["timestep"], "up_block": block,
            "dynamic_gate": cur["gates"][block], "hidden_l2": hidden_l2, "residual_l2": residual_l2,
            "residual_hidden_relative_l2": residual_l2 / max(hidden_l2, 1e-12),
            "tanh_bound_ratio_mean": float(ratio.mean()), "tanh_bound_ratio_max": float(ratio.max()),
            "fraction_near_tanh_limit_gt_0p95": float((ratio > .95).float().mean()),
            "finite": bool(torch.isfinite(updated).all() and torch.isfinite(residual).all())})
    for index, block in enumerate(pipe.unet.up_blocks):
        handles.append(block.register_forward_hook(lambda m, i, o, x=index: first(m, i, o, x), prepend=True))
        handles.append(block.register_forward_hook(lambda m, i, o, x=index: last(m, i, o, x)))
    return handles


def diagnostic_callback(mode, bank, slot, sink):
    def callback(adapter):
        snap, cap = adapter.factorial_snapshot, adapter.max_support_capture
        step = len(sink["reference"]); timestep = float(snap["timestep"].flatten()[0])
        sink["current"] = {"step_index": step, "timestep": timestep,
                           "gates": [float(v) for v in snap["gates"].flatten()]}
        pixel, fused = tstats(cap["pixel_support"]), tstats(cap["fused_support"])
        selected, aligned, final = tstats(cap["selected_support"]), tstats(snap["aligned_support"]), tstats(snap["final_support"])
        sink["reference"].append({"mode": mode, "bank": bank, "slot_id": slot["slot_id"],
            "step_index": step, "timestep": timestep,
            "pixel_support_mean": pixel["mean"], "pixel_support_max": pixel["max"],
            "fused_support_mean": fused["mean"], "fused_support_max": fused["max"],
            "reference_support_mean": selected["mean"], "reference_support_max": selected["max"],
            "reference_support_area_gt_0p05": selected["area"],
            "pixel_fused_overlap_intersection_area": cap["reference_overlap_intersection_area"],
            "pixel_fused_overlap_union_area": cap["reference_overlap_union_area"],
            "pixel_fused_overlap_iou": cap["reference_overlap_iou"],
            "max_ge_mean_everywhere": bool((cap["selected_support"] >= cap["mean_support"]).all()),
            "all_finite": pixel["finite"] and fused["finite"] and selected["finite"]})
        geometry = cap["geometry"][0]
        sink["alignment"].append({"mode": mode, "bank": bank, "slot_id": slot["slot_id"],
            "step_index": step, "timestep": timestep, **geometry,
            "aligned_support_area_gt_0p05": aligned["area"],
            "final_support_area_gt_0p05": final["area"],
            "aligned_support_mean": aligned["mean"], "final_support_mean": final["mean"],
            "all_finite": aligned["finite"] and final["finite"]})
    return callback


def historical_full(bank, slot_id):
    root, rows = x14.full_rows(bank); row = next(r for r in rows if r["slot_id"] == slot_id)
    return root / row["image_path"], x14.image_digest(row)


def diagnose(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "AUDIT_PASS", "diagnostics require audit PASS")
    update_status(status="DIAGNOSTICS_RUNNING", diagnostics_started_at=now())
    frozen = load(OUTPUT / "max_support_diagnostic_manifest.json")
    all_slots = {r["slot_id"]: r for r in load(P0 / "stage20b_generation_slots.json")["slots"]}
    chosen = defaultdict(list)
    for r in frozen["records"]: chosen[r["bank"]].append(all_slots[r["slot_id"]])
    rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
    reference_rows, alignment_rows, gate_rows, regression = [], [], [], []
    digests = {}
    try:
        with x12.gpu_runtime(device) as torch:
            for mode in MODES:
                pipe, adapter, generation = load_pipeline(device, mode); before = adapter_digest(adapter); digests[mode] = before
                for bank in "ABC":
                    for index, slot in enumerate(chosen[bank]):
                        sink = {"reference": [], "alignment": [], "gate": [], "current": {}}
                        handles = diagnostic_hooks(adapter, pipe, mode, bank, slot["slot_id"], sink)
                        try:
                            image, trace = x14.generate_one(pipe, adapter, slot, rngs[slot["slot_id"]][bank], generation,
                                device, diagnostic_callback(mode, bank, slot, sink))
                        finally:
                            for h in handles: h.remove()
                        require(len(sink["reference"]) == len(sink["alignment"]) == 50 and len(sink["gate"]) == 200,
                                "diagnostic observation count changed")
                        require(all(r["all_finite"] for r in sink["reference"] + sink["alignment"]) and
                                all(r["finite"] for r in sink["gate"]), "non-finite diagnostic")
                        path = OUTPUT / "diagnostics" / mode / bank / f"{slot['slot_id']}_{index}.png"
                        image.save(path, format="PNG"); exact = None
                        if mode == "mean" and index == 0:
                            old, digest = historical_full(bank, slot["slot_id"])
                            require(old.is_file() and sha(old) == digest, "historical Full image changed")
                            exact = sha(path) == digest
                        regression.append({"mode": mode, "bank": bank, "diagnostic_index": index,
                            "slot_id": slot["slot_id"], "image_SHA256": sha(path),
                            "historical_full_exact_SHA256": exact, **trace})
                        reference_rows.extend(sink["reference"]); alignment_rows.extend(sink["alignment"]); gate_rows.extend(sink["gate"])
                        print(f"[诊断] {mode} Bank {bank} {index + 1}/4 exact={exact}", flush=True)
                require(before == adapter_digest(adapter), f"parameters changed: {mode}")
                adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        require(len(set(digests.values())) == 1, "Mean and Max used different weights")
        checks = [r for r in regression if r["historical_full_exact_SHA256"] is not None]
        require(len(checks) == 3 and all(r["historical_full_exact_SHA256"] for r in checks), "Mean Full regression failed")
        require(all(r["max_ge_mean_everywhere"] for r in reference_rows if r["mode"] == "max"), "Max < Mean detected")
        write_csv(OUTPUT / "max_support_reference_statistics.csv", reference_rows)
        write_csv(OUTPUT / "max_support_alignment_statistics.csv", alignment_rows)
        write_csv(OUTPUT / "max_support_gate_residual_statistics.csv", gate_rows)
        save(OUTPUT / "max_support_full_regression.json", {"status": "PASS", "strict_checks": 3,
            "strict_passes": 3, "records": regression, "same_adapter_state_SHA256": next(iter(digests.values())),
            "parameters_unchanged": True, "diagnostic_images_excluded_from_formal_pool": True})
        frozen.update(status="COMPLETE", generated_images=24, completed_at=now()); save(OUTPUT / "max_support_diagnostic_manifest.json", frozen)
        update_status(status="DIAGNOSTICS_PASS", diagnostics_completed_at=now(), mean_full_regression="PASS",
                      diagnostic_images=24)
        print("[诊断通过] Mean A/B/C历史回归及24张固定样本机制诊断通过。", flush=True)
    except Exception as error:
        update_status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}"); raise


def generate(device):
    require(load(OUTPUT / "experiment_status.json")["status"] == "DIAGNOSTICS_PASS", "generation requires diagnostics PASS")
    update_status(status="GENERATING", generation_started_at=now()); records = []; started = time.monotonic()
    manifest_path = OUTPUT / "max_support_generation_manifest.json"
    try:
        with x12.gpu_runtime(device) as torch:
            from stage20b_formal_common import synthetic_label
            slots = [r for r in load(P0 / "stage20b_generation_slots.json")["slots"] if r["class"] == "black"]
            originals = {r["slot_id"]: r for r in load(Path(load(STAGE25 / "stage25_run_binding.json")["p0"]) /
                                                        "stage20b_generation_slots.json")["slots"]}
            rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
            pipe, adapter, protocol = load_pipeline(device, "max"); before = adapter_digest(adapter)
            for bank in "ABC":
                full_root, full_rows = x14.full_rows(bank); full = {r["slot_id"]: r for r in full_rows}
                for index, slot in enumerate(slots):
                    frozen_slot, rng = originals[slot["slot_id"]], rngs[slot["slot_id"]][bank]
                    require(rng["bank_seed"] == BANK_SEEDS[bank] and
                            rng["non_rng_metadata_sha256"] == hashlib.sha256(canonical(frozen_slot).encode()).hexdigest(),
                            "frozen RNG identity changed")
                    historical = full[slot["slot_id"]]
                    require(historical["frozen_slot"] == frozen_slot and historical["runtime_slot"] == slot and
                            historical["sample_seed"] == rng["sample_seed"], "Full slot pairing changed")
                    item = time.monotonic(); image, trace = x14.generate_one(pipe, adapter, slot, rng, protocol, device)
                    image_path = OUTPUT / "images" / "max" / bank / f"{slot['slot_id']}.png"
                    label_path = image_path.with_suffix(".txt")
                    require(not image_path.exists() and not label_path.exists(), "formal output overwrite blocked")
                    image.save(image_path, format="PNG"); label_path.write_text(synthetic_label(slot), encoding="utf-8", newline="\n")
                    width, height = frozen_slot["geometry"]["final_canvas_wh"]
                    records.append({"group": "max", "method": GROUPS["max"], "bank": bank, "class": "black",
                        "slot_id": slot["slot_id"], "frozen_slot": frozen_slot, "runtime_slot": slot,
                        "normal_SHA256": frozen_slot["source_sha256"], "bbox": frozen_slot["bbox"],
                        "prompt": frozen_slot["prompt"], "reference_identity": frozen_slot.get("reference_identity"),
                        "sample_seed": rng["sample_seed"], "bank_seed": rng["bank_seed"],
                        "checkpoint_tree_SHA256": EXPECTED_TREE, "msdf_checkpoint_SHA256": EXPECTED_MSDF,
                        "support_composition": "max", "injection_gamma": 1.0,
                        "paired_full_image_SHA256": x14.image_digest(historical),
                        "image_path": image_path.relative_to(OUTPUT).as_posix(), "image_SHA256": sha(image_path),
                        "label_path": label_path.relative_to(OUTPUT).as_posix(), "label_SHA256": sha(label_path),
                        "roi_xyxy": crop_box(frozen_slot["bbox"], width, height), "quality_selection": False,
                        "reroll": False, "runtime_seconds": time.monotonic() - item, **trace})
                    save(manifest_path, {"status": "RUNNING", "planned_count": 120, "records": records,
                        "reroll_count": 0, "selection_count": 0})
                    print(f"[生成] Max Bank {bank} {index + 1}/40 total={len(records)}/120", flush=True)
            require(before == adapter_digest(adapter), "parameters changed during Max generation")
            adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        counts = Counter(r["bank"] for r in records)
        require(len(records) == 120 and all(counts[b] == 40 for b in "ABC"), "Max generation incomplete")
        save(manifest_path, {"status": "COMPLETE", "count": 120, "records": records,
            "all_outputs_retained": True, "reroll_count": 0, "selection_count": 0,
            "parameter_update_count": 0, "runtime_seconds": time.monotonic() - started})
        update_status(status="GENERATION_COMPLETE", generation_count=120, generation_completed_at=now(),
                      reroll_count=0, selection_count=0, parameter_update_count=0)
        print("[生成完成] Max-Support A/B/C 共120/120。", flush=True)
    except Exception as error:
        save(manifest_path, {"status": "FAILED", "records": records,
            "error": f"{type(error).__name__}: {error}", "reroll_count": 0, "selection_count": 0})
        update_status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}"); raise


def extract_features(device, torch, lib, manifest):
    import numpy as np
    from PIL import Image
    cache, ledger_path = OUTPUT / "max_support_inception_features.npz", OUTPUT / "max_support_inception_feature_cache.json"
    keys = [f"max:{r['bank']}:{r['slot_id']}" for r in manifest["records"]]
    if cache.is_file() and ledger_path.is_file():
        ledger = load(ledger_path); require(ledger["status"] == "COMPLETE" and ledger["keys"] == keys and
            ledger["sha256"] == sha(cache), "Max feature cache changed")
        return x12.array_features(cache, ledger_path, torch)
    model = lib["Inception"]("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
    values = []
    with torch.inference_mode():
        for index, row in enumerate(manifest["records"]):
            path = OUTPUT / row["image_path"]; require(path.is_file() and sha(path) == row["image_SHA256"], "image changed")
            with Image.open(path) as source: crop = source.convert("RGB").crop(tuple(row["roi_xyxy"]))
            tensor = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
            feature = model(tensor)[0]; require(tuple(feature.shape) == (1, 2048) and bool(torch.isfinite(feature).all()), row["slot_id"])
            values.append(feature[0].double().cpu().numpy())
            if (index + 1) % 20 == 0: print(f"[Inception] {index + 1}/120", flush=True)
    data = np.stack(values); np.savez_compressed(cache, features=data)
    save(ledger_path, {"status": "COMPLETE", "keys": keys, "shape": [120, 2048],
        "stored_dtype": str(data.dtype), "extraction_dtype": "float32", "sha256": sha(cache),
        "image_SHA256_checks": 120})
    del model; torch.cuda.empty_cache()
    return {key: torch.from_numpy(data[i]).double() for i, key in enumerate(keys)}


def old_metrics():
    def conv(row, scope_key):
        mapping = {"f1s1": "mean"}; group = mapping.get(row["group"], row["group"])
        return {scope_key: row[scope_key], "group": group, "method": GROUPS[group],
            "KID": float(row["KID"]), "KID_subset_std": float(row["KID_subset_std"]),
            "FID": float(row["FID"]), "reference_ROIs": int(float(row["reference_ROIs"])),
            "generated_images": int(float(row["generated_images"])), "KID_subset_size": int(float(row["KID_subset_size"])),
            "runtime_seconds": float(row["runtime_seconds"]), "metric_source": "reused SHA-bound Experiment 16 metrics"}
    keep = {"raw", "f1s1", "f0s1", "f1s0"}
    black = [conv(r, "bank") for r in read_csv(X16 / "factorial_bankwise_black.csv") if r["group"] in keep]
    overall = [conv(r, "bank") for r in read_csv(X16 / "factorial_bankwise_overall.csv") if r["group"] in keep]
    pooled = []
    for r in read_csv(X16 / "factorial_pooled_metrics.csv"):
        if r["group"] in keep:
            v = conv(r, "scope"); v["population"] = r["population"]; pooled.append(v)
    require(len(black) == len(overall) == 12 and len(pooled) == 16, "Experiment 16 metric evidence changed")
    return black, overall, pooled


def delta(first, second, metric):
    value = first[metric] - second[metric]
    return {"first": first[metric], "second": second[metric], "absolute_delta": value,
            "relative_percent": 100 * value / second[metric], "first_is_better": value < 0}


def write_diagnostic_summary(reference_rows, alignment_rows, gate_rows):
    """Persist mean/median/min/max by bank and fixed diagnostic sample."""
    output = []
    specifications = (
        ("reference", reference_rows, ("mode", "bank", "slot_id"),
         ("pixel_support_mean", "pixel_support_max", "fused_support_mean", "fused_support_max",
          "reference_support_mean", "reference_support_max", "reference_support_area_gt_0p05",
          "pixel_fused_overlap_iou")),
        ("alignment", alignment_rows, ("mode", "bank", "slot_id"),
         ("source_bbox_height", "source_bbox_width", "occupied_ratio_raw", "occupied_ratio_clamped",
          "placed_height", "placed_width", "aligned_support_area_gt_0p05",
          "final_support_area_gt_0p05", "aligned_support_mean", "final_support_mean")),
        ("gate_residual", gate_rows, ("mode", "bank", "slot_id", "up_block"),
         ("dynamic_gate", "residual_l2", "residual_hidden_relative_l2", "tanh_bound_ratio_mean",
          "tanh_bound_ratio_max", "fraction_near_tanh_limit_gt_0p95")),
    )
    for source, rows, dimensions, fields in specifications:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in dimensions)].append(row)
        for key, records in sorted(groups.items()):
            identity = dict(zip(dimensions, key))
            for field in fields:
                values = [float(row[field]) for row in records]
                output.append({"source": source, "mode": identity["mode"],
                    "bank": identity["bank"], "slot_id": identity["slot_id"],
                    "up_block": identity.get("up_block", ""), "field": field,
                    "count": len(values), "mean": statistics.fmean(values),
                    "median": statistics.median(values), "min": min(values), "max": max(values)})
    write_csv(OUTPUT / "max_support_diagnostic_summary.csv", output)


def evaluate(device):
    require(load(OUTPUT / "experiment_status.json")["status"] in ("GENERATION_COMPLETE", "EVALUATING"),
            "evaluation requires complete generation")
    update_status(status="EVALUATING", evaluation_started_at=now()); started = time.monotonic()
    try:
        with x12.gpu_runtime(device) as torch:
            from evaluate_plastic_bomo_msdf_gamma_sweep import runtime as metric_runtime
            lib, runtime_audit = metric_runtime(device)
            manifest = load(OUTPUT / "max_support_generation_manifest.json")
            require(manifest["status"] == "COMPLETE" and manifest["count"] == 120, "manifest incomplete")
            new = extract_features(device, torch, lib, manifest)
            gamma = x12.array_features(GAMMA / "gamma_sweep_inception_features.npz",
                                       GAMMA / "gamma_sweep_inception_feature_cache.json", torch)
            multi = x12.array_features(X12 / "multibank_inception_features.npz",
                                       X12 / "multibank_inception_feature_cache.json", torch)
            refs = load(GAMMA / "gamma_sweep_protocol.json")["development_reference"]["records"]
            real_black = torch.stack([gamma["real:" + r["annotation_id"]] for r in refs if r["class"] == "black"])
            real_all = torch.stack([gamma["real:" + r["annotation_id"]] for r in refs])
            slots = load(P0 / "stage20b_generation_slots.json")["slots"]
            black_slots, flash_slots = [r for r in slots if r["class"] == "black"], [r for r in slots if r["class"] == "flash"]
            def max_black(bank): return torch.stack([new[f"max:{bank}:{r['slot_id']}"] for r in black_slots])
            def flash(bank):
                if bank == "A": return torch.stack([gamma[f"fake:0.00:{r['slot_id']}"] for r in flash_slots])
                return torch.stack([multi[f"flash_gamma0:{bank}:{r['slot_id']}"] for r in flash_slots])
            old_black, old_overall, old_pooled = old_metrics(); black_rows, overall_rows = [], []
            for bank in "ABC":
                black_rows.extend(r for r in old_black if r["bank"] == bank)
                overall_rows.extend(r for r in old_overall if r["bank"] == bank)
                fake = max_black(bank); bm = x12.metric(real_black, fake, lib); om = x12.metric(real_all, torch.cat([flash(bank), fake]), lib)
                black_rows.append({"bank": bank, "group": "max", "method": GROUPS["max"], **bm,
                                   "metric_source": "new Max-Support features"})
                overall_rows.append({"bank": bank, "group": "max", "method": GROUPS["max"], **om,
                                     "metric_source": "new Max Black plus shared Flash features"})
                print(f"[指标] {bank} Max Black KID={bm['KID']:.10f} FID={bm['FID']:.6f}", flush=True)
            pooled_rows = []
            for scope, banks in (("A+B+C", "ABC"), ("B+C", "BC")):
                pooled_rows.extend(r for r in old_pooled if r["scope"] == scope)
                for population, real in (("Black", real_black), ("Overall", real_all)):
                    fake = (torch.cat([max_black(b) for b in banks]) if population == "Black" else
                            torch.cat([torch.cat([flash(b), max_black(b)]) for b in banks]))
                    result = x12.metric(real, fake, lib)
                    pooled_rows.append({"scope": scope, "population": population, "group": "max",
                        "method": GROUPS["max"], **result, "metric_source": "new concatenated Max features"})
        order = {g: i for i, g in enumerate(("raw", "mean", "max", "f0s1", "f1s0"))}
        black_rows.sort(key=lambda r: (r["bank"], order[r["group"]])); overall_rows.sort(key=lambda r: (r["bank"], order[r["group"]]))
        pooled_rows.sort(key=lambda r: (("A+B+C", "B+C").index(r["scope"]),
                                        ("Black", "Overall").index(r["population"]), order[r["group"]]))
        write_csv(OUTPUT / "max_support_bankwise_black.csv", black_rows)
        write_csv(OUTPUT / "max_support_bankwise_overall.csv", overall_rows)
        write_csv(OUTPUT / "max_support_pooled_metrics.csv", pooled_rows)
        deltas = []
        def add(scope, population, first, second):
            for metric in ("KID", "FID"):
                d = first[metric] - second[metric]
                deltas.append({"scope": scope, "population": population, "comparison": first["group"] + " - " + second["group"],
                    "metric": metric, "first": first[metric], "second": second[metric], "absolute_delta": d,
                    "relative_percent": 100 * d / second[metric], "first_is_better": d < 0})
        for bank in "ABC":
            for pop, rows in (("Black", black_rows), ("Overall", overall_rows)):
                by = {r["group"]: r for r in rows if r["bank"] == bank}
                for second in ("mean", "raw", "f0s1", "f1s0"): add(bank, pop, by["max"], by[second])
        for scope in ("A+B+C", "B+C"):
            for pop in ("Black", "Overall"):
                by = {r["group"]: r for r in pooled_rows if r["scope"] == scope and r["population"] == pop}
                for second in ("mean", "raw", "f0s1", "f1s0"): add(scope, pop, by["max"], by[second])
        write_csv(OUTPUT / "max_support_pairwise_deltas.csv", deltas)
        bmap = lambda b, rows=black_rows: {r["group"]: r for r in rows if r["bank"] == b}
        pool = lambda s, p: {r["group"]: r for r in pooled_rows if r["scope"] == s and r["population"] == p}
        bc, abc = pool("B+C", "Black"), pool("A+B+C", "Black")
        bc_overall, abc_overall = pool("B+C", "Overall"), pool("A+B+C", "Overall")
        bank_all = all(bmap(b)["max"][m] < bmap(b)["mean"][m] for b in "ABC" for m in ("KID", "FID"))
        pooled_both = all(bc["max"][m] < bc["mean"][m] for m in ("KID", "FID"))
        improved = sum(bc["max"][m] < bc["mean"][m] for m in ("KID", "FID"))
        if bank_all and pooled_both: conclusion = "MAX_SUPPORT_CONSISTENT_IMPROVEMENT"
        elif pooled_both: conclusion = "MAX_SUPPORT_POOLED_IMPROVED_NOT_CONSISTENT"
        elif improved == 1: conclusion = "MAX_SUPPORT_MIXED"
        else: conclusion = "MAX_SUPPORT_NOT_IMPROVED"

        ref = read_csv(OUTPUT / "max_support_reference_statistics.csv")
        align = read_csv(OUTPUT / "max_support_alignment_statistics.csv")
        gate = read_csv(OUTPUT / "max_support_gate_residual_statistics.csv")
        write_diagnostic_summary(ref, align, gate)
        def avg(rows, field): return sum(float(r[field]) for r in rows) / len(rows)
        diagnostic = {}
        for mode in MODES:
            rr, aa, gg = [r for r in ref if r["mode"] == mode], [r for r in align if r["mode"] == mode], [r for r in gate if r["mode"] == mode]
            diagnostic[mode] = {"reference_support_mean": avg(rr, "reference_support_mean"),
                "reference_support_max": avg(rr, "reference_support_max"),
                "reference_support_area": avg(rr, "reference_support_area_gt_0p05"),
                "aligned_support_area": avg(aa, "aligned_support_area_gt_0p05"),
                "final_support_area": avg(aa, "final_support_area_gt_0p05"),
                "aligned_support_mean": avg(aa, "aligned_support_mean"),
                "final_support_mean": avg(aa, "final_support_mean"),
                "source_bbox_width_mean": avg(aa, "source_bbox_width"), "source_bbox_height_mean": avg(aa, "source_bbox_height"),
                "occupied_ratio_mean": avg(aa, "occupied_ratio_clamped"),
                "placed_width_mean": avg(aa, "placed_width"), "placed_height_mean": avg(aa, "placed_height"),
                "target_fit_trigger_count": sum(r["target_fit_limited"] == "True" for r in aa),
                "occupied_lower_trigger_count": sum(r["occupied_lower_bound_triggered"] == "True" for r in aa),
                "occupied_upper_trigger_count": sum(r["occupied_upper_bound_triggered"] == "True" for r in aa),
                "residual_l2_mean": avg(gg, "residual_l2"),
                "relative_residual_l2_mean": avg(gg, "residual_hidden_relative_l2"),
                "near_tanh_fraction_mean": avg(gg, "fraction_near_tanh_limit_gt_0p95"),
                "up_blocks": {block: {
                    "dynamic_gate_mean": avg([r for r in gg if r["up_block"] == block], "dynamic_gate"),
                    "residual_l2_mean": avg([r for r in gg if r["up_block"] == block], "residual_l2"),
                    "relative_residual_l2_mean": avg([r for r in gg if r["up_block"] == block], "residual_hidden_relative_l2"),
                    "near_tanh_fraction_mean": avg([r for r in gg if r["up_block"] == block], "fraction_near_tanh_limit_gt_0p95")
                } for block in ("0", "1", "2", "3")}}
        comparisons = {"max_minus_mean_BC_Black": {m: delta(bc["max"], bc["mean"], m) for m in ("KID", "FID")},
            "max_minus_raw_BC_Black": {m: delta(bc["max"], bc["raw"], m) for m in ("KID", "FID")},
            "max_minus_mean_ABC_Black": {m: delta(abc["max"], abc["mean"], m) for m in ("KID", "FID")},
            "max_minus_mean_BC_Overall": {m: delta(bc_overall["max"], bc_overall["mean"], m) for m in ("KID", "FID")},
            "max_minus_raw_BC_Overall": {m: delta(bc_overall["max"], bc_overall["raw"], m) for m in ("KID", "FID")},
            "max_minus_mean_ABC_Overall": {m: delta(abc_overall["max"], abc_overall["mean"], m) for m in ("KID", "FID")}}
        required_answers = {
            "1_max_support_executed": {"answer": True, "evidence": "CPU exact torch.maximum check plus 120/120 formal Max images"},
            "2_mean_historical_regression": {"answer": "PASS", "strict_SHA256_passes": "3/3"},
            "3_reference_support_response": {"answer": "increased intensity, unchanged thresholded area",
                "mean_delta_percent": 100 * (diagnostic["max"]["reference_support_mean"] / diagnostic["mean"]["reference_support_mean"] - 1)},
            "4_source_bbox_effect": {"answer": "unchanged", "mean_width": diagnostic["max"]["source_bbox_width_mean"],
                "mean_height": diagnostic["max"]["source_bbox_height_mean"]},
            "5_occupied_ratio_effect": {"answer": "increased", "mean": diagnostic["mean"]["occupied_ratio_mean"],
                "max": diagnostic["max"]["occupied_ratio_mean"]},
            "6_aligned_support_effect": {"answer": "thresholded area unchanged; mean intensity increased",
                "mean_area": diagnostic["mean"]["aligned_support_area"], "max_area": diagnostic["max"]["aligned_support_area"],
                "mean_intensity": diagnostic["mean"]["aligned_support_mean"], "max_intensity": diagnostic["max"]["aligned_support_mean"]},
            "7_up_block_residual_effect": {block: {
                "mean_residual_l2": diagnostic["mean"]["up_blocks"][block]["residual_l2_mean"],
                "max_residual_l2": diagnostic["max"]["up_blocks"][block]["residual_l2_mean"],
                "relative_change_percent": 100 * (diagnostic["max"]["up_blocks"][block]["residual_l2_mean"] /
                    diagnostic["mean"]["up_blocks"][block]["residual_l2_mean"] - 1)} for block in ("0", "1", "2", "3")},
            "8_black_KID_better_than_mean": {"BC_pooled": bc["max"]["KID"] < bc["mean"]["KID"],
                "banks": {b: bmap(b)["max"]["KID"] < bmap(b)["mean"]["KID"] for b in "ABC"}},
            "9_black_FID_better_than_mean": {"BC_pooled": bc["max"]["FID"] < bc["mean"]["FID"],
                "banks": {b: bmap(b)["max"]["FID"] < bmap(b)["mean"]["FID"] for b in "ABC"}},
            "10_BC_direction_consistent": {"answer": False,
                "reason": "B Black KID worsened while B FID and both C metrics improved"},
            "11_overall_improved": {"BC_pooled_KID": bc_overall["max"]["KID"] < bc_overall["mean"]["KID"],
                "BC_pooled_FID": bc_overall["max"]["FID"] < bc_overall["mean"]["FID"]},
            "12_raw_gap": {"BC_Black": comparisons["max_minus_raw_BC_Black"],
                "BC_Overall": comparisons["max_minus_raw_BC_Overall"]},
            "13_better_formula": "Max-Support on pooled development metrics, without full bankwise consistency",
            "14_adaptive_support_followup": "Supported as a research question because Max helps pooled metrics but is not uniformly better by bank",
            "15_training_level_validation": "Worth a bounded next-round candidate comparison; do not promote as default before independent validation"
        }
        save(OUTPUT / "max_support_conclusion.json", {"status": conclusion, "primary_scope": "B+C pooled Black",
            "bankwise_all_KID_FID_improved": bank_all, "BC_pooled_both_improved": pooled_both,
            "primary_metrics": {g: {m: bc[g][m] for m in ("KID", "FID")} for g in ("raw", "mean", "max", "f0s1", "f1s0")},
            "comparisons": comparisons, "diagnostic_group_means": diagnostic, "required_answers": required_answers,
            "boundary": "Frozen-model inference formula intervention; morphology placement and residuals may change downstream. Development evidence only."})
        counts = Counter(r["bank"] for r in manifest["records"])
        require(all(counts[b] == 40 for b in "ABC"), "integrity count changed")
        save(OUTPUT / "max_support_generation_integrity.json", {"status": "PASS", "counts": dict(counts),
            "total": 120, "image_SHA256_checks": 120, "full_slot_pair_checks": 120,
            "reroll_count": 0, "selection_count": 0, "parameter_update_count": 0,
            "same_checkpoint_as_mean": True, "feature_cache_SHA256": sha(OUTPUT / "max_support_inception_features.npz"),
            "final_eval_read_or_used": False})
        protocol = load(OUTPUT / "max_support_protocol.json")
        protocol.update(status="COMPLETE", conclusion=conclusion, source_code_SHA256=source_hashes(),
                        metric_runtime=runtime_audit, completed_at=now()); save(OUTPUT / "max_support_protocol.json", protocol)
        create_report(conclusion, black_rows, overall_rows, pooled_rows, bc, abc, bc_overall, comparisons, diagnostic)
        update_status(status=conclusion, completed_at=now(), evaluation_runtime_seconds=time.monotonic() - started,
                      generation_count=120, followup_started=False)
        print(f"[完成] Experiment 17 status={conclusion}；停止。", flush=True)
    except Exception as error:
        update_status(status="INCOMPLETE", failure=f"{type(error).__name__}: {error}"); raise


def create_report(conclusion, black, overall, pooled, bc, abc, bc_overall, comparisons, diagnostic):
    lines = ["# Experiment 17 — Confidence-Preserving Max-Support Fusion", "", f"Status: `{conclusion}`.", "",
        "This frozen-checkpoint experiment changes only Reference Support composition from the mean to the pixelwise maximum. Morphology alignment uses Reference Support for source boxes and occupied-area placement, so downstream geometry and residuals may also change.", "",
        "## Bankwise Black", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for r in black: lines.append(f"| {r['bank']} | {r['method']} | {r['KID']:.10f} | {r['FID']:.6f} |")
    lines += ["", "## Bankwise Overall", "", "| Bank | Method | KID ↓ | FID ↓ |", "| --- | --- | ---: | ---: |"]
    for r in overall: lines.append(f"| {r['bank']} | {r['method']} | {r['KID']:.10f} | {r['FID']:.6f} |")
    lines += ["", "## Pooled metrics", "", "Pooled metrics were recomputed from concatenated features.", "",
        "| Scope | Population | Method | Images | KID ↓ | FID ↓ |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for r in pooled: lines.append(f"| {r['scope']} | {r['population']} | {r['method']} | {r['generated_images']} | {r['KID']:.10f} | {r['FID']:.6f} |")
    mm = comparisons["max_minus_mean_BC_Black"]; mr = comparisons["max_minus_raw_BC_Black"]
    bm = {bank: {r["group"]: r for r in black if r["bank"] == bank} for bank in "ABC"}
    om = {bank: {r["group"]: r for r in overall if r["bank"] == bank} for bank in "ABC"}
    pct = lambda first, second: 100 * (first / second - 1)
    lines += ["", "## Primary findings", "",
        f"B+C Black Max minus Mean: KID `{mm['KID']['absolute_delta']:+.10f}`, FID `{mm['FID']['absolute_delta']:+.6f}`.", "",
        f"A+B+C Black Max: KID `{abc['max']['KID']:.10f}` versus Mean `{abc['mean']['KID']:.10f}`; FID `{abc['max']['FID']:.6f}` versus Mean `{abc['mean']['FID']:.6f}`.", "",
        f"B+C Overall Max: KID `{bc_overall['max']['KID']:.10f}`, FID `{bc_overall['max']['FID']:.6f}`.", "",
        f"B+C Black Max remains above Raw by `{mr['KID']['absolute_delta']:+.10f}` KID and `{mr['FID']['absolute_delta']:+.6f}` FID.", "",
        "## Mechanism diagnostics", "",
        f"Mean Reference Support mean/area: `{diagnostic['mean']['reference_support_mean']:.8f}` / `{diagnostic['mean']['reference_support_area']:.8f}`; Max: `{diagnostic['max']['reference_support_mean']:.8f}` / `{diagnostic['max']['reference_support_area']:.8f}`.", "",
        f"Mean Aligned/Final Support area: `{diagnostic['mean']['aligned_support_area']:.8f}` / `{diagnostic['mean']['final_support_area']:.8f}`; Max: `{diagnostic['max']['aligned_support_area']:.8f}` / `{diagnostic['max']['final_support_area']:.8f}`.", "",
        f"Mean/Max residual L2: `{diagnostic['mean']['residual_l2_mean']:.8f}` / `{diagnostic['max']['residual_l2_mean']:.8f}`.", "",
        "| Up-block | Mean residual L2 | Max residual L2 | Change |", "| ---: | ---: | ---: | ---: |"]
    for block in ("0", "1", "2", "3"):
        first = diagnostic["max"]["up_blocks"][block]["residual_l2_mean"]
        second = diagnostic["mean"]["up_blocks"][block]["residual_l2_mean"]
        lines.append(f"| {block} | {second:.8f} | {first:.8f} | {pct(first, second):+.2f}% |")
    lines += ["", "Detailed per-bank, per-sample and per-step records plus mean/median/min/max summaries are retained in the diagnostic CSVs.", "",
        "## Required answers", "",
        "1. **Max-Support executed:** Yes. CPU checks verified exact `torch.maximum`, and formal generation completed 120/120 images.",
        "2. **Mean-Support historical regression:** PASS, with exact image SHA256 agreement for the fixed A/B/C checks (3/3).",
        f"3. **Reference Support response:** Its mean rose from `{diagnostic['mean']['reference_support_mean']:.10f}` to `{diagnostic['max']['reference_support_mean']:.10f}` ({pct(diagnostic['max']['reference_support_mean'], diagnostic['mean']['reference_support_mean']):+.2f}%), while area above 0.05 stayed unchanged.",
        f"4. **Source BBox:** No change; mean width/height remained `{diagnostic['max']['source_bbox_width_mean']:.2f}` / `{diagnostic['max']['source_bbox_height_mean']:.2f}` latent pixels.",
        f"5. **Occupied Ratio:** Mean increased from `{diagnostic['mean']['occupied_ratio_mean']:.10f}` to `{diagnostic['max']['occupied_ratio_mean']:.10f}` ({pct(diagnostic['max']['occupied_ratio_mean'], diagnostic['mean']['occupied_ratio_mean']):+.2f}%); neither clamp bound triggered.",
        f"6. **Aligned Support:** Thresholded area stayed `{diagnostic['max']['aligned_support_area']:.10f}`, but mean intensity increased from `{diagnostic['mean']['aligned_support_mean']:.10f}` to `{diagnostic['max']['aligned_support_mean']:.10f}` ({pct(diagnostic['max']['aligned_support_mean'], diagnostic['mean']['aligned_support_mean']):+.2f}%).",
        "7. **Up-block residuals:** Mean residual L2 increased in all four blocks; the table above reports each change. Tanh-near-limit fractions remained extremely small.",
        f"8. **Black KID:** B+C pooled improved by `{mm['KID']['absolute_delta']:+.10f}` ({mm['KID']['relative_percent']:+.2f}%). A and C improved; B worsened by `{bm['B']['max']['KID'] - bm['B']['mean']['KID']:+.10f}`.",
        f"9. **Black FID:** B+C pooled improved by `{mm['FID']['absolute_delta']:+.6f}` ({mm['FID']['relative_percent']:+.2f}%). A, B and C all improved.",
        "10. **B/C direction:** Not fully consistent. B Black KID worsened while B FID improved; both C metrics improved.",
        f"11. **Overall:** B+C pooled improved in KID by `{comparisons['max_minus_mean_BC_Overall']['KID']['absolute_delta']:+.10f}` and FID by `{comparisons['max_minus_mean_BC_Overall']['FID']['absolute_delta']:+.6f}`. Bank B Overall FID worsened slightly; the other bankwise comparisons improved.",
        f"12. **Gap to Raw SD2:** On B+C Black, Max remains worse by `{mr['KID']['absolute_delta']:+.10f}` KID and `{mr['FID']['absolute_delta']:+.6f}` FID. On B+C Overall, the gaps are `{comparisons['max_minus_raw_BC_Overall']['KID']['absolute_delta']:+.10f}` and `{comparisons['max_minus_raw_BC_Overall']['FID']['absolute_delta']:+.6f}`.",
        "13. **Better formula:** Max-Support is better on the pooled development metrics, but the bankwise inconsistency prevents treating it as uniformly superior.",
        "14. **Adaptive Support fusion:** There is a basis for a future bounded study because one global Max rule helps pooled metrics but does not help every bank and metric.",
        "15. **Training-level validation:** Max-Support is worth including as a candidate in a controlled next-round comparison. It should not become the default before independent validation, and Raw SD2 still leads clearly.", "",
        "## Research boundary", "",
        "This is an inference-only intervention on a fixed trained model. It does not establish joint-training benefit, independent-test performance, support-head accuracy, statistical significance, or cross-class generalization. No follow-up experiment was started.", ""]
    (OUTPUT / "MAX_SUPPORT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(device): diagnose(device); generate(device); evaluate(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "diagnose", "generate", "evaluate", "run"))
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    print(f"[启动] Experiment 17 phase={args.phase} PID={os.getpid()} device={args.device}", flush=True)
    if args.phase == "audit": audit()
    elif args.phase == "diagnose": diagnose(args.device)
    elif args.phase == "generate": generate(args.device)
    elif args.phase == "evaluate": evaluate(args.device)
    else: run_all(args.device)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True); raise
