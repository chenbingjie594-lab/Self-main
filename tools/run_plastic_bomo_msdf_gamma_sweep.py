"""Prepare and generate the frozen Bank-A MSDF final-residual gamma sweep.

This script never trains.  It reuses the frozen Stage25 class checkpoints,
Bank-A slots/RNG, and V2 TRAIN allowlist.  Gamma is applied only after the
existing per-up-block tanh residual bound.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
SOURCE_RUN = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214"
P0 = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/server_runtime_20261013_080546_552698/p0_view"
GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASSES = ("flash", "black")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gamma_tag(gamma):
    return f"gamma_{gamma:.2f}".replace(".", "p")


def tree_digest(path):
    files = {p.relative_to(path).as_posix(): sha(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), files


def crop_box(bbox, width, height, margin=0.25):
    cx, cy, bw, bh = bbox
    x0 = (cx - (0.5 + margin) * bw) * width
    x1 = (cx + (0.5 + margin) * bw) * width
    y0 = (cy - (0.5 + margin) * bh) * height
    y1 = (cy + (0.5 + margin) * bh) * height
    result = [max(0, math.floor(x0)), max(0, math.floor(y0)),
              min(width, math.ceil(x1)), min(height, math.ceil(y1))]
    require(result[2] > result[0] and result[3] > result[1], f"empty ROI: {bbox}")
    return result


def source_audit():
    adapter = REPO / "diffusers/pipelines/stable_diffusion/msdf_guidance.py"
    text = adapter.read_text(encoding="utf-8")
    required = ("defect_residual =", "pixel_residual =", "gates = torch.sigmoid(gate_logits)",
                "limit * torch.tanh", "if self.injection_gamma != 1.0", "hidden + residual")
    require(all(token in text for token in required), "MSDF injection implementation is incomplete")
    paths = [REPO / name for name in (
        "train_rda.py", "inference.py",
        "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_magic.py",
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py",
        "tools/stage25_msdf_generator.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "tools/run_plastic_bomo_msdf_gamma_sweep.py",
        "tools/evaluate_plastic_bomo_msdf_gamma_sweep.py",
    )]
    return {p.relative_to(REPO).as_posix(): sha(p) for p in paths}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "gamma_sweep_protocol.json"
    if existing.exists():
        print("[准备] 已存在冻结协议，保持不变。", flush=True)
        return load(existing)
    require(not any(output.iterdir()), f"refusing non-empty unprepared output: {output}")
    binding = load(SOURCE_RUN / "stage25_run_binding.json")
    frozen = load(SOURCE_RUN / "generator_checkpoint_binding.json")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rng_records = load(P0 / "stage20b_rng_manifest.json")["records"]
    generation = load(P0 / "stage20b_generation_protocol.json")
    allowlist_path = P0 / "generator_train_allowlist.json"
    allowlist = load(allowlist_path)["inputs"]
    require(len(slots) == 80 and Counter(x["class"] for x in slots) == {"flash": 40, "black": 40}, "bad slots")
    require(len(allowlist) == 172 and Counter(x["class"] for x in allowlist) == {"flash": 75, "black": 97}, "bad TRAIN reference")
    references = []
    for row in sorted(allowlist, key=lambda x: x["annotation_id"]):
        path = Path(row["path"])
        require(row["split"] == "train" and path.is_file() and sha(path) == row["sha256"], f"reference changed: {path}")
        width, height = row["geometry"]["original_canvas_wh"]
        references.append({
            "annotation_id": row["annotation_id"], "class": row["class"],
            "canonical_image_id": row["canonical_image_id"], "source_group_id": row["source_group_id"],
            "image_path": str(path), "image_sha256": row["sha256"], "canvas_wh": [width, height],
            "bbox": row["geometry"]["final_bbox_normalized_xywh"],
            "roi_xyxy": crop_box(row["geometry"]["final_bbox_normalized_xywh"], width, height),
        })
    checkpoints = {}
    for cls in CLASSES:
        row = frozen["classes"][cls]["checkpoint"]
        path = SOURCE_RUN / row["path"]
        digest, files = tree_digest(path)
        require(digest == row["tree_sha256"] and files == row["sha256"], f"checkpoint changed: {cls}")
        checkpoints[cls] = {"path": str(path), "tree_sha256": digest,
                            "msdf_sha256": files["msdf.pt"], "successful_updates": frozen["classes"][cls]["successful_updates"]}
    protocol = {
        "status": "FROZEN_BEFORE_GENERATION", "created_at": now(),
        "objective": "exploratory MSDF final bounded residual injection strength sweep",
        "gamma_values": list(GAMMAS), "gamma_location": "after tanh bound and nan_to_num, immediately before hidden + residual",
        "gamma_1_behavior": "conditional multiply skipped; original floating-point path preserved",
        "only_control_variable": "MSDF final residual injection gamma",
        "training_performed": False, "bank": "A", "slot_count_per_gamma": 80,
        "same_slots_normal_bbox_prompt_seed_steps_for_all_gamma": True,
        "slot_class_counts": dict(Counter(x["class"] for x in slots)),
        "generation_protocol": generation,
        "generation_protocol_sha256": sha(P0 / "stage20b_generation_protocol.json"),
        "slots_sha256": sha(P0 / "stage20b_generation_slots.json"),
        "rng_manifest_sha256": sha(P0 / "stage20b_rng_manifest.json"),
        "source_bank_A_manifest_sha256": sha(SOURCE_RUN / "formal_bank_A_manifest.json"),
        "checkpoints": checkpoints,
        "source_run": str(SOURCE_RUN), "source_run_binding_sha256": sha(SOURCE_RUN / "stage25_run_binding.json"),
        "base_model": binding["base_model"],
        "development_reference": {
            "split": "V2 TRAIN", "image_role": "all original annotated defect ROIs in generator TRAIN allowlist",
            "count": len(references), "class_counts": dict(Counter(x["class"] for x in references)),
            "allowlist_sha256": sha(allowlist_path),
            "checkpoint_training_overlap_possible": True,
            "independent_validation": False,
            "selection_use": "exploratory screening only",
            "final_eval_read_or_used": False,
            "roi_crop": "normalized bbox expanded 25% on each side; floor/ceil and image-boundary clip",
            "records": references,
        },
        "metrics": {
            "ROI_KID": "primary; torch-fidelity polynomial KID; 100 subsets; size min(50,nreal,nfake); seed42",
            "ROI_FID": "auxiliary; torch-fidelity Inception-v3-compat 2048",
            "LPIPS": "Bank-A exploratory diversity reference: official AlexNet v0.1 over deterministic adjacent pairs within each class after sorting slot_id; 20 disjoint pairs/class; ROI resize256x256",
            "historical_final_eval_scores_comparable": False,
        },
        "implementation_sha256": source_audit(),
    }
    save(output / "development_reference_manifest.json", {"status": "FROZEN", "records": references})
    save(output / "gamma_sweep_protocol.json", protocol)
    save(output / "gamma_sweep_status.json", {
        "status": "PREPARED", "started_at": now(), "full_experiment_complete": False,
        "training_count": 0, "final_eval_content_read": 0, "gamma_values": list(GAMMAS),
        "output_image_count": 0, "sampling_forward_count": 0,
    })
    print("[准备完成] V2 TRAIN ROI=172；Bank A slots=80；未读取 final_eval。", flush=True)
    return protocol


def cpu_regression(output):
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter, _safe_rms
    torch.manual_seed(20261020)
    adapter = MSDFAdapter(4, 8, [4], max_residual_ratio=0.25).eval()
    with torch.no_grad():
        adapter.projections[0].weight.fill_(0.03125)
        adapter.projections[0].bias.fill_(0.0078125)
    adapter._feature = torch.randn(1, 8, 3, 3)
    adapter._support = torch.sigmoid(torch.randn(1, 1, 3, 3))
    adapter._gates = torch.tensor([[0.37]])
    adapter._cfg = False
    hidden = torch.randn(1, 4, 6, 6)
    feature = torch.nn.functional.interpolate(adapter._feature, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
    low = torch.nn.functional.avg_pool2d(feature, 3, stride=1, padding=1)
    feature = low + 0.25 * (feature - low)
    support = torch.nn.functional.interpolate(adapter._support, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
    residual = adapter.projections[0](feature)
    residual = torch.nan_to_num(residual) * support
    residual = residual * adapter._gates[:, 0].reshape(-1, 1, 1, 1)
    limit = adapter.max_residual_ratio * _safe_rms(hidden, (1, 2, 3)).reshape(-1, 1, 1, 1).clamp_min(1e-4)
    old = hidden + torch.nan_to_num(limit * torch.tanh(residual.float() / limit.clamp_min(1e-6))).to(hidden.dtype)
    hook = adapter._make_hook(0)
    adapter.set_injection_gamma(1.0)
    new = hook(None, None, hidden.clone())
    adapter.set_injection_gamma(0.5)
    half = hook(None, None, hidden.clone())
    adapter.set_injection_gamma(0.0)
    zero = hook(None, None, hidden.clone())
    record = {
        "status": "CPU_HOOK_PASS", "gamma_1_bitwise_equal_to_pre_interface_formula": bool(torch.equal(old, new)),
        "gamma_1_max_abs_difference": float((old-new).abs().max()),
        "gamma_0_bitwise_equal_hidden": bool(torch.equal(zero, hidden)),
        "gamma_0_max_abs_difference": float((zero-hidden).abs().max()),
        "gamma_0p5_final_contribution_max_error": float(((half-hidden)-0.5*(new-hidden)).abs().max()),
        "end_to_end": "PENDING", "checked_at": now(),
    }
    require(record["gamma_1_bitwise_equal_to_pre_interface_formula"] and record["gamma_0_bitwise_equal_hidden"]
            and record["gamma_0p5_final_contribution_max_error"] < 1e-6, f"CPU regression failed: {record}")
    save(output / "gamma_regression_check.json", record)
    print("[回归] CPU hook：gamma=1 逐位一致；gamma=0 原样返回；gamma=.5 作用于最终残差。", flush=True)


def configure_runtime(torch, device):
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def load_pipeline(checkpoint, generation, device, gamma):
    import torch
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    expected_pipe = REPO / "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py"
    require(Path(inspect.getfile(StableDiffusionInpaintPipeline)).resolve() == expected_pipe, "pipeline import shadowing")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    pipe.disable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(generation["scheduler_config"])
    for key, value in generation["scheduler_config"].items():
        require(getattr(pipe.scheduler.config, key) == value, f"scheduler changed: {key}")
    adapter = load_msdf_adapter(Path(checkpoint) / "msdf.pt", pipe.unet, device).eval()
    adapter.set_injection_gamma(gamma)
    return pipe, adapter


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
    def prepare_hook(_module, args, kwargs):
        inp = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        require(list(inp.shape)[1:] == [9, 64, 64] and inp.shape[0] == 2, "unexpected inference tensor")
        adapter.prepare(reference.float(), mask_t, inp[:1, 4:5].float(), timestep,
                        pipe.scheduler.config.num_train_timesteps, True, reference_pixels=pixels)
    handle = pipe.unet.register_forward_pre_hook(prepare_hook, with_kwargs=True)
    try:
        return sample(pipe, slot, rng, generation, device)
    finally:
        handle.remove()


def generate(output, device):
    import torch
    configure_runtime(torch, device)
    protocol = load(output / "gamma_sweep_protocol.json")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rngs = {r["slot_id"]: r["banks"]["A"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
    generation = load(P0 / "stage20b_generation_protocol.json")
    old_manifest = load(SOURCE_RUN / "formal_bank_A_manifest.json")
    old_by_slot = {r["slot_id"]: r for r in old_manifest["records"]}
    status = load(output / "gamma_sweep_status.json")
    status.update(status="GENERATING", device=device, generation_started_at=now())
    save(output / "gamma_sweep_status.json", status)
    regression = load(output / "gamma_regression_check.json")

    # Two end-to-end samples cover both frozen class checkpoints.  Exact PNG
    # equality permits reuse of all already-frozen gamma=1 Bank-A artifacts.
    if regression.get("end_to_end") != "PASS":
        rows = []
        for cls in CLASSES:
            checkpoint = Path(protocol["checkpoints"][cls]["path"])
            pipe, adapter = load_pipeline(checkpoint, generation, device, 1.0)
            slot = next(x for x in slots if x["class"] == cls)
            image, trace = generate_one(pipe, adapter, slot, rngs[slot["slot_id"]], generation, device)
            temp = output / "regression" / (slot["slot_id"] + ".png")
            temp.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp, format="PNG")
            old = SOURCE_RUN / old_by_slot[slot["slot_id"]]["image_path"]
            rows.append({"class": cls, "slot_id": slot["slot_id"], "new_sha256": sha(temp),
                         "frozen_original_sha256": sha(old), "exact_png_equal": sha(temp) == sha(old), **trace})
            adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        require(all(x["exact_png_equal"] for x in rows), f"gamma=1 end-to-end regression failed: {rows}")
        regression.update(end_to_end="PASS", end_to_end_records=rows,
                          gamma_1_all_assets_reused_after_two_class_exact_regression=True)
        save(output / "gamma_regression_check.json", regression)
        status["sampling_forward_count"] = status.get("sampling_forward_count", 0) + 2
        save(output / "gamma_sweep_status.json", status)
        print("[回归] gamma=1 两类端到端 PNG 与冻结 MSDF Bank A 完全一致。", flush=True)

    for gamma in (1.0, 0.0, 0.25, 0.50, 0.75):
        tag = gamma_tag(gamma)
        root = output / "images" / tag
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = output / f"{tag}_manifest.json"
        records = load(manifest_path).get("records", []) if manifest_path.exists() else []
        completed = {r["slot_id"]: r for r in records if (output / r["image_path"]).is_file()}
        started = time.monotonic()
        if gamma == 1.0:
            for slot in slots:
                old_row = old_by_slot[slot["slot_id"]]
                source = SOURCE_RUN / old_row["image_path"]
                target = root / (slot["slot_id"] + ".png")
                if not target.exists():
                    shutil.copy2(source, target)
                require(sha(target) == old_row["image_sha256"], f"gamma1 copy changed: {target}")
                completed[slot["slot_id"]] = {
                    "slot_id": slot["slot_id"], "class": slot["class"], "gamma": gamma,
                    "image_path": target.relative_to(output).as_posix(), "image_sha256": sha(target),
                    "source": "reused frozen gamma=1 Stage25 Bank-A image after exact two-class regression",
                    "sample_seed": rngs[slot["slot_id"]]["sample_seed"], "bank_seed": rngs[slot["slot_id"]]["bank_seed"],
                    "roi_xyxy": crop_box(slot["bbox"], *slot["geometry"]["final_canvas_wh"]),
                }
        else:
            for cls in CLASSES:
                checkpoint = Path(protocol["checkpoints"][cls]["path"])
                pipe, adapter = load_pipeline(checkpoint, generation, device, gamma)
                selected = [x for x in slots if x["class"] == cls]
                for slot in selected:
                    if slot["slot_id"] in completed:
                        continue
                    before = time.monotonic()
                    image, trace = generate_one(pipe, adapter, slot, rngs[slot["slot_id"]], generation, device)
                    target = root / (slot["slot_id"] + ".png")
                    image.save(target, format="PNG")
                    completed[slot["slot_id"]] = {
                        "slot_id": slot["slot_id"], "class": cls, "gamma": gamma,
                        "image_path": target.relative_to(output).as_posix(), "image_sha256": sha(target),
                        "source": "new frozen Bank-A sampling", "sample_seed": rngs[slot["slot_id"]]["sample_seed"],
                        "bank_seed": rngs[slot["slot_id"]]["bank_seed"],
                        "roi_xyxy": crop_box(slot["bbox"], *slot["geometry"]["final_canvas_wh"]),
                        "runtime_seconds": time.monotonic() - before, **trace,
                    }
                    ordered = [completed[x["slot_id"]] for x in slots if x["slot_id"] in completed]
                    save(manifest_path, {"status": "RUNNING", "gamma": gamma, "count": len(ordered), "records": ordered})
                    status["sampling_forward_count"] = status.get("sampling_forward_count", 0) + 1
                    status["output_image_count"] = sum(load(output / f"{gamma_tag(g)}_manifest.json").get("count", 0)
                                                       for g in GAMMAS if (output / f"{gamma_tag(g)}_manifest.json").exists())
                    save(output / "gamma_sweep_status.json", status)
                    print(f"[生成] {tag} {len(ordered)}/80 {slot['slot_id']} {time.monotonic()-before:.1f}s", flush=True)
                adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        ordered = [completed[x["slot_id"]] for x in slots]
        require(len(ordered) == 80 and Counter(x["class"] for x in ordered) == {"flash": 40, "black": 40}, f"incomplete {tag}")
        save(manifest_path, {"status": "COMPLETE", "gamma": gamma, "count": 80,
                            "runtime_seconds_this_invocation": time.monotonic()-started,
                            "quality_selection": False, "reroll_count": 0, "records": ordered})
        print(f"[完成] {tag}: 80/80", flush=True)
    status = load(output / "gamma_sweep_status.json")
    status.update(status="GENERATION_COMPLETE", output_image_count=400, generation_completed_at=now())
    save(output / "gamma_sweep_status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "regression", "generate", "all"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output == OUTPUT.resolve(), "only the frozen gamma-sweep output directory is allowed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    print(f"[启动] MSDF gamma sweep mode={args.mode} PID={os.getpid()}；不训练。", flush=True)
    if args.mode in ("prepare", "all"):
        prepare(output)
    if args.mode in ("regression", "all"):
        require((output / "gamma_sweep_protocol.json").is_file(), "run prepare first")
        cpu_regression(output)
    if args.mode in ("generate", "all"):
        require(load(output / "gamma_regression_check.json")["status"] == "CPU_HOOK_PASS", "run regression first")
        generate(output, args.device)
    print(f"[退出] mode={args.mode} 完成。", flush=True)


if __name__ == "__main__":
    main()
