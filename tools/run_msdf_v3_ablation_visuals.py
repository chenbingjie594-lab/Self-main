#!/usr/bin/env python3
"""Generate the fixed eight-slot MSDF-v3 training-ablation visual comparison."""

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
P0 = REPO / "results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"
VARIANTS = {
    "full": (REPO / "model/msdf_v3_sd2_2000_s42", "full"),
    "no_latent": (REPO / "model/msdf_v3_no_latent_sd2_2000_s42", "no_latent"),
    "no_morph": (REPO / "model/msdf_v3_no_morph_sd2_2000_s42", "no_morph"),
}
CLASS_DIR = {"flash": "01_Flash_point", "black": "02_Big_black_spots"}
BANK_SEQUENCE = ("A", "B", "C", "A")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checkpoint_files(root):
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def checkpoint_binding(root):
    files = checkpoint_files(root)
    require(files, f"empty checkpoint: {root}")
    rows = {path.relative_to(root).as_posix(): sha(path) for path in files}
    return {
        "root_name": root.name,
        "file_count": len(rows),
        "files": rows,
        "tree_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
    }


def choose_slots(slots, rngs):
    selected = []
    for cls in ("flash", "black"):
        rows = [row for row in slots if row["class"] == cls]
        require(len(rows) == 40, f"expected 40 frozen {cls} slots")
        indices = [round(x) for x in (0, (len(rows) - 1) / 3, 2 * (len(rows) - 1) / 3, len(rows) - 1)]
        require(indices == [0, 13, 26, 39], indices)
        for rank, (index, bank) in enumerate(zip(indices, BANK_SEQUENCE)):
            row = rows[index]
            selected.append({"class": cls, "rank": rank, "source_index": index,
                             "bank": bank, "slot": row, "rng": rngs[row["slot_id"]][bank]})
    return selected


def input_binding(item, output):
    import numpy as np
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    slot = item["slot"]
    normal, mask = crop_and_mask(slot, use_normal=True)
    reference, reference_mask = crop_and_mask(
        {**slot, "path": slot["source_path"], "sha256": slot["source_sha256"]})
    require(np.array_equal(np.asarray(mask), np.asarray(reference_mask)),
            f"normal/reference masks differ: {slot['slot_id']}")
    root = output / "inputs" / item["bank"] / slot["slot_id"]
    root.mkdir(parents=True, exist_ok=False)
    paths = {"normal_crop": root / "normal_crop.png", "reference_crop": root / "reference_crop.png",
             "mask": root / "mask.png"}
    normal.save(paths["normal_crop"], format="PNG")
    reference.save(paths["reference_crop"], format="PNG")
    mask.save(paths["mask"], format="PNG")
    for path in (Path(slot["normal_path"]), Path(slot["source_path"])):
        require(path.is_file(), f"missing source: {path}")
    require(sha(slot["normal_path"]) == slot["normal_sha256"], "normal source SHA mismatch")
    require(sha(slot["source_path"]) == slot["source_sha256"], "reference source SHA mismatch")
    # Stage20B binds decoded masks as JSON-encoded dimensions followed by L bytes.
    mask_pixels_sha = hashlib.sha256(json.dumps([512, 512]).encode() + mask.tobytes()).hexdigest()
    require(mask_pixels_sha == slot["geometry"]["mask_decoded_sha256"],
            f"decoded mask SHA mismatch: {slot['slot_id']}")
    return {
        "normal_path": slot["normal_path"], "normal_sha256": slot["normal_sha256"],
        "reference_path": slot["source_path"], "reference_sha256": slot["source_sha256"],
        "normal_input_path": str(paths["normal_crop"].resolve()),
        "normal_input_sha256": sha(paths["normal_crop"]),
        "reference_input_path": str(paths["reference_crop"].resolve()),
        "reference_input_sha256": sha(paths["reference_crop"]),
        "mask_input_path": str(paths["mask"].resolve()), "mask_input_sha256": sha(paths["mask"]),
        "mask_decoded_sha256": mask_pixels_sha,
    }


def roi_xyxy(slot):
    width, height = slot["geometry"]["final_canvas_wh"]
    cx, cy, bw, bh = slot["bbox"]
    return [max(0, math.floor((cx - bw / 2) * width)),
            max(0, math.floor((cy - bh / 2) * height)),
            min(width, math.ceil((cx + bw / 2) * width)),
            min(height, math.ceil((cy + bh / 2) * height))]


def preflight(output, device):
    import torch
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    protocol = load(P0 / "stage20b_generation_protocol.json")
    report = {"status": "RUNNING", "device": device, "checkpoints": {},
              "stage20b": {"slots_sha256": sha(P0 / "stage20b_generation_slots.json"),
                           "rng_sha256": sha(P0 / "stage20b_rng_manifest.json"),
                           "protocol_sha256": sha(P0 / "stage20b_generation_protocol.json")}}
    save(output / "checkpoint_preflight.json", report)
    for variant, (model_root, expected) in VARIANTS.items():
        report["checkpoints"][variant] = {}
        for cls, dirname in CLASS_DIR.items():
            root = model_root / "Plastic_Bomo" / dirname
            binding = checkpoint_binding(root)
            checkpoint = torch.load(root / "msdf.pt", map_location="cpu", weights_only=False)
            stored = checkpoint.get("training_ablation", "full" if checkpoint.get(
                "morphology_alignment_enabled", True) else "no_morph")
            require(stored == expected, f"{variant}/{cls}: expected {expected}, got {stored}")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(root), local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
                safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
            pipe.scheduler = DDIMScheduler.from_config(protocol["scheduler_config"])
            adapter = load_msdf_adapter(root / "msdf.pt", pipe.unet, device)
            norms = {name: float(value.float().norm()) for name, value in checkpoint["state_dict"].items()
                     if name.startswith("projections.") and name.endswith(".weight")}
            require(len(norms) == 4 and all(value > 0 for value in norms.values()),
                    f"{variant}/{cls}: inactive projections")
            report["checkpoints"][variant][cls] = {
                "status": "PASS", "training_ablation": stored, "binding": binding,
                "msdf_sha256": sha(root / "msdf.pt"), "projection_weight_norms": norms,
                "strict_runtime_load": True,
            }
            adapter.detach(); del adapter, pipe, checkpoint
            gc.collect(); torch.cuda.empty_cache()
            save(output / "checkpoint_preflight.json", report)
            print(f"[预检] {variant}/{cls} PASS", flush=True)
    report["status"] = "PASS"
    save(output / "checkpoint_preflight.json", report)
    return report


def generate(output, device, preflight_report):
    import numpy as np
    import torch
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    from generate_plastic_bomo_stage20b_banks import sample
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    protocol = load(P0 / "stage20b_generation_protocol.json")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rngs = {row["slot_id"]: row["banks"] for row in load(P0 / "stage20b_rng_manifest.json")["records"]}
    selected = choose_slots(slots, rngs)
    selection = {"status": "FROZEN_BEFORE_GENERATION",
                 "rule": "original Stage20B class order; evenly spaced indices 0,13,26,39; banks A,B,C,A by rank; no image inspection",
                 "records": [{k: item[k] for k in ("class", "rank", "source_index", "bank")} |
                             {"slot_id": item["slot"]["slot_id"], "sample_seed": item["rng"]["sample_seed"],
                              "bank_seed": item["rng"]["bank_seed"]} for item in selected]}
    save(output / "fixed_selection.json", selection)
    inputs = {item["slot"]["slot_id"] + ":" + item["bank"]: input_binding(item, output) for item in selected}
    parameters = {key: protocol[key] for key in ("num_inference_steps", "guidance_scale", "scheduler",
        "scheduler_config", "eta", "strength", "height", "width", "num_images_per_prompt",
        "negative_prompt", "clip_skip", "padding_mask_crop", "precision", "output_encoding")}
    manifests = {}
    bypass = {"status": "RUNNING", "definition": "same checkpoint/input/protocol/seed; injection_gamma 1 versus 0", "records": []}
    save(output / "checks/active_vs_bypass.json", bypass)
    for variant, (model_root, expected) in VARIANTS.items():
        manifest_path = output / "generation" / variant / "formal_manifest.json"
        records = []
        save(manifest_path, {"status": "RUNNING", "records": records})
        for cls, dirname in CLASS_DIR.items():
            root = model_root / "Plastic_Bomo" / dirname
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(root), local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
                safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
            pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
            pipe.disable_xformers_memory_efficient_attention()
            pipe.scheduler = DDIMScheduler.from_config(protocol["scheduler_config"])
            adapter = load_msdf_adapter(root / "msdf.pt", pipe.unet, device)
            require(adapter.training_ablation == expected, f"runtime ablation mismatch: {variant}/{cls}")
            for rank, item in enumerate(row for row in selected if row["class"] == cls):
                slot, rng = item["slot"], item["rng"]
                require(hashlib.sha256(canonical(slot).encode()).hexdigest() == rng["non_rng_metadata_sha256"],
                        f"frozen slot/RNG mismatch: {slot['slot_id']}")
                donor, mask = crop_and_mask({**slot, "path": slot["source_path"], "sha256": slot["source_sha256"]})
                pixels = torch.from_numpy(np.asarray(donor).copy()).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
                mask_t = torch.from_numpy(np.asarray(mask).copy())[None, None].to(device).float() / 255
                with torch.inference_mode():
                    reference = pipe.vae.encode(pixels.half()).latent_dist.mode() * pipe.vae.config.scaling_factor

                def prepare(_module, args, kwargs):
                    inp = args[0] if args else kwargs["sample"]
                    timestep = args[1] if len(args) > 1 else kwargs["timestep"]
                    require(list(inp.shape)[1:] == [9, 64, 64] and inp.shape[0] == 2, "invalid CFG inference input")
                    adapter.prepare(reference.float(), mask_t, inp[:1, 4:5].float(), timestep,
                                    pipe.scheduler.config.num_train_timesteps, True, pixels)

                handle = pipe.unet.register_forward_pre_hook(prepare, with_kwargs=True)
                try:
                    adapter.set_injection_gamma(1.0)
                    started = time.monotonic()
                    image, trace = sample(pipe, slot, rng, protocol, device)
                    elapsed = time.monotonic() - started
                    image_path = output / "generation" / variant / "images" / item["bank"] / (slot["slot_id"] + ".png")
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    require(not image_path.exists(), f"refusing overwrite: {image_path}")
                    image.save(image_path, format="PNG")
                    binding = inputs[slot["slot_id"] + ":" + item["bank"]]
                    record = {"variant": variant, "bank": item["bank"], "class": cls, "slot_id": slot["slot_id"],
                              "frozen_slot": slot, "runtime_slot": slot, **binding, **trace,
                              "mask_sha256": binding["mask_decoded_sha256"], "roi_xyxy": roi_xyxy(slot),
                              "image_path": str(image_path.resolve()), "image_sha256": sha(image_path),
                              "checkpoint_tree_sha256": preflight_report["checkpoints"][variant][cls]["binding"]["tree_sha256"],
                              "msdf_checkpoint_sha256": preflight_report["checkpoints"][variant][cls]["msdf_sha256"],
                              "generation_protocol_sha256": sha(P0 / "stage20b_generation_protocol.json"),
                              "inference_parameters": parameters, "injection_gamma": 1.0,
                              "runtime_seconds": elapsed, "quality_selection": False, "reroll": False}
                    records.append(record)
                    save(manifest_path, {"status": "RUNNING", "variant": variant, "records": records})
                    print(f"[生成] {variant}/{cls} {rank + 1}/4 {item['bank']} {slot['slot_id']}", flush=True)
                    if rank == 0:
                        adapter.set_injection_gamma(0.0)
                        bypass_image, bypass_trace = sample(pipe, slot, rng, protocol, device)
                        bypass_path = output / "checks" / "bypass_images" / variant / (cls + ".png")
                        bypass_path.parent.mkdir(parents=True, exist_ok=True)
                        bypass_image.save(bypass_path, format="PNG")
                        active_array = np.asarray(image, dtype=np.int16)
                        bypass_array = np.asarray(bypass_image, dtype=np.int16)
                        delta = np.abs(active_array - bypass_array)
                        check = {"variant": variant, "class": cls, "slot_id": slot["slot_id"], "bank": item["bank"],
                                 "active_image_sha256": sha(image_path), "bypass_image_sha256": sha(bypass_path),
                                 "same_seed_rng_after": trace["RNG_after_sha256"] == bypass_trace["RNG_after_sha256"],
                                 "changed_pixels": int(np.count_nonzero(delta)), "mean_abs_uint8": float(delta.mean()),
                                 "max_abs_uint8": int(delta.max()), "active_differs_from_bypass": bool(np.any(delta))}
                        require(check["same_seed_rng_after"] and check["active_differs_from_bypass"],
                                f"active/bypass ineffective: {variant}/{cls}")
                        bypass["records"].append(check)
                        save(output / "checks/active_vs_bypass.json", bypass)
                finally:
                    handle.remove()
            adapter.detach(); del adapter, pipe
            gc.collect(); torch.cuda.empty_cache()
        require(len(records) == 8, f"{variant}: expected 8 images")
        manifest = {"status": "COMPLETE", "variant": variant, "count": 8, "records": records,
                    "fixed_selection_sha256": sha(output / "fixed_selection.json"), "all_outputs_retained": True,
                    "quality_selection": False, "reroll_count": 0}
        save(manifest_path, manifest); manifests[variant] = manifest_path
    bypass["status"] = "PASS"
    require(len(bypass["records"]) == 6 and all(row["active_differs_from_bypass"] for row in bypass["records"]),
            "six active/bypass checks required")
    save(output / "checks/active_vs_bypass.json", bypass)
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "runs/msdf_v3_ablation/visual_evaluation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(not args.output.exists(), f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    status = {"status": "RUNNING", "started_at_unix": time.time(), "training_count": 0,
              "kid_fid_count": 0, "detector_training_count": 0}
    save(args.output / "status.json", status)
    try:
        report = preflight(args.output, args.device)
        manifests = generate(args.output, args.device, report)
        status.update(status="GENERATION_COMPLETE", completed_at_unix=time.time(),
                      generated_images=24, bypass_diagnostic_images=6,
                      manifests={key: str(value) for key, value in manifests.items()})
        save(args.output / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}", failed_at_unix=time.time())
        save(args.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
