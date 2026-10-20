#!/usr/bin/env python3
"""Generate the frozen Plastic_Bomo Exp2 AnomalyDiffusion A/B/C banks."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--ad-source", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--spatial-encoder", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument("--ddim-eta", type=float, default=1.0)
    return parser.parse_args()


def load_slots(prepared):
    plan_path = prepared / "generation_slots.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = plan["methods"]["anomalydiffusion"]
    require(plan["status"] in {"FROZEN", "FROZEN_GENERATION_PLAN_NOT_SAMPLED"},
            "generation_slots.json is not a frozen, unsampled plan")
    require(len(rows) == 240, f"expected 240 AnomalyDiffusion slots, got {len(rows)}")
    counts = Counter((row["bank"], row["class_name"]) for row in rows)
    expected = Counter({(bank, cls): 40 for bank in "ABC" for cls in ("flash", "black")})
    require(counts == expected, f"invalid bank/class counts: {dict(counts)}")
    require(len({(row["bank"], row["slot_id"]) for row in rows}) == 240, "duplicate bank/slot")
    require(all(row["status"] == "PLANNED_NOT_SAMPLED" for row in rows), "non-planned slot found")
    ordered = []
    for bank in "ABC":
        ordered.extend(row for row in rows if row["bank"] == bank)
    return plan_path, plan, ordered


def seed_everything(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 63 - 1))
    torch.cuda.manual_seed_all(seed % (2 ** 63 - 1))


def label_text(row):
    class_id = {"flash": 0, "black": 1}[row["class_name"]]
    return str(class_id) + " " + " ".join(f"{float(value):.6f}" for value in row["bbox"]) + "\n"


def main():
    args = parse_args()
    require(args.ddim_steps == 200 and args.ddim_eta == 1.0,
            "Exp2 protocol requires native DDIM steps=200 and eta=1.0")
    required = [args.prepared, args.ad_source, args.project_config, args.base_model,
                args.spatial_encoder, args.embeddings, args.training_checkpoint]
    for path in required:
        require(path.exists(), f"missing required input: {path}")
    require(not args.output.exists(), f"output already exists; refusing overwrite/reroll: {args.output}")

    plan_path, plan, rows = load_slots(args.prepared)
    args.output.mkdir(parents=True)
    status_path = args.output / "generation_status.json"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = {
        "status": "STARTING", "method": "AnomalyDiffusion", "started_utc": started,
        "planned_count": 240, "completed_count": 0, "current_slot": None,
        "quality_selection": False, "reroll_count": 0,
    }
    write_json(status_path, status)

    try:
        print("[启动] 校验冻结槽位与训练权重绑定。", flush=True)
        bindings = {}
        for name, path in (("generation_slots", plan_path), ("project_config", args.project_config),
                           ("base_model", args.base_model), ("spatial_encoder", args.spatial_encoder),
                           ("embeddings", args.embeddings), ("training_checkpoint", args.training_checkpoint)):
            size = path.stat().st_size
            bindings[name] = {"path": str(path.resolve()), "size": size,
                              "mtime_ns": path.stat().st_mtime_ns,
                              "sha256": sha256(path) if size < 100 * 1024 * 1024 else None,
                              "sha256_status": "COMPLETE" if size < 100 * 1024 * 1024 else "DEFERRED"}
        binding_doc = {
            "status": "IDENTITY_RECORDED_SHA256_PENDING", "method": "AnomalyDiffusion",
            "sampling_weights": ["base_model", "spatial_encoder", "embeddings"],
            "training_provenance": "training_checkpoint", "artifacts": bindings,
        }
        write_json(args.output / "generator_checkpoint_binding.json", binding_doc)

        os.environ["AD_NAME_ANOMALY_FILE"] = str((args.prepared / "name-anomaly.txt").resolve())
        os.chdir(args.ad_source)
        sys.path.insert(0, str(args.ad_source))
        import numpy as np
        import torch
        from PIL import Image
        from omegaconf import OmegaConf
        from ldm.data.personalized import imagenet_templates_small
        from ldm.util import instantiate_from_config

        require(torch.cuda.is_available(), "CUDA is unavailable")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = OmegaConf.load(str(args.project_config))
        config.model.params.ckpt_path = str(args.base_model.resolve())
        require(int(config.model.params.generation_resolution) == 512, "generation_resolution must be 512")
        require(int(config.model.params.spatial_condition_resolution) == 256,
                "spatial_condition_resolution must be 256")
        require(int(config.model.params.image_size) == 64, "latent image_size must be 64")

        status.update(status="LOADING_MODEL", current_slot=None)
        write_json(status_path, status)
        print("[运行中] 加载基础模型；随后只加载本次训练得到的空间编码器和嵌入权重。", flush=True)
        model = instantiate_from_config(config.model)
        model = model.cuda().eval()
        model.prepare_spatial_encoder(optimze_together=True)
        spatial_state = torch.load(args.spatial_encoder, map_location="cpu")
        model.embedding_manager.spatial_encoder_model.load_state_dict(spatial_state, strict=True)
        model.embedding_manager.load(str(args.embeddings))
        model.eval()
        model.embedding_manager.spatial_encoder_model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        print("[检查通过] 模型与训练权重加载完成；开始正式固定槽位采样。", flush=True)

        records = []
        bank_records = {bank: [] for bank in "ABC"}
        verified = set()
        for bank in "ABC":
            (args.output / f"synthetic_{bank}").mkdir()
            write_json(args.output / f"formal_bank_{bank}_manifest.json", {
                "status": "PENDING", "method": "AnomalyDiffusion", "bank": bank,
                "count": 0, "planned_count": 80, "quality_selection": False,
                "reroll_count": 0, "records": [],
            })

        for index, row in enumerate(rows, start=1):
            bank = row["bank"]
            slot = row["frozen_slot"]
            require(row["frozen_slot_sha256"] == hashlib.sha256(
                json.dumps(slot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(), f"frozen slot digest mismatch: {row['slot_id']}")
            image_path = args.prepared / row["image_path"]
            mask_path = args.prepared / row["mask_path"]
            for path, expected_hash in ((image_path, row["image_sha256"]),
                                        (mask_path, row["mask_sha256"]),
                                        (Path(slot["normal_path"]), slot["normal_sha256"])):
                key = (str(path), expected_hash)
                if key not in verified:
                    require(path.exists(), f"missing slot asset: {path}")
                    require(sha256(path) == expected_hash, f"slot asset digest mismatch: {path}")
                    verified.add(key)

            seed = int(row["seed"])
            seed_everything(seed, torch, np)
            caption = random.choice(imagenet_templates_small).format("*")
            with Image.open(image_path) as image:
                image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            with Image.open(mask_path) as mask:
                mask_array = (np.asarray(mask.convert("L"), dtype=np.float32) >= 128).astype(np.float32)
            require(image_array.shape == (512, 512, 3), f"invalid image shape: {image_array.shape}")
            require(mask_array.shape == (512, 512), f"invalid mask shape: {mask_array.shape}")
            batch = {
                "image": torch.from_numpy(image_array.copy()).unsqueeze(0),
                "mask": torch.from_numpy(mask_array.copy()).unsqueeze(0),
                "caption": [caption],
                "name": ["Plastic_Bomo+" + row["class_name"]],
            }

            status.update(status="RUNNING", current_slot=row["slot_id"], completed_count=len(records))
            write_json(status_path, status)
            print(f"[运行中] {index}/240 bank={bank} class={row['class_name']} "
                  f"slot={row['slot_id']} seed={seed}", flush=True)
            with torch.inference_mode():
                generated = model.log_images(
                    batch, N=1, sample=False, inpaint=True, unconditional_only=True,
                    adaptive_mask=False, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
                )["samples_inpainting"]
            require(tuple(generated.shape) == (1, 3, 512, 512),
                    f"invalid generated tensor shape: {tuple(generated.shape)}")
            require(bool(torch.isfinite(generated).all()), "non-finite generated tensor")
            patch = generated[0].detach().float().cpu().clamp(-1, 1)
            patch_array = ((patch.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
            patch_image = Image.fromarray(patch_array, "RGB")
            with Image.open(slot["normal_path"]) as normal:
                full = normal.convert("RGB")
            crop_x, crop_y, _, _ = slot["geometry"]["fixed_crop_xyxy"]
            left, top, right, bottom = slot["geometry"]["paste_valid_crop_region_xyxy"]
            full.paste(patch_image.crop((left - crop_x, top - crop_y, right - crop_x, bottom - crop_y)),
                       (left, top))
            require(list(full.size) == slot["geometry"]["final_canvas_wh"], "invalid final canvas size")

            output_dir = args.output / f"synthetic_{bank}"
            output_image = output_dir / f"{row['slot_id']}.png"
            output_label = output_dir / f"{row['slot_id']}.txt"
            temporary_image = output_image.with_name(output_image.stem + ".tmp.png")
            full.save(temporary_image, format="PNG")
            os.replace(temporary_image, output_image)
            output_label.write_text(label_text(row), encoding="utf-8")
            record = {
                "slot_id": row["slot_id"], "bank": bank, "class": row["class_name"],
                "conditioning_normal_id": row["conditioning_normal_id"],
                "bank_seed": int(row["bank_seed"]), "sample_seed": seed, "caption": caption,
                "frozen_slot": slot, "frozen_slot_sha256": row["frozen_slot_sha256"],
                "image_path": output_image.relative_to(args.output).as_posix(),
                "image_sha256": sha256(output_image),
                "label_path": output_label.relative_to(args.output).as_posix(),
                "label_sha256": sha256(output_label),
                "patch_rgb_sha256": hashlib.sha256(patch_array.tobytes()).hexdigest(),
                "decoded_RGB_sha256": hashlib.sha256(np.asarray(full).tobytes()).hexdigest(),
                "finite_pipeline_pixels": True, "formal_synthetic": True,
                "quality_selection": False, "reroll_count": 0,
                "sampling": {"sampler": "DDIM", "steps": args.ddim_steps,
                             "eta": args.ddim_eta, "adaptive_mask": False},
            }
            records.append(record)
            bank_records[bank].append(record)
            write_json(args.output / f"formal_bank_{bank}_manifest.json", {
                "status": "COMPLETE" if len(bank_records[bank]) == 80 else "RUNNING",
                "method": "AnomalyDiffusion", "bank": bank, "count": len(bank_records[bank]),
                "planned_count": 80, "quality_selection": False, "reroll_count": 0,
                "records": bank_records[bank],
            })
            status.update(completed_count=len(records), current_slot=None)
            write_json(status_path, status)

        final_counts = Counter((record["bank"], record["class"]) for record in records)
        expected_counts = Counter({(bank, cls): 40 for bank in "ABC" for cls in ("flash", "black")})
        require(len(records) == 240 and final_counts == expected_counts, "final slot coverage failure")
        print("[运行中] 生成完成；计算大权重 SHA256 并冻结最终绑定。", flush=True)
        for name, artifact in bindings.items():
            if artifact["sha256"] is None:
                path = Path(artifact["path"])
                require(path.stat().st_size == artifact["size"] and path.stat().st_mtime_ns == artifact["mtime_ns"],
                        f"checkpoint changed during generation: {path}")
                artifact["sha256"] = sha256(path)
                artifact["sha256_status"] = "COMPLETE"
        binding_doc["status"] = "FROZEN"
        write_json(args.output / "generator_checkpoint_binding.json", binding_doc)
        write_json(args.output / "formal_synthetic_manifest.json", {
            "status": "FROZEN", "method": "AnomalyDiffusion", "count": len(records),
            "bank_counts": dict(Counter(record["bank"] for record in records)),
            "class_counts": dict(Counter(record["class"] for record in records)),
            "quality_selection": False, "reroll_count": 0,
            "generation_slots_sha256": bindings["generation_slots"]["sha256"],
            "generator_checkpoint_binding_sha256": sha256(args.output / "generator_checkpoint_binding.json"),
            "records": records,
        })
        write_json(args.output / "formal_synthetic_validity.json", {
            "status": "PASS", "method": "AnomalyDiffusion", "expected_count": 240,
            "actual_count": len(records), "exact_frozen_slot_coverage": True,
            "all_outputs_finite": True, "quality_selection": False, "reroll_count": 0,
        })
        status.update(status="COMPLETE", completed_count=240, current_slot=None,
                      finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(status_path, status)
        print(f"[完成] AnomalyDiffusion 正式生成完成：240/240；输出={args.output}", flush=True)
    except BaseException as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                      failed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(status_path, status)
        traceback.print_exc()
        print(f"[已退出] AnomalyDiffusion 生成失败：{type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
