"""Prepare exact frozen512 TRAIN crops and normal/weak-bbox generation slots."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import chapter3_exp1_quality_common as c
import chapter3_exp2_baseline_common as b


def crop_and_mask(path, digest, geometry):
    from PIL import Image
    c.checked_file(path, digest)
    with Image.open(path) as im:
        c.require(list(im.size) == geometry["original_canvas_wh"], "V2_CANVAS_CHANGED", str(path))
        crop = im.convert("RGB").crop(geometry["fixed_crop_xyxy"])
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, tuple(geometry["rasterized_mask_xyxy_exclusive"]))
    actual = hashlib.sha256(json.dumps([512, 512]).encode()+mask.tobytes()).hexdigest()
    c.require(actual == geometry["mask_decoded_sha256"] and crop.size == (512, 512),
              "V2_MASK_OR_CROP_CHANGED", str(path))
    return crop, mask


def source_path(real_root, original):
    value = str(original).replace("\\", "/")
    c.require("/real_only/" in value, "V2_SOURCE_PATH_INVALID", value)
    return c.inside(real_root, value.rsplit("/real_only/", 1)[1])


def export_pair(output, stem, crop, mask):
    c.require(stem and "/" not in stem and "\\" not in stem, "V2_NAME_INVALID", stem)
    folder = output / "assets"
    folder.mkdir(exist_ok=True)
    paths = {"image": folder/(stem+".png"), "mask": folder/(stem+"_mask.png")}
    crop.save(paths["image"], format="PNG")
    mask.save(paths["mask"], format="PNG")
    return {k+suffix: (p.relative_to(output).as_posix() if suffix == "_path" else c.sha(p))
            for k, p in paths.items() for suffix in ("_path", "_sha256")}


def prepare(args):
    cfg = c.load(c.REPO / "configs/plastic_bomo_chapter3_exp2.json")
    exp1, _, reference = b.frozen_inputs(cfg)
    output = Path(args.output).resolve()
    c.require(not output.exists(), "V2_PREPARED_OUTPUT_EXISTS", str(output))
    with b.forbid_content_reads(b.forbidden_paths(exp1, reference, args.real_root)) as guard:
        audit, plans = b.verify_train_metadata(cfg, exp1, True, args.real_root, args.normal_root)
        p0 = c.resolve(exp1["p0_root"])
        train = c.load(p0/"generator_train_allowlist.json")["inputs"]
        slots = c.load(p0/"stage20b_generation_slots.json")["slots"]
        output.mkdir(parents=True)
        records = []
        for row in train:
            image, mask = crop_and_mask(source_path(args.real_root, row["path"]), row["sha256"], row["geometry"])
            exported = export_pair(output, row["annotation_id"], image, mask)
            records.append({**exported, **{k: row[k] for k in ("annotation_id", "class", "source_group_id", "split")},
                "source_sha256": row["sha256"], "geometry": row["geometry"],
                "canvas_wh": [512, 512], "mask_is_human_segmentation": False})
        c.require(Counter(r["class"] for r in records) == {"flash": 75, "black": 97}, "V2_TRAIN_COUNT_CHANGED", "172")
        c.save(output/"training_manifest.json", dict(status="TRAIN_ONLY_ASSETS_PREPARED", records=records,
            source_bindings=cfg["frozen_bindings_sha256"], count=len(records)))
        slot_assets = {}
        for slot in slots:
            normal = c.inside(args.normal_root, slot["normal_path"].replace("\\", "/").rsplit("/", 1)[-1])
            image, mask = crop_and_mask(normal, slot["normal_sha256"], slot["geometry"])
            slot_assets[slot["slot_id"]] = export_pair(output, slot["slot_id"]+"_normal", image, mask)
        generation = {m: [{**r, **slot_assets[r["slot_id"]]} for r in rows] for m, rows in plans.items()}
        c.save(output/"generation_slots.json", dict(status="FROZEN_GENERATION_PLAN_NOT_SAMPLED", methods=generation,
            seed_policy=cfg["slot_seed"], count_per_method=240, quality_selection=False, reroll=False))
        (output/"name-anomaly.txt").write_text("Plastic_Bomo+flash\nPlastic_Bomo+black", encoding="utf-8", newline="\n")
        c.save(output/"prepared_binding.json", dict(status="FROZEN", training_manifest_sha256=c.sha(output/"training_manifest.json"),
            generation_slots_sha256=c.sha(output/"generation_slots.json"), name_anomaly_sha256=c.sha(output/"name-anomaly.txt")))
        c.save(output/"preparation_audit.json", {**audit, "status": "V2_BASELINE_ASSETS_PREPARED_NOT_TRAINED",
            "final_eval_open_attempts_blocked": guard["blocked_attempts"], "normal_filtering": False,
            "legacy_converters_executed": False, "formal_generation_authorized": False})
    print(f"[已退出] 已准备172个TRAIN裁剪和80个固定normal槽位；未训练、未生成。目录：{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_root", type=Path, required=True)
    parser.add_argument("--normal_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("[启动] Exp2 V2数据接入；不合并旧train/val，不筛normal，不读取final_eval内容。", flush=True)
    try:
        prepare(args)
    except Exception as exc:
        reason = exc.status if isinstance(exc,c.Stop) else type(exc).__name__
        print(f"[已退出] 数据接入失败：{reason}: {exc}；退出码=1。保留输出供审计，重跑请用新目录。", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
