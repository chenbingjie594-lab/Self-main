"""Evaluate the fixed Plastic_Bomo class-adaptive MSDF gamma composition.

This experiment performs no training, sampling, image copying, or image
encoding.  It combines the already-frozen gamma=0 Flash images with the
already-frozen gamma=1 Black images and recomputes aggregate KID/FID from the
SHA-bound feature cache created by the preceding gamma sweep.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_class_adaptive_gamma"
P0 = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/server_runtime_20261013_080546_552698/p0_view"
SOURCE_RUN = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214"
CLASSES = ("flash", "black")
METHODS = (
    ("Uniform gamma=0", {"flash": 0.0, "black": 0.0}),
    ("Uniform gamma=1", {"flash": 1.0, "black": 1.0}),
    ("Class-Adaptive gamma", {"flash": 0.0, "black": 1.0}),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def crop_box(bbox, width, height, margin=0.25):
    cx, cy, bw, bh = bbox
    return [
        max(0, math.floor((cx - (0.5 + margin) * bw) * width)),
        max(0, math.floor((cy - (0.5 + margin) * bh) * height)),
        min(width, math.ceil((cx + (0.5 + margin) * bw) * width)),
        min(height, math.ceil((cy + (0.5 + margin) * bh) * height)),
    ]


def gamma_tag(value):
    return f"gamma_{value:.2f}".replace(".", "p")


def verify_source_artifact_manifest():
    manifest = load(SOURCE / "gamma_sweep_artifact_manifest.json")
    checked = 0
    if isinstance(manifest.get("files"), list):
        for row in manifest["files"]:
            path = SOURCE / row["path"]
            require(path.is_file(), f"DATA_REQUIRED: source artifact missing: {path}")
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                    f"DATA_REQUIRED: source artifact changed: {path}")
            checked += 1
    elif isinstance(manifest.get("sha256"), dict):
        for name, digest in manifest["sha256"].items():
            path = SOURCE / name
            require(path.is_file() and sha(path) == digest,
                    f"DATA_REQUIRED: source artifact changed: {path}")
            checked += 1
    else:
        raise RuntimeError("DATA_REQUIRED: unsupported source artifact manifest")
    return checked


def integrity_audit():
    required = [
        SOURCE / "gamma_sweep_protocol.json",
        SOURCE / "gamma_sweep_results.csv",
        SOURCE / "gamma_sweep_classwise.csv",
        SOURCE / "gamma_sweep_artifact_manifest.json",
        SOURCE / "gamma_0p00_manifest.json",
        SOURCE / "gamma_1p00_manifest.json",
        SOURCE / "gamma_sweep_inception_features.npz",
        SOURCE / "gamma_sweep_inception_feature_cache.json",
        SOURCE / "gamma_sweep_lpips_pairs.csv",
        P0 / "stage20b_generation_slots.json",
        P0 / "stage20b_rng_manifest.json",
        P0 / "stage20b_generation_protocol.json",
        SOURCE_RUN / "formal_bank_A_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "DATA_REQUIRED: " + ", ".join(missing))
    source_artifacts_checked = verify_source_artifact_manifest()
    protocol = load(SOURCE / "gamma_sweep_protocol.json")
    require(protocol["bank"] == "A" and protocol["slot_count_per_gamma"] == 80,
            "DATA_REQUIRED: preceding sweep is not frozen Bank A")
    require(protocol["same_slots_normal_bbox_prompt_seed_steps_for_all_gamma"],
            "DATA_REQUIRED: preceding sweep did not freeze conditions")
    require(sha(P0 / "stage20b_generation_slots.json") == protocol["slots_sha256"],
            "DATA_REQUIRED: frozen slot file changed")
    require(sha(P0 / "stage20b_rng_manifest.json") == protocol["rng_manifest_sha256"],
            "DATA_REQUIRED: frozen RNG file changed")
    require(sha(P0 / "stage20b_generation_protocol.json") == protocol["generation_protocol_sha256"],
            "DATA_REQUIRED: generation protocol changed")
    require(sha(SOURCE_RUN / "formal_bank_A_manifest.json") == protocol["source_bank_A_manifest_sha256"],
            "DATA_REQUIRED: source Bank-A manifest changed")

    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rngs = {row["slot_id"]: row["banks"]["A"] for row in load(P0 / "stage20b_rng_manifest.json")["records"]}
    formal = load(SOURCE_RUN / "formal_bank_A_manifest.json")
    formal_by_slot = {row["slot_id"]: row for row in formal["records"]}
    require(len(slots) == 80 and Counter(row["class"] for row in slots) == {"flash": 40, "black": 40},
            "DATA_REQUIRED: frozen slot population changed")
    require(formal["bank"] == "A" and formal["count"] == 80, "DATA_REQUIRED: source bank changed")
    for slot in slots:
        source_row = formal_by_slot.get(slot["slot_id"])
        require(source_row is not None and source_row["frozen_slot"] == slot,
                f"DATA_REQUIRED: frozen slot binding changed: {slot['slot_id']}")
        require(source_row["sample_seed"] == rngs[slot["slot_id"]]["sample_seed"] and
                source_row["bank_seed"] == rngs[slot["slot_id"]]["bank_seed"] == 42,
                f"DATA_REQUIRED: Bank-A seed changed: {slot['slot_id']}")

    source_manifests = {}
    by_gamma = {}
    image_checks = 0
    for gamma in (0.0, 1.0):
        path = SOURCE / f"{gamma_tag(gamma)}_manifest.json"
        manifest = load(path)
        require(manifest["status"] == "COMPLETE" and manifest["gamma"] == gamma and manifest["count"] == 80,
                f"DATA_REQUIRED: incomplete gamma manifest: {gamma}")
        require(Counter(row["class"] for row in manifest["records"]) == {"flash": 40, "black": 40},
                f"DATA_REQUIRED: class coverage changed: {gamma}")
        rows = {row["slot_id"]: row for row in manifest["records"]}
        require(list(rows) == [slot["slot_id"] for slot in slots],
                f"DATA_REQUIRED: frozen slot order changed: {gamma}")
        for slot in slots:
            row = rows[slot["slot_id"]]
            expected_roi = crop_box(slot["bbox"], *slot["geometry"]["final_canvas_wh"])
            require(row["class"] == slot["class"] and row["gamma"] == gamma,
                    f"DATA_REQUIRED: class/gamma mismatch: {slot['slot_id']}")
            require(row["sample_seed"] == rngs[slot["slot_id"]]["sample_seed"] and row["bank_seed"] == 42,
                    f"DATA_REQUIRED: sample seed changed: {slot['slot_id']}")
            require(row["roi_xyxy"] == expected_roi,
                    f"DATA_REQUIRED: ROI/bbox changed: {slot['slot_id']}")
            image = SOURCE / row["image_path"]
            require(image.is_file() and sha(image) == row["image_sha256"],
                    f"DATA_REQUIRED: image missing or changed: {image}")
            image_checks += 1
        source_manifests[gamma] = manifest
        by_gamma[gamma] = rows

    selected = []
    for slot in slots:
        gamma = 0.0 if slot["class"] == "flash" else 1.0
        row = by_gamma[gamma][slot["slot_id"]]
        selected.append({
            "slot_id": slot["slot_id"], "class": slot["class"], "bank": "A", "gamma": gamma,
            "source_manifest": f"../plastic_bomo_msdf_gamma_sweep/{gamma_tag(gamma)}_manifest.json",
            "source_image_path": f"../plastic_bomo_msdf_gamma_sweep/{row['image_path']}",
            "image_sha256": row["image_sha256"], "sample_seed": row["sample_seed"],
            "bank_seed": row["bank_seed"], "roi_xyxy": row["roi_xyxy"],
        })
    require(Counter(row["class"] for row in selected) == {"flash": 40, "black": 40},
            "DATA_REQUIRED: class-adaptive selection incomplete")
    require(all(row["gamma"] == (0.0 if row["class"] == "flash" else 1.0) for row in selected),
            "DATA_REQUIRED: class-adaptive gamma mapping wrong")

    raw_hashes = {
        "gamma_sweep_protocol.json": sha(SOURCE / "gamma_sweep_protocol.json"),
        "gamma_sweep_results.csv": sha(SOURCE / "gamma_sweep_results.csv"),
        "gamma_sweep_classwise.csv": sha(SOURCE / "gamma_sweep_classwise.csv"),
        "gamma_sweep_artifact_manifest.json": sha(SOURCE / "gamma_sweep_artifact_manifest.json"),
        "gamma_0p00_manifest.json": sha(SOURCE / "gamma_0p00_manifest.json"),
        "gamma_1p00_manifest.json": sha(SOURCE / "gamma_1p00_manifest.json"),
        "gamma_sweep_inception_features.npz": sha(SOURCE / "gamma_sweep_inception_features.npz"),
        "gamma_sweep_inception_feature_cache.json": sha(SOURCE / "gamma_sweep_inception_feature_cache.json"),
        "gamma_sweep_lpips_pairs.csv": sha(SOURCE / "gamma_sweep_lpips_pairs.csv"),
        "stage20b_generation_slots.json": sha(P0 / "stage20b_generation_slots.json"),
        "stage20b_rng_manifest.json": sha(P0 / "stage20b_rng_manifest.json"),
        "formal_bank_A_manifest.json": sha(SOURCE_RUN / "formal_bank_A_manifest.json"),
    }
    audit = {
        "status": "PASS", "checked_at": now(), "source_artifacts_checked": source_artifacts_checked,
        "source_manifest_records_checked": 160, "source_images_sha256_checked": image_checks,
        "selected_image_count": 80, "selected_class_counts": {"flash": 40, "black": 40},
        "selected_gamma_mapping": {"flash": 0.0, "black": 1.0}, "bank": "A",
        "frozen_slot_count": 80, "frozen_slot_binding_checks": 80,
        "normal_bbox_prompt_binding": "PASS via exact frozen_slot equality and protocol SHA256",
        "seed_binding": "PASS", "roi_binding": "PASS", "image_sha256_binding": "PASS",
        "images_copied": 0, "images_modified": 0, "images_generated": 0,
        "training_count": 0, "resampling_count": 0, "quality_selection_count": 0,
        "historical_results_modified": False, "final_eval_read_or_used": False,
        "raw_input_sha256": raw_hashes,
    }
    return protocol, slots, source_manifests, selected, audit


def load_features(protocol, source_manifests):
    import numpy as np
    import torch

    cache = SOURCE / "gamma_sweep_inception_features.npz"
    ledger_path = SOURCE / "gamma_sweep_inception_feature_cache.json"
    ledger = load(ledger_path)
    require(ledger["sha256"] == sha(cache) and ledger["shape"] == [572, 2048],
            "DATA_REQUIRED: Inception feature cache binding changed")
    expected_keys = ["real:" + row["annotation_id"] for row in protocol["development_reference"]["records"]]
    for gamma in (0.0, 0.25, 0.5, 0.75, 1.0):
        manifest = load(SOURCE / f"{gamma_tag(gamma)}_manifest.json")
        expected_keys.extend(f"fake:{gamma:.2f}:" + row["slot_id"] for row in manifest["records"])
    require(ledger["keys"] == expected_keys, "DATA_REQUIRED: feature input order changed")
    with np.load(cache, allow_pickle=False) as archive:
        values = archive["features"].copy()
    require(values.shape == (572, 2048) and bool(np.isfinite(values).all()),
            "DATA_REQUIRED: invalid cached features")
    return {key: torch.from_numpy(values[index]).double().cpu() for index, key in enumerate(expected_keys)}, ledger


def compute_metric(real, fake):
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    options = {
        "kid_subsets": 100, "kid_subset_size": min(50, len(real), len(fake)),
        "rng_seed": 42, "verbose": False, "kid_degree": 3, "kid_gamma": None, "kid_coef0": 1.0,
    }
    started = time.monotonic()
    kid = kid_features_to_metric(real, fake, **options)
    fid = fid_statistics_to_metric(fid_features_to_statistics(real), fid_features_to_statistics(fake), verbose=False)
    return {
        "KID": float(kid["kernel_inception_distance_mean"]),
        "KID_subset_std": float(kid["kernel_inception_distance_std"]),
        "FID": float(fid["frechet_inception_distance"]),
        "KID_subsets": 100, "KID_subset_size": options["kid_subset_size"],
        "runtime_seconds": time.monotonic() - started,
    }


def recompute_class_adaptive(protocol, selected, features):
    import torch

    records = []
    for cohort in ("overall", *CLASSES):
        real_rows = [row for row in protocol["development_reference"]["records"]
                     if cohort == "overall" or row["class"] == cohort]
        fake_rows = [row for row in selected if cohort == "overall" or row["class"] == cohort]
        real = torch.stack([features["real:" + row["annotation_id"]] for row in real_rows]).double().cpu()
        fake = torch.stack([features[f"fake:{row['gamma']:.2f}:" + row["slot_id"]] for row in fake_rows]).double().cpu()
        metric = compute_metric(real, fake)
        records.append({"class": cohort, "real_roi_count": len(real_rows),
                        "generated_roi_count": len(fake_rows), **metric})
        print(f"[指标] Class-Adaptive {cohort}: KID={metric['KID']:.10f} FID={metric['FID']:.6f}", flush=True)
    return records


def load_lpips(selected):
    rows = read_csv(SOURCE / "gamma_sweep_lpips_pairs.csv")
    chosen = []
    selected_ids = {cls: sorted(row["slot_id"] for row in selected if row["class"] == cls) for cls in CLASSES}
    for cls in CLASSES:
        gamma = 0.0 if cls == "flash" else 1.0
        class_rows = [row for row in rows if float(row["gamma"]) == gamma and row["class"] == cls]
        require(len(class_rows) == 20, f"DATA_REQUIRED: LPIPS pairs missing: {cls}")
        expected = [(selected_ids[cls][2*i], selected_ids[cls][2*i+1]) for i in range(20)]
        actual = [(row["slot_id_1"], row["slot_id_2"]) for row in class_rows]
        require(actual == expected, f"DATA_REQUIRED: LPIPS adjacent pairing changed: {cls}")
        chosen.extend(class_rows)
    values = {cls: [float(row["LPIPS"]) for row in chosen if row["class"] == cls] for cls in CLASSES}
    return {
        "flash": sum(values["flash"]) / len(values["flash"]),
        "black": sum(values["black"]) / len(values["black"]),
        "overall": sum(values["flash"] + values["black"]) / 40,
        "pair_count": 40,
        "reuse_policy": "Exact preceding-sweep LPIPS values reused after source-image SHA and deterministic adjacent-slot pair validation.",
    }


def source_result(gamma, cohort):
    if cohort == "overall":
        rows = read_csv(SOURCE / "gamma_sweep_results.csv")
        row = next(item for item in rows if float(item["gamma"]) == gamma)
        return {"KID": float(row["Overall_KID"]), "KID_subset_std": float(row["Overall_KID_subset_std"]),
                "FID": float(row["Overall_FID"]), "LPIPS": float(row["Overall_LPIPS"]),
                "generated_images": int(row["generated_images"]), "reference_rois": int(row["reference_rois"])}
    rows = read_csv(SOURCE / "gamma_sweep_classwise.csv")
    row = next(item for item in rows if float(item["gamma"]) == gamma and item["class"].lower() == cohort)
    return {"KID": float(row["KID"]), "KID_subset_std": float(row["KID_subset_std"]),
            "FID": float(row["FID"]), "LPIPS": float(row["LPIPS"]),
            "generated_images": int(row["generated_images"]), "reference_rois": int(row["reference_rois"])}


def publish(protocol, selected, audit, adaptive, lpips, ledger, wall_seconds):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw_binding = canonical_sha(audit["raw_input_sha256"])
    save(OUTPUT / "class_adaptive_manifest.json", {
        "status": "COMPLETE", "method": "Class-Adaptive gamma", "bank": "A", "count": 80,
        "gamma_setting": {"flash": 0.0, "black": 1.0}, "images_copied": 0,
        "raw_input_SHA256": raw_binding, "records": selected,
    })
    protocol_out = {
        "status": "FROZEN_AND_COMPLETE", "created_at": now(),
        "objective": "Test whether fixed class-adaptive MSDF residual gamma improves aggregate Bank-A development metrics.",
        "groups": {
            "G0": {"method": "Uniform gamma=0", "gamma_setting": {"flash": 0.0, "black": 0.0}, "source": "reused"},
            "G1": {"method": "Uniform gamma=1", "gamma_setting": {"flash": 1.0, "black": 1.0}, "source": "reused"},
            "G2": {"method": "Class-Adaptive gamma", "gamma_setting": {"flash": 0.0, "black": 1.0}, "source": "existing-image references"},
        },
        "sample_count_per_group": 80, "bank": "A", "training_performed": False,
        "generation_performed": False, "image_copy_or_encoding_performed": False,
        "same_frozen_slots_and_conditions": True,
        "reference": {"split": "V2 TRAIN", "count": 172, "class_counts": {"flash": 75, "black": 97},
                      "checkpoint_training_overlap_possible": True, "independent_validation": False,
                      "final_eval_read_or_used": False},
        "metrics": protocol["metrics"],
        "overall_metric_rule": "Concatenate 40 gamma=0 Flash and 40 gamma=1 Black cached Inception features, then recompute against all 172 real ROI features; never average class KID/FID.",
        "feature_cache": {"shape": ledger["shape"], "sha256": ledger["sha256"], "input_order_exactly_reused": True},
        "lpips_rule": "Reuse the exact 20 adjacent-slot pairs per class after SHA/pair verification; not comparable to historical A/B/C same-slot LPIPS.",
        "selection_bias": "Class gamma choices were selected after observing this same development reference.",
        "historical_final_eval_comparable": False, "raw_input_SHA256": audit["raw_input_sha256"],
        "implementation_SHA256": sha(Path(__file__)),
        "prohibited_extensions_started": False,
    }
    save(OUTPUT / "class_adaptive_protocol.json", protocol_out)
    save(OUTPUT / "class_adaptive_integrity_audit.json", audit)

    adaptive_by_class = {row["class"]: row for row in adaptive}
    tolerance = 1e-12
    expected_flash = source_result(0.0, "flash")
    expected_black = source_result(1.0, "black")
    checks = {
        "Flash_KID_abs_error_vs_gamma0": abs(adaptive_by_class["flash"]["KID"] - expected_flash["KID"]),
        "Flash_FID_abs_error_vs_gamma0": abs(adaptive_by_class["flash"]["FID"] - expected_flash["FID"]),
        "Black_KID_abs_error_vs_gamma1": abs(adaptive_by_class["black"]["KID"] - expected_black["KID"]),
        "Black_FID_abs_error_vs_gamma1": abs(adaptive_by_class["black"]["FID"] - expected_black["FID"]),
    }
    require(all(value <= tolerance for value in checks.values()), f"NUMERICAL_INCONSISTENCY: {checks}")
    checks.update(status="PASS", tolerance=tolerance)

    result_rows = []
    class_rows = []
    for method, setting in METHODS:
        if method == "Class-Adaptive gamma":
            overall = adaptive_by_class["overall"]
            values = {cls: adaptive_by_class[cls] for cls in CLASSES}
            overall_lpips = lpips["overall"]
            class_lpips = {cls: lpips[cls] for cls in CLASSES}
            mode = "recomputed_from_combined_cached_features"
        else:
            gamma = setting["flash"]
            overall = source_result(gamma, "overall")
            values = {cls: source_result(gamma, cls) for cls in CLASSES}
            overall_lpips = overall["LPIPS"]
            class_lpips = {cls: values[cls]["LPIPS"] for cls in CLASSES}
            mode = "reused_complete_preceding_sweep_result"
        result_rows.append({
            "method": method, "gamma_setting": json.dumps(setting, sort_keys=True),
            "generated_images": 80, "reference_rois": 172,
            "Overall_KID": overall["KID"], "Overall_KID_subset_std": overall["KID_subset_std"],
            "Overall_FID": overall["FID"], "Overall_LPIPS": overall_lpips,
            "Raw_input_SHA256": raw_binding, "metric_source": mode,
            "metric_runtime_seconds": overall.get("runtime_seconds", 0.0),
            "reference_source": "V2 TRAIN; 172 ROIs; possible checkpoint-training overlap",
            "historical_results_modified": False,
        })
        for cls in CLASSES:
            class_rows.append({
                "method": method, "gamma": setting[cls], "class": cls.capitalize(),
                "generated_images": 40, "reference_rois": 75 if cls == "flash" else 97,
                "KID": values[cls]["KID"], "KID_subset_std": values[cls]["KID_subset_std"],
                "FID": values[cls]["FID"], "LPIPS": class_lpips[cls],
                "Raw_input_SHA256": raw_binding, "metric_source": mode,
                "metric_runtime_seconds": values[cls].get("runtime_seconds", 0.0),
                "historical_results_modified": False,
            })
    write_csv(OUTPUT / "class_adaptive_results.csv", list(result_rows[0]), result_rows)
    write_csv(OUTPUT / "class_adaptive_classwise.csv", list(class_rows[0]), class_rows)

    by_method = {row["method"]: row for row in result_rows}
    adaptive_result = by_method["Class-Adaptive gamma"]
    differences = {}
    for baseline in ("Uniform gamma=0", "Uniform gamma=1"):
        differences[baseline] = {
            "Overall_KID": adaptive_result["Overall_KID"] - by_method[baseline]["Overall_KID"],
            "Overall_FID": adaptive_result["Overall_FID"] - by_method[baseline]["Overall_FID"],
            "Overall_LPIPS": adaptive_result["Overall_LPIPS"] - by_method[baseline]["Overall_LPIPS"],
        }
    best_kid = min(result_rows, key=lambda row: row["Overall_KID"])["method"]
    best_fid = min(result_rows, key=lambda row: row["Overall_FID"])["method"]
    comparison = {
        "status": "COMPLETE", "numerical_consistency": checks,
        "differences_class_adaptive_minus_baseline": differences,
        "best_Overall_KID": best_kid, "best_Overall_FID": best_fid,
        "class_adaptive_better_than_both_uniform_on_KID": all(differences[name]["Overall_KID"] < 0 for name in differences),
        "class_adaptive_better_than_both_uniform_on_FID": all(differences[name]["Overall_FID"] < 0 for name in differences),
        "KID_improves_but_FID_degrades_vs_any_baseline": any(
            value["Overall_KID"] < 0 and value["Overall_FID"] > 0 for value in differences.values()),
        "independent_validation_recommended": True,
        "interpretation": "Exploratory Bank-A/V2-TRAIN result with development-set selection bias.",
    }
    save(OUTPUT / "class_adaptive_comparison.json", comparison)

    lines = [
        "# MSDF Class-Adaptive Gamma Experiment", "",
        "Status: `COMPLETE`. No training, sampling, regeneration, image copying, image encoding, or historical-result modification was performed.", "",
        "## Protocol boundary", "",
        "G2 references the existing 40 Flash images at gamma=0 and 40 Black images at gamma=1 from the same frozen Bank A. Every selected PNG, slot, ROI and seed passed SHA/binding checks. Overall KID/FID was recomputed from the combined 80-image feature distribution against the same 172 V2 TRAIN ROIs; it is not an average of class scores.", "",
        "The V2 TRAIN reference may overlap checkpoint training data, and the class-specific gamma choice was made after observing this development set. Results are exploratory and cannot be compared directly with historical final_eval values.", "",
        "## Three-group comparison", "",
        "| Method | Overall KID ↓ | Overall FID ↓ | Flash KID ↓ | Black KID ↓ | Flash FID ↓ | Black FID ↓ | Overall LPIPS ↑ |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        c = {item["class"].lower(): item for item in class_rows if item["method"] == row["method"]}
        lines.append(f"| {row['method']} | {row['Overall_KID']:.10f} | {row['Overall_FID']:.6f} | {c['flash']['KID']:.10f} | {c['black']['KID']:.10f} | {c['flash']['FID']:.6f} | {c['black']['FID']:.6f} | {row['Overall_LPIPS']:.6f} |")
    lines += ["", "## Differences: Class-Adaptive minus baseline", "",
              "| Baseline | Δ Overall KID | Δ Overall FID | Δ Overall LPIPS |", "| --- | ---: | ---: | ---: |"]
    for baseline, values in differences.items():
        lines.append(f"| {baseline} | {values['Overall_KID']:+.10f} | {values['Overall_FID']:+.6f} | {values['Overall_LPIPS']:+.6f} |")
    kid_answer = "does" if comparison["class_adaptive_better_than_both_uniform_on_KID"] else "does not"
    fid_answer = "does" if comparison["class_adaptive_better_than_both_uniform_on_FID"] else "does not"
    tradeoff = "A KID improvement/FID degradation tradeoff exists against at least one baseline." if comparison["KID_improves_but_FID_degrades_vs_any_baseline"] else "No KID improvement/FID degradation tradeoff occurs against either baseline."
    lines += ["", "## Findings", "",
              f"1. Class-adaptive gamma {kid_answer} beat both uniform baselines on Overall KID.",
              f"2. Class-adaptive gamma {fid_answer} beat both uniform baselines on Overall FID.",
              f"3. Best Overall KID: **{best_kid}**. Best Overall FID: **{best_fid}**.",
              f"4. {tradeoff}",
              "5. Independent, source-group-isolated validation is warranted only as confirmation; this experiment cannot establish final performance.", "",
              "The experiment stops here. No wider gamma search, regeneration, retraining, BootstrapGuard, morphology ablation, or downstream detector training was started.", ""]
    (OUTPUT / "CLASS_ADAPTIVE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    save(OUTPUT / "class_adaptive_status.json", {
        "status": "COMPLETE", "completed_at": now(), "method_count": 3,
        "selected_image_count": 80, "reference_roi_count": 172,
        "generation_count": 0, "training_count": 0, "image_copy_count": 0,
        "historical_results_modified": False, "final_eval_read_or_used": False,
        "metric_runtime_seconds": wall_seconds,
        "best_Overall_KID": best_kid, "best_Overall_FID": best_fid,
        "independent_validation_recommended": True, "expanded_search_started": False,
        "BootstrapGuard": False, "downstream_detector_training_started": False,
    })


def main():
    started = time.monotonic()
    if OUTPUT.exists():
        raise RuntimeError(f"refusing existing output directory: {OUTPUT}")
    print(f"[启动] MSDF class-adaptive gamma evaluation PID={__import__('os').getpid()}；不训练、不生成。", flush=True)
    protocol, _slots, manifests, selected, audit = integrity_audit()
    print("[完整性] gamma=0/1 共160张源图与80张G2引用通过SHA、Bank A、slot、seed、ROI绑定。", flush=True)
    features, ledger = load_features(protocol, manifests)
    adaptive = recompute_class_adaptive(protocol, selected, features)
    lpips = load_lpips(selected)
    publish(protocol, selected, audit, adaptive, lpips, ledger, time.monotonic() - started)
    print("[完成] 三组比较与数值一致性检查通过；停止，不启动扩展实验。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
