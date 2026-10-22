"""Experiment 7: matched Raw SD2 versus class-adaptive MSDF evaluation.

No model is trained or sampled.  The script verifies the frozen Bank-A image
population, reuses the SHA-bound Raw ROI features from Experiment 1, reuses the
Gamma Sweep real/MSDF features, and recomputes all four distributions against
the same 172 V2 TRAIN ROI references.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_raw_matched_comparison"
RAW = REPO / "results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"
MSDF = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214"
P0 = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/server_runtime_20261013_080546_552698/p0_view"
EXP1 = REPO / "results_for_gpt/plastic_bomo_chapter3_exp1_msdf_generation_quality"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
ADAPTIVE = REPO / "results_for_gpt/plastic_bomo_msdf_class_adaptive_gamma"
CLASSES = ("flash", "black")
METHODS = (
    ("Raw SD2", "G0", {"flash": "raw", "black": "raw"}),
    ("MSDF Uniform gamma=0", "G1", {"flash": 0.0, "black": 0.0}),
    ("MSDF Uniform gamma=1", "G2", {"flash": 1.0, "black": 1.0}),
    ("MSDF Class-Adaptive gamma", "G3", {"flash": 0.0, "black": 1.0}),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


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
    result = [
        max(0, math.floor((cx - (0.5 + margin) * bw) * width)),
        max(0, math.floor((cy - (0.5 + margin) * bh) * height)),
        min(width, math.ceil((cx + (0.5 + margin) * bw) * width)),
        min(height, math.ceil((cy + (0.5 + margin) * bh) * height)),
    ]
    require(result[2] > result[0] and result[3] > result[1], f"invalid ROI: {bbox}")
    return result


def gamma_tag(value):
    return f"gamma_{value:.2f}".replace(".", "p")


def verify_hash_map(root, manifest_path, field="sha256"):
    manifest = load(manifest_path)
    records = manifest[field]
    for name, digest in records.items():
        path = root / name
        require(path.is_file() and sha(path) == digest, f"DATA_REQUIRED: artifact changed: {path}")
    return len(records)


def verify_gamma_artifacts():
    manifest = load(GAMMA / "gamma_sweep_artifact_manifest.json")
    count = 0
    if isinstance(manifest.get("files"), list):
        for row in manifest["files"]:
            path = GAMMA / row["path"]
            require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                    f"DATA_REQUIRED: gamma artifact changed: {path}")
            count += 1
    elif isinstance(manifest.get("sha256"), dict):
        count = verify_hash_map(GAMMA, GAMMA / "gamma_sweep_artifact_manifest.json")
    else:
        raise RuntimeError("DATA_REQUIRED: unsupported gamma artifact manifest")
    return count


def audit_inputs():
    required = [
        RAW / "formal_bank_A_manifest.json", RAW / "formal_result_artifact_manifest.json",
        RAW / "formal_synthetic_validity.json", RAW / "generator_checkpoint_binding.json",
        MSDF / "formal_bank_A_manifest.json",
        P0 / "stage20b_generation_slots.json", P0 / "stage20b_rng_manifest.json",
        EXP1 / "visual_comparison_manifest.json", EXP1 / "metric_runtime.json",
        EXP1 / "inception_feature_cache.json", EXP1 / "inception_features.npz",
        EXP1 / "experiment1_artifact_manifest.json",
        GAMMA / "gamma_sweep_protocol.json", GAMMA / "gamma_sweep_results.csv",
        GAMMA / "gamma_sweep_classwise.csv", GAMMA / "gamma_sweep_metric_runtime.json",
        GAMMA / "gamma_sweep_inception_feature_cache.json", GAMMA / "gamma_sweep_inception_features.npz",
        GAMMA / "gamma_0p00_manifest.json", GAMMA / "gamma_1p00_manifest.json",
        ADAPTIVE / "class_adaptive_results.csv", ADAPTIVE / "class_adaptive_classwise.csv",
        ADAPTIVE / "class_adaptive_manifest.json", ADAPTIVE / "class_adaptive_status.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "DATA_REQUIRED: " + ", ".join(missing))

    raw_artifacts = load(RAW / "formal_result_artifact_manifest.json")["sha256"]
    for name in ("formal_bank_A_manifest.json", "formal_synthetic_validity.json", "generator_checkpoint_binding.json"):
        require(raw_artifacts.get(name) == sha(RAW / name), f"DATA_REQUIRED: Raw artifact changed: {name}")
    exp_artifacts = load(EXP1 / "experiment1_artifact_manifest.json")["sha256"]
    for name in ("visual_comparison_manifest.json", "metric_runtime.json", "inception_feature_cache.json"):
        require(exp_artifacts.get(name) == sha(EXP1 / name), f"DATA_REQUIRED: Exp1 artifact changed: {name}")
    gamma_artifact_checks = verify_gamma_artifacts()

    protocol = load(GAMMA / "gamma_sweep_protocol.json")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rngs = {row["slot_id"]: row["banks"]["A"] for row in load(P0 / "stage20b_rng_manifest.json")["records"]}
    require(protocol["bank"] == "A" and protocol["slot_count_per_gamma"] == 80,
            "DATA_REQUIRED: gamma protocol is not frozen Bank A")
    require(sha(P0 / "stage20b_generation_slots.json") == protocol["slots_sha256"] and
            sha(P0 / "stage20b_rng_manifest.json") == protocol["rng_manifest_sha256"],
            "DATA_REQUIRED: slot/RNG binding changed")
    require(len(slots) == 80 and Counter(row["class"] for row in slots) == {"flash": 40, "black": 40},
            "DATA_REQUIRED: frozen slot population changed")

    raw_manifest = load(RAW / "formal_bank_A_manifest.json")
    msdf_manifest = load(MSDF / "formal_bank_A_manifest.json")
    require(raw_manifest["status"] == msdf_manifest["status"] == "COMPLETE" and
            raw_manifest["bank"] == msdf_manifest["bank"] == "A" and
            raw_manifest["count"] == msdf_manifest["count"] == 80,
            "DATA_REQUIRED: Raw/MSDF Bank A incomplete")
    raw_by_slot = {row["slot_id"]: row for row in raw_manifest["records"]}
    msdf_by_slot = {row["slot_id"]: row for row in msdf_manifest["records"]}
    require(set(raw_by_slot) == set(msdf_by_slot) == {row["slot_id"] for row in slots},
            "PROTOCOL_MISMATCH: Bank A slot IDs differ")

    gamma_by_value = {}
    for gamma in (0.0, 1.0):
        manifest = load(GAMMA / f"{gamma_tag(gamma)}_manifest.json")
        require(manifest["status"] == "COMPLETE" and manifest["count"] == 80 and manifest["gamma"] == gamma,
                f"DATA_REQUIRED: gamma={gamma} manifest incomplete")
        gamma_by_value[gamma] = {row["slot_id"]: row for row in manifest["records"]}
        require(set(gamma_by_value[gamma]) == set(raw_by_slot), f"PROTOCOL_MISMATCH: gamma={gamma} slots differ")

    adaptive = load(ADAPTIVE / "class_adaptive_manifest.json")
    require(adaptive["status"] == "COMPLETE" and adaptive["count"] == 80 and len(adaptive["records"]) == 80,
            "DATA_REQUIRED: local class-adaptive manifest incomplete")
    adaptive_by_slot = {row["slot_id"]: row for row in adaptive["records"]}
    require(set(adaptive_by_slot) == set(raw_by_slot), "PROTOCOL_MISMATCH: class-adaptive slots differ")

    raw_checkpoint = load(RAW / "generator_checkpoint_binding.json")["classes"]
    raw_image_checks = gamma_image_checks = slot_checks = 0
    for slot in slots:
        slot_id = slot["slot_id"]
        raw_row, msdf_row = raw_by_slot[slot_id], msdf_by_slot[slot_id]
        expected_rng = rngs[slot_id]
        expected_roi = crop_box(slot["bbox"], *slot["geometry"]["final_canvas_wh"])
        require(raw_row["frozen_slot"] == msdf_row["frozen_slot"] == slot,
                f"PROTOCOL_MISMATCH: frozen slot/normal/bbox/prompt changed: {slot_id}")
        for row in (raw_row, msdf_row):
            require(row["class"] == slot["class"] and row["bank"] == "A" and
                    row["sample_seed"] == expected_rng["sample_seed"] and row["bank_seed"] == 42,
                    f"PROTOCOL_MISMATCH: class/seed changed: {slot_id}")
        raw_image = RAW / raw_row["image_path"]
        require(raw_image.is_file() and sha(raw_image) == raw_row["image_sha256"],
                f"DATA_REQUIRED: Raw image changed: {raw_image}")
        raw_image_checks += 1
        for gamma in (0.0, 1.0):
            row = gamma_by_value[gamma][slot_id]
            image = GAMMA / row["image_path"]
            require(row["class"] == slot["class"] and row["sample_seed"] == expected_rng["sample_seed"] and
                    row["bank_seed"] == 42 and row["roi_xyxy"] == expected_roi,
                    f"PROTOCOL_MISMATCH: gamma condition changed: {gamma}:{slot_id}")
            require(image.is_file() and sha(image) == row["image_sha256"],
                    f"DATA_REQUIRED: gamma image changed: {image}")
            gamma_image_checks += 1
        chosen_gamma = 0.0 if slot["class"] == "flash" else 1.0
        chosen = gamma_by_value[chosen_gamma][slot_id]
        adaptive_row = adaptive_by_slot[slot_id]
        require(adaptive_row["class"] == slot["class"] and adaptive_row["bank"] == "A" and
                adaptive_row["gamma"] == chosen_gamma and adaptive_row["image_sha256"] == chosen["image_sha256"] and
                adaptive_row["sample_seed"] == expected_rng["sample_seed"] and
                adaptive_row["roi_xyxy"] == expected_roi,
                f"PROTOCOL_MISMATCH: adaptive binding changed: {slot_id}")
        slot_checks += 1

    checkpoint_distinction = {}
    for cls in CLASSES:
        raw_sha = raw_checkpoint[cls]["checkpoint"]["tree_sha256"]
        msdf_sha = protocol["checkpoints"][cls]["tree_sha256"]
        require(raw_sha != msdf_sha, f"PROTOCOL_MISMATCH: Raw/MSDF checkpoint unexpectedly identical: {cls}")
        checkpoint_distinction[cls] = {"raw_tree_sha256": raw_sha, "msdf_tree_sha256": msdf_sha,
                                       "different_checkpoints": True}

    visual = load(EXP1 / "visual_comparison_manifest.json")
    visual_a = {row["slot_id"]: row for row in visual["records"] if row["bank"] == "A"}
    require(len(visual_a) == 80 and set(visual_a) == set(raw_by_slot),
            "DATA_REQUIRED: Exp1 Raw feature population mismatch")
    for slot in slots:
        row, frozen = visual_a[slot["slot_id"]], raw_by_slot[slot["slot_id"]]
        expected_roi = crop_box(slot["bbox"], *slot["geometry"]["final_canvas_wh"])
        require(row["raw_image_sha256"] == frozen["image_sha256"] and row["roi_xyxy"] == expected_roi and
                row["conditioning_normal_id"] == slot["conditioning_normal_id"] and
                row["conditioning_normal_sha256"] == slot["normal_sha256"] and
                row["sample_seed"] == rngs[slot["slot_id"]]["sample_seed"] and row["bank_seed"] == 42,
                f"DATA_REQUIRED: Exp1 Raw feature binding mismatch: {slot['slot_id']}")

    raw_inputs = {
        "raw_bank_A_manifest": sha(RAW / "formal_bank_A_manifest.json"),
        "msdf_bank_A_manifest": sha(MSDF / "formal_bank_A_manifest.json"),
        "slots": sha(P0 / "stage20b_generation_slots.json"),
        "rng": sha(P0 / "stage20b_rng_manifest.json"),
        "exp1_visual_manifest": sha(EXP1 / "visual_comparison_manifest.json"),
        "exp1_feature_cache": sha(EXP1 / "inception_features.npz"),
        "gamma_protocol": sha(GAMMA / "gamma_sweep_protocol.json"),
        "gamma_feature_cache": sha(GAMMA / "gamma_sweep_inception_features.npz"),
        "gamma_0_manifest": sha(GAMMA / "gamma_0p00_manifest.json"),
        "gamma_1_manifest": sha(GAMMA / "gamma_1p00_manifest.json"),
        "class_adaptive_manifest": sha(ADAPTIVE / "class_adaptive_manifest.json"),
    }
    audit = {
        "status": "PASS", "checked_at": now(), "bank": "A", "matched_slot_count": 80,
        "class_counts": {"flash": 40, "black": 40}, "slot_condition_checks": slot_checks,
        "raw_image_sha256_checks": raw_image_checks, "gamma_image_sha256_checks": gamma_image_checks,
        "class": "PASS", "conditioning_normal": "PASS", "bbox": "PASS", "prompt": "PASS",
        "generation_seed": "PASS", "bank_seed": "PASS", "frozen_slot": "PASS", "ROI": "PASS",
        "checkpoint_distinction": checkpoint_distinction,
        "raw_artifact_bindings_checked": 3, "gamma_artifact_bindings_checked": gamma_artifact_checks,
        "images_generated": 0, "images_replaced": 0, "images_selected": 0,
        "training_count": 0, "historical_results_modified": False, "final_eval_read_or_used": False,
        "raw_input_SHA256": raw_inputs, "raw_input_binding_SHA256": canonical_sha(raw_inputs),
    }
    return protocol, slots, raw_by_slot, gamma_by_value, audit


def runtime_audit():
    exp = load(EXP1 / "metric_runtime.json")
    gamma = load(GAMMA / "gamma_sweep_metric_runtime.json")
    require(exp["package_versions"]["torch-fidelity"] == gamma["torch_fidelity"] == "0.3.0",
            "DATA_REQUIRED: torch-fidelity version mismatch")
    require(exp["weights"]["inception"]["sha256"] == gamma["weights"]["weights-inception-2015-12-05-6726825d.pth"]["sha256"] ==
            "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
            "DATA_REQUIRED: Inception weight mismatch")
    for name in ("torch_fidelity.feature_extractor_inceptionv3", "torch_fidelity.metric_fid", "torch_fidelity.metric_kid"):
        require(exp["metric_source_files"][name]["sha256"] == gamma["source_files"][name]["sha256"],
                f"DATA_REQUIRED: metric implementation mismatch: {name}")
    modules = ("torch_fidelity.feature_extractor_inceptionv3", "torch_fidelity.metric_fid", "torch_fidelity.metric_kid")
    current = {name: {"file": Path(importlib.import_module(name).__file__).name,
                      "sha256": sha(importlib.import_module(name).__file__)} for name in modules}
    for name in modules:
        require(current[name]["sha256"] == gamma["source_files"][name]["sha256"],
                f"DATA_REQUIRED: current metric implementation changed: {name}")
    return {
        "status": "PASS", "python": sys.version, "torch_fidelity": metadata.version("torch-fidelity"),
        "source_files": current, "inception_weight_sha256": exp["weights"]["inception"]["sha256"],
        "feature_extraction": "float32 eval/inference_mode; uint8 RGB; native torch-fidelity preprocessing",
        "metric_calculation": "float64 CPU", "KID_subsets": 100,
        "KID_subset_size": "min(50,nreal,nfake)", "KID_seed": 42,
        "KID_kernel": {"type": "polynomial", "degree": 3, "gamma": None, "coef0": 1.0},
        "ROI": "normalized bbox expanded 25% per side; floor lower, ceil upper, clipped",
        "new_inception_forward_count": 0, "network_download_count": 0,
    }


def load_feature_populations(protocol, slots, raw_by_slot, gamma_by_value):
    import numpy as np
    import torch

    exp_ledger = load(EXP1 / "inception_feature_cache.json")
    exp_cache = EXP1 / "inception_features.npz"
    require(exp_ledger["sha256"] == sha(exp_cache) and exp_ledger["feature_count"] == 1068 and
            exp_ledger["dimension"] == 2048 and
            exp_ledger["metric_runtime_sha256"] == sha(EXP1 / "metric_runtime.json"),
            "DATA_REQUIRED: Exp1 feature cache binding changed")
    with np.load(exp_cache, allow_pickle=False) as archive:
        exp_values = archive["features"].copy()
    require(exp_values.shape == (1068, 2048) and exp_values.dtype == np.float64 and bool(np.isfinite(exp_values).all()),
            "DATA_REQUIRED: invalid Exp1 features")
    exp_features = {key: exp_values[index] for index, key in enumerate(exp_ledger["keys"])}
    raw_keys = ["raw:roi:A:" + slot["slot_id"] for slot in slots]
    require(all(key in exp_features for key in raw_keys), "DATA_REQUIRED: Raw Bank-A ROI features missing")
    raw_array = np.stack([exp_features[key] for key in raw_keys])

    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw_cache = OUTPUT / "raw_bank_A_inception_features.npz"
    np.savez_compressed(raw_cache, features=raw_array)
    raw_cache_audit = {
        "status": "COMPLETE", "feature_count": 80, "dimension": 2048,
        "stored_dtype": str(raw_array.dtype), "feature_extraction_dtype": "float32",
        "keys": raw_keys, "sha256": sha(raw_cache),
        "source_cache": "Experiment 1 frozen Inception feature cache",
        "source_cache_sha256": sha(exp_cache), "source_cache_ledger_sha256": sha(EXP1 / "inception_feature_cache.json"),
        "source_images_sha256_verified": 80, "source_rois_verified": 80,
        "same_extractor_as_gamma_sweep": True, "new_inception_forward_count": 0,
    }
    save(OUTPUT / "raw_bank_A_inception_feature_cache.json", raw_cache_audit)

    gamma_ledger = load(GAMMA / "gamma_sweep_inception_feature_cache.json")
    gamma_cache = GAMMA / "gamma_sweep_inception_features.npz"
    require(gamma_ledger["sha256"] == sha(gamma_cache) and gamma_ledger["shape"] == [572, 2048],
            "DATA_REQUIRED: Gamma feature cache binding changed")
    expected = ["real:" + row["annotation_id"] for row in protocol["development_reference"]["records"]]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        manifest = load(GAMMA / f"{gamma_tag(value)}_manifest.json")
        expected.extend(f"fake:{value:.2f}:" + row["slot_id"] for row in manifest["records"])
    require(gamma_ledger["keys"] == expected, "DATA_REQUIRED: Gamma feature order changed")
    with np.load(gamma_cache, allow_pickle=False) as archive:
        gamma_values = archive["features"].copy()
    require(gamma_values.shape == (572, 2048) and gamma_values.dtype == np.float64 and
            bool(np.isfinite(gamma_values).all()), "DATA_REQUIRED: invalid Gamma features")
    gamma_features = {key: gamma_values[index] for index, key in enumerate(expected)}
    raw_features = {slot["slot_id"]: torch.from_numpy(raw_array[index]).double().cpu()
                    for index, slot in enumerate(slots)}
    return raw_features, {key: torch.from_numpy(value).double().cpu() for key, value in gamma_features.items()}, raw_cache_audit


def calculate_metrics(protocol, slots, raw_features, gamma_features):
    import torch
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    records = []
    for cohort in ("overall", *CLASSES):
        references = [row for row in protocol["development_reference"]["records"]
                      if cohort == "overall" or row["class"] == cohort]
        real = torch.stack([gamma_features["real:" + row["annotation_id"]] for row in references]).double().cpu()
        real_stats = fid_features_to_statistics(real)
        selected_slots = [slot for slot in slots if cohort == "overall" or slot["class"] == cohort]
        for method, group, setting in METHODS:
            fake_values = []
            for slot in selected_slots:
                choice = setting[slot["class"]]
                if choice == "raw":
                    fake_values.append(raw_features[slot["slot_id"]])
                else:
                    fake_values.append(gamma_features[f"fake:{choice:.2f}:" + slot["slot_id"]])
            fake = torch.stack(fake_values).double().cpu()
            options = {"kid_subsets": 100, "kid_subset_size": min(50, len(real), len(fake)),
                       "rng_seed": 42, "verbose": False, "kid_degree": 3, "kid_gamma": None, "kid_coef0": 1.0}
            started = time.monotonic()
            kid = kid_features_to_metric(real, fake, **options)
            fid = fid_statistics_to_metric(real_stats, fid_features_to_statistics(fake), verbose=False)
            row = {
                "method": method, "group": group, "class": cohort,
                "real_roi_count": len(real), "generated_roi_count": len(fake),
                "KID": float(kid["kernel_inception_distance_mean"]),
                "KID_subset_std": float(kid["kernel_inception_distance_std"]),
                "FID": float(fid["frechet_inception_distance"]),
                "KID_subsets": 100, "KID_subset_size": options["kid_subset_size"],
                "metric_runtime_seconds": time.monotonic() - started,
            }
            records.append(row)
            print(f"[指标] {group} {cohort}: KID={row['KID']:.10f} FID={row['FID']:.6f}", flush=True)
    return records


def regression_checks(records):
    def row(group, cohort="overall"):
        return next(item for item in records if item["group"] == group and item["class"] == cohort)

    prior_overall = {
        "G1": next(item for item in read_csv(GAMMA / "gamma_sweep_results.csv") if float(item["gamma"]) == 0.0),
        "G2": next(item for item in read_csv(GAMMA / "gamma_sweep_results.csv") if float(item["gamma"]) == 1.0),
        "G3": next(item for item in read_csv(ADAPTIVE / "class_adaptive_results.csv") if item["method"] == "Class-Adaptive gamma"),
    }
    expected_prompt = {
        "G1": {"KID": 0.0510316436, "FID": 166.880548},
        "G2": {"KID": 0.0515171327, "FID": 165.268961},
        "G3": {"KID": 0.0498916065, "FID": 165.131136},
    }
    checks = {}
    for group in ("G1", "G2", "G3"):
        source = prior_overall[group]
        source_kid = float(source["Overall_KID"])
        source_fid = float(source["Overall_FID"])
        current = row(group)
        record = {
            "KID_abs_error_vs_full_prior": abs(current["KID"] - source_kid),
            "FID_abs_error_vs_full_prior": abs(current["FID"] - source_fid),
            "KID_abs_error_vs_prompt_rounded": abs(current["KID"] - expected_prompt[group]["KID"]),
            "FID_abs_error_vs_prompt_rounded": abs(current["FID"] - expected_prompt[group]["FID"]),
        }
        require(record["KID_abs_error_vs_full_prior"] <= 1e-12 and record["FID_abs_error_vs_full_prior"] <= 1e-10,
                f"NUMERICAL_INCONSISTENCY: {group}: {record}")
        require(record["KID_abs_error_vs_prompt_rounded"] <= 5e-11 and
                record["FID_abs_error_vs_prompt_rounded"] <= 5e-7,
                f"NUMERICAL_INCONSISTENCY_ROUNDED: {group}: {record}")
        checks[group] = record
    checks["status"] = "PASS"
    return checks


def percent(delta, baseline):
    return 100.0 * delta / baseline


def publish(protocol, audit, runtime, raw_cache_audit, records, regression, wall_seconds):
    def metric(group, cohort="overall"):
        return next(row for row in records if row["group"] == group and row["class"] == cohort)

    result_rows, class_rows = [], []
    for method, group, setting in METHODS:
        overall = metric(group)
        result_rows.append({
            "group": group, "method": method, "gamma_setting": json.dumps(setting, sort_keys=True),
            "generated_images": 80, "reference_rois": 172,
            "Overall_KID": overall["KID"], "Overall_KID_subset_std": overall["KID_subset_std"],
            "Overall_FID": overall["FID"], "metric_runtime_seconds": overall["metric_runtime_seconds"],
            "Raw_input_SHA256": audit["raw_input_binding_SHA256"],
            "reference_source": "V2 TRAIN; 172 ROIs; possible checkpoint-training overlap",
            "historical_results_modified": False,
        })
        for cls in CLASSES:
            value = metric(group, cls)
            class_rows.append({
                "group": group, "method": method, "class": cls.capitalize(),
                "gamma": setting[cls], "generated_images": 40,
                "reference_rois": value["real_roi_count"], "KID": value["KID"],
                "KID_subset_std": value["KID_subset_std"], "FID": value["FID"],
                "metric_runtime_seconds": value["metric_runtime_seconds"],
                "Raw_input_SHA256": audit["raw_input_binding_SHA256"],
                "historical_results_modified": False,
            })
    write_csv(OUTPUT / "raw_matched_results.csv", list(result_rows[0]), result_rows)
    write_csv(OUTPUT / "raw_matched_classwise.csv", list(class_rows[0]), class_rows)

    comparisons = {}
    raw = {"Overall_KID": metric("G0")["KID"], "Overall_FID": metric("G0")["FID"],
           "Flash_KID": metric("G0", "flash")["KID"], "Flash_FID": metric("G0", "flash")["FID"],
           "Black_KID": metric("G0", "black")["KID"], "Black_FID": metric("G0", "black")["FID"]}
    for group in ("G1", "G2", "G3"):
        current = {"Overall_KID": metric(group)["KID"], "Overall_FID": metric(group)["FID"],
                   "Flash_KID": metric(group, "flash")["KID"], "Flash_FID": metric(group, "flash")["FID"],
                   "Black_KID": metric(group, "black")["KID"], "Black_FID": metric(group, "black")["FID"]}
        comparisons[f"{group}_minus_Raw"] = {
            name: {"absolute": current[name] - raw[name],
                   "relative_percent": percent(current[name] - raw[name], raw[name])}
            for name in raw
        }
    g2 = {"Overall_KID": metric("G2")["KID"], "Overall_FID": metric("G2")["FID"],
          "Flash_KID": metric("G2", "flash")["KID"], "Flash_FID": metric("G2", "flash")["FID"],
          "Black_KID": metric("G2", "black")["KID"], "Black_FID": metric("G2", "black")["FID"]}
    g3 = {"Overall_KID": metric("G3")["KID"], "Overall_FID": metric("G3")["FID"],
          "Flash_KID": metric("G3", "flash")["KID"], "Flash_FID": metric("G3", "flash")["FID"],
          "Black_KID": metric("G3", "black")["KID"], "Black_FID": metric("G3", "black")["FID"]}
    comparisons["G3_minus_G2"] = {
        name: {"absolute": g3[name] - g2[name], "relative_percent": percent(g3[name] - g2[name], g2[name])}
        for name in g2
    }

    kid_improved = g3["Overall_KID"] < raw["Overall_KID"]
    fid_improved = g3["Overall_FID"] < raw["Overall_FID"]
    if kid_improved and fid_improved:
        conclusion = "EXPLORATORY_BOTH_METRICS_IMPROVED"
    elif kid_improved or fid_improved:
        conclusion = "EXPLORATORY_MIXED_RESULT"
    else:
        conclusion = "EXPLORATORY_NOT_IMPROVED"
    flash_kid_improved = g3["Flash_KID"] < raw["Flash_KID"]
    flash_fid_improved = g3["Flash_FID"] < raw["Flash_FID"]
    black_kid_worse = g3["Black_KID"] > raw["Black_KID"]
    black_fid_worse = g3["Black_FID"] > raw["Black_FID"]
    comparison = {
        "status": conclusion, "numerical_regression": regression,
        "differences": comparisons,
        "answers": {
            "G3_Overall_KID_lower_than_Raw": kid_improved,
            "G3_Overall_FID_lower_than_Raw": fid_improved,
            "G3_Black_KID_worse_than_Raw": black_kid_worse,
            "G3_Black_FID_worse_than_Raw": black_fid_worse,
            "G3_Flash_KID_better_than_Raw": flash_kid_improved,
            "G3_Flash_FID_better_than_Raw": flash_fid_improved,
            "G3_vs_G2_improvement_source": "Flash only; G3 and G2 use identical gamma=1 Black images/features.",
            "overall_KID_improved_while_Black_KID_worsened_vs_Raw": kid_improved and black_kid_worse,
        },
        "interpretation": "Exploratory Bank-A/V2-TRAIN comparison; TRAIN may overlap generator training and class gamma was development-selected.",
        "recommended_next_step": (
            "Freeze the class-adaptive setting and test it with source-group-isolated validation and multiple banks."
            if conclusion == "EXPLORATORY_BOTH_METRICS_IMPROVED" else
            "Stop gamma tuning and move to training-level morphology-guidance or dual-branch-fusion ablation."
        ),
    }
    save(OUTPUT / "raw_matched_comparison.json", comparison)
    protocol_out = {
        "status": "COMPLETE", "experiment": "Experiment 7 — Raw SD2 vs Class-Adaptive MSDF",
        "groups": {group: {"method": method, "setting": setting} for method, group, setting in METHODS},
        "bank": "A", "frozen_slots": 80, "class_counts": {"flash": 40, "black": 40},
        "training_performed": False, "generation_performed": False, "selection_performed": False,
        "reference": {"source": "V2 TRAIN", "ROI_count": 172, "class_counts": {"flash": 75, "black": 97},
                      "checkpoint_training_overlap_possible": True, "independent_validation": False,
                      "final_eval_read_or_used": False},
        "metrics": {"primary": "ROI KID", "auxiliary": "ROI FID", **runtime},
        "overall_rule": "Each Overall metric is recomputed from all 80 generated features against all 172 real features; class metrics are never averaged.",
        "feature_reuse": {"Raw_SD2": "80 SHA-bound Bank-A ROI features selected from Experiment 1 cache",
                          "MSDF_and_real": "SHA-bound Gamma Sweep cache", "new_Inception_forwards": 0},
        "raw_feature_cache": {"file": "raw_bank_A_inception_features.npz", "sha256": raw_cache_audit["sha256"]},
        "raw_input_SHA256": audit["raw_input_SHA256"],
        "implementation_SHA256": sha(Path(__file__)),
        "historical_results_modified": False,
        "prohibited_followup_started": False,
    }
    save(OUTPUT / "raw_matched_protocol.json", protocol_out)
    save(OUTPUT / "raw_matched_integrity_audit.json", audit)
    save(OUTPUT / "raw_matched_status.json", {
        "status": conclusion, "completed_at": now(), "method_count": 4,
        "matched_slot_count": 80, "reference_roi_count": 172,
        "new_Inception_forward_count": 0, "training_count": 0, "generation_count": 0,
        "image_replacement_count": 0, "selection_count": 0,
        "historical_results_modified": False, "final_eval_read_or_used": False,
        "metric_runtime_seconds": wall_seconds, "numerical_regression": "PASS",
        "expanded_gamma_search_started": False, "MSDF_module_modified": False,
        "BootstrapGuard": False, "detector_training_started": False,
    })

    lines = [
        "# Experiment 7 — Raw SD2 vs Class-Adaptive MSDF", "",
        f"Status: `{conclusion}`.", "",
        "No training, generation, replacement, filtering, gamma search, MSDF modification, BootstrapGuard, or detector training was performed.", "",
        "## Matched protocol", "",
        "All four groups use the same 80 frozen Bank-A slots (Flash 40, Black 40), conditions, bboxes, prompts, sample seeds, bank seed and ROI geometry. Raw SD2 uses its independently trained Raw checkpoint. MSDF gamma=0 only disables the final residual contribution and is not Raw SD2.", "",
        "All metrics use the same 172 V2 TRAIN real ROI features. Overall KID/FID was recomputed from each complete 80-image distribution. Raw features were reused only after the Experiment 1 cache, images, slots and crops passed SHA/binding checks; no new Inception forward was needed.", "",
        "## Complete matched results", "",
        "| Group | Method | Overall KID ↓ | Overall FID ↓ | Flash KID ↓ | Flash FID ↓ | Black KID ↓ | Black FID ↓ |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, group, _setting in METHODS:
        lines.append(f"| {group} | {method} | {metric(group)['KID']:.10f} | {metric(group)['FID']:.6f} | {metric(group,'flash')['KID']:.10f} | {metric(group,'flash')['FID']:.6f} | {metric(group,'black')['KID']:.10f} | {metric(group,'black')['FID']:.6f} |")
    lines += ["", "## Differences", "",
              "Negative values mean the first method is lower/better for KID or FID.", "",
              "| Comparison | Δ Overall KID | Relative KID | Δ Overall FID | Relative FID | Δ Flash KID | Δ Black KID |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("G1_minus_Raw", "G2_minus_Raw", "G3_minus_Raw", "G3_minus_G2"):
        value = comparisons[name]
        lines.append(f"| {name.replace('_',' ')} | {value['Overall_KID']['absolute']:+.10f} | {value['Overall_KID']['relative_percent']:+.3f}% | {value['Overall_FID']['absolute']:+.6f} | {value['Overall_FID']['relative_percent']:+.3f}% | {value['Flash_KID']['absolute']:+.10f} | {value['Black_KID']['absolute']:+.10f} |")
    black_kid_delta = comparisons["G3_minus_Raw"]["Black_KID"]
    black_fid_delta = comparisons["G3_minus_Raw"]["Black_FID"]
    flash_kid_delta = comparisons["G3_minus_Raw"]["Flash_KID"]
    flash_fid_delta = comparisons["G3_minus_Raw"]["Flash_FID"]
    lines += ["", "## Required findings", "",
              f"1. G3 Overall KID {'is' if kid_improved else 'is not'} lower than Raw SD2 ({g3['Overall_KID']:.10f} vs {raw['Overall_KID']:.10f}).",
              f"2. G3 Overall FID {'is' if fid_improved else 'is not'} lower than Raw SD2 ({g3['Overall_FID']:.6f} vs {raw['Overall_FID']:.6f}).",
              f"3. Black KID is {black_kid_delta['absolute']:+.10f} ({black_kid_delta['relative_percent']:+.3f}%) and Black FID is {black_fid_delta['absolute']:+.6f} ({black_fid_delta['relative_percent']:+.3f}%) relative to Raw.",
              f"4. Flash KID is {flash_kid_delta['absolute']:+.10f} ({flash_kid_delta['relative_percent']:+.3f}%) and Flash FID is {flash_fid_delta['absolute']:+.6f} ({flash_fid_delta['relative_percent']:+.3f}%) relative to Raw.",
              "5. G3's change from G2 comes entirely from Flash because both groups use the same gamma=1 Black images and features.",
              f"6. Overall KID {'improves' if kid_improved else 'does not improve'} while Black KID {'worsens' if black_kid_worse else 'does not worsen'} versus Raw.", "",
              "## Interpretation boundary and next step", "",
              "This is a Bank-A development comparison using V2 TRAIN references that may overlap generator training. The class gamma was selected on this development reference. These values are not directly comparable to historical final_eval and do not establish final-paper superiority or SOTA.", "",
              comparison["recommended_next_step"], "",
              "The experiment stops here; no follow-up experiment was started.", ""]
    (OUTPUT / "RAW_MATCHED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.monotonic()
    if OUTPUT.exists():
        raise RuntimeError(f"refusing existing output directory: {OUTPUT}")
    print(f"[启动] Experiment 7 matched Raw/MSDF evaluation PID={__import__('os').getpid()}；不训练、不生成。", flush=True)
    protocol, slots, raw_by_slot, gamma_by_value, audit = audit_inputs()
    print("[完整性] Raw/MSDF 80个Bank-A槽位及240张相关源图SHA/条件绑定通过。", flush=True)
    runtime = runtime_audit()
    raw_features, gamma_features, raw_cache_audit = load_feature_populations(
        protocol, slots, raw_by_slot, gamma_by_value
    )
    print("[特征] 已从SHA绑定缓存建立Raw Bank-A 80×2048缓存；新增Inception forward=0。", flush=True)
    records = calculate_metrics(protocol, slots, raw_features, gamma_features)
    regression = regression_checks(records)
    print("[回归] G1/G2/G3与冻结结果一致。", flush=True)
    publish(protocol, audit, runtime, raw_cache_audit, records, regression, time.monotonic() - started)
    print("[完成] Experiment 7报告完成；停止，不启动后续实验。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
