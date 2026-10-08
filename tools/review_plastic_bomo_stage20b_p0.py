"""Review downloaded P0 metadata only; never open data or load/run models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def htext(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ledger(root, filename):
    root = Path(root).resolve()
    rows = []
    for name, expected in load(root / filename)["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("LEDGER_PATH_OUTSIDE_RESULT_DIRECTORY: " + name)
        actual = sha(path) if path.is_file() else None
        rows.append({"file": name, "expected_sha256": expected,
                     "actual_sha256": actual, "pass": actual == expected})
    return {"files": rows, "count": len(rows), "pass": bool(rows) and all(x["pass"] for x in rows)}


def remove_fields(row, fields):
    return {key: value for key, value in row.items() if key not in fields}


def review(repo, server, local, stage20a, protocol):
    cfg = load(protocol)
    local_files = ledger(local, "p0_frozen_artifact_manifest.json")
    server_files = ledger(server, "p0_frozen_artifact_manifest.json")
    stage20a_files = ledger(stage20a, "frozen_artifact_manifest.json")
    checks = {"local_frozen_files": local_files["pass"],
              "downloaded_server_files": server_files["pass"],
              "stage20a_frozen_files": stage20a_files["pass"]}
    s = load(server / "stage20b_p0_status.json")
    zero_fields = ("generator_training_count", "synthetic_generation_count", "detector_training_count",
                   "official_final_eval_forward", "final_eval_image_or_label_content_read",
                   "DeepPCB", "BootstrapGuard", "TDCRG_implementation")
    checks["server_structural_authorization_only"] = (
        s["status"] == "STAGE20B_STRUCTURAL_PREFLIGHT_PASS" and s["scope"] == "SERVER_PREFLIGHT"
        and s["STAGE20B_EXECUTION_AUTHORIZED"] is True and s["local_structural_preflight_pass"] is True
        and s["runtime_preflight_is_not_forward_test"] is True
        and all(s.get(k) == 0 for k in zero_fields)
        and all(s.get(k) is False for k in ("TDCRG_DEVELOPMENT_AUTHORIZED", "Stage20A_modified", "stage20b_auto_start")))
    checks["protocol_and_executed_script_identity"] = (
        s["protocol_sha256"] == sha(protocol)
        and s["script_sha256"] == sha(repo / "tools/preflight_plastic_bomo_stage20b_p0.py"))
    binding = load(server / "stage20b_stage20a_binding_audit.json")
    frozen_a = load(stage20a / "frozen_artifact_manifest.json")["sha256"]
    checks["authoritative_stage20a_binding"] = (
        sha(stage20a / "frozen_artifact_manifest.json") == cfg["stage20a_frozen_manifest_sha256"]
        and binding["status"] == "PASS" and binding["stage20a_unchanged"] is True
        and binding["stage20a_commit"] == cfg["stage20a_commit"]
        and set(binding["artifacts"]) == set(frozen_a)
        and all(row["expected_sha256"] == row["actual_sha256"] == frozen_a[name]
                for name, row in binding["artifacts"].items()))
    local_binding = load(server / "stage20b_local_p0_binding_audit.json")
    checks["local_freeze_binding"] = (
        local_binding["status"] == "PASS"
        and local_binding["local_manifest_sha256"] == sha(local / "p0_frozen_artifact_manifest.json")
        and all(local_binding[k] is True for k in ("config_bytes_identical", "all_80_slot_semantics_identical", "path_relocation_only")))
    slots = load(server / "stage20b_generation_slots.json")["slots"]
    lslots = load(local / "stage20b_generation_slots.json")["slots"]
    checks["80_slots_same_semantics_except_runtime_paths"] = (
        len(slots) == 80 and Counter(x["class"] for x in slots) == cfg["budget"]
        and len({x["slot_id"] for x in slots}) == len({x["source_group_id"] for x in slots}) == 80
        and [remove_fields(x, ("source_path", "normal_path")) for x in slots]
        == [remove_fields(x, ("source_path", "normal_path")) for x in lslots])
    split = load(server / "stage20b_split_recheck.json")
    checks["same_v2_split_no_known_leakage"] = (
        split == load(local / "stage20b_split_recheck.json") and split["status"] == "PASS"
        and not any(split[k] for k in ("cross_source_groups", "cross_rgb_aliases", "cross_known_edges"))
        and split["physical_source_complete"] is False and split["final_eval_content_read"] == 0)
    allow = load(server / "generator_train_allowlist.json")
    lallow = load(local / "generator_train_allowlist.json")
    checks["172_train_annotations_666_verified_assets"] = (
        allow["status"] == "PASS" and len(allow["inputs"]) == 172
        and Counter(x["class"] for x in allow["inputs"]) == cfg["expected"]["train_boxes"]
        and all(x["split"] == "train" for x in allow["inputs"])
        and allow["source_assets_hash_verified"] == s["TRAIN_assets_hash_verified"] == 666
        and all(allow[k] == 0 for k in ("final_eval_input_count", "unknown_split_input_count", "quality_filter_count"))
        and [remove_fields(x, ("path", "full_label_path")) for x in allow["inputs"]]
        == [remove_fields(x, ("path", "full_label_path")) for x in lallow["inputs"]])
    normals = load(server / "stage20b_normal_pool_freeze.json")
    lnormals = load(local / "stage20b_normal_pool_freeze.json")
    checks["394_identical_train_normals_128_excluded"] = (
        normals["status"] == "PASS" and normals["count"] == len(normals["records"]) == 394
        and normals["excluded_count"] == 128 and normals["pool_changed"] is False
        and [remove_fields(x, ("runtime_path",)) for x in normals["records"]]
        == [remove_fields(x, ("runtime_path",)) for x in lnormals["records"]])
    checks["identical_mask_geometry_and_inference_protocol"] = all(
        load(server / name) == load(local / name)
        for name in ("stage20b_geometry_audit.json", "stage20b_generation_protocol.json", "stage20b_future_baseline_gate.json"))
    rr = load(server / "stage20b_realrepeat_manifest.json")
    lrr = load(local / "stage20b_realrepeat_manifest.json")
    checks["rr_slotwise_full_images_and_complete_labels"] = (
        len(rr["records"]) == 80 and rr["replay_exposures"] == rr["unique_donor_images"] == 80
        and rr["max_donor_reuse"] == 1 and rr["class_role_counts"] == cfg["budget"]
        and all(x["bbox_crop_replay"] is False and x["split"] == "train" for x in rr["records"])
        and [remove_fields(x, ("full_image_path", "full_labels_path")) for x in rr["records"]]
        == [remove_fields(x, ("full_image_path", "full_labels_path")) for x in lrr["records"]])
    rng = load(server / "stage20b_rng_manifest.json")
    pairing = load(server / "stage20b_bank_pairing_audit.json")
    slot_by = {x["slot_id"]: x for x in slots}
    rng_ok = len(rng["records"]) == len(pairing["records"]) == 80
    rng_ok &= rng["torch_Generator_instantiated"] is False and rng["generation_count"] == 0
    rng_ok &= {x["slot_id"] for x in rng["records"]} == set(slot_by)
    for row in rng["records"]:
        expected_hash = htext(canonical(slot_by[row["slot_id"]]))
        rng_ok &= set(row["banks"]) == set(cfg["banks"])
        for bank, seed in cfg["banks"].items():
            expected_seed = int(htext("Plastic_Bomo_Generation_V2" + row["slot_id"] + str(seed))[:16], 16)
            rng_ok &= row["banks"][bank] == {"bank_seed": seed, "sample_seed": expected_seed,
                                           "non_rng_metadata_sha256": expected_hash}
    rng_ok &= {x["slot_id"] for x in pairing["records"]} == set(slot_by)
    rng_ok &= all(x["non_rng_metadata_sha256"] == htext(canonical(slot_by[x["slot_id"]]))
                  and x["all_non_rng_fields_identical"] is True and x["only_RNG_varies"] is True
                  for x in pairing["records"])
    checks["80_paired_slots_three_predeclared_rng_banks"] = bool(rng_ok)
    order = load(server / "stage20b_training_order_spec.json")
    source = load(stage20a / "real_source_registry.json")["records"]
    roles = {"base_" + x["node_id"] for x in source if x["asset_role"] == "detector" and x["split_v2"] == "train"}
    roles |= {"extra_" + x["slot_id"] for x in slots}
    order_ok = order == load(local / "stage20b_training_order_spec.json") and len(order["epochs"]) == 150
    for epoch, row in enumerate(order["epochs"]):
        sequence = sorted(roles, key=lambda r: (htext(cfg["protocol"] + "42" + str(epoch) + r), r))
        seeds = [int(htext(cfg["protocol"] + "augmentation42" + str(epoch) + str(i))[:16], 16) for i in range(216)]
        order_ok &= (row["epoch_0based"] == epoch and row["role_sequence"] == sequence
                     and row["sequence_sha256"] == htext(canonical(sequence)) and row["augmentation_seeds"] == seeds
                     and row["optimizer_attempt_after_positions_0based"] == [63, 127, 191, 215]
                     and row["accumulation_group_lengths"] == [64, 64, 64, 24])
    order_ok &= order["scheduled_draws"] == order["scheduled_batches"] == 32400
    order_ok &= order["scheduled_optimizer_attempts"] == 600
    checks["150_epoch_shared_role_rng_attempt_schedule"] = bool(order_ok)
    base = load(server / "sd2_base_weight_binding.json")
    checks["public_sd2_weight_and_metadata_binding"] = (
        base["status"] == "PASS" and not base["errors"] and base["loaded_as_model"] is False
        and base["public_reference"] == cfg["base_public_reference"]
        and all(base["files"][name]["sha256"] == ref["sha256"] and base["files"][name]["size"] == ref["size"]
                for name, ref in cfg["base_public_reference"]["weight_files"].items())
        and all(base["files"][name]["actual_git_blob"] == ref
                for name, ref in cfg["base_public_reference"]["metadata_git_blobs"].items()))
    detector = load(server / "detector_initialization_binding.json")
    checks["same_original_pretrained_yolo11s"] = (
        detector["status"] == "PASS" and detector["sha256"] == cfg["detector_weight_sha256"]
        and detector["model_loaded"] is False and detector["same_for_all_arms"] is True)
    runtime = load(server / "stage20b_runtime_binding.json")
    env = runtime["environment"]
    required = ("torch", "ultralytics", "transformers", "accelerate", "safetensors", "numpy", "PyYAML")
    checks["recorded_runtime_metadata_and_pinned_generator_code"] = (
        runtime["scope"] == "SERVER_PREFLIGHT" and runtime["imports_models"] is False
        and env["model_imported"] is False and all(env["versions"].get(k) for k in required)
        and all(x["actual_sha256"] == x["sha256"] == sha(repo / x["path"]) for x in runtime["generator_code"]))
    train = load(server / "stage20b_generator_training_protocol.json")
    det = load(server / "stage20b_detector_protocol.json")
    resolved = load(server / "stage20b_detector_resolved_runtime_arguments.json")
    checks["frozen_training_and_detector_arguments"] = (
        all(train.get(k) == v for k, v in cfg["generator_training"].items())
        and train["allowlist_sha256"] == sha(server / "generator_train_allowlist.json")
        and det["training"] == cfg["detector"] and det["evaluation"] == cfg["detector_evaluation"]
        and det["runtime_binding_sha256"] == sha(server / "stage20b_runtime_binding.json")
        and resolved["all_remaining_defaults_materialized_and_frozen"] is True
        and resolved["default_yaml_sha256"] == env["source_files"]["ultralytics/cfg/default.yaml"]["sha256"]
        and all(resolved["arguments"][k] == cfg["detector"][k] for k in resolved["protocol_override_fields"])
        and all(v == cfg["detector"][k] for k, v in resolved["custom_enforced_rules"].items()))
    return {"status": "STAGE20B_P0_SERVER_REVIEW_PASS" if all(checks.values()) else "STAGE20B_P0_SERVER_REVIEW_FAILED",
            "authoritative_server_status": s["status"], "server_directory": server.relative_to(repo).as_posix(),
            "checks": checks, "hash_ledgers": {"Stage20A": stage20a_files, "local_P0": local_files, "server_P0": server_files},
            "server_status": s, "runtime_versions": env["versions"],
            "runtime_source_hashes": {k: v["sha256"] for k, v in env["source_files"].items()},
            "review_scope": "downloaded metadata integrity and cross-artifact consistency; no remote assets or models accessed",
            "gpu_forward_or_numerical_determinism_verified": False,
            "trained_checkpoint_bytes_bound": False, "physical_source_complete": False,
            "official_final_eval_content_read_by_review": 0, "automatic_formal_execution": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("server", "local", "stage20a", "protocol", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("."))
    a = p.parse_args()
    repo = a.repo.resolve()
    if a.output.exists():
        raise RuntimeError("REVIEW_OUTPUT_EXISTS_NO_OVERWRITE: " + str(a.output))
    result = review(repo, a.server.resolve(), a.local.resolve(), a.stage20a.resolve(), a.protocol.resolve())
    result["review_script_sha256"] = sha(Path(__file__))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "server_review_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "hash_ledger_counts": {k: v["count"] for k, v in result["hash_ledgers"].items()}}, indent=2))
    if not all(result["checks"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
