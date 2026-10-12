"""Read-only downloaded A1 review; never opens dev/final-eval image or label content."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import finalize_plastic_bomo_stage21b_a1 as F
import stage21b_a1_common as C


def independent_results(metrics):
    C.require(set(metrics) == set(C.ARMS), "REVIEW_INCOMPLETE_ARMS")
    for row in metrics.values():
        values = [row["map5095"], *(row["per_class"][c]["ap5095"] for c in ("flash", "black"))]
        C.require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v) and 0 <= v <= 1 for v in values), "REVIEW_INVALID_METRICS")
    mean = lambda xs: math.fsum(xs) / len(xs)
    delta = {b: 100 * (metrics[b + "_curriculum"]["map5095"] - metrics[b + "_uniform"]["map5095"])
             for b in C.BANKS}
    classwise = {c: {b: 100 * (metrics[b + "_curriculum"]["per_class"][c]["ap5095"]
                                - metrics[b + "_uniform"]["per_class"][c]["ap5095"])
                    for b in C.BANKS} for c in ("flash", "black")}
    classes = {c: mean(list(v.values())) for c, v in classwise.items()}
    values = {mode: [100 * metrics[b + "_" + mode]["map5095"] for b in C.BANKS]
              for mode in ("uniform", "curriculum")}
    sd = {mode: math.sqrt(math.fsum((v - mean(vs)) ** 2 for v in vs) / (len(vs) - 1))
          for mode, vs in values.items()}
    gates = {"Gate_A": mean(list(delta.values())) >= .5, "Gate_B": sum(v > 0 for v in delta.values()) >= 2,
             "Gate_C": all(v >= -2 for v in classes.values()), "Gate_D": sd["curriculum"] <= sd["uniform"]}
    return {"status": "SYNTHETIC_EXPOSURE_TIMING_PREMISE_SUPPORTED" if all(gates.values())
                       else "SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED",
            **gates, **{f"delta_{b}_pp": v for b, v in delta.items()}, "mean_delta_pp": mean(list(delta.values())),
            "Flash_mean_delta_pp": classes["flash"], "Black_mean_delta_pp": classes["black"],
            "uniform_bank_std_pp": sd["uniform"], "curriculum_bank_std_pp": sd["curriculum"],
            "strict_bank_wins": sum(v > 0 for v in delta.values()), "paired_class_deltas_pp": classwise}


def verify_ledger(root, name):
    doc = C.load(Path(root) / name)
    for relative, expected in doc["sha256"].items():
        C.require(C.sha(C.safe_relative(root, relative)) == expected, "REVIEW_ARTIFACT_CHANGED: " + relative)
    return doc


def metadata_local_path(value):
    portable = value.replace("\\", "/")
    if portable.startswith("results_for_gpt/"):
        relative = portable
    else:
        C.require("/results_for_gpt/" in portable, "REVIEW_UNEXPECTED_METADATA_PATH")
        relative = "results_for_gpt/" + portable.split("/results_for_gpt/", 1)[1]
    return C.safe_relative(C.REPO, relative)


def review(run):
    run = Path(run).resolve()
    ledger = verify_ledger(run, "stage21b_result_manifest.json")
    C.require(ledger["full_experiment_complete"] is True, "REVIEW_INCOMPLETE_EXPERIMENT")
    C.verify_prepared(run)
    original = run.parent
    C.require(C.sha(original / "stage21b_artifact_manifest.json") == C.ORIGINAL_LEDGER_SHA,
              "REVIEW_ORIGINAL_AUDIT_CHANGED")
    verify_ledger(original, "stage21b_artifact_manifest.json")
    C.require(C.sha(run / "stage21b_split_manifest.json") == C.sha(original / "stage21b_split_manifest.json"),
              "REVIEW_SPLIT_CHANGED")
    evidence = C.load(run / "source_evidence_index_A1.json")
    for relative, expected in evidence["code_sha256"].items():
        C.require(C.sha(C.safe_relative(C.REPO, relative)) == expected, "REVIEW_EXECUTABLE_CHANGED: " + relative)
    for row in evidence["records"]:
        C.require(row["kind"] in ("metadata_only", "metadata_ledger")
                  and C.sha(metadata_local_path(row["file"])) == row["sha256"], "REVIEW_SOURCE_EVIDENCE_CHANGED")
    C.require(C.sha(C.REPO / "configs/plastic_bomo_stage21b_a1.json") == evidence["config_sha256"],
              "REVIEW_CONFIG_CHANGED")
    cfg = C.load(run / "stage21b_a1_protocol_snapshot.json"); C.check_cfg(cfg)
    base = C.load(run / "stage21b_base_protocol_snapshot.json")
    split = C.load(run / "stage21b_split_manifest.json")
    dev = {r["source_group_id"] for r in split["groups"] if r["split"] == "internal_dev"}
    allowed = C.amended_allowlist(C.load(original / "stage21b_synthetic_allowlist.json"), dev)
    C.require(allowed == C.load(run / "stage21b_synthetic_allowlist_v2.json"), "REVIEW_ALLOWLIST_CHANGED")
    real_roles = [r["role_id"] for r in split["records"] if r["internal_split"] == "internal_train"]
    u, c, pairing, equality = C.make_schedules(real_roles, [r["role_id"] for r in allowed["records"]], base, cfg)
    for name, doc in (("uniform_schedule_v2", u), ("curriculum_schedule_v2", c),
                      ("schedule_pairing_audit_v2", pairing), ("exposure_equality_audit_v2", equality)):
        C.require(doc == C.load(run / (name + ".json")), "REVIEW_SCHEDULE_CHANGED: " + name)
    exposure = C.load(run / "stage21b_generator_exposure_limitation.json")
    C.require(exposure["generator_annotation_inputs"] == 33 and exposure["per_class"] == {"black": 23, "flash": 10}
              and exposure["generator_internal_dev_historical_exposure"] is True
              and exposure["PIPELINE_INDEPENDENT_HOLDOUT"] is False, "REVIEW_GENERATOR_LIMITATION_CHANGED")
    assets = C.load(run / "server_source_asset_hash_audit.json")
    C.require(assets["status"] == "TRAIN_AND_SYNTHETIC_HASH_VERIFIED" and assets["synthetic_assets_verified"] == 366
              and assets["real_TRAIN_assets_verified"] == 218 and assets["internal_dev_content_read"] == 0
              and assets["official_final_eval_content_read"] == 0, "REVIEW_SOURCE_ASSET_AUDIT_FAILED")
    # Independently hash all six downloaded weights, without torch.load or any forward.
    checkpoints = C.checkpoint_barrier(run)
    C.require(checkpoints == C.load(run / "detector_checkpoint_binding.json"), "REVIEW_CHECKPOINT_CHANGED")
    # Existing trace auditor has one save call: capture it rather than rewrite frozen server output.
    with patch.object(F, "save"):
        actual_pair = F.inspect_draw_logs(run)
    C.require(actual_pair == C.load(run / "augmentation_pairing_audit.json"), "REVIEW_DRAW_AUDIT_CHANGED")
    initial = C.digest(C.load(run / "initialization_reference.json"))
    training = {arm: C.load(run / (arm + "_training.json")) for arm in C.ARMS}
    for arm, tr in training.items():
        C.require(tr["initialization_sha256"] == initial and tr["EMA_updates"] == 450
                  and tr["nonfinite_losses"] == 0 and tr["internal_dev_content_read"] == 0
                  and tr["official_final_eval_content_read"] == 0
                  and not tr["early_stopping_used"] and not tr["best_checkpoint_selection"], "REVIEW_TRAINING_INVALID")
    reference = training[C.ARMS[0]]
    C.require(all(tr["optimizer_groups"] == reference["optimizer_groups"] for tr in training.values()),
              "REVIEW_OPTIMIZER_CHANGED")
    lr = lambda tr: [q["learning_rates"] for e in tr["epoch_records"] for q in e["attempts"]]
    C.require(all(lr(tr) == lr(reference) for tr in training.values()), "REVIEW_LEARNING_RATE_PAIRING_FAILED")
    safety = C.load(run / "single_batch_safety_audit.json")
    C.require(safety["status"] == "PASS" and safety["optimizer_step_count"] == 0 and safety["internal_train_only"]
              and set(safety["schedules"]) == {"uniform", "curriculum"}, "REVIEW_SAFETY_FAILED")
    for row in safety["schedules"].values():
        C.require(row["status"] == "PASS" and row["optimizer_steps"] == 0 and row["finite_gradients"]
                  and row["nonzero_gradients"] and row["parameters_unchanged"]
                  and math.isfinite(row["loss"]) and row["initialization_sha256"] == initial, "REVIEW_SAFETY_FAILED")
    guards = {tag: C.load(run / (tag + "_file_access_guard.json"))
              for tag in ("parent", "safety", *C.ARMS, "internal_dev")}
    zero_keys = ("official_blocked_open_attempts", "internal_dev_blocked_open_attempts",
                 "official_final_eval_image_read", "official_final_eval_label_read", "official_final_eval_forward")
    C.require(all(all(g[k] == 0 for k in zero_keys) for g in guards.values()), "REVIEW_ACCESS_GUARD_VIOLATION")
    ev = C.load(run / "internal_dev_evaluation_audit.json")
    expected_nodes = {r["node_id"] for r in split["records"] if r["internal_split"] == "internal_dev"}
    targets = dict(Counter(c for r in split["records"] if r["internal_split"] == "internal_dev" for c in r["class_instances"]))
    C.require(ev["status"] == "COMPLETE" and ev["rounds_started"] == ev["rounds_completed"] == 1
              and ev["internal_dev_forwards"] == 162 and ev["state_at_first_dev_content_access"] == "ALL_SIX_FROZEN"
              and ev["checkpoint_binding_sha256"] == C.sha(run / "detector_checkpoint_binding.json"), "REVIEW_EVAL_BARRIER_FAILED")
    metrics = {arm: C.load(run / (arm + "_metrics.json")) for arm in C.ARMS}
    for arm, row in metrics.items():
        C.require(ev["arms"][arm]["image_forwards"] == 27 and len(ev["arms"][arm]["node_ids"]) == 27
                  and set(ev["arms"][arm]["node_ids"]) == expected_nodes
                  and row["checkpoint_sha256"] == checkpoints["arms"][arm]["sha256"]
                  and row["target_counts"] == targets, "REVIEW_EVAL_POPULATION_OR_CHECKPOINT_CHANGED")
    result = independent_results(metrics)
    state = C.load(run / "stage21b_status.json")
    native, headroom, classes, variance = F.calculate(metrics, cfg)
    for key, value in native.items():
        C.require(state[key] == value, "REVIEW_SERVER_RESULT_CHANGED: " + key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            C.require(math.isclose(result[key], value, abs_tol=1e-11, rel_tol=0), "REVIEW_INDEPENDENT_CALCULATION_FAILED")
        else:
            C.require(result[key] == value, "REVIEW_INDEPENDENT_CALCULATION_FAILED")
    for name, doc in (("stage21b_headroom", headroom), ("stage21b_classwise", classes), ("stage21b_variance", variance)):
        C.require(C.load(run / (name + ".json")) == doc, "REVIEW_DERIVED_RESULT_CHANGED")
    updates = actual_pair["successful_updates"]
    C.require(state["full_experiment_complete"] and state["phase"] == "COMPLETE" and state["detector_training_count"] == 6
              and state["successful_updates"] == updates and state["optimizer_step_count"] == sum(updates.values())
              and state["internal_dev_eval_rounds"] == 1 and state["internal_dev_forward"] == 162
              and state["generation_count"] == state["official_final_eval_access_count"] == state["BootstrapGuard"] == state["DeepPCB"] == 0
              and state["pipeline_independent_holdout_claim"] is False
              and state["confirmatory_execution_authorized"] is False and state["Stage21C_auto_start"] is False,
              "REVIEW_STATUS_COUNTS_INVALID")
    return {"review_status": "LOCAL_DOWNLOADED_STAGE21B_A1_REVIEW_PASS", "run": run.name,
            "server_result_manifest_sha256": C.sha(run / "stage21b_result_manifest.json"),
            "metadata_files_verified": len(ledger["sha256"]), "checkpoint_bytes_verified": 6,
            "draw_records_verified": 6 * 25500, "augmentation_pairing": "PASS",
            "actual_optimizer_groups_and_lr_pairing": "PASS", "source_isolation": "PASS",
            "original_audit_and_split_unchanged": True, "internal_dev_unique_images": 27,
            "successful_updates": updates, "successful_updates_equal": len(set(updates.values())) == 1,
            "scheduled_attempts_each_arm": 450, "actual_draws_each_arm": 25500,
            "independently_recomputed": result, "official_final_eval_access_in_local_review": 0,
            "internal_dev_image_or_label_content_read_in_local_review": 0, "optimizer_steps_in_local_review": 0,
            "forward_count_in_local_review": 0, "frozen_server_outputs_modified": False,
            "dev_population": state["dev_population"], "generator_dev_annotation_inputs": 33,
            "pipeline_independent_holdout_claim": False,
            "guard_evidence_scope": "pinned execution code + downloaded counters; not an OS-level access trace",
            "next_experiment_automatically_started": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    C.require(not a.output.exists() and not a.output.resolve().is_relative_to(a.run.resolve()), "REVIEW_USE_NEW_SEPARATE_OUTPUT")
    result = review(a.run)
    a.output.mkdir(parents=True)
    C.save(a.output / "local_review.json", result)
    s = result["independently_recomputed"]
    text = f"""# Stage21B A1 downloaded-result review

Review: `{result['review_status']}`. Experiment: `{s['status']}`.

- Original entry audit and split bytes unchanged; only the four authorized conditioning-group exclusions, Flash32 / Black29.
- All 53 server metadata files, six downloaded last.pt files and 153000 draw-log records verified. No checkpoint loading/forward and no dev/final-eval image or label content opened by this review.
- Actual augmentation pairing, frozen initialization, optimizer groups and all 450-attempt learning-rate sequences match.
- Each arm: 150 epochs, 25500 draws, 450 scheduled attempts; actual successful updates 442–443, disclosed, not strictly equal and not compensated.
- Curriculum-minus-Uniform: A {s['delta_A_pp']:+.4f}, B {s['delta_B_pp']:+.4f}, C {s['delta_C_pp']:+.4f} pp; mean {s['mean_delta_pp']:+.4f} pp; wins {s['strict_bank_wins']}/3.
- Flash mean {s['Flash_mean_delta_pp']:+.4f} pp; Black mean {s['Black_mean_delta_pp']:+.4f} pp.
- Across-bank sample SD (ddof=1): Uniform {s['uniform_bank_std_pp']:.4f}, Curriculum {s['curriculum_bank_std_pp']:.4f} pp.
- Gate A={s['Gate_A']}, B={s['Gate_B']}, C={s['Gate_C']}, D={s['Gate_D']}. Lower spread does not offset lower utility or Black-class degradation.

This is **DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR**, not an independent pipeline holdout: the frozen generator historically used 33 dev-group annotation inputs (Black23 / Flash10). The result rejects this one fixed late schedule under this protocol; it does not reject every possible curriculum or establish an official V2 final-eval outcome. File-access claims rely on pinned code and reported counters, not an OS sandbox.

No bound server file was rewritten. No generation, training, evaluation rerun, new selector, weighting, BootstrapGuard, TDCRG, DeepPCB or next stage was started. Stop after GitHub archival.
"""
    (a.output / "LOCAL_REVIEW.md").write_text(text, encoding="utf-8", newline="\n")
    C.save(a.output / "review_artifact_manifest.json", {"sha256": {n: C.sha(a.output / n)
           for n in ("local_review.json", "LOCAL_REVIEW.md")}, "review_script_sha256": C.sha(__file__)})
    print(C.canonical({"review_status": result["review_status"], "status": s["status"], "mean_delta_pp": s["mean_delta_pp"]}))


if __name__ == "__main__":
    main()
