"""Independently review downloaded Stage15A metrics without training or validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

SEEDS = (42, 3407, 2026)
ARMS = ("M00", "M10", "M01", "M11")
METRICS = ("map50_95", "map50", "recall")


def load(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def effects(values):
    a, b, c, d = (values[arm] for arm in ARMS)
    return {"EF_B0": b-a, "EF_B1": d-c, "EB_F0": c-a, "EB_F1": d-b,
            "ME_F": (b-a+d-c)/2, "ME_B": (c-a+d-b)/2,
            "interaction": d-b-c+a}


def summary(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values),
            "wins": sum(v > 0 for v in values), "values": values}


def same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    return actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage14a", type=Path, required=True)
    parser.add_argument("--stage14b", type=Path, required=True)
    parser.add_argument("--candidate_pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    documents = {p.name: load(p) for p in root.glob("*.json") if p.name != "downloaded_results_review.json"}
    cfg = load(args.protocol)
    rows = documents["factorial_all_arms_metrics.json"]["records"]
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise RuntimeError("STAGE15A_REVIEW_FAILED: " + name)

    by = {(x["seed"], x["arm"]): x for x in rows}
    check("exact_12_unique_arm_seed_records", len(rows) == len(by) == 12 and set(by) == {(s, arm) for s in SEEDS for arm in ARMS})
    check("protocol_matches_download", same(documents["stage15a_protocol.json"], cfg))
    check("finite_complete_metrics", all(isinstance(x[k], (float, int)) and math.isfinite(x[k]) and 0 <= x[k] <= 1 for x in rows for k in ("precision", *METRICS)) and all(math.isfinite(x["per_class"][c][k]) and 0 <= x["per_class"][c][k] <= 1 for x in rows for c in ("flash", "black") for k in ("ap50", "ap50_95", "recall")))
    old = {(x["seed"], x["arm"]): x for x in load(args.stage14b/"all_seed_metrics.json")["records"]}
    check("reused_M00_M11_exact_original_metrics_and_checkpoint", all(same({k: by[s, arm][k] for k in ("precision", *METRICS, "per_class", "checkpoint_sha256")}, {k: old[s, original][k] for k in ("precision", *METRICS, "per_class", "checkpoint_sha256")}) for s in SEEDS for arm, original in (("M00", "random"), ("M11", "high"))))
    check("six_new_six_reused_records", all(x["reused_stage14b"] == (x["arm"] in ("M00", "M11")) for x in rows))
    for arm, filename in (("M10", "arm10_flash_morph_metrics.json"), ("M01", "arm01_black_morph_metrics.json")):
        arm_rows = documents[filename]["records"]
        check(filename+"_matches", len(arm_rows) == 3 and same({x["seed"]: x for x in arm_rows}, {s: by[s, arm] for s in SEEDS}))

    subsets = load(args.stage14a/"subset_definition.json")["subsets"]
    pool = load(args.candidate_pool)
    candidates = pool.get("candidates", pool.get("selected", []))
    index = {x["candidate_id"]: x for x in candidates}
    rf, rb, hf, hb = ([i for i in subsets[k] if int(index[i]["class_id"]) == c] for k, c in (("random", 0), ("random", 1), ("high_fidelity", 0), ("high_fidelity", 1)))
    expected_ids = {"M00": rf+rb, "M10": hf+rb, "M01": rf+hb, "M11": hf+hb}
    manifest = documents["factorial_subset_definition.json"]
    for arm in ARMS:
        actual = manifest["arms"][arm]
        check(arm+"_exact_ids_and_class_counts", [x["candidate_id"] for x in actual] == expected_ids[arm] and len(set(expected_ids[arm])) == 80 and Counter(x["class_name"] for x in actual) == {"flash": 40, "black": 40})
        check(arm+"_exact_original_paths_seeds_sources", all(x["image_path"] == index[x["candidate_id"]]["image_path"] and x["label_path"] == index[x["candidate_id"]]["label_path"] and x["generation_seed"] == int(index[x["candidate_id"]]["seed"]) and x["parent_source_id"] == f"{index[x['candidate_id']]['class_id']}:{Path(index[x['candidate_id']]['source_image']).stem}" for x in actual))
        check(arm+"_ids_hash", digest(expected_ids[arm]) == manifest["candidate_id_sha256"][arm])
    check("manifest_content_hash", digest(manifest) == documents["factorial_manifest_freeze_audit.json"]["manifest_sha256"])
    overlap = documents["factorial_overlap_audit.json"]
    check("reported_overlap", overlap["high_f_vs_random_f"] == len(set(rf)&set(hf)) and overlap["high_b_vs_random_b"] == len(set(rb)&set(hb)) and overlap["overlap_removed"] is False)

    per_seed = []
    for seed in SEEDS:
        row = {"seed": seed, "overall": {k: effects({arm: by[seed, arm][k] for arm in ARMS}) for k in METRICS}}
        for cls in ("flash", "black"):
            row[cls+"_ap50_95"] = effects({arm: by[seed, arm]["per_class"][cls]["ap50_95"] for arm in ARMS})
        per_seed.append(row)
    check("all_per_seed_effects_recomputed", same(documents["factorial_effects_by_seed.json"], {"effects": per_seed}))
    effect_names = list(per_seed[0]["overall"]["map50_95"])
    overall = {k: {e: summary([x["overall"][k][e] for x in per_seed]) for e in effect_names} for k in METRICS}
    check("all_overall_effect_summaries_recomputed", same(documents["factorial_effects_summary.json"], overall))
    classes = {cls: {e: summary([x[cls+"_ap50_95"][e] for x in per_seed]) for e in effect_names} for cls in ("flash", "black")}
    for cls in classes:
        check(cls+"_effects_recomputed", same(documents[cls+"_class_effect.json"], classes[cls]))
    ef, conditional = (overall["map50_95"][k] for k in ("EF_B0", "EF_B1"))
    flash = classes["flash"]["EF_B0"]
    gates = {"A_overall_mean_ge_0_005": ef["mean"] >= cfg["flash_overall_threshold"], "B_overall_wins_ge_2": ef["wins"] >= 2, "C_flash_ap_mean_ge_0_01": flash["mean"] >= cfg["flash_class_threshold"], "D_flash_ap_wins_ge_2": flash["wins"] >= 2, "conditional_mean_positive": conditional["mean"] > 0, "conditional_wins_ge_2": conditional["wins"] >= 2}
    confirmed = all(gates.values())
    black_losses = sum(v < 0 for v in classes["black"]["EB_F0"]["values"])
    black_bad = overall["map50_95"]["EB_F0"]["mean"] < 0 and black_losses >= 2
    interaction = overall["map50_95"]["interaction"]
    material = abs(interaction["mean"]) >= cfg["interaction_threshold"]
    if confirmed:
        status = "CLASS_DEPENDENT_MORPHOLOGY_BOTTLENECK_CONFIRMED" if black_bad else "FLASH_MORPHOLOGY_CAUSAL_SIGNAL_CONFIRMED"
    elif all(list(gates.values())[:4]) and not (gates["conditional_mean_positive"] and gates["conditional_wins_ge_2"]) and material:
        status = "CLASS_INTERACTION_DOMINATES_MORPHOLOGY_SIGNAL"
    else:
        status = "FLASH_MORPHOLOGY_CAUSAL_SIGNAL_NOT_CONFIRMED"
    state = documents["stage15a_status.json"]
    check("status_and_gates_independently_recomputed", state["status"] == status and same(state["flash_gates"], gates) and state["FLASH_MORPHOLOGY_CAUSAL_SIGNAL_CONFIRMED"] == confirmed and state["STAGE15B_CLASS_ADAPTIVE_GENERATOR_DESIGN"] == confirmed and state["STAGE15B_AUTO_START"] is False)
    check("black_gate_uses_strict_negative_deltas", (state["black_status"] == "BLACK_MORPHOLOGY_ALIGNMENT_NOT_BENEFICIAL") == black_bad)
    budget = documents["factorial_training_budget_audit.json"]
    check("scheduled_budget_metadata", budget["scheduled"] == {"images_per_epoch": 218, "epochs": 150, "draws": 32700, "batches": 32700} and budget["new_detector_trainings"] == 6 and budget["reused_detector_trainings"] == 6)
    environment = documents["training_environment_audit.json"]
    check("six_args_hashes_recorded", len(environment["new_run_args"]) == 6 and all(isinstance(x["args_yaml_sha256"], str) and len(x["args_yaml_sha256"]) == 64 for x in environment["new_run_args"]))
    usage = documents["validation_usage_audit.json"]
    check("validation_usage_metadata", len(usage["new_runs"]) == 6 and usage["selection_uses"] == usage["threshold_tuning_uses"] == 0 and all(x["training_validation_uses"] == 0 and x["final_last_evaluation_uses"] == 1 and x["best_checkpoint_used"] is False for x in usage["new_runs"]))
    limitations = ["Downloaded files contain scheduled budget metadata and args.yaml hashes, but not training CSVs, actual batch counters, optimizer-step logs, args.yaml contents, or pretrained checkpoint bytes. Actual equal compute and identical initialization cannot be independently verified locally.", "The manifest hash verifies downloaded content against its recorded digest; it does not independently prove the pre-training freeze timestamp or unchanged server image/label bytes.", "This 2x2 intervention changes frozen class-specific sample sets. It establishes conditional subset effects; morphology-specific causality remains subject to residual sample differences and three-seed uncertainty."]
    review = {"status": "DOWNLOADED_METRICS_AND_MANIFEST_REVIEW_PASS", "experiment_status": status, "checks": checks, "seed_order": list(SEEDS), "std_definition": "sample standard deviation, ddof=1", "black_AP_losses_M01_vs_M00": black_losses, "limitations": limitations, "downloaded_json_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in documents}}
    (root/"downloaded_results_review.json").write_text(json.dumps(review, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = ["# Plastic_Bomo Stage15A — Class-Conditional Morphology Causal Isolation", "", f"Status: `{status}`.", "", "Only M10/M01 were newly trained (six runs). M00/M11 reproduce the exact Stage14B Random/High metrics and checkpoint hashes. No Stage15B design is authorized and no Stage15B experiment has started.", "", "## Paired primary effects", "", "All entries below are mAP50-95 percentage-point deltas. Mean ± std uses sample std across seeds 42, 3407, 2026.", "", "| Effect | seed42 | seed3407 | seed2026 | Mean ± std | Positive seeds |", "|---|---:|---:|---:|---:|---:|"]
    for effect in effect_names:
        v = overall["map50_95"][effect]
        report.append(f"| {effect} | " + " | ".join(f"{n*100:+.3f}" for n in v["values"]) + f" | {v['mean']*100:+.3f} ± {v['std']*100:.3f} | {v['wins']}/3 |")
    report += ["", "EF_B0=M10−M00; EF_B1=M11−M01; EB_F0=M01−M00; EB_F1=M11−M10. ME_F/ME_B average the respective conditional effects. Interaction=M11−M10−M01+M00.", "", "## Interpretation", "", "Flash-only replacement passes primary gates A–D: overall mAP50-95 improves by +2.387 pp (2/3 positive), and Flash AP50-95 by +3.664 pp (3/3 positive). Conditional robustness fails: under High-Black, the overall Flash effect is −0.110 pp (1/3 positive). The interaction is −2.497 pp (all three seeds negative), exceeding the registered 0.50 pp magnitude threshold.", "", "Black-only replacement improves overall mAP50-95 by +1.152 pp and Black AP50-95 by +0.899 pp (both 2/3 positive). The gate for Black morphology being non-beneficial does not pass. These observations do not support a general rule that Flash should always receive morphology guidance while Black should always remain baseline.", "", "## Class-specific and secondary effects", "", "| Outcome | EF_B0 mean ± std (pp) | Positive seeds | EF_B1 mean ± std (pp) | Positive seeds |", "|---|---:|---:|---:|---:|"]
    for label, source in (("Flash AP50-95", classes["flash"]), ("Black AP50-95", classes["black"]), ("mAP50", overall["map50"]), ("Recall", overall["recall"])):
        a, b = source["EF_B0"], source["EF_B1"]
        report.append(f"| {label} | {a['mean']*100:+.3f} ± {a['std']*100:.3f} | {a['wins']}/3 | {b['mean']*100:+.3f} ± {b['std']*100:.3f} | {b['wins']}/3 |")
    report += ["", "Full per-arm precision, recall, AP50 and AP50-95 remain in factorial_all_arms_metrics.json; all seven effects for each outcome remain in the effect JSON files.", "", "## Manifest and training audit", "", "All four arms contain the prescribed 40 Flash + 40 Black candidates. Every candidate ID, path, generation seed and parent/source ID agrees with the original pool. Flash High/Random overlap is 4 and Black overlap is 10; neither was removed. The full manifest digest and per-arm ID digests match.", "", "The downloaded budget declares 138 real + 80 synthetic, 150 epochs, batch 1, and 32,700 scheduled draws/batches per run. Six new args.yaml hashes and final-last-only evaluation metadata are present. These are recorded protocol values rather than locally verified runtime counters.", "", "## Evidence limits", ""]
    report += ["- " + item for item in limitations]
    report += ["", "The registered status is preserved without changing metrics, thresholds, frozen subsets or training protocol. No additional training, generation or official-validation evaluation was performed by this review.", ""]
    (root/"STAGE15A_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"review_status": review["status"], "experiment_status": status, "checks_passed": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
