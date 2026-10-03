"""Apply the frozen Stage15B dose/composition gates after all three M50 runs."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from plastic_bomo_stage15b_common import ARMS, CLASSES, SEEDS, load, paired_summary, save, verify_preparation

OUTCOMES = ("map50_95", "map50", "recall", "flash_ap50_95", "black_ap50_95")

def calculate_effects(rows):
    by = {(int(x["seed"]), x["arm"]): x for x in rows}
    if len(rows) != len(by) or set(by) != {(s,a) for s in SEEDS for a in ARMS}:
        raise RuntimeError("STAGE15B_REQUIRES_15_UNIQUE_ARM_SEED_METRICS")
    def value(row, outcome):
        v = row[outcome] if outcome in ("map50_95", "map50", "recall") else row["per_class"][outcome.split("_")[0]]["ap50_95"]
        if not math.isfinite(v) or not 0 <= v <= 1: raise RuntimeError("STAGE15B_NONFINITE_OR_INVALID_METRIC")
        return v
    per = []
    for seed in SEEDS:
        outcomes = {}
        for outcome in OUTCOMES:
            m = {a: value(by[seed,a], outcome) for a in ARMS}; single = (m["M10"]+m["M01"])/2
            outcomes[outcome] = {"arm_values": m, "M40_single": single, "D50_00": m["M50"]-m["M00"],
                "D_balance": m["M50"]-single, "D80_50": m["M11"]-m["M50"],
                "D50_10": m["M50"]-m["M10"], "D50_01": m["M50"]-m["M01"]}
        per.append({"seed": seed, "outcomes": outcomes})
    summary = {outcome: {effect: paired_summary([x["outcomes"][outcome][effect] for x in per])
        for effect in ("D50_00", "D_balance", "D80_50", "D50_10", "D50_01")} for outcome in OUTCOMES}
    return by, per, summary

def decide(summary, by, cfg):
    primary = summary["map50_95"]; thresholds = cfg["gates"]
    gates = {"A_balanced_gain_ge_0_005": primary["D50_00"]["mean"] >= thresholds["balanced_gain_min"],
        "B_balanced_wins_ge_2": primary["D50_00"]["wins"] >= thresholds["balanced_gain_wins_min"],
        "C_same_dose_mean_ge_minus_0_005": primary["D_balance"]["mean"] >= thresholds["same_dose_min"],
        "D_excess_mean_le_minus_0_005": primary["D80_50"]["mean"] <= thresholds["excess_penalty_max"],
        "E_excess_losses_ge_2": primary["D80_50"]["losses"] >= thresholds["excess_penalty_losses_min"]}
    balanced = gates["A_balanced_gain_ge_0_005"] and gates["B_balanced_wins_ge_2"]
    singles = {arm: paired_summary([by[s,arm]["map50_95"]-by[s,"M00"]["map50_95"] for s in SEEDS]) for arm in ("M10", "M01")}
    positive_singles = all(x["mean"] > 0 and x["wins"] >= 2 for x in singles.values())
    if all(gates.values()): status = "MORPHOLOGY_DOSE_EFFECT_SUPPORTED"
    elif not balanced and positive_singles: status = "CLASS_COMPOSITION_INTERACTION_SUPPORTED"
    elif balanced: status = "MORPHOLOGY_SIGNAL_PRESENT_DOSE_HYPOTHESIS_NOT_CONFIRMED"
    else: status = "DOSE_VS_CLASS_INTERACTION_UNRESOLVED"
    improvements = sum(summary[c+"_ap50_95"]["D50_00"]["mean"] > 0 for c in CLASSES)
    state = {"status": status, "gates": gates, "single_class_reference_effects": singles,
        "CLASS_INTERACTION_AS_PRIMARY_EXPLANATION": False if status == "MORPHOLOGY_DOSE_EFFECT_SUPPORTED" else True if status == "CLASS_COMPOSITION_INTERACTION_SUPPORTED" else None,
        "STAGE16_MORPHOLOGY_DISTRIBUTION_CALIBRATED_GENERATOR": status == "MORPHOLOGY_DOSE_EFFECT_SUPPORTED",
        "STAGE16_TASK_AWARE_GENERATION": status == "CLASS_COMPOSITION_INTERACTION_SUPPORTED",
        "STAGE16_AUTO_START": False, "classwise_status": "RESIDUAL_CLASS_DEPENDENCE_PRESENT" if improvements == 1 else "BOTH_CLASSES_IMPROVED" if improvements == 2 else "NO_POSITIVE_CLASS_MEAN",
        "new_detector_runs": 3, "existing_detector_runs_retrained": 0,
        "new_generation": 0, "generator_training": 0, "generator_modification": 0,
        "deep_pcb_training": 0, "deep_pcb_generation": 0, "bootstrap_guard_modification": 0}
    return state

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); root = a.output; cfg = load(a.protocol); verify_preparation(root, cfg)
    features = load(root/"feature_diversity_audit.json"); budget = load(root/"training_budget_equivalence.json")
    if features["status"] != "COMPLETE_BEFORE_NEW_DETECTOR_TRAINING" or budget.get("runtime_audit_pending", True):
        raise RuntimeError("STAGE15B_AUDITS_INCOMPLETE")
    rows = load(root/"all_five_arms_metrics.json")["records"]; by, per, summary = calculate_effects(rows)
    state = decide(summary, by, cfg)
    save(root/"dose_effects_by_seed.json", {"primary_metric": "map50_95", "effects": per})
    save(root/"dose_effects_summary.json", {"std_ddof": 1, "outcomes": summary})
    save(root/"classwise_dose_effects.json", {"status": state["classwise_status"], "flash": summary["flash_ap50_95"], "black": summary["black_ap50_95"],
        "all_arm_AP50_95": {c: {arm: paired_summary([by[s,arm]["per_class"][c]["ap50_95"] for s in SEEDS]) for arm in ARMS} for c in CLASSES}})
    overlap = load(root/"membership_overlap_audit.json")
    state["nominal_vs_actual_dose_limitation"] = overlap["interpretation"]
    state["concentration_diagnostics_define_status"] = False
    save(root/"stage15b_status.json", state)
    report = ["# Plastic_Bomo Stage15B — Morphology Dose vs Class Interaction", "", f"Status: `{state['status']}`.", "",
        "BalancedMorph50 is a mechanistic probe built from frozen HighMorph/Random lists. It is not a proposed final selector. Only three M50 detectors were trained; all twelve Stage15A metrics were reused. No Stage16 process was started.", "",
        "## Paired downstream effects", "", "Primary outcome: mAP50-95. Values are percentage-point deltas, mean ± sample std over seeds 42, 3407, 2026.", "",
        "| Effect | seed42 | seed3407 | seed2026 | Mean ± std | Positive / negative |", "|---|---:|---:|---:|---:|---:|"]
    for key in ("D50_00", "D_balance", "D80_50"):
        z = summary["map50_95"][key]
        report.append(f"| {key} | "+" | ".join(f"{v*100:+.3f}" for v in z["values"])+f" | {z['mean_pp']:+.3f} ± {z['std_pp']:.3f} | {z['wins']}/3 / {z['losses']}/3 |")
    report += ["", "D50_00=M50−M00; D_balance=M50−(M10+M01)/2; D80_50=M11−M50.", "", "## Registered gates", ""]
    report += [f"- {k}: {'PASS' if v else 'FAIL'}." for k,v in state["gates"].items()]
    report += ["", "## Class-specific and secondary outcomes", "", "| Outcome | M50−M00 mean ± std (pp) | Wins | M11−M50 mean ± std (pp) | Negative seeds |", "|---|---:|---:|---:|---:|"]
    for outcome in ("map50", "recall", "flash_ap50_95", "black_ap50_95"):
        first, excess = summary[outcome]["D50_00"], summary[outcome]["D80_50"]
        report.append(f"| {outcome} | {first['mean_pp']:+.3f} ± {first['std_pp']:.3f} | {first['wins']}/3 | {excess['mean_pp']:+.3f} ± {excess['std_pp']:.3f} | {excess['losses']}/3 |")
    report += ["", f"Class-specific assessment: `{state['classwise_status']}`.", "", "| Arm | Flash AP50-95 by seed (42, 3407, 2026), % | Black AP50-95 by seed, % |", "|---|---|---|"]
    for arm in ARMS:
        report.append(f"| {arm} | "+" | ".join(", ".join(f"{by[s,arm]['per_class'][c]['ap50_95']*100:.3f}" for s in SEEDS) for c in CLASSES)+" |")
    report += ["", "## Training-free concentration diagnostics", "", "Features use the existing DWBG real-only OOF teacher checkpoints and P3/P4/P5 detection-head-input ROIs. Distances and effective rank are computed within each fold and then averaged, never between incompatible teacher spaces. Historical real feature banks contain teacher-training records; they are not mislabeled as holdout features.", "", "| Group | Diagnostic | M11 | Mean M10/M01/M50 | M11 minus comparison |", "|---|---|---:|---:|---:|"]
    for group, metrics in features["M11_concentration_diagnostics"].items():
        for key, v in metrics.items(): report.append(f"| {group} | {key} | {v['M11']:.6f} | {v['mean_M10_M01_M50']:.6f} | {v['M11_minus_comparators']:+.6f} |")
    morphology = load(root/"morphology_distribution_audit.json")["arms"]
    report += ["", "| Group | Morphology std M11 | Mean std M10/M01/M50 |", "|---|---:|---:|"]
    for c in ("overall", *CLASSES):
        mean_std = sum(morphology[arm][c]["std"] for arm in ("M10", "M01", "M50"))/3
        report.append(f"| {c} | {morphology['M11'][c]['std']:.6f} | {mean_std:.6f} |")
    report += ["", "Concentration metrics are separate descriptive outcomes and do not determine the registered downstream gates. No weighted quality score or feedback into selection was used.", "", "## Dose membership and evidence limits", "", "| Arm | Nominal HighMorph role count | Actual membership in frozen HighMorph list |", "|---|---:|---:|"]
    for arm in ARMS:
        z = overlap["actual_vs_nominal_high_dose"][arm]; report.append(f"| {arm} | {z['nominal_high_role_count']} | {z['actual_frozen_high_membership_count']} |")
    report += ["", overlap["interpretation"], "", "New M50 runs export raw args.yaml, 150-epoch CSVs, primary image draws/batches and optimizer attempts/successes. Historical arms retain only their original scheduled-budget evidence; exact historical optimizer-step and pretrained-byte equality are not asserted. Mosaic source-image usage is not equated with primary loader draws.", "", "This single probe uses existing sample sets and three detector seeds. Even if the dose gates pass, compression is a mechanism hypothesis to assess against the separate diagnostics, not a proven consequence of morphology similarity alone. No additional dose sweep, generation, DeepPCB experiment or BootstrapGuard modification is authorized by this finalizer.", ""]
    (root/"STAGE15B_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(state["status"])

if __name__ == "__main__": main()
