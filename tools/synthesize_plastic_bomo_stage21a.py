"""Stage21A: stdlib-only evidence synthesis; no model/data runtime imports.

Reads only metadata blobs at the pinned formal Git revision. --write creates a
new, fixed Stage21A result directory; --check never writes. Deleted historical
worktree files are neither restored nor used as authoritative evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

REVISION = "26d19a7906b4339c059d9897d9a29c4cbbee2762"
RESULT = "results_for_gpt/plastic_bomo_stage21a_route_decision"
PREFIX = "results_for_gpt/"
ROOTS = {
    "14a": PREFIX + "plastic_bomo_stage14a_generator_bottleneck/",
    "14b": PREFIX + "plastic_bomo_stage14b_fidelity_confirmation/",
    "15a": PREFIX + "plastic_bomo_stage15a_class_conditional_morphology/",
    "15b": PREFIX + "plastic_bomo_stage15b_morphology_dose/",
    "16r": PREFIX + "plastic_bomo_stage16a_r0_traceability_preflight/",
    "17a": PREFIX + "plastic_bomo_stage17a_vanilla_utility_stability/detector_repair_20261007_080108/results/",
    "18a": PREFIX + "plastic_bomo_stage18a_generator_base_viability_audit/server_inventory_20261008_042413/",
    "20a": PREFIX + "plastic_bomo_stage20a_v2_protocol/",
    "p0": PREFIX + "plastic_bomo_stage20b_p0_preflight/",
    "p0s": PREFIX + "plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115/",
    "r0": PREFIX + "plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254/",
    "20b": PREFIX + "plastic_bomo_stage20b_formal/manual_C_20261010_032543/",
    "bg": PREFIX + "bootstrap_guard_downstream/",
}
ZERO_ACTIVITY = {
    "training": 0, "generator_training_count": 0, "generation": 0,
    "synthetic_generation_count": 0, "sampling_count": 0,
    "detector_training_count": 0, "optimizer_step_count": 0,
    "official_final_eval_forward": 0, "final_eval_image_or_label_content_read": 0,
    "BootstrapGuard": 0, "DeepPCB": 0, "TDCRG_implementation": 0,
    "new_module_implementation": 0,
}
REQUIRED_STAGES = (
    "Stage14A", "Stage14B", "Stage15A", "Stage15B", "Stage16A",
    "Stage16A-R0", "Stage17A", "Stage18A", "Stage20A", "Stage20B-P0",
    "Stage20B-R0", "Stage20B",
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


class Evidence:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.blobs = {}
        self.checks = []
        self.text_decoding = {}

    def read(self, path):
        if path not in self.blobs:
            if not path.startswith((PREFIX, "configs/")) or ".." in Path(path).parts:
                raise ValueError("metadata path outside evidence scope")
            self.blobs[path] = subprocess.check_output(
                ["git", "-C", str(self.repo), "show", REVISION + ":" + path],
                stderr=subprocess.PIPE,
            )
        blob = self.blobs[path]
        if path.endswith(".json"):
            return json.loads(blob.decode("utf-8-sig"))
        try:
            result = blob.decode("utf-8-sig")
            self.text_decoding[path] = "UTF8_STRICT"
        except UnicodeDecodeError:
            # Some legacy reports contain damaged non-UTF8 punctuation. Keep
            # the original byte digest; numbers/statuses come from strict JSON.
            result = blob.decode("utf-8-sig", errors="replace")
            self.text_decoding[path] = "UTF8_REPLACEMENT_FOR_LEGACY_REPORT_ONLY; numeric evidence uses strict JSON"
        return result

    def get(self, root, name):
        return self.read(ROOTS[root] + name)

    def require(self, name, condition):
        self.checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise ValueError("STAGE21A_EVIDENCE_CHECK_FAILED: " + name)

    def close(self, name, actual, expected):
        self.require(name, math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10))

    def sources(self):
        return [{"path": path, "revision": REVISION,
                 "sha256": hashlib.sha256(blob).hexdigest(),
                 "read_method": "git show pinned_revision:path; metadata only",
                 "text_decoding": self.text_decoding.get(path, "UTF8_STRICT_JSON"),
                 "url": "https://github.com/chenbingjie594-lab/Self-main/blob/" + REVISION + "/" + path}
                for path, blob in sorted(self.blobs.items())]


def entry(stage, question, protocol, result, supported, unsupported, confirmatory,
          validation, selector, sources, limits=()):
    return {
        "stage": stage, "question": question, "protocol": protocol, "result": result,
        "what_is_supported": supported, "what_is_not_supported": unsupported,
        "whether_confirmatory": confirmatory,
        "confirmatory_definition": "whether a prespecified confirmation/calibration was attempted, NOT whether it passed",
        "whether_validation_used": validation[0], "validation_use_scope": validation[1],
        "whether_selector_involved": selector[0], "selector_scope": selector[1],
        "sources": sources, "limits": list(limits),
    }


def build(repo):
    e = Evidence(repo)
    # Read complete formal reports, not just status names. They contain caveats
    # on provenance, legacy pool filtering, reused arms and successful updates.
    for k, name in {
        "14a": "STAGE14A_REPORT.md", "14b": "STAGE14B_REPORT.md",
        "15a": "STAGE15A_REPORT.md", "15b": "STAGE15B_REPORT.md",
        "16r": "SERVER_STOP_REVIEW.md", "17a": "STAGE17A_REPORT.md",
        "18a": "STAGE18A_REPORT.md", "20a": "STAGE20A_REPORT.md",
        "p0s": "STAGE20B_P0_REPORT.md", "r0": "STAGE20B_R0_REPORT.md",
        "20b": "STAGE20B_REPORT.md", "bg": "BOOTSTRAP_GUARD_DOWNSTREAM_REPORT.md",
    }.items():
        e.get(k, name)
    s14a = e.get("14a", "stage14a_status.json")
    s14b = e.get("14b", "stage14b_status.json")
    fidelity = e.get("14b", "completed_fidelity_audit.json")
    replication = e.get("14b", "three_seed_summary.json")
    s15a = e.get("15a", "stage15a_status.json")
    effects = e.get("15a", "factorial_effects_summary.json")["map50_95"]
    s15b = e.get("15b", "stage15b_status.json")
    dose = e.get("15b", "dose_effects_summary.json")
    s16r = e.get("16r", "stage16a_r0_status.json")
    p16r = e.get("16r", "stage16a_r0_protocol.json")
    s17a = e.get("17a", "stage17a_status.json")
    s18a = e.get("18a", "stage18a_status.json")
    alternatives = e.get("18a", "stage18b_candidate_selection.json")
    s20a = e.get("20a", "stage20a_status.json")
    sp0 = e.get("p0s", "stage20b_p0_status.json")
    sr0 = e.get("r0", "stage20b_r0_status.json")
    for root, filename in [
        ("14b", "stage14b_protocol.json"), ("14b", "validation_usage_audit.json"),
        ("15a", "stage15a_protocol.json"), ("15a", "validation_usage_audit.json"),
        ("15b", "stage15b_protocol.json"), ("15b", "validation_usage_audit.json"),
        ("20a", "future_baseline_gate.json"), ("20a", "v2_generator_baseline_selection.json"),
    ]:
        e.get(root, filename)
    e.read("configs/plastic_bomo_stage14a.json")
    pointer = e.read(PREFIX + "plastic_bomo_stage20b_formal/authoritative_formal_run.json")
    formal = e.get("20b", "stage20b_status.json")
    headroom = e.get("20b", "stage20b_headroom.json")
    classwise = e.get("20b", "stage20b_classwise.json")
    nuisance = e.get("20b", "stage20b_generation_nuisance.json")
    recovery = e.get("20b", "manual_arm_C_recovery.json")
    compute = e.get("20b", "detector_compute_audit.json")
    gate = e.get("p0", "stage20b_future_baseline_gate.json")
    genprotocol = e.get("p0", "stage20b_generation_protocol.json")
    e.get("p0", "stage20b_detector_protocol.json")
    e.get("20b", "formal_protocol_binding.json")
    e.get("20b", "generator_checkpoint_binding.json")
    e.get("20b", "final_eval_access_audit.json")
    e.get("20b", "formal_result_artifact_manifest.json")
    metrics = {arm: e.get("20b", ("RR" if arm == "RR" else "Raw" + arm) + "_metrics.json")
               for arm in ("RR", "A", "B", "C")}
    banks = {b: e.get("20b", "formal_bank_" + b + "_manifest.json") for b in "ABC"}
    bg = e.get("bg", "paired_deltas.json")
    e.get("bg", "validated_per_seed.json")
    e.get("bg", "dataset_audit.json")

    e.require("authoritative run unchanged", pointer["run"] == "manual_C_20261010_032543")
    e.require("authoritative artifact digest unchanged",
              pointer["artifact_manifest_sha256"] == hashlib.sha256(e.blobs[ROOTS["20b"] + "formal_result_artifact_manifest.json"]).hexdigest())
    e.require("formal stop preserved", formal["status"] == headroom["status"] == pointer["status"] == "GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED")
    e.require("no TDCRG authorization", formal["TDCRG_DEVELOPMENT_AUTHORIZED"] is False)
    delta = {b: 100 * (metrics[b]["map5095"] - metrics["RR"]["map5095"]) for b in "ABC"}
    mean_delta = statistics.mean(delta.values())
    std_delta = statistics.stdev(delta.values())
    bank_range = 100 * (max(metrics[b]["map5095"] for b in "ABC") - min(metrics[b]["map5095"] for b in "ABC"))
    class_delta = {c: {b: 100 * (metrics[b]["per_class"][c]["ap5095"] - metrics["RR"]["per_class"][c]["ap5095"])
                       for b in "ABC"} for c in ("flash", "black")}
    class_mean = {c: statistics.mean(values.values()) for c, values in class_delta.items()}
    for b in "ABC":
        e.close("raw " + b + " delta", delta[b], headroom["delta_" + b + "_pp"])
        e.close("macro AP identity " + b, statistics.mean(class_delta[c][b] for c in class_delta), delta[b])
    for c in class_delta:
        e.close("class mean " + c, class_mean[c], classwise["classes"][c]["mean_delta_pp"])
    for name, value, expected in [
        ("mean", mean_delta, headroom["mean_delta_pp"]), ("sample SD", std_delta, headroom["std_delta_pp"]),
        ("range", bank_range, nuisance["range_pp"]), ("nuisance SD", std_delta, nuisance["sample_std_pp"]),
    ]:
        e.close(name, value, expected)
    wins = sum(value > 0 for value in delta.values())
    gates = {"A": mean_delta >= gate["A_mean_raw_bank_minus_RR_ge_pp"],
             "B": wins >= gate["B_strict_bank_wins_ge"],
             "C": all(v >= gate["C_every_class_mean_delta_ge_pp"] for v in class_mean.values())}
    e.require("three frozen gates all fail", not any(gates.values()))
    for b in "ABC":
        e.require("recorded gate " + b, gates[b] == headroom["Gate_" + b])
        e.require("80 raw samples " + b, banks[b]["count"] == len(banks[b]["records"]) == 80)
        e.require("no quality selection/reroll " + b, banks[b]["quality_selection"] is False and banks[b]["reroll_count"] == 0)
        e.require("raw per-class40/40 " + b,
                  all(sum(row["class"] == c for row in banks[b]["records"]) == 40 for c in ("flash", "black")))
        e.require("every record raw " + b, all(row["formal_synthetic"] is True and row["quality_selection"] is False for row in banks[b]["records"]))
    arows = {row["slot_id"]: row for row in banks["A"]["records"]}
    e.require("unique A slots", len(arows) == 80)
    for b in "BC":
        rows = {row["slot_id"]: row for row in banks[b]["records"]}
        e.require("same unique slots " + b, rows.keys() == arows.keys() and len(rows) == 80)
        for sid, row in rows.items():
            ar = arows[sid]
            e.require("same slot/checkpoint/conditioning " + b + ":" + sid,
                      row["frozen_slot"] == ar["frozen_slot"]
                      and row["non_rng_metadata_sha256"] == ar["non_rng_metadata_sha256"]
                      and row["checkpoint_tree_sha256"] == ar["checkpoint_tree_sha256"])
            e.require("different paired RNG " + b + ":" + sid,
                      row["sample_seed"] != ar["sample_seed"] and row["RNG_before_sha256"] != ar["RNG_before_sha256"])
    arms = compute["arms"]
    for field in ("epochs", "actual_draws", "attempted_optimizer_updates", "successful_optimizer_updates",
                  "AMP_skipped_updates", "order_spec_sha256", "initial_parameter_binding_sha256"):
        e.require("matched primary " + field, len({r[field] for r in arms.values()}) == 1)
    e.require("primary epoch150", all(r["epochs"] == 150 for r in arms.values()))
    e.require("C restart epoch1, not resume", recovery["restart_epoch"] == 1 and recovery["restarted_detector_arms"] == ["C"])
    e.require("source unchanged", recovery["source_run_modified"] is False and recovery["source_run_unchanged_SHA_verified"] is True)
    e.require("kill cause not independently verified", recovery["external_kill_cause_independently_verified"] is False)
    e.require("no alternative generator authorized", alternatives["selected_generator"] is None and not s18a["STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED"])
    e.require("unchanged frozen generation protocol", genprotocol["raw_output_policy"]["quality_selection"] is False
              and genprotocol["raw_output_policy"]["all_successful_outputs_used"] is True)

    legacy_limits = [
        "Legacy subset/pool evidence, not untouched raw generation; Stage18A documents recomposition/filtering and unresolved historical isolation.",
        "Final validation metrics were consulted by subsequent stages; this sequence is exploratory at route level, not independent replication of a preregistered generator mechanism.",
        "Do not pool historical effect sizes with V2 or compare absolute mAP across different splits/protocols.",
    ]
    final_val = (True, "Recorded final-last evaluation, no training-time validation/threshold tuning; earlier pool-construction validation involvement unresolved.")
    subset_selector = (True, "Frozen subsets from a recomposed/filtered DWBG pool; Random within that pool is not raw selector-free generation.")
    no_val = (False, "No evaluation forward or new validation image/label content access in this stage.")
    no_selector = (False, "No quality scoring, Top-K, reroll, manual filtering or BootstrapGuard.")
    timeline = [
        entry("Stage14A", "Do image properties identify a generator bottleneck?",
              "Legacy 138 real + 80 extras, 40/class; YOLO11s seed42, 150 epochs, last.pt screening.",
              {"status": s14a["status"], "high_minus_random_pp": s14a["morphology_matched_subset_signal"]["high_minus_random_pp"],
               "low_minus_random_pp": s14a["morphology_matched_subset_signal"]["low_minus_random_pp"], "fidelity_protocol_complete": False},
              ["Morphology-matched high/low subset contrast exists."],
              ["Broad fidelity bottleneck; crop-KID/nearest LPIPS were unavailable at this stage.", "Raw generator utility."],
              False, final_val, subset_selector, [ROOTS["14a"] + "stage14a_status.json", ROOTS["14a"] + "STAGE14A_REPORT.md"],
              legacy_limits + ["Preserve corrected status, not the superseded FIDELITY_BOTTLENECK_SUPPORTED label.",
                               "Historical report's task-aware-generation recommendation is not present authorization."]),
        entry("Stage14B", "Does the subset signal replicate, and is it broad visual fidelity?",
              "Same frozen High/Random/Low/RR; detector seeds42/3407/2026, 150 epochs, last.pt; KID set-level only.",
              {"status": s14b["status"], "attribution": fidelity["status"],
               "paired_deltas": replication["deltas"], "fidelity_metrics": fidelity["metrics"]},
              ["High-Random +1.043pp, 2/3 wins; a conditional subset signal with class dependence."],
              ["A morphology-specific generator mechanism or broadly improved pixel/perceptual fidelity.", "Raw synthetic headroom."],
              True, final_val, subset_selector, [ROOTS["14b"] + n for n in ("stage14b_status.json", "three_seed_summary.json", "completed_fidelity_audit.json", "STAGE14B_REPORT.md")], legacy_limits),
        entry("Stage15A", "Does class-conditional replacement isolate an additive morphology benefit?",
              "2x2 M00/M10/M01/M11; only M10/M01 newly trained, reuse exact Stage14B M00/M11, three detector seeds.",
              {"status": s15a["status"], "primary_effects_raw": effects, "generator_design_authorized": False},
              ["Conditional sample-set effects and negative class interaction (-2.497pp)."],
              ["Flash-always-guided/Black-always-baseline rule.", "Morphology-specific causality; selected subsets differ in other respects."],
              True, final_val, subset_selector, [ROOTS["15a"] + n for n in ("stage15a_status.json", "factorial_effects_summary.json", "STAGE15A_REPORT.md")],
              legacy_limits + ["Historical scheduled budgets are recorded; runtime optimizer/init equality not independently verified."]),
        entry("Stage15B", "Does a balanced intermediate morphology dose explain utility?",
              "One frozen M50 probe (20High+20Random/class); three new detectors, twelve historical records reused; no dose sweep.",
              {"status": s15b["status"], "gates": s15b["gates"], "dose_effects": dose},
              ["M50-M00 +0.674pp; residual class dependence; descriptive concentration differences."],
              ["Registered dose hypothesis (C/D fail).", "Concentration causes harm or morphology maximization guarantees utility."],
              True, final_val, subset_selector, [ROOTS["15b"] + n for n in ("stage15b_status.json", "dose_effects_summary.json", "STAGE15B_REPORT.md")],
              legacy_limits + ["Nominal/actual High membership differs because High/Random overlap; 544 matched attempts, 532/534/534 successful M50 updates."]),
        entry("Stage16A", "Can original task guidance use an isolated, exact real-parent OOF teacher?",
              "Structural preflight only; exact parent mapping required, no substitute nearest/synthetic-ID teacher.",
              {"status": p16r["original_stage16a_stop"], "evidence_level": "FORMALLY_CARRIED_FORWARD_STOP_RECORD",
               "primary_server_status_blob_available_in_pinned_commit": False},
              ["Original server stop is explicitly carried forward in the frozen Stage16A-R0 protocol."],
              ["Task guidance efficacy or failure.", "Independent reconstruction of the missing original server status JSON."],
              False, no_val, no_selector, [ROOTS["16r"] + "stage16a_r0_protocol.json", ROOTS["16r"] + "SERVER_STOP_REVIEW.md"],
              ["A different LOCAL_PREFLIGHT_SNAPSHOT exists; it must not be relabeled as the original server execution."]),
        entry("Stage16A-R0", "Do annotation provenance and lineage permit task-guidance qualification?",
              "Frozen legacy138 images/168 boxes; >=5 accepted source groups/class required; annotation entry gate first.",
              {"status": s16r["status"], "gates": s16r["gates"],
               "accepted_unique_source_groups_per_class": s16r["accepted_unique_source_groups_per_class"]},
              ["Submitted evidence was insufficient at entry; all later runtime gates NOT_RUN."],
              ["Labels must be incorrect or no manual annotation ever existed.", "Task guidance efficacy; actual task-loss manipulation."],
              False, no_val, no_selector, [ROOTS["16r"] + n for n in ("stage16a_r0_status.json", "stage16a_r0_protocol.json", "SERVER_STOP_REVIEW.md")]),
        entry("Stage17A", "Does an auditable train-only Vanilla baseline beat RealRepeat reproducibly?",
              "Authorized train-only rebuild, separately isolated normals, 80 fixed slots x3 generation banks; detector seed42, epoch150last.",
              {"status": s17a["status"], "mean_delta_pp": 100*s17a["mean_delta"], "wins": s17a["wins_vs_RR"],
               "class_mean_delta_pp": {k: 100*v for k, v in s17a["mean_class_deltas_raw"].items()},
               "successful_update_range": s17a["successful_step_range"]},
              ["Within this legacy frozen calibration, raw banks all lose to RR (0/3); baseline development not authorized."],
              ["Three detector-seed replication.", "Direct numerical pooling with V2.", "Universally useless generation."],
              True, (True, "Recorded final-last evaluation under legacy dataset-defined annotation protocol."), no_selector,
              [ROOTS["17a"] + n for n in ("stage17a_status.json", "STAGE17A_REPORT.md")],
              ["Known overlap in earlier rebuild led to a separate train-only rebuild; contaminated checkpoint is not the authoritative final arm.",
               "Scheduled budget equivalence, not exact successful-update equality; annotations/unknown physical sources remain qualified.",
               "Actual stored status is SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE; no invented synonym replaces it."]),
        entry("Stage18A", "Is another auditable, raw generator baseline already available?",
              "Existing-asset audit, LOCAL then SERVER_EXISTING_EVIDENCE; code/checkpoint/lineage/annotation/selector checks, no model run.",
              {"status": s18a["status"], "scope": s18a["scope"], "selected_generator": alternatives["selected_generator"],
               "STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED": False},
              ["No candidate qualifies in inspected evidence; DWBG761 is a derived pool, not untouched raw MSDF."],
              ["Global absence or inferiority of AnomalyDiffusion/DefectFill/MSDF.", "A scientific reason to switch generator families immediately."],
              False, no_val, (True, "Historical selector dependence was audited, not executed."),
              [ROOTS["18a"] + n for n in ("stage18a_status.json", "stage18b_candidate_selection.json", "STAGE18A_REPORT.md")],
              ["Explicit inspected roots, not exhaustive server inventory; asset completeness is not a comparative scientific-fit study."]),
        entry("Stage20A", "Can a NEW group-isolated protocol permit raw SD2 calibration?",
              "V2 seed2026, known-source group train/final_eval75/25; TRAIN-only generator/normals; baseline gates frozen before performance.",
              {"status": s20a["status"], "baseline_calibration_authorized": True, "TDCRG_DEVELOPMENT_AUTHORIZED": False},
              ["Protocol readiness with no KNOWN cross-split source conflict."],
              ["Utility or quality; exhaustive physical-source independence."], False,
              (True, "Administrative old-eval image/label registration only: 46/46; zero model forwards or performance-based split choice."), no_selector,
              [ROOTS["20a"] + n for n in ("stage20a_status.json", "STAGE20A_REPORT.md", "future_baseline_gate.json")],
              ["New V2 is not a retrospective rescue of Stage17/18. physical_source_complete=false; dataset-defined labels and weak bbox masks."]),
        entry("Stage20B-P0", "Are fixed slots, assets, paired RNG and exposure structurally bound?",
              "Authoritative server_preflight_20261008_084115; frozen local metadata reused, no server slot redesign or GPU forward test.",
              {"status": sp0["status"], "slots": sp0["slots"], "TRAIN_normals": sp0["TRAIN_normals"],
               "TRAIN_assets_hash_verified": sp0["TRAIN_assets_hash_verified"], "runtime_preflight_is_not_forward_test": True},
              ["Structural binding pass, no headroom evidence."], ["Numerical/runtime qualification or synthetic utility."],
              False, no_val, no_selector, [ROOTS["p0s"] + n for n in ("stage20b_p0_status.json", "STAGE20B_P0_REPORT.md")]),
        entry("Stage20B-R0", "Does the frozen runtime pass differentiation, RNG and budget qualification?",
              "Authoritative server_runtime_20261008_121254; finite native backward and deterministic public-base probes; no optimizer step.",
              {"status": sr0["status"], "formal_execution_ready": sr0["STAGE20B_FORMAL_EXECUTION_READY"],
               "runtime_probe_generated_images": sr0["runtime_probe_generated_images"], "formal_synthetic_generation_count": 0},
              ["Runtime qualification of tested frozen paths; four non-formal probe images."],
              ["Synthetic headroom, trained-checkpoint quality, long-run stability."], False, no_val, no_selector,
              [ROOTS["r0"] + n for n in ("stage20b_r0_status.json", "STAGE20B_R0_REPORT.md")],
              ["The four public-base runtime probe images must not be hidden in an unqualified claim of zero physical sampling."]),
        entry("Stage20B", "Does selector-free raw SD2 V2 provide stable positive headroom?",
              "136 real+80extras, 40/class; A/B/C generation42/3407/2026; detector42,150epochs, sameinit/order/augmentation; one final_eval barrier.",
              {"status": headroom["status"], "delta_pp": delta, "mean_delta_pp": mean_delta, "sample_sd_pp": std_delta,
               "wins": wins, "gates": gates, "class_mean_delta_pp": class_mean},
              ["Stable positive headroom NOT confirmed; one positive realization; bank range7.6383pp; Black-centered degradation."],
              ["TDCRG authorization, best-bank rescue, population-level RNG causal effect or permanent Black quality failure."],
              True, (True, "One unified final_eval round after all4 checkpoint freezes:48 images/60 boxes,192 image-forwards; primaryepoch150last."),
              no_selector, [ROOTS["20b"] + n for n in ("stage20b_status.json", "stage20b_headroom.json", "stage20b_classwise.json", "manual_arm_C_recovery.json", "STAGE20B_REPORT.md")],
              ["Three generation realizations, one detector seed, one split; not population variance estimation or detector-seed confirmation.",
               "C restarts epoch1-150, not epoch77 resume; observed interrupted C physical compute separately disclosed.",
               "No KNOWN source leakage only; dataset-defined annotation utility, not exhaustive physical-defect truth."]),
    ]

    supported = [
        {"id": "S1", "claim": "Stable positive raw SD2 synthetic utility was not demonstrated under frozen V2.", "evidence": ["Stage20B"], "scope": "mean-2.5901pp,1/3wins, three gates fail; not a claim of negative expectation for every future bank"},
        {"id": "S2", "claim": "Downstream performance varied by7.64pp across the three generation-RNG realizations under the frozen protocol.", "evidence": ["Stage20B"], "scope": "observed paired bank variation, not a population causal-effect magnitude"},
        {"id": "S3", "claim": "RawC beat RR by+1.3781pp; one positive realization is insufficient for baseline confirmation.", "evidence": ["Stage20B"], "scope": "observed fact ONLY; no selection of C for rescue"},
        {"id": "S4", "claim": "Black AP is the major contributor to Stage20B mean failure.", "evidence": ["Stage20B"], "scope": "downstream outcome attribution, not a generator image-quality mechanism"},
        {"id": "S5", "claim": "BootstrapGuard selection utility and raw generator utility are scientifically separate questions.", "evidence": ["Stage18A", "Stage20B", "BootstrapGuard legacy downstream"], "scope": "historical conditional selection comparison cannot confirm V2 raw headroom"},
    ]
    diagnostic = [
        {"id": "D1", "claim": "Morphology alignment is a subset utility signal, not established broad fidelity causality.", "evidence": ["Stage14A", "Stage14B"]},
        {"id": "D2", "claim": "Class-specific subset interaction is observed; a class-adaptive generator mechanism remains unproven.", "evidence": ["Stage15A", "Stage15B"]},
        {"id": "D3", "claim": "Positive RawC motivates an instability question, not a stable-headroom conclusion.", "evidence": ["Stage20B"], "label": "ONE_POSITIVE_GENERATION_REALIZATION", "distinction": "RawC numerical win is SUPPORTED; its generalization/mechanism interpretation is DIAGNOSTIC_ONLY."},
        {"id": "D4", "claim": "Sample-set composition may relate to utility, but no responsible image property or optimization cause is isolated.", "evidence": ["Stage14B", "Stage15A", "Stage15B", "Stage20B"]},
    ]
    unsupported = [
        {"id": "N1", "claim": "Visual fidelity implies downstream utility.", "reason": "Stage14A missing perceptual metrics; Stage14B KID does not improve over Random and LPIPS is essentially equal."},
        {"id": "N2", "claim": "Morphology maximization guarantees utility.", "reason": "Stage15A interaction and Stage15B failed dose gates; subset interventions are not clean generator interventions."},
        {"id": "N3", "claim": "Raw SD2 baseline provides stable positive headroom.", "reason": "Stage17A failed its own calibration; Stage20B independently fails its frozen V2 gates; do not pool magnitudes."},
        {"id": "N4", "claim": "TDCRG is a currently justified confirmatory next step.", "reason": "Stage20B gates A/B/C all fail; its authorization remains false."},
        {"id": "N5", "claim": "Synthetic generation is universally useless.", "reason": "Limited datasets/protocols, RawC win, and separate selected-subset signals do not support universal dismissal."},
    ]
    shared = {"generator_checkpoints": True, "80_slots": True, "conditioning_normals": True,
              "prompts": True, "geometry": True, "scheduler_and_inference_parameters": True,
              "detector_initialization": True, "detector_seed": True, "role_order_and_augmentation_seeds": True,
              "primary_exposure_and_successful_updates": True}
    recovery_boundary = {
        "source_stopped_run": "formal_20261009_060552", "authoritative_recovered_run": pointer["run"],
        "C_restart": "original public initialization + seed42, epoch1 through150",
        "epoch77_resume": False, "matched_primary_arms": compute["arms"],
        "extra_interrupted_C_physical_compute": recovery["interrupted_C_observed_counters"],
        "interrupted_counters_may_be_lower_bounds": recovery["interrupted_counters_may_be_lower_bounds"],
        "extra_interrupted_compute_in_matched_primary": False,
        "external_SIGKILL_cause_independently_verified": False,
        "source_run_unchanged_SHA_verified": recovery["source_run_unchanged_SHA_verified"],
        "reused": ["both generators", "all240raw outputs", "RR/A/B primary detector checkpoints"],
        "earlier_OOM_runs_preserved": ["formal_20261009_043250", "formal_20261009_052723"],
        "all_failed_attempt_physical_compute_is_primary_matched": False,
        "early_OOM_full_physical_compute_totals_reconstructed": False,
    }
    candidates = [
        {"id": "A", "direction": "synthetic-real curriculum / exposure scheduling",
         "question": "With an unchanged generator, all frozen synthetic samples retained and fixed total real/synthetic exposure, can temporal organization reduce across-realization utility variability without class AP sacrifice?",
         "evidence_basis": "Stage20B instability motivates an integration question, but does not isolate exposure timing as its cause.",
         "minimality": "One conceptual training-time axis; no new generator, image score, learned gate or sample ranking.",
         "risk": "Schedule search on the existing final_eval would be post-hoc tuning; changing total exposure would confound integration with quantity.",
         "selector_boundary": "No dropping samples, quality-based order, class suppression or zero exposure; same complete frozen input population.",
         "priority": "PRIMARY_PROBLEM_FORMULATION_ONLY", "effectiveness_supported": False},
        {"id": "B", "direction": "uncertainty-aware synthetic weighting",
         "question": "Can training-time uncertainty regulate synthetic gradient contribution without excluding samples?",
         "evidence_basis": "No calibrated uncertainty evidence in Stage20B.",
         "minimality": "Less minimal: an uncertainty definition, calibration and scale introduce additional assumptions.",
         "risk": "Per-sample zero/near-zero weights or quality rank is a soft selector and overlaps BootstrapGuard; uncertainty can favor easy examples.",
         "selector_boundary": "No permanent sample gate/rank/topK; all samples and nonzero exposure retained. Merely calling scores 'weights' is insufficient separation.",
         "priority": "DEFER_HIGH_SELECTOR_OVERLAP", "effectiveness_supported": False},
        {"id": "C", "direction": "class-adaptive synthetic-real optimization",
         "question": "Can an integration rule avoid class-asymmetric harm without selecting samples or permanently disabling a class?",
         "evidence_basis": "V2 degradation is concentrated in Black, but Stage17A degradation was stronger in Flash; no universal class rule.",
         "minimality": "Potentially small, but a class rule needs independent train-only evidence rather than fitting these APs.",
         "risk": "Post-hoc Black-only penalty/suppression would tune to observed final_eval; changed per-class exposure confounds mechanism.",
         "selector_boundary": "Retain both classes and complete sample sets; no Black filtering or final_eval-driven class weights.",
         "priority": "SECONDARY_HYPOTHESIS_NOT_BLACK_RESCUE", "effectiveness_supported": False},
        {"id": "D", "direction": "synthetic consistency regularization",
         "question": "Can an explicit training-time consistency constraint improve robustness to synthetic variation?",
         "evidence_basis": "Prediction inconsistency was not measured as the failure cause.",
         "minimality": "Requires a transformation/target-consistency assumption not supplied by current evidence.",
         "risk": "Wrong or weak labels can be reinforced; extra forwards/targets and hyperparameters muddy compute attribution.",
         "selector_boundary": "All samples retained; no consistency-based acceptance/rejection.",
         "priority": "DEFER_UNMEASURED_MECHANISM", "effectiveness_supported": False},
        {"id": "E", "direction": "distribution-aware data integration",
         "question": "Can real/synthetic distribution information control joint optimization without post-generation sample ranking?",
         "evidence_basis": "Historical concentration diagnostics do not prove distribution mismatch causes V2 failure.",
         "minimality": "Highest conceptual burden: representation, distribution and aggregation choices.",
         "risk": "A nearest-real/guardrail score used for per-sample exclusion would reproduce selector behavior; teacher-space and domain assumptions unresolved.",
         "selector_boundary": "Training-level aggregate integration only; no novelty/fidelity quality score or pool filtering.",
         "priority": "DEFER_COMPLEXITY_AND_BG_OVERLAP", "effectiveness_supported": False},
    ]
    options = [
        {"option": "A", "route": "SD2 + direct TDCRG development", "status": "OPTION_A_CURRENTLY_NOT_AUTHORIZED",
         "scientific_justification": "Insufficient: prespecified prerequisite headroom gates all fail.",
         "risk_of_post_hoc_tuning": "High: constructing a task loss to rescue a known failed baseline without its entry criterion.",
         "experimental_cost": "High: generator mechanism, isolated teachers, training, repeated generation and downstream detectors.",
         "clean_causal_interpretation": "Not available now; failed entry gate cannot be reinterpreted as mechanism authorization.",
         "overlap_with_BootstrapGuard": "Different nominal intervention stage, but teacher/quality guidance and later selection could confound attribution.",
         "thesis_coherence": "A generator-first thesis premise is not supported by current independent baseline evidence.",
         "implementation_authorized": False},
        {"option": "B", "route": "Switch to another published generator family", "status": "OPTION_B_NOT_RECOMMENDED_AS_IMMEDIATE_NEXT_STEP",
         "scientific_justification": "Insufficient in existing evidence: Stage18A is an asset audit, not a demonstrated pre-existing family-to-problem fit rationale.",
         "risk_of_post_hoc_tuning": "High if families are swept after SD2 failure until one gives positive AP.",
         "experimental_cost": "High: reconstruct clean lineage, annotation contract, comparable training and raw bank calibration.",
         "clean_causal_interpretation": "A clean future comparison is conceivable only after an independent scientific rationale and protocol review; not authorized here.",
         "overlap_with_BootstrapGuard": "Only distinguishable if genuinely raw outputs; existing derived DWBG pool is not a raw alternative.",
         "thesis_coherence": "A published family alone is not a novel contribution, and asset absence is not proof of family inferiority.",
         "automatic_family_recommendation": None, "implementation_authorized": False},
        {"option": "C", "route": "Pause generator-mechanism-first; reframe innovation1 as non-selector training/data integration",
         "status": "OPTION_C_PREFERRED_PROBLEM_REFRAMING_ONLY",
         "scientific_justification": "Observed instability and class asymmetry justify asking how fixed synthetic inputs participate in optimization; no integration solution is yet validated.",
         "risk_of_post_hoc_tuning": "Still material: fitting weights/curricula to RawC or Black final_eval would invalidate confirmation. Label all candidates as exploratory hypotheses.",
         "experimental_cost": "Conceptual work is zero training now; any future training still needs independent protocol authorization. Avoid generator retraining by definition.",
         "clean_causal_interpretation": "Potentially cleaner if generator, all samples, total exposure, labels and model are fixed and only a prespecified integration factor varies; no experiment is designed here.",
         "overlap_with_BootstrapGuard": "Distinct only without retain/drop/rank decisions; uncertainty weighting and distribution guardrails can otherwise become soft selectors.",
         "thesis_coherence": "Innovation1: how to use synthetic data in optimization; innovation2: which synthetic data to retain. Method novelty/effectiveness still unproved.",
         "implementation_authorized": False},
    ]
    base = {"stage": "Stage21A", "scope": "EXISTING_FORMAL_EVIDENCE_SYNTHESIS_ONLY", "source_revision": REVISION}
    status = {
        **base, "status": "GENERATOR_ROUTE_NOT_JUSTIFIED_CURRENTLY", "full_evidence_synthesis_complete": True,
        "Stage20B_formal_status_preserved": formal["status"], "RawC_label": "ONE_POSITIVE_GENERATION_REALIZATION",
        "OPTION_A": options[0]["status"], "OPTION_B": options[1]["status"], "OPTION_C": options[2]["status"],
        "TDCRG_DEVELOPMENT_AUTHORIZED": False, "GENERATION_AUTHORIZED": False, "DETECTOR_TRAINING_AUTHORIZED": False,
        "NEW_TRAINING_MECHANISM_IMPLEMENTATION_AUTHORIZED": False, "STAGE21B_EXECUTION_AUTHORIZED": False,
        "stage21b_auto_start": False, "BootstrapGuard_execution_authorized": False,
        "historical_results_modified": False, "Stage20A_modified": False, "Stage20B_modified": False,
        "activity_counts": ZERO_ACTIVITY,
        "route_choice_basis": "evidence sufficiency and separation of contributions, not a requirement to obtain positive results",
        "future_work": "Conceptual hypothesis review and independently authorized protocol only; no new experiment/module launched.",
    }
    findings = {**base, "SUPPORTED": supported, "DIAGNOSTIC_ONLY": diagnostic, "NOT_SUPPORTED": unsupported,
                "not_supported_does_not_mean_disproved_universally": True, "cross_protocol_pooling_performed": False,
                "post_hoc_problem_reframing": True, "current_final_eval_reusable_as_independent_confirmation": False}
    failure = {**base, "status": formal["status"], "primary_metric": "epoch150 last.pt mAP50-95",
               "arms": {arm: {"map50_95": row["map5095"], "map50_95_percent": 100*row["map5095"],
                              "delta_vs_RR_pp": 0 if arm == "RR" else delta[arm]}
                        for arm, row in metrics.items()},
               "mean_delta_pp": mean_delta, "sample_sd_delta_pp": std_delta, "wins": wins,
               "gates": {g: {"pass": passed, "frozen_rule": rule} for g, passed, rule in (
                   ("A", gates["A"], "mean>=+0.50pp"), ("B", gates["B"], "strict wins>=2/3"),
                   ("C", gates["C"], "every class mean>=-2.0pp"))},
               "class_mean_delta_pp": class_mean, "class_contribution_to_macro_mean_pp": {c: v/2 for c, v in class_mean.items()},
               "contribution_definition": "Existing two-class macro mAP identity: half of each AP delta; not a new quality score.",
               "manual_C_recovery_boundary": recovery_boundary,
               "mechanism_responsible_for_failure_identified": False,
               "unidentified_causes": ["image fidelity", "label/weak-mask mismatch", "sample composition", "optimization dynamics"],
               "unidentified_causes_are_hypotheses_not_findings": True}
    rng = {**base, "status": "OBSERVED_GENERATION_REALIZATION_VARIABILITY", "shared_controls": shared,
           "paired_bank_metadata_independently_rechecked": True, "only_design_factor_differing": "generation RNG realization",
           "generation_seeds": nuisance["bank_seeds"], "detector_seed": 42,
           "sample_sd_pp": std_delta, "range_pp": bank_range, "n_generation_banks": 3, "n_detector_seeds": 1,
           "recommended_wording": "Downstream performance varied by7.64pp across the three generation-RNG realizations under the frozen protocol.",
           "causal_population_effect_size_estimated": False, "population_variance_estimated_with_precision": False,
           "best_bank_selected_for_future_use": False, "bank_C_confirmatory_status": "ONE_POSITIVE_GENERATION_REALIZATION",
           "limits": ["RNG necessarily changes generated pixels/sample composition, not just a seed label.",
                      "One detector seed and split; common RR, so bank deltas are paired and not independent RR replicates.",
                      "C timing differs due to disclosed restart; no independent external-kill proof.",
                      "No new bank, seed sweep or image-property analysis to rescue the result."]}
    asymmetry = {**base, "status": "CLASS_ASYMMETRIC_UTILITY_FAILURE", "class_bank_delta_pp": class_delta,
                 "class_mean_delta_pp": class_mean, "contribution_to_macro_mean_pp": {c: v/2 for c, v in class_mean.items()},
                 "black_negative_banks": sum(v < 0 for v in class_delta["black"].values()),
                 "supported_wording": "The downstream degradation is concentrated in Black AP under the current protocol.",
                 "Black_generator_quality_bad_proven": False, "Black_morphology_wrong_proven": False,
                 "label_error_proven": False, "Flash_no_harm_equivalence_proven": False,
                 "Flash_limit": "Mean near zero is descriptive; A loses Flash AP and no equivalence test was conducted.",
                 "cross_stage_caution": "Stage17A Flash/Black means were-8.4172/-1.7475pp under another protocol; no persistent Black-only failure rule or pooled class effect."}
    bgdelta = bg["random"]["map5095"]["mean_delta"]
    separation = {**base, "innovation1_question": "How should synthetic data be used during detector optimization?",
                  "innovation2_question": "Which synthetic samples should be retained?",
                  "innovation1_intervention": "training-time integration of an unchanged complete sample population",
                  "BootstrapGuard_intervention": "post-generation sample selection",
                  "legacy_BG_minus_Random_mean_pp": 100*bgdelta,
                  "historical_claim_scope": "Historical selector comparison conditional on its frozen derived pool/protocol; not a clean V2 replication or raw generator confirmation.",
                  "Stage18A_lineage_qualification_applies": True,
                  "separate_contribution_contract": [
                      "Innovation1 cannot include post-generation inclusion/exclusion, Top-K, best-of-N or permanent sample gates.",
                      "Continuous sample weighting can overlap a soft selector; nomenclature alone does not establish independence.",
                      "Baseline utility must be evaluated without BootstrapGuard; never retrofit selected utility as raw utility.",
                      "Keep component effects and budgets separate; a combined pipeline gain alone cannot identify either contribution.",
                      "Do not tune or run BootstrapGuard now, and do not transfer historical absolute AP to V2.",
                  ], "BootstrapGuard_implemented_modified_or_run_in_stage21a": False,
                  "independent_V2_BootstrapGuard_utility_confirmed": False}
    outputs = {
        "generator_evidence_timeline.json": {**base, "entries": timeline, "pooling_performed": False, "legacy_limits": legacy_limits},
        "supported_findings.json": findings,
        "stage20b_failure_decomposition.json": failure,
        "rng_instability_interpretation.json": rng,
        "class_asymmetry_interpretation.json": asymmetry,
        "innovation_route_options.json": {**base, "options": options, "preferred_option": "C", "preference_is_problem_formulation_not_method_validation": True},
        "innovation1_reframing_candidates.json": {**base, "candidates": candidates, "primary_question": candidates[0]["question"],
                                                    "new_experimental_protocol_designed": False, "method_novelty_established": False,
                                                    "all_candidates_concept_only": True, "execution_authorized": False},
        "bootstrapguard_separation.json": separation,
        "stage21a_status.json": status,
    }
    e.require("all mandatory timeline stages", tuple(x["stage"] for x in timeline) == REQUIRED_STAGES)
    e.require("zero new activity", all(v == 0 for v in ZERO_ACTIVITY.values()))
    e.close("additive macro failure contribution", sum(v/2 for v in class_mean.values()), mean_delta)
    e.require("no candidate implementation authorized", all(not x["implementation_authorized"] for x in options))
    # Prevent malformed floats or numpy-like implicit serialization; stdlib only.
    for value in outputs.values():
        encoded(value)
    outputs["source_evidence_index.json"] = {**base, "sources": e.sources(), "primary_source_files_modified": False,
        "read_strategy": "Pinned Git metadata, including deleted historical worktree files; no restore, server access or image/label content reads.",
        "Stage16A_stop_record_limitation": "Original server status blob is unavailable at this revision; cite its explicit frozen Stage16A-R0 declaration, not the local snapshot.",
        "existing_results_used_to_choose_route": True, "new_method_hypotheses_are_post_hoc_exploratory": True}
    outputs["evidence_verification.json"] = {**base, "status": "PASS", "checks": e.checks,
        "check_count": len(e.checks), "source_metadata_blob_count": len(e.blobs),
        "GPU_execution_or_AP_recalculation_from_predictions": False,
        "verification_scope": "Recompute published-metric arithmetic and gates; compare pinned slot/checkpoint/compute/recovery metadata. No model execution."}
    outputs["STAGE21A_REPORT.md"] = render_report(outputs, bgdelta)
    outputs[".gitattributes"] = "* -text -whitespace\n"
    return outputs


def render_report(outputs, bgdelta):
    f = outputs["stage20b_failure_decomposition.json"]
    rows = "\n".join(f"| {a} | {r['map50_95_percent']:.4f} | {r['delta_vs_RR_pp']:+.4f} |" for a, r in f["arms"].items())
    historical = "\n".join(f"| {x['stage']} | `{x['result']['status']}` | {'确认/校准尝试' if x['whether_confirmatory'] else '筛查/资格/审计'} |" for x in outputs["generator_evidence_timeline.json"]["entries"])
    return f"""# Stage21A — 证据综合与生成路线决策

正式状态：`GENERATOR_ROUTE_NOT_JUSTIFIED_CURRENTLY`。

当前不继续 Plastic_Bomo 的 generator-mechanism-first 路线。不实现 TDCRG，不立即换生成器。优先重新定义第一创新为**非 selector 的 training/data-integration 问题**；这是问题层面的选择，不是已验证的方法，也不是新实验授权。BootstrapGuard 保持独立第二创新。

本阶段训练、生成、检测训练、optimizer、final_eval forward、新模块实现、BootstrapGuard、DeepPCB 均为0。只读取 Git 中已有审计/指标 metadata；没有打开 final_eval 图像/标签，没有新模型运行，也没有修改 Stage14–20 的文件。

## 1. 三个问题的回答

A. 当前仍有充分依据继续 generator-mechanism-first 吗？**没有。** Legacy 子集的收益没有确认生成器机制；Stage17A 和新的 V2 Stage20B 各自在其协议内未确认 raw headroom。不能把历史子集、资格检查或一次 RawC 胜出升级为生成创新授权。

B. 第一创新应转向什么问题？优先问：**不改变生成器、不筛样、保留完整冻结合成集合，并固定总 real/synthetic exposure 时，训练过程中的使用时序能否降低不同生成 realization 的 utility 波动，同时避免类别 AP 受损？** 现有实验没有操纵时序，因此没有证明它是原因或有效解法。Stage21A 不指定 schedule、权重、阈值、训练配置，也不启动 Stage21B。

C. BootstrapGuard 如何独立？第一创新问“How should synthetic data be used during detector optimization?”；BootstrapGuard 问“Which synthetic samples should be retained?”。前者的干预在训练使用方式，后者在生成后的保留集合。不能用后者的历史收益解释前者的 raw baseline，也不能以近零 sample weighting 偷换为另一个 selector。

## 2. Stage20B 结论原样保留

权威运行：`manual_C_20261010_032543`；来源停止运行：`formal_20261009_060552`。

正式状态始终为 `GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED`；`TDCRG_DEVELOPMENT_AUTHORIZED=false`。

| Arm | mAP50-95 (%) | 相对 RR (pp) |
|---|---:|---:|
{rows}

平均 delta={f['mean_delta_pp']:+.4f}pp；样本 SD={f['sample_sd_delta_pp']:.4f}pp；wins=1/3。冻结 gate A（均值≥+0.50pp）、B（严格胜出≥2/3）、C（每类均值≥−2.0pp）全部 FAIL。没有修改指标、阈值、seed 或成功定义。

RawC 只记为 `ONE_POSITIVE_GENERATION_REALIZATION`。其胜出是可核实事实，但“因此 baseline 稳定有效”不受支持。不能挑 C、增加 bank、调参或加入 selector 来挽救结论。

## 3. Manual C recovery 与计算边界

C 从 original public initialization、seed42 **epoch1 重新训练到150**，不是 epoch77 resume。两类生成器、240个 raw 输出和已完成 RR/A/B 复用；原停止运行及先前两次 Flash OOM 记录保留。

主要比较 RR/A/B/C 各150epochs、32400draws、600optimizer attempts、592successful updates、8AMP skips，init/order/augmentation metadata 匹配。旧 C 的额外物理计算单列：已观察76epochs、16416draws、304attempts、296successful updates、8AMP skips，可能只是下界；它不进入 matched primary exposure。早期 OOM 的完整物理计算总量没有重建，不能声称全部实际花费都相等。

用户报告可能误杀进程，退出码−9已记录；**外部 SIGKILL 原因没有独立证实**。旧运行文件未改。

## 4. 历史证据时间线

“确认/校准尝试”表示执行过预设验证问题，不表示结论通过。

| 阶段 | 保留的正式状态 | 证据类型 |
|---|---|---|
{historical}

完整 question/protocol/result/support/non-support/validation/selector 字段及来源见 [generator_evidence_timeline.json](generator_evidence_timeline.json)。

- Stage14A：已纠正为 `IMAGE_PROPERTY_BOTTLENECK_NOT_IDENTIFIED`。当时 crop-KID/real-nearest LPIPS 缺失，high/low 是 morphology-matched 子集；+2.286/−3.935pp 只作诊断。
- Stage14B：High−Random +1.043±1.387pp，2/3wins；Flash +3.406pp、Black −1.320pp。High 的 set-KID×1000=44.722，Random=40.011；nearest-real LPIPS≈0.20905/0.20935。不是普遍 pixel/perceptual fidelity 改善。KID 没有用于 per-sample 排序。
- Stage15A：Flash条件效应在Random-Black下+2.387pp、在High-Black下−0.110pp；interaction−2.497pp。可支持条件子集效应，但不能锁定 morphology 特异机制。没有独立验证历史全部 optimizer/init 原始字节。
- Stage15B：M50−M00 +0.674pp；M50−mean(M10,M01)−1.095pp；M11−M50 +0.368pp；C/D gates失败。High/Random overlap 导致 nominal/physical dose不一致，浓缩指标不能证明 utility 损害。
- Stage16A/16A-R0：mapping/provenance 入口停止，**不是 task guidance 无效**的实验。原 Stage16A server status blob在当前提交中缺失：引用 R0 冻结 protocol 的明确 stop declaration，不能把 LOCAL snapshot 冒充服务器原结果。R0 accepted groups两类均0，后续GPU gates未运行；也不能由此断言标签一定错。
- Stage17A：权威 train-only rebuild 的三 bank mean−5.0824pp、0/3wins，存储状态为 `SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE`。Flash/Black−8.4172/−1.7475pp；成功更新range2。先前有重叠的 rebuild 并非最终权威模型。
- Stage18A：SERVER_EXISTING_EVIDENCE 中没有合格替代，非全球不存在。AnomalyDiffusion/DefectFill/MSDF缺少完整合格链；DWBG761是重组/过滤派生池，不是761个原始MSDF输出。历史 Random是过滤池内随机，不是 raw utility。
- Stage20A/P0/R0：分别证明新协议、结构绑定、已测运行路径的资格；不是收益验证。Stage20A曾行政性读取旧eval的46图/46label以重分组，没有forward；R0实际生成过4张公开base探针，**非formal synthetic**，不能抹去物理采样事实。

Stage14–15是旧标签/来源边界下的探索性子集链；后续使用了先前validation结果形成问题。不能把这条连续探索链包装为独立 generator-mechanism confirmation。Stage20A重分组后 V2 与历史 split/protocol 不同，**没有跨协议绝对AP比较、平均效应合并或历史阈值转移**。以前报告的下一步建议不等于当前授权。

## 5. 三层证据

SUPPORTED：V2未示稳定raw正收益；三生成realizations有大幅观察波动；RawC一次胜出但未确认baseline；Black贡献主要平均损失；selector utility与raw utility是独立命题。

DIAGNOSTIC_ONLY：morphology alignment信号、class-specific子集interaction、RawC正结果的推广/机制解释、sample composition与utility可能有关。

NOT_SUPPORTED：visual fidelity必然带来utility、morphology最大化保证utility、raw SD2有稳定正headroom、立即TDCRG是合格确认步骤、所有synthetic都无用。

上述分类区分“数字事实”与“机制/泛化推断”，见 [supported_findings.json](supported_findings.json)。未支持不等于普遍被证伪。

## 6. RNG instability 与类别非对称

A/B/C共用 class checkpoints、80slots、normal/prompt/geometry、scheduler/采样参数以及 detector init/seed/order/augmentation/exposure。成对metadata再次核对；设计上只改变 generation RNG，生成像素/集合内容随之变化。

准确表述：**downstream performance varied by7.64pp across the three generation-RNG realizations under the frozen protocol.** 样本SD3.8279pp，range7.6383pp。这不是“RNG导致7.64pp的总体因果效应”，更不是三个detector seeds；三个bank、单个split/seed不足以估计总体variance或找到导致波动的图像机制。

类别状态为 `CLASS_ASYMMETRIC_UTILITY_FAILURE`。Flashmean+0.0357pp，Blackmean−5.2160pp，Black在三个bank均负。按现有两类macro AP恒等式：Flash贡献+0.0179pp、Black贡献−2.6080pp，总计−2.5901pp；这只是原指标的代数拆分，不是新加权质量分数。

只能说 **downstream degradation is concentrated in Black AP under the current protocol**。没有直接证据证明Black生成质量坏、morphology错或标注错。Flash均值近0也不证明无害（A的Flash下降）。Stage17A反而Flash下降更多，不能据此制定普遍Black-only惩罚/排除规则。

## 7. 三条路线的正式选择

| 选项 | 当前决定 | 理由与主要风险 |
|---|---|---|
| A：直接SD2+TDCRG | `OPTION_A_CURRENTLY_NOT_AUTHORIZED` | 三个前置gate全失败；高成本，事后救结果风险高，缺少合格机制起点 |
| B：换published generator | `OPTION_B_NOT_RECOMMENDED_AS_IMMEDIATE_NEXT_STEP` | 资产审计不是独立scientific-fit理由；不能逐个换到正结果，也不能由资产不齐断言family不适合 |
| C：非selector训练/数据整合 | 优先**问题重定义**，不授权实现 | 固定生成器和完整数据，可更清楚区分优化参与方式；仍须警惕软selector、exposure混杂和事后调参 |

三个选项均按 scientific justification、post-hoc risk、cost、causal interpretation、BootstrapGuard overlap、thesis coherence独立评价，详见 [innovation_route_options.json](innovation_route_options.json)。没有按“必须获得正结果”选择；没有新外部文献适配性比较或generator family推荐。

## 8. 第一创新候选仅作概念分析

优先是synthetic-real exposure **使用时序**的问题。其他候选依次作为待审假设：uncertainty-aware weighting、class-adaptive optimization、consistency regularization、distribution-aware integration。

它们均**没有有效性证据**。uncertainty weighting可能变为soft selector；class rule可能事后针对Black调参；consistency可能强化错/弱标签且改变compute；distribution integration依赖尚未证实的表征/分布假设且容易重叠BG。不存在现成的“复杂模块方案”。完整风险和非selector边界见 [innovation1_reframing_candidates.json](innovation1_reframing_candidates.json)。

未来即使考虑实验，也必须另行授权、预先固定非selector干预及预算、保留全集合，并说明与已有final_eval接触的隔离方案。当前这些最终指标已经用于选择研究问题，不能再拿同一final_eval调参后声称独立确认。Stage21A没有给出新实验协议或启动命令。

## 9. BootstrapGuard贡献边界

历史报告中的BG−Random平均+{100*bgdelta:.4f}pp属于其旧派生池/协议下的conditional selector结果；不搬到V2，不用于否定Stage20B。Stage18A关于旧池lineage/validation未知的限制继续有效。当前没有独立V2 BG confirmation，也没有运行BG。

贡献1保留完整冻结集合、干预训练使用方式；贡献2干预生成后保留集合。不能把continuous weights换名当成独立贡献，也不能只报告组合pipeline收益而掩盖各自作用/预算。

## 10. 可复核文件与停止点

[source_evidence_index.json](source_evidence_index.json) 给出每份metadata的固定commit、路径、SHA256及GitHub链接；删除的历史worktree文件只读Git，不恢复、不改写。[evidence_verification.json](evidence_verification.json) 记录独立指标代数、gate、成对slot/checkpoint、计算预算与恢复边界检查。

所有来源锁定 `{REVISION}`。本地复核仅CPU/标准库，没有重算GPU/AP预测。此报告是既有final指标的路线综合，不是新确认实验、新方法有效性或论文新颖性证明。

停止于 Stage21A 文件与GitHub提交。**不训练、不生成、不实现TDCRG/新generator/curriculum/weighting/regularization，不运行BootstrapGuard/DeepPCB，不启动Stage21B。**
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true", help="create only the new fixed Stage21A metadata directory")
    modes.add_argument("--check", action="store_true", help="read-only reproducibility check of every output")
    a = p.parse_args()
    repo = a.repo.resolve()
    outputs = build(repo)
    out = repo / RESULT
    data = {name: value.encode("utf-8") if isinstance(value, str) else encoded(value) for name, value in outputs.items()}
    # Bind all generated deliverables (the manifest cannot recursively bind itself).
    data["stage21a_artifact_manifest.json"] = encoded({
        "source_revision": REVISION,
        "files": {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(data.items())},
        "self_hash_included": False,
    })
    if a.write:
        if out.exists():
            raise RuntimeError("STAGE21A_OUTPUT_EXISTS: use --check; never overwrite existing evidence")
        out.mkdir(parents=True)
        for name, blob in data.items():
            (out / name).write_bytes(blob)
    else:
        for name, blob in data.items():
            if (out / name).read_bytes() != blob:
                raise RuntimeError("STAGE21A_OUTPUT_MISMATCH: " + name)
    print(json.dumps({"status": outputs["stage21a_status.json"]["status"], "mode": "WRITE_NEW_METADATA" if a.write else "READ_ONLY_CHECK",
                      "files": len(data), "evidence_checks": outputs["evidence_verification.json"]["check_count"],
                      "new_activity": ZERO_ACTIVITY}, indent=2))


if __name__ == "__main__":
    main()
