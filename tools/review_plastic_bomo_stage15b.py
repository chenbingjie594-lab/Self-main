"""Review downloaded Stage15B evidence on CPU, without evaluation or training.

Recompute outcomes with stdlib statistics and feature rank with SVD (the
production audit uses a Gram eigendecomposition). Preserve all raw outputs.
"""
from __future__ import annotations
import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from plastic_bomo_stage15b_common import load, digest, sha, save, select_balanced, candidate_record

SEEDS = (42, 3407, 2026)
ARMS = ("M00", "M10", "M01", "M11", "M50")
CLASSES = ("flash", "black")
OUTCOMES = ("map50_95", "map50", "recall", "flash_ap50_95", "black_ap50_95")
EFFECTS = ("D50_00", "D_balance", "D80_50", "D50_10", "D50_01")


def same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-10)
    return actual == expected


def summary(values):
    return {"seed_order": list(SEEDS), "values": values, "mean": statistics.mean(values), "std": statistics.stdev(values),
            "mean_pp": statistics.mean(values)*100, "std_pp": statistics.stdev(values)*100,
            "wins": sum(v > 0 for v in values), "losses": sum(v < 0 for v in values), "ties": sum(v == 0 for v in values)}


def independent_features(features, epsilon):
    x = np.asarray(features, np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) <= epsilon).any():
        raise RuntimeError("STAGE15B_REVIEW_INVALID_FEATURES")
    x /= np.linalg.norm(x, axis=1)[:, None]
    distances = np.clip(1-x@x.T, 0, 2)
    mean_pair = float(distances[np.triu_indices(len(x), 1)].mean())
    np.fill_diagonal(distances, np.inf)
    eigenvalues = np.linalg.svd(x-x.mean(0), compute_uv=False)**2/(len(x)-1)
    total = float(eigenvalues.sum())
    prob = eigenvalues[eigenvalues > 0]/total if total > epsilon else np.array([])
    er = float(np.exp(-np.sum(prob*np.log(prob+epsilon)))) if len(prob) else 0.0
    return {"mean_pairwise_cosine_distance": mean_pair,
            "median_nearest_neighbor_distance": float(np.median(distances.min(1))),
            "covariance_effective_rank": er, "normalized_ER": er/x.shape[1]}


def inspect_training_csv(path, suppressed_validation_calls):
    with Path(path).open(encoding="utf8", newline="") as stream:
        header, *rows = list(csv.reader(stream))
    if header[:5] != ["epoch","time","train/box_loss","train/cls_loss","train/dfl_loss"]:
        raise RuntimeError("STAGE15B_REVIEW_CSV_HEADER_UNEXPECTED")
    if len(rows) != 150 or [int(r[0]) for r in rows] != list(range(1,151)):
        raise RuntimeError("STAGE15B_REVIEW_CSV_EPOCHS_INVALID")
    if not all(math.isfinite(float(v)) for r in rows for v in r):
        raise RuntimeError("STAGE15B_REVIEW_CSV_NONFINITE")
    ragged = [i+1 for i,r in enumerate(rows) if len(r) != len(header)]
    if ragged and not (ragged == [150] and len(rows[-1]) == 8 and len(header) == 15 and suppressed_validation_calls == 1):
        raise RuntimeError("STAGE15B_REVIEW_CSV_UNEXPLAINED_RAGGED_ROWS")
    return {"epochs":150,"header_columns":len(header),"last_row_columns":len(rows[-1]),
            "ragged_epochs":ragged,"training_loss_prefix_valid":True,
            "final_row_learning_rates_shifted_under_metric_headers":bool(ragged),
            "CSV_metric_columns_used_for_final_AP":False,
            "final_AP_source":"m50_s<seed>_raw_metrics.json; epoch150 last.pt independent validation"}


def review(root, cfg, stage14a, stage14b, stage15a, candidate_pool, bank_root):
    docs = {p.name: load(p) for p in root.glob("*.json") if p.name != "downloaded_results_review.json"}
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition: raise RuntimeError("STAGE15B_REVIEW_FAILED: "+name)
    check("download_matches_frozen_config", docs["stage15b_protocol.json"] == cfg)
    manifest = docs["balanced_morph50_manifest.json"]; freeze = docs["manifest_freeze_audit.json"]
    check("manifest_and_protocol_content_hashes", freeze["manifest_content_sha256"] == digest(manifest) and freeze["protocol_content_sha256"] == digest(cfg))
    pool = load(candidate_pool); index = {r["candidate_id"]: r for r in pool["candidates"]}
    subsets = load(stage14a/"subset_definition.json")["subsets"]
    morphology_doc = load(stage14b/"per_sample_morphology_distance.json")
    morphology = {r["candidate_id"]: r["morphology_distance"] for r in morphology_doc["records"]}
    check("exact_frozen_input_hashes", freeze["input_content_sha256"] == {"pool": digest(pool), "subsets": digest(subsets), "morphology": digest(morphology_doc)})
    selected, cap = select_balanced(subsets, index, morphology, cfg)
    check("deterministic_selection_reproduced", manifest["records"] == selected and cap == manifest["common_parent_cap"] == freeze["parent_cap"])
    check("unique_80_candidates_four_20_roles_shared_parent_cap", len({r["candidate_id"] for r in selected}) == 80 and Counter((r["class_name"], r["role"]) for r in selected) == {(c,r):20 for c in CLASSES for r in ("high_morph", "random")} and max(Counter(r["parent_id"] for r in selected).values()) <= cap)
    arms = docs["all_arm_candidate_records.json"]["arms"]
    prior = load(stage15a/"factorial_subset_definition.json")["arms"]
    check("five_arms_M50_matches", set(arms) == set(ARMS) and arms["M50"] == selected)
    check("detector_config_unchanged_from_Stage15A", cfg["detector"] == load(stage15a/"stage15a_protocol.json")["detector"])
    for arm in ARMS:
        rows = arms[arm]
        check(arm+"_unique_80_class_counts", len(rows) == len({r["candidate_id"] for r in rows}) == 80 and Counter(r["class_name"] for r in rows) == {"flash":40, "black":40})
        check(arm+"_unaltered_pool_metadata", all(r == candidate_record(index[r["candidate_id"]], r["role"], morphology[r["candidate_id"]]) for r in rows))
        if arm != "M50":
            check(arm+"_exact_historical_sample_order", [r["candidate_id"] for r in rows] == [r["candidate_id"] for r in prior[arm]])
        overlap = docs["membership_overlap_audit.json"]["actual_vs_nominal_high_dose"][arm]
        check(arm+"_actual_vs_nominal_membership", overlap == {"nominal_high_role_count":sum(r["role"] == "high_morph" for r in rows), "actual_frozen_high_membership_count":sum(r["candidate_id"] in subsets["high_fidelity"] for r in rows)})
        for group in ("overall", *CLASSES):
            group_rows = [r for r in rows if group == "overall" or r["class_name"] == group]
            a = np.array([r["morphology_distance"] for r in group_rows])
            expected = {"n":len(a), "mean":float(a.mean()), "median":float(np.median(a)), "std":float(a.std()), "std_ddof":0,
                        **{k:float(np.quantile(a,q)) for k,q in (("q25",.25),("q75",.75),("q90",.9))}}
            check(arm+"_"+group+"_morphology_stats", same(docs["morphology_distribution_audit.json"]["arms"][arm][group], expected))
            sh = Counter(str(r["generation_seed"]) for r in group_rows); ph = Counter(r["parent_id"] for r in group_rows)
            def entropy(h): return -sum(n/len(group_rows)*math.log(n/len(group_rows)) for n in h.values())
            expected_source = {"n":len(group_rows), "generation_seed_entropy":entropy(sh), "parent_source_entropy":entropy(ph), "entropy_base":"natural_log",
                               "unique_parent_count":len(ph), "max_parent_reuse":max(ph.values()), "generation_seed_histogram":dict(sh), "parent_histogram":dict(ph)}
            check(arm+"_"+group+"_source_stats", same(docs["source_diversity_audit.json"]["arms"][arm][group], expected_source))
    for cls_id, cls in enumerate(CLASSES):
        for role, key in (("high_morph","high_fidelity"),("random","random")):
            audit = docs["generation_seed_balance_audit.json"]["classes"][cls][role]
            available = [index[i] for i in subsets[key] if int(index[i]["class_id"]) == cls_id and (role != "random" or i not in subsets["high_fidelity"])]
            actual = {str(s):sum(r["class_name"] == cls and r["role"] == role and r["generation_seed"] == s for r in selected) for s in cfg["generation_seeds"]}
            check(cls+"_"+role+"_seed_balance", audit["selected"] == actual and audit["available_after_overlap_priority"] == {str(s):sum(int(r["seed"]) == s for r in available) for s in cfg["generation_seeds"]} and audit["all_counts_2_or_3"] == all(n in (2,3) for n in actual.values()))

    assets = docs["dataset_asset_audit.json"]
    real, built = assets["real_train"], assets["dataset_train_population"]
    check("dataset_record_hashes_counts", len(real) == 138 and len(built) == 218 and digest(real) == assets["real_train_content_sha256"] and digest(built) == assets["dataset_train_content_sha256"])
    built_index = {r["name"].split('.')[0]: r for r in built}
    for i, row in enumerate(sorted(real, key=lambda r: Path(r["name"]).stem)):
        check(f"real_link_bytes_{i}", all(built_index[f"base_{i:03d}"][k] == row[k] for k in ("image_sha256","label_sha256")))
    check("all_synthetic_asset_ids", [r["candidate_id"] for r in assets["synthetic_assets"]] == [r["candidate_id"] for r in selected])
    for i, row in enumerate(assets["synthetic_assets"]):
        check(f"synthetic_link_bytes_{i}", all(built_index[f"extra_{i:03d}"][k] == row[k] for k in ("image_sha256","label_sha256")))

    feature_doc = docs["feature_diversity_audit.json"]; aggregate = {arm:{c:{} for c in ("overall",*CLASSES)} for arm in ARMS}
    external_banks = {}
    for fold in cfg["features"]["folds"]:
        npz = root/f"synthetic_oof_features_fold{fold}.npz"; meta = docs[f"synthetic_oof_features_fold{fold}_audit.json"]; fp = meta["fingerprint"]
        check(f"fold{fold}_npz_byte_hash", sha(npz) == meta["npz_sha256"] == feature_doc["teachers"][fold]["features_sha256"])
        check(f"fold{fold}_manifest_input_hashes", fp["manifest"] == digest(manifest) and fp["all_arm_candidates"] == digest(arms))
        teacher = feature_doc["teachers"][fold]
        check(f"fold{fold}_recorded_teacher_fingerprint_consistency", fp["weights"] == teacher["weights_sha256"] and fp["metadata"] == teacher["bank_metadata_sha256"] and fp["imgsz"] == cfg["features"]["imgsz"])
        bank = bank_root/f"real_feature_bank_fold{fold}.npz"; bank_meta = bank_root/f"real_feature_bank_fold{fold}_metadata.json"
        external_banks[str(fold)] = {"original_real_bank_available_locally": bank.is_file() and bank_meta.is_file(), "recorded_bank_sha256":fp["bank"]}
        if bank.is_file() and bank_meta.is_file():
            check(f"fold{fold}_historical_bank_hashes", fp["bank"] == sha(bank) and fp["metadata"] == digest(load(bank_meta)))
        check(f"fold{fold}_synthetic_image_hash_consistency", all(fp["candidate_images"][r["candidate_id"]] == r["image_sha256"] for r in assets["synthetic_assets"]))
        with np.load(npz, allow_pickle=False) as data:
            ids = data["candidate_ids"].astype(str).tolist(); features = data["features"].copy()
            check(f"fold{fold}_exact_union_and_shape", ids == sorted({r["candidate_id"] for rows in arms.values() for r in rows}) and features.shape == (len(ids), feature_doc["teachers"][fold]["feature_dim"]) and int(data["fold"]) == fold)
        pos = {cid:i for i,cid in enumerate(ids)}
        for arm in ARMS:
            for group in ("overall",*CLASSES):
                rows = [r for r in arms[arm] if group == "overall" or r["class_name"] == group]
                stats = independent_features(features[[pos[r["candidate_id"]] for r in rows]], cfg["features"]["effective_rank_epsilon"])
                stored = feature_doc["arms_by_fold"][arm][str(fold)][group]
                check(f"fold{fold}_{arm}_{group}_feature_SVD_recomputation", same({k:stored[k] for k in stats},stats))
                for k,v in stats.items(): aggregate[arm][group].setdefault(k,[]).append(v)
    aggregate = {arm:{c:{k:statistics.mean(v) for k,v in m.items()} for c,m in groups.items()} for arm,groups in aggregate.items()}
    check("feature_fold_means", same(feature_doc["fold_mean"],aggregate))
    concentration = {c:{k:{"M11":aggregate["M11"][c][k], "mean_M10_M01_M50":statistics.mean(aggregate[a][c][k] for a in ("M10","M01","M50")), "M11_minus_comparators":aggregate["M11"][c][k]-statistics.mean(aggregate[a][c][k] for a in ("M10","M01","M50"))} for k in aggregate["M11"][c]} for c in ("overall",*CLASSES)}
    check("feature_concentration_deltas", same(feature_doc["M11_concentration_diagnostics"],concentration))

    records = docs["all_five_arms_metrics.json"]["records"]; by = {(r["seed"],r["arm"]):r for r in records}
    check("exact_15_arm_seed_metrics", len(records) == len(by) == 15 and set(by) == {(s,a) for s in SEEDS for a in ARMS})
    old_rows = load(stage15a/"factorial_all_arms_metrics.json")["records"]
    check("twelve_historical_metrics_byte_values_preserved", [r for r in records if r["arm"] != "M50"] == old_rows)
    initialization = docs["training_initialization_audit.json"]
    check("initialization_input_hashes", initialization["reused_metrics_content_sha256"] == digest(old_rows) and initialization["manifest_content_sha256"] == digest(manifest) and initialization["protocol_content_sha256"] == digest(cfg))
    check("three_new_metrics_export_match", docs["balanced_morph50_metrics.json"]["records"] == [by[s,"M50"] for s in SEEDS])
    budget = docs["training_budget_equivalence.json"]; environment = docs["training_environment_audit.json"]
    check("fixed_schedule_3_new_12_reused", all(budget[k] == v for k,v in {"base_real":138,"synthetic":80,"train_images":218,"epochs":150,"batch":1,"scheduled_draws":32700,"scheduled_batches":32700,"new_runs":3,"reused_runs":12,"runtime_audit_pending":False}.items()))
    csv_audits = {}
    for seed in SEEDS:
        row = docs[f"m50_s{seed}_raw_metrics.json"]; ledger = docs[f"epoch_exposure_m50_s{seed}.json"]; epochs = ledger["epochs"]
        check(f"s{seed}_raw_metric_values", row == by[seed,"M50"] and row["manifest_content_sha256"] == digest(manifest))
        check(f"s{seed}_raw_ultralytics_metrics", all(row[k] == row["raw_results_dict"][original] for k,original in (("precision","metrics/precision(B)"),("recall","metrics/recall(B)"),("map50","metrics/mAP50(B)"),("map50_95","metrics/mAP50-95(B)"))))
        check(f"s{seed}_aggregate_equals_class_mean", all(math.isclose(row[k],statistics.mean(row["per_class"][c][ck] for c in CLASSES),abs_tol=1e-12) for k,ck in (("map50_95","ap50_95"),("map50","ap50"),("recall","recall"))))
        check(f"s{seed}_150_sequential_epochs_218_draws_batches", [e["epoch"] for e in epochs] == list(range(1,151)) and all(e["batches"] == e["primary_image_draws"] == 218 for e in epochs))
        prev_s,prev_a = 0,0
        for epoch in epochs:
            succ,att = epoch["cumulative_successful_optimizer_steps"], epoch["cumulative_optimizer_step_attempts"]
            check(f"s{seed}_epoch{epoch['epoch']}_update_counters", 0 <= succ-prev_s <= att-prev_a)
            prev_s,prev_a = succ,att
        expected_budget = {"epochs":150,"primary_draws":32700,"batches":32700,"optimizer_step_attempts":prev_a,"successful_optimizer_steps":prev_s,"amp_guarded_skips":prev_a-prev_s}
        check(f"s{seed}_runtime_budget_counter_consistency", budget["M50_runtime"][str(seed)] == expected_budget and ledger["successful_optimizer_steps"] == prev_s and ledger["optimizer_step_attempts"] == prev_a)
        args_path = root/f"m50_s{seed}_args.yaml"; csv_path = root/f"m50_s{seed}_training_results.csv"; actual = yaml.safe_load(args_path.read_text(encoding="utf8"))
        env = next(r for r in environment["new_runs"] if r["seed"] == seed)
        check(f"s{seed}_raw_args_and_csv_hash", env["actual_args"] == actual and env["args_sha256"] == sha(args_path) and env["training_csv_sha256"] == sha(csv_path))
        expected_args = {k:cfg["detector"][k] for k in ("imgsz","epochs","patience","batch","optimizer","mosaic","mixup","copy_paste","close_mosaic","deterministic")}
        expected_args.update(seed=seed,val=False,rect=False,augment=False,model=cfg["detector"]["model"])
        check(f"s{seed}_actual_args_match_frozen_protocol", all(actual.get(k) == v for k,v in expected_args.items()))
        csv_audits[str(seed)] = inspect_training_csv(csv_path, ledger["suppressed_framework_validation_calls"])
        check(f"s{seed}_raw_CSV_150_epochs_training_prefix", csv_audits[str(seed)]["training_loss_prefix_valid"])
    usage = docs["validation_usage_audit.json"]
    check("final_last_only_validation_usage_metadata", usage["selection_uses"] == usage["threshold_tuning_uses"] == 0 and usage["validation_integrity_sha256"] == assets["official_val_integrity_sha256"] and len(usage["new_runs"]) == 3 and {r["seed"] for r in usage["new_runs"]} == set(SEEDS) and all(r["training_validation_model_forwards"] == 0 and r["final_last_evaluation_uses"] == 1 and r["best_pt_used"] is False for r in usage["new_runs"]))

    per = []
    for seed in SEEDS:
        outcomes = {}
        for outcome in OUTCOMES:
            values = {arm:by[seed,arm][outcome] if outcome in OUTCOMES[:3] else by[seed,arm]["per_class"][outcome.split('_')[0]]["ap50_95"] for arm in ARMS}
            check(f"s{seed}_{outcome}_finite_metrics", all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()))
            m00,m10,m01,m11,m50 = (values[a] for a in ARMS); single = (m10+m01)/2
            outcomes[outcome] = {"arm_values":values,"M40_single":single,"D50_00":m50-m00,"D_balance":m50-single,"D80_50":m11-m50,"D50_10":m50-m10,"D50_01":m50-m01}
        per.append({"seed":seed,"outcomes":outcomes})
    summaries = {o:{e:summary([r["outcomes"][o][e] for r in per]) for e in EFFECTS} for o in OUTCOMES}
    check("every_paired_delta_and_sample_std_recomputed", same(docs["dose_effects_by_seed.json"],{"primary_metric":"map50_95","effects":per}) and same(docs["dose_effects_summary.json"],{"std_ddof":1,"outcomes":summaries}))
    class_doc = docs["classwise_dose_effects.json"]
    for c in CLASSES:
        check(c+"_class_effects_recomputed", same(class_doc[c],summaries[c+"_ap50_95"]) and same(class_doc["all_arm_AP50_95"][c],{a:summary([by[s,a]["per_class"][c]["ap50_95"] for s in SEEDS]) for a in ARMS}))
    z = summaries["map50_95"]; t = cfg["gates"]
    gates = {"A_balanced_gain_ge_0_005":z["D50_00"]["mean"] >= t["balanced_gain_min"], "B_balanced_wins_ge_2":z["D50_00"]["wins"] >= t["balanced_gain_wins_min"],
             "C_same_dose_mean_ge_minus_0_005":z["D_balance"]["mean"] >= t["same_dose_min"], "D_excess_mean_le_minus_0_005":z["D80_50"]["mean"] <= t["excess_penalty_max"], "E_excess_losses_ge_2":z["D80_50"]["losses"] >= t["excess_penalty_losses_min"]}
    singles = {a:summary([by[s,a]["map50_95"]-by[s,"M00"]["map50_95"] for s in SEEDS]) for a in ("M10","M01")}
    balanced = list(gates.values())[0] and list(gates.values())[1]
    status = "MORPHOLOGY_DOSE_EFFECT_SUPPORTED" if all(gates.values()) else "CLASS_COMPOSITION_INTERACTION_SUPPORTED" if not balanced and all(r["mean"] > 0 and r["wins"] >= 2 for r in singles.values()) else "MORPHOLOGY_SIGNAL_PRESENT_DOSE_HYPOTHESIS_NOT_CONFIRMED" if balanced else "DOSE_VS_CLASS_INTERACTION_UNRESOLVED"
    state = docs["stage15b_status.json"]
    check("objective_status_and_five_gates", state["status"] == status and state["gates"] == gates and same(state["single_class_reference_effects"], singles))
    positive_classes = sum(summaries[c+"_ap50_95"]["D50_00"]["mean"] > 0 for c in CLASSES)
    check("classwise_status", state["classwise_status"] == class_doc["status"] == ("RESIDUAL_CLASS_DEPENDENCE_PRESENT" if positive_classes == 1 else "BOTH_CLASSES_IMPROVED" if positive_classes == 2 else "NO_POSITIVE_CLASS_MEAN"))
    check("no_unauthorized_next_stage", state["STAGE16_AUTO_START"] is False and state["STAGE16_MORPHOLOGY_DISTRIBUTION_CALIBRATED_GENERATOR"] == (status == "MORPHOLOGY_DOSE_EFFECT_SUPPORTED") and state["STAGE16_TASK_AWARE_GENERATION"] == (status == "CLASS_COMPOSITION_INTERACTION_SUPPORTED"))
    limitations = [
        "No model checkpoint or image bytes are re-executed by this CPU review. Server-recorded hashes are cross-checked against downloaded ledgers and NPZs; original training images/checkpoints and validation images are not independently rehashed locally.",
        "The freeze hashes prove internal content consistency, not an independently timestamped pre-training freeze. Zero selection/tuning validation usage is supported by code and recorded metadata, not an external execution trace.",
        "Historical Stage15A runtime optimizer counters and pretrained bytes remain unavailable. New M50 runs share 544 attempted updates but have 532/534/534 successful updates due to AMP skips. Exact successful-update equality is not claimed.",
        "All three raw training CSVs have 15 header columns but only 8 columns at epoch150: bypassed final framework validation returned no metric placeholders, so the three learning rates appear under metric headers. The first five training columns and epoch count remain valid. Final detector APs use separately exported last.pt validation JSON, never the ragged CSV metric columns. Original CSVs are preserved unchanged.",
        "Historical nominal HighMorph doses differ from actual High-list membership because Random/High lists overlap. M00/M10/M01/M11/M50 actual membership is 14/50/44/80/40. Parent reuse and generation-seed concentrations also differ; this is not a one-variable randomized causal dose experiment.",
        "M11 is narrower in overall pairwise feature distance and morphology std, but classwise feature diagnostics are not uniformly narrower. Concentration alone does not demonstrate a downstream penalty or choose a generator mechanism.",
    ]
    return {"status":"DOWNLOADED_RESULTS_REVIEW_PASS", "experiment_status":status, "checks":checks,
            "primary_effects":z,"class_effects":{c:summaries[c+"_ap50_95"]["D50_00"] for c in CLASSES},
            "gates":gates,"limitations":limitations,"raw_training_CSV_format_audit":csv_audits,
            "optional_original_bank_verification":external_banks,"training_or_evaluation_performed":False,
            "downloaded_json_byte_sha256":{name:sha(root/name) for name in docs},
            "reviewed_NPZ_byte_sha256":{f"fold{f}":sha(root/f"synthetic_oof_features_fold{f}.npz") for f in cfg["features"]["folds"]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","stage14a","stage14b","stage15a","candidate_pool","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--feature_bank_root",type=Path,default=Path("results/dwbg/v2/fixed_v3/feature_banks"))
    a = p.parse_args()
    result = review(a.output,load(a.protocol),a.stage14a,a.stage14b,a.stage15a,a.candidate_pool,a.feature_bank_root)
    save(a.output/"downloaded_results_review.json",result)
    report_path = a.output/"STAGE15B_REPORT.md"
    marker = "\n## Downloaded-result independent review\n"
    report = report_path.read_text(encoding="utf8").split(marker)[0].rstrip()
    section = [marker, "", f"CPU-only review: `{result['status']}`, {len(result['checks'])} checks passed. All 15 arm/seed metric records, 12 unchanged historical records, 3 raw final validation exports, 3 epoch ledgers/args/CSVs, frozen sample selection and all 3 feature NPZ caches were cross-checked. Paired effects/sample std were independently recomputed; feature rank was recomputed with SVD rather than the production Gram eigendecomposition.", "",
        "Registered gates A/B/E pass; C/D fail. M50−M00 is +0.674 ± 2.367 pp, but M50−mean(M10,M01) is −1.095 ± 2.873 pp. M11−M50 is +0.368 ± 3.527 pp: two negative seeds do not establish the required negative mean dose penalty. Flash improves by +2.076 pp on average, while Black decreases by −0.727 pp. Neither Stage16 generator direction is authorized.", "",
        "Successful updates are 532/534/534 despite identical 544 attempted updates. The review does not upgrade schedule equivalence into exact successful-compute equality.", "", "### Review limitations", ""]
    section += ["- "+item for item in result["limitations"]]
    section += ["", "Machine-readable checks, hashes and explicit per-seed CSV format diagnostics are retained in `downloaded_results_review.json`. No raw metrics, CSVs, thresholds, samples or historical results were changed.", ""]
    report_path.write_text(report+"\n".join(section),encoding="utf8")
    print(result["status"]+"; "+result["experiment_status"]+"; checks="+str(len(result["checks"])))


if __name__ == "__main__": main()
