"""Read-only audit of Stage24A text artifacts; no ML imports or asset payloads."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


REQUIRED_FILES = (
    "STAGE24A_REPORT.md", "literature_search_scope.json",
    "primary_source_registry.json", "method_comparison_matrix.json",
    "weak_annotation_gap.json", "small_low_contrast_gap.json",
    "background_preservation_gap.json", "few_shot_adaptation_gap.json",
    "scale_structure_gap.json", "annotation_reliability_gap.json",
    "stage20b_baseline_claim_audit.json", "candidate_gap_ranking.json",
    "innovation1_primary_candidate.json", "stage24a_status.json",
)
LEVELS = {"STRONG_GAP", "PLAUSIBLE_GAP", "WEAK_OR_ALREADY_COVERED", "NOT_VERIFIED"}
MATRIX_FIELDS = (
    "paper_id", "method", "year", "venue_status", "primary_source_url",
    "generator_family", "few_shot_setting", "required_real_anomaly_count",
    "normal_images_required", "annotation_requirement",
    "local_or_full_image_generation", "small_defect_handling",
    "low_contrast_handling", "background_preservation", "location_control",
    "shape_control", "appearance_control", "cross_seed_robustness",
    "diversity_mechanism", "sample_selector_filter",
    "downstream_detector_evaluation", "public_code", "public_dataset",
    "evidence_anchors",
)
CANDIDATE_FIELDS = (
    "research_problem", "why_it_matters", "closest_prior_art",
    "what_prior_art_does", "what_remains_unsolved", "why_Plastic_Bomo_can_test_it",
    "minimum_experiment_needed", "possible_external_dataset",
    "overlap_with_BootstrapGuard", "novelty_risk", "qualification",
)
ACTIVITY_FIELDS = (
    "training_count", "generator_training_count", "generation_count",
    "synthetic_generation_count", "sampling_count", "detector_training_count",
    "detector_forward_count", "backward_count", "optimizer_step_count",
    "new_module_implementation_count", "residual_computation_count",
    "perceptual_metric_forward_count", "official_final_eval_forward",
    "official_final_eval_image_or_label_content_read", "BootstrapGuard", "DeepPCB",
)


def read_json(path):
    def reject_constant(value):
        raise ValueError("Non-JSON constant: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    errors = []
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)

    check("required_files", all((root / name).is_file() for name in REQUIRED_FILES))
    docs = {p.name: read_json(p) for p in root.glob("*.json")}
    registry = docs["primary_source_registry.json"]["reviewed_papers"]
    matrix = docs["method_comparison_matrix.json"]["methods"]
    ids = {row["paper_id"] for row in registry}
    check("twenty_unique_sources", len(registry) == len(ids) == 20)
    check("matrix_registry_one_to_one", len(matrix) == 20 and {x["paper_id"] for x in matrix} == ids)
    check("matrix_required_fields", all(set(MATRIX_FIELDS) <= set(row) for row in matrix))
    check("annotation_roles_separate", all(
        {"training", "inference", "assessment_or_synthetic_label"} <= set(row["annotation_requirement"])
        for row in matrix
    ))
    check("mandatory_methods", {"AnomalyDiffusion", "DefectFill", "GAA", "UniScale"}
          <= {row["method"] for row in matrix})
    check("primary_sources_only", all(
        row["full_text_url"].startswith(("https://arxiv.org/", "https://openaccess.thecvf.com/",
                                         "https://www.ecva.net/", "https://ojs.aaai.org/"))
        and row["review_depth"] == "METHODS_AND_EXPERIMENTS_TARGETED_REVIEW"
        for row in registry
    ))
    gap_names = ("weak_annotation_gap.json", "small_low_contrast_gap.json",
                 "background_preservation_gap.json", "few_shot_adaptation_gap.json",
                 "scale_structure_gap.json", "annotation_reliability_gap.json")
    check("six_gap_families", {docs[n]["family"] for n in gap_names} == set("ABCDEF"))
    check("gap_levels_allowed", all(
        item in LEVELS for n in gap_names for key, item in walk(docs[n]) if key == "level"
    ))
    ranking = docs["candidate_gap_ranking.json"]
    candidates = ranking["candidates"]
    check("at_most_three_candidates", len(candidates) <= 3)
    check("candidate_fields", all(set(CANDIDATE_FIELDS) <= set(c) for c in candidates))
    check("candidate_levels", all(c["level"] in LEVELS for c in candidates))
    check("candidate_priors_resolved", all(
        len(c["closest_prior_art"]) >= 2 and set(c["closest_prior_art"]) <= ids for c in candidates
    ))
    referenced = set()
    for name in REQUIRED_FILES:
        if name.endswith(".json"):
            referenced.update(re.findall(r'"(P\d{2})"', (root / name).read_text(encoding="utf-8")))
    check("all_paper_ids_resolved", referenced <= ids)
    status = docs["stage24a_status.json"]
    primary = docs["innovation1_primary_candidate.json"]
    check("decision_consistent", status["status"] == ranking["status"] == primary["status"])
    check("negative_no_primary", status["status"] == "NO_CLEAR_INNOVATION1_GAP_IDENTIFIED"
          and status["primary_candidate"] is None and primary["primary_candidate"] is None
          and ranking["primary_qualified_count"] == 0
          and not any(c["primary_qualified"] for c in candidates))
    check("all_activity_zero", all(status[k] == 0 for k in ACTIVITY_FIELDS))
    check("all_authorizations_closed", all(
        item is False for name in REQUIRED_FILES if name.endswith(".json")
        for key, item in walk(docs[name])
        if key.upper().endswith("AUTHORIZED") or key == "authorization"
    ))
    check("no_payload_or_history_mutation", status["image_or_weight_payloads_opened"] is False
          and status["historical_results_modified"] is False
          and status["models_images_labels_splits_modified"] is False
          and status["stopped_routes_reopened"] is False)
    baseline = docs["stage20b_baseline_claim_audit.json"]
    source_results = []
    for row in baseline["sources"]:
        path = row["path"]
        allowed = path.endswith(".py") or path.endswith(".json")
        if not allowed:
            raise ValueError("Non-text source path rejected: " + path)
        disk_hash = sha((repo / path).read_bytes())
        git_hash = sha(subprocess.check_output(
            ["git", "-C", str(repo), "show", baseline["base_commit"] + ":" + path]
        ))
        source_results.append({"id": row["id"], "path": path,
                               "working_hash_matches": disk_hash == row["sha256"],
                               "base_commit_hash_matches": git_hash == row["sha256"]})
    check("twenty_pinned_source_metadata_hashes", len(source_results) == 20 and all(
        r["working_hash_matches"] and r["base_commit_hash_matches"] for r in source_results
    ))
    check("historical_ledger_unchanged", sha((repo / baseline["prior_ledger_path"]).read_bytes())
          == baseline["prior_ledger_sha256"])
    manifest = docs.get("artifact_manifest.json")
    if manifest:
        check("artifact_hashes", all(
            Path(r["path"]).name == r["path"] and sha((root / r["path"]).read_bytes()) == r["sha256"]
            for r in manifest["files"]
        ))
    result = {"status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors,
              "reviewed_methods": len(matrix), "qualified_primary_candidates": 0,
              "verified_source_metadata_files": len(source_results), "source_results": source_results,
              "verification_scope": "TEXT_SCHEMA_AND_GIT_HASH_ONLY_NOT_LITERATURE_REPLICATION",
              "model_forward_count": 0, "optimizer_step_count": 0, "files_written_by_verifier": 0}
    print(json.dumps(result, indent=2, allow_nan=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
