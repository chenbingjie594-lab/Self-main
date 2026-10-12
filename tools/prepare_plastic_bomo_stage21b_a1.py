"""Freeze A1 metadata without reading internal_dev content or modifying the entry audit."""
import argparse
from collections import Counter
from pathlib import Path
import sys

from stage21b_a1_common import *
from prepare_plastic_bomo_stage21b import verify_assets


def prepare(a):
    cfg = load(a.protocol); check_cfg(cfg)
    original = Path(a.original)
    require(sha(original / "stage21b_artifact_manifest.json") == ORIGINAL_LEDGER_SHA, "STAGE21B_ORIGINAL_AUDIT_CHANGED")
    old_manifest = load(original / "stage21b_artifact_manifest.json")
    for name, expected in old_manifest["sha256"].items():
        require(sha(safe_relative(original, name)) == expected, "STAGE21B_ORIGINAL_AUDIT_CHANGED: " + name)
    old_status = load(original / "stage21b_status.json")
    require(old_status["status"] == "STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP"
            and old_status["detector_training_count"] == old_status["optimizer_step_count"] == 0, "STAGE21B_ORIGINAL_AUDIT_CHANGED")
    base_protocol = REPO / "configs/plastic_bomo_stage21b.json"
    require(sha(base_protocol) == old_status["protocol_sha256"], "STAGE21B_ORIGINAL_PROTOCOL_CHANGED")
    base_cfg = load(base_protocol)
    evidence = []
    md = bound_metadata(a.stage20a, "frozen_artifact_manifest.json", STAGE20A_LEDGER,
        ("real_source_registry.json", "annotation_provenance_v2.json", "normal_source_registry.json", "source_group_registry.json"), evidence)
    bmd = bound_metadata(a.stage20b, "formal_result_artifact_manifest.json", STAGE20B_LEDGER,
        tuple(f"formal_bank_{b}_manifest.json" for b in BANKS) + ("formal_synthetic_validity.json",), evidence)
    pmd = bound_metadata(a.p0, "p0_frozen_artifact_manifest.json", P0_LEDGER,
        ("generator_train_allowlist.json", "stage20b_detector_protocol.json", "stage20b_detector_resolved_runtime_arguments.json",
         "stage20b_runtime_binding.json", "detector_initialization_binding.json"), evidence)
    rmd = bound_metadata(a.r0, "r0_artifact_manifest.json", R0_LEDGER_SHA,
        ("runtime_environment.json", "detector_model_load_audit.json"), evidence)
    require(bmd["formal_synthetic_validity.json"]["status"] == "PASS", "STAGE21B_SYNTHETIC_VALIDITY_CHANGED")
    registry = md["real_source_registry.json"]["records"]
    train, dev, _ = group_split([x for x in registry if x["asset_role"] == "detector" and x["split_v2"] == "train"], base_cfg)
    require((len(train), len(dev)) == (107,27), "STAGE21B_SPLIT_CHANGED")
    real = real_records(registry, md["annotation_provenance_v2.json"]["records"], train)
    split = load(original / "stage21b_split_manifest.json")
    require(real == split["records"] and split["image_counts"] == {"internal_train":109,"internal_dev":27}, "STAGE21B_SPLIT_CHANGED")
    old_allow = synthetic_audit({b: bmd[f"formal_bank_{b}_manifest.json"] for b in BANKS}, real, train, dev,
        md["normal_source_registry.json"]["records"], md["source_group_registry.json"]["records"])
    require(old_allow["records"] == load(original / "stage21b_synthetic_allowlist.json")["records"], "STAGE21B_ORIGINAL_ALLOWLIST_CHANGED")
    allow = amended_allowlist(old_allow, dev)
    gen = [x for x in pmd["generator_train_allowlist.json"]["inputs"] if x["source_group_id"] in dev]
    require(len(gen) == 33 and Counter(x["class"] for x in gen) == {"black":23,"flash":10}, "STAGE21B_GENERATOR_EXPOSURE_EVIDENCE_CHANGED")
    u, c, pairing, equality = make_schedules([r["role_id"] for r in real if r["internal_split"] == "internal_train"],
                                           [r["role_id"] for r in allow["records"]], base_cfg, cfg)
    output = Path(a.output)
    require(not output.exists() and output.resolve() != original.resolve(), "STAGE21B_OUTPUT_EXISTS_USE_NEW_DIRECTORY")
    output.mkdir(parents=True)
    verified_copy(original / "stage21b_split_manifest.json", output / "stage21b_split_manifest.json", sha(original / "stage21b_split_manifest.json"))
    save(output / "protocol_amendment_A1.json", {"version":"A1", "user_authorized": True, "protocol":cfg,
        "parent_commit":cfg["parent_commit"], "original_artifact_manifest_sha256": ORIGINAL_LEDGER_SHA,
        "original_split_manifest_sha256":sha(original / "stage21b_split_manifest.json"),
        "split_unchanged":True, "additional_exclusion_count":4, "reason":"KNOWN_SOURCE_GROUP_LEAKAGE_EXCLUSION",
        "new_generation":False, "old_audit_preserved":True})
    save(output / "stage21b_synthetic_allowlist_v2.json", allow)
    save(output / "stage21b_split_isolation_audit_v2.json", {"status":"KNOWN_SOURCE_GROUP_ISOLATION_PASS",
        "donor_dev_intersection":[], "conditioning_dev_intersection":[], "train_dev_group_intersection":[],
        "excluded_conditioning_slots":allow["conditioning_dev_excluded"], "split_unchanged":True,
        "physical_source_complete":False, "pipeline_independent_holdout_claim":False})
    save(output / "stage21b_generator_exposure_limitation.json", {"generator_internal_dev_historical_exposure":True,
        "generator_annotation_inputs":33, "per_class":{"black":23,"flash":10},
        "annotation_ids":[x["annotation_id"] for x in gen],
        "dev_population":"DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR",
        "PIPELINE_INDEPENDENT_HOLDOUT":False, "generator_retraining":0, "generation_count":0})
    for name, data in (("uniform_schedule_v2",u),("curriculum_schedule_v2",c)):
        save(output / (name + ".json"), data, compact=True)
    save(output / "schedule_pairing_audit_v2.json", pairing)
    save(output / "exposure_equality_audit_v2.json", equality)
    for name, data in (("stage20b_detector_protocol_snapshot",pmd["stage20b_detector_protocol.json"]),
                       ("stage20b_resolved_arguments_snapshot",pmd["stage20b_detector_resolved_runtime_arguments.json"]),
                       ("r0_initialization_snapshot",rmd["detector_model_load_audit.json"]),
                       ("r0_runtime_environment_snapshot",rmd["runtime_environment.json"]),
                       ("stage21b_base_protocol_snapshot",base_cfg), ("stage21b_a1_protocol_snapshot",cfg)):
        save(output / (name + ".json"),data)
    save(output / "source_evidence_index_A1.json", {"records":evidence, "official_metrics_read":0,
        "original_split_sha256":sha(original / "stage21b_split_manifest.json"),
        "code_sha256":{f"tools/{n}":sha(REPO / "tools" / n) for n in NEW_CODE + HELPERS},
        "config_sha256":sha(a.protocol)})
    # Crucially: real internal_dev labels/pixels are NOT hashed or decoded here.
    asset_audit = verify_assets(allow, a.stage20b, [r for r in real if r["internal_split"] == "internal_train"], a.real_root)
    asset_audit["internal_dev_content_read"] = 0
    save(output / "source_asset_hash_audit_A1.json",asset_audit)
    s = initial_status(); s["scope"] = a.scope
    save(output / "stage21b_status.json",s)
    from finalize_plastic_bomo_stage21b_a1 import write_report
    write_report(output)
    files = sorted(p.name for p in output.iterdir() if p.is_file() and p.name not in ("stage21b_status.json","STAGE21B_REPORT.md"))
    save(output / "amended_metadata_manifest.json", {"version":"A1", "sha256":{n:sha(output / n) for n in files},
        "runtime_NOT_RUN":True, "split_unchanged":True})
    print(canonical({"status":s["status"],"output":str(output),"synthetic_per_class":allow["eligible_per_class"],
        "total_draws":25500,"attempts":450,"original_split_unchanged":True,"internal_dev_content_read":0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",type=Path,default=Path("configs/plastic_bomo_stage21b_a1.json"))
    p.add_argument("--original",type=Path,default=Path("results_for_gpt/plastic_bomo_stage21b_timing_premise"))
    p.add_argument("--stage20a",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20a_v2_protocol"))
    p.add_argument("--stage20b",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"))
    p.add_argument("--p0",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"))
    p.add_argument("--r0",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254"))
    p.add_argument("--real_root",type=Path)
    p.add_argument("--scope",default="LOCAL_PREPARATION",choices=("LOCAL_PREPARATION","SERVER_PREPARATION"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try: prepare(a)
    except Exception as error:
        print(canonical({"status":"STAGE21B_A1_PREPARATION_FAILED","detail":str(error),"training_started":False}),file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__": sys.exit(main())
