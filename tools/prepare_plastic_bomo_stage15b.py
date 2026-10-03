"""Freeze BalancedMorph50 before diagnostics/training; link unchanged images and labels."""
from __future__ import annotations
import argparse
import os
from collections import Counter
from pathlib import Path
from plastic_bomo_stage15b_common import (CLASSES, candidate_record, dataset_layout, digest, distribution,
    frozen_save, image_files, label_dir, load, parent, population, save, select_balanced, sha, source_diversity)

def link(source, destination):
    source = Path(source).resolve(); destination = Path(destination)
    if not source.is_file(): raise FileNotFoundError("STAGE15B_ASSET_MISSING: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.exists() or not os.path.samefile(source, destination):
            raise RuntimeError("STAGE15B_DATASET_CONFLICT: " + str(destination))
    else:
        try: os.symlink(source, destination)
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 1314: raise
            # Windows may deny symlink creation; a hardlink preserves exact bytes.
            os.link(source, destination)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "stage14a", "stage14b", "stage15a", "candidate_pool", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--real_data_yaml", type=Path); p.add_argument("--dataset_root", type=Path)
    p.add_argument("--manifest_only", action="store_true")
    a = p.parse_args(); cfg = load(a.protocol); out = a.output; out.mkdir(parents=True, exist_ok=True)
    if load(a.stage15a/"stage15a_status.json")["status"] != "CLASS_INTERACTION_DOMINATES_MORPHOLOGY_SIGNAL":
        raise RuntimeError("STAGE15B_SOURCE_STAGE_STATUS_CHANGED")
    if cfg["detector"] != load(a.stage15a/"stage15a_protocol.json")["detector"]:
        raise RuntimeError("STAGE15B_DETECTOR_PROTOCOL_CHANGED")
    raw = load(a.candidate_pool); candidates = raw.get("candidates", raw.get("selected", []))
    index = {x["candidate_id"]: x for x in candidates}
    if len(index) != len(candidates): raise RuntimeError("STAGE15B_POOL_DUPLICATE_IDS")
    subsets = load(a.stage14a/"subset_definition.json")["subsets"]
    morphology_doc = load(a.stage14b/"per_sample_morphology_distance.json")
    morphology = {x["candidate_id"]: x["morphology_distance"] for x in morphology_doc["records"]}
    required = set(subsets["random"]) | set(subsets["high_fidelity"])
    if not required <= index.keys() or not required <= morphology.keys(): raise RuntimeError("STAGE15B_FROZEN_INPUTS_INCOMPLETE")
    if any(int(index[i]["seed"]) not in cfg["generation_seeds"] for i in required): raise RuntimeError("STAGE15B_GENERATION_SEED_CHANGED")
    selected, cap = select_balanced(subsets, index, morphology, cfg)
    counts = Counter((x["class_name"], x["role"]) for x in selected)
    if counts != {(c, r): 20 for c in CLASSES for r in ("high_morph", "random")}:
        raise RuntimeError("STAGE15B_ROLE_QUOTA_INVALID")
    manifest = {"status": "FROZEN_BEFORE_NEW_DETECTOR_TRAINING", "arm": "M50", "records": selected,
                "common_parent_cap": cap, "selection_rule": cfg["selection"], "overlap_policy": cfg["overlap_policy"],
                "morphology_distance_used_for_selection": False, "validation_used_for_selection": False}
    frozen_save(out/"balanced_morph50_manifest.json", manifest)
    frozen_save(out/"stage15b_protocol.json", cfg)
    frozen_save(out/"manifest_freeze_audit.json", {"status": "PASS", "manifest_content_sha256": digest(manifest),
        "protocol_content_sha256": digest(cfg), "input_content_sha256": {"pool": digest(raw), "subsets": digest(subsets), "morphology": digest(morphology_doc)},
        "class_role_counts": {c: {r: counts[c,r] for r in ("high_morph", "random")} for c in CLASSES},
        "total": 80, "unique_candidate_ids": 80, "parent_cap": cap, "max_parent_reuse": max(Counter(x["parent_id"] for x in selected).values())})
    original_arms = load(a.stage15a/"factorial_subset_definition.json")["arms"]
    all_arms = {}
    for arm, old in original_arms.items():
        all_arms[arm] = []
        for x in old:
            row = index[x["candidate_id"]]
            if x["image_path"] != row["image_path"] or x["label_path"] != row["label_path"]: raise RuntimeError("STAGE15B_EXISTING_MANIFEST_CHANGED")
            role = "high_morph" if (arm in ("M11", "M10") and int(row["class_id"]) == 0) or (arm in ("M11", "M01") and int(row["class_id"]) == 1) else "random"
            all_arms[arm].append(candidate_record(row, role, morphology[row["candidate_id"]]))
    all_arms["M50"] = selected
    frozen_save(out/"all_arm_candidate_records.json", {"arms": all_arms})
    overlap = {"overlap_priority": cfg["overlap_policy"], "selected_within_arm_duplicate_count": 0, "classes": {}, "actual_vs_nominal_high_dose": {}}
    for cls in CLASSES:
        c = CLASSES.index(cls); high = {i for i in subsets["high_fidelity"] if int(index[i]["class_id"]) == c}; rnd = {i for i in subsets["random"] if int(index[i]["class_id"]) == c}
        overlap["classes"][cls] = {"frozen_high_random_overlap": len(high & rnd), "random_candidates_excluded_by_high_membership": sorted(high & rnd), "selected_role_overlap": 0}
    high_all = set(subsets["high_fidelity"])
    for arm, records in all_arms.items():
        overlap["actual_vs_nominal_high_dose"][arm] = {"nominal_high_role_count": sum(x["role"] == "high_morph" for x in records), "actual_frozen_high_membership_count": sum(x["candidate_id"] in high_all for x in records)}
    overlap["interpretation"] = "Nominal role dose is matched; frozen High/Random overlap means the historical arms do not have exact physical High-membership doses of 0/40/80. Report this limitation without changing any arm."
    frozen_save(out/"membership_overlap_audit.json", overlap)
    md = {}; sd = {}
    for arm, records in all_arms.items():
        groups = {"overall": records, **{c: [x for x in records if x["class_name"] == c] for c in CLASSES}}
        md[arm] = {c: distribution([x["morphology_distance"] for x in group]) for c, group in groups.items()}
        sd[arm] = {c: source_diversity(group) for c, group in groups.items()}
    frozen_save(out/"morphology_distribution_audit.json", {"status": "COMPLETE_BEFORE_TRAINING", "source": "Stage14B frozen per-sample morphology distance", "arms": md, "selection_feedback": False})
    frozen_save(out/"source_diversity_audit.json", {"status": "COMPLETE_BEFORE_TRAINING", "arms": sd, "selection_feedback": False,
        "class_role_seed_counts": {c: {r: {str(s): sum(x["class_name"] == c and x["role"] == r and x["generation_seed"] == s for x in selected) for s in cfg["generation_seeds"]} for r in ("high_morph", "random")} for c in CLASSES}})
    seed_balance = {}
    for c in CLASSES:
        cid = CLASSES.index(c); high_ids = set(subsets["high_fidelity"])
        seed_balance[c] = {}
        for role, key in (("high_morph", "high_fidelity"), ("random", "random")):
            eligible = [index[i] for i in subsets[key] if int(index[i]["class_id"]) == cid and (role != "random" or i not in high_ids)]
            available = {str(s): sum(int(x["seed"]) == s for x in eligible) for s in cfg["generation_seeds"]}
            actual = {str(s): sum(x["class_name"] == c and x["role"] == role and x["generation_seed"] == s for x in selected) for s in cfg["generation_seeds"]}
            seed_balance[c][role] = {"available_after_overlap_priority": available, "selected": actual,
                "all_counts_2_or_3": all(n in (2,3) for n in actual.values()),
                "fallback_rule": cfg["selection"], "parent_cap_is_shared_over_full_arm": cap}
    frozen_save(out/"generation_seed_balance_audit.json", {"status": "DETERMINISTIC_BALANCE_WITH_FROZEN_SUPPORT_AND_PARENT_CAP", "classes": seed_balance})
    if a.manifest_only:
        print("STAGE15B_MANIFEST_FROZEN; parent_cap="+str(cap)); return
    if a.real_data_yaml is None or a.dataset_root is None: p.error("dataset build requires --real_data_yaml and --dataset_root")
    images, val, _ = dataset_layout(a.real_data_yaml); real = image_files(images)
    if len(real) != 138: raise RuntimeError("STAGE15B_REAL_TRAIN_COUNT_MISMATCH")
    labels = label_dir(images); root = a.dataset_root/"M50"
    assets = []
    for i, x in enumerate(sorted(real, key=lambda p: p.stem)):
        # Match Stage15A's base/extra index layout so filename sorting does not
        # introduce a separate primary-loader ordering change.
        alias = f"base_{i:03d}"
        link(x, root/"images/train"/(alias+x.suffix)); link(labels/(x.stem+".txt"), root/"labels/train"/(alias+".txt"))
    for i, x in enumerate(selected):
        name = f"extra_{i:03d}"
        link(x["image_path"], root/"images/train"/(name+Path(x["image_path"]).suffix)); link(x["label_path"], root/"labels/train"/(name+".txt"))
        lines = [line.split() for line in Path(x["label_path"]).read_text().splitlines() if line.strip()]
        if len(lines) != 1 or len(lines[0]) != 5 or int(float(lines[0][0])) != x["class_id"]: raise RuntimeError("STAGE15B_SYNTHETIC_LABEL_INVALID")
        assets.append({"candidate_id": x["candidate_id"], "image_sha256": sha(x["image_path"]), "label_sha256": sha(x["label_path"])})
    if len(image_files(root/"images/train")) != 218 or len(list((root/"labels/train").glob("*.txt"))) != 218:
        raise RuntimeError("STAGE15B_DATASET_HAS_STALE_OR_MISSING_FILES")
    import yaml
    doc = {"path": str(root.resolve()), "train": "images/train", "val": str(val.resolve()), "nc": 2, "names": list(CLASSES)}
    yaml_path = root/"data.yaml"; content = yaml.safe_dump(doc, sort_keys=False)
    if yaml_path.exists() and yaml_path.read_text(encoding="utf-8") != content: raise RuntimeError("STAGE15B_DATA_YAML_CHANGED")
    yaml_path.write_text(content, encoding="utf-8")
    frozen_save(out/"dataset_asset_audit.json", {"status": "PASS", "synthetic_assets": assets, "real_train": population(images),
        "real_train_content_sha256": digest(population(images)), "data_yaml": str(yaml_path.resolve()),
        "dataset_train_population": population(root/"images/train"), "dataset_train_content_sha256": digest(population(root/"images/train")),
        "dataset_index_layout": "Stage15A base_000..base_137 then extra_000..extra_079",
        "official_val_path": str(val.resolve()), "official_val_integrity_sha256": digest(population(val)),
        "validation_pixel_selection_use": 0, "no_bbox_or_image_edits": True})
    save(out/"training_budget_equivalence.json", {"status": "PREPARED_SCHEDULE_EQUIVALENT", "base_real": 138, "synthetic": 80, "class_counts": {"flash": 40, "black": 40},
        "train_images": 218, "epochs": 150, "batch": 1, "scheduled_draws": 32700, "scheduled_batches": 32700,
        "new_runs": 3, "reused_runs": 12, "runtime_audit_pending": True})
    print("STAGE15B_DATASET_PREPARED")

if __name__ == "__main__": main()
