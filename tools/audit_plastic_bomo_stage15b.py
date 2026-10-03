"""Training-free diversity diagnostics using the existing frozen DWBG OOF teachers."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from plastic_bomo_stage15b_common import (CLASSES, digest, feature_diversity, frozen_save, load, save, sha, verify_preparation)

FEATURE_METRICS = ("mean_pairwise_cosine_distance", "median_nearest_neighbor_distance", "covariance_effective_rank", "normalized_ER")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "prepared", "feature_bank_root", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--oof_integrity", type=Path, required=True)
    p.add_argument("--device", default="0")
    a = p.parse_args(); cfg = load(a.protocol); manifest = verify_preparation(a.prepared, cfg)
    arms = load(a.prepared/"all_arm_candidate_records.json")["arms"]
    rows = {x["candidate_id"]: x for records in arms.values() for x in records}; ids = sorted(rows)
    original = load(a.oof_integrity)
    if any(original[k] for k in ("missing_images", "duplicate_images_across_folds", "missing_instances")):
        raise RuntimeError("STAGE15B_HISTORICAL_OOF_INTEGRITY_FAILED")
    bank_manifest = load(a.feature_bank_root/"feature_bank_manifest.json")
    if bank_manifest["source"] != "real_training_only" or sorted(x["fold"] for x in bank_manifest["folds"]) != [0,1,2]:
        raise RuntimeError("STAGE15B_REAL_ONLY_TEACHERS_REQUIRED")
    from PIL import Image
    from dwbg_feature_extraction import DetectInputExtractor
    a.output.mkdir(parents=True, exist_ok=True); fold_audits = []; arm_metrics = {arm: {} for arm in arms}
    for fold in cfg["features"]["folds"]:
        bank_path = a.feature_bank_root/f"real_feature_bank_fold{fold}.npz"
        meta_path = a.feature_bank_root/f"real_feature_bank_fold{fold}_metadata.json"
        meta = load(meta_path); weights = Path(meta["weights"])
        if not weights.is_file(): raise FileNotFoundError("STAGE15B_EXISTING_OOF_CHECKPOINT_MISSING: "+str(weights))
        with np.load(bank_path, allow_pickle=False) as bank:
            dim = int(bank["features"].shape[1]); bank_count = len(bank["features"])
        fingerprint = {"manifest": digest(manifest), "all_arm_candidates": digest(arms), "weights": sha(weights),
            "bank": sha(bank_path), "metadata": digest(meta), "imgsz": cfg["features"]["imgsz"],
            "extractor_code": sha(Path(__file__).with_name("dwbg_feature_extraction.py")),
            "candidate_images": {cid: sha(rows[cid]["image_path"]) for cid in ids}}
        cache = a.output/f"synthetic_oof_features_fold{fold}.npz"; cache_meta = a.output/f"synthetic_oof_features_fold{fold}_audit.json"
        if cache.exists() != cache_meta.exists(): raise RuntimeError("STAGE15B_INCOMPLETE_FEATURE_CACHE: "+str(cache))
        if cache.exists():
            old = load(cache_meta)
            if old["fingerprint"] != fingerprint or old["npz_sha256"] != sha(cache): raise RuntimeError("STAGE15B_FEATURE_CACHE_CHANGED")
            with np.load(cache, allow_pickle=False) as data:
                x = data["features"].copy(); cached_ids = data["candidate_ids"].astype(str).tolist()
            if cached_ids != ids: raise RuntimeError("STAGE15B_FEATURE_CACHE_ID_MISMATCH")
        else:
            encoder = DetectInputExtractor(weights, a.device, cfg["features"]["imgsz"])
            try:
                vectors = []
                for cid in ids:
                    row = rows[cid]
                    with Image.open(row["image_path"]) as image: width, height = image.size
                    vectors.append(encoder.encode(row["image_path"], row["bbox_xyxy"], width, height))
                x = np.stack(vectors).astype(np.float32)
            finally:
                encoder.close()
            del encoder
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            np.savez_compressed(cache, features=x, candidate_ids=np.asarray(ids), fold=np.array(fold))
            save(cache_meta, {"fingerprint": fingerprint, "npz_sha256": sha(cache)})
        if x.shape != (len(ids), dim): raise RuntimeError("STAGE15B_HISTORICAL_FEATURE_DIM_MISMATCH")
        pos = {cid: i for i, cid in enumerate(ids)}
        for arm, records in arms.items():
            groups = {"overall": records, **{c: [r for r in records if r["class_name"] == c] for c in CLASSES}}
            arm_metrics[arm][str(fold)] = {group: feature_diversity(x[[pos[r["candidate_id"]] for r in selected]], cfg["features"]["effective_rank_epsilon"]) for group, selected in groups.items()}
        fold_audits.append({"fold": fold, "weights": str(weights), "weights_sha256": fingerprint["weights"],
            "bank_metadata_sha256": fingerprint["metadata"], "existing_real_bank_records": bank_count,
            "real_bank_record_split": "teacher-training real images, as recorded by historical metadata; not mislabeled as held-out",
            "features_npz": cache.name, "features_sha256": sha(cache), "feature_dim": dim})
        print(f"STAGE15B_FEATURE_AUDIT_FOLD_{fold}_COMPLETE", flush=True)
    aggregate = {arm: {c: {key: float(np.mean([fold[c][key] for fold in per_fold.values()])) for key in FEATURE_METRICS} for c in ("overall", *CLASSES)} for arm, per_fold in arm_metrics.items()}
    narrowing = {c: {key: {"M11": aggregate["M11"][c][key], "mean_M10_M01_M50": float(np.mean([aggregate[arm][c][key] for arm in ("M10", "M01", "M50")])),
        "M11_minus_comparators": float(aggregate["M11"][c][key]-np.mean([aggregate[arm][c][key] for arm in ("M10", "M01", "M50")]))} for key in FEATURE_METRICS} for c in ("overall", *CLASSES)}
    result = {"status": "COMPLETE_BEFORE_NEW_DETECTOR_TRAINING", "manifest_content_sha256": digest(manifest),
        "feature_protocol": cfg["features"], "teachers": fold_audits, "arms_by_fold": arm_metrics, "fold_mean": aggregate,
        "M11_concentration_diagnostics": narrowing, "selection_feedback": False, "weighted_quality_score": False,
        "official_validation_use_count": 0, "new_teacher_training_count": 0}
    frozen_save(a.output/"feature_diversity_audit.json", result)
    print("STAGE15B_PRETRAINING_FEATURE_AUDIT_COMPLETE")

if __name__ == "__main__": main()
