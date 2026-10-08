"""Read-only, CPU-only existing-asset audit. Never imports generation/training code.

Declared metadata, observed files, and proven lineage are deliberately separate.
Discovering a checkpoint or legal labels NEVER upgrades isolation to confirmed.
Missing remote paths are unavailable in this scope, not proof of global absence.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "train", "test", "val",
             "validation", "eval", "ground_truth", "datasets", "node_modules"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def evidence(repo, relative):
    if not relative:
        return {"path": None, "available": False}
    p = repo / relative
    if p.is_file():
        return {"path": str(p), "available": True, "sha256": sha(p),
                "origin": "WORKTREE", "text": p.read_text(encoding="utf-8-sig", errors="replace")}
    # Deleted historical tools belong to the user; read Git blobs, never restore.
    r = subprocess.run(["git", "-C", str(repo), "show", "HEAD:" + relative],
                       capture_output=True, check=False)
    if r.returncode == 0:
        return {"path": relative, "available": True,
                "sha256": hashlib.sha256(r.stdout).hexdigest(),
                "origin": "HEAD_GIT_BLOB", "text": r.stdout.decode("utf-8", "replace")}
    return {"path": str(p), "available": False, "origin": "UNAVAILABLE"}


def bbox_valid(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in box):
        return False
    x, y, w, h = box
    return w > 0 and h > 0 and x-w/2 >= -1e-8 and y-h/2 >= -1e-8 and x+w/2 <= 1+1e-8 and y+h/2 <= 1+1e-8


def resolve_record_path(value, repo):
    if not value:
        return None
    # Never reinterpret Linux absolute paths as local relative paths on Windows.
    if os.name == "nt" and str(value).startswith("/"):
        return None
    p = Path(value)
    return p if p.is_absolute() else repo / p


def audit_annotations(records, repo):
    seen_i, seen_l = set(), set()
    rows, legal = [], Counter()
    for r in records:
        ip, lp = r.get("image_path"), r.get("label_path")
        image, label = resolve_record_path(ip, repo), resolve_record_path(lp, repo)
        errors = []
        duplicate = (bool(ip) and ip in seen_i) or (bool(lp) and lp in seen_l)
        if ip: seen_i.add(ip)
        if lp: seen_l.add(lp)
        if duplicate: errors.append("DUPLICATE_IMAGE_OR_LABEL_PATH")
        if not bbox_valid(r.get("bbox")): errors.append("INVALID_DECLARED_BBOX")
        cid = r.get("class_id")
        if type(cid) is not int or cid not in (0, 1): errors.append("INVALID_DECLARED_CLASS")
        cls = "flash" if cid == 0 else "black" if cid == 1 else "unknown"
        expected_name = {0: "01_Flash_point", 1: "02_Big_black_spots"}.get(cid)
        if r.get("class_name") not in (None, cls, expected_name): errors.append("CLASS_MISMATCH")
        image_exists = image is not None and image.is_file()
        label_exists = label is not None and label.is_file()
        if not image_exists: errors.append("IMAGE_UNAVAILABLE_IN_AUDIT_SCOPE")
        if not label_exists: errors.append("LABEL_UNAVAILABLE_IN_AUDIT_SCOPE")
        if image_exists:
            try:
                from PIL import Image
                with Image.open(image) as im:
                    if min(im.size) <= 0: errors.append("INVALID_IMAGE_DIMENSIONS")
                    im.verify()
            except ImportError:
                errors.append("IMAGE_DECODER_UNAVAILABLE")
            except Exception as e:
                errors.append("IMAGE_DECODE_FAILED:" + type(e).__name__)
        label_rows = []
        if label_exists:
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip(): continue
                try:
                    v = [float(x) for x in line.split()]
                    if len(v) != 5 or v[0] != cid or not bbox_valid(v[1:]):
                        errors.append("INVALID_BBOX_OR_CLASS_IN_LABEL")
                    label_rows.append(v)
                except ValueError:
                    errors.append("INVALID_LABEL_SYNTAX")
            if not label_rows: errors.append("EMPTY_LABEL")
            if len(label_rows) != 1: errors.append("EXPECTED_ONE_DECLARED_OBJECT")
            if len(label_rows) == 1 and len(label_rows[0]) == 5 and bbox_valid(r.get("bbox")):
                if any(abs(a-b) > 1e-6 for a, b in zip(label_rows[0][1:], r["bbox"])):
                    errors.append("MANIFEST_LABEL_BBOX_MISMATCH")
        if not errors: legal[cls] += 1
        rows.append({"candidate_id": r.get("candidate_id"), "image_path": ip,
                     "label_path": lp, "class": cls, "image_exists": image_exists,
                     "label_exists": label_exists, "image_sha256": sha(image) if image_exists else None,
                     "label_sha256": sha(label) if label_exists else None,
                     "legal": not errors, "errors": sorted(set(errors))})
    return {"status": "PASS" if rows and all(x["legal"] for x in rows) else "DOWNSTREAM_UTILITY_NOT_TESTABLE",
            "declared_record_count": len(rows), "verified_image_count": sum(x["image_exists"] for x in rows),
            "verified_label_count": sum(x["label_exists"] for x in rows),
            "legal_class_counts": {c: legal[c] for c in ("flash", "black")},
            "error_counts": dict(Counter(e for x in rows for e in x["errors"])), "records": rows,
            "geometry_validity_is_not_annotation_semantic_verification": True}


def inventory(root):
    root = Path(root)
    out = {"root": str(root), "exists": root.is_dir(), "files": [], "errors": [],
           "excluded_directories": sorted(SKIP_DIRS), "validation_access": 0}
    if not root.is_dir(): return out
    def onerror(e): out["errors"].append(str(e))
    for parent, dirs, files in os.walk(root, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and "deeppcb" not in d.lower())
        for name in sorted(files):
            p = Path(parent) / name
            if p.suffix.lower() not in IMAGE_EXT | WEIGHT_EXT | {".json", ".yaml", ".yml", ".sh", ".txt"}: continue
            # Inventory generated/output roots only; no real dataset roots accepted.
            row = {"path": str(p), "relative": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                   "kind": "weight" if p.suffix.lower() in WEIGHT_EXT else "image" if p.suffix.lower() in IMAGE_EXT else "metadata"}
            if row["kind"] != "image": row["sha256"] = sha(p)
            out["files"].append(row)
    out["counts"] = dict(Counter(x["kind"] for x in out["files"]))
    return out


def selector_classification(arm):
    arm = arm.lower()
    if arm in {"random", "real_random_matched", "m00"}: return "RANDOM_SYNTHETIC"
    if arm in {"va", "vb", "vc", "vanilla_a", "vanilla_b", "vanilla_c"}: return "RAW_SYNTHETIC"
    if arm in {"high", "low", "m11"} or any(x in arm for x in ("morph", "m10", "m01", "m50", "fidelity", "context", "novel", "dwbg", "guard")):
        return "SELECTED_SYNTHETIC"
    return "UNKNOWN_SELECTION"


def downstream(repo):
    names = ["bootstrap_guard_downstream/validated_per_seed.json",
             "plastic_bomo_stage14b_fidelity_confirmation/all_seed_metrics.json",
             "plastic_bomo_stage15a_class_conditional_morphology/factorial_all_arms_metrics.json",
             "plastic_bomo_stage15b_morphology_dose/all_five_arms_metrics.json"]
    files, records = [], []
    for name in names:
        p = repo / "results_for_gpt" / name
        if not p.is_file():
            files.append({"path": str(p), "exists": False}); continue
        data = load(p)
        files.append({"path": str(p), "exists": True, "sha256": sha(p)})
        rr = data.get("records", []) if isinstance(data, dict) else data
        if not isinstance(rr, list): continue
        for r in rr:
            if not isinstance(r, dict): continue
            arm = str(r.get("arm", "unknown")); classification = selector_classification(arm)
            control = arm.lower() in {"real_only", "real_repeat", "rr", "realrepeat"}
            records.append({"evidence_file": name, "arm": arm, "seed": r.get("seed"),
                            "classification": classification, "is_real_control": control,
                            "raw_record": r, "interpretation": "REAL_CONTROL" if control else "SUBSET_UTILITY_SIGNAL",
                            "generator_utility_confirmed": False,
                            "headroom_admissible": False,
                            "blocked_reason": None if control else "Pool upstream filtering/recomposition and isolation unresolved; random within this pool is not raw-generator headroom."})
    stage17 = repo / "results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability/detector_repair_20261007_080108/results"
    for name in ("realrepeat_metrics.json", "vanilla_A_metrics.json", "vanilla_B_metrics.json", "vanilla_C_metrics.json"):
        f = stage17 / name
        if not f.is_file(): continue
        r = load(f); control = r["arm"] == "RR"
        files.append({"path": str(f), "exists": True, "sha256": sha(f)})
        records.append({"evidence_file": str(f), "arm": r["arm"], "seed": r["seed"],
                        "classification": "UNKNOWN_SELECTION" if control else "RAW_SYNTHETIC",
                        "is_real_control": control, "raw_record": r,
                        "interpretation": "REAL_CONTROL" if control else "FORMAL_NEGATIVE_VANILLA_HEADROOM_EVIDENCE",
                        "generator_utility_confirmed": False, "headroom_admissible": not control,
                        "failed_generator_must_not_be_reselected": True})
    return {"files": files, "records": records, "new_detector_forwards": 0,
            "arm_alias_evidence": "configs/plastic_bomo_stage15a.json: M00=Random-F/Random-B, M11=High-F/High-B; Stage14B high/low are morphology subsets",
            "selected_results_do_not_unlock_generator_development": True,
            "unknown_selection_is_not_raw": True}


def build_report(status, registry, lineage):
    table = "\n".join(f"| {r['generator_id']} | {r['isolation_status']} | {r['synthetic_image_count']} | {r['verified_legal_class_counts']} |" for r in registry)
    return f"""# Stage18A — Existing Generator Base Viability Audit

## 1. Why stop the current Vanilla base?

Stage17A clean Vanilla A/B/C lost to RealRepeat by -3.4533/-5.4312/-6.3627 pp;
mean -5.0824 pp, wins 0/3. Flash/Black mean deltas were -8.4172/-1.7475 pp.
These are three generation banks with detector seed 42, not a three-detector-seed claim.
`VANILLA_STAGE17A = NOT_VIABLE_AS_GENERATOR_BASE`. No extra bank, seed, tuning or module is authorized.
The 1.4857 pp nuisance SD / 2.9715 pp threshold applies ONLY to this Vanilla protocol, not another generator.

## 2. Is an auditable alternative already available?

`{status['status']}` in scope `{status['scope']}`. No alternative has a complete train-only lineage chain in the inspected evidence.
This is not proof that missing server assets do not exist. Server inventory pending: {status['server_inventory_pending']}.
AnomalyDiffusion code and a Plastic_Bomo launcher exist, but execution, output and checkpoint linkage are unconfirmed.
DefectFill has a prepared INPUT conversion report, not proof of generated outputs. Its local step_200 log identifies concrete/crack, not Plastic_Bomo.
Its inference code allows fallback to test masks if train masks are absent; actual historical use is unknown.
AnomalyDiffusion native generation writes image/mask outputs, which do not establish a detector bbox manifest.
Historical SD checkpoint files alone cannot establish the frozen training split. MSDF-v3 checkpoint identity is unverified.
RDA/CARF/DHFG configurations and historical notes are evidence of experiments, not verified clean output banks.

| Family | Isolation | Manifest-declared synthetic count | Verified legal Flash/Black |
|---|---|---:|---|
{table}

## 3. Are there enough raw synthetic images and legal annotations?

The DWBG manifest declares {lineage['candidate_count']} records, Flash {lineage['class_counts']['flash']}, Black {lineage['class_counts']['black']}.
Declared bboxes are checked separately from actual label files. Missing remote files are not counted as verified legal samples.
Images, masks, input conversions, triplet previews and unrelated-dataset demonstrations are not interchangeable with a legal Plastic_Bomo detection bank.
No family has verified all requirements for a matched 138-real + 80-synthetic vs 138-real + 80-RealRepeat screen.
Annotation caveats remain: 88 Flash / 30 Black XML lineage; 25 supplemental Black images / 50 boxes are historical prelabels without resolved instance verification;
suspected omissions and Small/Big Black ambiguity remain unchanged. No labels or splits were edited.

## 4. Is the evidence selector independent?

DWBG is post-generation pool construction/scoring/selection here, not an additional UNet generator architecture.
Historical builder code performs residual extraction, quality/attribute rejection, train-box geometry resizing and background recomposition.
Support-based YOLO boxes are constructed after recomposition. The 761 records have matching composition/quality fields; this supports the derivation,
but exact execution/checkpoint lineage remains unresolved. They must NOT be called 761 untouched raw MSDF outputs.
Stage14 Random IDs are checked against this pool and the older Random manifest. That manifest explicitly samples `manifold_valid` only,
with parent/seed caps. Random-within-a-filtered-pool is not proof of selector-free raw utility.
Detector scores are present in the scored manifest; this alone does not prove detector-guided generation, nor absence of earlier rejection.
Actual validation involvement in historical pool construction is unresolved, not asserted zero.
HighMorph/M10/M01/BalancedMorph50 and BootstrapGuard results are subset/selection signals, not generator utility confirmation.

## 5. Authorize one Stage18B screen?

`STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED = false`. No candidate selected. Current evidence supports retiring the generator-first route
on this frozen dataset, subject to the stated unresolved remote-asset scope; it does not establish universal absence of an alternative generator.
The supplied server command only inventories existing assets and checks already-declared synthetic labels. It never generates or trains;
finding files cannot automatically pass isolation. A new evidence review is required before any authorization.

All activity counts (sampling, generator/detector training, optimizer steps, validation access, DeepPCB, BootstrapGuard changes) are zero.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=Path("configs/plastic_bomo_stage18a.json"))
    p.add_argument("--repo_root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=["LOCAL_EXISTING_EVIDENCE", "SERVER_EXISTING_EVIDENCE"], default="LOCAL_EXISTING_EVIDENCE")
    p.add_argument("--code_parent", type=Path)
    p.add_argument("--download_root", type=Path)
    p.add_argument("--asset_root", type=Path, action="append", default=[])
    p.add_argument("--input_conversion_report", type=Path,
                   help="Optional existing DefectFill INPUT conversion metadata, not generated images")
    p.add_argument("--reuse_inventory", type=Path,
                   help="Reuse a completed Stage18A inventory, with its original observed file hashes; no claim of fresh file verification")
    a = p.parse_args(); repo = a.repo_root.resolve(); cfg = load(repo / a.protocol)
    if a.output.exists():
        if not a.reuse_inventory or a.reuse_inventory.resolve().parent != a.output.resolve():
            raise RuntimeError("STAGE18A_OUTPUT_EXISTS_USE_NEW_DIRECTORY: " + str(a.output))
        previous = load(a.output / "stage18a_status.json")
        if previous["scope"] != a.scope or not (a.output / "existing_generator_registry.json").is_file():
            raise RuntimeError("STAGE18A_REUSE_SCOPE_MISMATCH")
    a.output.mkdir(parents=True, exist_ok=True); out = a.output
    roots = list(a.asset_root)
    cp = a.code_parent or repo.parent
    roots += [cp / n for n in cfg["external_repository_names"]]
    if a.download_root and a.download_root.is_dir():
        roots += [d / "Plastic_Bomo" for d in sorted(a.download_root.iterdir())
                  if d.is_dir() and d.name.startswith("baseline")]
    # Never accept the official dataset root as a discovery root.
    for r in roots:
        if r.name.lower() in SKIP_DIRS | {"dataset", "plastic_bomo", "real_only"}:
            raise RuntimeError("STAGE18A_ASSET_ROOT_MUST_NOT_BE_REAL_DATASET: " + str(r))
    if a.reuse_inventory:
        cached = load(a.reuse_inventory)
        if cached["scope"] != a.scope: raise RuntimeError("STAGE18A_REUSE_SCOPE_MISMATCH")
        inv = cached["roots"]
    else:
        inv = [inventory(r) for r in dict.fromkeys(roots)]
    save(out / "asset_inventory.json", {"scope": a.scope, "roots": inv,
         "coverage": "explicit roots only; skipped train/test/val/eval and DeepPCB", "global_absence_claim": False})
    pool_path = repo / cfg["pool_manifest"]; pool = load(pool_path); candidates = pool["candidates"]
    pool_by_id = {x["candidate_id"]: x for x in candidates}
    counts = Counter("flash" if x["class_id"] == 0 else "black" for x in candidates)
    ann = audit_annotations(candidates, repo)
    random = load(repo / cfg["random_manifest"])
    subsets = load(repo / cfg["stage14a_subsets"])
    rids = [x["candidate_id"] for x in random["selected"]]
    sids = subsets["subsets"]["random"]
    snapshots = []
    for f in ["tools/build_dwbg_candidates.py", "tools/build_dasr_detection.py",
              "tools/score_dwbg_candidates_v2.py", "tools/select_dwbg_candidates_v2.py",
              "configs/msdf.json", "experiments/baseline_split70_s42.json", "网页端交接记忆.md"]:
        e = evidence(repo, f)
        if e.get("text") is not None:
            fn = "evidence_" + Path(f).name + ".txt"
            (out / fn).write_text(e.pop("text"), encoding="utf-8")
            e["snapshot"] = fn
            e["snapshot_sha256"] = sha(out / fn)
        snapshots.append(e)
    lineage = {"status": "PROCESS_DERIVATION_SUPPORTED_EXACT_CHECKPOINT_AND_ISOLATION_UNRESOLVED",
        "candidate_count": len(candidates), "class_counts": dict(counts), "manifest_sha256": sha(pool_path),
        "pool_path": str(pool_path), "merged_from": pool.get("merged_from"),
        "generator_architecture_evidence": "configs/msdf.json enables MSDF on SD2; exact used checkpoint not linked by hashes",
        "expected_checkpoint_path": "model/msdf_v3_sd2_2000_s42", "DWBG_role": "POST_GENERATION_CONSTRUCTION_SCORING_SELECTION",
        "all_761_untouched_raw": False,
        "composition_record_count": sum("composition" in x for x in candidates),
        "source_metrics_record_count": sum("source_metrics" in x for x in candidates),
        "source_output_directories": sorted({str(x.get("source_image", "")).rsplit("/image/", 1)[0] for x in candidates}),
        "background_directories": sorted({str(x.get("background", "")).rsplit("/", 1)[0] for x in candidates}),
        "generation_seed_distribution": dict(Counter(str(x.get("seed")) for x in candidates)),
        "annotation_method": "residual support bbox after geometry recomposition; historical builder code and manifest fields",
        "code_evidence": snapshots,
        "stage14_random": {"pool_id_count": len(sids), "all_ids_in_pool": all(x in pool_by_id for x in sids),
           "exact_order_equals_historical_random_manifest": sids == rids,
           "exact_set_equals_historical_random_manifest": set(sids) == set(rids),
           "historical_random_selection": random.get("selection"),
           "max_per_parent_source": random.get("max_per_parent_source"), "max_per_seed": random.get("max_per_seed")},
        "validation_use_in_historical_construction": "UNRESOLVED",
        "detector_feedback": "detector scoring is recorded; feedback into original diffusion sampling NOT PROVEN",
        "bootstrap_guard_or_DQ_preselection": "UNRESOLVED; do not conflate later selected arms with the entire pool",
        "quality_geometry_preprocessing": "historical builder explicitly filters and recomposes before scoring"}
    save(out / "msdf_dwbg_pool_lineage_audit.json", lineage)
    historical = downstream(repo); save(out / "historical_downstream_evidence_audit.json", historical)
    registry, isolation, annotations, selectors = [], {}, {}, {}
    for family in cfg["historical_families"]:
        gid = family["id"]; external = cp / family["external"] if family.get("external") else repo
        code = evidence(external, family["code"])
        if code.get("text") is not None:
            snap = "evidence_" + gid + "_code.txt"
            (out / snap).write_text(code.pop("text"), encoding="utf-8")
            code["snapshot"] = snap
            code["snapshot_sha256"] = sha(out / snap)
        native_code = None
        if family.get("external"):
            native_file = "generate_with_mask.py" if gid == "ANOMALYDIFFUSION" else "inference.py"
            native_code = evidence(external, native_file)
            if native_code.get("text") is not None:
                snap = "evidence_" + gid + "_inference.txt"
                (out / snap).write_text(native_code.pop("text"), encoding="utf-8")
                native_code["snapshot"] = snap
                native_code["snapshot_sha256"] = sha(out / snap)
        config = evidence(repo, family.get("config")); config.pop("text", None)
        checkpoint = repo / family["checkpoint"] if family.get("checkpoint") else None
        checkpoint_files = []
        if checkpoint and checkpoint.is_dir():
            checkpoint_files = [{"path": str(x), "sha256": sha(x)} for x in sorted(checkpoint.rglob("*"))
                                if x.is_file() and x.suffix.lower() in WEIGHT_EXT]
        relevant = [x for x in inv if Path(x["root"]).is_relative_to(external)] if family.get("external") else []
        if gid == "HISTORICAL_SD_BASELINE": relevant = [x for x in inv if "baseline" in x["root"].lower()]
        has_images = any(x.get("counts", {}).get("image", 0) for x in relevant)
        annotation = ann if gid == "MSDF_V3_DWBG_POOL" else {"status": "IMAGE_ONLY_ASSET" if has_images else "DOWNSTREAM_UTILITY_NOT_TESTABLE",
            "declared_record_count": None, "verified_image_count": None, "verified_label_count": 0,
            "legal_class_counts": {"flash": 0, "black": 0}, "reason": "No Plastic_Bomo output-to-bbox manifest bound to this family was verified."}
        annotations[gid] = annotation
        isolation[gid] = {"status": "ISOLATION_UNRESOLVED",
            "generator_training_frozen_train_only": "UNRESOLVED", "checkpoint_validation_test_contact": "UNRESOLVED",
            "normal_background_train_only": "UNRESOLVED", "synthetic_target_train_only": "UNRESOLVED",
            "annotation_validation_independent": "UNRESOLVED",
            "reason": "Configuration/intended train paths do not prove actual checkpoint or source-parent lineage.",
            "do_not_transfer_stage17a_contamination_to_unlinked_checkpoints": True}
        selectors[gid] = {"raw_random_decoupled_confirmed": False, "bootstrap_guard_independent_confirmed": False,
            "validation_independent_confirmed": False, "status": "UNRESOLVED"}
        if gid == "MSDF_V3_DWBG_POOL":
            selectors[gid]["status"] = "FILTERED_RECOMPOSED_POOL_NOT_UNTOUCHED_RAW"
        row = {"generator_id": gid, "generator_name": family["name"], "method_family": family["family"],
            "code_path": code, "config_path": config, "checkpoint_path": str(checkpoint) if checkpoint else None,
            "native_generation_code": native_code,
            "checkpoint_exists": bool(checkpoint_files), "checkpoint_sha256_if_available": checkpoint_files,
            "other_observed_asset_roots": relevant, "training_data_source": None, "training_split": "UNRESOLVED",
            "observed_assets_are_not_proven_family_checkpoint_or_output_bindings": True,
            "training_image_count": None, "validation_used_for_generator_training": "UNRESOLVED",
            "test_used_for_generator_training": "UNRESOLVED", "generation_output_dirs": [family["output"]] if family.get("output") else [],
            "synthetic_image_count": len(candidates) if gid == "MSDF_V3_DWBG_POOL" else None,
            "synthetic_annotation_count": len(candidates) if gid == "MSDF_V3_DWBG_POOL" else None,
            "counts_are_manifest_declared_not_file_verified": True,
            "class_counts": dict(counts) if gid == "MSDF_V3_DWBG_POOL" else {"flash": None, "black": None},
            "generation_seed_information": lineage["generation_seed_distribution"] if gid == "MSDF_V3_DWBG_POOL" else "UNRESOLVED",
            "parent/source_information_available": "source output path but no original training parent identity" if gid == "MSDF_V3_DWBG_POOL" else False,
            "conditioning_information_available": gid == "MSDF_V3_DWBG_POOL", "mask_information_available": "UNRESOLVED",
            "annotation_generation_method": lineage["annotation_method"] if gid == "MSDF_V3_DWBG_POOL" else "UNRESOLVED",
            "downstream_detector_results_if_any": [x["path"] for x in historical["files"] if x["exists"]] if gid == "MSDF_V3_DWBG_POOL" else [],
            "selector_used_in_historical_result": True if gid == "MSDF_V3_DWBG_POOL" else "UNRESOLVED",
            "selector_name_if_any": ["quality_gate", "manifold_valid", "DWBG", "morphology/context subsets", "BootstrapGuard selected arms"] if gid == "MSDF_V3_DWBG_POOL" else [],
            "evidence_paths": [code, config], "isolation_status": "ISOLATION_UNRESOLVED",
            "verified_legal_class_counts": annotation["legal_class_counts"], "eligible": False,
            "eligibility_failures": ["TRAIN_ONLY_NOT_CONFIRMED", "SELECTOR_INDEPENDENCE_NOT_CONFIRMED", "MATCHED_PROTOCOL_NOT_ESTABLISHED"]}
        if min(annotation["legal_class_counts"].values()) < 40: row["eligibility_failures"].append("LESS_THAN_40_VERIFIED_LEGAL_SAMPLES_PER_CLASS")
        registry.append(row)
    stage17 = load(repo / cfg["stage17a_results"] / "stage17a_status.json")
    save(out / "stage17a_stop_evidence.json", {"authoritative_commit": cfg["authoritative_stage17a_commit"],
         "source_status": stage17, "source_sha256": sha(repo / cfg["stage17a_results"] / "stage17a_status.json"),
         "VANILLA_STAGE17A": "NOT_VIABLE_AS_GENERATOR_BASE", "nuisance_threshold_scope": "STAGE17A_VANILLA_ONLY"})
    save(out / "existing_generator_registry.json", {"scope": a.scope, "families": registry,
         "Stage17A_Vanilla": "NOT_VIABLE_AS_GENERATOR_BASE", "counts_unknown_are_null_not_zero": True})
    save(out / "generator_isolation_audit.json", isolation)
    save(out / "generator_annotation_audit.json", annotations)
    save(out / "generator_selector_dependency_audit.json", selectors)
    for gid, fn in [("ANOMALYDIFFUSION", "anomalydiffusion_asset_audit.json"), ("DEFECTFILL", "defectfill_asset_audit.json")]:
        row = next(x for x in registry if x["generator_id"] == gid)
        example = cp / "DefectFill-main" / "step_200" / "inference_log.json"
        example_info = {"path": str(example), "exists": example.is_file()}
        if example.is_file():
            log = load(example)
            example_info.update({"sha256": sha(example), "object_class": log.get("object_class"),
                                 "defect_type": log.get("defect_type"), "checkpoint": log.get("checkpoint"),
                                 "generated_records": len(log.get("results", []))})
        conversion = None
        if a.input_conversion_report and a.input_conversion_report.is_file():
            conversion = {"path": str(a.input_conversion_report), "sha256": sha(a.input_conversion_report),
                          "metadata": load(a.input_conversion_report), "kind": "PREPARED_INPUT_NOT_SYNTHETIC"}
        save(out / fn, {"registry": row, "visual_quality_is_not_utility": True,
            "prepared_input_dataset_is_not_synthetic": True, "unrelated_dataset_outputs_are_not_Plastic_Bomo": True,
            "example_log": example_info if gid == "DEFECTFILL" else None,
            "input_conversion": conversion if gid == "DEFECTFILL" else None,
            "native_code_diagnostics": "train-mask fallback to test-mask is allowed; actual use unresolved" if gid == "DEFECTFILL" else "image/mask/ori/recon outputs, not a proven YOLO bbox manifest"})
    overlap = repo / "results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability/local_exact_overlap_audit/exact_overlap_status.json"
    if overlap.is_file():
        isolation["HISTORICAL_STAGE17A_REBUILD"] = {"status": "KNOWN_VALIDATION_CONTAMINATION",
             "generator_id": "baseline_stage17a_rebuild_split70_s42_noise_0.0",
             "evidence": load(overlap), "evidence_sha256": sha(overlap),
             "current_clean_Stage17A_is_a_different_checkpoint": True,
             "cannot_transfer_to_other_unlinked_historical_checkpoints": True}
        save(out / "generator_isolation_audit.json", isolation)
    save(out / "stage18b_candidate_selection.json", {"eligible_candidates": [], "selected_generator": None,
         "ranking": cfg["ranking"], "quality_and_map_used_for_ranking": False,
         "STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED": False, "auto_start": False})
    save(out / "stage18a_protocol.json", cfg)
    status = {"status": "NO_AUDITABLE_ALTERNATIVE_GENERATOR_BASE", "scope": a.scope,
        "qualification": "No candidate supported by inspected evidence; not an exhaustive claim of remote nonexistence.",
        "server_inventory_pending": a.scope != "SERVER_EXISTING_EVIDENCE",
        "asset_inventory_is_explicit_roots_not_exhaustive_server_search": True,
        "global_asset_absence_proven": False, "full_asset_isolation_reconstruction_complete": False,
        "VANILLA_STAGE17A": "NOT_VIABLE_AS_GENERATOR_BASE",
        "GENERATOR_FIRST_INNOVATION_ROUTE_CURRENT_DATASET": "RETIRED",
        "route_status_basis": "current auditable evidence; unresolved assets cannot authorize development",
        "STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED": False, "stage18b_auto_start": False,
        "sampling_count": 0, "generator_training_count": 0, "detector_training_count": 0,
        "optimizer_step_count": 0, "official_validation_use_count": 0,
        "deep_pcb_training": 0, "deep_pcb_generation": 0, "bootstrap_guard_modification": 0,
        "labels_modified": False, "splits_modified": False, "historical_results_modified": False}
    save(out / "stage18a_status.json", status)
    save(out / "execution_audit.json", {"utc": datetime.now(timezone.utc).isoformat(),
         "script_sha256": sha(Path(__file__)), "protocol_sha256": sha(repo / a.protocol),
         "inventory_reused": bool(a.reuse_inventory),
         "inventory_reuse_limitation": "Cached file hashes retain their original observation time, not freshly reverified" if a.reuse_inventory else None,
         "model_loading_count": 0, "validation_files_read": 0,
         "evidence_inputs": [{"path": str(x), "sha256": sha(x)} for x in
             [pool_path, repo / cfg["random_manifest"], repo / cfg["stage14a_subsets"]]]})
    (out / "STAGE18A_REPORT.md").write_text(build_report(status, registry, lineage), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
