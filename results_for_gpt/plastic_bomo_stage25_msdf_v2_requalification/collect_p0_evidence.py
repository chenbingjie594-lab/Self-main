"""Read-only Stage25 P0 evidence collector, NOT a training/generation runner.

No imports of torch/diffusers/ultralytics. Hash only explicit V2 TRAIN paths,
frozen historical comparator files and public weights; never scan final_eval.
Writes JSON to stdout, not to historical directories.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_name(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def canonical(value):
    # Same canonicalization as frozen Stage20B, including ensure_ascii=False.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def audit(args):
    repo = args.repo.resolve()
    stage20a = repo / "results_for_gpt/plastic_bomo_stage20a_v2_protocol"
    local_p0 = repo / "results_for_gpt/plastic_bomo_stage20b_p0_preflight"
    server_p0 = local_p0 / "server_preflight_20261008_084115"
    formal = repo / "results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"
    cache = {}

    def bound(path, expected):
        path = Path(path).resolve()
        if path not in cache:
            cache[path] = sha(path) if path.is_file() else None
        return {"path": str(path), "expected_sha256": expected,
                "actual_sha256": cache[path], "match": cache[path] == expected}

    source_paths = (
        "diffusers/pipelines/stable_diffusion/msdf_guidance.py", "train_rda.py",
        "run_train_msdf.py", "inference.py",
        "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_magic.py",
        "configs/msdf.json", "tools/build_dwbg_candidates.py",
        "tools/score_dwbg_candidates_v2.py", "tools/select_dwbg_candidates_v2.py",
        "tools/dwbg_v2_utils.py", "tools/train_plastic_bomo_stage20b_generator.py",
        "tools/generate_plastic_bomo_stage20b_banks.py",
        "tools/train_plastic_bomo_stage20b_detector.py", "tools/stage20b_detector_controller.py",
        "tools/stage20b_formal_common.py", "tools/preflight_plastic_bomo_stage20b_p0.py",
        "diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py",
        "diffusers/schedulers/scheduling_ddim.py",
    )
    sources = []
    for name in source_paths:
        raw = subprocess.check_output(["git", "-C", str(repo), "show", args.base_commit + ":" + name])
        expected = hashlib.sha256(raw).hexdigest()
        current = repo / name
        sources.append({"path": name, "base_commit": args.base_commit, "sha256": expected,
                        "worktree_exists": current.is_file(),
                        "worktree_matches_if_present": sha(current) == expected if current.is_file() else None,
                        "read_from_git_if_worktree_missing": not current.is_file()})
    manifests = {}
    for label, root, name in (("stage20a", stage20a, "frozen_artifact_manifest.json"),
                               ("local_p0", local_p0, "p0_frozen_artifact_manifest.json"),
                               ("server_p0", server_p0, "p0_frozen_artifact_manifest.json"),
                               ("stage20b", formal, "formal_result_artifact_manifest.json")):
        path = root / name
        rows = [bound(root / relative, expected) for relative, expected in load(path)["sha256"].items()]
        manifests[label] = {"path": str(path), "sha256": sha(path), "files": rows,
                            "all_match": all(row["match"] for row in rows)}
    split = load(stage20a / "v2_split_manifest.json")
    group_split = split["group_splits"]
    # Inspect metadata for final_eval group IDs, never its image/label content.
    train_groups = {k for k, v in group_split.items() if v == "train"}
    eval_groups = {k for k, v in group_split.items() if v == "final_eval"}
    allow = load(local_p0 / "generator_train_allowlist.json")["inputs"]
    normal = load(local_p0 / "stage20b_normal_pool_freeze.json")["records"]
    requests = {}
    for row in allow:
        if row["split"] != "train" or row["source_group_id"] not in train_groups:
            raise ValueError("Non-TRAIN asset rejected before opening: " + row["annotation_id"])
        requests[repo / Path(row["path"].replace("\\", "/"))] = row["sha256"]
        requests[repo / Path(row["full_label_path"].replace("\\", "/"))] = row["full_label_sha256"]
    for row in normal:
        if row["split"] != "train" or row["known_source_group"] not in train_groups:
            raise ValueError("Non-TRAIN normal rejected before opening: " + row["normal_id"])
        requests[Path(row["runtime_path"].replace("\\", "/"))] = row["sha256"]
    train_assets = [bound(path, expected) for path, expected in requests.items()]
    server_allow = load(server_p0 / "generator_train_allowlist.json")["inputs"]
    def semantic_training(rows):
        return [{k: row[k] for k in ("annotation_id", "canonical_image_id", "source_group_id",
                                     "class", "sha256", "full_label_sha256", "geometry")}
                for row in rows]
    data = {"annotation_counts": dict(Counter(r["class"] for r in allow)),
            "unique_real_images": len({r["canonical_image_id"] for r in allow}),
            "unique_real_source_groups": len({r["source_group_id"] for r in allow}),
            "normal_count": len(normal), "asset_count": len(train_assets), "assets": train_assets,
            "all_asset_hashes_match": all(r["match"] for r in train_assets),
            "local_server_training_semantics_equal": semantic_training(allow) == semantic_training(server_allow),
            "train_groups_disjoint_from_final_eval": not bool(train_groups & eval_groups),
            "input_groups_are_frozen_train": all(r["source_group_id"] in train_groups for r in allow),
            "normal_groups_are_frozen_train": all(r["known_source_group"] in train_groups for r in normal),
            "physical_source_complete": False, "unknown_lineage_proven_absent": False,
            "image_decodes": 0, "final_eval_image_or_label_content_read": 0}
    slots = load(server_p0 / "stage20b_generation_slots.json")["slots"]
    rng = {r["slot_id"]: r["banks"] for r in load(server_p0 / "stage20b_rng_manifest.json")["records"]}
    slot_by_id = {s["slot_id"]: s for s in slots}
    banks = {}
    for bank, bank_seed in (("A", 42), ("B", 3407), ("C", 2026)):
        mf = formal / ("formal_bank_" + bank + "_manifest.json")
        rows = load(mf)["records"]
        assets = []
        checks = []
        for row in rows:
            slot = slot_by_id[row["slot_id"]]
            derive = int(hashlib.sha256(("Plastic_Bomo_Generation_V2" + row["slot_id"] + str(bank_seed)).encode()).hexdigest()[:16], 16)
            checks.append(row["frozen_slot"] == slot and row["bank_seed"] == bank_seed
                          and row["sample_seed"] == derive == rng[row["slot_id"]][bank]["sample_seed"]
                          and row["quality_selection"] is False)
            assets += [bound(formal / row["image_path"], row["image_sha256"]),
                       bound(formal / row["label_path"], row["label_sha256"])]
        banks[bank] = {"manifest_sha256": sha(mf), "count": len(rows),
                       "class_counts": dict(Counter(r["class"] for r in rows)),
                       "exact_slots_seed_derivation_equal": all(checks),
                       "unique_slots": len({r["slot_id"] for r in rows}),
                       "all_asset_hashes_match": all(r["match"] for r in assets), "assets": assets,
                       "images_decoded_or_quality_scored": 0}
    generators = {}
    for cls, row in load(formal / "generator_checkpoint_binding.json")["classes"].items():
        cp = row["checkpoint"]
        assets = [bound(formal / cp["path"] / name, digest) for name, digest in cp["sha256"].items()]
        generators[cls] = {"successful_updates": row["successful_updates"],
                           "tree_sha256": cp["tree_sha256"], "tree_hash_matches":
                           hashlib.sha256(canonical(cp["sha256"]).encode()).hexdigest() == cp["tree_sha256"],
                           "all_asset_hashes_match": all(r["match"] for r in assets), "assets": assets}
    detectors = {}
    for arm, row in load(formal / "detector_checkpoint_binding.json")["arms"].items():
        detectors[arm] = {**row, "asset": bound(formal / row["path"], row["sha256"])}
    base_weights = {"model_root": str(repo / "sd2-community/stable-diffusion-2-inpainting")}
    w = load(server_p0 / "sd2_base_weight_binding.json")["files"]
    base_weights["assets"] = [bound(Path(base_weights["model_root"]) / name, row["sha256"])
                              for name, row in w.items()]
    base_weights["all_asset_hashes_match"] = all(r["match"] for r in base_weights["assets"])
    # No MODEL deserialization. This subprocess asks only package metadata/specs.
    probe = """import importlib.metadata as m, importlib.util as u, json, sys
names=('torch','torchvision','tqdm','transformers','accelerate','safetensors','ultralytics','numpy','Pillow','PyYAML')
rows={}
for n in names:
 try: v=m.version(n)
 except m.PackageNotFoundError: v=None
 module={'Pillow':'PIL','PyYAML':'yaml'}.get(n,n)
 rows[n]={'version':v,'import_spec_exists':u.find_spec(module) is not None}
print(json.dumps({'python':sys.executable,'packages':rows}))
"""
    runtime = json.loads(subprocess.check_output([str(args.runtime_python), "-c", probe], cwd=repo))
    previous = load(repo / "results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254/stage20b_runtime_environment.json")
    runtime["historical_package_versions"] = previous["package_versions"]
    runtime["missing_generator_dependencies"] = [n for n in ("torch", "torchvision", "tqdm", "transformers", "accelerate", "safetensors")
                                                  if not runtime["packages"][n]["import_spec_exists"]]
    runtime["package_versions_equal"] = {n: runtime["packages"][n]["version"] == v for n, v in previous["package_versions"].items()
                                          if n in runtime["packages"]}
    try:
        runtime["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        runtime["gpu_inventory"] = "UNAVAILABLE"
    runtime["historical_raw_training_peak_vram_bytes"] = {c: load(formal / ("generator_training_" + c + ".json"))["peak_vram_bytes"] for c in ("flash", "black")}
    runtime["runtime_forward_backward_test"] = "NOT_RUN"
    return {"base_commit": args.base_commit, "scope": "LOCAL_P0_READ_ONLY", "sources": sources,
            "frozen_manifests": manifests, "v2_data": data, "baseline_raw_banks": banks,
            "baseline_generators": generators, "baseline_detectors": detectors,
            "public_base_weights": base_weights, "runtime": runtime,
            "experiments_executed": False, "optimizer_step_count": 0,
            "new_final_eval_content_read": 0, "files_written_by_collector": 0}


def compact(value):
    """Keep failures, counts and a digest of every checked row for small stdout."""
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key in ("files", "assets") and isinstance(item, list):
                output[key + "_verification"] = {
                    "checked": len(item), "matched": sum(bool(r["match"]) for r in item),
                    "failed": [r for r in item if not r["match"]],
                    "all_checked_rows_sha256": hashlib.sha256(canonical(item).encode()).hexdigest(),
                }
            else:
                output[key] = compact(item)
        return output
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--base_commit", default="91a6fcb4882ea056d796a27367ebd20f2695c3fe")
    parser.add_argument("--runtime_python", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(compact(result) if args.summary else result, indent=2, allow_nan=False))
