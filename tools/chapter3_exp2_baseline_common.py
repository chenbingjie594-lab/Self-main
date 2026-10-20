"""Exp2 P0 only: metadata/provenance, no ML imports or experimental execution."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata as metadata
import math
import os
from pathlib import Path
import platform
import sys

import chapter3_exp1_quality_common as c

METHODS = {"anomalydiffusion": "AnomalyDiffusion", "defectfill": "DefectFill"}
ORDER = ("Raw SD2", "AnomalyDiffusion", "DefectFill", "MSDF")
ZERO = dict(generator_training_count=0, synthetic_generation_count=0,
            detector_training_count=0, optimizer_step_count=0,
            BootstrapGuard=0, DWBG=0, selector=0, method_innovation_change_count=0,
            official_final_eval_forward=0, final_eval_image_or_label_content_read=0,
            inception_image_forward_count=0, DeepPCB=0)


def slot_seed(method, slot_id, bank_seed):
    c.require(method in METHODS and isinstance(slot_id, str) and slot_id
              and type(bank_seed) is int, "SEED_INPUT_INVALID", (method, slot_id, bank_seed))
    payload = c.canonical(dict(method=method, slot_id=slot_id, bank_seed=bank_seed))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def native_encoder_token_rows(batch_size, image_size):
    """Static shape deduction, NOT an executed neural-network test."""
    c.require(batch_size > 0 and image_size in (256, 512), "SHAPE_INPUT_INVALID", image_size)
    spatial = image_size // 4
    for _ in range(6):
        spatial = (spatial + 1) // 2  # k3, s2, p1
    return batch_size * spatial * spatial


def path_key(path):
    return os.path.normcase(os.path.realpath(os.fsdecode(path)))


@contextmanager
def forbid_content_reads(paths):
    """Deny exact final_eval assets; never ban every old physical /val/ folder."""
    forbidden = {path_key(p) for p in paths}
    state = {"active": True, "blocked_attempts": 0}

    def hook(event, args):
        if state["active"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            if path_key(args[0]) in forbidden:
                state["blocked_attempts"] += 1
                raise c.Stop("FINAL_EVAL_CONTENT_FORBIDDEN_IN_P0", os.fsdecode(args[0]))

    sys.addaudithook(hook)
    try:
        yield state
    finally:
        state["active"] = False


def frozen_inputs(cfg):
    for name, digest in cfg["frozen_bindings_sha256"].items():
        c.checked_file(c.inside(c.REPO, name), digest)
    exp1_cfg = c.load(c.resolve(cfg["exp1_config"]))
    root = c.resolve(cfg["exp1_root"])
    artifact = c.load(root / "experiment1_artifact_manifest.json")
    c.require(artifact["status"] == "COMPLETE", "EXP1_INCOMPLETE", str(root))
    # Only metric/manifest metadata is opened here. No reference/image/label contents.
    names = ("generation_quality_summary.json", "metric_protocol.json", "metric_runtime.json",
             "real_visual_reference_manifest.json", "visual_comparison_manifest.json", "experiment1_status.json")
    for name in names:
        c.checked_file(c.inside(root, name), artifact["sha256"][name])
    summary = c.load(root / "generation_quality_summary.json")
    c.require(c.load(root / "experiment1_status.json")["full_experiment_complete"],
              "EXP1_INCOMPLETE", "metric status")
    c.require(set(summary["methods"]) == {"Raw SD2", "MSDF"}, "EXP1_METHODS_CHANGED", "Raw/MSDF")
    for method in summary["methods"].values():
        for region in ("full_image", "roi"):
            for cohort in ("overall", "flash", "black"):
                for metric in ("FID", "KID"):
                    c.require(math.isfinite(method[region][cohort][metric]["mean"]),
                              "EXP1_METRICS_INVALID", (region, cohort, metric))
        for cohort in ("overall", "flash", "black"):
            c.require(math.isfinite(method["LPIPS"][cohort]["mean"]), "EXP1_METRICS_INVALID", cohort)
    return exp1_cfg, summary, c.load(root / "real_visual_reference_manifest.json")


def forbidden_paths(exp1_cfg, reference, real_root=None):
    result = []
    for row in reference["images"]:
        result.extend((c.resolve(row["image_path"]), c.resolve(row["label_path"])))
    registry = c.load(c.resolve(exp1_cfg["stage20a_root"]) / "real_source_registry.json")["records"]
    for row in registry:
        if row.get("asset_role") != "detector" or row.get("split_v2") != "final_eval":
            continue
        path = str(row["current_path"])
        result.append(path)
        if real_root:
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            for folder in ("train", "val", "final_eval"):
                result.extend((Path(real_root) / "images" / folder / name,
                               Path(real_root) / "labels" / folder / (Path(name).stem + ".txt")))
    return result


def verify_train_metadata(cfg, exp1_cfg, verify_assets=False, real_root=None, normal_root=None):
    p0 = c.resolve(exp1_cfg["p0_root"])
    train = c.load(p0 / "generator_train_allowlist.json")["inputs"]
    slots = c.load(p0 / "stage20b_generation_slots.json")["slots"]
    normals = c.load(p0 / "stage20b_normal_pool_freeze.json")["records"]
    rng = c.load(p0 / "stage20b_rng_manifest.json")["records"]
    registry = c.load(c.resolve(exp1_cfg["stage20a_root"]) / "real_source_registry.json")["records"]
    held_out = [r for r in registry if r.get("split_v2") == "final_eval"]
    final_nodes = {r["node_id"] for r in held_out}
    final_groups = {r.get("known_source_group_id") for r in held_out} - {None, ""}
    c.require(len(train) == 172 and Counter(r["class"] for r in train) == {"flash": 75, "black": 97},
              "TRAIN_ALLOWLIST_CHANGED", "172 annotations, flash75/black97")
    c.require(all(r["split"] == "train" and r["source_group_id"] not in final_groups
                  and r["canonical_image_id"] not in final_nodes for r in train),
              "TRAIN_FINAL_EVAL_OVERLAP", "positive V2 membership, not physical directory name")
    c.require(len(normals) == 394 and all(r["split"] == "train" and r["confirmatory_allowed"]
              and r["known_source_group"] not in final_groups for r in normals), "NORMAL_POOL_CHANGED", "394 TRAIN normals")
    c.require(len(slots) == len({r["slot_id"] for r in slots}) == 80
              and Counter(r["class"] for r in slots) == {"flash": 40, "black": 40}, "SLOT_PAIRING_INVALID", "40+40")
    annotations = {r["annotation_id"] for r in train}
    normal_ids = {r["normal_id"]: r["sha256"] for r in normals}
    rng_by_id = {r["slot_id"]: r for r in rng}
    c.require(len(rng_by_id) == len(rng) == 80, "RNG_MANIFEST_INVALID", "80 distinct slots")
    plans = {m: [] for m in METHODS}
    for s in sorted(slots, key=lambda x: x["slot_id"]):
        c.require(s["split"] == "train" and s["annotation_id"] in annotations
                  and s["source_group_id"] not in final_groups
                  and s["geometry"]["generator_canvas_wh"] == [512, 512]
                  and s["geometry"]["resize_scale_xy"] == [1.0, 1.0]
                  and s["geometry"]["mask_is_human_segmentation"] is False
                  and normal_ids[s["conditioning_normal_id"]] == s["normal_sha256"],
                  "SLOT_GEOMETRY_OR_LINEAGE_INVALID", s["slot_id"])
        g = s["geometry"]
        x0, y0, x1, y1 = g["fixed_crop_xyxy"]
        c.require(x1-x0 == y1-y0 == 512 and g["crop_shifted_to_rescue"] is False,
                  "SLOT_GEOMETRY_OR_LINEAGE_INVALID", s["slot_id"])
        digest = hashlib.sha256(c.canonical(s).encode()).hexdigest()
        for bank, bank_seed in cfg["banks"].items():
            old_rng = rng_by_id[s["slot_id"]]["banks"][bank]
            c.require(old_rng["bank_seed"] == bank_seed and old_rng["non_rng_metadata_sha256"] == digest,
                      "RNG_MANIFEST_INVALID", (bank, s["slot_id"]))
            for m in METHODS:
                plans[m].append(dict(method=METHODS[m], class_name=s["class"], bank=bank,
                    slot_id=s["slot_id"], bank_seed=bank_seed, seed=slot_seed(m, s["slot_id"], bank_seed),
                    conditioning_normal_id=s["conditioning_normal_id"], normal_sha256=s["normal_sha256"],
                    bbox=s["bbox"], crop=g["fixed_crop_xyxy"], frozen_slot=s,
                    frozen_slot_sha256=digest, status="PLANNED_NOT_SAMPLED"))
    checked = {}
    if verify_assets:
        c.require(real_root and normal_root, "TRAIN_ROOTS_REQUIRED", "--real_root and --normal_root")
        assets = []
        for r in train:
            original = r["path"].replace("\\", "/")
            c.require("/real_only/" in original, "TRAIN_PATH_INVALID", original)
            suffix = original.rsplit("/real_only/", 1)[1]
            assets.append((c.inside(real_root, suffix), r["sha256"]))
        assets += [(c.inside(normal_root, r["runtime_path"].replace("\\", "/").rsplit("/", 1)[-1]), r["sha256"]) for r in normals]
        for path, digest in assets:
            key = path_key(path)
            if key not in checked:
                c.checked_file(path, digest)
                checked[key] = digest
            c.require(checked[key] == digest, "TRAIN_ASSET_BINDING_CONFLICT", str(path))
    audit = dict(status="PASS", annotations=172, train_annotations_per_class={"flash": 75, "black": 97},
        normal_pool=394, slots=80, slots_per_class={"flash": 40, "black": 40},
        assets_status="SHA256_VERIFIED" if verify_assets else "NOT_RUN_METADATA_ONLY",
        unique_train_asset_content_reads=len(checked),
        TRAIN_final_eval_source_group_overlap=0, weak_bbox_is_not_dense_segmentation=True,
        no_physical_val_folder_split_inference=True, historical_slots_unchanged=True,
        normal_filtering=False, sample_seed_policy=cfg["slot_seed"], **ZERO)
    return audit, plans


def source_audit(root, spec):
    rows = []
    for name, expected in spec["source_text_sha256_LF"].items():
        path = c.inside(root, name)
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else None
        rows.append(dict(file=name, expected_sha256_LF=expected, actual_sha256_LF=actual,
                         status="PASS" if actual == expected else "MISSING" if actual is None else "CHANGED"))
    return dict(status="PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
                scope="selected pinned text files, LF-normalized only; no upstream modification",
                root=str(root), repository=spec["repository"], commit=spec["commit"], records=rows)


def dependency_inventory():
    packages = ("torch", "torchvision", "pytorch-lightning", "omegaconf", "transformers",
                "einops", "kornia", "test-tube", "torchmetrics", "timm", "scipy", "lpips", "torch-fidelity")
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), executable=sys.executable, packages=versions,
                status="INVENTORY_ONLY_NOT_RUNTIME_QUALIFICATION", ML_imports=0,
                package_install_or_upgrade_count=0,
                warning="Official AD environment is Python3.8/Torch1.12/Lightning1.5. Presence alone does not prove runtime compatibility.")


def qualifications(cfg, source, checkpoint, dependencies):
    present = checkpoint.is_file()
    ad_reasons = ["NATIVE_256_CORE_PATH_NOT_STRICT_512_READY",
                  "RESOLUTION_ADAPTER_NOT_IMPLEMENTED_OR_RUNTIME_QUALIFIED",
                  "JOINT_PAPER_VS_REPOSITORY_BUDGET_AND_PER_CLASS_MAPPING_NOT_FROZEN"]
    if source["status"] != "PASS":
        ad_reasons.append("PINNED_SOURCE_MISSING_OR_CHANGED")
    if not present:
        ad_reasons.append("NATIVE_LDM_CHECKPOINT_MISSING")
    ad = dict(method="AnomalyDiffusion", status="FAIL", execution_authorized=False,
        verdict="CONDITIONALLY_IMPLEMENTABLE_NOT_QUALIFIED_NOT_A_METHOD_IMPOSSIBILITY_CLAIM",
        reasons=ad_reasons, source=source,
        checkpoint=dict(path=str(checkpoint), status="PRESENT_UNVERIFIED" if present else "MISSING",
                        sha256=c.sha(checkpoint) if present else None, model_forward="NOT_RUN"),
        dependencies=dependencies,
        gates=dict(code=source["status"], native_weights="PRESENT_UNVERIFIED" if present else "MISSING",
            TRAIN_only_metadata="PASS", separate_class_training="NOT_QUALIFIED",
            strict_512_local_generation="FAIL_UNMODIFIED_UPSTREAM", raw_ABC="INTERFACE_MAPPING_PLANNED_NOT_RUN",
            lossless_output="INTERFACE_MAPPING_PLANNED_NOT_RUN", runtime="NOT_RUN"),
        static_shape_check=dict(kind="ANALYTIC_NOT_RUNTIME", native_image=256, V2_image=512,
            native_encoder_rows_for_B1=native_encoder_token_rows(1, 256),
            direct_512_encoder_rows_for_B1=native_encoder_token_rows(1, 512),
            note="A native256 mask-conditioning encoder could be retained while adapting diffusion512/latent64. Not implemented here."), **ZERO)
    df = dict(method="DefectFill", status="FAIL", execution_authorized=False,
        verdict="PROVENANCE_BLOCKED_NOT_PROVEN_WEAK_BBOX_INCOMPATIBLE",
        reasons=["AUTHOR_IMPLEMENTATION_NOT_IDENTIFIED_IN_BOUNDED_SEARCH",
                 "NO_VERIFIED_NATIVE_RUNTIME_OR_CHECKPOINT_CONTRACT"],
        source=cfg["defectfill"], gates=dict(code="PROVENANCE_BLOCKED", native_weights="NOT_QUALIFIED",
            dependencies="NOT_RUN_NO_AUTHOR_SOURCE", TRAIN_only_metadata="PASS",
            separate_class_training="NOT_QUALIFIED", strict_512_local_generation="NOT_QUALIFIED",
            raw_ABC="REQUIRES_DECLARED_WITHOUT_LFS_COMPARISON", lossless_output="NOT_QUALIFIED", runtime="NOT_RUN"),
        note="Raw outputs without LFS follow the user's rule, but cannot be labeled the full LFS paper pipeline. No unofficial method adopted.", **ZERO)
    return {"anomalydiffusion": ad, "defectfill": df}


def summary_outputs(output, exp1, cfg, quals):
    """Copy bound existing results; never fabricate or recompute baseline scores."""
    methods = {m: exp1["methods"].get(m) for m in ORDER}
    for region in ("full_image", "roi"):
        c.save(output / f"generation_quality_{region}_all.json", dict(status="BASELINE_PROTOCOL_BLOCKED",
            methods={m: s[region] if s else None for m, s in methods.items()},
            metric_source="unchanged frozen Experiment 1 results for Raw/MSDF; new baselines NOT_RUN",
            new_metric_calculation_count=0, D1_lowrank_FID_imported=False))
    classes = {cohort: {m: dict(full_image=s["full_image"][cohort], roi=s["roi"][cohort],
        LPIPS=s["LPIPS"][cohort]) if s else None for m, s in methods.items()} for cohort in ("flash", "black")}
    c.save(output / "generation_quality_classwise_all.json", dict(status="BASELINE_PROTOCOL_BLOCKED", classes=classes))

    def display(x):
        return f"{x['mean']:.6g} ± {x['sd']:.6g}"

    overall, classwise = [], []
    for m, s in methods.items():
        overall.append([m, *([display(s["roi"]["overall"][k]) for k in ("FID", "KID")]
            + [display(s["LPIPS"]["overall"]) ] if s else [None]*3),
            "REUSED_FROZEN_EXP1" if s else "NOT_RUN_P0_BLOCKED"])
        classwise.append([m, *([display(s["roi"][cohort][k]) if k != "LPIPS" else display(s["LPIPS"][cohort])
            for cohort in ("flash", "black") for k in ("FID", "KID", "LPIPS")] if s else [None]*6),
            "REUSED_FROZEN_EXP1" if s else "NOT_RUN_P0_BLOCKED"])
    c.csv_write(output / "paper_table_generation_quality_all.csv", ["Method", "FID↓", "KID↓", "LPIPS↑", "Status"], overall)
    c.csv_write(output / "paper_table_generation_quality_classwise_all.csv",
                ["Method", "Flash FID", "Flash KID", "Flash LPIPS", "Black FID", "Black KID", "Black LPIPS", "Status"], classwise)
    status = dict(status="BASELINE_PROTOCOL_BLOCKED", phase="P0_QUALIFICATION_ONLY",
        full_experiment_complete=False, Raw_present=True, MSDF_present=True,
        existing_results_present_scope="SHA-bound completed Exp1 metric/manifest metadata; source image files not reopened",
        methods=methods, metric_summary=methods, Raw_MSDF_metrics_recomputed=False,
        qualification_only=True, long_run_launcher_implemented=False, training_or_generation_auto_start=False, **ZERO)
    for key, label in METHODS.items():
        status.update({label+"_qualification": quals[key]["status"], label+"_generated_count": 0,
            label+"_planned_slot_count": 240, label+"_successful_outputs": 0, label+"_failed_outputs": 0,
            label+"_train_status_flash": "NOT_RUN_P0_BLOCKED", label+"_train_status_black": "NOT_RUN_P0_BLOCKED"})
    c.save(output / "exp2_status.json", status)
    return status
