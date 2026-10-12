"""Stage21B A1 contracts. No generator/selector and no changes to Stage20B code."""
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

from prepare_plastic_bomo_stage21b import (ARMS, BANKS, NAMESPACE, AuditStop, balanced_merge,
    bound_metadata, canonical, digest, group_split, load, occurrence_seed, real_records,
    require, safe_relative, save, sha, status_record, synthetic_audit, schedule_audit,
    STAGE20A_LEDGER, STAGE20B_LEDGER, P0_LEDGER)
from stage20b_detector_controller import Stage20BDetectorController

REPO = Path(__file__).resolve().parents[1]
ORIGINAL_LEDGER_SHA = "039896c1d5e10b5821f0c781216d29327058f92547269fd6e14fd659fe58c9dc"
R0_LEDGER_SHA = "282a381e2dc402f372204c45df3cf7d137d9e437b0b2aac0835ec32643b59f85"
EXCLUSIONS = {
    "black_ann_071f815d8a58e681f5a2cae5ad368b34c8aeb19bb9966083bc83a34f3e96daa1",
    "black_ann_37dc244042f3f61f90f6768ebe1ad7324feaaa2e48025b1c61f65ab2b7dfadb4",
    "flash_ann_7d22d7b137469454a63acb1fd7a702eff9aed47b127143a422122fec43183c07",
    "flash_ann_a035013bf058ed34572388912740a76fcd6a63eae3cea0ed5d9bd9fa294f7bb8",
}
NEW_CODE = ("stage21b_a1_common.py", "prepare_plastic_bomo_stage21b_a1.py",
            "run_plastic_bomo_stage21b_a1.py", "execute_plastic_bomo_stage21b_a1.py",
            "finalize_plastic_bomo_stage21b_a1.py")
HELPERS = ("prepare_plastic_bomo_stage21b.py", "qualify_plastic_bomo_stage20b_r0.py",
           "stage20b_runtime_contract.py", "stage20b_detector_controller.py",
           "evaluate_plastic_bomo_stage20b.py", "stage20b_formal_common.py")


def check_cfg(cfg):
    expected = {"namespace": NAMESPACE, "split_seed": 2101, "internal_train_images": 109,
        "internal_dev_images": 27, "train_groups": 107, "dev_groups": 27,
        "synthetic_per_class": {"flash": 32, "black": 29}, "epochs": 150, "detector_seed": 42,
        "late_start_epoch_0based": 50, "late_even_synthetic": 91, "late_odd_synthetic": 92,
        "draws_per_epoch": 170, "real_draws": 16350, "synthetic_draws": 9150, "total_draws": 25500,
        "accumulation_groups": [64, 64, 42], "optimizer_attempts": 450, "occurrences_per_role": 150,
        "exclusion_reason": "KNOWN_SOURCE_GROUP_LEAKAGE_EXCLUSION", "arms": list(ARMS),
        "amendment": "A1", "parent_commit": "e10096073514aa0e2a15e6541489123b9e81f270",
        "generator_dev_inputs": {"black": 23, "flash": 10}, "automatic_next_stage": False,
        "dev_population": "DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR",
        "official_final_eval": "FORBIDDEN_CONTENT_AND_FORWARD",
        "gates": {"mean_delta_pp_min": 0.5, "strict_bank_wins_min": 2, "class_mean_delta_pp_min": -2.0, "std_ddof": 1}}
    require(all(cfg.get(k) == v for k, v in expected.items())
            and set(cfg["excluded_conditioning_slots"]) == EXCLUSIONS
            and len(cfg["excluded_conditioning_slots"]) == 4, "STAGE21B_AMENDMENT_CHANGED")


def verify_prepared(root):
    root = Path(root)
    manifest = load(root / "amended_metadata_manifest.json")
    for name, expected in manifest["sha256"].items():
        require(sha(safe_relative(root, name)) == expected, "STAGE21B_FROZEN_METADATA_CHANGED: " + name)
    return manifest


def amended_allowlist(old, dev):
    require({r["slot_id"] for r in old["conditioning_dev_overlaps"]} == EXCLUSIONS,
            "STAGE21B_SOURCE_ISOLATION_FAILED: discovered conflicts do not equal authorized four")
    kept = [deepcopy(r) for r in old["records"] if r["slot_id"] not in EXCLUSIONS]
    require(len(kept) == 61 and Counter(r["class"] for r in kept) == {"flash": 32, "black": 29},
            "STAGE21B_SYNTHETIC_COUNTS_CHANGED")
    require(all(r["source_group_id"] not in dev and r["conditioning_normal_source_group"] not in dev for r in kept),
            "STAGE21B_SOURCE_ISOLATION_FAILED")
    exclusions = [{"slot_id": r["slot_id"], "class": r["class"], "affected_banks": list(BANKS),
                   "reason": "KNOWN_SOURCE_GROUP_LEAKAGE_EXCLUSION"} for r in old["records"] if r["slot_id"] in EXCLUSIONS]
    return {"version": "A1_v2", "records": kept, "eligible_per_class": {"flash": 32, "black": 29},
        "donor_dev_excluded": deepcopy(old["donor_dev_excluded"]), "conditioning_dev_excluded": exclusions,
        "same_slot_ids_all_banks": True, "quality_selection": False, "all_legal_synthetic_used": True,
        "no_replacement_or_generation": True, "population": "61 frozen raw slots per bank"}


def make_schedules(real_roles, syn_roles, base_cfg, cfg):
    check_cfg(cfg)
    require(len(real_roles) == 109 and len(syn_roles) == 61, "STAGE21B_COMPUTE_PAIRING_INVALID")
    roles = sorted(real_roles) + sorted(syn_roles)
    require(len(set(roles)) == 170 and all(r.startswith("base_") for r in real_roles)
            and all(r.startswith("extra_") for r in syn_roles), "STAGE21B_COMPUTE_PAIRING_INVALID")
    index = {r: i for i, r in enumerate(roles)}
    cyclic = lambda rs: sorted(rs, key=lambda r: (digest([NAMESPACE, "role_order", r]), r))
    rr, sr = cyclic(real_roles), cyclic(syn_roles)
    real_queue = [[index[r], k] for k in range(150) for r in rr]
    syn_queue = [[index[r], k] for k in range(150) for r in sr]
    uniform = [balanced_merge(real_queue[e * 109:(e + 1) * 109], syn_queue[e * 61:(e + 1) * 61]) for e in range(150)]
    curriculum = [real_queue[e * 170:(e + 1) * 170] for e in range(50)]
    ri, si = 8500, 0
    for e in range(100):
        ns = 91 if e % 2 == 0 else 92
        nr = 170 - ns
        curriculum.append(balanced_merge(real_queue[ri:ri + nr], syn_queue[si:si + ns]))
        ri += nr; si += ns
    require(ri == 16350 and si == 9150, "STAGE21B_COMPUTE_PAIRING_INVALID")
    common = {"version": "A1_v2", "roles": roles, "draw_encoding": "[role_index, occurrence_index]",
        "draws_per_epoch": 170, "epochs": 150, "scheduled_draws": 25500, "scheduled_optimizer_attempts": 450,
        "per_role_total_exposure": 150, "accumulation_group_lengths_per_epoch": [64, 64, 42],
        "augmentation_seed_formula": cfg["paired_augmentation"], "detector_seed": 42,
        "native_warmup_draws": 510, "no_quality_selection": True}
    def pack(name, seq):
        return {**common, "mode": name, "epoch_draws": seq, "sequence_sha256": digest([x for e in seq for x in e])}
    u, c = pack("UniformMix", uniform), pack("LateSyntheticCurriculum", curriculum)
    pairing, equality = schedule_audit(u, c, base_cfg)
    equality["strict_fixed_compute_requires_equal_actual_successful_updates"] = False
    equality["compute_definition"] = cfg["compute_definition"]
    equality["per_epoch_tail_normalizer"] = 42
    validate_schedules(u, c, base_cfg, cfg)
    return u, c, pairing, equality


def validate_schedules(u, c, base_cfg, cfg):
    check_cfg(cfg)
    schedule_audit(u, c, base_cfg)
    require(len(u["epoch_draws"]) == len(c["epoch_draws"]) == 150
            and u["scheduled_draws"] == c["scheduled_draws"] == 25500
            and u["scheduled_optimizer_attempts"] == c["scheduled_optimizer_attempts"] == 450
            and u["accumulation_group_lengths_per_epoch"] == c["accumulation_group_lengths_per_epoch"] == [64,64,42],
            "STAGE21B_COMPUTE_PAIRING_INVALID")
    for e in range(150):
        us = sum(u["roles"][i].startswith("extra_") for i, _ in u["epoch_draws"][e])
        cs = sum(c["roles"][i].startswith("extra_") for i, _ in c["epoch_draws"][e])
        require(us == 61 and cs == (0 if e < 50 else 91 + (e - 50) % 2), "STAGE21B_COMPUTE_PAIRING_INVALID")


class TimingController(Stage20BDetectorController):
    """Reuse the tested callback loop/backward; adapt only occurrence order and N=170."""
    def __init__(self, frozen_training, schedule, base_cfg):
        self.training = frozen_training
        self.order = {"epochs": []}
        for e, draws in enumerate(schedule["epoch_draws"]):
            self.order["epochs"].append({"epoch_0based": e,
                "role_sequence": [schedule["roles"][i] for i, _ in draws],
                "augmentation_seeds": [occurrence_seed(schedule["roles"][i], k, base_cfg) for i, k in draws],
                "optimizer_attempt_after_positions_0based": [63,127,169], "accumulation_group_lengths": [64,64,42]})

    def learning_rates(self, epoch, position, param_groups):
        args = self.training
        factor = max(1 - epoch / args["epochs"], 0) * (1 - args["lrf"]) + args["lrf"]
        ni, nw = epoch * 170 + position, round(args["warmup_epochs"] * 170)
        result = []
        for group in param_groups:
            target = group.get("initial_lr", args["lr0"]) * factor
            initial = args["warmup_bias_lr"] if group.get("param_group") == "bias" else 0.0
            result.append(initial + (target - initial) * ni / nw if nw > 0 and ni < nw else target)
        return result


def initial_status():
    s = status_record("STAGE21B_A1_PREPARED_SERVER_RUNTIME_PENDING")
    s.update(amendment="A1", phase="PREPARED", server_training_runtime_implemented=True,
        internal_train_images=109, internal_dev_images=27, synthetic_flash=32, synthetic_black=29,
        synthetic_total=61, total_draws_per_arm=25500, scheduled_optimizer_attempts=450,
        successful_updates={a: None for a in ARMS}, official_final_eval_access_count=0,
        internal_dev_eval_rounds=0, generator_internal_dev_historical_exposure=True,
        pipeline_independent_holdout_claim=False, dev_population="DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR",
        Stage21B_original_audit_modified=False, completed_arms=[], compute_definition="fixed scheduled draws and attempts; AMP successful-update counts disclosed")
    return s


def update(run, **fields):
    s = load(Path(run) / "stage21b_status.json"); s.update(fields)
    save(Path(run) / "stage21b_status.json", s)


def verified_copy(src, dst, expected):
    src, dst = Path(src), Path(dst)
    require(src.is_file() and sha(src) == expected, "STAGE21B_SOURCE_ASSET_CHANGED: " + str(src))
    require(not dst.exists(), "STAGE21B_OUTPUT_EXISTS: " + str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    require(sha(dst) == expected, "STAGE21B_COPY_HASH_MISMATCH")


def check_runtime(p0, r0, detector):
    md = bound_metadata(p0, "p0_frozen_artifact_manifest.json", P0_LEDGER,
        ("stage20b_runtime_binding.json", "stage20b_detector_resolved_runtime_arguments.json", "detector_initialization_binding.json"), [])
    rmd = bound_metadata(r0, "r0_artifact_manifest.json", R0_LEDGER_SHA,
        ("runtime_environment.json", "detector_model_load_audit.json"), [])
    runtime = md["stage20b_runtime_binding.json"]["environment"]
    actual = {}
    for package, expected in runtime["versions"].items():
        actual[package] = importlib.metadata.version(package)
        require(actual[package] == expected, "STAGE21B_RUNTIME_CHANGED: " + package)
    dist = importlib.metadata.distribution("ultralytics")
    for name, row in runtime["source_files"].items():
        path = Path(dist.locate_file(name)) if name.startswith("ultralytics/") else REPO / name
        require(sha(path) == row["sha256"], "STAGE21B_RUNTIME_SOURCE_CHANGED: " + name)
    require(sha(detector) == md["detector_initialization_binding.json"]["sha256"], "STAGE21B_INITIALIZATION_CHANGED")
    return md, rmd, actual


def child_context(run):
    run = Path(run).resolve()
    require(os.environ.get("STAGE21B_BINDING_SHA256") == sha(run / "execution_binding.json"), "STAGE21B_CHILD_BINDING_MISSING")
    b = load(run / "execution_binding.json")
    for name, h in b["code_sha256"].items():
        require(sha(REPO / name) == h, "STAGE21B_EXECUTABLE_CHANGED: " + name)
    verify_prepared(run)
    require(load(run / "stage21b_status.json")["status"] == "STAGE21B_A1_RUNNING", "STAGE21B_CHILD_STATE_INVALID")
    return b


class ContentGuard:
    """Exact V2 membership denial plus native image-reader input allowlists.

    Not an OS sandbox: covers Python/audit file opens, numpy.fromfile and
    cv2.imread; all native dataset image paths must additionally be allowlisted.
    """
    def __init__(self, official_paths, dev_paths, official_prefixes=(), dev_prefixes=(), allow_dev=False):
        self.official = {str(Path(p).resolve()) for p in official_paths}
        self.dev = {str(Path(p).resolve()) for p in dev_paths}
        self.official_prefixes = [Path(p).resolve() for p in official_prefixes]
        self.dev_prefixes = [Path(p).resolve() for p in dev_prefixes]
        self.allow_dev = allow_dev
        self.armed = False
        self.native_images = set()
        self.counts = {"official_blocked_open_attempts": 0, "internal_dev_blocked_open_attempts": 0,
                       "official_final_eval_image_read": 0, "official_final_eval_label_read": 0,
                       "official_final_eval_forward": 0, "internal_dev_content_allowed": allow_dev}

    def check(self, path, native=False):
        if not isinstance(path, (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(path)).resolve(); s = str(p)
        if s in self.official or any(p.is_relative_to(q) for q in self.official_prefixes):
            self.counts["official_blocked_open_attempts"] += 1
            raise AuditStop("STAGE21B_OFFICIAL_FINAL_EVAL_ACCESS_FORBIDDEN")
        if not self.allow_dev and (s in self.dev or any(p.is_relative_to(q) for q in self.dev_prefixes)):
            self.counts["internal_dev_blocked_open_attempts"] += 1
            raise AuditStop("STAGE21B_INTERNAL_DEV_BARRIER_VIOLATION")
        if native and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            require(s in self.native_images, "STAGE21B_NATIVE_IMAGE_NOT_ALLOWLISTED: " + s)

    @contextmanager
    def python_guard(self):
        def hook(event, args):
            if self.armed and event == "open": self.check(args[0])
        sys.addaudithook(hook); self.armed = True
        try: yield self
        finally: self.armed = False

    @contextmanager
    def native_guard(self):
        import cv2
        import numpy as np
        from unittest.mock import patch
        oldcv, oldnp = cv2.imread, np.fromfile
        def imread(path, *args, **kw):
            self.check(path, native=True); return oldcv(path, *args, **kw)
        def fromfile(path, *args, **kw):
            self.check(path, native=True); return oldnp(path, *args, **kw)
        with patch.object(cv2, "imread", imread), patch.object(np, "fromfile", fromfile): yield self


def guard_for(run, b, allow_dev=False):
    official = []
    for row in load(Path(b["stage20a"]) / "real_source_registry.json")["records"]:
        if row["asset_role"] == "detector" and row["split_v2"] == "final_eval":
            im = PurePosixPath(row["portable_locator"]["relative_path"])
            official += [Path(b["real_root"]) / str(im), Path(b["real_root"]) / "labels" / row["split_membership_old"] / (im.stem + ".txt")]
    require(len(official) == 96, "STAGE21B_OFFICIAL_DENYLIST_INVALID")
    dev = []
    for row in load(Path(run) / "stage21b_split_manifest.json")["records"]:
        if row["internal_split"] == "internal_dev":
            dev += [Path(b["real_root"]) / row[k + "_relative_path"] for k in ("image", "label")]
    return ContentGuard(official, dev,
        official_prefixes=[Path(b["stage20b"]) / "final_eval_inputs", Path(b["stage20b"]).parent / "formal_20261009_060552/final_eval_inputs"],
        dev_prefixes=[Path(run) / "internal_dev_inputs"], allow_dev=allow_dev)


@contextmanager
def gpu_guard(run, b, tag, allow_dev=False):
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden, set_gpu_runtime
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", YOLO_AUTOINSTALL="false",
        YOLO_OFFLINE="true", NO_ALBUMENTATIONS_UPDATE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
    g = guard_for(run, b, allow_dev)
    try:
        with ExitStack() as stack:
            stack.enter_context(g.python_guard()); stack.enter_context(network_forbidden())
            import torch
            set_gpu_runtime(torch, b["device"])
            stack.enter_context(g.native_guard())
            expected = load(Path(run) / "r0_runtime_environment_snapshot.json")
            actual = {"python": sys.version, "torch_version": torch.__version__, "CUDA_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability())}
            require(all(actual[k] == expected[k] for k in actual), "STAGE21B_GPU_RUNTIME_CHANGED")
            yield torch, g
    finally:
        save(Path(run) / (tag + "_file_access_guard.json"), g.counts)


def checkpoint_barrier(run):
    rows = {}
    for arm in ARMS:
        a = load(Path(run) / (arm + "_training.json"))
        require(a["status"] == "COMPLETE" and a["epochs"] == 150 and a["actual_draws"] == 25500
                and a["attempted_optimizer_updates"] == 450, "STAGE21B_EVALUATION_BARRIER_FAILED")
        p = safe_relative(run, a["last_checkpoint"])
        require(sha(p) == a["last_sha256"], "STAGE21B_CHECKPOINT_CHANGED")
        rows[arm] = {"path": a["last_checkpoint"], "sha256": a["last_sha256"], "epochs": 150}
    return {"status": "ALL_SIX_EPOCH150_LAST_FROZEN", "arms": rows, "validation_selection": False}
