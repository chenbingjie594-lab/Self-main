"""Stage21B metadata/asset entry audit. No models, training, generation or evaluation.

The immutable 80/20 group split is never rebalanced. A conditioning-background
overlap stops preparation rather than silently extending the donor allowlist.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

NAMESPACE = "Plastic_Bomo_Stage21B_TrainOnly"
STAGE20A_LEDGER = "2caa71e8cb3af1b6cae70b7d70974d7ccf6d70d3ea3eee37b4e22bbad1c2a079"
STAGE20B_LEDGER = "3911775b0bfd25b2d700be8e0a1b82a6b5177d3dc3d9700d5242daa724689af5"
P0_LEDGER = "6660bb09bda4fdbfa09dc91e0b3bc9e3a5bcf3feb46a04a6307235f46aa29f3d"
BANKS = ("A", "B", "C")
ARMS = tuple(f"{bank}_{mode}" for bank in BANKS for mode in ("uniform", "curriculum"))


class AuditStop(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise AuditStop(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value, compact=False):
    text = canonical(value) if compact else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8", newline="\n")


def safe_relative(root, relative):
    require(isinstance(relative, str) and "\\" not in relative, "nonportable asset locator")
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and ":" not in relative, "asset path traversal")
    result = (Path(root) / relative).resolve()
    require(result.is_relative_to(Path(root).resolve()), "asset path escaped its bound root")
    return result


def validate_config(cfg):
    expected = {"namespace": NAMESPACE, "split_seed": 2101, "train_group_fraction": 0.8,
                "epochs": 150, "detector_seed": 42, "late_start_epoch_0based": 50,
                "exposures_per_role": 150, "gradient_accumulation": 64,
                "minimum_synthetic_per_class": 20, "arms": list(ARMS)}
    require(all(cfg.get(k) == v for k, v in expected.items()), "frozen Stage21B design changed")
    require(cfg["gates"] == {"mean_delta_pp_min": 0.5, "strict_bank_wins_min": 2,
                "class_mean_delta_pp_min": -2.0, "std_ddof": 1, "curriculum_std_not_greater": True},
            "frozen premise gates changed")


def bound_metadata(root, ledger_name, expected, names, evidence):
    """Only explicit JSON metadata is read. Never walk a heavy/final-eval ledger."""
    root = Path(root)
    ledger_path = root / ledger_name
    require(sha(ledger_path) == expected, "authoritative ledger changed: " + str(ledger_path))
    ledger = load(ledger_path)["sha256"]
    values = {}
    evidence.append({"file": str(ledger_path), "sha256": expected, "kind": "metadata_ledger"})
    for name in names:
        require(name.endswith(".json") and name in ledger, "unbound metadata: " + name)
        path = safe_relative(root, name)
        require(sha(path) == ledger[name], "frozen metadata changed: " + str(path))
        evidence.append({"file": str(path), "sha256": ledger[name], "kind": "metadata_only"})
        values[name] = load(path)
    return values


def group_split(real, cfg):
    validate_config(cfg)
    require(len({x["node_id"] for x in real}) == len(real), "duplicate detector node")
    groups = sorted({x["known_source_group_id"] for x in real},
                    key=lambda g: (digest([NAMESPACE, 2101, g]), g))
    cutoff = len(groups) * 4 // 5
    train, dev = set(groups[:cutoff]), set(groups[cutoff:])
    require(train and dev, "empty group split")
    require(not train & dev, "group split overlap")
    return train, dev, [{"source_group_id": g, "stable_sha256": digest([NAMESPACE, 2101, g]),
                        "split": "internal_train" if g in train else "internal_dev"} for g in groups]


def real_records(registry, annotations, train_groups):
    labels, classes = {}, {}
    for ann in annotations:
        if ann["split_v2"] == "train":
            labels.setdefault(ann["node_id"], set()).add(ann["label_sha256"])
            classes.setdefault(ann["node_id"], []).append(ann["class"])
    records = []
    for row in registry:
        if row["asset_role"] != "detector" or row["split_v2"] != "train":
            continue
        require(len(labels.get(row["node_id"], set())) == 1, "ambiguous TRAIN label hash")
        relative = row["portable_locator"]["relative_path"]
        image = PurePosixPath(relative)
        label = "labels/" + row["split_membership_old"] + "/" + image.stem + ".txt"
        records.append({"role_id": "base_" + row["node_id"], "node_id": row["node_id"],
            "canonical_image_id": row["canonical_image_id"],
            "source_group_id": row["known_source_group_id"], "split_v2": "train",
            "internal_split": "internal_train" if row["known_source_group_id"] in train_groups else "internal_dev",
            "image_relative_path": relative, "label_relative_path": label,
            "image_sha256": row["file_sha256"], "label_sha256": next(iter(labels[row["node_id"]])),
            "class_instances": classes[row["node_id"]], "width": row["width"], "height": row["height"]})
    require(len(records) == 136, "expected V2 TRAIN136, not another historical split")
    return sorted(records, key=lambda x: x["role_id"])


def synthetic_audit(banks, real, train, dev, normals, source_groups):
    require(set(banks) == set(BANKS), "three banks required")
    indices = {}
    for bank, manifest in banks.items():
        rows = manifest["records"]
        require(manifest["status"] == "COMPLETE" and manifest["count"] == 80 and len(rows) == 80
                and manifest["quality_selection"] is False and manifest["reroll_count"] == 0,
                "Raw bank is not the original frozen unselected bank")
        indices[bank] = {r["slot_id"]: r for r in rows}
        require(len(indices[bank]) == 80 and all(r["bank"] == bank for r in rows), "duplicate/misnamed bank slot")
    require(all(set(indices[b]) == set(indices["A"]) for b in BANKS), "bank slot IDs differ")
    real_by_node = {r["node_id"]: r for r in real}
    normal_by_id = {}
    for row in normals:
        normal_by_id.setdefault(row["normal_id"], []).append(row)
    group_nodes = {r["source_group_id"]: set(r["nodes"]) for r in source_groups}
    kept, excluded, overlaps = [], [], []
    for slot_id in sorted(indices["A"]):
        first = indices["A"][slot_id]
        slot = first["frozen_slot"]
        donor = real_by_node.get(slot["node_id"])
        require(donor is not None and slot["split"] == "train"
                and donor["source_group_id"] == slot["source_group_id"]
                and donor["image_sha256"] == slot["source_sha256"], "slot donor not a bound V2 TRAIN image")
        require(slot["source_group_id"] in train | dev, "unassigned donor group")
        for bank in BANKS:
            row = indices[bank][slot_id]
            require(row["frozen_slot"] == slot and row["role_id"] == first["role_id"]
                    and row["class"] == first["class"] and row["label_sha256"] == first["label_sha256"]
                    and row["non_rng_metadata_sha256"] == first["non_rng_metadata_sha256"]
                    and row["checkpoint_tree_sha256"] == first["checkpoint_tree_sha256"]
                    and row["quality_selection"] is False and row["formal_synthetic"] is True,
                    "non-RNG bank metadata or labels differ")
        if slot["source_group_id"] in dev:
            excluded.append({"slot_id": slot_id, "class": first["class"],
                             "source_group_id": slot["source_group_id"], "reason": "DONOR_INTERNAL_DEV",
                             "excluded_in_banks": list(BANKS)})
            continue
        matched = [n for n in normal_by_id.get(slot["conditioning_normal_id"], [])
                   if n["sha256"] == slot["normal_sha256"] and n["known_source_group"] == slot["normal_source_group"]]
        require(matched and all(n["split"] == "train" and n["confirmatory_allowed"] for n in matched),
                "normal conditioning provenance is not frozen V2 TRAIN")
        normal = matched[0]
        require(normal["node_id"] in group_nodes.get(slot["normal_source_group"], set()), "normal group membership missing")
        item = {"slot_id": slot_id, "role_id": first["role_id"], "class": first["class"],
                "source_group_id": slot["source_group_id"], "conditioning_normal_id": slot["conditioning_normal_id"],
                "conditioning_normal_source_group": slot["normal_source_group"],
                "banks": {b: {k: indices[b][slot_id][k] for k in
                    ("image_path", "image_sha256", "label_path", "label_sha256", "sample_seed", "checkpoint_tree_sha256")}
                    for b in BANKS}}
        kept.append(item)
        if slot["normal_source_group"] in dev:
            related = [r for r in real if r["source_group_id"] == slot["normal_source_group"]]
            require(related and all(r["internal_split"] == "internal_dev"
                    and r["node_id"] in group_nodes[slot["normal_source_group"]] for r in related), "inconsistent dev ancestry")
            overlaps.append({"slot_id": slot_id, "class": first["class"], "affected_banks": list(BANKS),
                "donor_source_group": slot["source_group_id"], "donor_split": "internal_train",
                "normal_source_group": slot["normal_source_group"], "normal_node_id": normal["node_id"],
                "normal_id": normal["normal_id"], "normal_path_metadata_only": slot["normal_path"],
                "normal_sha256_metadata_only": slot["normal_sha256"], "internal_dev_relatives": related,
                "evidence": "frozen normal registry and source_group_registry place conditioning normal and dev detector node in same known source group",
                "additional_exclusion_applied": False})
    counts = {c: sum(r["class"] == c for r in kept) for c in ("flash", "black")}
    return {"records": kept, "donor_dev_excluded": excluded, "eligible_per_class": counts,
            "conditioning_dev_overlaps": overlaps, "quality_selection": False,
            "same_slot_ids_all_banks": True, "all_donor_eligible_slots_retained": True,
            "filter_rule": "only donor group internal_train; conditioning conflicts reported, NOT silently removed"}


def occurrence_seed(role, occurrence, cfg):
    return int(digest([cfg["namespace"], "augmentation", role, occurrence, cfg["detector_seed"]])[:16], 16) % (2**63)


def balanced_merge(real, synthetic):
    """Fixed integer interleaving, no scores/ranking/search; each input appears once."""
    total, ns = len(real) + len(synthetic), len(synthetic)
    ri = si = 0
    result = []
    for i in range(total):
        if (i + 1) * ns // total > i * ns // total:
            result.append(synthetic[si]); si += 1
        else:
            result.append(real[ri]); ri += 1
    return result


def schedules(real_roles, synthetic_roles, cfg):
    validate_config(cfg)
    roles = sorted(real_roles) + sorted(synthetic_roles)
    require(len(set(roles)) == len(roles) and real_roles and synthetic_roles, "empty/overlapping role sets")
    nreal, draws, epochs = len(real_roles), len(roles), cfg["epochs"]
    start = cfg["late_start_epoch_0based"]
    require(start * draws <= epochs * nreal, "late schedule impossible without increasing real exposure")
    idx = {role: i for i, role in enumerate(roles)}
    ordered = lambda rs: sorted(rs, key=lambda r: (digest([NAMESPACE, "role_order", r]), r))
    rr, sr = ordered(real_roles), ordered(synthetic_roles)
    uniform = []
    for e in range(epochs):
        r = sorted(rr, key=lambda role: (digest([NAMESPACE, "uniform_real", e, role]), role))
        s = sorted(sr, key=lambda role: (digest([NAMESPACE, "uniform_synthetic", e, role]), role))
        uniform.extend(balanced_merge([[idx[role], e] for role in r], [[idx[role], e] for role in s]))
    real_queue = [[idx[r], e] for e in range(epochs) for r in rr]
    syn_queue = [[idx[s], e] for e in range(epochs) for s in sr]
    warmup_draws = start * draws
    curriculum = real_queue[:warmup_draws] + balanced_merge(real_queue[warmup_draws:], syn_queue)
    groups = [min(cfg["gradient_accumulation"], draws - p) for p in range(0, draws, cfg["gradient_accumulation"])]
    common = {"status": "FROZEN_DESIGN_ONLY_NOT_EXECUTED", "execution_authorized": False,
              "roles": roles, "draw_encoding": "[role_index into roles, occurrence_index]; occurrence_id=role_id+':' + occurrence_index",
              "occurrence_seed_formula": cfg["paired_augmentation"], "detector_seed": 42,
              "draws_per_epoch": draws, "epochs": epochs, "per_role_total_exposure": epochs,
              "accumulation_group_lengths_per_epoch": groups,
              "scheduled_optimizer_attempts": epochs * len(groups),
              "scheduled_draws": draws * epochs,
              "native_warmup_draws": round(3.0 * draws),
              "learning_rate_progress": "Stage20B linear epoch decay and 3-epoch native LR warmup, with this split's constant draws_per_epoch",
              "schedule_does_not_authorize_training": True}
    def pack(mode, sequence):
        return {**common, "mode": mode, "sequence_sha256": digest(sequence),
                "epoch_draws": [sequence[e * draws:(e + 1) * draws] for e in range(epochs)]}
    return pack("UniformMix", uniform), pack("LateSyntheticCurriculum", curriculum)


def schedule_audit(uniform, curriculum, cfg):
    require(uniform["roles"] == curriculum["roles"], "paired roles changed")
    roles = uniform["roles"]
    u = [tuple(x) for e in uniform["epoch_draws"] for x in e]
    c = [tuple(x) for e in curriculum["epoch_draws"] for x in e]
    expected = Counter((i, k) for i in range(len(roles)) for k in range(cfg["epochs"]))
    require(Counter(u) == Counter(c) == expected, "occurrence multiset mismatch")
    require(digest([list(x) for x in u]) == uniform["sequence_sha256"]
            and digest([list(x) for x in c]) == curriculum["sequence_sha256"], "draw ordering digest mismatch")
    seeds = [[roles[i], k, occurrence_seed(roles[i], k, cfg)] for i, k in sorted(expected)]
    warmup = c[:cfg["late_start_epoch_0based"] * len(roles)]
    require(all(roles[i].startswith("base_") for i, _ in warmup), "synthetic in first third")
    require(all(len(e) == len(roles) for e in uniform["epoch_draws"] + curriculum["epoch_draws"]), "epoch draws differ")
    require(uniform["scheduled_optimizer_attempts"] == curriculum["scheduled_optimizer_attempts"], "attempt counts differ")
    counts = {r: cfg["epochs"] for r in roles}
    totals = {"total_draws": len(u), "total_real_draws": sum(n for r, n in counts.items() if r.startswith("base_")),
              "total_synthetic_draws": sum(n for r, n in counts.items() if r.startswith("extra_")),
              "optimizer_attempts": uniform["scheduled_optimizer_attempts"], "per_role_counts": counts}
    pairing = {"status": "DESIGN_PAIRING_PASS_NOT_RUNTIME_TESTED", "occurrence_multiset_equal": True,
               "augmentation_seed_multiset_equal": True, "occurrence_seed_binding_sha256": digest(seeds),
               "augmentation_seed_depends_on_epoch_or_position": False,
               "same_slot_role_seed_across_banks": True, "actual_transformed_tensor_pairing_test": "NOT_RUN",
               "temporal_sequence_differs": u != c, "early_synthetic_draws": 0}
    equality = {"status": "DESIGN_EXPOSURE_EQUALITY_PASS_NOT_EXECUTED", "uniform": totals, "curriculum": totals,
                "per_occurrence_counts_exactly_one": True, "successful_updates": "NOT_RUN",
                "strict_fixed_compute_requires_equal_actual_successful_updates": True,
                "augmented_input_hashes": "NOT_RUN", "actual_draw_counts": "NOT_RUN"}
    return pairing, equality


def verify_assets(allowlist, bank_root, real, real_root=None):
    records = []
    for row in allowlist["records"]:
        for bank in BANKS:
            asset = row["banks"][bank]
            for kind in ("image", "label"):
                path = safe_relative(bank_root, asset[kind + "_path"])
                require(path.is_file() and sha(path) == asset[kind + "_sha256"], "synthetic asset missing or modified: " + str(path))
                records.append({"kind": "synthetic_" + kind, "bank": bank, "slot_id": row["slot_id"],
                                "relative_path": asset[kind + "_path"], "sha256": asset[kind + "_sha256"]})
    if real_root is not None:
        for row in real:
            require(row["split_v2"] == "train", "attempt to verify official final_eval content")
            for kind in ("image", "label"):
                path = safe_relative(real_root, row[kind + "_relative_path"])
                require(path.is_file() and sha(path) == row[kind + "_sha256"], "V2 TRAIN asset missing or modified: " + str(path))
                records.append({"kind": "V2_TRAIN_" + kind, "node_id": row["node_id"],
                                "relative_path": row[kind + "_relative_path"], "sha256": row[kind + "_sha256"]})
    return {"status": "SYNTHETIC_HASH_VERIFIED" if real_root is None else "TRAIN_AND_SYNTHETIC_HASH_VERIFIED",
            "synthetic_assets_verified": len(allowlist["records"]) * 3 * 2,
            "real_TRAIN_assets_verified": 0 if real_root is None else len(real) * 2,
            "real_TRAIN_hash_check": "NOT_RUN_LOCAL_SERVER_PATHS_UNAVAILABLE" if real_root is None else "PASS",
            "official_final_eval_content_read": 0, "records": records}


def status_record(status):
    return {"status": status, "phase": "ENTRY_AUDIT_NOT_TRAINING", "STAGE21B_EXECUTION_AUTHORIZED": False,
            **{k: "NOT_RUN" for k in ("Gate_A", "Gate_B", "Gate_C", "Gate_D")},
            **{k: None for k in ("delta_A_pp", "delta_B_pp", "delta_C_pp", "mean_delta_pp",
                "uniform_bank_std_pp", "curriculum_bank_std_pp", "Flash_mean_delta_pp", "Black_mean_delta_pp")},
            "final_eval_access_count": 0, "official_final_eval_forward": 0,
            "final_eval_image_read": 0, "final_eval_label_read": 0,
            "generation_count": 0, "detector_training_count": 0, "optimizer_step_count": 0,
            "internal_dev_forward": 0, "BootstrapGuard": 0, "DeepPCB": 0,
            "Stage20A_modified": False, "Stage20B_modified": False, "Stage21C_auto_start": False,
            "full_experiment_complete": False}


def prepare(a):
    cfg = load(a.protocol)
    validate_config(cfg)
    evidence = []
    md = bound_metadata(a.stage20a, "frozen_artifact_manifest.json", STAGE20A_LEDGER,
        ("real_source_registry.json", "annotation_provenance_v2.json", "normal_source_registry.json", "source_group_registry.json"), evidence)
    bankmd = bound_metadata(a.stage20b, "formal_result_artifact_manifest.json", STAGE20B_LEDGER,
        tuple(f"formal_bank_{b}_manifest.json" for b in BANKS) + ("formal_synthetic_validity.json",), evidence)
    p0md = bound_metadata(a.p0, "p0_frozen_artifact_manifest.json", P0_LEDGER,
        ("generator_train_allowlist.json", "stage20b_detector_protocol.json"), evidence)
    require(bankmd["formal_synthetic_validity.json"]["status"] == "PASS", "Stage20B synthetic validity not PASS")
    registry = md["real_source_registry.json"]["records"]
    v2train = [x for x in registry if x["asset_role"] == "detector" and x["split_v2"] == "train"]
    require(len(v2train) == 136, "V2 TRAIN136 unavailable")
    train, dev, group_rows = group_split(v2train, cfg)
    real = real_records(registry, md["annotation_provenance_v2.json"]["records"], train)
    banks = {b: bankmd[f"formal_bank_{b}_manifest.json"] for b in BANKS}
    allowlist = synthetic_audit(banks, real, train, dev, md["normal_source_registry.json"]["records"],
                              md["source_group_registry.json"]["records"])
    official = [x for x in registry if x["asset_role"] == "detector" and x["split_v2"] == "final_eval"]
    require(not ({x["file_sha256"] for x in official} & {x["image_sha256"] for x in real}), "V2TRAIN/official content alias in metadata")
    generator_dev = [x for x in p0md["generator_train_allowlist.json"]["inputs"] if x["source_group_id"] in dev]
    insufficient = any(n < 20 for n in allowlist["eligible_per_class"].values())
    conflicts = allowlist["conditioning_dev_overlaps"]
    status = ("STAGE21B_INSUFFICIENT_TRAIN_SYNTHETIC" if insufficient else
              "STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP" if conflicts else
              "STAGE21B_METADATA_PREPARATION_COMPLETE_RUNTIME_NOT_IMPLEMENTED")
    output = Path(a.output)
    require(not output.exists(), "STAGE21B_OUTPUT_EXISTS_USE_NEW_AUDIT_DIRECTORY: " + str(output))
    output.mkdir(parents=True)
    split = {"namespace": NAMESPACE, "seed": 2101, "method": cfg["split_method"],
             "group_cutoff": len(train), "group_count": len(train | dev), "no_rebalance": True,
             "groups": group_rows, "records": real,
             "image_counts": dict(Counter(r["internal_split"] for r in real)),
             "annotation_class_counts": {s: dict(Counter(c for r in real if r["internal_split"] == s for c in r["class_instances"]))
                                         for s in ("internal_train", "internal_dev")},
             "physical_source_complete": False, "isolation_scope": "frozen known source groups, not unknown physical ancestry"}
    save(output / "stage21b_split_manifest.json", split)
    allowlist["STAGE21B_EXECUTION_AUTHORIZED"] = False
    save(output / "stage21b_synthetic_allowlist.json", allowlist)
    isolation = {"status": "FAIL_CONDITIONING_BACKGROUND_OVERLAP" if conflicts else "KNOWN_GROUP_ISOLATION_PASS",
                 "detector_train_dev_group_intersection": [], "synthetic_donor_dev_group_intersection": [],
                 "conditioning_background_dev_overlaps": conflicts,
                 "additional_slots_removed": 0, "split_modified_after_counts": False,
                 "generator_training_already_saw_internal_dev": bool(generator_dev),
                 "generator_internal_dev_input_count": len(generator_dev),
                 "generator_internal_dev_per_class": dict(Counter(x["class"] for x in generator_dev)),
                 "generator_dev_input_annotation_ids": [x["annotation_id"] for x in generator_dev],
                 "interpretation_limit": "Even if conditioning conflicts are separately resolved, this would be a detector-holdout paired timing test conditional on frozen generators, not a generator-independent holdout.",
                 "official_final_eval_content_access_count": 0}
    save(output / "stage21b_split_isolation_audit.json", isolation)
    u, c = schedules([r["role_id"] for r in real if r["internal_split"] == "internal_train"],
                     [r["role_id"] for r in allowlist["records"]], cfg)
    pairing, equality = schedule_audit(u, c, cfg)
    save(output / "uniform_schedule.json", u, compact=True)
    save(output / "curriculum_schedule.json", c, compact=True)
    save(output / "schedule_pairing_audit.json", pairing)
    save(output / "exposure_equality_audit.json", equality)
    frozen_protocol = p0md["stage20b_detector_protocol.json"]
    save(output / "stage20b_detector_protocol_snapshot.json", frozen_protocol)
    save(output / "source_evidence_index.json", {"source_commit": "cc20872afd191c59616c2ab230e58e69c6fba157",
         "protocol_sha256": sha(a.protocol), "records": evidence,
         "Stage20B_final_eval_metric_files_read": 0, "Stage20B_heavy_ledger_traversed": False})
    current = status_record(status)
    current.update(eligible_per_class=allowlist["eligible_per_class"], conditioning_conflict_slots=len(conflicts),
                   split_manifest_sha256=sha(output / "stage21b_split_manifest.json"),
                   train_draws_per_arm=u["scheduled_draws"], scheduled_optimizer_attempts_per_arm=u["scheduled_optimizer_attempts"],
                   requires_user_direction=bool(conflicts), server_training_runtime_implemented=False,
                   scope=a.scope, protocol_sha256=sha(a.protocol))
    save(output / "stage21b_status.json", current)
    try:
        assets = verify_assets(allowlist, a.stage20b, real, a.real_root)
        save(output / "source_asset_hash_audit.json", assets)
    except AuditStop as error:
        save(output / "source_asset_hash_audit.json", {"status": "ASSET_HASH_VERIFICATION_FAILED", "detail": str(error),
             "official_final_eval_content_read": 0})
        current.update(status="STAGE21B_SOURCE_ASSETS_UNAVAILABLE_OR_MODIFIED", detail=str(error))
        save(output / "stage21b_status.json", current)
    report(output, current, split, allowlist, isolation, equality)
    files = sorted(p.name for p in output.iterdir() if p.is_file())
    save(output / "stage21b_artifact_manifest.json", {"sha256": {n: sha(output / n) for n in files},
         "status": current["status"], "experiment_complete": False, "no_overwrite": True})
    print(json.dumps(current, indent=2, ensure_ascii=False))
    return 2 if not current["STAGE21B_EXECUTION_AUTHORIZED"] else 0


def report(out, status, split, allowlist, isolation, equality):
    counts = allowlist["eligible_per_class"]
    totals = equality["uniform"]
    overlaps = allowlist["conditioning_dev_overlaps"]
    conflict_counts = Counter(r["class"] for r in overlaps)
    remaining = {c: n - conflict_counts[c] for c, n in counts.items()}
    lines = ["# Stage21B — Train-only Synthetic Exposure Timing Premise Test", "",
        "状态：`" + status["status"] + "`。入口审计完成，六个 detector **均未启动**；这不是 premise PASS/FAIL。", "",
        "## 固定划分及资产", "",
        f"仅 V2 TRAIN136。namespace `{NAMESPACE}`、seed2101；134 个已知来源组按 SHA256 排序，前 floor(80%×134)=107 组训练，其余27组内部评估。",
        f"internal_train={split['image_counts']['internal_train']}，internal_dev={split['image_counts']['internal_dev']}；无重平衡或改 seed。",
        f"按 donor-only 规则保留 {sum(counts.values())} 个 slot：flash={counts['flash']}，black={counts['black']}。三 bank 相同 IDs；其余15个 donor-dev slot 三 bank 一起排除。", "",
        "## 需要先解决的来源冲突", "",
        f"{len(overlaps)} 个 donor 合格 slot 的 conditioning normal 与 internal_dev detector 图属于同一冻结 known source group。",
        "此证据来自 Stage20A normal/source-group registry 与 Stage20B frozen_slot；未读取官方 final_eval 图、标签或指标。",
        "不能将 donor-only 隔离写成完整的 detector train/dev 来源隔离。未擅自扩展排除、重新生成、更换背景或修改 split。", "",
        "| slot_id | class | normal/dev group | internal_dev relative |", "|---|---|---|---|"]
    for row in overlaps:
        relatives = ", ".join(r["image_relative_path"] for r in row["internal_dev_relatives"])
        lines.append(f"| `{row['slot_id']}` | {row['class']} | `{row['normal_source_group']}` | {relatives} |")
    lines += ["", "需要新的明确授权：是否在三个 bank 同步增加 **conditioning group 不得属于 internal_dev** 的纯泄漏排除规则？",
        f"如果授权，仅这{len(overlaps)}个 slot 额外排除，预计 flash={remaining['flash']}、black={remaining['black']}，仍≥20；目前**没有执行这项额外排除**。",
        "不调整现有 split/seed，不补样本，不按质量删样本。批准后应建立新的 allowlist/schedule 版本，保留本次审计。", "",
        "## 另一个解释限制", "",
        f"原 frozen generator 已使用其中 {isolation['generator_internal_dev_input_count']} 个 internal_dev 所属 TRAIN 输入（按生成器 annotation 输入计数）：{isolation['generator_internal_dev_per_class']}。",
        "因此未来只能声称固定 generator 条件下的 detector 内部留出 timing premise；不能声称完整生成—检测管线均未见过 internal_dev。此共同历史暴露不能通过仅排除 donor 消除。", "",
        "## 当前 donor-only schedule 数学审计（禁止执行）", "",
        f"每 epoch {u_draws(totals)} draws，150 epochs；每个 real/synthetic role 均150次；每 arm real={totals['total_real_draws']}、synthetic={totals['total_synthetic_draws']}、total={totals['total_draws']}。",
        f"Uniform 与 Curriculum occurrence multiset 相同；前50 epochs synthetic=0，其余100 epochs分摊。每 epoch accumulation=[64,64,46]，共 {totals['optimizer_attempts']} attempts。",
        "paired augmentation seed 仅依赖 role identity、occurrence index、seed42；不依赖 epoch/position。仅种子数学配对已审计，实际 augmentation tensor 配对与 GPU 运行均 NOT_RUN。",
        "保持 Stage20B detector 超参数快照。随内部训练集变小，draws/epoch、tail-group 长度与原216-role协议不同；这是划分的必要算术结果，不是超参数搜索。",
        "实际 successful AMP updates 尚未知；不通过追加 steps 强行匹配 compute。", "",
        "## 未执行与输出边界", "",
        "没有六 arm metrics/headroom/classwise/variance 文件：没有训练或内部评估，不能伪造结果。",
        "Gate_A/B/C/D=NOT_RUN；所有 delta/std=null。未实现/启动服务器训练 runtime，不能用本目录 schedule 直接开训。",
        "official final_eval content/forward=0；internal_dev forward=0；generation=0；detector training=0；optimizer steps=0；BootstrapGuard=0；DeepPCB=0。",
        "Stage20A/Stage20B 历史文件未修改。未自动继续 Stage21C。", ""]
    (out / "STAGE21B_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def u_draws(totals):
    return totals["total_draws"] // 150


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/plastic_bomo_stage21b.json"))
    parser.add_argument("--stage20a", type=Path, default=Path("results_for_gpt/plastic_bomo_stage20a_v2_protocol"))
    parser.add_argument("--stage20b", type=Path, default=Path("results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"))
    parser.add_argument("--p0", type=Path, default=Path("results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"))
    parser.add_argument("--real_root", type=Path, help="optional actual server V2 real_only root; only TRAIN assets hashed")
    parser.add_argument("--scope", choices=("LOCAL_PREPARATION", "SERVER_ENTRY_AUDIT"), default="LOCAL_PREPARATION")
    parser.add_argument("--output", type=Path, default=Path("results_for_gpt/plastic_bomo_stage21b_timing_premise"))
    args = parser.parse_args()
    try:
        return prepare(args)
    except (AuditStop, OSError, KeyError, ValueError) as error:
        print(json.dumps({"status": "STAGE21B_ENTRY_AUDIT_FAILED", "detail": str(error),
                          "STAGE21B_EXECUTION_AUTHORIZED": False, "training_started": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
