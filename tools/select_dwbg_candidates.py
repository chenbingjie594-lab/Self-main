"""Create leakage-free Real-only, Random, and DWBG-selected YOLO datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from .dwbg_utils import native
except ImportError:
    from dwbg_utils import native


CLASS_NAMES = {0: "Flash point", 1: "Big black spots"}
POLICY_NAMES = {0: "flash", 1: "black"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--scored_candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dwbg_stage2.json"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--flash_count", type=int, required=True)
    parser.add_argument("--black_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def interval_boundary_score(confidence, iou, low, high, min_iou):
    """Hard-but-valid Flash score: exactly one inside interval, decays outside."""
    if float(iou) < float(min_iou):
        return 0.0
    confidence = float(confidence)
    if low <= confidence <= high:
        return 1.0
    if confidence < low:
        # Below-boundary samples become invalid quickly rather than being rewarded.
        return float(max(0.0, confidence / max(low, 1e-8)) ** 2)
    return float(max(0.0, 1.0 - (confidence - high) / max(1.0 - high, 1e-8)))


def moderate_difficulty_score(confidence, iou, target, sigma, min_iou):
    confidence, iou = float(confidence), float(iou)
    bell = math.exp(-0.5 * ((confidence - float(target)) / max(float(sigma), 1e-8)) ** 2)
    return float(bell * min(max(iou / max(float(min_iou), 1e-8), 0.0), 1.0))


def weakness_lookup(profile, class_id, dimension, name):
    rows = profile["class_profiles"][str(class_id)][dimension]
    for row in rows:
        if row["name"] == name:
            return float(row["weakness"])
    return 0.0


def score_candidate(candidate, profile, config):
    """Class-specific utility; intentionally not a shared cross-class formula."""
    class_id = int(candidate["class_id"])
    policy = config[POLICY_NAMES[class_id]]
    scale = weakness_lookup(profile, class_id, "scale", candidate["scale_bin"])
    contrast = weakness_lookup(profile, class_id, "contrast", candidate["contrast_bin"])
    morphology = weakness_lookup(profile, class_id, "morphology", candidate["morphology_bin"])
    weights = policy["weights"]
    reasons = []
    if class_id == 0:
        boundary = interval_boundary_score(candidate["median_confidence"], candidate["median_iou"],
                                           policy["flash_conf_low"], policy["flash_conf_high"], policy["min_iou"])
        weakness = weights["scale"] * scale + weights["contrast"] * contrast + weights["morphology"] * morphology
        final = weights["boundary"] * boundary + weakness
        if boundary > 0:
            reasons.append("flash_boundary")
        if candidate["scale_bin"] == "large":
            reasons.append("large_scale")
        if candidate["contrast_bin"] == "high":
            reasons.append("high_contrast")
        if candidate["morphology_bin"] == "compact":
            reasons.append("compact_morphology")
        return {"scale_weakness": scale, "contrast_weakness": contrast, "morphology_weakness": morphology,
                "boundary_score": boundary, "detector_difficulty_score": None,
                "weakness_score": weakness, "final_score": final, "selection_reason": reasons}
    difficulty = moderate_difficulty_score(candidate["median_confidence"], candidate["median_iou"],
                                            policy["difficulty_target"], policy["difficulty_sigma"], policy["min_iou"])
    weakness = weights["scale"] * scale + weights["contrast"] * contrast + weights["morphology"] * morphology
    final = weakness + weights["detector_difficulty"] * difficulty
    if candidate["scale_bin"] == "tiny":
        reasons.append("tiny_scale")
    if candidate["contrast_bin"] == "low":
        reasons.append("low_contrast")
    if candidate["morphology_bin"] == "elongated":
        reasons.append("elongated_morphology")
    if difficulty > 0.5:
        reasons.append("moderate_detector_difficulty")
    return {"scale_weakness": scale, "contrast_weakness": contrast, "morphology_weakness": morphology,
            "boundary_score": None, "detector_difficulty_score": difficulty,
            "weakness_score": weakness, "final_score": final, "selection_reason": reasons}


def quota_bonus(candidate, selected, target_count, policy, config):
    bonus, satisfied = 0.0, []
    quota_bonus = float(config["selection"]["quota_bonus"])
    for dimension, requirements in policy["marginal_quotas"].items():
        name = candidate[f"{dimension}_bin"]
        ratio = requirements.get(name)
        if ratio is None:
            continue
        needed = max(1, int(math.ceil(target_count * float(ratio))))
        current = sum(item[f"{dimension}_bin"] == name for item in selected)
        if current < needed:
            bonus += quota_bonus * (needed - current) / needed
            satisfied.append(f"quota_{dimension}_{name}")
    return bonus, satisfied


def diversity_bonus(candidate, selected, config):
    if not selected:
        return sum(float(config["selection"][key]) for key in (
            "source_diversity_bonus", "seed_diversity_bonus", "geometry_diversity_bonus"))
    source_seen = {item["reference_image"] for item in selected}
    seed_seen = {item["seed"] for item in selected}
    geometry_seen = {(item["scale_bin"], item["contrast_bin"], item["morphology_bin"]) for item in selected}
    bonus = 0.0
    if candidate["reference_image"] not in source_seen:
        bonus += float(config["selection"]["source_diversity_bonus"])
    if candidate["seed"] not in seed_seen:
        bonus += float(config["selection"]["seed_diversity_bonus"])
    geometry = (candidate["scale_bin"], candidate["contrast_bin"], candidate["morphology_bin"])
    if geometry not in geometry_seen:
        bonus += float(config["selection"]["geometry_diversity_bonus"])
    return bonus


def select_dwbg(candidates, count, profile, config):
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)}")
    class_id = int(candidates[0]["class_id"])
    policy = config[POLICY_NAMES[class_id]]
    pool = [dict(candidate, **score_candidate(candidate, profile, config)) for candidate in candidates]
    selected = []
    while len(selected) < count:
        ranked = []
        for candidate in pool:
            if candidate.get("_selected"):
                continue
            q_bonus, q_reasons = quota_bonus(candidate, selected, count, policy, config)
            d_bonus = diversity_bonus(candidate, selected, config)
            gain = float(candidate["final_score"] + q_bonus + d_bonus)
            ranked.append((gain, candidate["candidate_id"], candidate, q_reasons))
        if not ranked:
            raise RuntimeError("DWBG pool exhausted during selection")
        _, _, chosen, q_reasons = max(ranked, key=lambda item: (item[0], item[1]))
        chosen["_selected"] = True
        chosen["selection_reason"] = chosen["selection_reason"] + q_reasons
        selected.append(chosen)
    for item in selected:
        item.pop("_selected", None)
    return selected


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_real(real_root, destination):
    for split in ("train", "val"):
        for image in sorted((real_root / "images" / split).iterdir()):
            if image.is_file():
                link_or_copy(image, destination / "images" / split / image.name)
                label = real_root / "labels" / split / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(f"Missing real label: {label}")
                link_or_copy(label, destination / "labels" / split / label.name)


def validation_hash(root):
    items = []
    for kind in ("images", "labels"):
        folder = root / kind / "val"
        for path in sorted(folder.iterdir()):
            if path.is_file():
                items.append((f"{kind}/{path.name}", hashlib.sha256(path.read_bytes()).hexdigest()))
    return items


def write_yaml(root, group):
    (root / f"{group}.yaml").write_text(
        f"path: {(root / group).resolve().as_posix()}\ntrain: images/train\nval: images/val\n\n"
        "names:\n  0: Flash point\n  1: Big black spots\n", encoding="utf-8")


def add_synthetic(selected, destination):
    for item in selected:
        image = Path(item["image_path"])
        label = Path(item["label_path"])
        name = f"{item['candidate_id']}.jpg"
        link_or_copy(image, destination / "images" / "train" / name)
        link_or_copy(label, destination / "labels" / "train" / f"{Path(name).stem}.txt")


def distribution(items, class_id):
    subset = [item for item in items if int(item["class_id"]) == class_id]
    total = max(len(subset), 1)
    output = {}
    for dimension in ("scale", "contrast", "morphology"):
        count = Counter(item[f"{dimension}_bin"] for item in subset)
        output[dimension] = {name: {"count": int(value), "ratio": float(value / total)} for name, value in sorted(count.items())}
    output["confidence"] = {"mean": float(np.mean([i["median_confidence"] for i in subset])) if subset else 0.0,
                            "median": float(np.median([i["median_confidence"] for i in subset])) if subset else 0.0,
                            "p25": float(np.quantile([i["median_confidence"] for i in subset], .25)) if subset else 0.0,
                            "p75": float(np.quantile([i["median_confidence"] for i in subset], .75)) if subset else 0.0}
    output["iou"] = {"mean": float(np.mean([i["median_iou"] for i in subset])) if subset else 0.0,
                      "median": float(np.median([i["median_iou"] for i in subset])) if subset else 0.0}
    return output


def print_summary(random_items, dwbg_items):
    print("=" * 50 + "\nDWBG Stage-2 Selection Summary\n" + "=" * 50)
    for class_id in (0, 1):
        print(f"\n{CLASS_NAMES[class_id]}")
        for name, items in (("Random", random_items), ("DWBG", dwbg_items)):
            d = distribution(items, class_id)
            print(f"{name}: n={sum(int(x['class_id']) == class_id for x in items)} "
                  f"conf(median)={d['confidence']['median']:.3f} IoU(median)={d['iou']['median']:.3f}")
            print("  " + " | ".join(f"{dim}: " + ", ".join(f"{k}={v['ratio']:.0%}" for k, v in d[dim].items())
                                     for dim in ("scale", "contrast", "morphology")))
        if class_id == 0:
            boundary = sum(item.get("boundary_score", 0) == 1.0 for item in dwbg_items if int(item["class_id"]) == 0)
            total = max(1, sum(int(item["class_id"]) == 0 for item in dwbg_items))
            print(f"DWBG Flash boundary candidates: {boundary}/{total} ({boundary / total:.0%})")
    print("=" * 50)


def main():
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {args.output_root}")
    if min(args.flash_count, args.black_count) < 1:
        raise ValueError("Synthetic class counts must be positive")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    scored = json.loads(args.scored_candidates.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = scored.get("candidates", [])
    by_class = {class_id: [c for c in candidates if int(c["class_id"]) == class_id] for class_id in (0, 1)}
    requested = {0: args.flash_count, 1: args.black_count}
    rng = random.Random(args.seed)
    random_selected, dwbg_selected = [], []
    for class_id, count in requested.items():
        if len(by_class[class_id]) < count:
            raise ValueError(f"Class {class_id}: need {count} candidates, found {len(by_class[class_id])}")
        random_selected.extend(rng.sample(by_class[class_id], count))
        dwbg_selected.extend(select_dwbg(by_class[class_id], count, profile, config))
    args.output_root.mkdir(parents=True, exist_ok=True)
    for group in ("real_only", "real_random", "real_dwbg"):
        copy_real(args.real_root, args.output_root / group)
        write_yaml(args.output_root, group)
    add_synthetic(random_selected, args.output_root / "real_random")
    add_synthetic(dwbg_selected, args.output_root / "real_dwbg")
    baseline_hash = validation_hash(args.output_root / "real_only")
    if any(validation_hash(args.output_root / group) != baseline_hash for group in ("real_random", "real_dwbg")):
        raise RuntimeError("Validation differs between Real-only, Random, and DWBG datasets")
    common = {"version": 1, "profile": str(args.profile.resolve()), "scored_candidates": str(args.scored_candidates.resolve()),
              "counts": {"flash": args.flash_count, "black": args.black_count}, "seed": args.seed}
    (args.output_root / "random_manifest.json").write_text(json.dumps(native({**common, "method": "random", "selected": random_selected}), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "dwbg_manifest.json").write_text(json.dumps(native({**common, "method": "dwbg", "config": config, "selected": dwbg_selected}), ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(random_selected, dwbg_selected)
    print(f"DWBG datasets and manifests: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
