"""Frozen-pool DWBG-v2 final selection helpers.

This module intentionally only consumes scored candidates.  It neither calls
diffusion nor YOLO, so it can be used for the final audit without changing the
candidate pool.
"""
from __future__ import annotations

import math
import random
from collections import Counter


def parent_source_id(item):
    """The MSDF output, not its composition background, defines a parent."""
    return str(item.get("source_image") or item.get("reference_image"))


def hard_valid(item, class_id, cfg):
    if not item.get("manifold_valid", False):
        return False
    if float(item.get("median_iou", 0.0)) < 0.5:  # fixed final definition
        return False
    return class_id != 0 or float(item.get("median_confidence", 0.0)) >= float(cfg["flash"]["flash_conf_floor"])


def quota_requirements(class_id, hard_count, cfg):
    policy = cfg["flash" if class_id == 0 else "black"]
    return {(dimension, name): int(math.ceil(hard_count * float(ratio)))
            for dimension, bins in policy["marginal_quotas"].items()
            for name, ratio in bins.items()}


def quota_counts(items, requirements):
    return {key: sum(x.get(f"{key[0]}_bin") == key[1] for x in items)
            for key in requirements}


def _allowed(item, selected, max_per_source, max_per_seed):
    parent = parent_source_id(item)
    return (sum(parent_source_id(x) == parent for x in selected) < max_per_source and
            sum(int(x.get("seed", -1)) == int(item.get("seed", -2)) for x in selected) < max_per_seed)


def _candidate_gain(item, selected, requirements, utility):
    counts = quota_counts(selected, requirements)
    gain = 0
    for key, required in requirements.items():
        if item.get(f"{key[0]}_bin") == key[1] and counts[key] < required:
            # Scarcer and overlapping quota candidates are selected first.
            gain += 10.0 + (required - counts[key]) / max(required, 1)
    return gain + float(item.get(utility, 0.0))


def constrained_hard_selection(pool, count, class_id, cfg, utility="final_score", trials=160, seed=42):
    """Find an actual quota/cap-valid subset through deterministic multi-start greedy search.

    A returned subset is a constructive feasibility proof.  The pool has only
    three marginal quota dimensions; repeated varied ordering avoids treating
    independent marginal counts as a feasibility certificate.
    """
    requirements = quota_requirements(class_id, count, cfg)
    caps = cfg["selection"]
    valid = [dict(x, parent_source_id=parent_source_id(x)) for x in pool if hard_valid(x, class_id, cfg)]
    rng = random.Random(seed + class_id * 100003)
    best, best_key = None, None
    for _ in range(trials):
        selected = []
        counts = {key: 0 for key in requirements}
        source_counts, seed_counts = Counter(), Counter()
        # Random tie noise changes which overlapping quota combination is tried;
        # utility remains the dominant criterion after quota coverage.
        remaining = list(valid)
        while len(selected) < count:
            allowed = [x for x in remaining if source_counts[parent_source_id(x)] < caps["max_per_source"] and seed_counts[int(x.get("seed", -1))] < caps["max_per_seed"]]
            if not allowed:
                break
            def rank(x):
                gain = sum(10.0 + (required-counts[key])/max(required,1)
                           for key,required in requirements.items()
                           if x.get(f"{key[0]}_bin")==key[1] and counts[key] < required)
                return gain + float(x.get(utility, 0.0)) + rng.random()*1e-5
            choice = max(allowed, key=lambda x: (rank(x), x["candidate_id"]))
            selected.append(choice)
            source_counts[parent_source_id(choice)] += 1
            seed_counts[int(choice.get("seed", -1))] += 1
            for key in counts:
                counts[key] += int(choice.get(f"{key[0]}_bin")==key[1])
            remaining.remove(choice)
        if len(selected) != count:
            continue
        deficit = sum(max(0, requirements[k] - counts[k]) for k in requirements)
        quality = sum(float(x.get(utility, 0.0)) for x in selected)
        key = (deficit == 0, -deficit, quality)
        if best_key is None or key > best_key:
            best, best_key = selected, key
        if deficit == 0:
            # Continue enough trials to retain a high-utility valid set.
            pass
    return best or [], requirements, bool(best_key and best_key[0])


def strict_anchor_pool(rows, class_id, original_only=True):
    out = [x for x in rows if x.get("manifold_valid", False) and float(x.get("median_iou", 0)) >= .5
           and float(x.get("median_confidence", 0)) >= .25]
    if original_only and class_id == 0:
        out = [x for x in out if not x.get("target_expansion", False)]
    return out


def choose_anchors(rows, selected_hard, count, class_id, cfg):
    """Strict anchors first, then the specified two-level fallbacks."""
    caps = cfg["selection"]
    used_ids = {x["candidate_id"] for x in selected_hard}
    selected = []
    levels = [
        ("strict", lambda x: x.get("manifold_valid", False) and float(x.get("median_iou", 0)) >= .5 and float(x.get("median_confidence", 0)) >= .25),
        ("level1_conf", lambda x: x.get("manifold_valid", False) and float(x.get("median_iou", 0)) >= .5 and float(x.get("median_confidence", 0)) >= .15),
        ("level2_iou", lambda x: x.get("manifold_valid", False) and float(x.get("median_iou", 0)) >= .4 and float(x.get("median_confidence", 0)) >= .15 and float(x.get("consensus_score", 0)) >= float(caps["anchor_min_consensus"])),
        ("hard_pool", lambda x: hard_valid(x, class_id, cfg)),
    ]
    for reason, predicate in levels:
        pool = [x for x in rows if x["candidate_id"] not in used_ids and predicate(x)]
        if class_id == 0 and reason == "strict":
            original = [x for x in pool if not x.get("target_expansion", False)]
            if len(original) >= count - len(selected): pool = original
        pool.sort(key=lambda x: (float(x.get("anchor_score", 0)), x["candidate_id"]), reverse=True)
        for item in pool:
            if len(selected) >= count: break
            if not _allowed(item, selected_hard + selected, caps["max_per_source"], caps["max_per_seed"]): continue
            chosen = dict(item); chosen["fallback_reason"] = None if reason == "strict" else reason
            selected.append(chosen); used_ids.add(chosen["candidate_id"])
        if len(selected) >= count: break
    return selected


def diversity_report(items):
    return {
        "selected_candidates": len(items),
        "unique_parent_sources": len({parent_source_id(x) for x in items}),
        "unique_backgrounds": len({str(x.get("background", "")) for x in items}),
        "unique_seeds": len({int(x.get("seed", -1)) for x in items}),
        "max_candidates_from_one_parent_source": max(Counter(parent_source_id(x) for x in items).values(), default=0),
    }
