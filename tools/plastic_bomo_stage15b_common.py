"""Frozen selection and descriptive statistics for the Stage15B probe."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

CLASSES = ("flash", "black")
SEEDS = (42, 3407, 2026)
ARMS = ("M00", "M10", "M01", "M11", "M50")

def load(p):
    def reject(v): raise ValueError("NONFINITE_JSON: " + v)
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=reject)

def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""): h.update(part)
    return h.hexdigest()

def frozen_save(path, value):
    path = Path(path)
    if path.exists():
        if load(path) != value: raise RuntimeError("STAGE15B_FROZEN_ARTIFACT_CHANGED: " + str(path))
    else: save(path, value)

def parent(row):
    return f"{int(row['class_id'])}:{Path(row['source_image']).stem}"

def candidate_record(row, role, morphology):
    return {"candidate_id": row["candidate_id"], "class_name": CLASSES[int(row["class_id"])],
            "class_id": int(row["class_id"]), "role": role, "generation_seed": int(row["seed"]),
            "parent_id": parent(row), "image_path": row["image_path"], "label_path": row["label_path"],
            "bbox_xyxy": list(row["bbox_xyxy"]), "morphology_distance": float(morphology)}

def pick(candidates, quota, cap, used_parents, seeds):
    """Greedy round-robin in frozen metadata order; no scores or RNG."""
    remaining = {x["candidate_id"]: x for x in candidates}
    counts = Counter({str(s): 0 for s in seeds}); chosen = []
    while len(chosen) < quota:
        eligible = [x for x in remaining.values() if used_parents[parent(x)] < cap]
        if not eligible: return None
        row = min(eligible, key=lambda x: (counts[str(int(x["seed"]))], str(int(x["seed"])), x["candidate_id"]))
        chosen.append(row); counts[str(int(row["seed"]))] += 1
        used_parents[parent(row)] += 1; del remaining[row["candidate_id"]]
    return chosen

def high_pick_with_random_capacity(high, random, quota, random_quota, cap, used_parents, seeds):
    """Skip only choices making the remaining two fixed quotas infeasible.

    For two roles, the three max-flow cut conditions are exactly: capacity
    for high, capacity for random, and shared capacity for their sum.
    The ordering remains seed-count/seed/ID; no phenotype score is consulted.
    """
    remaining = {x["candidate_id"]: x for x in high}; counts = Counter({str(s): 0 for s in seeds}); chosen = []
    rc = Counter(parent(x) for x in random)
    while len(chosen) < quota:
        eligible = sorted((x for x in remaining.values() if used_parents[parent(x)] < cap),
            key=lambda x: (counts[str(int(x["seed"]))], str(int(x["seed"])), x["candidate_id"]))
        accepted = None
        for row in eligible:
            trial_used = used_parents.copy(); trial_used[parent(row)] += 1
            hc = Counter(parent(x) for cid,x in remaining.items() if cid != row["candidate_id"])
            parents = hc.keys() | rc.keys(); available = {key: max(0, cap-trial_used[key]) for key in parents}
            need_h = quota-len(chosen)-1
            if (sum(min(hc[k], available[k]) for k in parents) >= need_h and
                sum(min(rc[k], available[k]) for k in parents) >= random_quota and
                sum(min(hc[k]+rc[k], available[k]) for k in parents) >= need_h+random_quota):
                accepted = row; break
        if accepted is None: return None
        chosen.append(accepted); counts[str(int(accepted["seed"]))] += 1
        used_parents[parent(accepted)] += 1; del remaining[accepted["candidate_id"]]
    return chosen

def select_balanced(subsets, index, morphology, cfg):
    groups = {}
    for cls_id in (0, 1):
        groups[cls_id] = {role: [index[i] for i in subsets[key] if int(index[i]["class_id"]) == cls_id]
                          for role, key in (("high_morph", "high_fidelity"), ("random", "random"))}
        for values in groups[cls_id].values():
            if len(values) != 40: raise RuntimeError("STAGE15B_SOURCE_QUOTA_INVALID")
    for cap in cfg["parent_caps"]:
        chosen = []; caps_ok = True
        # One common cap for both roles and both classes, counted over the whole arm.
        used = Counter()
        for cls_id in (0, 1):
            high_ids = {x["candidate_id"] for x in groups[cls_id]["high_morph"]}
            random_available = [x for x in groups[cls_id]["random"] if x["candidate_id"] not in high_ids]
            for role, quota in (("high_morph", cfg["high_per_class"]), ("random", cfg["random_per_class"])):
                eligible = groups[cls_id][role]
                if role == "random":
                    selected = pick(random_available, quota, cap, used, cfg["generation_seeds"])
                else:
                    selected = high_pick_with_random_capacity(eligible, random_available, quota, cfg["random_per_class"], cap, used, cfg["generation_seeds"])
                if selected is None: caps_ok = False; break
                chosen.extend(candidate_record(x, role, morphology[x["candidate_id"]]) for x in selected)
            if not caps_ok: break
        if caps_ok:
            if len(chosen) != 80 or len({x["candidate_id"] for x in chosen}) != 80: raise RuntimeError("STAGE15B_DUPLICATE_CANDIDATES")
            return chosen, cap
    raise RuntimeError("STAGE15B_PARENT_CAP_INFEASIBLE_AT_2")

def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all(): raise RuntimeError("STAGE15B_INVALID_DISTRIBUTION")
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75)),
            "q90": float(np.quantile(a, .90)), "std": float(a.std()), "std_ddof": 0}

def entropy(counts):
    total = sum(counts.values())
    return float(-sum((n/total)*math.log(n/total) for n in counts.values() if n))

def source_diversity(rows):
    seeds = Counter(str(x["generation_seed"]) for x in rows); parents = Counter(x["parent_id"] for x in rows)
    return {"n": len(rows), "generation_seed_entropy": entropy(seeds), "parent_source_entropy": entropy(parents),
            "entropy_base": "natural_log", "unique_parent_count": len(parents), "max_parent_reuse": max(parents.values()),
            "generation_seed_histogram": dict(sorted(seeds.items())), "parent_histogram": dict(sorted(parents.items()))}

def feature_diversity(features, epsilon=1e-12):
    x = np.asarray(features, np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all(): raise RuntimeError("STAGE15B_INVALID_FEATURES")
    norms = np.linalg.norm(x, axis=1)
    if (norms <= epsilon).any(): raise RuntimeError("STAGE15B_ZERO_FEATURE")
    x = x / norms[:, None]; distances = np.clip(1-x@x.T, 0, 2)
    pairs = distances[np.triu_indices(len(x), 1)]
    np.fill_diagonal(distances, np.inf)
    centered = x-x.mean(0); eigenvalues = np.maximum(np.linalg.eigvalsh(centered@centered.T/(len(x)-1)), 0)
    total = eigenvalues.sum()
    if total <= epsilon: er = 0.; degenerate = True
    else:
        probabilities = eigenvalues[eigenvalues > 0]/total
        er = float(np.exp(-np.sum(probabilities*np.log(probabilities+epsilon)))); degenerate = False
    return {"n": len(x), "feature_dim": x.shape[1], "mean_pairwise_cosine_distance": float(pairs.mean()),
            "median_nearest_neighbor_distance": float(np.median(distances.min(1))),
            "covariance_effective_rank": er, "normalized_ER": er/x.shape[1],
            "degenerate_covariance": degenerate, "effective_rank_epsilon": epsilon,
            "covariance_input": "L2-normalized detector-head-input vectors", "maximum_sample_rank": len(x)-1}

def verify_preparation(root, cfg):
    root = Path(root); manifest = load(root/"balanced_morph50_manifest.json"); audit = load(root/"manifest_freeze_audit.json")
    if digest(manifest) != audit["manifest_content_sha256"]: raise RuntimeError("STAGE15B_MANIFEST_CHANGED")
    if digest(cfg) != audit["protocol_content_sha256"]: raise RuntimeError("STAGE15B_PROTOCOL_CHANGED")
    if len(manifest["records"]) != 80 or len({x["candidate_id"] for x in manifest["records"]}) != 80:
        raise RuntimeError("STAGE15B_MANIFEST_INVALID")
    return manifest

def label_dir(image_dir):
    image_dir = Path(image_dir)
    parts = list(image_dir.parts)
    if "images" not in parts: raise RuntimeError("STAGE15B_IMAGES_LAYOUT_INVALID: " + str(image_dir))
    parts[len(parts)-1-parts[::-1].index("images")] = "labels"
    return Path(*parts)

def dataset_layout(yaml_path):
    import yaml
    path = Path(yaml_path).resolve(); doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = Path(doc.get("path", str(path.parent)))
    if not base.is_absolute(): base = path.parent/base
    def resolve(key):
        value = doc[key]
        if not isinstance(value, str): raise RuntimeError("STAGE15B_REQUIRES_SINGLE_DIRECTORY_SPLIT: " + key)
        p = Path(value); return p if p.is_absolute() else base/p
    names = doc["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    if len(names) != 2 or "flash" not in str(names[0]).lower() or "black" not in str(names[1]).lower():
        raise RuntimeError("STAGE15B_CLASS_MAPPING_INVALID")
    return resolve("train"), resolve("val"), doc

def image_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

def population(images):
    return [{"name": p.name, "image_sha256": sha(p), "label_sha256": sha(label_dir(p.parent)/(p.stem+".txt"))} for p in image_files(images)]

def paired_summary(values):
    a = np.asarray(values, dtype=float)
    if len(a) != 3 or not np.isfinite(a).all(): raise RuntimeError("STAGE15B_PAIRED_VALUES_INVALID")
    return {"seed_order": list(SEEDS), "values": a.tolist(), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "mean_pp": float(a.mean()*100), "std_pp": float(a.std(ddof=1)*100),
            "wins": int((a>0).sum()), "losses": int((a<0).sum()), "ties": int((a==0).sum())}
