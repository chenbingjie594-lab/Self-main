"""Evaluate the frozen AnomalyDiffusion 240-image set under the Exp1 protocol.

The evaluator reuses Exp1 real Inception features and frozen Raw SD2/MSDF
results.  It extracts only AnomalyDiffusion full-image/ROI features and never
trains, samples, selects images, or runs a detector.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

from chapter3_exp1_quality_common import (
    BANKS, CLASSES, REPO, Stop, checked_file, csv_write, dependencies, load,
    mean_sd, portable, require, resolve, save, sha,
)
from evaluate_plastic_bomo_chapter3_exp1 import (
    heartbeat, kid_options, offline_metrics, open_rgb, runtime_modules,
    subset_coverage,
)


CONFIG = REPO / "configs/plastic_bomo_chapter3_exp2_ad_visual_eval.json"
METHOD = "AnomalyDiffusion"
ALL_METHODS = ("Raw SD2", "MSDF", METHOD)


def paths(cfg):
    return (
        resolve(cfg["exp1_output"]),
        resolve(cfg["ad_generation_root"]),
        resolve(cfg["output"]),
        resolve(cfg["protocol_source"]),
    )


def frozen_files(cfg):
    exp1, ad_root, _, _ = paths(cfg)
    return {
        "exp1_inception_features": exp1 / "inception_features.npz",
        "exp1_inception_feature_cache": exp1 / "inception_feature_cache.json",
        "exp1_visual_comparison_manifest": exp1 / "visual_comparison_manifest.json",
        "exp1_real_visual_reference_manifest": exp1 / "real_visual_reference_manifest.json",
        "exp1_generation_quality_summary": exp1 / "generation_quality_summary.json",
        "exp1_metric_runtime": exp1 / "metric_runtime.json",
        "ad_formal_synthetic_manifest": ad_root / "formal_synthetic_manifest.json",
    }


def verify_frozen_files(cfg):
    files = frozen_files(cfg)
    for name, path in files.items():
        checked_file(path, cfg["frozen_sha256"][name])
    return files


def prepare(cfg, config_path):
    files = verify_frozen_files(cfg)
    exp1, ad_root, output, protocol_path = paths(cfg)
    protocol = load(protocol_path)
    require(protocol["banks"] == list(BANKS) and protocol["classes"] == list(CLASSES),
            "PROTOCOL_CHANGED", "Exp1 banks/classes")
    require(protocol["crop"]["context_margin_each_side"] == .25,
            "PROTOCOL_CHANGED", "Exp1 ROI crop")
    pairs = load(files["exp1_visual_comparison_manifest"])
    reference = load(files["exp1_real_visual_reference_manifest"])
    ad_manifest = load(files["ad_formal_synthetic_manifest"])
    require(ad_manifest["status"] == "FROZEN" and ad_manifest["method"] == METHOD,
            "AD_MANIFEST_INVALID", ad_manifest.get("status"))
    require(ad_manifest["count"] == 240 and ad_manifest["bank_counts"] == {b: 80 for b in BANKS},
            "AD_MANIFEST_INVALID", "240 images / 80 per bank required")
    require(ad_manifest["class_counts"] == {"black": 120, "flash": 120}
            and not ad_manifest["quality_selection"] and ad_manifest["reroll_count"] == 0,
            "AD_MANIFEST_INVALID", "class counts, selection, or reroll")

    pair_map = {(r["bank"], r["slot_id"]): r for r in pairs["records"]}
    ad_map = {(r["bank"], r["slot_id"]): r for r in ad_manifest["records"]}
    require(len(pair_map) == len(ad_map) == 240 and set(pair_map) == set(ad_map),
            "SLOT_ALIGNMENT_INVALID", "Exp1 and AnomalyDiffusion bank/slot sets differ")
    records = []
    for pair in pairs["records"]:
        row = ad_map[(pair["bank"], pair["slot_id"])]
        require(row["class"] == pair["class"]
                and row["frozen_slot"]["bbox"] == pair["bbox"]
                and row["frozen_slot_sha256"] == pair["slot_metadata_sha256"],
                "SLOT_ALIGNMENT_INVALID", pair["slot_id"])
        image = (ad_root / row["image_path"]).resolve()
        require(image.is_relative_to(ad_root), "INPUT_PATH_INVALID", row["image_path"])
        checked_file(image, row["image_sha256"])
        records.append({
            "bank": pair["bank"], "slot_id": pair["slot_id"], "class": pair["class"],
            "image_path": portable(image), "image_sha256": row["image_sha256"],
            "canvas_wh": pair["canvas_wh"], "bbox": pair["bbox"],
            "roi_xyxy": pair["roi_xyxy"], "frozen_slot_sha256": row["frozen_slot_sha256"],
        })
    require(len(records) == 240, "SLOT_ALIGNMENT_INVALID", len(records))

    output.mkdir(parents=True, exist_ok=True)
    prepared = {
        "status": "FROZEN", "method": METHOD, "count": 240,
        "bank_counts": {b: sum(r["bank"] == b for r in records) for b in BANKS},
        "class_counts": {c: sum(r["class"] == c for r in records) for c in CLASSES},
        "same_slots_as_exp1": True, "no_filtering": True, "no_reroll": True,
        "records": records,
    }
    prepared_path = output / "anomalydiffusion_evaluation_manifest.json"
    if prepared_path.exists():
        require(load(prepared_path) == prepared, "PREPARED_INPUT_CHANGED", str(prepared_path))
    else:
        save(prepared_path, prepared)
    binding = {
        "status": "PREPARED", "config": portable(config_path),
        "config_sha256": sha(config_path), "protocol": portable(protocol_path),
        "protocol_sha256": sha(protocol_path),
        "implementation_sha256": {
            portable(Path(__file__)): sha(__file__),
            "tools/evaluate_plastic_bomo_chapter3_exp1.py": sha(Path(__file__).with_name("evaluate_plastic_bomo_chapter3_exp1.py")),
            "tools/chapter3_exp1_quality_common.py": sha(Path(__file__).with_name("chapter3_exp1_quality_common.py")),
        },
        "frozen_files": {portable(p): sha(p) for p in files.values()},
        "prepared_manifest_sha256": sha(prepared_path),
        "exp1_real_feature_count_reused": 108,
        "anomalydiffusion_feature_count_to_extract": 480,
        "generator_training_count": 0, "generation_count": 0,
        "detector_evaluation_count": 0,
    }
    binding_path = output / "evaluation_binding.json"
    if binding_path.exists():
        require(load(binding_path) == binding, "PREPARED_INPUT_CHANGED", str(binding_path))
    else:
        save(binding_path, binding)
    status_path = output / "evaluation_status.json"
    if not status_path.exists() or not load(status_path).get("complete"):
        save(status_path, {
            "status": "PREPARED", "complete": False, "method": METHOD,
            "training_count": 0, "generation_count": 0, "detector_evaluation_count": 0,
        })
    return pairs, reference, prepared, protocol


def verify_prepared(cfg, config_path):
    files = verify_frozen_files(cfg)
    exp1, _, output, protocol_path = paths(cfg)
    binding = load(output / "evaluation_binding.json")
    require(binding["config_sha256"] == sha(config_path)
            and binding["protocol_sha256"] == sha(protocol_path),
            "PREPARED_INPUT_CHANGED", "config/protocol")
    expected_implementation = {
        portable(Path(__file__)): sha(__file__),
        "tools/evaluate_plastic_bomo_chapter3_exp1.py": sha(Path(__file__).with_name("evaluate_plastic_bomo_chapter3_exp1.py")),
        "tools/chapter3_exp1_quality_common.py": sha(Path(__file__).with_name("chapter3_exp1_quality_common.py")),
    }
    require(binding["implementation_sha256"] == expected_implementation,
            "EVALUATION_CODE_CHANGED", "frozen evaluator implementation")
    for name, digest in binding["frozen_files"].items():
        checked_file(resolve(name), digest)
    checked_file(output / "anomalydiffusion_evaluation_manifest.json",
                 binding["prepared_manifest_sha256"])
    prepared = load(output / "anomalydiffusion_evaluation_manifest.json")
    for row in prepared["records"]:
        checked_file(resolve(row["image_path"]), row["image_sha256"])
    return (
        load(files["exp1_visual_comparison_manifest"]),
        load(files["exp1_real_visual_reference_manifest"]), prepared, load(protocol_path),
    )


def ad_items(prepared):
    items = []
    for r in prepared["records"]:
        for region in ("full_image", "roi"):
            items.append({
                "key": f'ad:{region}:{r["bank"]}:{r["slot_id"]}',
                "region": region, "role": METHOD, "bank": r["bank"],
                "slot_id": r["slot_id"], "classes": [r["class"]],
                "path": r["image_path"], "canvas_wh": r["canvas_wh"],
                "crop": r["roi_xyxy"] if region == "roi" else None,
            })
    require(len(items) == 480 and len({r["key"] for r in items}) == 480,
            "FEATURE_POPULATION_INVALID", "480 unique AD full/ROI inputs")
    return items


def real_items(reference):
    result = []
    for r in reference["images"]:
        result.append({"key": "real:full:" + r["node_id"], "region": "full_image",
                       "role": "real", "classes": r["classes"]})
    for r in reference["instances"]:
        result.append({"key": "real:roi:" + r["annotation_id"], "region": "roi",
                       "role": "real", "classes": [r["class"]]})
    require(len(result) == 108, "REFERENCE_POPULATION_INVALID", len(result))
    return result


def load_exp1_features(cfg, lib):
    exp1, _, _, _ = paths(cfg)
    ledger = load(exp1 / "inception_feature_cache.json")
    require(ledger["feature_count"] == 1068 and ledger["dimension"] == 2048,
            "FEATURE_CACHE_BINDING_CHANGED", "Exp1 feature ledger")
    with lib["np"].load(exp1 / "inception_features.npz", allow_pickle=False) as saved:
        array = saved["features"].copy()
    require(array.shape == (1068, 2048) and bool(lib["np"].isfinite(array).all()),
            "FEATURE_CACHE_BINDING_CHANGED", "Exp1 feature matrix")
    return {key: lib["torch"].from_numpy(array[i]) for i, key in enumerate(ledger["keys"])}


def extract_ad_features(output, items, lib, runtime):
    torch, np, Image = lib["torch"], lib["np"], lib["Image"]
    cache = output / "anomalydiffusion_inception_features.npz"
    ledger = output / "anomalydiffusion_inception_feature_cache.json"
    keys = [r["key"] for r in items]
    if ledger.exists():
        meta = load(ledger)
        checked_file(cache, meta["sha256"])
        require(meta["keys"] == keys and meta["metric_runtime_sha256"] == sha(output / "metric_runtime.json"),
                "FEATURE_CACHE_BINDING_CHANGED", str(cache))
        with np.load(cache, allow_pickle=False) as saved:
            array = saved["features"].copy()
        print("[复用] AnomalyDiffusion 480 份 Inception 特征缓存。", flush=True)
    else:
        model = lib["Inception"](
            name="inception-v3-compat", features_list=["2048"],
            feature_extractor_weights_path=runtime["weights"]["inception"]["path"],
        ).float().eval().requires_grad_(False).to(runtime["device"])
        values = []
        with torch.inference_mode(), heartbeat("AnomalyDiffusion 标准 Inception 特征提取；不训练、不生成"):
            for i, item in enumerate(items):
                image = open_rgb(item, Image)
                data = np.asarray(image, dtype=np.uint8).copy()
                x = torch.from_numpy(data).permute(2, 0, 1)[None].to(runtime["device"])
                value = model(x)[0]
                require(tuple(value.shape) == (1, 2048) and bool(torch.isfinite(value).all()),
                        "NONFINITE_INCEPTION_FEATURE", item["key"])
                values.append(value[0].double().cpu().numpy())
                if (i + 1) % 40 == 0 or i + 1 == len(items):
                    print(f"[Inception] {i+1}/{len(items)} 个 AD full/ROI 输入。", flush=True)
        array = np.stack(values)
        np.savez_compressed(cache, features=array)
        save(ledger, {
            "status": "COMPLETE", "sha256": sha(cache), "keys": keys,
            "feature_count": 480, "dimension": 2048,
            "metric_runtime_sha256": sha(output / "metric_runtime.json"),
            "only_anomalydiffusion_new_features": True,
        })
        del model
        if runtime["device"].startswith("cuda:"):
            torch.cuda.empty_cache()
    require(array.shape == (480, 2048) and bool(np.isfinite(array).all()),
            "FEATURE_CACHE_BINDING_CHANGED", "AD feature matrix")
    return {key: torch.from_numpy(array[i]) for i, key in enumerate(keys)}


def distribution_metrics(output, real, generated, features, protocol, lib, runtime):
    torch, np = lib["torch"], lib["np"]
    partial_path = output / "anomalydiffusion_distribution_metrics_raw.json"
    expected = [(region, cohort, bank) for region in ("full_image", "roi")
                for cohort in ("overall", *CLASSES) for bank in BANKS]
    records = []
    if partial_path.exists():
        partial = load(partial_path)
        records = partial.get("records", []) if partial.get("status") in ("RUNNING", "COMPLETE") else []
        actual = [(r.get("region"), r.get("cohort"), r.get("bank")) for r in records]
        require(actual == expected[:len(actual)] and all(r.get("method") == METHOD for r in records),
                "PARTIAL_METRICS_INVALID", actual)
        if records:
            print(f"[断点续跑] 复用已完成的 {len(records)}/18 组 FID/KID。", flush=True)
    completed = {(r["region"], r["cohort"], r["bank"]) for r in records}
    sampling = {}
    for region in ("full_image", "roi"):
        for cohort in ("overall", *CLASSES):
            actual = [r for r in real if r["region"] == region
                      and (cohort == "overall" or cohort in r["classes"])]
            real_features = torch.stack([features[r["key"]] for r in actual]).double().cpu()
            print(f"[FID/KID] {region}/{cohort}，复用固定真实特征 {len(actual)}。", flush=True)
            pending = [bank for bank in BANKS if (region, cohort, bank) not in completed]
            real_stats = lib["fid_stats"](real_features) if pending else None
            for bank in BANKS:
                fake = [r for r in generated if r["region"] == region and r["bank"] == bank
                        and (cohort == "overall" or cohort in r["classes"])]
                require(len(fake) == (80 if cohort == "overall" else 40),
                        "BANKWISE_RESULTS_INCOMPLETE", (region, cohort, bank))
                fake_features = torch.stack([features[r["key"]] for r in fake]).double().cpu()
                options = kid_options(protocol, runtime["package_versions"]["torch-fidelity"],
                                      len(actual), len(fake))
                coverage = subset_coverage(np, len(actual), len(fake), options)
                sampling[f"{region}/{cohort}/{bank}"] = {
                    "real_keys": [r["key"] for r in actual],
                    "generated_slot_ids": [r["slot_id"] for r in fake],
                    "options": options, "coverage": coverage,
                }
                if (region, cohort, bank) in completed:
                    print(f"[复用] {bank} {region}/{cohort} 已完成。", flush=True)
                    continue
                with heartbeat(f"{METHOD} {bank} {region}/{cohort} FID/KID"):
                    kid = lib["kid"](real_features, fake_features, **options)
                    fid = lib["fid"](real_stats, lib["fid_stats"](fake_features), verbose=False)
                values = [fid["frechet_inception_distance"],
                          kid["kernel_inception_distance_mean"],
                          kid["kernel_inception_distance_std"]]
                require(all(math.isfinite(v) for v in values), "NONFINITE_METRIC", (region, cohort, bank))
                row = {
                    "method": METHOD, "bank": bank, "region": region, "cohort": cohort,
                    "real_count": len(actual), "generated_count": len(fake),
                    "FID": float(values[0]), "KID": float(values[1]),
                    "KID_subset_std": float(values[2]), "KID_subset_std_ddof": 0,
                    "KID_subsets": options["kid_subsets"],
                    "KID_subset_size": options["kid_subset_size"], "KID_scale": 1.0,
                    "all_inputs_used": True,
                }
                records.append(row)
                save(partial_path, {"status": "RUNNING", "records": records})
                print(f"[完成] {bank} {region}/{cohort}: FID={values[0]:.6g} KID={values[1]:.6g}", flush=True)
    require(len(records) == 18, "BANKWISE_RESULTS_INCOMPLETE", len(records))
    save(partial_path, {"status": "COMPLETE", "records": records})
    save(output / "anomalydiffusion_kid_subset_sampling_audit.json", sampling)
    return records


def lpips_metrics(output, prepared, lib, runtime, protocol):
    torch, np, Image = lib["torch"], lib["np"], lib["Image"]
    old_hub = torch.hub.get_dir()
    alex = Path(runtime["weights"]["alexnet"]["path"])
    torch.hub.set_dir(str(alex.parent.parent))
    try:
        model = lib["lpips"].LPIPS(
            net="alex", version="0.1", lpips=True, pretrained=True, pnet_rand=False,
            model_path=runtime["weights"]["lpips_linear"]["path"], eval_mode=True, verbose=False,
        )
    finally:
        torch.hub.set_dir(old_hub)
    model = model.eval().requires_grad_(False).to(runtime["device"])
    by_slot = {}
    for r in prepared["records"]:
        by_slot.setdefault(r["slot_id"], {})[r["bank"]] = r
    rows = []
    with torch.inference_mode(), heartbeat("AnomalyDiffusion ROI LPIPS A-B/A-C/B-C"):
        for index, (slot_id, banks) in enumerate(sorted(by_slot.items())):
            require(set(banks) == set(BANKS), "LPIPS_SLOT_COVERAGE_INVALID", slot_id)
            tensors = {}
            for bank, row in banks.items():
                image = open_rgb({"path": row["image_path"], "canvas_wh": row["canvas_wh"],
                                  "crop": row["roi_xyxy"]}, Image)
                image = image.resize(tuple(protocol["LPIPS"]["resize_wh"]), Image.Resampling.BILINEAR)
                pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
                tensors[bank] = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(runtime["device"])
            values = {f"LPIPS_{a}_{b}": float(model(tensors[a], tensors[b], normalize=False).item())
                      for a, b in itertools.combinations(BANKS, 2)}
            require(all(math.isfinite(v) for v in values.values()), "NONFINITE_LPIPS", slot_id)
            rows.append({"method": METHOD, "slot_id": slot_id, "class": banks["A"]["class"],
                         **values, "LPIPS_slot": sum(values.values()) / 3})
            if (index + 1) % 20 == 0:
                print(f"[LPIPS] {index+1}/80 slots。", flush=True)
    del model
    if runtime["device"].startswith("cuda:"):
        torch.cuda.empty_cache()
    csv_write(output / "anomalydiffusion_lpips_per_slot.csv",
              ["Method", "slot_id", "class", "LPIPS_A_B", "LPIPS_A_C", "LPIPS_B_C", "LPIPS_slot"],
              [[r["method"], r["slot_id"], r["class"], r["LPIPS_A_B"], r["LPIPS_A_C"],
                r["LPIPS_B_C"], r["LPIPS_slot"]] for r in rows])
    return rows


def summarize_ad(records, lpips_rows):
    summary = {}
    for region in ("full_image", "roi"):
        summary[region] = {}
        for cohort in ("overall", *CLASSES):
            rows = [r for r in records if r["region"] == region and r["cohort"] == cohort]
            require(len(rows) == 3 and {r["bank"] for r in rows} == set(BANKS),
                    "BANKWISE_RESULTS_INCOMPLETE", (region, cohort))
            summary[region][cohort] = {
                "FID": mean_sd([r["FID"] for r in rows]),
                "KID": mean_sd([r["KID"] for r in rows]),
                "bankwise": {r["bank"]: r for r in rows},
            }
    summary["LPIPS"] = {}
    for cohort in ("overall", *CLASSES):
        rows = [r for r in lpips_rows if cohort == "overall" or r["class"] == cohort]
        require(len(rows) == (80 if cohort == "overall" else 40),
                "LPIPS_SLOT_COVERAGE_INVALID", cohort)
        summary["LPIPS"][cohort] = mean_sd([r["LPIPS_slot"] for r in rows])
    return summary


def display(value):
    return f'{value["mean"]:.6g} ± {value["sd"]:.6g}'


def publish(output, cfg, records, lpips_rows, ad_summary, reference, runtime):
    exp1, _, _, _ = paths(cfg)
    old = load(exp1 / "generation_quality_summary.json")
    methods = {"Raw SD2": old["methods"]["Raw SD2"],
               "MSDF": old["methods"]["MSDF"], METHOD: ad_summary}
    combined = {
        "status": "COMPLETE", "methods": methods,
        "primary": "defect-region KID", "FID_auxiliary_small_sample_warning": True,
        "bank_SD_is_not_independent_training_seed_uncertainty": True,
        "LPIPS_is_diversity_not_realism_or_higher_quality": True,
        "reference_image_count": reference["image_count"],
        "reference_roi_count": reference["instance_count"],
        "training_count": 0, "generation_count": 0, "detector_evaluation_count": 0,
    }
    save(output / "generation_quality_summary_three_methods.json", combined)
    save(output / "anomalydiffusion_generation_quality_roi.json", {
        "status": "COMPLETE", "region": "roi", "method": METHOD,
        "results": ad_summary["roi"],
        "bankwise": [r for r in records if r["region"] == "roi"],
    })
    save(output / "anomalydiffusion_generation_quality_full_image.json", {
        "status": "COMPLETE", "region": "full_image", "method": METHOD,
        "results": ad_summary["full_image"],
        "bankwise": [r for r in records if r["region"] == "full_image"],
    })
    save(output / "anomalydiffusion_lpips_summary.json", {
        "status": "COMPLETE", "interpretation": "appearance diversity only; not realism",
        "slot_pair_count": 240, "slot_count": 80, "results": ad_summary["LPIPS"], "SD_ddof": 1,
    })
    table = []
    for method in ALL_METHODS:
        s = methods[method]
        table.append([method, display(s["roi"]["overall"]["FID"]),
                      display(s["roi"]["overall"]["KID"]), display(s["LPIPS"]["overall"])])
    csv_write(output / "paper_table_generation_quality_three_methods.csv",
              ["Method", "FID↓", "KID↓", "LPIPS↑"], table)
    class_rows = []
    for method in ALL_METHODS:
        s = methods[method]
        class_rows.append([method, *[display(s["roi"][c][k]) if k != "LPIPS"
                                     else display(s["LPIPS"][c])
                                     for c in CLASSES for k in ("FID", "KID", "LPIPS")]])
    csv_write(output / "paper_table_generation_quality_classwise_three_methods.csv",
              ["Method", "Flash FID", "Flash KID", "Flash LPIPS",
               "Black FID", "Black KID", "Black LPIPS"], class_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
    colors = ["#4477AA", "#EE8844", "#228833"]
    for ax, metric in zip(axes, ("FID", "KID", "LPIPS")):
        values = [methods[m]["LPIPS"]["overall"] if metric == "LPIPS"
                  else methods[m]["roi"]["overall"][metric] for m in ALL_METHODS]
        ax.bar(ALL_METHODS, [v["mean"] for v in values], yerr=[v["sd"] for v in values],
               color=colors, capsize=4, width=.62)
        ax.set_title("LPIPS appearance diversity" if metric == "LPIPS" else metric + " (lower is better)")
        ax.tick_params(axis="x", rotation=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Defect-region frozen visual metrics")
    fig.savefig(output / "generation_quality_three_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Experiment 2 — AnomalyDiffusion frozen visual evaluation", "",
        "Status: `COMPLETE`.", "",
        "No training, image generation, selection, reroll, or detector evaluation ran. "
        "The evaluator reused the frozen Experiment 1 real Inception features and Raw SD2/MSDF results, "
        "then evaluated all 240 AnomalyDiffusion images under the same protocol.", "",
        "## Defect-region primary table", "",
        "| Method | FID (bank mean ± SD) | KID (bank mean ± SD, unscaled) | LPIPS (slot mean ± SD) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, fid, kid, lpips in table:
        lines.append(f"| {method} | {fid} | {kid} | {lpips} |")
    lines += ["", "Primary analysis is defect-region KID. FID is auxiliary because the reference sample is small. "
              "LPIPS measures A/B/C appearance diversity within each frozen slot and is not a realism score.", "",
              "## Class-wise defect-region results", "",
              "| Method | Flash FID | Flash KID | Flash LPIPS | Black FID | Black KID | Black LPIPS |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in class_rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += ["", "FID/KID were computed independently for banks A/B/C, with 80 generated images per bank "
              "and 40 per class. SD uses ddof=1 across the three banks. KID uses the same fixed seed 42, "
              "100 subsets, and subset size min(50, n_real, n_generated) as Experiment 1.", "",
              "The ROI is the original normalized defect box expanded by 25% on every side, then floor/ceil "
              "rounded and clipped to the image. LPIPS AlexNet v0.1 compares A-B, A-C, and B-C crops for "
              "each slot at 256×256 and summarizes the 80 per-slot means.", ""]
    (output / "ANOMALYDIFFUSION_VISUAL_EVALUATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n")
    status = {
        "status": "COMPLETE", "complete": True, "method": METHOD,
        "ROI_FID": ad_summary["roi"]["overall"]["FID"],
        "ROI_KID": ad_summary["roi"]["overall"]["KID"],
        "LPIPS": ad_summary["LPIPS"]["overall"],
        "Flash": {"FID": ad_summary["roi"]["flash"]["FID"],
                  "KID": ad_summary["roi"]["flash"]["KID"],
                  "LPIPS": ad_summary["LPIPS"]["flash"]},
        "Black": {"FID": ad_summary["roi"]["black"]["FID"],
                  "KID": ad_summary["roi"]["black"]["KID"],
                  "LPIPS": ad_summary["LPIPS"]["black"]},
        "new_inception_feature_count": 480, "reused_exp1_feature_count": 108,
        "lpips_pair_count": 240, "training_count": 0, "generation_count": 0,
        "detector_evaluation_count": 0, "metric_runtime_sha256": sha(output / "metric_runtime.json"),
    }
    save(output / "evaluation_status.json", status)
    artifacts = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*")
                 if p.is_file() and p.name != "evaluation_artifact_manifest.json"
                 and p.suffix in (".json", ".md", ".csv", ".png")}
    save(output / "evaluation_artifact_manifest.json", {
        "status": "COMPLETE", "sha256": artifacts,
        "ad_feature_cache_sha256": sha(output / "anomalydiffusion_inception_features.npz"),
        "old_exp1_files_modified": False,
    })


def evaluate(cfg, config_path, device):
    exp1, _, output, _ = paths(cfg)
    status_path = output / "evaluation_status.json"
    if status_path.exists() and load(status_path).get("complete"):
        manifest = load(output / "evaluation_artifact_manifest.json")
        for name, digest in manifest["sha256"].items():
            checked_file(output / name, digest)
        checked_file(output / "anomalydiffusion_inception_features.npz",
                     manifest["ad_feature_cache_sha256"])
        print("[已完成] 现有评价结果及哈希完整；不重复计算。", flush=True)
        return
    pairs, reference, prepared, protocol = verify_prepared(cfg, config_path)
    dep = dependencies(
        protocol,
        load(exp1 / "metric_runtime.json")["weights"]["inception"]["path"],
        load(exp1 / "metric_runtime.json")["weights"]["alexnet"]["path"],
    )
    require(dep["status"] == "PASS", "DEPENDENCY_REQUIRED", dep["errors"])
    with offline_metrics():
        lib, runtime = runtime_modules(protocol, dep, device)
        require(runtime == load(exp1 / "metric_runtime.json"), "METRIC_RUNTIME_CHANGED",
                "Exp2 must exactly match frozen Exp1 metric runtime and cuda device")
        runtime_path = output / "metric_runtime.json"
        if runtime_path.exists():
            require(load(runtime_path) == runtime, "METRIC_RUNTIME_CHANGED", str(runtime_path))
        else:
            save(runtime_path, runtime)
        save(status_path, {"status": "RUNNING", "complete": False, "method": METHOD,
                          "training_count": 0, "generation_count": 0, "detector_evaluation_count": 0})
        generated = ad_items(prepared)
        real = real_items(reference)
        features = load_exp1_features(cfg, lib)
        features.update(extract_ad_features(output, generated, lib, runtime))
        require(all(r["key"] in features for r in (*real, *generated)),
                "FEATURE_CACHE_BINDING_CHANGED", "missing real or AD feature")
        records = distribution_metrics(output, real, generated, features, protocol, lib, runtime)
        lpips_rows = lpips_metrics(output, prepared, lib, runtime, protocol)
        ad_summary = summarize_ad(records, lpips_rows)
        verify_prepared(cfg, config_path)
        after = dependencies(protocol, runtime["weights"]["inception"]["path"],
                             runtime["weights"]["alexnet"]["path"])
        require(after["package_versions"] == dep["package_versions"] and after["weights"] == dep["weights"],
                "METRIC_RUNTIME_CHANGED", "packages or weights changed during evaluation")
        publish(output, cfg, records, lpips_rows, ad_summary, reference, runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "evaluate", "all"), default="all")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="0", help="visible CUDA index, cuda:N, or cpu")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load(config_path)
    _, _, output, _ = paths(cfg)
    require(output == resolve(cfg["output"]), "OUTPUT_PATH_INVALID", str(output))
    device = "cuda:" + args.device if args.device.isdecimal() else args.device
    print(f"[启动] Exp2 AnomalyDiffusion 冻结视觉评价；mode={args.mode}；PID={os.getpid()}。", flush=True)
    print("[约束] 不训练、不生成、不选图、不运行检测器；仅复用 Exp1 并评价现有 240 张。", flush=True)
    if args.mode in ("preflight", "all"):
        prepare(cfg, config_path)
        print("[检查通过] 240 张 AD 图与 Exp1 槽位、类别、ROI 和 SHA256 全部绑定。", flush=True)
    if args.mode in ("evaluate", "all"):
        evaluate(cfg, config_path, device)
        print("[已完成] AnomalyDiffusion FID/KID/LPIPS 冻结评价完成。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Stop as error:
        print(f"[已退出] {error.status}: {error.detail}", flush=True)
        raise SystemExit(2)
