"""Evaluate the frozen MSDF gamma sweep on the V2 TRAIN development ROIs."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
GAMMAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASSES = ("flash", "black")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tag(gamma):
    return f"gamma_{gamma:.2f}".replace(".", "p")


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def open_crop(path, box, Image):
    with Image.open(path) as source:
        image = source.convert("RGB")
    return image.crop(tuple(box))


def runtime(device):
    import numpy as np
    import torch
    import lpips
    from PIL import Image
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric
    require(torch.cuda.is_available(), "CUDA unavailable for frozen neural metrics")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    home = Path.home() / ".cache/torch/hub/checkpoints"
    inception = home / "weights-inception-2015-12-05-6726825d.pth"
    alexnet = home / "alexnet-owt-7be5be79.pth"
    linear = Path(metadata.distribution("lpips").locate_file("lpips/weights/v0.1/alex.pth"))
    expected = {
        inception: "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
        alexnet: "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
        linear: "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    }
    for path, digest in expected.items():
        require(path.is_file() and sha(path) == digest, f"metric weight changed: {path}")
    modules = ("torch_fidelity.feature_extractor_inceptionv3", "torch_fidelity.metric_fid",
               "torch_fidelity.metric_kid", "lpips.lpips", "lpips.pretrained_networks")
    audit = {
        "status": "FROZEN_BEFORE_METRICS", "device": device, "python": sys.version,
        "torch": torch.__version__, "CUDA": torch.version.cuda,
        "torch_fidelity": metadata.version("torch-fidelity"), "lpips": metadata.version("lpips"),
        "weights": {p.name: {"path": str(p), "sha256": d} for p, d in expected.items()},
        "source_files": {name: {"path": importlib.import_module(name).__file__,
                                "sha256": sha(importlib.import_module(name).__file__)} for name in modules},
        "TF32": False, "autocast": False, "network_downloads": 0,
    }
    require(audit["torch_fidelity"] == "0.3.0", "metric API version changed")
    return {"np": np, "torch": torch, "lpips": lpips, "Image": Image, "Inception": FeatureExtractorInceptionV3,
            "fid_stats": fid_features_to_statistics, "fid": fid_statistics_to_metric,
            "kid": kid_features_to_metric, "inception_weight": inception,
            "alexnet_weight": alexnet, "linear_weight": linear}, audit


def inputs(output, protocol):
    rows = []
    for reference in protocol["development_reference"]["records"]:
        path = Path(reference["image_path"])
        require(path.is_file() and sha(path) == reference["image_sha256"], f"reference changed: {path}")
        rows.append({"key": "real:" + reference["annotation_id"], "role": "real", "class": reference["class"],
                     "path": str(path), "roi": reference["roi_xyxy"]})
    for gamma in GAMMAS:
        manifest = load(output / f"{tag(gamma)}_manifest.json")
        require(manifest["status"] == "COMPLETE" and manifest["count"] == 80, f"incomplete {gamma}")
        for row in manifest["records"]:
            path = output / row["image_path"]
            require(path.is_file() and sha(path) == row["image_sha256"], f"generated changed: {path}")
            rows.append({"key": f"fake:{gamma:.2f}:" + row["slot_id"], "role": "fake", "gamma": gamma,
                         "class": row["class"], "slot_id": row["slot_id"], "path": str(path), "roi": row["roi_xyxy"]})
    require(len(rows) == 572 and len({x["key"] for x in rows}) == 572, "feature population changed")
    return rows


def features(output, rows, lib, device):
    torch, np, Image = lib["torch"], lib["np"], lib["Image"]
    cache = output / "gamma_sweep_inception_features.npz"
    ledger = output / "gamma_sweep_inception_feature_cache.json"
    keys = [x["key"] for x in rows]
    if cache.exists() and ledger.exists():
        info = load(ledger)
        require(info["keys"] == keys and info["sha256"] == sha(cache), "feature cache binding changed")
        with np.load(cache, allow_pickle=False) as archive:
            array = archive["features"].copy()
    else:
        model = lib["Inception"]("inception-v3-compat", ["2048"],
                                  feature_extractor_weights_path=str(lib["inception_weight"])).float().eval().requires_grad_(False).to(device)
        values = []
        with torch.inference_mode():
            for index, row in enumerate(rows):
                image = open_crop(row["path"], row["roi"], Image)
                x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)[None].to(device)
                value = model(x)[0]
                require(tuple(value.shape) == (1, 2048) and bool(torch.isfinite(value).all()), row["key"])
                values.append(value[0].double().cpu().numpy())
                if (index + 1) % 20 == 0 or index + 1 == len(rows):
                    print(f"[Inception] {index+1}/{len(rows)} ROI", flush=True)
        array = np.stack(values)
        np.savez_compressed(cache, features=array)
        save(ledger, {"status": "COMPLETE", "keys": keys, "sha256": sha(cache), "shape": list(array.shape)})
        del model; torch.cuda.empty_cache()
    require(array.shape == (572, 2048) and bool(np.isfinite(array).all()), "invalid feature cache")
    return {key: torch.from_numpy(array[index]) for index, key in enumerate(keys)}


def distribution(rows, feats, lib):
    torch = lib["torch"]
    result = []
    for cohort in ("overall", *CLASSES):
        real_rows = [x for x in rows if x["role"] == "real" and (cohort == "overall" or x["class"] == cohort)]
        real = torch.stack([feats[x["key"]] for x in real_rows]).double().cpu()
        real_stats = lib["fid_stats"](real)
        for gamma in GAMMAS:
            fake_rows = [x for x in rows if x["role"] == "fake" and x["gamma"] == gamma
                         and (cohort == "overall" or x["class"] == cohort)]
            fake = torch.stack([feats[x["key"]] for x in fake_rows]).double().cpu()
            options = {"kid_subsets": 100, "kid_subset_size": min(50, len(real_rows), len(fake_rows)),
                       "rng_seed": 42, "verbose": False, "kid_degree": 3, "kid_gamma": None, "kid_coef0": 1.0}
            before = time.monotonic()
            kid = lib["kid"](real, fake, **options)
            fid = lib["fid"](real_stats, lib["fid_stats"](fake), verbose=False)
            result.append({
                "gamma": gamma, "class": cohort, "real_roi_count": len(real_rows),
                "generated_roi_count": len(fake_rows), "KID": float(kid["kernel_inception_distance_mean"]),
                "KID_subset_std": float(kid["kernel_inception_distance_std"]),
                "FID": float(fid["frechet_inception_distance"]), "KID_subsets": 100,
                "KID_subset_size": options["kid_subset_size"], "metric_runtime_seconds": time.monotonic()-before,
            })
            print(f"[指标] gamma={gamma:.2f} {cohort}: KID={result[-1]['KID']:.8f} FID={result[-1]['FID']:.5f}", flush=True)
    return result


def lpips_values(output, rows, lib, device):
    torch, np, Image = lib["torch"], lib["np"], lib["Image"]
    old_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(lib["alexnet_weight"].parent.parent))
    try:
        model = lib["lpips"].LPIPS(net="alex", version="0.1", lpips=True, pretrained=True,
                                    pnet_rand=False, model_path=str(lib["linear_weight"]), eval_mode=True, verbose=False)
    finally:
        torch.hub.set_dir(old_hub)
    model = model.eval().requires_grad_(False).to(device)
    pair_rows = []
    with torch.inference_mode():
        for gamma in GAMMAS:
            for cls in CLASSES:
                candidates = sorted((x for x in rows if x["role"] == "fake" and x["gamma"] == gamma and x["class"] == cls),
                                    key=lambda x: x["slot_id"])
                require(len(candidates) == 40, f"LPIPS coverage {gamma} {cls}")
                for pair_index in range(20):
                    first, second = candidates[2*pair_index:2*pair_index+2]
                    tensors = []
                    for row in (first, second):
                        image = open_crop(row["path"], row["roi"], Image).resize((256, 256), Image.Resampling.BILINEAR)
                        pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
                        tensors.append(torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device))
                    value = float(model(tensors[0], tensors[1], normalize=False).item())
                    require(math.isfinite(value), "nonfinite LPIPS")
                    pair_rows.append({"gamma": gamma, "class": cls, "pair_index": pair_index,
                                      "slot_id_1": first["slot_id"], "slot_id_2": second["slot_id"], "LPIPS": value})
            print(f"[LPIPS] gamma={gamma:.2f}: 40 固定配对", flush=True)
    del model; torch.cuda.empty_cache()
    write_csv(output / "gamma_sweep_lpips_pairs.csv",
              ["gamma", "class", "pair_index", "slot_id_1", "slot_id_2", "LPIPS"], pair_rows)
    return pair_rows


def generation_runtime(output, gamma):
    manifest = load(output / f"{tag(gamma)}_manifest.json")
    values = [x.get("runtime_seconds") for x in manifest["records"] if x.get("runtime_seconds") is not None]
    return sum(values) if values else manifest.get("runtime_seconds_this_invocation", 0.0)


def publish(output, protocol, distributions, pairs, audit, wall_seconds):
    checkpoint = protocol["checkpoints"]
    class_rows, overall_rows = [], []
    for gamma in GAMMAS:
        runtime_s = generation_runtime(output, gamma)
        for cohort in CLASSES:
            metric = next(x for x in distributions if x["gamma"] == gamma and x["class"] == cohort)
            values = [x["LPIPS"] for x in pairs if x["gamma"] == gamma and x["class"] == cohort]
            class_rows.append({
                "gamma": gamma, "class": cohort.capitalize(), "generated_images": 40,
                "reference_rois": metric["real_roi_count"], "KID": metric["KID"], "KID_subset_std": metric["KID_subset_std"],
                "FID": metric["FID"], "LPIPS": sum(values)/len(values), "LPIPS_pairs": len(values),
                "generation_runtime_seconds_all_classes": runtime_s,
                "checkpoint_tree_SHA256": checkpoint[cohort]["tree_sha256"],
                "msdf_checkpoint_SHA256": checkpoint[cohort]["msdf_sha256"],
            })
        metric = next(x for x in distributions if x["gamma"] == gamma and x["class"] == "overall")
        values = [x["LPIPS"] for x in pairs if x["gamma"] == gamma]
        overall_rows.append({
            "gamma": gamma, "generated_images": 80, "reference_rois": metric["real_roi_count"],
            "Overall_KID": metric["KID"], "Overall_KID_subset_std": metric["KID_subset_std"],
            "Overall_FID": metric["FID"], "Overall_LPIPS": sum(values)/len(values),
            "LPIPS_pairs": len(values), "generation_runtime_seconds": runtime_s,
            "Flash_checkpoint_tree_SHA256": checkpoint["flash"]["tree_sha256"],
            "Black_checkpoint_tree_SHA256": checkpoint["black"]["tree_sha256"],
        })
    write_csv(output / "gamma_sweep_results.csv", list(overall_rows[0]), overall_rows)
    write_csv(output / "gamma_sweep_classwise.csv", list(class_rows[0]), class_rows)
    save(output / "gamma_sweep_metric_details.json", {"status": "COMPLETE", "records": distributions,
                                                       "LPIPS_pair_count": len(pairs), "runtime": audit})

    def row(gamma, cohort):
        return next(x for x in class_rows if x["gamma"] == gamma and x["class"] == cohort.capitalize())
    best = min(GAMMAS, key=lambda g: (row(g, "black")["KID"], next(x for x in overall_rows if x["gamma"] == g)["Overall_KID"], row(g, "flash")["KID"]))
    black_values = [row(g, "black")["KID"] for g in GAMMAS]
    flash_values = [row(g, "flash")["KID"] for g in GAMMAS]
    black_improves_monotonically_with_increasing_gamma = all(
        black_values[i] >= black_values[i+1] for i in range(4)
    )
    black_improved = row(best, "black")["KID"] < row(1.0, "black")["KID"]
    gamma0_better = row(0.0, "black")["KID"] < row(1.0, "black")["KID"]
    flash_delta_best = row(best, "flash")["KID"] - row(1.0, "flash")["KID"]
    over_injection = black_improved and best < 1.0
    lines = [
        "# MSDF Feature Residual Injection Strength Sweep", "",
        "Status: `COMPLETE`. This was an inference-only Bank-A development sweep; no model was trained.", "",
        "## Protocol and interpretation limits", "",
        "All five gamma values used the same 80 frozen Bank-A slots, normals, bboxes, prompts, seeds, steps, scheduler, CFG and checkpoints. Gamma scales only the final tanh-bounded residual immediately before it is added to each UNet up-block. At gamma=1.0 the multiply is skipped and two class-specific end-to-end PNGs exactly matched the prior frozen MSDF outputs.", "",
        "The reference is all 172 original V2 TRAIN ROIs (Flash 75, Black 97). It may overlap the data used to train these checkpoints. These scores are exploratory screening results, not independent validation or final paper performance. `final_eval` was not read or used. Historical final_eval Raw SD2/MSDF scores are not directly comparable to this table.", "",
        "LPIPS is an exploratory Bank-A diversity reference from 20 deterministic adjacent slot pairs per class. It differs from the historical same-slot A/B/C LPIPS protocol and is not directly comparable to it.", "",
        "## Complete results", "",
        "| gamma | Flash KID ↓ | Black KID ↓ | Overall KID ↓ | Flash FID ↓ | Black FID ↓ | Overall FID ↓ | Flash LPIPS ↑ | Black LPIPS ↑ | Overall LPIPS ↑ |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for gamma in GAMMAS:
        f, b = row(gamma, "flash"), row(gamma, "black")
        o = next(x for x in overall_rows if x["gamma"] == gamma)
        lines.append(f"| {gamma:.2f} | {f['KID']:.8f} | {b['KID']:.8f} | {o['Overall_KID']:.8f} | {f['FID']:.5f} | {b['FID']:.5f} | {o['Overall_FID']:.5f} | {f['LPIPS']:.6f} | {b['LPIPS']:.6f} | {o['Overall_LPIPS']:.6f} |")
    if black_improves_monotonically_with_increasing_gamma:
        trend = "is monotonic in the opposite direction: Black KID improves as gamma increases, so reducing gamma steadily hurts Black KID"
    else:
        trend = "is not monotonic across the tested grid"
    flash_effect = "improves" if flash_delta_best < 0 else "worsens" if flash_delta_best > 0 else "is unchanged"
    lines += ["", "## Required findings", "",
              f"1. Lower residual strength {'improves' if black_improved else 'does not improve'} Black KID. The exploratory Black-KID best is gamma={best:.2f}: {row(best, 'black')['KID']:.8f}; gamma=0.75 is {row(0.75, 'black')['KID'] - row(1.0, 'black')['KID']:+.8f} and gamma=0 is {row(0.0, 'black')['KID'] - row(1.0, 'black')['KID']:+.8f} relative to gamma=1.00.",
              f"2. The Black improvement pattern {trend} when ordered from gamma=0.00 to 1.00.",
              f"3. At the Black-selected gamma, Flash KID {flash_effect} by {flash_delta_best:+.8f} versus gamma=1.00. All Flash values are retained in the table.",
              f"4. Gamma=0 {'is' if gamma0_better else 'is not'} better than gamma=1 for Black KID ({row(0.0, 'black')['KID']:.8f} vs {row(1.0, 'black')['KID']:.8f}).",
              f"5. This TRAIN-overlap screen {'shows exploratory evidence of excessive residual injection' if over_injection else 'does not show evidence of excessive residual injection'} at gamma=1.00. This conclusion requires multiple banks and source-group-isolated validation before formal use.", "",
              "The sweep stops here. No wider gamma search, new training, BootstrapGuard, multi-bank confirmation or second optimization round was started.", ""]
    (output / "GAMMA_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    status = load(output / "gamma_sweep_status.json")
    status.update({
        "status": "COMPLETE", "full_experiment_complete": True, "completed_at": now(),
        "gamma_values": list(GAMMAS), "generated_image_count_per_gamma": 80, "output_image_count": 400,
        "metric_reference_roi_count": 172, "metric_reference_class_counts": {"flash": 75, "black": 97},
        "best_exploratory_gamma_by_black_KID": best, "Black_KID_improved_vs_gamma1": black_improved,
        "Black_KID_monotonic_improvement_with_increasing_gamma": black_improves_monotonically_with_increasing_gamma,
        "gamma0_Black_KID_better_than_gamma1": gamma0_better,
        "evidence_of_excessive_injection_at_gamma1": over_injection,
        "evaluation_runtime_seconds": wall_seconds, "training_count": 0, "final_eval_content_read": 0,
        "second_round_started": False, "BootstrapGuard": False,
    })
    save(output / "gamma_sweep_status.json", status)
    public = [p for p in output.iterdir() if p.is_file() and p.name not in
              {"gamma_sweep_inception_features.npz"}]
    save(output / "gamma_sweep_artifact_manifest.json", {"status": "COMPLETE",
         "sha256": {p.name: sha(p) for p in sorted(public)},
         "excluded_large_cache": "gamma_sweep_inception_features.npz"})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output == OUTPUT.resolve(), "only the frozen gamma-sweep output is allowed")
    protocol = load(output / "gamma_sweep_protocol.json")
    require(load(output / "gamma_sweep_status.json")["status"] == "GENERATION_COMPLETE", "generation incomplete")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    print(f"[启动] gamma sweep ROI metrics PID={os.getpid()}；V2 TRAIN reference；不训练。", flush=True)
    lib, audit = runtime(args.device)
    save(output / "gamma_sweep_metric_runtime.json", audit)
    rows = inputs(output, protocol)
    feats = features(output, rows, lib, args.device)
    distributions = distribution(rows, feats, lib)
    pairs = lpips_values(output, rows, lib, args.device)
    publish(output, protocol, distributions, pairs, audit, time.monotonic()-started)
    print("[完成] 全部5个 gamma 的 ROI KID/FID/LPIPS 已输出；停止，不启动下一轮。", flush=True)


if __name__ == "__main__":
    main()
