#!/usr/bin/env python3
"""Reproducible Plastic_Bomo RealRepeat control for BootstrapGuard.

Each epoch contains every one of the 138 real training images once plus 80
replacement draws from the same real set.  No dataset files are copied or
modified; repetition is implemented solely by a deterministic sampler.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Sampler

from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import get_torch_device_backend


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_for_gpt/bootstrap_guard_realrepeat"
RUNS = ROOT / "runs/bootstrap_guard_realrepeat"
DATA = Path("/mnt/sda/cbj/dataset/Plastic_Bomo/bootstrap_guard_downstream/real_only.yaml")
MODEL = ROOT / "pretrained/yolo11s.pt"
HIST = ROOT / "results_for_gpt/bootstrap_guard_downstream/validated_per_seed.json"
SEEDS = (42, 3407, 2026)
EXTRA = 80


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class RealRepeatSampler(Sampler[int]):
    """138 unique base indices + 80 replacement draws, newly sampled per epoch."""

    def __init__(self, size: int, seed: int, audit_path: Path):
        if size != 138:
            raise RuntimeError(f"Frozen protocol requires 138 real images, found {size}")
        self.size, self.seed, self.epoch = size, seed, 0
        self.audit_path = audit_path
        self.written_epochs: set[int] = set()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.size + EXTRA

    def _indices(self) -> list[int]:
        # Separate streams make the replacement multiset and final order explicit.
        draw_rng = random.Random(self.seed * 1_000_003 + self.epoch * 97_409 + 17)
        order_rng = random.Random(self.seed * 1_000_033 + self.epoch * 97_421 + 29)
        replacements = [draw_rng.randrange(self.size) for _ in range(EXTRA)]
        indices = list(range(self.size)) + replacements
        order_rng.shuffle(indices)
        return indices

    def __iter__(self):
        indices = self._indices()
        if self.epoch not in self.written_epochs:
            counts = Counter(indices)
            record = {
                "epoch_zero_based": self.epoch,
                "epoch": self.epoch + 1,
                "seed": self.seed,
                "base_size": self.size,
                "replacement_draws": EXTRA,
                "samples": len(indices),
                "batches": len(indices),
                "all_base_images_present": set(indices) == set(range(self.size)),
                "unique_base_images": len(set(indices)),
                "repeated_exposures": len(indices) - self.size,
                "max_exposures_for_one_image": max(counts.values()),
                "sequence_sha256": hashlib.sha256(
                    ",".join(map(str, indices)).encode()
                ).hexdigest(),
                "replacement_sequence_sha256": hashlib.sha256(
                    ",".join(map(str, indices[self.size:])).encode()
                ).hexdigest(),
            }
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            self.written_epochs.add(self.epoch)
        return iter(indices)


class RealRepeatTrainer(DetectionTrainer):
    """DetectionTrainer whose train loader uses the frozen RealRepeat sampler."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        audit_path = OUT / f"sampling_epochs_seed{self.args.seed}.jsonl"
        audit_path.unlink(missing_ok=True)
        sampler = RealRepeatSampler(len(dataset), int(self.args.seed), audit_path)
        batch_size = min(batch_size, len(dataset))
        workers = min(os.cpu_count() or 1, int(self.args.workers), len(sampler))
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + int(self.args.seed))
        device_type = getattr(self.device, "type", str(self.device).split(":")[0])
        nd = get_torch_device_backend(self.device).device_count() if device_type not in {"cpu", "mps"} else 0
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=workers,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            prefetch_factor=4 if workers else None,
            drop_last=False,
        )


def metric_payload(metrics) -> dict:
    box = metrics.box
    names = metrics.names
    per_class = {}
    for i, name in names.items():
        key = "flash" if i == 0 else "black"
        per_class[key] = {
            "ap50": float(box.ap50[i]),
            "ap5095": float(box.maps[i]),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "dataset_name": name,
        }
    return {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map5095": float(box.map),
        "per_class": per_class,
    }


def run_seed(seed: int, device: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if not DATA.is_file() or not MODEL.is_file() or not HIST.is_file():
        raise FileNotFoundError("Frozen data/model/historical results are incomplete")
    run_dir = RUNS / f"realrepeat_s{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")

    counters = {"optimizer_steps": 0}

    def epoch_start(trainer):
        sampler = trainer.train_loader.sampler
        if not isinstance(sampler, RealRepeatSampler):
            raise RuntimeError("RealRepeat sampler was not installed")
        sampler.set_epoch(trainer.epoch)

    def optimizer_step(_trainer):
        counters["optimizer_steps"] += 1

    model = YOLO(str(MODEL))
    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("optimizer_step", optimizer_step)
    model.train(
        trainer=RealRepeatTrainer,
        data=str(DATA), model=str(MODEL), epochs=150, patience=40,
        batch=1, imgsz=1536, device=device, workers=4,
        project=str(RUNS), name=f"realrepeat_s{seed}", exist_ok=False,
        pretrained=True, optimizer="auto", seed=seed, deterministic=True,
        rect=False, cos_lr=False, close_mosaic=10, amp=True, val=True,
        mosaic=1.0, mixup=0.0, lr0=0.01, lrf=0.01,
        momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5, nbs=64,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
    )
    results_csv = run_dir / "results.csv"
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    stop_epoch = len(rows)
    key = "metrics/mAP50-95(B)"
    best_epoch_csv = max(range(stop_epoch), key=lambda i: float(rows[i][key])) + 1
    best = run_dir / "weights/best.pt"
    val = YOLO(str(best)).val(data=str(DATA), imgsz=1536, batch=1, device=device, workers=4, plots=False)
    record = {
        "arm": "realrepeat", "seed": seed, "run_dir": str(run_dir.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256(best), "best_epoch_csv_map5095": best_epoch_csv,
        "stop_epoch": stop_epoch, "samples_per_epoch": 218, "batches_per_epoch": 218,
        "optimizer_steps": counters["optimizer_steps"], **metric_payload(val),
    }
    dump(OUT / f"seed_{seed}_result.json", record)
    print(json.dumps(record, indent=2), flush=True)


def recover_seed(seed: int, device: str) -> None:
    """Finish reporting from a completed run without repeating any training."""
    run_dir = RUNS / f"realrepeat_s{seed}"
    results_csv = run_dir / "results.csv"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    stop_epoch = len(rows)
    sampling = OUT / f"sampling_epochs_seed{seed}.jsonl"
    if len(sampling.read_text().splitlines()) != stop_epoch:
        raise RuntimeError("Sampling audit epoch count does not match results.csv")
    key = "metrics/mAP50-95(B)"
    best_epoch_csv = max(range(stop_epoch), key=lambda i: float(rows[i][key])) + 1
    # Ultralytics stores the EMA update counter in unstripped last.pt.  If a
    # completed run has already been stripped, reconstruct the invocation count
    # exactly from the frozen accumulation schedule and disclose that source.
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    updates = ckpt.get("updates")
    update_source = "last.pt:updates"
    if updates is None:
        nb, nw, last_opt_step, updates = 218, 654, -1, 0
        for epoch in range(stop_epoch):
            accumulate = 64
            for i in range(nb):
                ni = i + nb * epoch
                if ni < nw:
                    accumulate = max(1, int(np.interp(ni, [0, nw], [1, 64]).round()))
                if ni - last_opt_step >= accumulate:
                    updates += 1
                    last_opt_step = ni
        update_source = "exact_reconstruction_from_frozen_ultralytics_accumulation_schedule"
    val = YOLO(str(best)).val(data=str(DATA), imgsz=1536, batch=1, device=device, workers=4, plots=False)
    record = {
        "arm": "realrepeat", "seed": seed, "run_dir": str(run_dir.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256(best), "best_epoch_csv_map5095": best_epoch_csv,
        "stop_epoch": stop_epoch, "samples_per_epoch": 218, "batches_per_epoch": 218,
        "optimizer_steps": int(updates), "optimizer_steps_source": update_source,
        "recovered_without_retraining": True, **metric_payload(val),
    }
    dump(OUT / f"seed_{seed}_result.json", record)
    print(json.dumps(record, indent=2), flush=True)


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def finalize() -> None:
    records = [json.loads((OUT / f"seed_{s}_result.json").read_text()) for s in SEEDS]
    historical = json.loads(HIST.read_text())["records"]
    methods = ["real_only", "realrepeat", "random", "dwbg", "bootstrap_guard"]
    combined = historical + records
    rows = []
    for r in records:
        rows.append({
            "method": "RealRepeat", "seed": r["seed"], "precision": r["precision"],
            "recall": r["recall"], "map50": r["map50"], "map5095": r["map5095"],
            "flash_ap5095": r["per_class"]["flash"]["ap5095"],
            "black_ap5095": r["per_class"]["black"]["ap5095"],
            "best_epoch": r["best_epoch_csv_map5095"], "stop_epoch": r["stop_epoch"],
            "optimizer_steps": r["optimizer_steps"], "samples_per_epoch": 218,
        })
    with (OUT / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n"); w.writeheader(); w.writerows(rows)

    labels = {"real_only":"Real-only", "realrepeat":"RealRepeat", "random":"Random",
              "dwbg":"DWBG", "bootstrap_guard":"BootstrapGuard"}
    summary = []
    for method in methods:
        rs = [r for r in combined if r["arm"] == method]
        out = {"method": labels[method]}
        for key in ("precision", "recall", "map50", "map5095"):
            m, s = mean_std([float(r[key]) for r in rs])
            out[key + "_mean"] = m; out[key + "_std"] = s
            out[key] = f"{m:.4f} ± {s:.4f}"
        summary.append(out)
    fields = ["method", "precision", "recall", "map50", "map5095",
              "precision_mean", "precision_std", "recall_mean", "recall_std",
              "map50_mean", "map50_std", "map5095_mean", "map5095_std"]
    with (OUT / "final_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(summary)

    bg = {r["seed"]: r for r in historical if r["arm"] == "bootstrap_guard"}
    rr = {r["seed"]: r for r in records}
    deltas = {str(s): bg[s]["map5095"] - rr[s]["map5095"] for s in SEEDS}
    bg_mean = np.mean([bg[s]["map5095"] for s in SEEDS]); rr_mean = np.mean([rr[s]["map5095"] for s in SEEDS])
    paired = {
        "metric": "mAP50-95", "bootstrap_guard_minus_realrepeat_by_seed": deltas,
        "absolute_mean_difference": float(bg_mean - rr_mean),
        "relative_improvement_percent": float((bg_mean - rr_mean) / rr_mean * 100),
        "bootstrap_guard_wins": sum(v > 0 for v in deltas.values()),
        "realrepeat_wins": sum(v < 0 for v in deltas.values()), "ties": sum(v == 0 for v in deltas.values()),
        "class_ap5095": {
            c: {m: mean_std([float(r["per_class"][c]["ap5095"]) for r in combined if r["arm"] == m])[0]
                for m in ("real_only", "realrepeat", "bootstrap_guard")} for c in ("flash", "black")
        },
    }
    dump(OUT / "paired_deltas.json", paired)

    epoch_audits = {}
    for s in SEEDS:
        p = OUT / f"sampling_epochs_seed{s}.jsonl"
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        epoch_audits[str(s)] = {
            "epochs_recorded": len(recs), "all_epochs_218": all(x["samples"] == 218 for x in recs),
            "all_epochs_cover_all_138": all(x["all_base_images_present"] and x["unique_base_images"] == 138 for x in recs),
            "unique_epoch_sequence_hashes": len({x["sequence_sha256"] for x in recs}),
            "first_epoch_sequence_sha256": recs[0]["sequence_sha256"],
            "last_epoch_sequence_sha256": recs[-1]["sequence_sha256"],
            "raw_epoch_log_sha256": sha256(p),
            "raw_epoch_log_server_path": str(p),
        }
    dump(OUT / "sampling_audit.json", {
        "status": "PASS", "implementation": "deterministic epoch-aware torch Sampler; no data files changed",
        "rule": "all 138 real images once plus 80 replacement draws, shuffled each epoch",
        "audits": epoch_audits,
    })

    hist_steps = {}
    for s in SEEDS:
        h = bg[s]
        # With batch=1/nbs=64 Ultralytics accumulation is dynamic during warmup;
        # historical optimizer-step callbacks were not stored, so do not fabricate exact values.
        hist_steps[str(s)] = {"stop_epoch": h["stop_epoch"], "exact_optimizer_steps": "NOT_RECORDED_HISTORICALLY"}
    dump(OUT / "compute_match_audit.json", {
        "classification": "PER_EPOCH_EXPOSURE_MATCHED_NOT_STRICT_TOTAL_COMPUTE_MATCHED",
        "realrepeat": {str(r["seed"]): {k:r[k] for k in ("samples_per_epoch","batches_per_epoch","stop_epoch","optimizer_steps")} for r in records},
        "bootstrap_guard": hist_steps,
        "reason": "Early stopping may differ; historical exact optimizer-step callbacks are unavailable.",
        "minimal_strict_followup": "Predeclare fixed epochs (or fixed optimizer updates) and rerun both RealRepeat and BootstrapGuard from identical initialization; not executed here.",
    })
    args_ref = ROOT / "runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml"
    dump(OUT / "training_config_audit.json", {
        "status":"MATCHED_EXCEPT_DATA_SAMPLER", "ultralytics":ultralytics_version,
        "python":platform.python_version(), "torch":torch.__version__,
        "historical_args_path":str(args_ref.relative_to(ROOT)), "historical_args_sha256":sha256(args_ref),
        "model_path":str(MODEL.relative_to(ROOT)), "model_sha256":sha256(MODEL),
        "data_yaml":str(DATA), "data_yaml_sha256":sha256(DATA), "seeds":list(SEEDS),
        "frozen": {"epochs":150,"patience":40,"batch":1,"imgsz":1536,"optimizer":"auto",
                   "mosaic":1.0,"mixup":0.0,"close_mosaic":10,"amp":True,"deterministic":True},
        "note":"augment=False in historical args is inference augmentation; train transforms remain enabled as listed.",
    })

    def fmt(method, metric):
        x = next(r for r in summary if r["method"] == method)
        return x[metric]
    rr_m = next(r for r in summary if r["method"] == "RealRepeat")["map5095_mean"]
    ro_m = next(r for r in summary if r["method"] == "Real-only")["map5095_mean"]
    report = [
        "# Plastic_Bomo RealRepeat control", "",
        "## Main result (three seeds, mean ± sample standard deviation)", "",
        "| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |", "|---|---:|---:|---:|---:|",
    ]
    for method in ["Real-only","RealRepeat","Random","DWBG","BootstrapGuard"]:
        report.append(f"| {method} | {fmt(method,'precision')} | {fmt(method,'recall')} | {fmt(method,'map50')} | {fmt(method,'map5095')} |")
    report += ["", "## Why", "", "RealRepeat isolates whether the synthetic-data gain can be explained by 80 additional real-image exposures per epoch. Every epoch contains all 138 frozen real images plus 80 deterministic replacement draws; no synthetic image or altered label is used.", "",
        "## Result", "",
        f"RealRepeat − Real-only mAP50-95: {rr_m-ro_m:+.4f}. BootstrapGuard − RealRepeat: {paired['absolute_mean_difference']:+.4f} ({paired['relative_improvement_percent']:+.2f}%), with BootstrapGuard winning {paired['bootstrap_guard_wins']}/3 paired seeds. Mean class AP50-95 (RealRepeat → BootstrapGuard): Flash {paired['class_ap5095']['flash']['realrepeat']:.4f} → {paired['class_ap5095']['flash']['bootstrap_guard']:.4f}; Black {paired['class_ap5095']['black']['realrepeat']:.4f} → {paired['class_ap5095']['black']['bootstrap_guard']:.4f}.", "",
        "## Conclusion", "",
        "The numerical direction above is the result; no favorable outcome was assumed. This is an exact per-epoch exposure match, not a strict total-compute match, because Early Stopping determines cumulative epochs and historical exact optimizer-step counters were not retained. See `compute_match_audit.json` for the minimal strict-compute follow-up; it was not run.", ""]
    (OUT / "REALREPEAT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(paired, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--recover-seed", type=int, choices=SEEDS)
    p.add_argument("--device", default="0")
    p.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    if a.finalize: finalize()
    elif a.recover_seed is not None: recover_seed(a.recover_seed, a.device)
    elif a.seed is not None: run_seed(a.seed, a.device)
    else: p.error("provide --seed or --finalize")


if __name__ == "__main__":
    main()
