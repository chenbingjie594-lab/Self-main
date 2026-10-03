"""Train exactly three BalancedMorph50 runs; retain raw metrics and runtime budgets."""
from __future__ import annotations
import argparse
import csv
import shutil
from pathlib import Path
from plastic_bomo_stage15b_common import (SEEDS, dataset_layout, digest, frozen_save, image_files, label_dir,
    load, population, save, sha, verify_preparation)

def metric(result):
    b = result.box
    indexes = {int(cid): i for i, cid in enumerate(b.ap_class_index)}
    if set(indexes) != {0,1}: raise RuntimeError("STAGE15B_VALIDATION_CLASS_METRICS_MISSING")
    return {"precision": float(b.mp), "recall": float(b.mr), "map50": float(b.map50), "map50_95": float(b.map),
            "per_class": {c: {"ap50": float(b.ap50[indexes[i]]), "ap50_95": float(b.ap[indexes[i]]),
                              "recall": float(b.r[indexes[i]])} for i,c in enumerate(("flash", "black"))},
            "raw_results_dict": {k: float(v) for k,v in result.results_dict.items()}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "prepared", "stage15a", "dataset_root", "model", "runs", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--device", default="0")
    a = p.parse_args(); cfg = load(a.protocol); det = cfg["detector"]; a.runs = a.runs.resolve()
    manifest = verify_preparation(a.prepared, cfg); assets = load(a.prepared/"dataset_asset_audit.json")
    features = load(a.prepared/"feature_diversity_audit.json")
    if features["status"] != "COMPLETE_BEFORE_NEW_DETECTOR_TRAINING" or features["manifest_content_sha256"] != digest(manifest):
        raise RuntimeError("STAGE15B_TRAINING_FREE_AUDIT_REQUIRED")
    yaml_path = (a.dataset_root/"M50/data.yaml").resolve(); images, val, _ = dataset_layout(yaml_path)
    if len(image_files(images)) != 218: raise RuntimeError("STAGE15B_TRAIN_COUNT_MISMATCH")
    if len(assets["synthetic_assets"]) != 80: raise RuntimeError("STAGE15B_SYNTHETIC_ASSET_AUDIT_INCOMPLETE")
    for asset, row in zip(assets["synthetic_assets"], manifest["records"]):
        if asset["candidate_id"] != row["candidate_id"] or sha(row["image_path"]) != asset["image_sha256"] or sha(row["label_path"]) != asset["label_sha256"]:
            raise RuntimeError("STAGE15B_SYNTHETIC_ASSET_CHANGED")
    if digest(population(images)) != assets["dataset_train_content_sha256"]: raise RuntimeError("STAGE15B_TRAIN_IMAGE_OR_LABEL_CHANGED")
    if digest(population(val)) != assets["official_val_integrity_sha256"]: raise RuntimeError("STAGE15B_VALIDATION_CHANGED")
    old_rows = load(a.stage15a/"factorial_all_arms_metrics.json")["records"]
    if len(old_rows) != 12 or len({(x["seed"], x["arm"]) for x in old_rows}) != 12:
        raise RuntimeError("STAGE15B_REUSED_METRICS_INCOMPLETE")
    a.output.mkdir(parents=True, exist_ok=True)
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    frozen_save(a.output/"training_initialization_audit.json", {"pretrained_path": str(a.model), "pretrained_sha256": sha(a.model),
        "protocol_content_sha256": digest(cfg), "manifest_content_sha256": digest(manifest),
        "reused_metrics_content_sha256": digest(old_rows), "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__})
    records = []; budget_runs = {}; usage = []; environment = []
    for seed in SEEDS:
        run = a.runs/f"m50_s{seed}"; last = run/"weights/last.pt"
        ledger = a.output/f"epoch_exposure_m50_s{seed}.json"; raw_metrics = a.output/f"m50_s{seed}_raw_metrics.json"
        state = {"epochs": [], "successful_optimizer_steps": 0, "optimizer_step_attempts": 0,
                 "suppressed_framework_validation_calls": 0, "validation_model_forwards_during_training": 0}
        class Trainer(DetectionTrainer):
            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                state["epoch_batches"] += 1; state["epoch_draws"] += int(batch["img"].shape[0])
                return batch
            def validate(self):
                # Some Ultralytics versions force final-epoch validation despite val=False.
                # Keep fitness constant and perform no model forward on official validation.
                state["suppressed_framework_validation_calls"] += 1
                return {}, 0.0
            def final_eval(self):
                # Explicit last.pt evaluation occurs below only after 150 epochs.
                return None
        def start(trainer):
            original = trainer.optimizer.step
            def step(*args, **kwargs):
                result = original(*args, **kwargs); state["successful_optimizer_steps"] += 1; return result
            trainer.optimizer.step = step
            attempt = trainer.optimizer_step
            def attempted(*args, **kwargs):
                state["optimizer_step_attempts"] += 1; return attempt(*args, **kwargs)
            trainer.optimizer_step = attempted
        def epoch_start(trainer):
            state["epoch_batches"] = 0; state["epoch_draws"] = 0
        def epoch_end(trainer):
            state["epochs"].append({"epoch": int(trainer.epoch)+1, "batches": state["epoch_batches"], "primary_image_draws": state["epoch_draws"],
                "cumulative_successful_optimizer_steps": state["successful_optimizer_steps"], "cumulative_optimizer_step_attempts": state["optimizer_step_attempts"]})
            save(ledger, {k:v for k,v in state.items() if not k.startswith("epoch_")})
        if not last.exists():
            if run.exists(): raise RuntimeError("STAGE15B_INCOMPLETE_RUN_EXISTS: "+str(run)+"; preserve the directory and choose a clean --runs root; do not reuse a partial checkpoint")
            model = YOLO(str(a.model)); model.add_callback("on_train_start", start); model.add_callback("on_train_epoch_start", epoch_start); model.add_callback("on_train_epoch_end", epoch_end)
            model.train(data=str(yaml_path), imgsz=det["imgsz"], epochs=det["epochs"], patience=det["patience"], batch=det["batch"],
                deterministic=det["deterministic"], seed=seed, optimizer=det["optimizer"], rect=False, augment=False,
                mosaic=det["mosaic"], mixup=det["mixup"], copy_paste=det["copy_paste"], close_mosaic=det["close_mosaic"],
                val=False, device=a.device, project=str(a.runs), name=run.name, exist_ok=False, trainer=Trainer)
            save(ledger, {k:v for k,v in state.items() if not k.startswith("epoch_")})
        if not ledger.exists(): raise RuntimeError("STAGE15B_CHECKPOINT_WITHOUT_RUNTIME_AUDIT")
        state = load(ledger)
        if len(state["epochs"]) != 150 or any(x["batches"] != 218 or x["primary_image_draws"] != 218 for x in state["epochs"]):
            raise RuntimeError("STAGE15B_RUNTIME_BUDGET_MISMATCH")
        csv_path = run/"results.csv"
        with csv_path.open(encoding="utf-8") as stream: csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != 150: raise RuntimeError("STAGE15B_TRAINING_CSV_EPOCH_MISMATCH")
        import yaml
        actual_args = yaml.safe_load((run/"args.yaml").read_text())
        expected = {k: det[k] for k in ("imgsz", "epochs", "patience", "batch", "optimizer", "mosaic", "mixup", "copy_paste", "close_mosaic", "deterministic")}
        expected.update({"seed": seed, "val": False, "rect": False, "augment": False})
        if any(actual_args.get(k) != v for k,v in expected.items()): raise RuntimeError("STAGE15B_ARGS_PROTOCOL_MISMATCH")
        shutil.copyfile(run/"args.yaml", a.output/f"m50_s{seed}_args.yaml"); shutil.copyfile(csv_path, a.output/f"m50_s{seed}_training_results.csv")
        cached_before_evaluation = raw_metrics.exists()
        if cached_before_evaluation:
            row = load(raw_metrics)
            if row["checkpoint_sha256"] != sha(last) or row["manifest_content_sha256"] != digest(manifest): raise RuntimeError("STAGE15B_METRICS_CHECKPOINT_CHANGED")
        else:
            result = YOLO(str(last)).val(data=str(yaml_path), split="val", imgsz=det["imgsz"], batch=1, device=a.device,
                verbose=False, plots=False, project=str(a.runs/"final_validation"), name=f"m50_s{seed}", exist_ok=True)
            row = {"arm": "M50", "seed": seed, "checkpoint": "epoch150 last.pt", "checkpoint_sha256": sha(last),
                "manifest_content_sha256": digest(manifest), **metric(result)}
            save(raw_metrics, row)
        records.append(row)
        budget_runs[str(seed)] = {"epochs": len(state["epochs"]), "primary_draws": sum(x["primary_image_draws"] for x in state["epochs"]),
            "batches": sum(x["batches"] for x in state["epochs"]), "optimizer_step_attempts": state["optimizer_step_attempts"],
            "successful_optimizer_steps": state["successful_optimizer_steps"], "amp_guarded_skips": state["optimizer_step_attempts"]-state["successful_optimizer_steps"]}
        usage.append({"arm": "M50", "seed": seed, "training_validation_model_forwards": state["validation_model_forwards_during_training"],
            "suppressed_framework_validation_calls": state["suppressed_framework_validation_calls"], "final_last_evaluation_uses": 1,
            "best_pt_used": False, "metric_cache_reused": cached_before_evaluation})
        environment.append({"seed": seed, "actual_args": actual_args, "args_sha256": sha(run/"args.yaml"), "training_csv_sha256": sha(csv_path)})
        print(f"STAGE15B_M50_SEED_{seed}_COMPLETE", flush=True)
    if digest(population(val)) != assets["official_val_integrity_sha256"]: raise RuntimeError("STAGE15B_VALIDATION_CHANGED_DURING_TRAINING")
    save(a.output/"balanced_morph50_metrics.json", {"records": records})
    save(a.output/"all_five_arms_metrics.json", {"records": [*old_rows, *records], "existing_arms_retrained": False})
    budget = load(a.prepared/"training_budget_equivalence.json"); budget.update({"status": "SCHEDULE_EQUIVALENT_WITH_M50_RUNTIME_VERIFIED", "runtime_audit_pending": False,
        "M50_runtime": budget_runs, "historical_arms": "Stage15A frozen scheduled budget; historical runtime counters unavailable", "historical_equal_successful_optimizer_steps_verified": False})
    save(a.output/"training_budget_equivalence.json", budget)
    save(a.output/"validation_usage_audit.json", {"status": "FINAL_LAST_ONLY", "new_runs": usage, "selection_uses": 0,
        "threshold_tuning_uses": 0, "validation_integrity_sha256": assets["official_val_integrity_sha256"], "existing_arms": "Stage15A metrics reused without new validation"})
    save(a.output/"training_environment_audit.json", {"status": "M50_ARGS_CHECKED_AGAINST_FROZEN_STAGE15A_PROTOCOL", "new_runs": environment,
        "initialization": load(a.output/"training_initialization_audit.json"), "historical_pretrained_byte_equality": "not independently verified; prior stage recorded only initialization protocol",
        "forced_framework_validation": "trainer validate/final_eval bypassed to enforce frozen val=False protocol", "hyperparameters_changed": False})
    print("STAGE15B_THREE_NEW_RUNS_COMPLETE")

if __name__ == "__main__": main()
