#!/usr/bin/env python3
"""Audit completed Flash/Black MSDF-v3 formal checkpoints."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("no_latent", "no_morph"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_pairs = {"01_Flash_point": 76, "02_Big_black_spots": 36}
    report = {"status": "PASS", "variant": args.variant, "classes": {}}
    for class_name, pair_count in expected_pairs.items():
        directory = args.model_root / "Plastic_Bomo" / class_name
        checkpoint_path = directory / "msdf.pt"
        metadata_path = directory / "msdf_training.json"
        history_path = directory / "training_history.csv"
        for path in (checkpoint_path, metadata_path, history_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with history_path.open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        adapter = MSDFAdapter(
            int(checkpoint["latent_channels"]), int(checkpoint["hidden_dim"]),
            checkpoint["block_channels"], float(checkpoint["context_scale"]),
            float(checkpoint["max_injection"]),
            float(checkpoint["max_residual_ratio"]),
            float(checkpoint["branch_dropout"]),
            float(checkpoint["pixel_support_weight"]),
            training_ablation=checkpoint["training_ablation"],
        )
        loaded = adapter.load_state_dict(checkpoint["state_dict"], strict=True)
        finite = all(
            not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
            for value in checkpoint["state_dict"].values()
        )
        projection_norms = {
            name: float(value.float().norm())
            for name, value in checkpoint["state_dict"].items()
            if name.startswith("projections.") and name.endswith(".weight")
        }
        steps_exact = (
            len(history) == 2000
            and [int(row["step"]) for row in history] == list(range(1, 2001))
        )
        support_zero = all(float(row["msdf_support"]) == 0.0 for row in history)
        contract = adapter.training_contract()
        checks = {
            "successful_steps_exact_1_to_2000": steps_exact,
            "training_pair_count": metadata.get("num_training_pairs") == pair_count,
            "variant_metadata": metadata.get("msdf_training_ablation") == args.variant,
            "variant_checkpoint": checkpoint.get("training_ablation") == args.variant,
            "strict_load_no_missing": not loaded.missing_keys,
            "strict_load_no_unexpected": not loaded.unexpected_keys,
            "all_tensors_finite": finite,
            "four_projection_weights_nonzero": (
                len(projection_norms) == 4 and all(value > 0 for value in projection_norms.values())
            ),
            "no_latent_encoder_frozen": (
                all(not parameter.requires_grad for parameter in adapter.encoder.parameters())
                if args.variant == "no_latent" else True
            ),
            "no_morph_support_loss_zero_all_steps": (
                support_zero if args.variant == "no_morph" else True
            ),
            "no_morph_contract": (
                not contract["morphology_alignment"]
                and not contract["morphology_support_loss"]
                if args.variant == "no_morph" else True
            ),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        if status == "FAIL":
            report["status"] = "FAIL"
        report["classes"][class_name] = {
            "status": status,
            "successful_steps": len(history),
            "training_pairs": metadata.get("num_training_pairs"),
            "nonfinite_gradient_skips_total": metadata.get("nonfinite_gradient_skips_total"),
            "training_contract": contract,
            "projection_weight_norms": projection_norms,
            "checkpoint_sha256": sha(checkpoint_path),
            "history_sha256": sha(history_path),
            "metadata_sha256": sha(metadata_path),
            "checks": checks,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
