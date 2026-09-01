"""Verify the frozen baseline config and optional dataset split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import enabled_modules, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/baseline_frozen_split70_s42.json",
    )
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def verify_manifest(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    errors = []
    for category, classes in manifest.get("categories", {}).items():
        for defect_class, splits in classes.items():
            train = {entry["image"] for entry in splits["train"]}
            evaluation = {entry["image"] for entry in splits["eval"]}
            overlap = sorted(train & evaluation)
            if overlap:
                errors.append(f"{category}/{defect_class} overlap: {overlap}")
            if not train or not evaluation:
                errors.append(f"{category}/{defect_class} has an empty split")
    if errors:
        raise ValueError("Invalid split manifest:\n" + "\n".join(errors))
    print(f"[OK] Split manifest has no train/eval overlap: {path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    errors = []
    if config["pipeline_mode"] != "custom":
        errors.append(
            "pipeline_mode must be custom so baseline and innovation ablations "
            "share exactly the same pipeline carrier"
        )
    active = enabled_modules(config)
    if active:
        errors.append(f"innovation modules must all be disabled: {active}")
    if config["modules"]["prompt_perturbation"]["noise_scale"] != 0:
        errors.append("prompt perturbation noise_scale must be zero")
    if config["guidance_scale"] != 7.5:
        errors.append("guidance_scale must be 7.5")
    if config["num_inference_steps"] != 50:
        errors.append("num_inference_steps must be 50")
    if config["seed"] != 42:
        errors.append("seed must be 42")
    if errors:
        raise ValueError("Frozen baseline verification failed:\n" + "\n".join(errors))

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    print(f"[OK] Frozen module-off ablation baseline: {args.config}")
    print(f"[OK] Config SHA256: {fingerprint}")
    if args.manifest:
        verify_manifest(args.manifest)


if __name__ == "__main__":
    main()
