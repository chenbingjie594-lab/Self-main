"""Train MSDF-only checkpoints for every selected defect class."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--defect_class", default=None)
    parser.add_argument("--output_root", default="./model/msdf_v3_sd2_2000_s42")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--unet_learning_rate", type=float, default=5e-6)
    parser.add_argument("--msdf_learning_rate", type=float, default=1e-4)
    parser.add_argument("--msdf_hidden_dim", type=int, default=64)
    parser.add_argument("--msdf_context_scale", type=float, default=1.75)
    parser.add_argument("--msdf_max_injection", type=float, default=0.75)
    parser.add_argument("--msdf_max_residual_ratio", type=float, default=0.25)
    parser.add_argument("--msdf_branch_dropout", type=float, default=0.20)
    parser.add_argument("--msdf_pixel_support_weight", type=float, default=0.50)
    parser.add_argument("--msdf_region_weight", type=float, default=0.75)
    parser.add_argument("--msdf_reconstruction_weight", type=float, default=0.50)
    parser.add_argument("--msdf_structure_weight", type=float, default=0.25)
    parser.add_argument("--msdf_background_weight", type=float, default=0.20)
    parser.add_argument("--msdf_scale_weight", type=float, default=0.25)
    parser.add_argument("--msdf_support_weight", type=float, default=0.25)
    parser.add_argument("--mask_jitter_radius", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max_nonfinite_gradient_skips", type=int, default=20)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    if "msdf_v2" in str(args.output_root).lower():
        parser.error(
            "MSDF v3 cannot use an msdf_v2 output directory. Use, for example, "
            "--output_root ./model/msdf_v3_sd2_2000_s42"
        )

    base_dir = Path(args.base_dir)
    model_path = Path(args.pretrained_model_name_or_path).expanduser()
    if not model_path.is_dir():
        parser.error(f"Pretrained model directory does not exist: {model_path}")
    required = ["model_index.json", "tokenizer", "text_encoder", "vae", "unet", "scheduler"]
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        parser.error(f"Pretrained model is incomplete; missing: {', '.join(missing)}")

    for category in args.categories:
        gt_root = base_dir / category / "ground_truth"
        if not gt_root.is_dir():
            parser.error(f"Ground-truth directory does not exist: {gt_root}")
        defects = (
            [args.defect_class]
            if args.defect_class
            else sorted(path.name for path in gt_root.iterdir() if path.is_dir())
        )
        for defect in defects:
            command = [
                sys.executable,
                "train_rda.py",
                "--model_path", str(model_path.resolve()),
                "--image_dir", str(base_dir / category / "test" / defect),
                "--mask_dir", str(gt_root / defect),
                "--output_dir", str(Path(args.output_root) / category / defect),
                "--rda_mode", "none",
                "--enable_msdf",
                "--train_steps", str(args.train_steps),
                "--batch_size", str(args.batch_size),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
                "--unet_learning_rate", str(args.unet_learning_rate),
                "--msdf_learning_rate", str(args.msdf_learning_rate),
                "--msdf_hidden_dim", str(args.msdf_hidden_dim),
                "--msdf_context_scale", str(args.msdf_context_scale),
                "--msdf_max_injection", str(args.msdf_max_injection),
                "--msdf_max_residual_ratio", str(args.msdf_max_residual_ratio),
                "--msdf_branch_dropout", str(args.msdf_branch_dropout),
                "--msdf_pixel_support_weight", str(args.msdf_pixel_support_weight),
                "--msdf_region_weight", str(args.msdf_region_weight),
                "--msdf_reconstruction_weight", str(args.msdf_reconstruction_weight),
                "--msdf_structure_weight", str(args.msdf_structure_weight),
                "--msdf_background_weight", str(args.msdf_background_weight),
                "--msdf_scale_weight", str(args.msdf_scale_weight),
                "--msdf_support_weight", str(args.msdf_support_weight),
                "--mask_jitter_radius", str(args.mask_jitter_radius),
                "--seed", str(args.seed),
                "--mixed_precision", args.mixed_precision,
                "--max_nonfinite_gradient_skips", str(args.max_nonfinite_gradient_skips),
            ]
            if args.use_8bit_adam:
                command.append("--use_8bit_adam")
            if args.no_gradient_checkpointing:
                command.append("--no_gradient_checkpointing")
            print(f"[MSDF] Training {category}/{defect}")
            environment = os.environ.copy()
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode != 0 and not args.continue_on_error:
                raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
