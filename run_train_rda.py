import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        required=True,
        help="Original Stable Diffusion inpainting model used by every defect class.",
    )
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--rda_mode",
        choices=["full", "loss_only", "none"],
        default="full",
        help="Use loss_only for the E1 auxiliary-loss control experiment.",
    )
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--defect_class", default=None)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--unet_learning_rate", type=float, default=5e-6)
    parser.add_argument("--rda_learning_rate", type=float, default=1e-4)
    parser.add_argument("--enable_carf", action="store_true")
    parser.add_argument("--enable_msdf", action="store_true")
    parser.add_argument("--msdf_learning_rate", type=float, default=1e-4)
    parser.add_argument("--msdf_hidden_dim", type=int, default=64)
    parser.add_argument("--msdf_context_scale", type=float, default=1.75)
    parser.add_argument("--msdf_max_injection", type=float, default=1.0)
    parser.add_argument("--msdf_region_weight", type=float, default=0.75)
    parser.add_argument("--msdf_reconstruction_weight", type=float, default=0.50)
    parser.add_argument("--msdf_structure_weight", type=float, default=0.25)
    parser.add_argument("--msdf_background_weight", type=float, default=0.20)
    parser.add_argument("--msdf_scale_weight", type=float, default=0.25)
    parser.add_argument("--msdf_support_weight", type=float, default=0.25)
    parser.add_argument("--carf_learning_rate", type=float, default=1e-4)
    parser.add_argument("--carf_hidden_dim", type=int, default=64)
    parser.add_argument("--carf_weight", type=float, default=0.5)
    parser.add_argument("--carf_consistency_weight", type=float, default=0.25)
    parser.add_argument("--carf_min_support", type=float, default=0.35)
    parser.add_argument("--carf_max_core_adjustment", type=float, default=0.75)
    parser.add_argument("--carf_max_context_adjustment", type=float, default=0.50)
    parser.add_argument("--carf_coverage_weight", type=float, default=0.50)
    parser.add_argument("--carf_min_core_coverage", type=float, default=0.90)
    parser.add_argument("--carf_region_weight", type=float, default=0.50)
    parser.add_argument("--carf_reconstruction_weight", type=float, default=0.50)
    parser.add_argument("--carf_structure_weight", type=float, default=0.20)
    parser.add_argument("--carf_background_weight", type=float, default=0.20)
    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--token_grid", type=int, default=4)
    parser.add_argument("--max_token_scale", type=float, default=0.25)
    parser.add_argument("--distribution_weight", type=float, default=0.25)
    parser.add_argument("--background_weight", type=float, default=0.25)
    parser.add_argument("--region_weight", type=float, default=1.0)
    parser.add_argument("--conditional_weight", type=float, default=1.0)
    parser.add_argument("--latent_reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--structure_weight", type=float, default=0.25)
    parser.add_argument(
        "--unet_train_scope",
        choices=["attention", "full", "frozen"],
        default="full",
    )
    parser.add_argument("--saliency_low_quantile", type=float, default=0.60)
    parser.add_argument("--saliency_high_quantile", type=float, default=0.95)
    parser.add_argument("--mask_jitter_radius", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", default="fp16")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    if args.output_root is None:
        if args.enable_msdf:
            args.output_root = "./model/msdf_v2_sd2_2000_s42"
        elif args.enable_carf and args.rda_mode == "none":
            args.output_root = "./model/carf_only_v4_sd2_2000_s42"
        elif args.enable_carf:
            args.output_root = "./model/rda_carf_v3_sd2_2000_s42"
        else:
            args.output_root = (
                "./model/rda_v4_sd2_2000_s42"
                if args.rda_mode == "full"
                else "./model/rda_e1_loss_only_sd2_2000_s42"
            )

    base_dir = Path(args.base_dir)
    model_path = Path(args.pretrained_model_name_or_path).expanduser()
    if not model_path.is_dir():
        parser.error(f"Pretrained model directory does not exist: {model_path}")
    required = ["model_index.json", "tokenizer", "text_encoder", "vae", "unet", "scheduler"]
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        parser.error(
            f"Pretrained model is incomplete; missing: {', '.join(missing)}"
        )
    model_path = model_path.resolve()
    for category in args.categories:
        gt_root = base_dir / category / "ground_truth"
        defects = (
            [args.defect_class]
            if args.defect_class
            else sorted(p.name for p in gt_root.iterdir() if p.is_dir())
        )
        for defect in defects:
            command = [
                sys.executable,
                "train_rda.py",
                "--model_path",
                str(model_path),
                "--image_dir",
                str(base_dir / category / "test" / defect),
                "--mask_dir",
                str(gt_root / defect),
                "--output_dir",
                str(Path(args.output_root) / category / defect),
                "--rda_mode",
                args.rda_mode,
                "--train_steps",
                str(args.train_steps),
                "--batch_size",
                str(args.batch_size),
                "--gradient_accumulation_steps",
                str(args.gradient_accumulation_steps),
                "--unet_learning_rate",
                str(args.unet_learning_rate),
                "--rda_learning_rate",
                str(args.rda_learning_rate),
                "--carf_learning_rate",
                str(args.carf_learning_rate),
                "--msdf_learning_rate",
                str(args.msdf_learning_rate),
                "--msdf_hidden_dim",
                str(args.msdf_hidden_dim),
                "--msdf_context_scale",
                str(args.msdf_context_scale),
                "--msdf_max_injection",
                str(args.msdf_max_injection),
                "--msdf_region_weight",
                str(args.msdf_region_weight),
                "--msdf_reconstruction_weight",
                str(args.msdf_reconstruction_weight),
                "--msdf_structure_weight",
                str(args.msdf_structure_weight),
                "--msdf_background_weight",
                str(args.msdf_background_weight),
                "--msdf_scale_weight",
                str(args.msdf_scale_weight),
                "--msdf_support_weight",
                str(args.msdf_support_weight),
                "--carf_hidden_dim",
                str(args.carf_hidden_dim),
                "--carf_weight",
                str(args.carf_weight),
                "--carf_consistency_weight",
                str(args.carf_consistency_weight),
                "--carf_min_support",
                str(args.carf_min_support),
                "--carf_max_core_adjustment",
                str(args.carf_max_core_adjustment),
                "--carf_max_context_adjustment",
                str(args.carf_max_context_adjustment),
                "--carf_coverage_weight",
                str(args.carf_coverage_weight),
                "--carf_min_core_coverage",
                str(args.carf_min_core_coverage),
                "--carf_region_weight",
                str(args.carf_region_weight),
                "--carf_reconstruction_weight",
                str(args.carf_reconstruction_weight),
                "--carf_structure_weight",
                str(args.carf_structure_weight),
                "--carf_background_weight",
                str(args.carf_background_weight),
                "--adapter_hidden_dim",
                str(args.adapter_hidden_dim),
                "--token_grid",
                str(args.token_grid),
                "--max_token_scale",
                str(args.max_token_scale),
                "--distribution_weight",
                str(args.distribution_weight),
                "--background_weight",
                str(args.background_weight),
                "--region_weight",
                str(args.region_weight),
                "--conditional_weight",
                str(args.conditional_weight),
                "--latent_reconstruction_weight",
                str(args.latent_reconstruction_weight),
                "--structure_weight",
                str(args.structure_weight),
                "--unet_train_scope",
                args.unet_train_scope,
                "--saliency_low_quantile",
                str(args.saliency_low_quantile),
                "--saliency_high_quantile",
                str(args.saliency_high_quantile),
                "--mask_jitter_radius",
                str(args.mask_jitter_radius),
                "--seed",
                str(args.seed),
                "--mixed_precision",
                args.mixed_precision,
            ]
            if args.use_8bit_adam:
                command.append("--use_8bit_adam")
            if args.enable_carf:
                command.append("--enable_carf")
            if args.enable_msdf:
                command.append("--enable_msdf")
            if args.no_gradient_checkpointing:
                command.append("--no_gradient_checkpointing")
            print(f"[RDA] Training {category}/{defect}")
            environment = os.environ.copy()
            environment.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode != 0 and not args.continue_on_error:
                raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
