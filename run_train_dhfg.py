"""Train DHFG-only or MSDF+DHFG checkpoints for selected defect classes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--defect_class", default=None)
    parser.add_argument(
        "--class_map_json",
        default=None,
        help='Optional mapping such as {"01_Flash_point": 0, "02_Big_black_spots": 1}.',
    )
    parser.add_argument("--enable_msdf", action="store_true")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--unet_learning_rate", type=float, default=5e-6)
    parser.add_argument("--dhfg_weight", type=float, default=0.10)
    parser.add_argument("--dhfg_feature_weight", type=float, default=0.25)
    parser.add_argument("--dhfg_response_weight", type=float, default=0.75)
    parser.add_argument("--dhfg_confidence_weight", type=float, default=0.50)
    parser.add_argument("--dhfg_polarity_weight", type=float, default=0.50)
    parser.add_argument("--dhfg_shape_weight", type=float, default=0.25)
    parser.add_argument("--dhfg_boundary_weight", type=float, default=0.50)
    parser.add_argument("--dhfg_background_weight", type=float, default=0.10)
    parser.add_argument("--dhfg_boundary_radius", type=int, default=8)
    parser.add_argument("--dhfg_context_scale", type=float, default=4.0)
    parser.add_argument("--dhfg_minimum_extent_ratio", type=float, default=0.25)
    parser.add_argument("--dhfg_max_timestep", type=int, default=200)
    parser.add_argument("--dhfg_interval", type=int, default=1)
    parser.add_argument("--dhfg_warmup_steps", type=int, default=200)
    parser.add_argument("--dhfg_calibration_samples", type=int, default=64)
    parser.add_argument("--dhfg_object_topk_fraction", type=float, default=0.05)
    parser.add_argument("--dhfg_object_topk_max", type=int, default=4)
    parser.add_argument("--dhfg_background_topk_fraction", type=float, default=0.01)
    parser.add_argument("--dhfg_background_topk_max", type=int, default=16)
    parser.add_argument("--dhfg_min_teacher_probability", type=float, default=0.01)
    parser.add_argument("--dhfg_confidence_beta", type=float, default=1.0)
    parser.add_argument("--dhfg_contrast_threshold", type=float, default=1.0)
    parser.add_argument("--dhfg_contrast_temperature", type=float, default=0.25)
    parser.add_argument("--dhfg_erasure_radius", type=int, default=4)
    parser.add_argument("--dhfg_erasure_blur_radius", type=int, default=8)
    parser.add_argument("--dhfg_delta_tolerance", type=float, default=0.25)
    parser.add_argument("--dhfg_minimum_teacher_delta", type=float, default=0.01)
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

    if args.output_root is None:
        args.output_root = (
            "./model/msdf_dhfg_v5_sd2_2000_s42"
            if args.enable_msdf
            else "./model/dhfg_v5_sd2_2000_s42"
        )

    base_dir = Path(args.base_dir)
    model_path = Path(args.pretrained_model_name_or_path).expanduser()
    teacher_path = Path(args.teacher_path).expanduser()
    if not model_path.is_dir():
        parser.error(f"Pretrained model directory does not exist: {model_path}")
    if not teacher_path.is_file():
        parser.error(f"DHFG teacher does not exist: {teacher_path}")
    class_map = {}
    if args.class_map_json:
        class_map = json.loads(Path(args.class_map_json).read_text(encoding="utf-8"))

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
                "--enable_dhfg",
                "--dhfg_teacher_path", str(teacher_path.resolve()),
                "--dhfg_class_name", defect,
                "--train_steps", str(args.train_steps),
                "--batch_size", str(args.batch_size),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
                "--unet_learning_rate", str(args.unet_learning_rate),
                "--dhfg_weight", str(args.dhfg_weight),
                "--dhfg_feature_weight", str(args.dhfg_feature_weight),
                "--dhfg_response_weight", str(args.dhfg_response_weight),
                "--dhfg_confidence_weight", str(args.dhfg_confidence_weight),
                "--dhfg_polarity_weight", str(args.dhfg_polarity_weight),
                "--dhfg_shape_weight", str(args.dhfg_shape_weight),
                "--dhfg_boundary_weight", str(args.dhfg_boundary_weight),
                "--dhfg_background_weight", str(args.dhfg_background_weight),
                "--dhfg_boundary_radius", str(args.dhfg_boundary_radius),
                "--dhfg_context_scale", str(args.dhfg_context_scale),
                "--dhfg_minimum_extent_ratio", str(args.dhfg_minimum_extent_ratio),
                "--dhfg_max_timestep", str(args.dhfg_max_timestep),
                "--dhfg_interval", str(args.dhfg_interval),
                "--dhfg_warmup_steps", str(args.dhfg_warmup_steps),
                "--dhfg_calibration_samples", str(args.dhfg_calibration_samples),
                "--dhfg_object_topk_fraction", str(args.dhfg_object_topk_fraction),
                "--dhfg_object_topk_max", str(args.dhfg_object_topk_max),
                "--dhfg_background_topk_fraction", str(args.dhfg_background_topk_fraction),
                "--dhfg_background_topk_max", str(args.dhfg_background_topk_max),
                "--dhfg_min_teacher_probability", str(args.dhfg_min_teacher_probability),
                "--dhfg_confidence_beta", str(args.dhfg_confidence_beta),
                "--dhfg_contrast_threshold", str(args.dhfg_contrast_threshold),
                "--dhfg_contrast_temperature", str(args.dhfg_contrast_temperature),
                "--dhfg_erasure_radius", str(args.dhfg_erasure_radius),
                "--dhfg_erasure_blur_radius", str(args.dhfg_erasure_blur_radius),
                "--dhfg_delta_tolerance", str(args.dhfg_delta_tolerance),
                "--dhfg_minimum_teacher_delta", str(args.dhfg_minimum_teacher_delta),
                "--mask_jitter_radius", str(args.mask_jitter_radius),
                "--seed", str(args.seed),
                "--mixed_precision", args.mixed_precision,
                "--max_nonfinite_gradient_skips", str(args.max_nonfinite_gradient_skips),
            ]
            if defect in class_map:
                command.extend(["--dhfg_class_id", str(class_map[defect])])
            if args.enable_msdf:
                command.extend(
                    [
                        "--enable_msdf",
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
                    ]
                )
            if args.use_8bit_adam:
                command.append("--use_8bit_adam")
            if args.no_gradient_checkpointing:
                command.append("--no_gradient_checkpointing")
            label = "MSDF+DHFG" if args.enable_msdf else "DHFG"
            print(f"[{label}] Training {category}/{defect}")
            environment = os.environ.copy()
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode != 0 and not args.continue_on_error:
                raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
