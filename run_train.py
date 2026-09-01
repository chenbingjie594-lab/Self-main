import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Training script for DreamBooth Inpainting.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory (e.g., ./mvtecad)")
    parser.add_argument("--defect_class", type=str, default=None, help="Specific defect class (e.g., color)")
    parser.add_argument("--category", type=str, nargs="+", default=None)
    parser.add_argument("--soft_mask", default=None, action="store_true")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./sd2-community/stable-diffusion-2-inpainting",
        help="Local Stable Diffusion inpainting directory or Hugging Face model id.",
    )
    parser.add_argument("--output_root", type=str, default="./model")
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks defect",
        help="Prompt used for real defect training images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo of defect",
        help="Prompt used by the prior-preservation branch.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default to reduce VRAM).",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use bitsandbytes 8-bit Adam. Requires: pip install bitsandbytes.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with later defect classes if one training process fails.",
    )
    parser.add_argument(
        "--text_noise_scale",
        type=float,
        default=1.0,
        help="Standard deviation of random noise added to encA/encB",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {base_dir}")

    model_arg = args.pretrained_model_name_or_path
    local_model = Path(model_arg).expanduser()
    if local_model.exists():
        model_arg = str(local_model.resolve())
        required = ["model_index.json", "tokenizer", "text_encoder", "vae", "unet", "scheduler"]
        missing = [name for name in required if not (Path(model_arg) / name).exists()]
        if missing:
            parser.error(f"Local base model is incomplete ({model_arg}); missing: {', '.join(missing)}")

    categories = args.category if args.category is not None else [p.name for p in base_dir.iterdir() if p.is_dir()]

    for category in categories:
        defect_classes = (
            [args.defect_class]
            if args.defect_class
            else [
                d
                for d in os.listdir(base_dir / category / "ground_truth")
                if (base_dir / category / "ground_truth" / d).is_dir()
            ]
        )

        for defect in defect_classes:
            instance_data_dir = base_dir / category / "test" / defect
            class_data_dir = instance_data_dir
            mask_data_dir = base_dir / category / "ground_truth" / defect
            output_dir = Path(args.output_root) / f"{args.output_name}_noise_{args.text_noise_scale}" / category / defect

            for label, directory in (("images", instance_data_dir), ("masks", mask_data_dir)):
                if not directory.is_dir():
                    raise FileNotFoundError(f"{label} directory does not exist: {directory}")

            command = [
                sys.executable, "train_dreambooth_noise.py",
                f"--pretrained_model_name_or_path={model_arg}",
                f"--instance_data_dir={instance_data_dir}",
                f"--class_data_dir={class_data_dir}",
                f"--output_dir={output_dir}",
                "--prior_loss_weight=1.0",
                f"--instance_prompt={args.instance_prompt}",
                f"--class_prompt={args.class_prompt}",
                "--resolution=512",
                f"--train_batch_size={args.train_batch_size}",
                f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
                f"--learning_rate={args.learning_rate}",
                "--lr_scheduler=constant",
                "--lr_warmup_steps=0",
                "--num_class_images=200",
                f"--max_train_steps={args.max_train_steps}",
                f"--text_noise_scale={args.text_noise_scale}",
                f"--mixed_precision={args.mixed_precision}",
                "--mvtecad",
                f"--mask_data_dir={mask_data_dir}",
                "--center_crop",
            ]

            if args.soft_mask:
                command.append("--soft_mask")
            if not args.no_gradient_checkpointing:
                command.append("--gradient_checkpointing")
            if args.use_8bit_adam:
                command.append("--use_8bit_adam")

            print(f"Running command for category '{category}', defect '{defect}'.")
            print(f"Base model: {model_arg}")
            child_env = os.environ.copy()
            child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            result = subprocess.run(command, check=False, env=child_env)
            if result.returncode != 0:
                message = f"Training failed for {category}/{defect} (exit code {result.returncode})"
                if args.continue_on_error:
                    print(f"[ERROR] {message}; continuing because --continue_on_error is set.")
                else:
                    raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()

