"""Train a Reference-Decoupled Anomaly Adapter from an inpainting base model.

The real defect image is the diffusion target. The supplied mask is treated as
an uncertain spatial prior, not as an exact segmentation annotation.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.rda_guidance import (
    RDA_FORMAT_VERSION,
    ReferenceTokenAdapter,
    append_reference_tokens,
)
from diffusers.pipelines.stable_diffusion.carf_guidance import (
    CARF_FORMAT_VERSION,
    CARFRegionRefiner,
    carf_weak_supervision_loss,
)
from diffusers.pipelines.stable_diffusion.rda_carf_attention import (
    install_rda_carf_attention,
)
from diffusers.pipelines.stable_diffusion.msdf_guidance import (
    MSDF_FORMAT_VERSION,
    build_msdf_adapter,
)
from diffusers.pipelines.stable_diffusion.dhfg_guidance import (
    DHFGTeacher,
    crop_mask_regions,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--rda_mode",
        choices=["full", "loss_only", "none"],
        default="full",
        help=(
            "full trains reference tokens and all auxiliary losses; loss_only "
            "disables the reference adapter while keeping the same auxiliary losses; "
            "none removes both the adapter and all RDA-specific losses for a clean "
            "baseline or CARF-only experiment."
        ),
    )
    parser.add_argument("--prompt", default="a photo of a sks defect")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--unet_learning_rate", type=float, default=5e-6)
    parser.add_argument("--rda_learning_rate", type=float, default=1e-4)
    parser.add_argument("--enable_carf", action="store_true")
    parser.add_argument("--enable_msdf", action="store_true")
    parser.add_argument("--enable_dhfg", action="store_true")
    parser.add_argument(
        "--dhfg_teacher_path",
        default=None,
        help="TorchScript teacher exported by tools/export_dhfg_teacher.py.",
    )
    parser.add_argument("--dhfg_class_name", default=None)
    parser.add_argument("--dhfg_class_id", type=int, default=None)
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
    parser.add_argument(
        "--background_weight",
        type=float,
        default=0.25,
        help="Preserve clean latent content outside the uncertain defect region.",
    )
    parser.add_argument("--region_weight", type=float, default=1.0)
    parser.add_argument("--conditional_weight", type=float, default=1.0)
    parser.add_argument("--latent_reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--structure_weight", type=float, default=0.25)
    parser.add_argument(
        "--unet_train_scope",
        choices=["attention", "full", "frozen"],
        default="full",
        help="Controlled UNet adaptation prevents coarse-mask geometry overfitting.",
    )
    parser.add_argument("--saliency_low_quantile", type=float, default=0.60)
    parser.add_argument("--saliency_high_quantile", type=float, default=0.95)
    parser.add_argument("--mask_jitter_radius", type=int, default=24)
    parser.add_argument("--mask_blur_radius", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--max_nonfinite_gradient_skips",
        type=int,
        default=20,
        help=(
            "Maximum consecutive fp16 gradient-overflow steps that GradScaler may skip "
            "before training is treated as genuinely unstable."
        ),
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable UNet gradient checkpointing (enabled by default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument("--use_8bit_adam", action="store_true")
    return parser.parse_args()


def _number_key(path):
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else path.stem


def assert_finite_module(module, label):
    """Refuse to train or save silently corrupted model parameters."""
    invalid = []
    for name, value in module.state_dict().items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            invalid.append(name)
            if len(invalid) >= 8:
                break
    if invalid:
        raise FloatingPointError(
            f"{label} contains NaN/Inf tensors: {', '.join(invalid)}"
        )


class PairedDefectDataset(Dataset):
    def __init__(
        self, image_dir, mask_dir, resolution, reference_cross_probability=0.30
    ):
        images = sorted(
            (p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
            key=_number_key,
        )
        masks = sorted(
            (p for p in Path(mask_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
            key=_number_key,
        )
        if not images or not masks:
            raise ValueError("No training images or masks were found")
        mask_by_stem = {p.stem: p for p in masks}
        pairs = [(p, mask_by_stem[p.stem]) for p in images if p.stem in mask_by_stem]
        if not pairs:
            if len(images) != len(masks):
                raise ValueError(
                    f"Cannot pair {len(images)} images with {len(masks)} masks"
                )
            pairs = list(zip(images, masks))
        self.pairs = pairs
        self.reference_cross_probability = float(reference_cross_probability)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        # Self-reference teaches morphology transfer. Occasional same-class
        # cross-reference prevents exact image copying and retains diversity.
        if (
            len(self.pairs) > 1
            and random.random() < self.reference_cross_probability
        ):
            offset = random.randint(1, len(self.pairs) - 1)
            reference_path, reference_mask_path = self.pairs[(index + offset) % len(self.pairs)]
            reference_is_self = False
        else:
            reference_path, reference_mask_path = image_path, mask_path
            reference_is_self = True
        image = self.image_transform(Image.open(image_path).convert("RGB"))
        mask = self.mask_transform(Image.open(mask_path).convert("L"))
        reference = self.image_transform(Image.open(reference_path).convert("RGB"))
        reference_mask = self.mask_transform(Image.open(reference_mask_path).convert("L"))
        return {
            "image": image,
            "mask": (mask >= 0.5).float(),
            "reference": reference,
            "reference_mask": (reference_mask >= 0.5).float(),
            "reference_is_self": reference_is_self,
        }


def make_soft_prior(mask, latent_size, radius):
    prior = F.interpolate(mask.float(), size=latent_size, mode="nearest")
    radius = max(0, int(radius))
    if radius:
        kernel = 2 * radius + 1
        prior = F.avg_pool2d(prior, kernel, stride=1, padding=radius)
        maximum = prior.flatten(1).amax(1).reshape(-1, 1, 1, 1).clamp_min(1e-6)
        prior = prior / maximum
    return prior.clamp(0, 1)


def augment_coarse_mask(mask, max_radius):
    """Randomly dilate coarse masks so their boundary cannot become a target.

    Erosion is intentionally avoided: with coarse annotations it can expose the
    real defect to the masked-image branch and turn training into copying.
    """
    max_radius = max(0, int(max_radius))
    if max_radius == 0:
        return mask
    radius = random.randint(0, max_radius)
    if radius == 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel, stride=1, padding=radius)


def build_defect_saliency(
    image, coarse_mask, latent_size, low_quantile, high_quantile
):
    """Extract high-confidence defect evidence without using mask boundaries.

    Local RGB residual highlights actual spots/flash structures. Quantiles are
    estimated only inside the coarse region, which remains an uncertain search
    area rather than a pixel-level target.
    """
    # Max aggregation preserves both tiny high-contrast cores and broad diffuse
    # halos. A single fixed window biased CARF v1 toward compact dark spots.
    residuals = []
    for kernel in (7, 15, 31, 63):
        local_background = F.avg_pool2d(
            image.float(), kernel, stride=1, padding=kernel // 2
        )
        residuals.append(
            (image.float() - local_background).abs().mean(dim=1, keepdim=True)
        )
    residual = torch.stack(residuals, dim=0).amax(dim=0)
    saliency = torch.zeros_like(residual)
    for index in range(image.shape[0]):
        values = residual[index][coarse_mask[index] > 0.5]
        if values.numel() < 4:
            continue
        low = torch.quantile(values, float(low_quantile))
        high = torch.quantile(values, float(high_quantile))
        sample = (residual[index] - low) / (high - low).clamp_min(1e-6)
        saliency[index] = sample.clamp(0, 1) * coarse_mask[index]
    saliency = F.interpolate(
        saliency, size=latent_size, mode="bilinear", align_corners=False
    )
    # Emphasize only reliable evidence and discard weak texture response.
    return saliency.square().clamp(0, 1)


def local_distribution_loss(prediction, target, weight):
    """Match GT local contrast and latent-channel statistics inside defects."""
    pred_residual = prediction - F.avg_pool2d(
        prediction, 5, stride=1, padding=2
    )
    target_residual = target - F.avg_pool2d(
        target, 5, stride=1, padding=2
    )
    denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)

    def statistics(value):
        mean = (value * weight).sum(dim=(2, 3), keepdim=True) / denominator
        variance = (
            (value - mean).square() * weight
        ).sum(dim=(2, 3), keepdim=True) / denominator
        magnitude = (value.abs() * weight).sum(
            dim=(2, 3), keepdim=True
        ) / denominator
        return mean, variance.sqrt(), magnitude

    pred_stats = statistics(pred_residual)
    target_stats = statistics(target_residual.detach())
    return sum(
        F.smooth_l1_loss(pred, real)
        for pred, real in zip(pred_stats, target_stats)
    )


def multiscale_local_consistency_loss(prediction, target, weight):
    """Preserve tiny morphology at latent resolution and coarser context."""
    total = prediction.new_zeros(())
    used = 0
    for scale in (1, 2, 4):
        if min(prediction.shape[-2:]) < scale:
            continue
        if scale == 1:
            pred_scale, target_scale, weight_scale = prediction, target, weight
        else:
            pred_scale = F.avg_pool2d(prediction, scale, stride=scale)
            target_scale = F.avg_pool2d(target, scale, stride=scale)
            weight_scale = F.avg_pool2d(weight, scale, stride=scale)
        difference = (pred_scale - target_scale.detach()).abs() * weight_scale
        total = total + difference.sum() / (
            weight_scale.sum() * prediction.shape[1]
        ).clamp_min(1.0)
        used += 1
    return total / max(used, 1)


def encode_prompt(tokenizer, text_encoder, text, batch_size, device):
    ids = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        embedding = text_encoder(ids)[0]
    return embedding.expand(batch_size, -1, -1)


def main():
    args = parse_args()
    if not 0 <= args.saliency_low_quantile < args.saliency_high_quantile <= 1:
        raise ValueError(
            "Require 0 <= saliency_low_quantile < "
            "saliency_high_quantile <= 1"
        )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.max_nonfinite_gradient_skips < 0:
        raise ValueError("max_nonfinite_gradient_skips cannot be negative")
    if args.enable_dhfg and not args.dhfg_teacher_path:
        raise ValueError("--dhfg_teacher_path is required with --enable_dhfg")
    if args.enable_dhfg and not Path(args.dhfg_teacher_path).is_file():
        raise ValueError(f"DHFG teacher does not exist: {args.dhfg_teacher_path}")
    if args.enable_dhfg and args.dhfg_class_id is None and not args.dhfg_class_name:
        raise ValueError("Pass --dhfg_class_name or --dhfg_class_id with DHFG")
    if args.dhfg_interval < 1:
        raise ValueError("dhfg_interval must be at least 1")
    if args.dhfg_calibration_samples < 1:
        raise ValueError("dhfg_calibration_samples must be at least 1")
    dhfg_weights = {
        "dhfg_weight": args.dhfg_weight,
        "dhfg_feature_weight": args.dhfg_feature_weight,
        "dhfg_response_weight": args.dhfg_response_weight,
        "dhfg_confidence_weight": args.dhfg_confidence_weight,
        "dhfg_polarity_weight": args.dhfg_polarity_weight,
        "dhfg_shape_weight": args.dhfg_shape_weight,
        "dhfg_boundary_weight": args.dhfg_boundary_weight,
        "dhfg_background_weight": args.dhfg_background_weight,
    }
    if any(value < 0 for value in dhfg_weights.values()):
        raise ValueError(f"DHFG weights must be non-negative: {dhfg_weights}")
    if args.dhfg_boundary_radius < 1:
        raise ValueError("dhfg_boundary_radius must be at least 1")
    if args.dhfg_contrast_temperature <= 0:
        raise ValueError("dhfg_contrast_temperature must be positive")
    if args.dhfg_erasure_radius < 0 or args.dhfg_erasure_blur_radius < 1:
        raise ValueError("invalid DHFG counterfactual erasure radii")
    if args.dhfg_delta_tolerance < 0:
        raise ValueError("dhfg_delta_tolerance must be non-negative")
    if args.dhfg_minimum_teacher_delta < 0:
        raise ValueError("dhfg_minimum_teacher_delta must be non-negative")
    if args.enable_msdf and (args.enable_carf or args.rda_mode != "none"):
        raise ValueError(
            "The first MSDF experiment must be MSDF-only: use --rda_mode none "
            "without --enable_carf"
        )
    if (
        args.rda_mode == "none"
        and not args.enable_carf
        and not args.enable_msdf
        and not args.enable_dhfg
    ):
        print(
            "[WARN] rda_mode=none without CARF is the clean diffusion baseline mode"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        compute_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        compute_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32
    use_autocast = device.type == "cuda" and compute_dtype != torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet"
    ).to(device)
    if not args.no_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    if args.unet_train_scope != "full":
        unet.requires_grad_(False)
        if args.unet_train_scope == "attention":
            for name, parameter in unet.named_parameters():
                if ".attn2." in name or name.startswith("conv_out."):
                    parameter.requires_grad_(True)
    unet.train()
    rda = None
    use_rda_auxiliary_losses = args.rda_mode in {"full", "loss_only"}
    if args.rda_mode == "full":
        rda = ReferenceTokenAdapter(
            latent_channels=int(vae.config.latent_channels),
            hidden_dim=args.adapter_hidden_dim,
            token_dim=int(text_encoder.config.hidden_size),
            token_grid=args.token_grid,
            max_token_scale=args.max_token_scale,
        ).to(device)
    carf = None
    if args.enable_carf:
        carf = CARFRegionRefiner(
            latent_channels=int(vae.config.latent_channels),
            hidden_dim=args.carf_hidden_dim,
            min_support=args.carf_min_support,
            max_core_adjustment=args.carf_max_core_adjustment,
            max_context_adjustment=args.carf_max_context_adjustment,
        ).to(device)
    msdf = None
    if args.enable_msdf:
        msdf = build_msdf_adapter(
            unet,
            latent_channels=int(vae.config.latent_channels),
            hidden_dim=args.msdf_hidden_dim,
            context_scale=args.msdf_context_scale,
            max_injection=args.msdf_max_injection,
            max_residual_ratio=args.msdf_max_residual_ratio,
            branch_dropout=args.msdf_branch_dropout,
            pixel_support_weight=args.msdf_pixel_support_weight,
        ).to(device)
        installed = msdf.attach(unet)
        print(f"[MSDF] multi-scale up-block injections={installed}")
    joint_spatial_attention = rda is not None and carf is not None
    if joint_spatial_attention:
        installed = install_rda_carf_attention(unet)
        print(f"[RDA+CARF] spatial attention processors={installed}")
    scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    dataset = PairedDefectDataset(
        args.image_dir,
        args.mask_dir,
        args.resolution,
        # MSDF v2 uses self-reference to learn morphology encoding and retains
        # cross-reference batches as category-level regularization. The adapter
        # sees only a local defect residual, not the full target latent.
        reference_cross_probability=0.30,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    dhfg = None
    dhfg_calibration = {}
    if args.enable_dhfg:
        dhfg = DHFGTeacher(
            args.dhfg_teacher_path,
            class_name=args.dhfg_class_name,
            class_id=args.dhfg_class_id,
            topk_fraction=args.dhfg_object_topk_fraction,
            maximum_object_locations=args.dhfg_object_topk_max,
            background_topk_fraction=args.dhfg_background_topk_fraction,
            maximum_background_locations=args.dhfg_background_topk_max,
            minimum_teacher_probability=args.dhfg_min_teacher_probability,
            confidence_beta=args.dhfg_confidence_beta,
            boundary_radius=args.dhfg_boundary_radius,
            contrast_threshold=args.dhfg_contrast_threshold,
            contrast_temperature=args.dhfg_contrast_temperature,
            erasure_radius=args.dhfg_erasure_radius,
            erasure_blur_radius=args.dhfg_erasure_blur_radius,
            delta_tolerance=args.dhfg_delta_tolerance,
            minimum_teacher_delta=args.dhfg_minimum_teacher_delta,
        ).to(device=device, dtype=compute_dtype)
        if dhfg.input_size % 8:
            raise ValueError("DHFG teacher input_size must be divisible by the VAE scale factor 8")
        calibration_count = min(len(dataset), args.dhfg_calibration_samples)
        if calibration_count == len(dataset):
            calibration_indices = list(range(len(dataset)))
        else:
            calibration_indices = np.linspace(
                0, len(dataset) - 1, calibration_count, dtype=int
            ).tolist()
        calibration_pixels, calibration_masks = [], []
        with torch.no_grad():
            for calibration_index in calibration_indices:
                calibration_sample = dataset[calibration_index]
                calibration_image = calibration_sample["image"].unsqueeze(0).to(
                    device=device, dtype=compute_dtype
                )
                calibration_mask = calibration_sample["mask"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                local_pixels, local_mask = crop_mask_regions(
                    (calibration_image + 1.0) / 2.0,
                    calibration_mask,
                    dhfg.input_size,
                    args.dhfg_context_scale,
                    args.dhfg_minimum_extent_ratio,
                )
                calibration_pixels.append(local_pixels)
                calibration_masks.append(local_mask)
            dhfg_calibration = dhfg.calibrate(
                calibration_pixels, calibration_masks
            )
        del calibration_pixels, calibration_masks
        print(
            f"[DHFG] teacher={args.dhfg_teacher_path} | "
            f"class={dhfg.class_id}:{dhfg.class_names[dhfg.class_id]} | "
            f"calibration={dhfg_calibration}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("Install bitsandbytes or omit --use_8bit_adam") from error
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    trainable_unet_parameters = [
        parameter for parameter in unet.parameters() if parameter.requires_grad
    ]
    if args.unet_train_scope == "attention" and not trainable_unet_parameters:
        raise RuntimeError("No UNet cross-attention parameters were selected")
    if args.rda_mode == "loss_only" and not trainable_unet_parameters:
        raise ValueError("loss_only mode requires a trainable UNet")
    optimizer_groups = [
        {
            "params": trainable_unet_parameters,
            "lr": args.unet_learning_rate,
        }
    ]
    if rda is not None:
        optimizer_groups.append(
            {"params": rda.parameters(), "lr": args.rda_learning_rate}
        )
    if carf is not None:
        optimizer_groups.append(
            {"params": carf.parameters(), "lr": args.carf_learning_rate}
        )
    if msdf is not None:
        optimizer_groups.append(
            {"params": msdf.parameters(), "lr": args.msdf_learning_rate}
        )
    optimizer_kwargs = {
        "betas": (0.9, 0.999),
        "weight_decay": 1e-2,
    }
    if optimizer_class is torch.optim.AdamW:
        # Foreach AdamW creates large temporary tensor lists on the first
        # optimizer step. Disabling it lowers the peak without changing the
        # optimizer equations, which matters for full-UNet 24 GiB training.
        optimizer_kwargs["foreach"] = False
    optimizer = optimizer_class(optimizer_groups, **optimizer_kwargs)
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and compute_dtype == torch.float16
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and compute_dtype == torch.float16
        )

    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=args.train_steps,
        desc=(
            "MSDF+DHFG"
            if msdf is not None and dhfg is not None
            else "MSDF"
            if msdf is not None
            else "DHFG"
            if dhfg is not None
            else "RDA"
        ),
    )
    last_losses = {}
    micro_step = 0
    successful_steps = 0
    consecutive_nonfinite_gradient_skips = 0
    total_nonfinite_gradient_skips = 0
    # Count successful optimizer updates rather than attempted updates. Mixed
    # precision can legitimately overflow once; GradScaler must lower its scale
    # and retry instead of shortening or aborting the requested training run.
    while successful_steps < args.train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        image = batch["image"].to(device=device, dtype=torch.float32)
        original_mask = batch["mask"].to(device=device, dtype=torch.float32)
        reference = batch["reference"].to(device=device, dtype=torch.float32)
        reference_mask = batch["reference_mask"].to(device=device, dtype=torch.float32)
        reference_is_self = batch["reference_is_self"].to(device=device)
        mask = augment_coarse_mask(original_mask, args.mask_jitter_radius)
        masked_image = image * (mask < 0.5)
        batch_size = image.shape[0]
        cond_embedding = encode_prompt(
            tokenizer, text_encoder, args.prompt, batch_size, device
        )
        uncond_embedding = encode_prompt(
            tokenizer, text_encoder, "", batch_size, device
        )

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=compute_dtype, enabled=use_autocast
        ):
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            masked_latents = vae.encode(masked_image).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            if rda is not None or msdf is not None:
                reference_latents = vae.encode(reference).latent_dist.mode()
                reference_latents = reference_latents * vae.config.scaling_factor
        soft_prior = make_soft_prior(
            mask, latents.shape[-2:], args.mask_blur_radius
        )
        defect_saliency = build_defect_saliency(
            image,
            original_mask,
            latents.shape[-2:],
            args.saliency_low_quantile,
            args.saliency_high_quantile,
        ).detach()
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        refined_prior = soft_prior
        carf_loss = torch.zeros((), device=device)
        carf_metrics = {}
        if carf is not None:
            # A nearby independently noised timestep supplies cheap temporal
            # consistency without a second UNet forward pass.
            second_timesteps = (timesteps + torch.randint_like(timesteps, -50, 51)).clamp(
                0, scheduler.config.num_train_timesteps - 1
            )
            second_noisy = scheduler.add_noise(latents, noise, second_timesteps)
            second_prior = carf(
                second_noisy.float(), masked_latents.float(), soft_prior,
                second_timesteps,
            )
            # Recompute the first view after the second call because the module
            # exposes its latest decomposition for loss logging/supervision.
            refined_prior = carf(
                noisy_latents.float(), masked_latents.float(), soft_prior, timesteps
            )
            current_core = carf.last_core_map
            current_context = carf.last_context_map
            current_ratio = defect_saliency.sum((2, 3)) / soft_prior.sum(
                (2, 3)
            ).clamp_min(1.0)
            # Geometry belongs to the current sample. A randomly selected
            # reference defect must not enlarge or shrink its support target.
            target_support_ratio = current_ratio.clamp(0.03, 0.95)
            carf_loss, carf_metrics = carf_weak_supervision_loss(
                refined_prior,
                defect_saliency,
                soft_prior,
                second_view=second_prior,
                core=current_core,
                context=current_context,
                target_support_ratio=target_support_ratio,
                consistency_weight=args.carf_consistency_weight,
                coverage_weight=args.carf_coverage_weight,
                min_core_coverage=args.carf_min_core_coverage,
            )
        # CARF-only refines the inpaint mask. In the joint model, preserve the
        # validated RDA mask channel and use CARF exclusively to gate reference
        # attention, preventing the two modules from rewriting the same input.
        unet_mask_prior = soft_prior if joint_spatial_attention else refined_prior
        model_input = torch.cat(
            [noisy_latents, unet_mask_prior.to(noisy_latents.dtype), masked_latents],
            dim=1,
        )
        doubled_input = torch.cat([model_input, model_input], dim=0)
        doubled_timesteps = torch.cat([timesteps, timesteps])
        if rda is not None:
            reference_tokens = rda(reference_latents.float(), reference_mask)
            adaptive_weight = (
                rda.last_token_scale.detach()
                / (
                    rda.max_token_scale
                    * torch.sigmoid(rda.token_gate.detach())
                ).clamp_min(1e-6)
            ).clamp(0.25, 2.0)
            embeddings = append_reference_tokens(
                torch.cat([uncond_embedding, cond_embedding], dim=0),
                reference_tokens,
                classifier_free_guidance=True,
            )
        else:
            # E1 control: use the same UNet targets and auxiliary losses, but
            # remove every reference-token path. A unit weight replaces the
            # reference-confidence multiplier used by the full model.
            adaptive_weight = torch.ones(
                batch_size, 1, device=device, dtype=torch.float32
            )
            embeddings = torch.cat([uncond_embedding, cond_embedding], dim=0)

        if msdf is not None:
            msdf.prepare(
                reference_latents.float(),
                reference_mask,
                soft_prior,
                timesteps,
                scheduler.config.num_train_timesteps,
                classifier_free_guidance=True,
                reference_pixels=reference,
            )

        with torch.autocast(
            device_type=device.type, dtype=compute_dtype, enabled=use_autocast
        ):
            attention_kwargs = None
            if joint_spatial_attention:
                spatial_gate = carf.attention_gate()
                attention_kwargs = {
                    "carf_attention_gate": torch.cat(
                        [spatial_gate, spatial_gate], dim=0
                    ),
                    "rda_token_count": rda.num_tokens,
                }
            prediction = unet(
                doubled_input,
                doubled_timesteps,
                encoder_hidden_states=embeddings,
                cross_attention_kwargs=attention_kwargs,
            ).sample
            prediction_uncond, prediction_cond = prediction.chunk(2)
            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {scheduler.config.prediction_type}"
                )

            conditional_loss = F.mse_loss(
                prediction_cond.float(), target.float()
            )
            region_loss = torch.zeros((), device=device)
            latent_reconstruction_loss = torch.zeros((), device=device)
            structure_loss = torch.zeros((), device=device)
            distribution_loss = torch.zeros((), device=device)
            background_loss = torch.zeros((), device=device)
            msdf_region_loss = torch.zeros((), device=device)
            msdf_reconstruction_loss = torch.zeros((), device=device)
            msdf_structure_loss = torch.zeros((), device=device)
            msdf_background_loss = torch.zeros((), device=device)
            msdf_scale_loss = torch.zeros((), device=device)
            msdf_support_loss = torch.zeros((), device=device)
            dhfg_feature_delta_loss = torch.zeros((), device=device)
            dhfg_response_delta_loss = torch.zeros((), device=device)
            dhfg_confidence_delta_loss = torch.zeros((), device=device)
            dhfg_polarity_loss = torch.zeros((), device=device)
            dhfg_shape_loss = torch.zeros((), device=device)
            dhfg_boundary_loss = torch.zeros((), device=device)
            dhfg_background_loss = torch.zeros((), device=device)
            dhfg_confidence = torch.zeros((), device=device)
            dhfg_reference_confidence = torch.zeros((), device=device)
            dhfg_confidence_delta = torch.zeros((), device=device)
            dhfg_reference_confidence_delta = torch.zeros((), device=device)
            dhfg_support = torch.zeros((), device=device)
            dhfg_background_response = torch.zeros((), device=device)
            dhfg_positive = torch.zeros((), device=device)
            dhfg_negative = torch.zeros((), device=device)
            dhfg_coverage = torch.zeros((), device=device)
            dhfg_extent = torch.zeros((), device=device)
            dhfg_objective = torch.zeros((), device=device)
            dhfg_active = torch.zeros((), device=device)
            dhfg_reliability = torch.zeros((), device=device)
            # Shared by the RDA and CARF-only quality objectives.
            pixel_error = (prediction_cond.float() - target.float()).square()
            if use_rda_auxiliary_losses:
                region_loss = (
                    pixel_error * (1.0 + args.region_weight * defect_saliency)
                ).mean()
            if (
                use_rda_auxiliary_losses
                and scheduler.config.prediction_type == "epsilon"
            ):
                alphas = scheduler.alphas_cumprod.to(
                    device=device, dtype=torch.float32
                )[timesteps].reshape(-1, 1, 1, 1)
                predicted_clean = (
                    noisy_latents.float()
                    - (1.0 - alphas).sqrt() * prediction_cond.float()
                ) / alphas.sqrt().clamp_min(1e-4)
                # Very early/high-noise estimates can have extreme magnitude.
                # Clamping only the auxiliary reconstruction branch prevents
                # those samples from dominating while preserving diffusion loss.
                predicted_clean = predicted_clean.clamp(-10.0, 10.0)
                reconstruction_error = (
                    predicted_clean - latents.detach().float()
                ).abs()
                latent_reconstruction_loss = (
                    reconstruction_error * defect_saliency
                ).sum() / (
                    defect_saliency.sum() * predicted_clean.shape[1]
                ).clamp_min(1.0)
                pred_dx = predicted_clean[..., 1:] - predicted_clean[..., :-1]
                true_dx = (
                    latents.detach().float()[..., 1:]
                    - latents.detach().float()[..., :-1]
                )
                pred_dy = predicted_clean[..., 1:, :] - predicted_clean[..., :-1, :]
                true_dy = (
                    latents.detach().float()[..., 1:, :]
                    - latents.detach().float()[..., :-1, :]
                )
                prior_x = (
                    defect_saliency[..., 1:] * defect_saliency[..., :-1]
                )
                prior_y = (
                    defect_saliency[..., 1:, :] * defect_saliency[..., :-1, :]
                )
                structure_loss = (
                    (pred_dx - true_dx).abs() * prior_x
                ).sum() / (
                    prior_x.sum() * predicted_clean.shape[1]
                ).clamp_min(1.0)
                structure_loss = structure_loss + (
                    (pred_dy - true_dy).abs() * prior_y
                ).sum() / (
                    prior_y.sum() * predicted_clean.shape[1]
                ).clamp_min(1.0)
                distribution_loss = local_distribution_loss(
                    predicted_clean,
                    latents.detach().float(),
                    defect_saliency,
                ) * alphas.mean() * adaptive_weight.mean()
                background_region = (
                    soft_prior if joint_spatial_attention else refined_prior
                )
                background_prior = (1.0 - background_region.float()).square()
                background_loss = (
                    reconstruction_error * background_prior
                ).sum() / (
                    background_prior.sum() * predicted_clean.shape[1]
                ).clamp_min(1.0)
            if use_rda_auxiliary_losses:
                diffusion_objective = (
                    region_loss
                    + args.conditional_weight * conditional_loss
                    + args.latent_reconstruction_weight
                    * adaptive_weight.mean()
                    * latent_reconstruction_loss
                    + args.structure_weight
                    * adaptive_weight.mean()
                    * structure_loss
                    + args.distribution_weight * distribution_loss
                    + args.background_weight * background_loss
                )
            elif msdf is not None:
                # MSDF supervises real local morphology in addition to injecting
                # reference features. Reliability weighting keeps high-noise x0
                # estimates from overwhelming the standard diffusion target.
                msdf_region_loss = (
                    pixel_error * defect_saliency
                ).sum() / (
                    defect_saliency.sum() * pixel_error.shape[1]
                ).clamp_min(1.0)
                msdf_support_loss = msdf.morphology_support_loss(
                    defect_saliency,
                    reference_is_self,
                )
                if scheduler.config.prediction_type == "epsilon":
                    alphas = scheduler.alphas_cumprod.to(
                        device=device, dtype=torch.float32
                    )[timesteps].reshape(-1, 1, 1, 1)
                    predicted_clean = (
                        noisy_latents.float()
                        - (1.0 - alphas).sqrt() * prediction_cond.float()
                    ) / alphas.sqrt().clamp_min(1e-4)
                    predicted_clean = predicted_clean.clamp(-10.0, 10.0)
                    target_clean = latents.detach().float()
                    reliability = alphas.clamp_min(0.05)
                    clean_error = (predicted_clean - target_clean).abs()
                    weighted_saliency = defect_saliency * reliability
                    msdf_reconstruction_loss = (
                        clean_error * weighted_saliency
                    ).sum() / (
                        weighted_saliency.sum() * predicted_clean.shape[1]
                    ).clamp_min(1.0)

                    pred_dx = predicted_clean[..., 1:] - predicted_clean[..., :-1]
                    true_dx = target_clean[..., 1:] - target_clean[..., :-1]
                    pred_dy = predicted_clean[..., 1:, :] - predicted_clean[..., :-1, :]
                    true_dy = target_clean[..., 1:, :] - target_clean[..., :-1, :]
                    weight_x = weighted_saliency[..., 1:] * weighted_saliency[..., :-1]
                    weight_y = weighted_saliency[..., 1:, :] * weighted_saliency[..., :-1, :]
                    msdf_structure_loss = (
                        ((pred_dx - true_dx).abs() * weight_x).sum()
                        / (weight_x.sum() * predicted_clean.shape[1]).clamp_min(1.0)
                        + ((pred_dy - true_dy).abs() * weight_y).sum()
                        / (weight_y.sum() * predicted_clean.shape[1]).clamp_min(1.0)
                    )
                    background_prior = (1.0 - soft_prior.float()).square() * reliability
                    msdf_background_loss = (
                        clean_error * background_prior
                    ).sum() / (
                        background_prior.sum() * predicted_clean.shape[1]
                    ).clamp_min(1.0)
                    msdf_scale_loss = multiscale_local_consistency_loss(
                        predicted_clean,
                        target_clean,
                        weighted_saliency,
                    )
                diffusion_objective = (
                    conditional_loss
                    + args.msdf_region_weight * msdf_region_loss
                    + args.msdf_reconstruction_weight * msdf_reconstruction_loss
                    + args.msdf_structure_weight * msdf_structure_loss
                    + args.msdf_background_weight * msdf_background_loss
                    + args.msdf_scale_weight * msdf_scale_loss
                    + args.msdf_support_weight * msdf_support_loss
                )
            elif carf is not None:
                # CARF v4 quality objective. A tiny defect contributes almost
                # nothing to a whole-image diffusion MSE, so v3 could learn a
                # useful region while leaving colour, boundary and halo quality
                # weakly constrained. Weight the denoising error by image
                # evidence and supervise the implied clean latent locally.
                carf_region_loss = (
                    pixel_error * defect_saliency
                ).sum() / (
                    defect_saliency.sum() * pixel_error.shape[1]
                ).clamp_min(1.0)
                carf_reconstruction_loss = torch.zeros((), device=device)
                carf_structure_loss = torch.zeros((), device=device)
                carf_background_loss = torch.zeros((), device=device)
                if scheduler.config.prediction_type == "epsilon":
                    alphas = scheduler.alphas_cumprod.to(
                        device=device, dtype=torch.float32
                    )[timesteps].reshape(-1, 1, 1, 1)
                    predicted_clean = (
                        noisy_latents.float()
                        - (1.0 - alphas).sqrt() * prediction_cond.float()
                    ) / alphas.sqrt().clamp_min(1e-4)
                    predicted_clean = predicted_clean.clamp(-10.0, 10.0)
                    clean_error = (predicted_clean - latents.detach().float()).abs()
                    # High-noise x0 estimates are less reliable; alpha weighting
                    # prevents them from dominating the local quality objective.
                    reliability = alphas.clamp_min(0.05)
                    carf_reconstruction_loss = (
                        clean_error * defect_saliency * reliability
                    ).sum() / (
                        (defect_saliency * reliability).sum()
                        * predicted_clean.shape[1]
                    ).clamp_min(1.0)
                    pred_dx = predicted_clean[..., 1:] - predicted_clean[..., :-1]
                    true_dx = latents.detach().float()[..., 1:] - latents.detach().float()[..., :-1]
                    pred_dy = predicted_clean[..., 1:, :] - predicted_clean[..., :-1, :]
                    true_dy = latents.detach().float()[..., 1:, :] - latents.detach().float()[..., :-1, :]
                    prior_x = defect_saliency[..., 1:] * defect_saliency[..., :-1]
                    prior_y = defect_saliency[..., 1:, :] * defect_saliency[..., :-1, :]
                    carf_structure_loss = (
                        ((pred_dx - true_dx).abs() * prior_x).sum()
                        / (prior_x.sum() * predicted_clean.shape[1]).clamp_min(1.0)
                        + ((pred_dy - true_dy).abs() * prior_y).sum()
                        / (prior_y.sum() * predicted_clean.shape[1]).clamp_min(1.0)
                    ) * reliability.mean()
                    background_prior = (1.0 - soft_prior.float()).square()
                    carf_background_loss = (
                        clean_error * background_prior * reliability
                    ).sum() / (
                        (background_prior * reliability).sum()
                        * predicted_clean.shape[1]
                    ).clamp_min(1.0)
                diffusion_objective = (
                    conditional_loss
                    + args.carf_region_weight * carf_region_loss
                    + args.carf_reconstruction_weight * carf_reconstruction_loss
                    + args.carf_structure_weight * carf_structure_loss
                    + args.carf_background_weight * carf_background_loss
                )
            else:
                # Clean CARF-only: standard conditional diffusion objective plus
                # CARF's own weak region supervision. No RDA loss is retained.
                diffusion_objective = conditional_loss
            if (
                dhfg is not None
                and scheduler.config.prediction_type == "epsilon"
                and micro_step % args.dhfg_interval == 0
            ):
                reliable_samples = timesteps <= args.dhfg_max_timestep
                if bool(reliable_samples.any()):
                    dhfg_active = torch.ones((), device=device)
                    dhfg_alphas = scheduler.alphas_cumprod.to(
                        device=device, dtype=torch.float32
                    )[timesteps[reliable_samples]].reshape(-1, 1, 1, 1)
                    dhfg_reliability = dhfg_alphas.mean().detach().clamp(0.05, 1.0)
                    dhfg_clean = (
                        noisy_latents[reliable_samples].float()
                        - (1.0 - dhfg_alphas).sqrt()
                        * prediction_cond[reliable_samples].float()
                    ) / dhfg_alphas.sqrt().clamp_min(1e-4)
                    # The local VAE decode is substantially cheaper than a full
                    # 512x512 decode and is the reason joint training fits 24 GiB.
                    dhfg_clean = dhfg_clean.clamp(-6.0, 6.0)
                    local_latents, _ = crop_mask_regions(
                        dhfg_clean,
                        original_mask[reliable_samples],
                        dhfg.input_size // 8,
                        args.dhfg_context_scale,
                        args.dhfg_minimum_extent_ratio,
                    )
                    generated_pixels = (
                        vae.decode(
                            local_latents.to(dtype=compute_dtype)
                            / vae.config.scaling_factor
                        ).sample.float()
                        + 1.0
                    ) / 2.0
                    dhfg_reference_pixels, dhfg_reference_mask = crop_mask_regions(
                        (image[reliable_samples] + 1.0) / 2.0,
                        original_mask[reliable_samples],
                        dhfg.input_size,
                        args.dhfg_context_scale,
                        args.dhfg_minimum_extent_ratio,
                    )
                    dhfg_losses = dhfg.guidance_losses(
                        generated_pixels.clamp(0, 1),
                        dhfg_reference_pixels.clamp(0, 1),
                        dhfg_reference_mask,
                    )
                    dhfg_feature_delta_loss = dhfg_losses["feature_delta"]
                    dhfg_response_delta_loss = dhfg_losses["response_delta"]
                    dhfg_confidence_delta_loss = dhfg_losses["confidence_delta_loss"]
                    dhfg_polarity_loss = dhfg_losses["polarity"]
                    dhfg_shape_loss = dhfg_losses["shape"]
                    dhfg_boundary_loss = dhfg_losses["boundary"]
                    dhfg_background_loss = dhfg_losses["false_background"]
                    dhfg_confidence = dhfg_losses["confidence"]
                    dhfg_reference_confidence = dhfg_losses["reference_confidence"]
                    dhfg_confidence_delta = dhfg_losses["confidence_delta"]
                    dhfg_reference_confidence_delta = dhfg_losses[
                        "reference_confidence_delta"
                    ]
                    dhfg_support = dhfg_losses["support"]
                    dhfg_background_response = dhfg_losses["background_response"]
                    dhfg_positive = dhfg_losses["positive"]
                    dhfg_negative = dhfg_losses["negative"]
                    dhfg_coverage = dhfg_losses["coverage"]
                    dhfg_extent = dhfg_losses["extent"]
                    warmup = min(
                        1.0,
                        float(successful_steps + 1)
                        / max(float(args.dhfg_warmup_steps), 1.0),
                    )
                    # Do not multiply by dhfg_interval here.  That unbiased but
                    # high-variance estimator caused a single feedback sample
                    # to dominate all other accumulated micro-steps.  The user
                    # facing dhfg_weight now means weight per active sample.
                    dhfg_objective = (
                        args.dhfg_feature_weight * dhfg_feature_delta_loss
                        + args.dhfg_response_weight * dhfg_response_delta_loss
                        + args.dhfg_confidence_weight * dhfg_confidence_delta_loss
                        + args.dhfg_polarity_weight * dhfg_polarity_loss
                        + args.dhfg_shape_weight * dhfg_shape_loss
                        + args.dhfg_boundary_weight * dhfg_boundary_loss
                        + args.dhfg_background_weight * dhfg_background_loss
                    ) * warmup * dhfg_reliability
            loss = (
                diffusion_objective
                + args.carf_weight * carf_loss
                + args.dhfg_weight * dhfg_objective
            ) / args.gradient_accumulation_steps

        if not bool(torch.isfinite(loss.detach())):
            diagnostics = {
                "conditional": float(conditional_loss.detach()),
                "msdf_region": float(msdf_region_loss.detach()),
                "msdf_reconstruction": float(msdf_reconstruction_loss.detach()),
                "msdf_structure": float(msdf_structure_loss.detach()),
                "msdf_background": float(msdf_background_loss.detach()),
                "msdf_scale": float(msdf_scale_loss.detach()),
                "msdf_support": float(msdf_support_loss.detach()),
                "dhfg_feature_delta": float(dhfg_feature_delta_loss.detach()),
                "dhfg_response_delta": float(dhfg_response_delta_loss.detach()),
                "dhfg_confidence_delta_loss": float(dhfg_confidence_delta_loss.detach()),
                "dhfg_polarity": float(dhfg_polarity_loss.detach()),
                "dhfg_shape": float(dhfg_shape_loss.detach()),
                "dhfg_boundary": float(dhfg_boundary_loss.detach()),
                "dhfg_background": float(dhfg_background_loss.detach()),
            }
            raise FloatingPointError(
                f"Non-finite training loss at micro_step={micro_step}: {diagnostics}"
            )
        scaler.scale(loss).backward()
        should_step = (
            micro_step + 1
        ) % args.gradient_accumulation_steps == 0
        if should_step:
            scaler.unscale_(optimizer)
            parameters_to_clip = list(trainable_unet_parameters)
            if rda is not None:
                parameters_to_clip += list(rda.parameters())
            if carf is not None:
                parameters_to_clip += list(carf.parameters())
            if msdf is not None:
                parameters_to_clip += list(msdf.parameters())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters_to_clip, args.max_grad_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                optimizer.zero_grad(set_to_none=True)
                if not scaler.is_enabled():
                    raise FloatingPointError(
                        f"Non-finite gradient norm at micro_step={micro_step} "
                        "without fp16 GradScaler"
                    )
                scaler.update()
                consecutive_nonfinite_gradient_skips += 1
                total_nonfinite_gradient_skips += 1
                print(
                    "[WARN] fp16 gradient overflow: skipped optimizer step "
                    f"at micro_step={micro_step}; "
                    f"consecutive={consecutive_nonfinite_gradient_skips}/"
                    f"{args.max_nonfinite_gradient_skips}; "
                    f"total={total_nonfinite_gradient_skips}; "
                    f"new_scale={scaler.get_scale():.1f}"
                )
                if (
                    consecutive_nonfinite_gradient_skips
                    > args.max_nonfinite_gradient_skips
                ):
                    raise FloatingPointError(
                        "Too many non-finite fp16 gradient steps; training is "
                        "genuinely unstable rather than experiencing a transient "
                        "loss-scale overflow"
                    )
                micro_step += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if msdf is not None:
                assert_finite_module(msdf, "MSDF")
            consecutive_nonfinite_gradient_skips = 0
            successful_steps += 1
            progress.update(1)
        last_losses = {
            "loss": float(loss.detach()) * args.gradient_accumulation_steps,
            "region": float(region_loss.detach()),
            "conditional": float(conditional_loss.detach()),
            "distribution": float(distribution_loss.detach()),
            "reconstruction": float(latent_reconstruction_loss.detach()),
            "structure": float(structure_loss.detach()),
            "background": float(background_loss.detach()),
            "saliency": float(defect_saliency.detach().mean()),
            "carf": float(carf_loss.detach()),
        }
        if msdf is not None:
            last_losses.update(
                msdf_region=float(msdf_region_loss.detach()),
                msdf_reconstruction=float(msdf_reconstruction_loss.detach()),
                msdf_structure=float(msdf_structure_loss.detach()),
                msdf_background=float(msdf_background_loss.detach()),
                msdf_scale=float(msdf_scale_loss.detach()),
                msdf_support=float(msdf_support_loss.detach()),
                msdf_self_reference=float(reference_is_self.float().mean().cpu()),
                msdf_gate=float(msdf.last_gates.mean().cpu()),
                msdf_gate_min=float(msdf.last_gates.min().cpu()),
                msdf_gate_max=float(msdf.last_gates.max().cpu()),
            )
        if dhfg is not None:
            last_losses.update(
                dhfg_feature_delta=float(dhfg_feature_delta_loss.detach()),
                dhfg_response_delta=float(dhfg_response_delta_loss.detach()),
                dhfg_confidence_delta_loss=float(dhfg_confidence_delta_loss.detach()),
                dhfg_polarity=float(dhfg_polarity_loss.detach()),
                dhfg_shape=float(dhfg_shape_loss.detach()),
                dhfg_boundary=float(dhfg_boundary_loss.detach()),
                dhfg_background=float(dhfg_background_loss.detach()),
                dhfg_confidence=float(dhfg_confidence.detach()),
                dhfg_reference_confidence=float(dhfg_reference_confidence.detach()),
                dhfg_confidence_delta=float(dhfg_confidence_delta.detach()),
                dhfg_reference_confidence_delta=float(
                    dhfg_reference_confidence_delta.detach()
                ),
                dhfg_support=float(dhfg_support.detach()),
                dhfg_background_response=float(dhfg_background_response.detach()),
                dhfg_positive=float(dhfg_positive.detach()),
                dhfg_negative=float(dhfg_negative.detach()),
                dhfg_coverage=float(dhfg_coverage.detach()),
                dhfg_extent=float(dhfg_extent.detach()),
                dhfg_active=float(dhfg_active.detach()),
                dhfg_reliability=float(dhfg_reliability.detach()),
            )
        if carf is not None and not use_rda_auxiliary_losses:
            last_losses.update(
                carf_region_quality=float(carf_region_loss.detach()),
                carf_reconstruction=float(carf_reconstruction_loss.detach()),
                carf_structure=float(carf_structure_loss.detach()),
                carf_background=float(carf_background_loss.detach()),
            )
        last_losses.update(
            {key: float(value.cpu()) for key, value in carf_metrics.items()}
        )
        if rda is not None:
            last_losses.update(
                {
                    "token_scale": float(
                        rda.last_token_scale.detach().mean().cpu()
                    ),
                    "token_scale_min": float(
                        rda.last_token_scale.detach().min().cpu()
                    ),
                    "token_scale_max": float(
                        rda.last_token_scale.detach().max().cpu()
                    ),
                }
            )
        if joint_spatial_attention:
            last_losses["carf_attention_alpha"] = float(
                torch.sigmoid(carf.attention_gate_logit.detach()).cpu()
            )
        progress.set_postfix(**{k: f"{v:.4f}" for k, v in last_losses.items()})
        micro_step += 1
    progress.close()

    assert_finite_module(unet, "UNet")
    if msdf is not None:
        assert_finite_module(msdf, "MSDF")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    unet.eval()
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        unet=unet.to(device="cpu", dtype=torch.float32),
        text_encoder=text_encoder.to(device="cpu", dtype=torch.float32),
        vae=vae.to(device="cpu", dtype=torch.float32),
    )
    pipeline.save_pretrained(output_dir)
    if rda is not None:
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in rda.state_dict().items()
                },
                "latent_channels": int(vae.config.latent_channels),
                "hidden_dim": args.adapter_hidden_dim,
                "token_dim": int(text_encoder.config.hidden_size),
                "token_grid": args.token_grid,
                "max_token_scale": args.max_token_scale,
                "format_version": RDA_FORMAT_VERSION,
            },
            output_dir / "rda.pt",
        )
    if carf is not None:
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in carf.state_dict().items()
                },
                "latent_channels": int(vae.config.latent_channels),
                "hidden_dim": args.carf_hidden_dim,
                "min_support": carf.min_support,
                "max_core_adjustment": carf.max_core_adjustment,
                "max_context_adjustment": carf.max_context_adjustment,
                "format_version": CARF_FORMAT_VERSION,
            },
            output_dir / "carf.pt",
        )
    if msdf is not None:
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in msdf.state_dict().items()
                },
                "latent_channels": msdf.latent_channels,
                "hidden_dim": msdf.hidden_dim,
                "block_channels": list(msdf.block_channels),
                "context_scale": msdf.context_scale,
                "max_injection": msdf.max_injection,
                "max_residual_ratio": msdf.max_residual_ratio,
                "branch_dropout": msdf.branch_dropout,
                "pixel_support_weight": msdf.pixel_support_weight,
                "format_version": MSDF_FORMAT_VERSION,
            },
            output_dir / "msdf.pt",
        )
    metadata = vars(args).copy()
    if msdf is not None:
        metadata["rda_training_version"] = "msdf_v3_bounded_dual_branch_injection"
        if dhfg is not None:
            metadata["rda_training_version"] = "msdf_v3_dhfg_v5_counterfactual_delta"
    elif dhfg is not None:
        metadata["rda_training_version"] = "dhfg_v5_counterfactual_delta_feedback"
    elif joint_spatial_attention:
        metadata["rda_training_version"] = "rda_v4_carf_v3_spatial_attention"
    elif carf is not None and rda is None:
        metadata["rda_training_version"] = "carf_only_v4_local_quality"
    elif carf is not None:
        metadata["rda_training_version"] = "adaptive_reference_gate_v4_carf_v2"
    else:
        metadata["rda_training_version"] = (
            "adaptive_reference_gate_v4"
            if rda is not None
            else "e1_loss_only_no_reference"
        )
    metadata["reference_tokens_enabled"] = rda is not None
    metadata["carf_enabled"] = carf is not None
    metadata["msdf_enabled"] = msdf is not None
    metadata["dhfg_enabled"] = dhfg is not None
    metadata["dhfg_calibration"] = dhfg_calibration
    if dhfg is not None:
        metadata["dhfg_teacher_classes"] = dhfg.class_names
        metadata["dhfg_resolved_class_id"] = dhfg.class_id
        metadata["dhfg_resolved_class_name"] = dhfg.class_names[dhfg.class_id]
    metadata["rda_auxiliary_losses_enabled"] = use_rda_auxiliary_losses
    metadata["num_training_pairs"] = len(dataset)
    metadata["nonfinite_gradient_skips_total"] = total_nonfinite_gradient_skips
    metadata["final"] = last_losses
    metadata_name = (
        "msdf_dhfg_training.json"
        if msdf is not None and dhfg is not None
        else "msdf_training.json"
        if msdf is not None
        else "dhfg_training.json"
        if dhfg is not None
        else "rda_training.json"
    )
    (output_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    method_name = (
        "MSDF+DHFG"
        if msdf is not None and dhfg is not None
        else "MSDF"
        if msdf is not None
        else "DHFG"
        if dhfg is not None
        else "RDA"
    )
    print(f"[{method_name}] saved complete checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
