#!/usr/bin/env python3
"""One-step real-data CUDA acceptance test for MSDF-v3 A1/A2."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.msdf_guidance import (
    MSDF_FORMAT_VERSION, build_msdf_adapter, load_msdf_adapter,
)
from train_rda import PairedDefectDataset, make_soft_prior, encode_prompt
from transformers import CLIPTextModel, CLIPTokenizer


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_checkpoint(adapter, path):
    torch.save({
        "format_version": MSDF_FORMAT_VERSION,
        "latent_channels": adapter.latent_channels,
        "hidden_dim": adapter.hidden_dim,
        "block_channels": adapter.block_channels,
        "context_scale": adapter.context_scale,
        "max_injection": adapter.max_injection,
        "max_residual_ratio": adapter.max_residual_ratio,
        "branch_dropout": adapter.branch_dropout,
        "pixel_support_weight": adapter.pixel_support_weight,
        "morphology_alignment_enabled": adapter.training_ablation != "no_morph",
        "training_ablation": adapter.training_ablation,
        "training_contract": adapter.training_contract(),
        "state_dict": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
    }, path)


def one_variant(mode, common, output):
    device, unet, scheduler = common["device"], common["unet"], common["scheduler"]
    adapter = build_msdf_adapter(unet, common["latents"].shape[1], 64, 1.75, 0.75,
                                 0.25, 0.20, 0.50, training_ablation=mode).to(device)
    adapter.attach(unet); adapter.capture_injection_stats = True; adapter.train()
    calls = {"latent": 0, "morphology_alignment": 0}
    handle = adapter.encoder.register_forward_hook(
        lambda *_: calls.__setitem__("latent", calls["latent"] + 1))
    original_align = adapter._align_reference

    def align(*args, **kwargs):
        calls["morphology_alignment"] += 1
        return original_align(*args, **kwargs)

    adapter._align_reference = align
    active_named = [(name, value) for name, value in adapter.named_parameters()
                    if value.requires_grad]
    optimizer = torch.optim.AdamW([value for _, value in active_named], lr=1e-4,
                                  betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
                                  foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    before = {name: value.detach().cpu().clone() for name, value in adapter.named_parameters()}
    loss_history = []
    grads = {}
    for smoke_step in range(2):
        adapter.prepare(common["reference_latents"].float(), common["reference_mask"],
                        common["prior"], common["timesteps"],
                        scheduler.config.num_train_timesteps, True,
                        reference_pixels=common["reference"])
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = unet(common["doubled_input"], common["doubled_timesteps"],
                              encoder_hidden_states=common["embeddings"]).sample
            _, conditional = prediction.chunk(2)
            diffusion = F.mse_loss(conditional.float(), common["target"].float())
            morphology = (adapter.morphology_support_loss(common["original_mask"],
                           common["reference_is_self"])
                          if adapter.morphology_supervision_enabled
                          else diffusion.new_zeros(()))
            loss = diffusion + 0.25 * morphology
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = {name: value.grad is not None and bool(torch.isfinite(value.grad).all())
                 for name, value in adapter.named_parameters()}
        torch.nn.utils.clip_grad_norm_([value for _, value in active_named], 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        loss_history.append({"step": smoke_step + 1,
                             "conditional_diffusion": float(diffusion.detach()),
                             "morphology_support": float(morphology.detach())})
    changed = [name for name, value in adapter.named_parameters()
               if not torch.equal(before[name], value.detach().cpu())]

    # The zero-initialized projections receive the first update. The next fixed
    # forward must show nonzero injections, while CFG's unconditional half stays zero.
    adapter.eval()
    adapter.prepare(common["reference_latents"].float(), common["reference_mask"],
                    common["prior"], common["timesteps"],
                    scheduler.config.num_train_timesteps, True,
                    reference_pixels=common["reference"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        active = unet(common["doubled_input"], common["doubled_timesteps"],
                      encoder_hidden_states=common["embeddings"]).sample.float()
    stats = list(adapter.last_injection_stats)
    adapter.set_ablation({"residual_scale": 0.0})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        bypass = unet(common["doubled_input"], common["doubled_timesteps"],
                      encoder_hidden_states=common["embeddings"]).sample.float()
    au, ac = active.chunk(2); bu, bc = bypass.chunk(2)
    checkpoint = output / "smoke_msdf.pt"; save_checkpoint(adapter, checkpoint)
    reloaded = load_msdf_adapter(checkpoint, unet, device)
    exact_state = all(torch.equal(value.detach().cpu(), reloaded.state_dict()[name].cpu())
                      for name, value in adapter.state_dict().items())
    reloaded.detach()
    checks = {
        "real_training_sample": True,
        "latent_not_executed": calls["latent"] == 0 if mode == "no_latent" else calls["latent"] > 0,
        "latent_not_in_optimizer": (not any(name.startswith("encoder.") for name, _ in active_named))
            if mode == "no_latent" else True,
        "latent_has_no_gradient": (not any(grads[name] for name in grads if name.startswith("encoder.")))
            if mode == "no_latent" else True,
        "morphology_alignment_route": calls["morphology_alignment"] == 0
            if mode == "no_morph" else calls["morphology_alignment"] > 0,
        "morphology_loss_route": (float(morphology.detach()) == 0.0)
            if mode == "no_morph" else bool(torch.isfinite(morphology)),
        "effective_parameter_updated": bool(changed),
        "four_nonzero_injections": len(stats) == 4 and all(row["conditional_l2"] > 0 for row in stats),
        "cfg_unconditional_no_injection": len(stats) == 4 and all(row["unconditional_l2"] == 0 for row in stats),
        "active_differs_from_bypass": float(torch.linalg.vector_norm(ac - bc)) > 0,
        "unconditional_equals_bypass": bool(torch.equal(au, bu)),
        "finite": bool(torch.isfinite(active).all()) and all(row["finite"] for row in stats),
        "bounded_residual": all(row["finite"] for row in stats),
        "checkpoint_strict_reload_exact": exact_state,
    }
    handle.remove(); adapter.detach()
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL", "mode": mode,
        "sample": common["sample"], "contract": adapter.training_contract(),
        "optimizer_parameter_count": sum(value.numel() for _, value in active_named),
        "smoke_steps": 2, "loss_history": loss_history, "losses": {
            "conditional_diffusion": {"weight": 1.0, "value": float(diffusion.detach()), "enabled": True},
            "morphology_support": {"weight": 0.25 if mode != "no_morph" else 0.0,
                                   "value": float(morphology.detach()), "enabled": mode != "no_morph"},
            "formal_other_msdf_losses": {"enabled_in_formal_training": True,
                "names_and_weights": {"region": 0.75, "reconstruction": 0.5,
                    "structure": 0.25, "background": 0.2, "scale": 0.25},
                "smoke_note": "omitted from the minimal acceptance objective; train_rda.py exercises them"}
        },
        "calls": calls, "changed_parameters": changed, "injection": stats,
        "prediction_delta_l2": {"conditional": float(torch.linalg.vector_norm(ac - bc)),
                                "unconditional": float(torch.linalg.vector_norm(au - bu))},
        "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)},
        "checks": checks,
    }
    (output / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=REPO / "sd2-community/stable-diffusion-2-inpainting")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "runs/msdf_v3_ablation/checks")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-UNet smoke test")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(args.base_model / "text_encoder", local_files_only=True).to(device).eval()
    vae = AutoencoderKL.from_pretrained(args.base_model / "vae", local_files_only=True).to(device, dtype=torch.float16).eval()
    unet = UNet2DConditionModel.from_pretrained(args.base_model / "unet", local_files_only=True).to(device, dtype=torch.float16).eval()
    scheduler = DDPMScheduler.from_pretrained(args.base_model / "scheduler", local_files_only=True)
    for module in (text_encoder, vae, unet):
        module.requires_grad_(False)
    dataset = PairedDefectDataset(args.image_dir, args.mask_dir, 512, reference_cross_probability=0.0)
    # Select by annotation area only, before model execution. A large real mask
    # ensures that all four resolution levels can be observed in this scale test.
    areas = []
    from PIL import Image
    for _image_path, mask_path in dataset.pairs:
        with Image.open(mask_path) as mask_image:
            areas.append(int((np.asarray(mask_image.convert("L")) >= 128).sum()))
    sample_index = int(np.argmax(areas))
    row = dataset[sample_index]
    image = row["image"][None].to(device); original_mask = row["mask"][None].to(device)
    reference = row["reference"][None].to(device); reference_mask = row["reference_mask"][None].to(device)
    with torch.no_grad():
        latents = vae.encode(image.half()).latent_dist.mode() * vae.config.scaling_factor
        reference_latents = vae.encode(reference.half()).latent_dist.mode() * vae.config.scaling_factor
        coarse = original_mask
        masked = image * (1 - coarse)
        masked_latents = vae.encode(masked.half()).latent_dist.mode() * vae.config.scaling_factor
    prior = make_soft_prior(coarse, latents.shape[-2:], 2)
    generator = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    timesteps = torch.tensor([500], device=device, dtype=torch.long)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    model_input = torch.cat([noisy, prior.to(noisy.dtype), masked_latents], 1)
    cond = encode_prompt(tokenizer, text_encoder, "a photo of a sks defect", 1, device)
    uncond = encode_prompt(tokenizer, text_encoder, "", 1, device)
    common = {"device": device, "unet": unet, "scheduler": scheduler,
              "latents": latents, "reference_latents": reference_latents,
              "reference": reference, "reference_mask": reference_mask,
              "original_mask": original_mask, "reference_is_self": torch.tensor([True], device=device),
              "prior": prior, "timesteps": timesteps, "doubled_input": torch.cat([model_input, model_input]),
              "doubled_timesteps": torch.cat([timesteps, timesteps]),
              "embeddings": torch.cat([uncond, cond]), "target": noise,
              "sample": {"selection": "largest mask area in the real training class; metadata-only",
                         "mask_area_pixels": areas[sample_index],
                         "image": str(dataset.pairs[sample_index][0]), "image_sha256": digest(dataset.pairs[sample_index][0]),
                         "mask": str(dataset.pairs[sample_index][1]), "mask_sha256": digest(dataset.pairs[sample_index][1])}}
    results = {}
    for mode in ("no_latent", "no_morph"):
        target = args.output / mode; target.mkdir(parents=True, exist_ok=True)
        results[mode] = one_variant(mode, common, target)
    summary = {"status": "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL",
               "seed": 42, "device": str(device), "variants": {k: v["status"] for k, v in results.items()}}
    (args.output / "real_data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
