#!/usr/bin/env python3
"""Fast CPU contract checks for MSDF-v3 training ablations."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = [type("R", (), {"out_channels": channels})()]

    def forward(self, value):
        return value


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList([Block(8), Block(8), Block(8), Block(8)])

    def forward(self, value):
        for block in self.up_blocks:
            value = block(value)
        return value


def check(mode):
    torch.manual_seed(42)
    adapter = MSDFAdapter(4, 8, [8] * 4, branch_dropout=0.2,
                          training_ablation=mode)
    unet = TinyUNet()
    adapter.attach(unet)
    adapter.capture_injection_stats = True
    # Zero initialization is intentional. Make projections nonzero only for
    # this routing test so injection and CFG can be observed before training.
    with torch.no_grad():
        for projection in adapter.projections:
            projection.weight.fill_(0.01)
    calls = {"latent": 0, "morph": 0}
    latent_handle = adapter.encoder.register_forward_hook(
        lambda *_: calls.__setitem__("latent", calls["latent"] + 1))
    original_align = adapter._align_reference

    def align(*args, **kwargs):
        calls["morph"] += 1
        return original_align(*args, **kwargs)

    adapter._align_reference = align
    adapter.train()
    latents = torch.randn(1, 4, 8, 8)
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    mask = torch.zeros(1, 1, 64, 64); mask[..., 20:40, 24:44] = 1
    prior = torch.zeros(1, 1, 8, 8); prior[..., 2:6, 3:7] = 1
    adapter.prepare(latents, mask, prior, torch.tensor([500]), 1000, True,
                    reference_pixels=pixels)
    output = unet(torch.randn(2, 8, 8, 8))
    loss = output.square().mean()
    loss.backward()
    active = [name for name, value in adapter.named_parameters()
              if value.requires_grad]
    optimizer = torch.optim.AdamW(
        [value for value in adapter.parameters() if value.requires_grad], lr=1e-4)
    optimizer_names = {name for name, value in adapter.named_parameters()
                       if any(value is item for group in optimizer.param_groups
                              for item in group["params"])}
    encoder_grad = any(value.grad is not None for value in adapter.encoder.parameters())
    stats = adapter.last_injection_stats
    checks = {
        "latent_calls_expected": calls["latent"] == (0 if mode == "no_latent" else 1),
        "morph_calls_expected": calls["morph"] == (0 if mode == "no_morph" else 1),
        "latent_optimizer_excluded": (not any(name.startswith("encoder.") for name in optimizer_names))
            if mode == "no_latent" else True,
        "latent_gradient_absent": (not encoder_grad) if mode == "no_latent" else encoder_grad,
        "morphology_loss_disabled": adapter.morphology_supervision_enabled == (mode != "no_morph"),
        "four_up_blocks_injected": len(stats) == 4,
        "cfg_unconditional_zero": all(row["unconditional_l2"] == 0 for row in stats),
        "cfg_conditional_nonzero": all(row["conditional_l2"] > 0 for row in stats),
        "all_finite": bool(torch.isfinite(output).all()) and all(row["finite"] for row in stats),
    }
    latent_handle.remove(); adapter.detach()
    return {
        "mode": mode, "status": "PASS" if all(checks.values()) else "FAIL",
        "contract": adapter.training_contract(), "calls": calls,
        "active_parameter_names": active, "checks": checks,
        "injection": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {mode: check(mode) for mode in ("full", "no_latent", "no_morph")}
    payload = {"status": "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL",
               "seed": 42, "synthetic_tensor_scope": "routing unit test only",
               "variants": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in results.items()}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
