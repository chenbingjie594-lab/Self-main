import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn as nn


MODULE_PATH = (
    Path(__file__).parents[1]
    / "diffusers"
    / "pipelines"
    / "stable_diffusion"
    / "msdf_guidance.py"
)
SPEC = importlib.util.spec_from_file_location("msdf_guidance_test", MODULE_PATH)
MSDF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MSDF)


class _ResidualMetadata(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.out_channels = channels

    def forward(self, sample):
        return sample


class _FakeUpBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = nn.ModuleList([_ResidualMetadata(channels)])

    def forward(self, sample):
        return sample


class _FakeUNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up_blocks = nn.ModuleList([_FakeUpBlock(value) for value in channels])


class MSDFAdapterTest(unittest.TestCase):
    def test_safe_rms_has_finite_gradient_at_zero(self):
        value = torch.zeros(2, 4, 8, 8, requires_grad=True)
        rms = MSDF._safe_rms(value, dims=(1, 2, 3))
        self.assertTrue(torch.isfinite(rms).all())
        rms.sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_cfg_injection_shape_and_gradient(self):
        unet = _FakeUNet([16, 8])
        adapter = MSDF.MSDFAdapter(4, 8, [16, 8])
        adapter.eval()
        self.assertEqual(adapter.attach(unet), 2)
        nn.init.constant_(adapter.projections[0].weight, 0.01)

        reference = torch.randn(1, 4, 8, 8)
        reference_mask = torch.zeros(1, 1, 8, 8)
        reference_mask[..., 2:5, 2:5] = 1
        target_mask = torch.zeros(1, 1, 8, 8)
        target_mask[..., 4:7, 4:7] = 1
        gates = adapter.prepare(
            reference,
            reference_mask,
            target_mask,
            torch.tensor([500]),
            1000,
            classifier_free_guidance=True,
            reference_pixels=torch.randn(1, 3, 64, 64),
        )
        self.assertEqual(tuple(gates.shape), (1, 2))

        sample = torch.zeros(2, 16, 8, 8)
        output = unet.up_blocks[0](sample)
        self.assertEqual(tuple(output.shape), tuple(sample.shape))
        self.assertTrue(torch.equal(output[0], sample[0]))
        self.assertGreater(float(output[1].abs().sum()), 0.0)
        output[1].sum().backward()
        self.assertIsNotNone(adapter.projections[0].weight.grad)

    def test_morphology_supervision_only_uses_self_reference(self):
        adapter = MSDF.MSDFAdapter(4, 8, [16, 8])
        reference = torch.randn(2, 4, 8, 8)
        reference_mask = torch.zeros(2, 1, 8, 8)
        reference_mask[0, :, 2:4, 1:6] = 1
        reference_mask[1, :, 1:6, 3:5] = 1
        target_mask = torch.zeros(2, 1, 8, 8)
        target_mask[:, :, 2:7, 2:7] = 1
        adapter.prepare(
            reference,
            reference_mask,
            target_mask,
            torch.tensor([250, 750]),
            1000,
            classifier_free_guidance=False,
            reference_pixels=torch.randn(2, 3, 64, 64),
        )
        target_support = torch.zeros_like(target_mask)
        target_support[0, :, 3:5, 2:6] = 1
        target_support[1, :, 2:6, 3:5] = 1
        loss = adapter.morphology_support_loss(
            target_support, torch.tensor([True, False])
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(adapter.support_head.weight.grad)
        self.assertIsNotNone(adapter.pixel_support_head.weight.grad)

    def test_injection_is_bounded_relative_to_hidden_rms(self):
        unet = _FakeUNet([16])
        adapter = MSDF.MSDFAdapter(
            4, 8, [16], max_injection=1.0, max_residual_ratio=0.25
        ).eval()
        adapter.attach(unet)
        nn.init.constant_(adapter.projections[0].weight, 100.0)
        nn.init.constant_(adapter.projections[0].bias, 100.0)
        reference_mask = torch.zeros(1, 1, 8, 8)
        reference_mask[..., 2:6, 2:6] = 1
        adapter.prepare(
            torch.randn(1, 4, 8, 8),
            reference_mask,
            reference_mask,
            torch.tensor([500]),
            1000,
            classifier_free_guidance=False,
            reference_pixels=torch.randn(1, 3, 64, 64),
        )
        sample = torch.ones(1, 16, 8, 8)
        output = unet.up_blocks[0](sample)
        delta = (output - sample).abs()
        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(float(delta.max()), 0.25001)

    def test_zero_aligned_feature_has_finite_full_backward(self):
        unet = _FakeUNet([16])
        adapter = MSDF.MSDFAdapter(4, 8, [16]).eval()
        adapter.attach(unet)
        for module in (adapter.encoder, adapter.pixel_encoder, adapter.fusion):
            for parameter in module.parameters():
                nn.init.zeros_(parameter)
        nn.init.zeros_(adapter.projections[0].weight)
        nn.init.ones_(adapter.projections[0].bias)
        reference_mask = torch.zeros(1, 1, 8, 8)
        reference_mask[..., 2:6, 2:6] = 1
        adapter.prepare(
            torch.zeros(1, 4, 8, 8),
            reference_mask,
            reference_mask,
            torch.tensor([500]),
            1000,
            classifier_free_guidance=False,
            reference_pixels=torch.zeros(1, 3, 64, 64),
        )
        output = unet.up_blocks[0](torch.ones(1, 16, 8, 8))
        output.sum().backward()
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_nonfinite_reference_cannot_break_geometry(self):
        adapter = MSDF.MSDFAdapter(4, 8, [16, 8])
        reference = torch.randn(1, 4, 8, 8)
        reference[..., 3, 3] = float("nan")
        reference_mask = torch.zeros(1, 1, 8, 8)
        reference_mask[..., 2:6, 2:6] = 1
        target_mask = torch.zeros(1, 1, 8, 8)
        target_mask[..., 3:7, 1:7] = 1
        pixels = torch.randn(1, 3, 64, 64)
        pixels[..., 20, 20] = float("inf")
        gates = adapter.prepare(
            reference,
            reference_mask,
            target_mask,
            torch.tensor([500]),
            1000,
            classifier_free_guidance=False,
            reference_pixels=pixels,
        )
        self.assertTrue(torch.isfinite(gates).all())
        self.assertTrue(torch.isfinite(adapter.last_aligned_support).all())


if __name__ == "__main__":
    unittest.main()
