import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "diffusers"
    / "pipelines"
    / "stable_diffusion"
    / "dhfg_guidance.py"
)
SPEC = importlib.util.spec_from_file_location("dhfg_guidance_test", MODULE_PATH)
DHFG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DHFG)


class DHFGGuidanceTest(unittest.TestCase):
    def test_local_crop_preserves_gradient(self):
        value = torch.randn(2, 4, 16, 16, requires_grad=True)
        mask = torch.zeros(2, 1, 64, 64)
        mask[0, :, 30:34, 20:25] = 1
        mask[1, :, 10:15, 40:45] = 1
        crop, crop_mask = DHFG.crop_mask_regions(value, mask, 12)
        self.assertEqual(tuple(crop.shape), (2, 4, 12, 12))
        self.assertEqual(tuple(crop_mask.shape), (2, 1, 12, 12))
        crop.square().mean().backward()
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_paired_confidence_uses_a_symmetric_tolerance(self):
        confidence = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
        loss = DHFG.paired_confidence_loss(
            confidence, torch.tensor([0.5, 0.5, 0.5]), tolerance=0.2
        )
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertLess(float(confidence.grad[0]), 0.0)
        self.assertEqual(float(confidence.grad[1]), 0.0)
        self.assertGreater(float(confidence.grad[2]), 0.0)

    def test_paired_confidence_large_gap_is_linear_not_quadratic(self):
        confidence = torch.tensor([-13.0], requires_grad=True)
        loss = DHFG.paired_confidence_loss(
            confidence, torch.tensor([0.0]), tolerance=0.0, beta=1.0
        )
        self.assertLess(float(loss), 13.0)
        self.assertGreater(float(loss), 10.0)
        loss.backward()
        self.assertAlmostEqual(float(confidence.grad), -1.0, places=5)

    def test_calibrated_interval_penalises_only_outside_real_band(self):
        values = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        loss = DHFG.calibrated_interval_loss(
            values, torch.tensor(0.0), torch.tensor(1.0), beta=0.5
        )
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertLess(float(values.grad[0]), 0.0)
        self.assertEqual(float(values.grad[1]), 0.0)
        self.assertGreater(float(values.grad[2]), 0.0)

    def test_photometric_signature_separates_bright_and_dark_defects(self):
        mask = torch.zeros(2, 1, 32, 32)
        mask[:, :, 12:20, 12:20] = 1
        pixels = torch.full((2, 3, 32, 32), 0.5)
        pixels[0, :, 14:18, 14:18] = 0.9
        pixels[1, :, 14:18, 14:18] = 0.1
        signature = DHFG._local_photometric_signature(pixels, mask, context_radius=3)
        self.assertGreater(float(signature["positive"][0]), float(signature["negative"][0]))
        self.assertGreater(float(signature["negative"][1]), float(signature["positive"][1]))
        self.assertTrue(torch.isfinite(signature["coverage"]).all())
        self.assertTrue(torch.isfinite(signature["extent"]).all())

    def test_masked_topk_uses_sparse_object_responses(self):
        values = torch.tensor([[[[0.0, 1.0, 8.0, 3.0, 2.0, 0.0]]]])
        mask = torch.tensor([[[[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]]])
        result = DHFG._masked_topk_mean(
            values, mask, fraction=0.50, maximum_count=2, empty_value=-1e4
        )
        self.assertAlmostEqual(float(result), 5.5)

    def test_tiny_support_survives_downsampling(self):
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 31:33, 31:33] = 1
        resized = DHFG._resize_support(mask, (8, 8))
        self.assertGreater(float(resized.sum()), 0.0)

    def test_feature_statistics_match_identical_features(self):
        feature = torch.randn(2, 8, 12, 12)
        mask = torch.zeros(2, 1, 48, 48)
        mask[:, :, 16:32, 18:34] = 1
        first = DHFG._masked_feature_statistics(feature, mask)
        second = DHFG._masked_feature_statistics(feature.clone(), mask)
        for left, right in zip(first, second):
            self.assertTrue(torch.allclose(left, right))

    def test_reference_response_selects_sparse_core_and_context(self):
        response = torch.zeros(1, 1, 8, 8)
        response[:, :, 3, 4] = 0.9
        response[:, :, 4, 4] = 0.8
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1
        core, context = DHFG._reference_core_context_support(
            response,
            mask,
            core_fraction=0.25,
            maximum_core_locations=2,
            context_radius=1,
        )
        self.assertEqual(float(core.sum()), 2.0)
        self.assertGreater(float(context.sum()), 0.0)
        self.assertEqual(float((core * context).sum()), 0.0)
        self.assertEqual(float(core[0, 0, 3, 4]), 1.0)

    def test_soft_core_uses_reference_contrast_when_teacher_is_weak(self):
        response = torch.full((1, 1, 8, 8), 0.01)
        contrast = torch.zeros(1, 1, 32, 32)
        contrast[:, :, 14:18, 14:18] = 4.0
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1
        core, context = DHFG._soft_reference_core_context_support(
            response, contrast, mask, context_radius=1
        )
        self.assertGreater(float(core.max()), 0.9)
        self.assertGreater(float(context.sum()), 0.0)
        self.assertTrue(torch.isfinite(core).all())

    def test_counterfactual_erase_reduces_local_spot_and_preserves_outside(self):
        pixels = torch.full((1, 3, 32, 32), 0.5)
        pixels[:, :, 15:17, 15:17] = 0.0
        support = torch.zeros(1, 1, 32, 32)
        support[:, :, 13:19, 13:19] = 1.0
        erased = DHFG.counterfactual_erase(pixels, support, blur_radius=4)
        self.assertGreater(float(erased[:, :, 15:17, 15:17].mean()), 0.3)
        self.assertTrue(torch.equal(erased[:, :, :10, :10], pixels[:, :, :10, :10]))

    def test_counterfactual_erase_preserves_input_dtype(self):
        pixels = torch.rand(1, 3, 16, 16, dtype=torch.float16)
        support = torch.zeros(1, 1, 16, 16)
        support[:, :, 6:10, 6:10] = 1
        erased = DHFG.counterfactual_erase(pixels, support, blur_radius=2)
        self.assertEqual(erased.dtype, torch.float16)

    def test_counterfactual_support_falls_back_to_physical_contrast(self):
        responses = [torch.full((1, 1, 8, 8), 0.01)]
        contrast = torch.zeros(1, 1, 32, 32)
        contrast[:, :, 15:17, 15:17] = 5.0
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1
        support = DHFG._counterfactual_support(
            responses, contrast, mask, (32, 32), erasure_radius=2
        )
        self.assertGreater(float(support[:, :, 15:17, 15:17].mean()), 0.1)
        self.assertEqual(float((support * (1 - mask)).sum()), 0.0)

    def test_boundary_loss_ignores_changes_inside_mask(self):
        reference = torch.zeros(1, 3, 32, 32)
        generated = reference.clone()
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 12:20, 12:20] = 1
        generated[:, :, 12:20, 12:20] = 1
        inside_only = DHFG.boundary_preservation_loss(
            generated, reference, mask, radius=4
        )
        self.assertEqual(float(inside_only), 0.0)
        generated[:, :, 10:12, 12:20] = 1
        leaking = DHFG.boundary_preservation_loss(
            generated, reference, mask, radius=4
        )
        self.assertGreater(float(leaking), 0.0)

    def test_paired_guidance_backpropagates_to_generated_pixels(self):
        class FakeTeacher(DHFG.DHFGTeacher):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.confidence_beta = 1.0
                self.context_radius = 1
                self.boundary_radius = 2
                self.contrast_threshold = 1.0
                self.contrast_temperature = 0.25
                self.erasure_radius = 1
                self.erasure_blur_radius = 2
                self.delta_tolerance = 0.1
                self.register_buffer("confidence_lower", torch.tensor(0.2))
                self.register_buffer("confidence_upper", torch.tensor(0.8))
                self.register_buffer("confidence_delta_lower", torch.tensor(-1.0))
                self.register_buffer("confidence_delta_upper", torch.tensor(1.0))
                self.register_buffer("background_upper", torch.tensor(0.1))

            def _measure(self, pixels, mask):
                confidence = pixels.mean((1, 2, 3))
                background = pixels.new_zeros(pixels.shape[0])
                return {
                    "features": [pixels],
                    "responses": [pixels.mean(1, keepdim=True).sigmoid()],
                    "confidence": confidence,
                    "background": background,
                }

        teacher = FakeTeacher()
        generated = torch.rand(1, 3, 16, 16, requires_grad=True)
        reference = torch.rand(1, 3, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 6:10, 6:10] = 1
        losses = teacher.guidance_losses(generated, reference, mask)
        objective = (
            losses["feature_delta"]
            + losses["response_delta"]
            + losses["confidence_delta_loss"]
            + losses["polarity"]
            + losses["shape"]
            + losses["boundary"]
            + losses["false_background"]
        )
        objective.backward()
        self.assertIsNotNone(generated.grad)
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
