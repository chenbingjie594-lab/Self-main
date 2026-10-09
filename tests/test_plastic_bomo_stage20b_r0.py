"""R0 CPU/metadata tests. These do NOT stand in for the server GPU qualification."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import stage20b_runtime_contract as contract
from stage20b_detector_controller import Stage20BDetectorController, accumulation_dryrun
import qualify_plastic_bomo_stage20b_r0 as qualifier

P0 = ROOT / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115'
STAGE20A = ROOT / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol'
CFG = contract.load(ROOT / 'configs/plastic_bomo_stage20b_p0.json')
ORDER = contract.load(P0 / 'stage20b_training_order_spec.json')
TRAIN = contract.load(P0 / 'stage20b_generator_training_protocol.json')


class RuntimeContractTests(unittest.TestCase):
    def test_frozen_stage20a_and_server_p0_ledgers(self):
        self.assertTrue(contract.verify_ledger(STAGE20A, 'frozen_artifact_manifest.json', contract.STAGE20A_MANIFEST_SHA))
        self.assertTrue(contract.verify_ledger(P0, 'p0_frozen_artifact_manifest.json', contract.P0_MANIFEST_SHA))

    def test_config_and_p0_script_unchanged(self):
        self.assertEqual(contract.sha(ROOT / 'configs/plastic_bomo_stage20b_p0.json'), contract.PROTOCOL_SHA)
        self.assertEqual(contract.sha(ROOT / 'tools/preflight_plastic_bomo_stage20b_p0.py'), contract.P0_SCRIPT_SHA)

    def test_changed_and_missing_file_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'artifact.json'
            artifact.write_bytes(b'{}')
            contract.save(root / 'ledger.json', {'sha256': {'artifact.json': contract.sha(artifact)}})
            expected = contract.sha(root / 'ledger.json')
            artifact.write_bytes(b'{ }')
            with self.assertRaises(contract.Stop) as error:
                contract.verify_ledger(root, 'ledger.json', expected)
            self.assertEqual(error.exception.status, 'FROZEN_RUNTIME_BINDING_CHANGED')
            artifact.unlink()  # Exact disposable test artifact only.
            with self.assertRaises(contract.Stop):
                contract.verify_ledger(root, 'ledger.json', expected)

    def test_ledger_traversal_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract.save(root / 'ledger.json', {'sha256': {'../outside.json': '0' * 64}})
            with self.assertRaises(contract.Stop):
                contract.verify_ledger(root, 'ledger.json', contract.sha(root / 'ledger.json'))

    def test_missing_or_changed_manifest_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(contract.Stop):
                contract.verify_ledger(directory, 'missing.json', '0' * 64)

    def test_all150_frozen_epochs_validate(self):
        contract.validate_schedule(CFG, ORDER)

    def test_five_seeds_independently_recomputed(self):
        for epoch, position in ((0, 0), (0, 63), (0, 64), (0, 215), (149, 215)):
            encoded = (CFG['protocol'] + 'augmentation' + '42' + str(epoch) + str(position)).encode()
            expected = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')
            self.assertEqual(contract.augmentation_seed(CFG['protocol'], epoch, position), expected)
            self.assertEqual(ORDER['epochs'][epoch]['augmentation_seeds'][position], expected)

    def test_role_seed_and_tail_changes_refused(self):
        for field in ('role_sequence', 'augmentation_seeds', 'accumulation_group_lengths'):
            changed = deepcopy(ORDER)
            row = changed['epochs'][0]
            if field == 'role_sequence':
                row[field][0], row[field][1] = row[field][1], row[field][0]
            else:
                row[field][-1] += 1
            with self.assertRaises(contract.Stop):
                contract.validate_schedule(CFG, changed)

    def test_generator2000_successful_not_attempted(self):
        base = contract.generator_budget_dryrun(TRAIN, {'flash': 75, 'black': 97})
        skips = contract.generator_budget_dryrun(TRAIN, {'flash': 75, 'black': 97}, (1, 7, 31, 1999))
        for cls in ('flash', 'black'):
            self.assertEqual(base[cls]['successful_updates'], 2000)
            self.assertEqual(base[cls]['attempted_updates'], 2000)
            self.assertEqual(skips[cls]['successful_updates'], 2000)
            self.assertEqual(skips[cls]['attempted_updates'], 2004)
            self.assertEqual(skips[cls]['simulated_amp_skips'], 4)
            self.assertTrue(skips[cls]['checkpoint_may_be_named_final_step2000'])
        self.assertEqual({x['draws'] for x in base['flash']['epoch_end_partial_groups']}, {3})
        self.assertEqual({x['draws'] for x in base['black']['epoch_end_partial_groups']}, {1})

    def test_changed_generator_budget_refused(self):
        changed = deepcopy(TRAIN)
        changed['max_train_steps'] = 1999
        with self.assertRaises(contract.Stop):
            contract.generator_budget_dryrun(changed, {'flash': 75, 'black': 97})
        with self.assertRaises(contract.Stop):
            contract.generator_budget_dryrun(TRAIN, {'flash': 0, 'black': 97})

    def test_binding_failure_writes_no_false_ready_or_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'r0'
            argv = ['qualify', '--base_model', str(ROOT / 'sd2-community/stable-diffusion-2-inpainting'),
                    '--detector_model', str(ROOT / 'pretrained/yolo11s.pt'), '--output', str(root)]
            with patch.object(sys, 'argv', argv), patch.object(qualifier, 'binding_recheck',
                    side_effect=contract.Stop('FROZEN_RUNTIME_BINDING_CHANGED', 'injected failure')), \
                    patch.object(qualifier, 'generator_qualification') as generator, \
                    patch.object(qualifier, 'detector_qualification') as detector, patch.dict(qualifier.os.environ):
                with self.assertRaises(SystemExit) as error:
                    qualifier.main()
                self.assertEqual(error.exception.code, 2)
                generator.assert_not_called()
                detector.assert_not_called()
            result = contract.load(root / 'stage20b_r0_status.json')
            self.assertEqual(result['status'], 'FROZEN_RUNTIME_BINDING_CHANGED')
            self.assertFalse(result['STAGE20B_FORMAL_EXECUTION_READY'])
            for key in ('optimizer_step_count', 'generator_training_count', 'synthetic_generation_count',
                        'detector_training_count', 'official_final_eval_forward', 'final_eval_image_or_label_content_read'):
                self.assertEqual(result[key], 0)
            self.assertTrue(all(contract.load(root / (x + '.json'))['status'] == 'NOT_RUN' for x in qualifier.OUTPUTS))

    def test_final_eval_guard_blocks_before_content_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata').mkdir()
            (root / 'p0').mkdir()
            train = root / 'data/images/train/allowed.jpg'
            contract.save(root / 'p0/generator_train_allowlist.json', {'inputs': [{'path': str(train)}]})
            rows = [{'split_v2': 'final_eval', 'asset_role': 'detector', 'current_path': f'old/images/val/{i}.jpg',
                     'split_membership_old': 'val'} for i in range(48)]
            contract.save(root / 'metadata/real_source_registry.json', {'records': rows})
            blocked = root / 'data/images/val/0.jpg'
            blocked.parent.mkdir(parents=True)
            blocked.write_bytes(b'not read by R0')
            counts = {'forbidden_final_eval_open_attempts': 0}
            with qualifier.final_eval_content_forbidden(root / 'metadata', root / 'p0', counts):
                with self.assertRaises(contract.Stop) as error:
                    blocked.read_bytes()
                self.assertEqual(error.exception.status, 'R0_FINAL_EVAL_CONTENT_ACCESS_FORBIDDEN')
                self.assertEqual(counts['forbidden_final_eval_open_attempts'], 1)
            # Dormant hook no longer interferes with disposable test cleanup.
            self.assertEqual(blocked.read_bytes(), b'not read by R0')

    def test_optimizer_step_guard_actually_intercepts(self):
        class DummyOptimizer:
            def step(self):
                raise AssertionError('must never reach actual step')
        torch = type('DummyTorch', (), {'optim': type('Optim', (), {'AdamW': DummyOptimizer, 'SGD': DummyOptimizer})})
        counts = {'forbidden_optimizer_step_calls': 0}
        with qualifier.optimizer_steps_forbidden(torch, counts):
            with self.assertRaises(contract.Stop) as error:
                DummyOptimizer().step()
            self.assertEqual(error.exception.status, 'R0_OPTIMIZER_STEP_FORBIDDEN')
        self.assertEqual(counts['forbidden_optimizer_step_calls'], 1)

    def test_no_gpu_pass_claim_in_cpu_test(self):
        # Importing the R0 CLI must not import torch or start probes/experiments.
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(qualifier.formal_plan()['formal_training_or_sampling_started'])


class DetectorControllerTests(unittest.TestCase):
    def test32400_draws600_groups_and_mean24_tail(self):
        result = accumulation_dryrun(CFG, ORDER)
        self.assertEqual((result['epochs'], result['draws'], result['scheduled_optimizer_attempts']), (150, 32400, 600))
        self.assertEqual(result['first_epoch_mean_gradients'], [32.5, 96.5, 160.5, 204.5])
        self.assertEqual([x['normalizer'] for x in result['attempt_groups'][:4]], [64, 64, 64, 24])
        self.assertEqual(sum(x['draws'] == 24 for x in result['attempt_groups']), 150)
        self.assertEqual(result['actual_optimizer_steps'], 0)

    def test_actual_callback_order_and_tail_attempt(self):
        controller = Stage20BDetectorController(CFG, ORDER)
        observed, groups = [], []
        counts = controller.run(lambda e, p, role, seed, length: observed.append((e, p, role, seed, length)),
                                lambda e, p, length: groups.append((e, p, length)))
        self.assertEqual(observed[0], (0, 0, ORDER['epochs'][0]['role_sequence'][0],
                                      ORDER['epochs'][0]['augmentation_seeds'][0], 64))
        self.assertEqual(groups[-1], (149, 215, 24))
        self.assertEqual(counts['draws'], len(observed))

    def test_cleanup_normal_and_exception_no_validator(self):
        controller = Stage20BDetectorController(CFG, ORDER)
        cleanup, forbidden_validator = Mock(), Mock(side_effect=AssertionError('validation forbidden'))
        controller.validator = controller.final_eval = controller.stopper = forbidden_validator
        controller.run(lambda *args: None, lambda *args: None, cleanup=cleanup)
        def fail(*args):
            raise ValueError('injected failure')
        with self.assertRaises(ValueError):
            controller.run(fail, lambda *args: None, cleanup=cleanup)
        self.assertEqual(cleanup.call_count, 2)
        forbidden_validator.assert_not_called()

    def test_warmup_bias_lr_and_fixed_adamw_betas(self):
        controller = Stage20BDetectorController(CFG, ORDER)
        groups = [{'param_group': name, 'initial_lr': .001, 'betas': (.9, .999)} for name in ('bias', 'weight', 'bn')]
        self.assertEqual(controller.learning_rates(0, 0, groups), [.1, 0., 0.])
        self.assertAlmostEqual(controller.learning_rates(0, 64, groups)[1], .001 * 64 / 648)
        final = controller.learning_rates(149, 215, groups)
        expected = .001 * ((1 - 149 / 150) * .99 + .01)
        self.assertTrue(all(abs(x - expected) < 1e-12 for x in final))
        self.assertTrue(all(g['betas'] == (.9, .999) for g in groups))

    def test_backward_helper_divides_by_actual_group_length(self):
        class Loss:
            def __init__(self, value): self.value = value
            def sum(self): return self
            def __truediv__(self, n): self.value /= n; return self
            def backward(self): self.gradient = self.value
        for n in (64, 24):
            loss = Loss(float(n))
            Stage20BDetectorController.backward(loss, n)
            self.assertEqual(loss.gradient, 1.)

    def test_ast_no_optimizer_step_high_level_train_or_val(self):
        for name in ('qualify_plastic_bomo_stage20b_r0.py', 'stage20b_detector_controller.py', 'stage20b_runtime_contract.py'):
            tree = ast.parse((ROOT / 'tools' / name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('step', 'val', 'final_eval', 'validate'))
                    if node.func.attr == 'train':
                        # nn.Module.train() sets mode; never YOLO.train()/BaseTrainer.train().
                        self.assertIn(ast.unparse(node.func.value), ('pipe.unet.requires_grad_(True)',))
            self.assertNotIn('__class__=', (ROOT / 'tools' / name).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
