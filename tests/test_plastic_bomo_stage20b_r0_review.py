"""Downloaded R0 evidence/tamper tests; never access original dataset or GPU."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import review_plastic_bomo_stage20b_r0 as reviewer

SERVER = ROOT / 'results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254'
P0 = ROOT / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115'
STAGE20A = ROOT / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol'


class R0ReviewTests(unittest.TestCase):
    def review(self, server):
        return reviewer.review(ROOT, server, P0, STAGE20A)

    def tamper_json_and_rehash(self, server, name, change):
        path = server / name
        value = reviewer.load(path)
        change(value)
        reviewer.save(path, value)
        manifest = reviewer.load(server / 'r0_artifact_manifest.json')
        manifest['sha256'][name] = reviewer.sha(path)
        reviewer.save(server / 'r0_artifact_manifest.json', manifest)

    def test_downloaded_server_independent_review_passes(self):
        result = self.review(SERVER)
        self.assertEqual(result['status'], 'STAGE20B_R0_SERVER_REVIEW_PASS')
        self.assertEqual(result['checks_passed'], result['checks_total'])
        self.assertEqual(result['checks_total'], 30)
        self.assertEqual(result['downloaded_artifacts']['count'], 27)
        self.assertTrue(result['STAGE20B_FORMAL_EXECUTION_READY'])
        self.assertFalse(result['TDCRG_DEVELOPMENT_AUTHORIZED'])
        self.assertFalse(result['formal_auto_start'])
        self.assertFalse(result['local_GPU_rerun'])

    def test_rehashed_rng_seed_tamper_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            server = Path(t) / 'server'
            shutil.copytree(SERVER, server)
            self.tamper_json_and_rehash(server, 'generator_rng_probe.json',
                lambda x: x['records'][0]['runs'][2].update(sample_seed=1))
            result = self.review(server)
            self.assertTrue(result['checks']['all27_downloaded_artifact_bytes'])
            self.assertFalse(result['checks']['A_B_noise_independence_and_frozen_rng_metadata'])
            self.assertFalse(result['STAGE20B_FORMAL_EXECUTION_READY'])

    def test_rehashed_optimizer_step_tamper_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            server = Path(t) / 'server'
            shutil.copytree(SERVER, server)
            self.tamper_json_and_rehash(server, 'stage20b_r0_status.json', lambda x: x.update(optimizer_step_count=1))
            result = self.review(server)
            self.assertFalse(result['checks']['no_formal_steps_training_sampling_eval_or_module_development'])
            self.assertFalse(result['STAGE20B_FORMAL_EXECUTION_READY'])

    def test_rehashed_parameter_invariance_tamper_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            server = Path(t) / 'server'
            shutil.copytree(SERVER, server)
            def change(x):
                hashes = x['classes'][0]['parameter_hashes_after']['unet']
                hashes[next(iter(hashes))] = '0' * 64
            self.tamper_json_and_rehash(server, 'generator_forward_backward_audit.json', change)
            result = self.review(server)
            self.assertFalse(result['checks']['both_generator_losses_gradients_and_parameter_invariance'])

    def test_rehashed_tail64_instead24_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            server = Path(t) / 'server'
            shutil.copytree(SERVER, server)
            self.tamper_json_and_rehash(server, 'detector_accumulation_dryrun.json',
                lambda x: x['attempt_groups'][3].update(normalizer=64))
            result = self.review(server)
            self.assertFalse(result['checks']['all600_group_attempts_and_32400_draws_independently_recomputed'])

    def test_missing_probe_is_reported_by_ledger(self):
        with tempfile.TemporaryDirectory() as t:
            server = Path(t)
            reviewer.save(server / 'r0_artifact_manifest.json', {'sha256': {'runtime_probe/missing.png': '0' * 64}})
            ledger = reviewer.downloaded_ledger(server)
            self.assertFalse(ledger['pass'])
            self.assertIsNone(ledger['files'][0]['actual_sha256'])

    def test_ledger_traversal_refused_before_reading(self):
        with tempfile.TemporaryDirectory() as t:
            server = Path(t)
            reviewer.save(server / 'r0_artifact_manifest.json', {'sha256': {'../outside.jpg': '0' * 64}})
            with self.assertRaisesRegex(ValueError, 'R0_LEDGER_PATH_OUTSIDE'):
                reviewer.downloaded_ledger(server)

    def test_nonfinite_numbers_and_zero_gradients_not_accepted(self):
        self.assertFalse(reviewer.finite(float('nan')))
        self.assertFalse(reviewer.finite(float('inf')))
        self.assertFalse(reviewer.valid_gradients({'all_gradients_finite': True, 'all_trainable_parameters_fp32': True,
            'nonzero_gradients': 0, 'non_none_gradients': 686, 'parameters': 686}))


if __name__ == '__main__':
    unittest.main()
