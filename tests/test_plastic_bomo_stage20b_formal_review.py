"""CPU arithmetic and fail-closed hash checks for the separate downloaded review."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_plastic_bomo_stage20b_formal import Review, recompute


class FormalReviewTests(unittest.TestCase):
    def metrics(self, values, black=None):
        return {name: {'map5095': v, 'per_class': {'flash': {'ap5095': v},
                'black': {'ap5095': black[i] if black else v}}}
                for i, (name, v) in enumerate(zip(('RR', 'RawA', 'RawB', 'RawC'), values))}

    def gate(self):
        return {'A_mean_raw_bank_minus_RR_ge_pp': .5, 'B_strict_bank_wins_ge': 2,
                'C_every_class_mean_delta_ge_pp': -2.}

    def test_failure_and_sample_sd(self):
        x = recompute(self.metrics([.3597019289721092, .2970998912654449,
                                   .33081838919630263, .37348318684014364]), self.gate())
        self.assertAlmostEqual(x['mean_delta_pp'], -2.5901439871478784)
        self.assertAlmostEqual(x['std_delta_pp'], 3.8278866892745036)
        self.assertEqual(x['wins_vs_RR'], 1)
        self.assertFalse(x['TDCRG_DEVELOPMENT_AUTHORIZED'])

    def test_equal_not_a_strict_win(self):
        x = recompute(self.metrics([.4, .4, .4, .5]), self.gate())
        self.assertEqual(x['wins_vs_RR'], 1)
        self.assertFalse(x['Gate_B'])

    def test_all_frozen_gates_required(self):
        x = recompute(self.metrics([.4, .41, .42, .43], black=[.4, .3, .3, .3]), self.gate())
        self.assertTrue(x['Gate_A'])
        self.assertTrue(x['Gate_B'])
        self.assertFalse(x['Gate_C'])
        self.assertFalse(x['TDCRG_DEVELOPMENT_AUTHORIZED'])

    def test_threshold_is_read_not_redefined(self):
        gate = self.gate()
        gate['A_mean_raw_bank_minus_RR_ge_pp'] = 10.
        self.assertFalse(recompute(self.metrics([.4, .41, .42, .43]), gate)['Gate_A'])

    def test_pass_authorization_is_not_automatic_execution(self):
        x = recompute(self.metrics([.4, .41, .42, .43]), self.gate())
        self.assertEqual(x['status'], 'GENERATOR_BASELINE_HEADROOM_CONFIRMED')
        self.assertTrue(x['TDCRG_DEVELOPMENT_AUTHORIZED'])

    def test_available_wrong_bytes_fail_even_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x'
            p.write_bytes(b'wrong')
            r = Review()
            r.verify(tmp, {'x': hashlib.sha256(b'correct').hexdigest()}, 'test', optional=True)
            self.assertFalse(r.checks[-1]['passed'])

    def test_missing_required_fails_optional_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Review()
            r.verify(tmp, {'x': 'missing'}, 'required')
            self.assertFalse(r.checks[-1]['passed'])
            r.verify(tmp, {'x': 'missing'}, 'optional', optional=True)
            self.assertTrue(r.checks[-1]['passed'])
            self.assertEqual(len(r.missing), 1)

    def test_path_escape_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Review()
            r.verify(tmp, {'../outside': 'x'}, 'escape', optional=True)
            self.assertFalse(r.checks[-1]['passed'])


if __name__ == '__main__':
    unittest.main()
