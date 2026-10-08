import copy
import json
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools'))
from audit_plastic_bomo_stage17a import baseline_errors, sample_seed


class Stage17ATests(unittest.TestCase):
    def setUp(self):
        self.baseline=json.loads((REPO/'configs/baseline_frozen_split70_s42.json').read_text(encoding='utf-8'))

    def test_exact_baseline(self):
        self.assertEqual(baseline_errors(self.baseline),[])

    def test_module_change_rejected(self):
        for name in self.baseline['modules']:
            b=copy.deepcopy(self.baseline);b['modules'][name]['enabled']=True
            self.assertTrue(baseline_errors(b))

    def test_normal_filter_not_retuned(self):
        b=copy.deepcopy(self.baseline);b['normal_filter']['min_mean_luminance']=29
        self.assertIn('BASELINE_NORMAL_FILTER_MISMATCH',baseline_errors(b))

    def test_stable_distinct_bank_seed_rule(self):
        values=[sample_seed(bank,f'slot{i}') for bank in (42,3407,2026) for i in range(80)]
        self.assertEqual(len(set(values)),240)
        self.assertTrue(all(0<=v<2**63-1 for v in values))
        self.assertEqual(sample_seed(42,'slot0'),sample_seed(42,'slot0'))

    def test_protocol_is_generation_variance_only(self):
        c=json.loads((REPO/'configs/plastic_bomo_stage17a.json').read_text(encoding='utf-8'))
        self.assertEqual(c['banks'],{'A':42,'B':3407,'C':2026})
        self.assertEqual(c['arms'],['RR','VA','VB','VC'])
        self.assertEqual(c['detector']['seed'],42)
        self.assertFalse(c['selector']);self.assertFalse(c['task_guidance'])
        self.assertFalse(c['stage17b_auto_start'])


if __name__=='__main__': unittest.main()
