import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_plastic_bomo_stage17a_normals_isolated import eligible_records, EXCLUDED
from verify_plastic_bomo_stage17a_normal_server import load


class IsolatedNormalTests(unittest.TestCase):
    def setUp(self):
        root = ROOT / 'results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability'
        self.rows = load(root / 'normal_lineage_local_audit/normal_origin_manifest.json')['records']
        self.accepted = {Path(k).name: v for k, v in load(root / 'trainonly_runtime_20261006_051524/frozen_probe_conditions.json')['normal_pool_files_sha256'].items()}

    def test_exact_174_no_mutation(self):
        before = copy.deepcopy(self.rows)
        new = eligible_records(self.rows, self.accepted, EXCLUDED)
        self.assertEqual(len(new), 174)
        self.assertFalse({r['normal_filename'] for r in new} & EXCLUDED)
        self.assertEqual(self.rows, before)
        self.assertFalse(any(r['related_validation_parents'] for r in new))

    def test_cannot_silently_add_or_remove_exclusions(self):
        for excluded in (EXCLUDED - {'076.jpg'}, EXCLUDED | {'000.jpg'}):
            with self.assertRaisesRegex(RuntimeError, 'EXACT_AUTHORIZED_SIX'):
                eligible_records(self.rows, self.accepted, excluded)

    def test_extra_relationship_aborts_instead_of_new_filter(self):
        rows = copy.deepcopy(self.rows)
        next(r for r in rows if r['normal_filename'] in self.accepted and r['normal_filename'] not in EXCLUDED)['related_validation_parents'] = [{'new': 'relationship'}]
        with self.assertRaisesRegex(RuntimeError, 'NO_AUTOMATIC_NEW_EXCLUSIONS'):
            eligible_records(rows, self.accepted, EXCLUDED)

    def test_changed_accepted_pool_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'ORIGINAL_ACCEPTED_POOL_CHANGED'):
            eligible_records(self.rows, {}, EXCLUDED)


if __name__ == '__main__':
    unittest.main()
