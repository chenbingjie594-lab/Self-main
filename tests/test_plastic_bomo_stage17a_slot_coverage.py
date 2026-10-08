import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_plastic_bomo_stage17a_slot_coverage import check_coverage, load
from probe_plastic_bomo_stage17a_frozen_slots import fixed_probe_slots


class CoverageTests(unittest.TestCase):
    def test_probe_subset_fixed_and_order_independent(self):
        rows = [{'slot_id': f'{c}_{i}', 'class': c} for c in ('flash','black') for i in range(40)]
        selected = fixed_probe_slots(rows)
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(r['class']=='flash' for r in selected),3)
        self.assertEqual(selected, fixed_probe_slots(list(reversed(rows))))

    def fixture(self):
        data = ROOT / 'dataset/plastic_bomo_stage17a_trainonly_local'
        if not data.is_dir(): self.skipTest('local exact-source test assets not installed')
        base = ROOT / 'results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability'
        slots = load(base / 'slots_frozen_20261007_020941/generation_slot_manifest.json')['slots']
        sources = load(base / 'trainonly_20261006_022714/preparation/trainonly_source_manifest.json')['records']
        frozen = load(ROOT / 'results_for_gpt/plastic_bomo_stage16a_r0_traceability_preflight/annotation_provenance_audit.json')['records']
        return slots, sources, frozen, data

    def test_all80_exact_coverage(self):
        audit = check_coverage(*self.fixture())
        self.assertEqual(audit['status'], 'SLOT_MASK_LABEL_COVERAGE_PASS')
        self.assertEqual(audit['slot_count'],80)
        self.assertEqual(audit['projected_label_counts'],{'flash':41,'black':44})
        self.assertFalse(audit['annotation_accuracy_or_generated_defect_identity_proven'])

    def test_incorrect_projection_cannot_pass(self):
        slots, sources, frozen, data = self.fixture()
        wrong = copy.deepcopy(slots[:1]); wrong[0]['synthetic_labels'][0]['yolo_xywh'][0] += .01
        audit = check_coverage(wrong, sources, frozen, data)
        self.assertEqual(audit['status'],'SLOT_MASK_LABEL_COVERAGE_FAILED')
        self.assertEqual(audit['failed_slot_ids'],[wrong[0]['slot_id']])


if __name__ == '__main__': unittest.main()
