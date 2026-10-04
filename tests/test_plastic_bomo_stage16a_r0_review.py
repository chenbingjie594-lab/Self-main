"""Review raw downloaded JSON only; these are not GPU or OOF isolation tests."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from audit_plastic_bomo_stage16a_r0 import load, save
from review_plastic_bomo_stage16a_r0 import review

RESULT = ROOT/'results_for_gpt/plastic_bomo_stage16a_r0_traceability_preflight'


class DownloadReviewTests(unittest.TestCase):
    def test_server_stop_is_consistent_not_pipeline_pass(self):
        out = review(RESULT, RESULT/'local_preflight_snapshot')
        self.assertEqual(out['status'], 'SERVER_STOP_REVIEW_PASS')
        self.assertEqual(out['checks_passed'], 567)
        self.assertFalse(out['gpu_tests_performed'])
        self.assertFalse(out['source_lineage_and_runtime_gates_verified'])

    def test_changed_label_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in RESULT.glob('*.json'):
                shutil.copyfile(path, target/path.name)
            doc = load(target/'annotation_provenance_audit.json')
            doc['records'][0]['annotation_sha256'] = 'changed'
            save(target/'annotation_provenance_audit.json', doc)
            out = review(target, RESULT/'local_preflight_snapshot')
            self.assertEqual(out['status'], 'SERVER_STOP_REVIEW_FAILED')
            self.assertFalse(out['local_server_image_and_label_equality'])

    def test_false_later_gate_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in RESULT.glob('*.json'):
                shutil.copyfile(path, target/path.name)
            doc = load(target/'differentiability_audit.json')
            doc.update(status='PASS', pass_claimed=True)
            save(target/'differentiability_audit.json', doc)
            out = review(target, RESULT/'local_preflight_snapshot')
            self.assertEqual(out['status'], 'SERVER_STOP_REVIEW_FAILED')


if __name__ == '__main__':
    unittest.main()
