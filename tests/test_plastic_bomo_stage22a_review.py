"""Check actual downloaded-result review and rejection of corrupted evidence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import review_plastic_bomo_stage22a as auditor


class Stage22AReviewTest(unittest.TestCase):
    root=Path(__file__).resolve().parents[1]/"results_for_gpt/plastic_bomo_stage22a_failure_mechanism"

    def test_downloaded_formal_records(self):
        result=auditor.review(self.root)
        self.assertEqual(result["status"],"STAGE22A_DOWNLOADED_RESULT_REVIEW_PASS")
        self.assertEqual(result["synthetic_negative_cosine_conflicts"],0)
        self.assertTrue(result["not_identified_even_without_annotation_count_guard"])

    def test_reject_changed_loss(self):
        original=auditor.load
        def corrupt(path):
            value=original(path)
            if Path(path).name=="loss_distribution_audit.json":
                value["records"][0]["loss"]["total_loss"]+=1
            return value
        with patch.object(auditor,"load",corrupt),self.assertRaises(RuntimeError):
            auditor.review(self.root)

    def test_reject_changed_triplet(self):
        original=auditor.load
        def corrupt(path):
            value=original(path)
            if Path(path).name=="matched_triplet_audit.json":
                value["records"][0]["sample_seeds"]["A"]+=1
            return value
        with patch.object(auditor,"load",corrupt),self.assertRaises(RuntimeError):
            auditor.review(self.root)

    def test_reject_eval_access_counter(self):
        original=auditor.load
        def corrupt(path):
            value=original(path)
            if Path(path).name=="stage22a_status.json":
                value["official_final_eval_forward"]=1
            return value
        with patch.object(auditor,"load",corrupt),self.assertRaises(RuntimeError):
            auditor.review(self.root)


if __name__=="__main__":unittest.main()
