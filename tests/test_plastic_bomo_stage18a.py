import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("stage18a", Path(__file__).parents[1] / "tools/audit_plastic_bomo_stage18a.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Stage18AAuditTests(unittest.TestCase):
    def test_bbox_rejects_nonfinite_outside_and_zero(self):
        for b in ([.5, .5, 0, .1], [.99, .5, .1, .1], [.5, float("nan"), .1, .1], [True, .5, .1, .1]):
            self.assertFalse(audit.bbox_valid(b))
        self.assertTrue(audit.bbox_valid([.5, .5, .1, .1]))

    def test_missing_assets_never_verified(self):
        with tempfile.TemporaryDirectory() as d:
            r = {"candidate_id": "x", "class_id": 0, "class_name": "01_Flash_point",
                 "image_path": "/missing/x.png", "label_path": "/missing/x.txt", "bbox": [.5, .5, .1, .1]}
            a = audit.audit_annotations([r], Path(d))
            self.assertEqual(a["legal_class_counts"], {"flash": 0, "black": 0})
            self.assertEqual(a["status"], "DOWNSTREAM_UTILITY_NOT_TESTABLE")

    def test_actual_label_corruption_class_and_duplicates(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); Image.new("RGB", (16, 16)).save(root / "x.png")
            (root / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            r = {"class_id": 0, "image_path": "x.png", "label_path": "x.txt", "bbox": [.5, .5, .1, .1]}
            self.assertEqual(audit.audit_annotations([r], root)["status"], "PASS")
            a = audit.audit_annotations([r, r], root)
            self.assertEqual(a["legal_class_counts"]["flash"], 1)
            self.assertIn("DUPLICATE_IMAGE_OR_LABEL_PATH", a["records"][1]["errors"])
            (root / "x.txt").write_text("1 0.5 0.5 0.1 0.1\n")
            self.assertFalse(audit.audit_annotations([r], root)["records"][0]["legal"])
            (root / "x.txt").write_text("")
            self.assertIn("EMPTY_LABEL", audit.audit_annotations([r], root)["records"][0]["errors"])

    def test_inventory_excludes_validation_and_deeppcb(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("val", "test", "train", "deeppcb_model", "outputs"):
                (root / name).mkdir(); (root / name / "a.json").write_text("{}")
            i = audit.inventory(root)
            self.assertEqual([x["relative"] for x in i["files"]], ["outputs/a.json"])
            self.assertEqual(i["validation_access"], 0)

    def test_selected_is_not_raw_and_unknown_fails_closed(self):
        self.assertEqual(audit.selector_classification("high_fidelity"), "SELECTED_SYNTHETIC")
        self.assertEqual(audit.selector_classification("M10"), "SELECTED_SYNTHETIC")
        self.assertEqual(audit.selector_classification("M11"), "SELECTED_SYNTHETIC")
        self.assertEqual(audit.selector_classification("high"), "SELECTED_SYNTHETIC")
        self.assertEqual(audit.selector_classification("M00"), "RANDOM_SYNTHETIC")
        self.assertEqual(audit.selector_classification("random"), "RANDOM_SYNTHETIC")
        self.assertEqual(audit.selector_classification("unknown"), "UNKNOWN_SELECTION")

    def test_protocol_forbids_experiments(self):
        c = json.loads((Path(__file__).parents[1] / "configs/plastic_bomo_stage18a.json").read_text())
        self.assertIn("official_validation_access", c["forbidden"])
        self.assertEqual(c["required_synthetic_class_counts"], {"flash": 40, "black": 40})


if __name__ == "__main__":
    unittest.main()
