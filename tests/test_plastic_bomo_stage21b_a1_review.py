"""Independent arithmetic and read-only downloaded-review contracts."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import review_plastic_bomo_stage21b_a1 as R


def fixtures():
    return {b + "_" + mode: {"map5095": v, "per_class": {c: {"ap5095": v} for c in ("flash", "black")}}
            for b, u, v in (("A", .3, .32), ("B", .32, .33), ("C", .34, .34))
            for mode, v in (("uniform", u), ("curriculum", v))}


class ReviewTest(unittest.TestCase):
    def test_gate_pass_and_tie_not_a_win(self):
        result = R.independent_results(fixtures())
        self.assertEqual(result["strict_bank_wins"], 2)
        self.assertEqual(result["status"], "SYNTHETIC_EXPOSURE_TIMING_PREMISE_SUPPORTED")

    def test_failure_not_rescued_by_spread(self):
        x = fixtures()
        for arm, row in x.items():
            if arm.endswith("curriculum"):
                row["map5095"] = .29
                row["per_class"]["black"]["ap5095"] = .25
        result = R.independent_results(x)
        self.assertFalse(result["Gate_A"])
        self.assertFalse(result["Gate_B"])
        self.assertFalse(result["Gate_C"])
        self.assertTrue(result["Gate_D"])
        self.assertEqual(result["status"], "SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED")

    def test_nonfinite_and_missing_arms_rejected(self):
        x = fixtures(); x["A_uniform"]["map5095"] = float("nan")
        with self.assertRaises(R.C.AuditStop): R.independent_results(x)
        x = fixtures(); del x["A_uniform"]
        with self.assertRaises(R.C.AuditStop): R.independent_results(x)

    def test_metadata_mapping_rejects_traversal_and_non_metadata(self):
        p = R.metadata_local_path("/mnt/sda/cbj/ImageGeneration/Self-main/results_for_gpt/x/y.json")
        self.assertEqual(p, REPO / "results_for_gpt/x/y.json")
        for p in ("/mnt/sda/dataset/image.jpg", "results_for_gpt/../../escape.json"):
            with self.assertRaises(R.C.AuditStop): R.metadata_local_path(p)

    def test_ledger_tampering_rejected_without_rewrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); f = root / "item.json"; f.write_text('{}', encoding="utf-8")
            doc = {"sha256": {"item.json": R.C.sha(f)}}
            (root / "ledger.json").write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(R.verify_ledger(root, "ledger.json"), doc)
            f.write_text('{"modified": true}', encoding="utf-8")
            before = f.read_bytes()
            with self.assertRaises(R.C.AuditStop): R.verify_ledger(root, "ledger.json")
            self.assertEqual(f.read_bytes(), before)

    def test_review_function_never_rewrites_or_runs_model(self):
        tree = ast.parse((REPO / "tools/review_plastic_bomo_stage21b_a1.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "review")
        attrs = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(attrs & {"write_text", "write_bytes", "save", "train", "val", "predict"})


if __name__ == "__main__":
    unittest.main()
