import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p0_review", ROOT / "tools/review_plastic_bomo_stage20b_p0.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
LOCAL = ROOT / "results_for_gpt/plastic_bomo_stage20b_p0_preflight"
SERVER = LOCAL / "server_preflight_20261008_084115"
STAGE20A = ROOT / "results_for_gpt/plastic_bomo_stage20a_v2_protocol"
PROTOCOL = ROOT / "configs/plastic_bomo_stage20b_p0.json"


class ReviewTests(unittest.TestCase):
    def test_downloaded_successful_server_review(self):
        result = reviewer.review(ROOT, SERVER, LOCAL, STAGE20A, PROTOCOL)
        self.assertEqual(result["status"], "STAGE20B_P0_SERVER_REVIEW_PASS")
        self.assertFalse(result["gpu_forward_or_numerical_determinism_verified"])
        self.assertFalse(result["automatic_formal_execution"])

    def test_changed_bytes_fail_ledger(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "test.json").write_bytes(b"{}")
            expected = reviewer.sha(root / "test.json")
            (root / "ledger.json").write_text(json.dumps({"sha256": {"test.json": expected}}), encoding="utf-8")
            (root / "test.json").write_bytes(b"{ }")
            self.assertFalse(reviewer.ledger(root, "ledger.json")["pass"])

    def test_missing_artifact_fails_ledger(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "ledger.json").write_text(json.dumps({"sha256": {"absent.json": "0" * 64}}), encoding="utf-8")
            result = reviewer.ledger(root, "ledger.json")
            self.assertFalse(result["pass"])
            self.assertIsNone(result["files"][0]["actual_sha256"])

    def test_ledger_cannot_escape_result_directory(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "ledger.json").write_text(json.dumps({"sha256": {"../outside.json": "0" * 64}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LEDGER_PATH_OUTSIDE"):
                reviewer.ledger(root, "ledger.json")

    def test_rehashed_rng_tamper_is_not_semantically_accepted(self):
        with tempfile.TemporaryDirectory(dir=LOCAL) as t:
            server = Path(t) / "tampered_server"
            shutil.copytree(SERVER, server)
            rng_path = server / "stage20b_rng_manifest.json"
            rng = reviewer.load(rng_path)
            rng["records"][0]["banks"]["A"]["sample_seed"] += 1
            rng_path.write_text(json.dumps(rng), encoding="utf-8")
            manifest_path = server / "p0_frozen_artifact_manifest.json"
            manifest = reviewer.load(manifest_path)
            manifest["sha256"][rng_path.name] = reviewer.sha(rng_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = reviewer.review(ROOT, server, LOCAL, STAGE20A, PROTOCOL)
            self.assertTrue(result["checks"]["downloaded_server_files"])
            self.assertFalse(result["checks"]["80_paired_slots_three_predeclared_rng_banks"])
            self.assertEqual(result["status"], "STAGE20B_P0_SERVER_REVIEW_FAILED")


if __name__ == "__main__":
    unittest.main()
