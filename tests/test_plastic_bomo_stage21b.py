"""Stdlib CPU tests for the entry audit, not detector/GPU qualification."""
from collections import Counter
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage21b", REPO / "tools/prepare_plastic_bomo_stage21b.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
CFG = M.load(REPO / "configs/plastic_bomo_stage21b.json")


def fixture_banks():
    reals, rows = [], []
    for i in range(80):
        group = "dev" if i == 0 else "train"
        reals.append({"node_id": f"node{i}", "source_group_id": group, "image_sha256": f"image{i}",
                      "internal_split": "internal_dev" if group == "dev" else "internal_train",
                      "image_relative_path": f"images/train/{i}.jpg"})
        normal_group = "dev" if i == 1 else "train"
        slot = {"slot_id": f"slot{i}", "node_id": f"node{i}", "source_group_id": group,
                "source_sha256": f"image{i}", "split": "train", "conditioning_normal_id": "normal_" + normal_group,
                "normal_sha256": "normalhash_" + normal_group, "normal_source_group": normal_group,
                "normal_path": "normal/" + normal_group + ".jpg"}
        rows.append({"slot_id": f"slot{i}", "bank": "A", "class": "flash" if i < 40 else "black",
            "role_id": f"extra_slot{i}", "frozen_slot": slot, "non_rng_metadata_sha256": "metadata",
            "checkpoint_tree_sha256": "checkpoint", "label_sha256": f"label{i}", "quality_selection": False,
            "formal_synthetic": True, "image_path": f"synthetic_A/{i}.png", "image_sha256": f"syn{i}",
            "label_path": f"synthetic_A/{i}.txt", "sample_seed": i})
    banks = {}
    for bank in M.BANKS:
        bank_rows = deepcopy(rows)
        for row in bank_rows:
            row["bank"] = bank
        banks[bank] = {"status": "COMPLETE", "count": 80, "quality_selection": False, "reroll_count": 0,
                       "records": bank_rows}
    normals = [{"normal_id": "normal_" + g, "sha256": "normalhash_" + g, "known_source_group": g,
                "node_id": "normal_node_" + g, "split": "train", "confirmatory_allowed": True} for g in ("train", "dev")]
    groups = [{"source_group_id": g, "nodes": ["normal_node_" + g] + [r["node_id"] for r in reals if r["source_group_id"] == g]}
              for g in ("train", "dev")]
    return banks, reals, {"train"}, {"dev"}, normals, groups


class Stage21BTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u, cls.c = M.schedules([f"base_{i}" for i in range(109)], [f"extra_{i}" for i in range(65)], CFG)

    def test_fixed_config_rejects_search(self):
        for key, value in (("split_seed", 2102), ("late_start_epoch_0based", 40), ("train_group_fraction", 0.9)):
            cfg = deepcopy(CFG); cfg[key] = value
            with self.assertRaises(M.AuditStop): M.validate_config(cfg)

    def test_group_split_permutation_invariant(self):
        rows = [{"node_id": str(i), "known_source_group_id": f"g{i // 2}"} for i in range(20)]
        a = M.group_split(rows, CFG)
        b = M.group_split(rows[::-1], CFG)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 8)
        self.assertEqual(len(a[1]), 2)
        self.assertFalse(a[0] & a[1])

    def test_duplicate_node_rejected(self):
        with self.assertRaises(M.AuditStop):
            M.group_split([{"node_id": "x", "known_source_group_id": "g"}] * 2, CFG)

    def test_all_occurrences_and_counts_exact(self):
        _, equality = M.schedule_audit(self.u, self.c, CFG)
        self.assertEqual(equality["uniform"], equality["curriculum"])
        self.assertEqual(equality["uniform"]["total_draws"], 26100)
        self.assertEqual(equality["uniform"]["total_real_draws"], 16350)
        self.assertEqual(equality["uniform"]["total_synthetic_draws"], 9750)
        self.assertEqual(set(equality["uniform"]["per_role_counts"].values()), {150})

    def test_curriculum_early_synthetic_zero(self):
        roles = self.c["roles"]
        self.assertTrue(all(roles[i].startswith("base_") for e in self.c["epoch_draws"][:50] for i, _ in e))
        self.assertEqual(sum(roles[i].startswith("extra_") for e in self.c["epoch_draws"][50:] for i, _ in e), 9750)

    def test_uniform_each_sample_once_per_epoch(self):
        for epoch, draws in enumerate(self.u["epoch_draws"]):
            self.assertEqual({i for i, _ in draws}, set(range(174)))
            self.assertTrue(all(k == epoch for _, k in draws))

    def test_fixed_compute_groups(self):
        for mode in (self.u, self.c):
            self.assertEqual(mode["draws_per_epoch"], 174)
            self.assertEqual(mode["accumulation_group_lengths_per_epoch"], [64, 64, 46])
            self.assertEqual(mode["scheduled_optimizer_attempts"], 450)
            self.assertFalse(mode["execution_authorized"])

    def test_occurrence_seed_independent_of_position(self):
        def seeds(schedule):
            return {(i, k): M.occurrence_seed(schedule["roles"][i], k, CFG) for e in schedule["epoch_draws"] for i, k in e}
        self.assertEqual(seeds(self.u), seeds(self.c))
        self.assertNotEqual(self.u["sequence_sha256"], self.c["sequence_sha256"])
        self.assertNotEqual(M.occurrence_seed("base_0", 0, CFG), M.occurrence_seed("base_0", 1, CFG))

    def test_occurrence_tampering_rejected(self):
        c = deepcopy(self.c); c["epoch_draws"][0][0] = c["epoch_draws"][0][1]
        with self.assertRaises(M.AuditStop): M.schedule_audit(self.u, c, CFG)

    def test_infeasible_late_schedule_not_rescued(self):
        with self.assertRaises(M.AuditStop): M.schedules(["base_x"], ["extra_a", "extra_b", "extra_c"], CFG)

    def test_balanced_merge_keeps_inputs(self):
        merged = M.balanced_merge(list(range(9)), list(range(10, 16)))
        self.assertEqual(Counter(merged), Counter(list(range(9)) + list(range(10, 16))))
        self.assertEqual(M.balanced_merge([], []), [])

    def test_donor_exclusion_same_banks_not_quality(self):
        result = M.synthetic_audit(*fixture_banks())
        self.assertEqual(len(result["records"]), 79)
        self.assertEqual(result["donor_dev_excluded"][0]["excluded_in_banks"], ["A", "B", "C"])
        self.assertFalse(result["quality_selection"])
        self.assertTrue(result["all_donor_eligible_slots_retained"])

    def test_conditioning_conflict_reported_not_silently_filtered(self):
        result = M.synthetic_audit(*fixture_banks())
        self.assertEqual([r["slot_id"] for r in result["conditioning_dev_overlaps"]], ["slot1"])
        self.assertIn("slot1", {r["slot_id"] for r in result["records"]})
        self.assertFalse(result["conditioning_dev_overlaps"][0]["additional_exclusion_applied"])

    def test_bank_inconsistency_rejected(self):
        args = fixture_banks(); args[0]["C"]["records"][2]["label_sha256"] = "changed"
        with self.assertRaises(M.AuditStop): M.synthetic_audit(*args)

    def test_path_traversal_and_asset_corruption_rejected(self):
        for path in ("../official.jpg", "/absolute", "C:/escape", "bad\\path"):
            with self.assertRaises(M.AuditStop): M.safe_relative(REPO, path)
        allow = {"records": [{"slot_id": "s", "banks": {b: {"image_path": "x", "image_sha256": "wrong"} for b in M.BANKS}}]}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(M.AuditStop): M.verify_assets(allow, folder, [])

    def test_verify_real_never_opens_official(self):
        with patch.object(M, "sha", side_effect=AssertionError("content was opened")):
            with self.assertRaises(M.AuditStop):
                M.verify_assets({"records": []}, REPO, [{"split_v2": "final_eval"}], real_root=REPO)

    def test_unknown_metrics_not_zero_or_pass(self):
        status = M.status_record("STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP")
        self.assertEqual(status["Gate_A"], "NOT_RUN")
        self.assertIsNone(status["mean_delta_pp"])
        self.assertFalse(status["STAGE21B_EXECUTION_AUTHORIZED"])
        self.assertFalse(status["full_experiment_complete"])
        for field in ("final_eval_access_count", "generation_count", "detector_training_count", "internal_dev_forward", "BootstrapGuard"):
            self.assertEqual(status[field], 0)

    def test_no_model_or_training_imports(self):
        tree = ast.parse((REPO / "tools/prepare_plastic_bomo_stage21b.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): names.update(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom): names.add((node.module or "").split(".")[0])
        self.assertFalse(names & {"torch", "ultralytics", "diffusers", "PIL", "numpy"})

    def test_existing_real_split_counts_and_background_evidence(self):
        root = REPO / "results_for_gpt/plastic_bomo_stage20a_v2_protocol"
        registry = M.load(root / "real_source_registry.json")["records"]
        real = [r for r in registry if r["asset_role"] == "detector" and r["split_v2"] == "train"]
        train, dev, _ = M.group_split(real, CFG)
        self.assertEqual((len(train), len(dev)), (107, 27))
        rows = M.real_records(registry, M.load(root / "annotation_provenance_v2.json")["records"], train)
        banks = {b: M.load(REPO / f"results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_{b}_manifest.json") for b in M.BANKS}
        result = M.synthetic_audit(banks, rows, train, dev, M.load(root / "normal_source_registry.json")["records"],
                                   M.load(root / "source_group_registry.json")["records"])
        self.assertEqual(result["eligible_per_class"], {"flash": 34, "black": 31})
        self.assertEqual(Counter(r["class"] for r in result["conditioning_dev_overlaps"]), {"flash": 2, "black": 2})

    def test_frozen_output_ledger_and_no_fabricated_metrics(self):
        root = REPO / "results_for_gpt/plastic_bomo_stage21b_timing_premise"
        manifest = M.load(root / "stage21b_artifact_manifest.json")
        self.assertFalse(manifest["experiment_complete"])
        self.assertTrue(all(M.sha(root / n) == h for n, h in manifest["sha256"].items()))
        self.assertFalse(list(root.glob("*_metrics.json")))
        assets = M.load(root / "source_asset_hash_audit.json")
        self.assertEqual(assets["synthetic_assets_verified"], 390)
        self.assertEqual(assets["real_TRAIN_assets_verified"], 0)
        self.assertEqual(assets["official_final_eval_content_read"], 0)

    def test_authority_ledger_tampering_rejected(self):
        with patch.object(M, "sha", return_value="modified"), patch.object(M, "load", side_effect=AssertionError("loaded after failed binding")):
            with self.assertRaises(M.AuditStop):
                M.bound_metadata(REPO, "ledger.json", "expected", ("metadata.json",), [])


if __name__ == "__main__":
    unittest.main()
