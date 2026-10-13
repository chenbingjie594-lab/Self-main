"""Standard-library-only verification of the stopped LOCAL Stage25 P0 record.

No model imports/forwards and no repeated reads of large image/weight assets.
The completed collector's byte-check evidence is separately pinned in JSON.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
REQUIRED = (
    "STAGE25_REPORT.md", "msdf_historical_component_audit.json",
    "msdf_selector_separation_audit.json", "msdf_v2_protocol.json", "msdf_v2_data_binding.json",
    "baseline_reuse_audit.json", "msdf_flash_training_audit.json", "msdf_black_training_audit.json",
    "msdf_checkpoint_binding.json", "msdf_bank_A_manifest.json", "msdf_bank_B_manifest.json",
    "msdf_bank_C_manifest.json", "msdf_generation_validity.json", "detector_A_training_audit.json",
    "detector_B_training_audit.json", "detector_C_training_audit.json", "detector_checkpoint_binding.json",
    "msdf_vs_raw_metrics.json", "msdf_vs_rr_metrics.json", "msdf_classwise.json",
    "msdf_bank_variance.json", "final_eval_access_audit.json", "stage25_status.json",
)


def load(name):
    def reject(value):
        raise ValueError("Non-JSON constant: " + value)
    return json.loads((ROOT / name).read_text(encoding="utf-8"), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Stage25P0Test(unittest.TestCase):
    def test_01_required_files_and_strict_json(self):
        self.assertTrue(all((ROOT / n).is_file() for n in REQUIRED))
        for path in ROOT.glob("*.json"):
            load(path.name)

    def test_02_not_a_scientific_failure(self):
        status = load("stage25_status.json")
        self.assertEqual(status["status"], "MSDF_V2_NOT_QUALIFIABLE")
        self.assertEqual(status["scope"], "CURRENT_LOCAL_P0_EXECUTION_ENVIRONMENT_ONLY")
        self.assertEqual(status["scientific_requalification_result"], "NOT_RUN")
        self.assertFalse(status["scientific_MSDF_failure_proven"])
        self.assertFalse(status["full_experiment_complete"])
        for k in ("MSDF_GENERATOR_IMPROVEMENT_SUPPORTED", "MSDF_AUGMENTATION_UTILITY_CONFIRMED",
                  "msdf_A_map", "msdf_B_map", "msdf_C_map", "mean_msdf_vs_raw_pp"):
            self.assertIsNone(status[k])

    def test_03_all_experiment_counters_zero(self):
        status = load("stage25_status.json")
        for key in ("safety_forward_count", "safety_backward_count", "generator_training_count",
                    "generation_count", "synthetic_generation_count", "detector_training_count",
                    "detector_forward_count", "optimizer_step_count", "official_final_eval_forward",
                    "final_eval_image_or_label_content_read", "BootstrapGuard", "DWBG", "selector",
                    "DeepPCB", "new_module_implementation_count", "MSDF_core_algorithm_change_count"):
            self.assertEqual(status[key], 0, key)
        for key, value in status.items():
            if key.endswith("AUTHORIZED"):
                self.assertIs(value, False, key)

    def test_04_actual_entry_failure(self):
        runtime = load("runtime_entry_audit.json")
        self.assertEqual(runtime["training_entry_probe"]["exit_code"], 1)
        self.assertIn("tqdm", runtime["training_entry_probe"]["error"])
        self.assertTrue(all(not r["import_spec_exists"] for r in runtime["dependencies"].values()))
        self.assertEqual(runtime["single_batch_safety_test"]["status"], "NOT_RUN")

    def test_05_source_freeze_preserved(self):
        evidence = load("p0_asset_hash_verification.json")
        self.assertEqual(len(evidence["pinned_sources"]), 18)
        for row in evidence["pinned_sources"]:
            raw = subprocess.check_output(["git", "-C", str(REPO), "show", row["base_commit"] + ":" + row["path"]])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
            current = REPO / row["path"]
            if row["worktree_exists"]:
                self.assertEqual(sha(current), row["sha256"])

    def test_06_existing_metadata_unchanged(self):
        evidence = load("p0_asset_hash_verification.json")
        for name, digest in evidence["extra_metadata_hashes"].items():
            self.assertEqual(sha(REPO / name), digest)
        for row in evidence["frozen_metadata_verification"].values():
            self.assertEqual(row["files_checked"], row["matched"])

    def test_07_train_identity_not_reshuffled(self):
        data = load("msdf_v2_data_binding.json")
        self.assertEqual(data["TRAIN_image_count"], 136)
        self.assertEqual(data["generator_instances"], {"flash": 75, "black": 97})
        self.assertEqual(data["TRAIN_normal_count"], 394)
        self.assertEqual(data["asset_hash_matches"], 666)
        self.assertEqual(data["known_train_final_eval_source_group_intersection"], [])
        self.assertFalse(data["split_changed"])
        self.assertTrue(data["local_server_normal_semantics_equal"])

    def test_08_baseline_reuse_not_assumed(self):
        reuse = load("baseline_reuse_audit.json")
        self.assertFalse(reuse["formal_baseline_reuse_authorized"])
        self.assertEqual(reuse["actual_V2_MSDF_forward_sampling_parity"], "NOT_RUN")
        for bank in reuse["verified_assets"]["raw_banks"].values():
            self.assertEqual(bank["count"], 80)
            self.assertEqual(bank["slot_and_seed_matches"], 80)
            self.assertEqual(bank["matched_assets"], 160)
        for arm, metric in reuse["historical_metrics_only"].items():
            self.assertEqual(metric["checkpoint_sha256"], reuse["verified_assets"]["raw_detectors"][arm]["actual_sha256"])

    def test_09_targets_and_gates_not_changed(self):
        protocol = load("msdf_v2_protocol.json")
        self.assertEqual(protocol["common_generator_training_target"]["max_train_steps"], 2000)
        self.assertEqual(protocol["common_generator_training_target"]["learning_rate"], 1e-6)
        self.assertEqual(protocol["bank_seeds"], {"A": 42, "B": 3407, "C": 2026})
        self.assertEqual(protocol["primary_gates"]["A_mean_msdf_minus_raw_pp"], {"operator": ">=", "threshold": .50})
        self.assertEqual(protocol["primary_gates"]["B_strict_matched_bank_wins"]["strict_win_operator"], ">")
        for cls in ("flash", "black"):
            self.assertEqual(protocol["primary_gates"]["C_" + cls + "_mean_delta_pp"]["threshold"], -2.)
        self.assertEqual(protocol["detector_exposure"]["total_scheduled_optimizer_attempts"], 600)
        self.assertEqual(protocol["MSDF_core_format_version"], 3)
        self.assertFalse(protocol["current_formal_execution_authorized"])

    def test_10_not_filtered_or_reused_MSDF(self):
        separation = load("msdf_selector_separation_audit.json")
        self.assertFalse(separation["historical_761_pool"]["used_as_stage25_data"])
        self.assertFalse(separation["train_saliency_is_selector"])
        core = ast.parse((REPO / "diffusers/pipelines/stable_diffusion/msdf_guidance.py").read_text(encoding="utf-8"))
        imports = [n.module or "" for n in ast.walk(core) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any("dwbg" in m.lower() or "ultralytics" in m.lower() for m in imports))
        self.assertEqual(load("msdf_historical_component_audit.json")["algorithm_changes"], [])

    def test_11_unit_slot_seed_without_rng_or_model(self):
        slot = "flash_ann_a2c65212fd1710b2950f3073e1605ef317d35df43baf4accc06fee16d1bcbc1d"
        seed = int(hashlib.sha256(("Plastic_Bomo_Generation_V2" + slot + "42").encode()).hexdigest()[:16], 16)
        self.assertEqual(seed, 15756173166268111651)

    def test_12_read_only_collector_utilities(self):
        spec = importlib.util.spec_from_file_location("stage25_p0_collector_test", ROOT / "collect_p0_evidence.py")
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        compact = collector.compact({"assets": [{"match": True}, {"match": False}]})
        self.assertEqual(compact["assets_verification"]["checked"], 2)
        self.assertEqual(compact["assets_verification"]["matched"], 1)
        self.assertEqual(len(compact["assets_verification"]["failed"]), 1)
        self.assertEqual(collector.path_name(r"C:\data\image.png"), "image.png")
        self.assertEqual(collector.canonical({"b": 2, "a": 1}), '{"a":1,"b":2}')

    def test_13_new_artifact_manifest_if_present(self):
        if not (ROOT / "artifact_manifest.json").exists():
            return
        for row in load("artifact_manifest.json")["files"]:
            self.assertEqual(Path(row["path"]).name, row["path"])
            self.assertEqual(sha(ROOT / row["path"]), row["sha256"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Stage25P0Test))
    print(json.dumps({"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
                      "failures": len(result.failures), "errors": len(result.errors),
                      "scope": "READ_ONLY_SCHEMA_SOURCE_AND_NUMERIC_UNIT_TESTS_NOT_GPU_SAFETY",
                      "model_forward_count": 0, "optimizer_step_count": 0}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
