"""CPU metadata tests; do not import torch or execute any experiment."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage21a", REPO / "tools/synthesize_plastic_bomo_stage21a.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage21ATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = MODULE.build(REPO)

    def test_required_outputs(self):
        required = {"STAGE21A_REPORT.md", "generator_evidence_timeline.json", "supported_findings.json",
                    "stage20b_failure_decomposition.json", "rng_instability_interpretation.json",
                    "class_asymmetry_interpretation.json", "innovation_route_options.json",
                    "innovation1_reframing_candidates.json", "bootstrapguard_separation.json", "stage21a_status.json"}
        self.assertTrue(required <= self.docs.keys())

    def test_authoritative_stop_and_no_experiment(self):
        x = self.docs["stage21a_status.json"]
        self.assertEqual(x["status"], "GENERATOR_ROUTE_NOT_JUSTIFIED_CURRENTLY")
        self.assertEqual(x["Stage20B_formal_status_preserved"], "GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED")
        for key in ("TDCRG_DEVELOPMENT_AUTHORIZED", "GENERATION_AUTHORIZED", "DETECTOR_TRAINING_AUTHORIZED",
                    "NEW_TRAINING_MECHANISM_IMPLEMENTATION_AUTHORIZED", "STAGE21B_EXECUTION_AUTHORIZED", "stage21b_auto_start"):
            self.assertIs(x[key], False)
        self.assertTrue(all(v == 0 for v in x["activity_counts"].values()))

    def test_metrics_independently_expected(self):
        x = self.docs["stage20b_failure_decomposition.json"]
        self.assertAlmostEqual(x["mean_delta_pp"], -2.5901439871478784)
        self.assertAlmostEqual(x["sample_sd_delta_pp"], 3.8278866892745036)
        self.assertEqual(x["wins"], 1)
        self.assertFalse(any(v["pass"] for v in x["gates"].values()))
        self.assertAlmostEqual(sum(x["class_contribution_to_macro_mean_pp"].values()), x["mean_delta_pp"])

    def test_manual_c_restart_not_resume(self):
        x = self.docs["stage20b_failure_decomposition.json"]["manual_C_recovery_boundary"]
        self.assertEqual(x["source_stopped_run"], "formal_20261009_060552")
        self.assertEqual(x["authoritative_recovered_run"], "manual_C_20261010_032543")
        self.assertIs(x["epoch77_resume"], False)
        self.assertIs(x["external_SIGKILL_cause_independently_verified"], False)
        self.assertEqual(x["extra_interrupted_C_physical_compute"]["epochs"], 76)
        self.assertIs(x["extra_interrupted_compute_in_matched_primary"], False)
        self.assertTrue(x["interrupted_counters_may_be_lower_bounds"])

    def test_rng_observation_not_population_causal_estimate(self):
        x = self.docs["rng_instability_interpretation.json"]
        self.assertAlmostEqual(x["range_pp"], 7.638329557469875)
        self.assertTrue(all(x["shared_controls"].values()))
        self.assertIs(x["causal_population_effect_size_estimated"], False)
        self.assertEqual(x["n_detector_seeds"], 1)
        self.assertEqual(x["bank_C_confirmatory_status"], "ONE_POSITIVE_GENERATION_REALIZATION")

    def test_class_asymmetry_not_generator_blame(self):
        x = self.docs["class_asymmetry_interpretation.json"]
        self.assertEqual(x["status"], "CLASS_ASYMMETRIC_UTILITY_FAILURE")
        self.assertEqual(x["black_negative_banks"], 3)
        self.assertIs(x["Black_generator_quality_bad_proven"], False)
        self.assertIs(x["Black_morphology_wrong_proven"], False)
        self.assertIs(x["Flash_no_harm_equivalence_proven"], False)

    def test_timeline_provenance_and_types(self):
        x = self.docs["generator_evidence_timeline.json"]
        self.assertEqual(tuple(row["stage"] for row in x["entries"]), MODULE.REQUIRED_STAGES)
        required = {"question", "protocol", "result", "what_is_supported", "what_is_not_supported",
                    "whether_confirmatory", "whether_validation_used", "whether_selector_involved", "sources"}
        for row in x["entries"]:
            self.assertTrue(required <= row.keys())
            self.assertTrue(row["sources"])
            for key in ("whether_confirmatory", "whether_validation_used", "whether_selector_involved"):
                self.assertIsInstance(row[key], bool)
        s16 = next(row for row in x["entries"] if row["stage"] == "Stage16A")
        self.assertIs(s16["result"]["primary_server_status_blob_available_in_pinned_commit"], False)
        s17 = next(row for row in x["entries"] if row["stage"] == "Stage17A")
        self.assertEqual(s17["result"]["status"], "SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE")

    def test_no_method_authorization_or_auto_family(self):
        x = self.docs["innovation_route_options.json"]
        self.assertEqual(x["preferred_option"], "C")
        self.assertEqual(x["options"][0]["status"], "OPTION_A_CURRENTLY_NOT_AUTHORIZED")
        self.assertEqual(x["options"][1]["status"], "OPTION_B_NOT_RECOMMENDED_AS_IMMEDIATE_NEXT_STEP")
        self.assertIsNone(x["options"][1]["automatic_family_recommendation"])
        self.assertTrue(all(v["implementation_authorized"] is False for v in x["options"]))
        c = self.docs["innovation1_reframing_candidates.json"]
        self.assertEqual(len(c["candidates"]), 5)
        self.assertTrue(all(v["effectiveness_supported"] is False for v in c["candidates"]))
        self.assertIs(c["new_experimental_protocol_designed"], False)

    def test_evidence_levels_and_bg_separation(self):
        x = self.docs["supported_findings.json"]
        self.assertGreaterEqual(len(x["SUPPORTED"]), 5)
        self.assertGreaterEqual(len(x["DIAGNOSTIC_ONLY"]), 4)
        self.assertGreaterEqual(len(x["NOT_SUPPORTED"]), 5)
        self.assertIs(x["cross_protocol_pooling_performed"], False)
        self.assertIs(x["current_final_eval_reusable_as_independent_confirmation"], False)
        bg = self.docs["bootstrapguard_separation.json"]
        self.assertNotEqual(bg["innovation1_question"], bg["innovation2_question"])
        self.assertIs(bg["independent_V2_BootstrapGuard_utility_confirmed"], False)

    def test_sources_hashes_and_all_checks_pass(self):
        index = self.docs["source_evidence_index.json"]
        for row in index["sources"]:
            self.assertEqual(row["revision"], MODULE.REVISION)
            self.assertEqual(len(row["sha256"]), 64)
        self.assertTrue(all(x["passed"] for x in self.docs["evidence_verification.json"]["checks"]))

    def test_strict_json_serialization(self):
        for key, value in self.docs.items():
            if key.endswith(".json"):
                self.assertEqual(json.loads(MODULE.encoded(value)), value)

    def test_stdlib_only_no_runtime_imports(self):
        tree = ast.parse((REPO / "tools/synthesize_plastic_bomo_stage21a.py").read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name.split(".")[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split(".")[0])
        self.assertTrue(set(imports) <= {"__future__", "argparse", "hashlib", "json", "math", "pathlib", "statistics", "subprocess"})


if __name__ == "__main__":
    unittest.main()
