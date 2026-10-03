"""Stage15B fixed-sample, diversity and preregistered decision checks (CPU)."""
import copy
import json
import sys
import subprocess
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from plastic_bomo_stage15b_common import feature_diversity, load, parent, pick, select_balanced, verify_preparation
from finalize_plastic_bomo_stage15b import calculate_effects, decide
from review_plastic_bomo_stage15b import independent_features, inspect_training_csv, review


class Stage15BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load(ROOT/"configs/plastic_bomo_stage15b.json")
        cls.subsets = load(ROOT/"results_for_gpt/plastic_bomo_stage14a_generator_bottleneck/subset_definition.json")["subsets"]
        cls.index = {x["candidate_id"]: x for x in load(ROOT/"results_for_gpt/scored_candidates_v2_fixed_v3_expanded.json")["candidates"]}
        cls.morphology = {x["candidate_id"]: x["morphology_distance"] for x in load(ROOT/"results_for_gpt/plastic_bomo_stage14b_fidelity_confirmation/per_sample_morphology_distance.json")["records"]}

    def test_frozen_population_roles_overlap_and_shared_parent_cap(self):
        rows, cap = select_balanced(self.subsets, self.index, self.morphology, self.cfg)
        self.assertEqual(len({r["candidate_id"] for r in rows}), 80)
        self.assertEqual(Counter((r["class_name"], r["role"]) for r in rows), {(c,role):20 for c in ("flash", "black") for role in ("high_morph", "random")})
        self.assertEqual(cap, 2)
        self.assertLessEqual(max(Counter(r["parent_id"] for r in rows).values()), cap)
        for r in rows:
            self.assertIn(r["candidate_id"], self.subsets["high_fidelity" if r["role"] == "high_morph" else "random"])
            if r["role"] == "random": self.assertNotIn(r["candidate_id"], self.subsets["high_fidelity"])

    def test_determinism_input_order_and_morphology_score_independence(self):
        rows, cap = select_balanced(self.subsets, self.index, self.morphology, self.cfg)
        reversed_lists = {k:list(reversed(v)) for k,v in self.subsets.items()}
        changed_scores = {k:100000-float(v) for k,v in self.morphology.items()}
        second, second_cap = select_balanced(reversed_lists, dict(reversed(list(self.index.items()))), changed_scores, self.cfg)
        self.assertEqual([x["candidate_id"] for x in rows], [x["candidate_id"] for x in second])
        self.assertEqual(cap, second_cap)

    def test_parent_quota_lookahead_resolves_greedy_failure(self):
        h = [self.index[i] for i in self.subsets["high_fidelity"] if int(self.index[i]["class_id"]) == 1]
        r = [self.index[i] for i in self.subsets["random"] if int(self.index[i]["class_id"]) == 1 and i not in self.subsets["high_fidelity"]]
        used = Counter(); pick(h,20,2,used,self.cfg["generation_seeds"])
        capacity = sum(min(n,2-used[k]) for k,n in Counter(parent(x) for x in r).items())
        self.assertEqual(capacity, 19)
        rows, _ = select_balanced(self.subsets,self.index,self.morphology,self.cfg)
        self.assertEqual(sum(x["class_name"]=="black" for x in rows),40)

    def test_rank_and_pairwise_distances(self):
        features = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        result = feature_diversity(features)
        self.assertAlmostEqual(result["mean_pairwise_cosine_distance"],4/3)
        self.assertAlmostEqual(result["median_nearest_neighbor_distance"],1.)
        self.assertAlmostEqual(result["covariance_effective_rank"],2., places=8)
        self.assertAlmostEqual(result["normalized_ER"],1., places=8)
        self.assertTrue(feature_diversity(np.ones((4,2)))["degenerate_covariance"])
        with self.assertRaises(RuntimeError): feature_diversity(np.zeros((4,2)))

    def test_independent_SVD_statistics(self):
        x = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        stats = independent_features(x,1e-12)
        self.assertAlmostEqual(stats['covariance_effective_rank'],2.)
        self.assertAlmostEqual(stats['mean_pairwise_cosine_distance'],4/3)
        with self.assertRaises(RuntimeError): independent_features(np.zeros((4,2)),1e-12)

    def review_args(self, output):
        return (output,self.cfg,ROOT/'results_for_gpt/plastic_bomo_stage14a_generator_bottleneck',
                ROOT/'results_for_gpt/plastic_bomo_stage14b_fidelity_confirmation',
                ROOT/'results_for_gpt/plastic_bomo_stage15a_class_conditional_morphology',
                ROOT/'results_for_gpt/scored_candidates_v2_fixed_v3_expanded.json',
                ROOT/'results/dwbg/v2/fixed_v3/feature_banks')

    def test_downloaded_results_review_and_tampered_summary_rejection(self):
        output = ROOT/'results_for_gpt/plastic_bomo_stage15b_morphology_dose'
        result = review(*self.review_args(output))
        self.assertEqual(result['status'],'DOWNLOADED_RESULTS_REVIEW_PASS')
        self.assertEqual(result['experiment_status'],'MORPHOLOGY_SIGNAL_PRESENT_DOSE_HYPOTHESIS_NOT_CONFIRMED')
        self.assertGreater(len(result['checks']),800)
        with tempfile.TemporaryDirectory(prefix='stage15b_review_test_') as directory:
            copy = Path(directory)/'results'; shutil.copytree(output,copy)
            path = copy/'dose_effects_summary.json'; doc = load(path)
            doc['outcomes']['map50_95']['D50_00']['mean'] += .01
            path.write_text(json.dumps(doc),encoding='utf8')
            with self.assertRaisesRegex(RuntimeError,'every_paired_delta_and_sample_std_recomputed'):
                review(*self.review_args(copy))

    def test_CSV_ragged_final_row_is_reported_not_silently_read_as_AP(self):
        path = ROOT/'results_for_gpt/plastic_bomo_stage15b_morphology_dose/m50_s42_training_results.csv'
        result = inspect_training_csv(path,1)
        self.assertEqual(result['ragged_epochs'],[150])
        self.assertFalse(result['CSV_metric_columns_used_for_final_AP'])
        with self.assertRaisesRegex(RuntimeError,'UNEXPLAINED_RAGGED_ROWS'):
            inspect_training_csv(path,0)
        with tempfile.TemporaryDirectory(prefix='stage15b_csv_test_') as directory:
            changed = Path(directory)/'results.csv'
            lines = path.read_text().splitlines(); lines[1] = ','.join(lines[1].split(',')[:-1])
            changed.write_text('\n'.join(lines),encoding='utf8')
            with self.assertRaisesRegex(RuntimeError,'UNEXPLAINED_RAGGED_ROWS'):
                inspect_training_csv(changed,1)

    def rows(self, values):
        return [{"seed":s,"arm":a,"map50_95":v,"map50":v+.2,"recall":v+.3,
                 "per_class":{c:{"ap50_95":v} for c in ("flash","black")}} for s in (42,3407,2026) for a,v in values.items()]

    def state(self, values):
        by, _, sm = calculate_effects(self.rows(values))
        return decide(sm,by,self.cfg)

    def test_dose_supported_only_all_five_gates_pass(self):
        state = self.state({"M00":.40,"M10":.42,"M01":.41,"M11":.39,"M50":.415})
        self.assertEqual(state["status"],"MORPHOLOGY_DOSE_EFFECT_SUPPORTED")
        self.assertTrue(all(state["gates"].values()))
        self.assertTrue(state["STAGE16_MORPHOLOGY_DISTRIBUTION_CALIBRATED_GENERATOR"])
        self.assertFalse(state["STAGE16_AUTO_START"])

    def test_class_composition_supported_when_balanced_fails(self):
        state = self.state({"M00":.40,"M10":.42,"M01":.41,"M11":.39,"M50":.399})
        self.assertEqual(state["status"],"CLASS_COMPOSITION_INTERACTION_SUPPORTED")
        self.assertTrue(state["STAGE16_TASK_AWARE_GENERATION"])

    def test_inadequate_excess_penalty_does_not_authorize_generator(self):
        state = self.state({"M00":.40,"M10":.42,"M01":.41,"M11":.42,"M50":.415})
        self.assertEqual(state["status"],"MORPHOLOGY_SIGNAL_PRESENT_DOSE_HYPOTHESIS_NOT_CONFIRMED")
        self.assertFalse(state["STAGE16_MORPHOLOGY_DISTRIBUTION_CALIBRATED_GENERATOR"])
        self.assertFalse(state["STAGE16_TASK_AWARE_GENERATION"])

    def test_duplicate_run_metrics_rejected(self):
        rows = self.rows({"M00":.40,"M10":.42,"M01":.41,"M11":.39,"M50":.415})
        with self.assertRaises(RuntimeError): calculate_effects(rows+[copy.deepcopy(rows[0])])

    def test_preparation_cli_dataset_build_and_repeat_preserve_manifest(self):
        # This fixture exercises actual paths, symlinks, labels, freeze checks and
        # dataset YAML on CPU; the byte fixtures are never passed to a detector.
        with tempfile.TemporaryDirectory(prefix="stage15b_prepare_test_") as directory:
            base = Path(directory); fixture_pool = copy.deepcopy(list(self.index.values()))
            required = set(self.subsets["random"]) | set(self.subsets["high_fidelity"])
            for row in fixture_pool:
                if row["candidate_id"] not in required: continue
                cid = row["candidate_id"]; ip = base/"candidates/images"/(cid+".jpg"); lp = base/"candidates/labels"/(cid+".txt")
                ip.parent.mkdir(parents=True,exist_ok=True); lp.parent.mkdir(parents=True,exist_ok=True)
                ip.write_bytes(cid.encode()); lp.write_text(f"{row['class_id']} 0.5 0.5 0.1 0.1\n")
                row.update(image_path=str(ip),label_path=str(lp))
            idx = {x["candidate_id"]:x for x in fixture_pool}
            old = load(ROOT/"results_for_gpt/plastic_bomo_stage15a_class_conditional_morphology/factorial_subset_definition.json")
            for records in old["arms"].values():
                for row in records:
                    row["image_path"] = idx[row["candidate_id"]]["image_path"]; row["label_path"] = idx[row["candidate_id"]]["label_path"]
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value),encoding="utf-8")
            write(base/"pool.json",{"candidates":fixture_pool})
            write(base/"stage14a/subset_definition.json",{"subsets":self.subsets})
            write(base/"stage14b/per_sample_morphology_distance.json",{"records":[{"candidate_id":k,"morphology_distance":v} for k,v in self.morphology.items()]})
            write(base/"stage15a/stage15a_status.json",{"status":"CLASS_INTERACTION_DOMINATES_MORPHOLOGY_SIGNAL"})
            write(base/"stage15a/stage15a_protocol.json",{"detector":self.cfg["detector"]})
            write(base/"stage15a/factorial_subset_definition.json",old)
            for split,n in (("train",138),("val",2)):
                images = base/"real/images"/split; labels = base/"real/labels"/split
                images.mkdir(parents=True); labels.mkdir(parents=True)
                for i in range(n):
                    (images/f"real_{i:03d}.jpg").write_bytes(f"fixture_{i}".encode()); (labels/f"real_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            import yaml
            yaml_path = base/"real/data.yaml"; yaml_path.write_text(yaml.safe_dump({"path":str(base/"real"),"train":"images/train","val":"images/val","names":["flash","black"]}))
            command = [sys.executable,str(ROOT/"tools/prepare_plastic_bomo_stage15b.py"),"--protocol",str(ROOT/"configs/plastic_bomo_stage15b.json"),
                "--stage14a",str(base/"stage14a"),"--stage14b",str(base/"stage14b"),"--stage15a",str(base/"stage15a"),
                "--candidate_pool",str(base/"pool.json"),"--real_data_yaml",str(yaml_path),"--dataset_root",str(base/"built"),"--output",str(base/"out")]
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            frozen = (base/"out/balanced_morph50_manifest.json").read_bytes()
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(frozen,(base/"out/balanced_morph50_manifest.json").read_bytes())
            self.assertEqual(len(list((base/"built/M50/images/train").glob("*.jpg"))),218)
            self.assertEqual(verify_preparation(base/"out",self.cfg)["arm"],"M50")


if __name__ == "__main__": unittest.main()
