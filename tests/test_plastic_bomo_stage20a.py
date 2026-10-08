import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage20a", ROOT / "tools/bootstrap_plastic_bomo_stage20a.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class Stage20ATest(unittest.TestCase):
    def test_split_is_order_independent(self):
        p = json.loads((ROOT / "data_protocols/plastic_bomo_generation_v2/bootstrap_spec.json").read_text())
        ids = ["sg_" + str(i) for i in range(100)]
        a = mod.stable_split(ids, p)
        self.assertEqual(a, mod.stable_split(ids[::-1], p))
        self.assertEqual(sum(v == "train" for v in a.values()), 75)

    def test_graph_known_relations_only(self):
        def rec(n, rgb, key=None):
            return dict(node_id=n, canonical_image_id=rgb, known_filename_source_key=key,
                        source_xml_metadata=None, possible_aliases=[], possible_sibling_crops=[])
        rows = [rec("a", "rgb1", "img_10"), rec("b", "rgb2", "img_10"),
                rec("c", "rgb1"), rec("unknown1", "rgb3"), rec("unknown2", "rgb4")]
        groups, edges = mod.make_graph(rows, [])
        self.assertEqual(len(groups), 3)
        self.assertEqual(rows[0]["known_source_group_id"], rows[1]["known_source_group_id"])
        self.assertEqual(rows[0]["known_source_group_id"], rows[2]["known_source_group_id"])
        self.assertNotEqual(rows[3]["known_source_group_id"], rows[4]["known_source_group_id"])
        self.assertEqual({e["type"] for e in edges}, {"EXACT_RGB_ALIAS", "KNOWN_SIBLING_CROP"})
        self.assertTrue(all(not g["physical_source_complete"] for g in groups))

    def test_filename_sibling_not_black_number_guess(self):
        self.assertEqual(mod.source_key("empty_img_1013_pre_part_1.jpg"), "img_1013")
        self.assertIsNone(mod.source_key("blackinst_1013.jpg"))

    def test_box_validation(self):
        self.assertTrue(mod.valid_box([.5, .5, .2, .2]))
        self.assertFalse(mod.valid_box([.01, .5, .2, .2]))
        self.assertFalse(mod.valid_box([.5, .5, float("nan"), .2]))

    def test_actual_frozen_outputs(self):
        root = ROOT / "results_for_gpt/plastic_bomo_stage20a_v2_protocol"
        if not (root / "stage20a_status.json").is_file():
            self.skipTest("audit has not completed")
        load = lambda n: json.loads((root / n).read_text(encoding="utf-8"))
        rows = load("real_source_registry.json")["records"]
        by = {r["node_id"]: r for r in rows}
        for e in load("source_group_graph.json")["edges"]:
            self.assertEqual(by[e["from"]]["split_v2"], by[e["to"]]["split_v2"])
        rgb = {}
        for r in rows:
            self.assertEqual(rgb.setdefault(r["decoded_rgb_sha256"], r["split_v2"]), r["split_v2"])
        anns = load("annotation_provenance_v2.json")
        self.assertEqual(sum(x["split_membership_old"] == "train" for x in anns["records"]), 168)
        self.assertEqual(len(anns["one_pixel_issues"]), 2)
        self.assertFalse(any(x["human_verification_status"] == "CONFIRMED" for x in anns["records"]))
        for x in load("normal_source_registry.json")["records"]:
            if x["confirmatory_allowed"]:
                self.assertEqual(x["split"], "train")
                self.assertEqual(x["status"], "NORMAL_SOURCE_RESOLVED")
        status = load("stage20a_status.json")
        for key in ("generator_training_count", "synthetic_generation_count", "detector_training_count",
                    "teacher_training_count", "optimizer_step_count", "official_validation_use_count"):
            self.assertEqual(status[key], 0)
        self.assertFalse(status["TDCRG_DEVELOPMENT_AUTHORIZED"])
        self.assertFalse(status["stage20b_auto_start"])


if __name__ == "__main__":
    unittest.main()
