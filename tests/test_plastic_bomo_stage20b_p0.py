import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("p0",ROOT/"tools/preflight_plastic_bomo_stage20b_p0.py")
p0=importlib.util.module_from_spec(spec);spec.loader.exec_module(p0)
CFG=p0.load(ROOT/"configs/plastic_bomo_stage20b_p0.json")


class P0Tests(unittest.TestCase):
    def test_missing_manifest_reports_resolved_path(self):
        with tempfile.TemporaryDirectory() as t:
            x=p0.inspect_frozen_manifest(Path(t),CFG["stage20a_frozen_manifest_sha256"])
            self.assertFalse(x["exists"])
            self.assertFalse(x["hash_match"])
            self.assertIsNone(x["actual_sha256"])
            self.assertTrue(Path(x["manifest_path"]).is_absolute())

    def test_crlf_change_is_identified_but_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            raw=b'{\n  "test": 1\n}\n';expected=p0.hashlib.sha256(raw).hexdigest()
            (Path(t)/"frozen_artifact_manifest.json").write_bytes(raw.replace(b"\n",b"\r\n"))
            x=p0.inspect_frozen_manifest(Path(t),expected)
            self.assertFalse(x["hash_match"])
            self.assertTrue(x["matches_if_CRLF_converted_to_LF_DIAGNOSTIC_ONLY"])
            self.assertFalse(x["byte_normalization_accepted"])

    def test_nested_upload_is_reported_without_path_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);nested=root/"duplicate_folder";nested.mkdir()
            (nested/"frozen_artifact_manifest.json").write_bytes(b"{}")
            x=p0.inspect_frozen_manifest(root,p0.hashlib.sha256(b"{}").hexdigest())
            self.assertFalse(x["hash_match"])
            self.assertFalse(x["automatic_path_fallback"])
            self.assertEqual(len(x["nearby_manifest_candidates_DIAGNOSTIC_ONLY"]),1)

    def test_unsupported_detector_arg_cannot_be_silently_dropped(self):
        with self.assertRaises(p0.Stop) as e:p0.resolve_detector_arguments({"batch":16},{"batch":1,"cutmix":0.0})
        self.assertEqual(e.exception.status,"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN")

    def test_explicit_detector_args_override_remaining_frozen_defaults(self):
        x,keys=p0.resolve_detector_arguments({"batch":16,"lr0":.01,"unused":False},{"batch":1,"lr0":.001,"max_grad_norm":10.0})
        self.assertEqual(x,{"batch":1,"lr0":.001,"unused":False})
        self.assertEqual(keys,["batch","lr0"])
    def test_changed_authoritative_manifest_stops_before_assets(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"frozen_artifact_manifest.json").write_text("{}",encoding="utf-8")
            args=SimpleNamespace(repo_root=ROOT,stage20a=root)
            with self.assertRaises(p0.Stop) as e:p0.run(args,CFG,root)
            self.assertEqual(e.exception.status,"STAGE20A_PROTOCOL_BINDING_FAILED")
    def test_windows_asset_names_are_portable_to_linux(self):
        self.assertEqual(p0.portable_name(r"C:\dataset\empty_images\empty_img_1_pre_part_0.jpg"),"empty_img_1_pre_part_0.jpg")
        self.assertEqual(p0.portable_name("/mnt/data/foo.jpg"),"foo.jpg")
    def test_crop_translation_and_inverse(self):
        box={"bbox":[.4,.6,.02,.02]}
        image={"width":4096,"height":1024}
        x=p0.geometry(box,image,CFG["geometry"])
        self.assertTrue(x["valid"])
        self.assertGreater(x["nonzero_mask_pixels"],0)
        for i in range(4):
            self.assertAlmostEqual(x["generator_bbox_xyxy"][i]+x["fixed_crop_xyxy"][i%2],x["original_bbox_xyxy"][i])
        self.assertFalse(x["mask_is_human_segmentation"])

    def test_truncation_does_not_move_crop(self):
        x=p0.geometry({"bbox":[.5,.5,.5,.5]},{"width":4096,"height":1024},CFG["geometry"])
        self.assertFalse(x["valid"])
        self.assertEqual(x["reason"],"TARGET_CROPPED_OR_INVALID")

    def test_edge_target_uses_padding_not_crop_rescue(self):
        x=p0.geometry({"bbox":[.01,.5,.01,.02]},{"width":4096,"height":1024},CFG["geometry"])
        self.assertTrue(x["valid"])
        self.assertLess(x["fixed_crop_xyxy"][0],0)
        self.assertFalse(x["crop_shifted_to_rescue"])

    def test_group_round_robin_and_unique_annotations(self):
        rows=[];by={}
        for i in range(4):
            by[str(i)]={"width":4096,"height":1024}
            for j in range(3):rows.append({"class":"flash","source_group_id":str(i),"annotation_id":f"ann{i}_{j}","node_id":str(i),"bbox":[.5,.5,.02,.02]})
        chosen,_=p0.choose_donors(rows,by,CFG,{"flash":6})
        again,_=p0.choose_donors(rows[::-1],by,CFG,{"flash":6})
        self.assertEqual(chosen,again)
        self.assertEqual(len({x[0]["source_group_id"] for x in chosen[:4]}),4)
        self.assertEqual(len({x[0]["annotation_id"] for x in chosen}),6)

    def test_insufficient_structural_slots_stops(self):
        with self.assertRaises(p0.Stop) as e:p0.choose_donors([],{},CFG,{"flash":40})
        self.assertEqual(e.exception.status,"INSUFFICIENT_STRUCTURALLY_VALID_SLOTS")

    def test_empty_base_is_not_public_initialization(self):
        with tempfile.TemporaryDirectory() as t:x=p0.model_binding(Path(t),CFG["base_public_reference"])
        self.assertEqual(x["status"],"SD2_BASE_WEIGHT_IDENTITY_UNRESOLVED")
        self.assertTrue(x["errors"])

    def test_frozen_local_outputs(self):
        out=ROOT/"results_for_gpt/plastic_bomo_stage20b_p0_preflight"
        if not (out/"stage20b_p0_status.json").is_file():self.skipTest("local preflight pending")
        slots=p0.load(out/"stage20b_generation_slots.json")["slots"]
        self.assertEqual(len(slots),80)
        self.assertEqual(len({x["annotation_id"] for x in slots}),80)
        self.assertTrue(all(x["split"]=="train" and x["geometry"]["valid"] for x in slots))
        for cls in ("flash","black"):
            chosen=[x for x in slots if x["class"]==cls]
            self.assertEqual(len(chosen),40)
            self.assertEqual(len({x["source_group_id"] for x in chosen}),40)
        rng=p0.load(out/"stage20b_rng_manifest.json")["records"]
        for r in rng:
            self.assertEqual(len({x["non_rng_metadata_sha256"] for x in r["banks"].values()}),1)
            self.assertEqual(len({x["sample_seed"] for x in r["banks"].values()}),3)
        order=p0.load(out/"stage20b_training_order_spec.json")
        self.assertEqual(order["scheduled_draws"],32400)
        self.assertEqual(order["scheduled_optimizer_attempts"],600)
        for e in order["epochs"]:
            self.assertEqual(len(set(e["role_sequence"])),216)
            self.assertEqual(sum(e["accumulation_group_lengths"]),216)
        status=p0.load(out/"stage20b_p0_status.json")
        self.assertFalse(status["STAGE20B_EXECUTION_AUTHORIZED"])
        for k in ("generator_training_count","synthetic_generation_count","detector_training_count","official_final_eval_forward","TDCRG_implementation"):
            self.assertEqual(status[k],0)
        self.assertFalse(status["TDCRG_DEVELOPMENT_AUTHORIZED"])


if __name__=="__main__":unittest.main()
