"""CPU-only contract tests; these synthetic fixtures are not experiment results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_plastic_bomo_stage22a import analyze, protocol_and_manifest, summary, cosine


class Stage22ATest(unittest.TestCase):
    def test_local_frozen_population(self):
        repo = Path(__file__).resolve().parents[1]
        protocol = repo / "configs/plastic_bomo_stage22a.json"
        cfg = json.loads(protocol.read_text(encoding="utf-8"))
        run = repo / cfg["stage21b_run"]
        _, manifest, real, slots, bindings = protocol_and_manifest(protocol, run, verify_files=False)
        self.assertEqual((len(real),len(slots)), (109,61))
        self.assertEqual([len(bindings[x]) for x in ("A","B","C")], [170]*3)
        self.assertEqual(manifest["real_images_by_membership"], {"flash":63,"black":46})
        self.assertTrue(all(len({s["banks"][b]["label_sha256"] for b in ("A","B","C")})==1
            for s in manifest["synthetic_slots"]))

    def test_numeric_helpers(self):
        self.assertEqual(summary(x for x in [1.,2.,3.])["median"], 2.)
        self.assertEqual(cosine(np.array([1.,0.]),np.array([-1.,0.])), -1.)
        self.assertIsNone(cosine(np.array([0.,0.]),np.array([1.,0.])))

    def test_analysis_output_schema_on_tiny_fixture(self):
        cfg = {"selected_parameters":{"backbone_late":"p1","neck":"p2","detect_head":"p3"},
            "feature_layers":{"backbone_late":10,"neck":19},"gradient_reference":"unit real means",
            "feature_roi":"test ROI","historical_stage21b_internal_dev_bank_order_descending":["C","A","B"]}
        raw, vectors, features, slots = [], [], {}, []
        for cls,base in (("flash",[1.,0.,0.,0.]),("black",[0.,1.,0.,0.])):
            for j in range(3):
                raw.append({"role_id":f"real_{cls}_{j}","bank":"real","slot_id":None,
                    "class_memberships":[cls],"geometry":{"annotation_instances":1},
                    "loss":{"box_loss":1.,"cls_loss":2.,"dfl_loss":3.,"total_loss":6.},
                    "gradient_norm_by_group":{"backbone_late":1.,"neck":2.,"detect_head":3.},
                    "gradient_norm_selected_concat":4.})
                v=np.array(base,dtype=np.float32); v[2]=.1*j; vectors.append(v/np.linalg.norm(v))
                for layer in cfg["feature_layers"]:
                    features.setdefault(("real",cls,layer),[]).append({"row_index":len(raw)-1,"vector":v})
            slot=f"slot_{cls}"
            slots.append({"slot_id":slot,"class":cls,
                "banks":{b:{"label_sha256":"same","sample_seed":k} for k,b in enumerate(("A","B","C"))}})
            for k,bank in enumerate(("A","B","C")):
                raw.append({"role_id":f"{slot}_{bank}","bank":bank,"slot_id":slot,
                    "class_memberships":[cls],"geometry":{"annotation_instances":1},
                    "loss":{"box_loss":1.+k,"cls_loss":2.,"dfl_loss":3.,"total_loss":6.+k},
                    "gradient_norm_by_group":{"backbone_late":1.,"neck":2.,"detect_head":3.},
                    "gradient_norm_selected_concat":4.+k})
                v=np.array(base,dtype=np.float32); v[3]=.1*k; vectors.append(v/np.linalg.norm(v))
                for layer in cfg["feature_layers"]:
                    features.setdefault((bank,cls,layer),[]).append({"row_index":len(raw)-1,"vector":v})
        manifest={"real_images_by_membership":{"flash":3,"black":3},"synthetic_slots":slots}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"model_parameter_binding.json").write_text("{}")
            (root/"diagnostic_protocol.json").write_text("{}")
            (root/"diagnostic_data_manifest.json").write_text("{}")
            analyze(cfg,manifest,raw,vectors,features,root,{"selected_parameters":{}},{})
            self.assertEqual(json.loads((root/"stage22a_status.json").read_text())["status"],
                "SYNTHETIC_OPTIMIZATION_MECHANISM_NOT_IDENTIFIED")
            self.assertEqual(len(json.loads((root/"matched_triplet_audit.json").read_text())["records"]),2)


if __name__ == "__main__": unittest.main()
