"""CPU contracts; not a substitute for mandatory SERVER single-batch GPU safety."""
from collections import Counter
from copy import deepcopy
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO / "tools"))
import stage21b_a1_common as M
import finalize_plastic_bomo_stage21b_a1 as F


def metric(arm,value=.3,flash=.3,black=.3):
    return {"status":"COMPLETE","arm":arm,"map5095":value,"evaluation_image_count":27,
        "primary_checkpoint":"epoch150 last.pt","dev_population":"DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR",
        "per_class":{"flash":{"ap5095":flash},"black":{"ap5095":black}}}


class Stage21BA1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=M.load(REPO / "configs/plastic_bomo_stage21b_a1.json")
        cls.base=M.load(REPO / "configs/plastic_bomo_stage21b.json")
        cls.original=REPO / "results_for_gpt/plastic_bomo_stage21b_timing_premise"
        cls.split=M.load(cls.original / "stage21b_split_manifest.json")
        cls.dev={r["source_group_id"] for r in cls.split["groups"] if r["split"] == "internal_dev"}
        cls.old=M.load(cls.original / "stage21b_synthetic_allowlist.json")
        cls.allow=M.amended_allowlist(cls.old,cls.dev)
        cls.u,cls.c,cls.pair,cls.eq=M.make_schedules(
            [r["role_id"] for r in cls.split["records"] if r["internal_split"] == "internal_train"],
            [r["role_id"] for r in cls.allow["records"]],cls.base,cls.cfg)
        cls.training=M.load(cls.original / "stage20b_detector_protocol_snapshot.json")["training"]

    def test_original_audit_immutable(self):
        self.assertEqual(M.sha(self.original / "stage21b_artifact_manifest.json"),M.ORIGINAL_LEDGER_SHA)
        ledger=M.load(self.original / "stage21b_artifact_manifest.json")["sha256"]
        self.assertTrue(all(M.sha(self.original / n) == h for n,h in ledger.items()))
        self.assertEqual(M.load(self.original / "stage21b_status.json")["status"],"STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP")

    def test_only_four_authorized_slots_excluded(self):
        removed={x["slot_id"] for x in self.old["records"]}-{x["slot_id"] for x in self.allow["records"]}
        self.assertEqual(removed,M.EXCLUSIONS)
        self.assertEqual(len(self.allow["records"]),61)
        self.assertEqual(Counter(r["class"] for r in self.allow["records"]),{"flash":32,"black":29})
        self.assertTrue(all(x["reason"] == "KNOWN_SOURCE_GROUP_LEAKAGE_EXCLUSION" for x in self.allow["conditioning_dev_excluded"]))
        self.assertFalse(self.allow["quality_selection"])

    def test_both_donor_and_conditioning_isolated(self):
        self.assertTrue(all(r["source_group_id"] not in self.dev and r["conditioning_normal_source_group"] not in self.dev for r in self.allow["records"]))
        self.assertTrue(all(set(r["banks"]) == set(M.BANKS) for r in self.allow["records"]))

    def test_no_extra_conflict_can_be_hidden(self):
        old=deepcopy(self.old);old["conditioning_dev_overlaps"].append({"slot_id":"new"})
        with self.assertRaises(M.AuditStop):M.amended_allowlist(old,self.dev)
        old=deepcopy(self.old);old["records"][0]["class"]="changed"
        # A non-excluded record must affect the exact legal count.
        for r in old["records"]:
            if r["slot_id"] not in M.EXCLUSIONS:r["class"]="changed";break
        with self.assertRaises(M.AuditStop):M.amended_allowlist(old,self.dev)

    def test_schedule_totals_exact(self):
        x=self.eq["uniform"]
        self.assertEqual((x["total_real_draws"],x["total_synthetic_draws"],x["total_draws"],x["optimizer_attempts"]),(16350,9150,25500,450))
        self.assertEqual(self.eq["uniform"],self.eq["curriculum"])
        self.assertEqual(set(x["per_role_counts"].values()),{150})

    def test_exact_even_odd_late_pattern(self):
        for e in range(150):
            ud=self.u["epoch_draws"][e];cd=self.c["epoch_draws"][e]
            self.assertEqual((len(ud),len(cd)),(170,170))
            us=sum(self.u["roles"][i].startswith("extra_") for i,k in ud)
            cs=sum(self.c["roles"][i].startswith("extra_") for i,k in cd)
            self.assertEqual(us,61);self.assertEqual(cs,0 if e < 50 else 91+(e-50)%2)

    def test_cyclic_role_order_no_epoch_quality_shuffle(self):
        def real_order(s,e):return [s["roles"][i] for i,k in s["epoch_draws"][e] if s["roles"][i].startswith("base_")]
        self.assertEqual(real_order(self.u,0),real_order(self.u,149))
        self.assertEqual(len(set(real_order(self.c,0))),109)
        self.assertEqual(real_order(self.c,0)[:61],real_order(self.c,0)[109:])

    def test_seeds_same_for_same_occurrence(self):
        def binding(s):return {(i,k):M.occurrence_seed(s["roles"][i],k,self.base) for e in s["epoch_draws"] for i,k in e}
        self.assertEqual(binding(self.u),binding(self.c))
        self.assertFalse(self.pair["augmentation_seed_depends_on_epoch_or_position"])
        self.assertEqual(len(binding(self.c)),25500)

    def test_controller_reuses_original_loop(self):
        self.assertIs(M.TimingController.run,M.Stage20BDetectorController.run)
        self.assertIs(M.TimingController.backward,M.Stage20BDetectorController.backward)
        ctrl=M.TimingController(self.training,self.c,self.base);normalizers=[];draws=[]
        totals=ctrl.run(lambda e,p,r,s,l:draws.append(l),lambda e,p,l:normalizers.append(l))
        self.assertEqual(totals,{"epochs":150,"draws":25500,"scheduled_optimizer_attempts":450})
        self.assertEqual(normalizers,[64,64,42]*150)
        self.assertEqual(Counter(draws),{64:19200,42:6300})

    def test_tail_normalized_by_actual_42(self):
        seen=[]
        class Loss:
            def sum(self):return self
            def __truediv__(self,n):seen.append(n);return self
            def backward(self):seen.append("backward")
        M.TimingController.backward(Loss(),42)
        self.assertEqual(seen,[42,"backward"])

    def test_lr_same_position_across_schedules(self):
        u=M.TimingController(self.training,self.u,self.base);c=M.TimingController(self.training,self.c,self.base)
        groups=[{"initial_lr":.001,"param_group":"bias"},{"initial_lr":.001,"param_group":"decay"}]
        self.assertEqual(u.learning_rates(0,0,groups),[.1,0])
        for e,p in ((0,0),(2,169),(3,0),(149,169)):self.assertEqual(u.learning_rates(e,p,groups),c.learning_rates(e,p,groups))

    def test_occurrence_mutation_rejected(self):
        c=deepcopy(self.c);c["epoch_draws"][50][0]=c["epoch_draws"][50][1]
        with self.assertRaises(M.AuditStop):M.validate_schedules(self.u,c,self.base,self.cfg)

    def test_cfg_rejects_seed_warmup_or_ratio_search(self):
        for k,v in (("split_seed",2102),("late_start_epoch_0based",20),("late_even_synthetic",92),("total_draws",25000)):
            cfg=deepcopy(self.cfg);cfg[k]=v
            with self.assertRaises(M.AuditStop):M.check_cfg(cfg)

    def test_file_guard_blocks_before_python_read(self):
        with tempfile.TemporaryDirectory() as d:
            official=Path(d)/"official.jpg";dev=Path(d)/"dev.txt";official.write_bytes(b"not read");dev.write_bytes(b"not read")
            g=M.ContentGuard([official],[dev])
            with g.python_guard():
                with self.assertRaises(M.AuditStop):official.read_bytes()
                with self.assertRaises(M.AuditStop):dev.read_text()
            self.assertEqual(g.counts["official_blocked_open_attempts"],1)
            self.assertEqual(g.counts["internal_dev_blocked_open_attempts"],1)
            self.assertEqual(g.counts["official_final_eval_image_read"],0)

    def test_dev_allowed_only_after_barrier_official_always_denied(self):
        g=M.ContentGuard([REPO/"official.jpg"],[REPO/"dev.jpg"],allow_dev=True)
        g.check(REPO/"dev.jpg")
        with self.assertRaises(M.AuditStop):g.check(REPO/"official.jpg")
        with self.assertRaises(M.AuditStop):g.check(REPO/"other.jpg",native=True)
        g.native_images.add(str((REPO/"dev.jpg").resolve()));g.check(REPO/"dev.jpg",native=True)

    def test_legacy_val_train_not_globally_blocked(self):
        g=M.ContentGuard([REPO/"images/val/final.jpg"],[])
        g.check(REPO/"images/val/legal_v2train.jpg")

    def test_all_six_frozen_barrier_required(self):
        a={"status":"COMPLETE","epochs":150,"actual_draws":25500,"attempted_optimizer_updates":450,"last_checkpoint":"last.pt","last_sha256":"h"}
        with patch.object(M,"load",return_value=a),patch.object(M,"sha",return_value="h"):
            self.assertEqual(len(M.checkpoint_barrier(REPO)["arms"]),6)
        with patch.object(M,"load",return_value={**a,"epochs":149}):
            with self.assertRaises(M.AuditStop):M.checkpoint_barrier(REPO)

    def test_four_gate_pass(self):
        metrics={a:metric(a) for a in M.ARMS}
        for b in M.BANKS:metrics[b+"_curriculum"]=metric(b+"_curriculum",.31,.31,.31)
        s,_,_,v=F.calculate(metrics,self.cfg)
        self.assertEqual(s["status"],"SYNTHETIC_EXPOSURE_TIMING_PREMISE_SUPPORTED")
        self.assertTrue(all(s[k] for k in ("Gate_A","Gate_B","Gate_C","Gate_D")))
        self.assertEqual(v["ddof"],1)

    def test_ties_are_not_strict_wins(self):
        s,*_=F.calculate({a:metric(a) for a in M.ARMS},self.cfg)
        self.assertFalse(s["Gate_A"]);self.assertFalse(s["Gate_B"]);self.assertTrue(s["Gate_D"])
        self.assertEqual(s["status"],"SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED")

    def test_class_sacrifice_gate(self):
        metrics={a:metric(a) for a in M.ARMS}
        for b in M.BANKS:metrics[b+"_curriculum"]=metric(b+"_curriculum",.31,.35,.25)
        s,*_=F.calculate(metrics,self.cfg);self.assertFalse(s["Gate_C"])
        self.assertAlmostEqual(s["Black_mean_delta_pp"],-5)

    def test_variance_gate_is_sample_sd(self):
        metrics={a:metric(a) for a in M.ARMS}
        for b,v in zip(M.BANKS,(.3,.32,.34)):metrics[b+"_curriculum"]=metric(b+"_curriculum",v)
        s,*_=F.calculate(metrics,self.cfg);self.assertFalse(s["Gate_D"])
        self.assertAlmostEqual(s["curriculum_bank_std_pp"],2)

    def test_incomplete_or_nonfinite_metrics_rejected(self):
        metrics={a:metric(a) for a in M.ARMS}
        metrics["A_uniform"]["map5095"]=float("nan")
        with self.assertRaises(M.AuditStop):F.calculate(metrics,self.cfg)
        del metrics["A_uniform"]
        with self.assertRaises(M.AuditStop):F.calculate(metrics,self.cfg)

    def test_unknown_status_no_fake_results(self):
        s=M.initial_status();self.assertEqual(s["Gate_A"],"NOT_RUN");self.assertIsNone(s["mean_delta_pp"])
        self.assertFalse(s["full_experiment_complete"]);self.assertEqual(s["internal_dev_eval_rounds"],0)
        self.assertFalse(s["pipeline_independent_holdout_claim"])

    def test_no_native_full_train_or_new_generator(self):
        for name in M.NEW_CODE:
            source=(REPO/"tools"/name).read_text(encoding="utf-8");tree=ast.parse(source)
            native_train=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == "train"]
            self.assertFalse(native_train,name)
            self.assertNotIn("StableDiffusion",source)
            self.assertNotIn("best.pt",source)

    def test_native_image_guard_blocks_before_decoder(self):
        calls=[]
        cv=SimpleNamespace(imread=lambda p,*a,**k:calls.append(p))
        np=SimpleNamespace(fromfile=lambda p,*a,**k:calls.append(p))
        official=REPO/"official.jpg";dev=REPO/"dev.jpg";train=REPO/"train.jpg"
        g=M.ContentGuard([official],[dev]);g.native_images.add(str(train.resolve()))
        with patch.dict(sys.modules,{"cv2":cv,"numpy":np}),g.native_guard():
            with self.assertRaises(M.AuditStop):cv.imread(str(official))
            with self.assertRaises(M.AuditStop):np.fromfile(str(dev))
            cv.imread(str(train));np.fromfile(str(train))
        self.assertEqual(calls,[str(train),str(train)])

    def test_actual_draw_log_pairing_all_occurrences_and_tamper(self):
        # Simulated traces exercise the auditor; they are never written as experimental output.
        docs={"stage21b_base_protocol_snapshot.json":self.base,
              "uniform_schedule_v2.json":self.u,"curriculum_schedule_v2.json":self.c}
        logs={}
        for arm in M.ARMS:
            mode=arm.split("_")[1];s=docs[mode+"_schedule_v2.json"]
            rows=[]
            for e,draws in enumerate(s["epoch_draws"]):
                for p,(i,k) in enumerate(draws):
                    rows.append({"epoch":e,"position":p,"role_index":i,"occurrence_index":k,
                        "augmentation_seed":M.occurrence_seed(s["roles"][i],k,self.base),"normalizer":64 if p<128 else 42,
                        "augmentation_sha256":"a"*64,"loss":1})
            logs[arm+"_draws.jsonl"]="".join(json.dumps(x)+"\n" for x in rows)
            docs[arm+"_training.json"]={"status":"COMPLETE","epochs":150,"actual_draws":25500,
                "attempted_optimizer_updates":450,"successful_optimizer_updates":450,"AMP_skipped_updates":0,
                "schedule_sha256":"hash","per_draw_log_sha256":"hash","initialization_sha256":"same",
                "epoch_records":[{"epoch_0based":e,"draws":170,"sequence_sha256":M.digest(draws),
                    "attempts":[{"position":p,"draws":n,"normalizer":n,"optimizer_step_executed":True,"AMP_skip":False}
                                for p,n in ((63,64),(127,64),(169,42))]} for e,draws in enumerate(s["epoch_draws"])]}
        written={}
        with patch.object(F,"load",side_effect=lambda p:docs[Path(p).name]),patch.object(F,"sha",return_value="hash"),\
             patch.object(Path,"open",lambda p,*a,**k:io.StringIO(logs[p.name])),\
             patch.object(F,"save",side_effect=lambda p,x:written.update({Path(p).name:x})):
            self.assertEqual(F.inspect_draw_logs(REPO)["status"],"PASS")
            logs["C_curriculum_draws.jsonl"]=logs["C_curriculum_draws.jsonl"].replace("a"*64,"b"*64,1)
            with self.assertRaises(M.AuditStop):F.inspect_draw_logs(REPO)
        self.assertEqual(written["augmentation_pairing_audit.json"]["banks"]["C"]["mismatched_occurrence_count"],1)


if __name__ == "__main__":unittest.main()
