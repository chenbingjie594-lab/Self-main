"""Train the single authorized RCCRepeat120 arm after frozen preflight gates."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
from types import MethodType
import numpy as np
from run_deeppcb_stage11b import Sampler,sha,metric,delta

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","prepared","dataset_root","real_dataset","model","stage11b","stage11b_runs","stage12a","stage11a","stage3r","runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);man=load(a.prepared/"coverage_manipulation_audit.json");freeze=load(a.prepared/"selection_freeze_audit.json")
 if man["status"]!="RCC_MANIPULATION_PASS" or sha(a.prepared/"rcc_selection.json")!=freeze["selection_sha256"]:raise RuntimeError("STAGE12B_PREFLIGHT_GATE_FAILED")
 import torch,yaml,ultralytics
 hist_budget=load(a.stage11b/"training_budget_audit.json");hist_args=hist_budget["training_arguments"]["uniform"];hist_last=a.stage11b_runs/"uniform_repeat120_s42/weights/last.pt";current_sha=sha(a.model);ck=torch.load(hist_last,map_location="cpu",weights_only=False);hist_version=str(ck.get("version","UNKNOWN"));current_version=str(ultralytics.__version__);expected={"imgsz":512,"epochs":150,"patience":151,"batch":8,"seed":42,"deterministic":True,"optimizer":"auto","mosaic":0.0,"mixup":0.0,"copy_paste":0.0,"cutmix":0.0,"val":False};env={"pretrained_weights_sha256_current":current_sha,"pretrained_weights_sha256_stage11b":hist_budget["pretrained_weights_sha256"],"pretrained_match":current_sha==hist_budget["pretrained_weights_sha256"],"ultralytics_current":current_version,"ultralytics_stage11b_checkpoint":hist_version,"ultralytics_match":current_version==hist_version,"historical_uniform_arguments":hist_args,"arguments_match":all(hist_args.get(k)==v for k,v in expected.items())};env["status"]="ENVIRONMENT_MATCH" if all((env["pretrained_match"],env["ultralytics_match"],env["arguments_match"])) else "HISTORICAL_COMPARATOR_ENVIRONMENT_MISMATCH";save(a.output/"historical_comparator_environment_audit.json",env)
 if env["status"]!="ENVIRONMENT_MATCH":save(a.output/"stage12b_status.json",{"status":env["status"],"STAGE12C_SYNTHETIC_COVERAGE_SELECTION_AUTHORIZED":False,"detector_training_count":0});raise RuntimeError(env["status"])
 seq=[]
 for e in range(cfg["epochs"]):
  rng=random.Random(cfg["detector_seed"]+e);base=[rng.choice([f"base_{i:03d}" for i in range(100)]) for _ in range(cfg["base_real_draws_per_epoch"])];extra=[f"extra_{i:03d}" for i in range(120)];rng.shuffle(extra);z=base+extra;rng.shuffle(z);seq.append(z)
 hashes=[hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest() for x in seq];schedule_sha=hashlib.sha256(json.dumps(hashes,separators=(",",":")).encode()).hexdigest();schedule={"stage11b_uniform_sequence_hashes_sha256":hist_budget["arms"]["uniform"]["sequence_hashes_sha256"],"rcc_sequence_hashes_sha256":schedule_sha,"same_base_and_shuffle_schedule":schedule_sha==hist_budget["arms"]["uniform"]["sequence_hashes_sha256"],"epochs":150,"draws_per_epoch":380,"base_draws_per_epoch":260,"extra_draws_per_epoch":120};save(a.output/"training_schedule_equivalence.json",schedule)
 if not schedule["same_base_and_shuffle_schedule"]:raise RuntimeError("STAGE12B_SCHEDULE_MISMATCH")
 from torch.utils.data import DataLoader
 from ultralytics import YOLO
 from ultralytics.data.build import seed_worker
 from ultralytics.models.yolo.detect import DetectionTrainer
 class Loader(DataLoader):
  def reset(self):return None
 state={"sampler":None,"steps":0,"epochs":[]}
 class Trainer(DetectionTrainer):
  def get_dataloader(self,path,batch_size=16,rank=0,mode="train"):
   if mode!="train":return super().get_dataloader(path,batch_size,rank,mode)
   ds=self.build_dataset(path,mode,batch_size);names=[Path(x).stem for x in ds.im_files]
   if set(names)!={*(f"base_{i:03d}" for i in range(100)),*(f"extra_{i:03d}" for i in range(120))}:raise RuntimeError("STAGE12B_DATASET_NAMES_CHANGED")
   state["sampler"]=Sampler(names,seq);return Loader(ds,batch_size=8,sampler=state["sampler"],num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
  def preprocess_batch(self,batch):self.consumed.extend(Path(x).stem for x in batch["im_file"]);return super().preprocess_batch(batch)
 def start(t):
  old=t.optimizer.step
  def step(self,*x,**kw):r=old(*x,**kw);state["steps"]+=1;return r
  t.optimizer.step=MethodType(step,t.optimizer)
 def es(t):state["sampler"].set_epoch(t.epoch);t.consumed=[]
 def ee(t):
  if t.consumed!=seq[t.epoch]:raise RuntimeError("STAGE12B_EXPOSURE_SEQUENCE_MISMATCH")
  state["epochs"].append({"epoch":t.epoch+1,"base_draws":260,"extra_draws":120,"total_draws":380,"batches":len(t.train_loader),"cumulative_optimizer_steps":state["steps"],"sequence_sha256":hashes[t.epoch],"consumed_sequence_verified":True});save(a.output/"epoch_exposure_rcc.json",{"epochs":state["epochs"]})
 run=a.runs/"rcc_repeat120_s42"
 if run.exists():raise RuntimeError("STAGE12B_RUN_EXISTS: "+str(run))
 m=YOLO(str(a.model));m.add_callback("on_train_start",start);m.add_callback("on_train_epoch_start",es);m.add_callback("on_train_epoch_end",ee);m.train(data=str((a.dataset_root/"rcc/data.yaml").resolve()),imgsz=512,epochs=150,patience=151,batch=8,deterministic=True,seed=42,optimizer="auto",mosaic=0.,mixup=0.,copy_paste=0.,cutmix=0.,val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer)
 last,best=run/"weights/last.pt",run/"weights/best.pt"
 if len(state["epochs"])!=150 or not last.is_file() or not best.is_file():raise RuntimeError("STAGE12B_TRAINING_INCOMPLETE")
 common={"data":str((a.real_dataset/"data.yaml").resolve()),"split":"val","imgsz":512,"device":a.device,"verbose":False,"plots":False,"project":str(a.runs/"final_validation"),"exist_ok":False};primary={"arm":"rcc","checkpoint":"epoch150 last.pt","checkpoint_sha256":sha(last),**metric(YOLO(str(last)).val(name="rcc_last",**common))};secondary={"arm":"rcc","checkpoint":"best.pt","checkpoint_sha256":sha(best),**metric(YOLO(str(best)).val(name="rcc_best",**common))};save(a.output/"primary_last_metrics.json",{"records":[primary]});save(a.output/"secondary_best_metrics.json",{"role":"secondary only","records":[secondary]})
 uniform=next(x for x in load(a.stage11b/"primary_last_metrics.json")["records"] if x["arm"]=="uniform");d=delta(primary,uniform);class_delta=[d["per_class"][c]["ap50_95_raw"] for c in CLASSES];A=d["overall"]["map50_95_raw"]>=cfg["downstream_map50_95_delta_min"];B=sum(x>=0 for x in class_delta)>=cfg["nonnegative_class_delta_min_count"] and min(class_delta)>=cfg["minimum_any_class_delta"];save(a.output/"rcc_vs_uniform_deltas.json",{"historical_uniform":uniform,"rcc":primary,"delta":d,"gates":{"map50_95_plus_0_5pp":A,"class_stability":B,"nonnegative_classes":sum(x>=0 for x in class_delta),"worst_class_delta":min(class_delta)}})
 # Frozen post-hoc diagnostics only; none participated in selection or the primary gate.
 bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");rcc_sel=load(a.prepared/"rcc_selection.json")["records"];uniform_sel=load(a.stage11b/"uniform_selection.json")["records"]
 with np.load(a.stage11a/"real_oof_gradient_embeddings.npz") as z:gg=z["embeddings"].astype(float);gids=z["instance_ids"].astype(str);gmap={iid:gg[i] for i,iid in enumerate(gids)}
 def entropy(v):
  from collections import Counter
  c=np.asarray(list(Counter(v).values()),float);p=c/c.sum();return float(-(p*np.log(p)).sum()/max(np.log(len(c)),1e-12)) if len(c)>1 else 0.
 def diagnostics(sel):
  by={(int(x["fold"]),int(x["class_id"])):[] for x in sel}
  for x in sel:by[(int(x["fold"]),int(x["class_id"]))].append(x)
  support=[];ess=[];gmis=[]
  for e in bankdoc["folds"]:
   f=int(e["fold"])
   with np.load(a.stage3r/f"oof_real_bank_corrected_fold{f}.npz") as z:X=z["features"].astype(float)
   X/=np.linalg.norm(X,axis=1,keepdims=True).clip(1e-12)
   for ci in range(6):
    jj=[i for i,x in enumerate(e["records"]) if int(x["class_id"])==ci];R=[e["records"][i] for i in jj];F=X[jj];lookup={x["instance_id"]:i for i,x in enumerate(R)};S=by.get((f,ci),[])
    if not S:continue
    J=[lookup[x["instance_id"]] for x in S];D=1-np.clip(F[J]@F.T,-1,1);rp=np.asarray([x["pair_id"] for x in R])
    for k,x in enumerate(S):support.append(float(D[k][rp!=x["pair_id"]].min()))
    DD=1-np.clip(F[J]@F[J].T,-1,1);realD=1-np.clip(F@F.T,-1,1);mask=(rp[:,None]!=rp[None,:])&(~np.eye(len(F),dtype=bool));sigma=float(np.median(realD[mask]));K=np.exp(-(DD*DD)/(2*max(sigma,1e-12)**2));ess.append((float(len(J)**2/(K*K).sum()/len(J)),len(J)))
    mr=np.mean([gmap[x["instance_id"]] for x in R],0);ms=np.mean([gmap[x["instance_id"]] for x in S],0);gmis.append((float(1-np.dot(mr,ms)/max(np.linalg.norm(mr)*np.linalg.norm(ms),1e-12)),len(S)))
  return {"support_nearest_real_q90_raw":float(np.quantile(support,.9)),"feature_ess_normalized":float(np.average([x[0] for x in ess],weights=[x[1] for x in ess])),"pair_entropy_normalized":entropy([x["pair_id"] for x in sel]),"gradient_mean_misalignment":float(np.average([x[0] for x in gmis],weights=[x[1] for x in gmis]))}
 post={"selection_blind_to_these_diagnostics":True,"coverage":{"uniform":man["uniform"],"rcc":man["rcc"]},"uniform":diagnostics(uniform_sel),"rcc":diagnostics(rcc_sel),"used_for_primary_conclusion":False};save(a.output/"posthoc_set_diagnostics.json",post)
 if A and B:status="REAL_COVERAGE_COMPLETION_CAUSALLY_SUPPORTED"
 elif d["overall"]["map50_95_raw"]<=0:status="COVERAGE_ASSOCIATION_NOT_CAUSALLY_SUPPORTED"
 else:status="REAL_COVERAGE_COMPLETION_MIXED"
 final={"status":status,"coverage_manipulation_status":man["status"],"rcc_minus_uniform_map50_95_pp":d["overall"]["map50_95_pp"],"downstream_gate":A,"class_stability_gate":B,"STAGE12C_SYNTHETIC_COVERAGE_SELECTION_AUTHORIZED":status=="REAL_COVERAGE_COMPLETION_CAUSALLY_SUPPORTED","STAGE12C_AUTOMATICALLY_STARTED":False,"detector_training_count":1,"synthetic_generation_count":0,"official_validation_evaluations":2};save(a.output/"stage12b_status.json",final);save(a.output/"stage12b_protocol.json",cfg)
 (a.output/"STAGE12B_REPORT.md").write_text(f"# DeepPCB Stage12B Real Coverage Completion Causal Pilot\n\nStatus: **{status}**\n\n- Coverage q90 relative improvement: {man['relative_q90_improvement']:.3%}\n- Classes with lower q90: {man['improved_class_count']}/6\n- RCC - historical Uniform mAP50-95: {d['overall']['map50_95_pp']:+.3f} pp\n- Nonnegative classes: {sum(x>=0 for x in class_delta)}/6\n- Worst class delta: {100*min(class_delta):+.3f} pp\n- Stage12C authorized: {final['STAGE12C_SYNTHETIC_COVERAGE_SELECTION_AUTHORIZED']}\n\nF4/F5 show the Stage12A association is imperfect; this real-only intervention isolates synthetic-quality confounding. No synthetic generation or comparator retraining was performed. Stage12C was not started.\n",encoding="utf-8");print(json.dumps(final,indent=2))
if __name__=="__main__":main()
