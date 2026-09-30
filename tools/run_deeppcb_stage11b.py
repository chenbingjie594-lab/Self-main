"""Train and evaluate the four frozen Stage11B real-replay allocation arms."""
from __future__ import annotations
import argparse,hashlib,json,random
from datetime import datetime,timezone
from pathlib import Path
from types import MethodType
import numpy as np

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
ARMS=("uniform","feature","hardness","gradient")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def metric(r):
 b=r.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i]),"recall":float(b.r[i])} for i,n in enumerate(CLASSES)}}
def archive_incomplete_run(run,output,arm,epochs):
 run=Path(run);exposure=Path(output)/f"epoch_exposure_{arm}.json";complete=False
 if exposure.is_file() and (run/"weights/last.pt").is_file() and (run/"weights/best.pt").is_file():
  try:complete=len(load(exposure).get("epochs",[]))==epochs
  except (OSError,ValueError,TypeError):complete=False
 if complete:raise RuntimeError("STAGE11B_FORMAL_RUN_ALREADY_COMPLETE_REFUSING_OVERWRITE: "+str(run))
 archive=run.parent/"_failed_runs"/(run.name+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));archive.parent.mkdir(parents=True,exist_ok=True)
 if archive.exists():raise RuntimeError("STAGE11B_FAILED_RUN_ARCHIVE_COLLISION: "+str(archive))
 run.rename(archive);return str(archive)
class Sampler:
 def __init__(self,names,seq):self.names=tuple(names);self.idx={x:i for i,x in enumerate(names)};self.seq=seq;self.epoch=0;self.observed=[]
 def __len__(self):return len(self.seq[self.epoch])
 def set_epoch(self,e):self.epoch=int(e)
 def __iter__(self):
  self.observed=[]
  for n in self.seq[self.epoch]:self.observed.append(n);yield self.idx[n]
def delta(a,b):
 return {"overall":{"map50_95_raw":a["map50_95"]-b["map50_95"],"map50_95_pp":100*(a["map50_95"]-b["map50_95"]),"map50_raw":a["map50"]-b["map50"],"map50_pp":100*(a["map50"]-b["map50"]),"mean_recall_raw":a["recall"]-b["recall"],"mean_recall_pp":100*(a["recall"]-b["recall"])},"per_class":{c:{"ap50_95_raw":a["per_class"][c]["ap50_95"]-b["per_class"][c]["ap50_95"],"ap50_95_pp":100*(a["per_class"][c]["ap50_95"]-b["per_class"][c]["ap50_95"])} for c in CLASSES}}
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","prepared","dataset_root","real_dataset","model","runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);freeze=load(a.prepared/"selection_freeze_audit.json");assert freeze["status"]=="SELECTIONS_FROZEN_BEFORE_DETECTOR_TRAINING"
 for arm in ARMS:
  if sha(a.prepared/f"{arm}_selection.json")!=freeze["selection_sha256"][arm]:raise RuntimeError("STAGE11B_SELECTION_CHANGED: "+arm)
 iso=load(a.prepared/"replay_isolation_audit.json");assert iso["status"]=="REPLAY_ISOLATION_PASS";build=load(a.prepared/"dataset_build_audit.json");assert all(build["arms"][x]["images"]==220 and build["arms"][x]["labels"]==220 for x in ARMS)
 seq=[]
 for e in range(cfg["epochs"]):
  rng=random.Random(cfg["detector_seed"]+e);base=[rng.choice([f"base_{i:03d}" for i in range(100)]) for _ in range(cfg["base_real_draws_per_epoch"])];extra=[f"extra_{i:03d}" for i in range(cfg["extra_replay_draws_per_epoch"])];rng.shuffle(extra);z=base+extra;rng.shuffle(z);seq.append(z)
 sequence_hashes=[hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest() for x in seq]
 import torch,yaml
 from torch.utils.data import DataLoader
 from ultralytics import YOLO
 from ultralytics.data.build import seed_worker
 from ultralytics.models.yolo.detect import DetectionTrainer
 class Loader(DataLoader):
  def reset(self):return None
 primary=[];secondary=[];budgets={};args_audit={};archived_failed_runs={}
 for arm in ARMS:
  run=a.runs/f"{arm}_repeat120_s42"
  if run.exists():archived_failed_runs[arm]=archive_incomplete_run(run,a.output,arm,cfg["epochs"]);save(a.output/"failed_run_recovery_audit.json",{"policy":"archive incomplete run only; never overwrite a completed 150-epoch formal run","archives":archived_failed_runs})
  state={"sampler":None,"steps":0,"epochs":[]}
  class Trainer(DetectionTrainer):
   def get_dataloader(self,path,batch_size=16,rank=0,mode="train"):
    if mode!="train":return super().get_dataloader(path,batch_size,rank,mode)
    ds=self.build_dataset(path,mode,batch_size);names=[Path(x).stem for x in ds.im_files]
    if set(names)!={*(f"base_{i:03d}" for i in range(100)),*(f"extra_{i:03d}" for i in range(120))}:raise RuntimeError("STAGE11B_DATASET_NAMES_CHANGED")
    state["sampler"]=Sampler(names,seq);return Loader(ds,batch_size=cfg["batch"],sampler=state["sampler"],num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(cfg["detector_seed"]))
   def preprocess_batch(self,batch):self.consumed.extend(Path(x).stem for x in batch["im_file"]);return super().preprocess_batch(batch)
  def start(t):
   old=t.optimizer.step
   def step(self,*x,**kw):r=old(*x,**kw);state["steps"]+=1;return r
   t.optimizer.step=MethodType(step,t.optimizer)
  def es(t):state["sampler"].set_epoch(t.epoch);t.consumed=[]
  def ee(t):
   expected=seq[t.epoch];actual=t.consumed
   if actual!=expected:
    first=next((i for i,(x,y) in enumerate(zip(actual,expected)) if x!=y),None);save(a.output/f"exposure_mismatch_{arm}.json",{"epoch":t.epoch+1,"expected_count":len(expected),"consumed_count":len(actual),"first_mismatch_index":first,"expected_at_mismatch":expected[first] if first is not None else None,"consumed_at_mismatch":actual[first] if first is not None and first<len(actual) else None,"sampler_yield_count_including_prefetch":len(state["sampler"].observed)});raise RuntimeError("STAGE11B_EXPOSURE_SEQUENCE_MISMATCH")
   base=sum(x.startswith("base_") for x in actual);extra=sum(x.startswith("extra_") for x in actual)
   if (base,extra,len(actual))!=(260,120,380):raise RuntimeError("STAGE11B_EXPOSURE_COUNT_MISMATCH")
   state["epochs"].append({"epoch":t.epoch+1,"base_draws":base,"extra_draws":extra,"total_draws":len(actual),"batches":len(t.train_loader),"cumulative_optimizer_steps":state["steps"],"sequence_sha256":sequence_hashes[t.epoch],"consumed_sequence_verified":True,"sampler_yield_count_including_prefetch":len(state["sampler"].observed),"sampler_prefetch_not_used_as_exposure_evidence":True})
   save(a.output/f"epoch_exposure_{arm}.json",{"epochs":state["epochs"]})
  m=YOLO(str(a.model));m.add_callback("on_train_start",start);m.add_callback("on_train_epoch_start",es);m.add_callback("on_train_epoch_end",ee);m.train(data=str((a.dataset_root/arm/"data.yaml").resolve()),imgsz=cfg["imgsz"],epochs=cfg["epochs"],patience=cfg["patience"],batch=cfg["batch"],deterministic=True,seed=cfg["detector_seed"],optimizer=cfg["optimizer"],mosaic=0.,mixup=0.,copy_paste=0.,cutmix=0.,val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer)
  last=run/"weights/last.pt";best=run/"weights/best.pt"
  if not last.is_file() or not best.is_file() or len(state["epochs"])!=150:raise RuntimeError("STAGE11B_TRAINING_INCOMPLETE: "+arm)
  train_args=yaml.safe_load((run/"args.yaml").read_text());args_audit[arm]={k:train_args.get(k) for k in ("model","imgsz","epochs","patience","batch","seed","deterministic","optimizer","mosaic","mixup","copy_paste","cutmix","val")}
  common={"data":str((a.real_dataset/"data.yaml").resolve()),"split":"val","imgsz":cfg["imgsz"],"device":a.device,"verbose":False,"plots":False,"project":str(a.runs/"final_validation"),"exist_ok":False}
  lm=metric(YOLO(str(last)).val(name=f"{arm}_last",**common));bm=metric(YOLO(str(best)).val(name=f"{arm}_best",**common));primary.append({"arm":arm,"checkpoint":"epoch150 last.pt","checkpoint_sha256":sha(last),**lm});secondary.append({"arm":arm,"checkpoint":"best.pt","checkpoint_sha256":sha(best),**bm});budgets[arm]={"epochs":len(state["epochs"]),"total_draws":sum(x["total_draws"] for x in state["epochs"]),"base_draws":sum(x["base_draws"] for x in state["epochs"]),"extra_draws":sum(x["extra_draws"] for x in state["epochs"]),"total_batches":sum(x["batches"] for x in state["epochs"]),"optimizer_steps":state["steps"],"sequence_hashes_sha256":hashlib.sha256(json.dumps(sequence_hashes,separators=(",",":")).encode()).hexdigest(),"last_sha256":sha(last),"best_sha256":sha(best)}
 save(a.output/"primary_last_metrics.json",{"primary_checkpoint":"epoch150 last.pt","detector_seed":42,"records":primary});save(a.output/"secondary_best_metrics.json",{"role":"secondary trajectory sanity only; forbidden for gate","records":secondary})
 schedule_fields=("epochs","total_draws","base_draws","extra_draws","total_batches","sequence_hashes_sha256");equal={k:len({v[k] for v in budgets.values()})==1 for k in schedule_fields};equal["successful_optimizer_steps"]=len({v["optimizer_steps"] for v in budgets.values()})==1;equal["training_arguments"]=len({json.dumps(v,sort_keys=True) for v in args_audit.values()})==1;expected={"imgsz":512,"epochs":150,"patience":151,"batch":8,"seed":42,"deterministic":True,"optimizer":"auto","mosaic":0.0,"mixup":0.0,"copy_paste":0.0,"cutmix":0.0,"val":False};arguments_match=all(all(v.get(k)==q for k,q in expected.items()) for v in args_audit.values());equal["arguments_match_frozen_protocol"]=arguments_match;required=("epochs","total_draws","base_draws","extra_draws","total_batches","sequence_hashes_sha256","training_arguments","arguments_match_frozen_protocol");budget_valid=all(equal[k] for k in required);step_values=[v["optimizer_steps"] for v in budgets.values()];budget_status="EQUAL_BUDGET_PASS" if budget_valid and equal["successful_optimizer_steps"] else "EQUAL_SCHEDULE_PASS_WITH_SUCCESSFUL_UPDATE_VARIATION" if budget_valid else "EQUAL_BUDGET_FAILED";save(a.output/"training_budget_audit.json",{"status":budget_status,"protocol_required_equal_fields":list(required),"arms":budgets,"equal":equal,"successful_optimizer_step_range":max(step_values)-min(step_values),"successful_optimizer_steps_are_reported_but_not_a_frozen_equal_schedule_gate":True,"training_arguments":args_audit,"frozen_expected_arguments":expected,"pretrained_weights_sha256":sha(a.model),"imgsz":512,"detector_seed":42})
 save(a.output/"validation_usage_audit.json",{"status":"FINAL_EVALUATION_ONLY","official_validation":"frozen DeepPCB official validation from real_dataset/data.yaml","selection_uses":0,"threshold_uses":0,"allocation_uses":0,"hyperparameter_tuning_uses":0,"training_epoch_uses":0,"early_stopping_uses":0,"final_checkpoint_evaluations":8,"evaluations_per_arm":["epoch150 last.pt primary","best.pt secondary"],"training_yaml_internal_val":"arm training set, with val=False; never official validation"})
 if not budget_valid:raise RuntimeError("STAGE11B_UNEQUAL_TRAINING_BUDGET")
 r={x["arm"]:x for x in primary};d={"gradient_minus_uniform":delta(r["gradient"],r["uniform"]),"gradient_minus_feature":delta(r["gradient"],r["feature"]),"gradient_minus_hardness":delta(r["gradient"],r["hardness"])};save(a.output/"allocation_primary_deltas.json",d)
 gu=d["gradient_minus_uniform"];class_delta=[gu["per_class"][c]["ap50_95_raw"] for c in CLASSES];A=gu["overall"]["map50_95_raw"]>=cfg["gate"]["gradient_minus_uniform_map50_95_min"];B=r["gradient"]["map50_95"]>r["feature"]["map50_95"];C=r["gradient"]["map50_95"]>r["hardness"]["map50_95"];D=sum(x>=0 for x in class_delta)>=cfg["gate"]["nonnegative_class_delta_min_count"] and min(class_delta)>=cfg["gate"]["minimum_any_class_delta"]
 if A and B and C and D:status="GRADIENT_ALLOCATION_REAL_REPLAY_SUPPORTED"
 elif gu["overall"]["map50_95_raw"]>0:status="GRADIENT_ALLOCATION_REAL_REPLAY_MIXED"
 else:status="GRADIENT_ALLOCATION_REAL_REPLAY_NOT_SUPPORTED"
 final={"status":status,"gates":{"A_beat_uniform_by_0_5pp":A,"B_strictly_beat_feature":B,"C_strictly_beat_hardness":C,"D_no_broad_class_collapse":D},"nonnegative_gradient_vs_uniform_classes":sum(x>=0 for x in class_delta),"worst_class_delta":min(class_delta),"STAGE11B_CROSS_ARCH_CONFIRMATION_AUTHORIZED":status=="GRADIENT_ALLOCATION_REAL_REPLAY_SUPPORTED","architecture_coupled_feasibility_only":True,"synthetic_count":0,"STAGE11BR_AUTOMATICALLY_STARTED":False};save(a.output/"stage11b_status.json",final)
 lines=["# DeepPCB Stage11B Real-Replay Allocation Pilot","",f"Status: **{status}**.","","Primary checkpoint: epoch150 `last.pt`; detector seed: 42; architecture: YOLO11s.","","## Frozen gates",""]+[f"- {k}: **{v}**" for k,v in final["gates"].items()]+["","## Primary mAP50-95",""]+[f"- {arm}: {r[arm]['map50_95']:.6f}" for arm in ARMS]+["",f"Gradient - Uniform: {gu['overall']['map50_95_pp']:+.3f} pp.","",f"Cross-architecture confirmation authorized: **{final['STAGE11B_CROSS_ARCH_CONFIRMATION_AUTHORIZED']}**.","","This is an architecture-coupled, single-seed real-only feasibility result. No synthetic generation, selector tuning, official-validation allocation, or post-hoc protocol change was used."];(a.output/"STAGE11B_REPORT.md").write_text("\n".join(lines)+"\n")
 print(json.dumps(final,indent=2))
if __name__=="__main__":main()
