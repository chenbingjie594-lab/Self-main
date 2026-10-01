"""Train the single frozen CPT120 detector arm and compare with Stage11B UniformRepeat120."""
from __future__ import annotations
import argparse,hashlib,json,os,random,shutil
from pathlib import Path
from types import MethodType
from collections import Counter

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():
  if dst.resolve()!=Path(src).resolve():raise RuntimeError("STAGE13A_DATASET_CONFLICT: "+str(dst))
 else:os.symlink(Path(src).resolve(),dst)
def metric(r):
 b=r.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i]),"recall":float(b.r[i])} for i,n in enumerate(CLASSES)}}
class Sampler:
 def __init__(self,names,seq):self.idx={x:i for i,x in enumerate(names)};self.seq=seq;self.epoch=0
 def __len__(self):return len(self.seq[self.epoch])
 def set_epoch(self,e):self.epoch=int(e)
 def __iter__(self):
  for n in self.seq[self.epoch]:yield self.idx[n]
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","prepared","stage11b","stage11b_runs","real_dataset","model","dataset_root","runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);mf=a.prepared/"cpt120_manifest.json";freeze=load(a.prepared/"cpt120_manifest_freeze.json")
 if sha(mf)!=freeze["sha256"]:raise RuntimeError("STAGE13A_MANIFEST_CHANGED")
 manifest=load(mf);assert manifest["status"]=="FROZEN" and manifest["count"]==120
 root=a.dataset_root/"cpt120";base_i={p.stem:p for p in (a.real_dataset/"images/train").iterdir()};base_l={p.stem:p for p in (a.real_dataset/"labels/train").iterdir()}
 if len(base_i)!=100 or set(base_i)!=set(base_l):raise RuntimeError("STAGE13A_BASE_DATASET_INVALID")
 for i,k in enumerate(sorted(base_i)):link(base_i[k],root/"images/train"/f"base_{i:03d}{base_i[k].suffix}");link(base_l[k],root/"labels/train"/f"base_{i:03d}.txt")
 for i,r in enumerate(sorted(manifest["records"],key=lambda x:(x["class_name"],x["donor_instance_id"]))):link(r["image_path"],root/"images/train"/f"extra_{i:03d}.png");link(r["label_path"],root/"labels/train"/f"extra_{i:03d}.txt")
 yp=root/"data.yaml";yp.parent.mkdir(parents=True,exist_ok=True);yp.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/train\nnc: 6\nnames: {json.dumps(list(CLASSES))}\n")
 seq=[]
 for e in range(150):
  rng=random.Random(42+e);base=[rng.choice([f"base_{i:03d}" for i in range(100)]) for _ in range(260)];extra=[f"extra_{i:03d}" for i in range(120)];rng.shuffle(extra);z=base+extra;rng.shuffle(z);seq.append(z)
 hashes=[hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest() for x in seq];seqhash=hashlib.sha256(json.dumps(hashes,separators=(",",":")).encode()).hexdigest();hist_budget=load(a.stage11b/"training_budget_audit.json");uniform=next(x for x in load(a.stage11b/"primary_last_metrics.json")["records"] if x["arm"]=="uniform")
 import torch,ultralytics,yaml
 historical=hist_budget["arms"]["uniform"];hist_args=hist_budget["training_arguments"]["uniform"];current={"pretrained_weights_sha256":sha(a.model),"ultralytics_version":ultralytics.__version__,"torch_version":torch.__version__,"cuda_major":str(torch.version.cuda).split(".")[0] if torch.version.cuda else None,"imgsz":512,"optimizer":"auto","seed":42,"epochs":150,"batch":8,"mosaic":0.0,"mixup":0.0,"copy_paste":0.0,"cutmix":0.0,"val":False,"base_sequence_hash":seqhash}
 checks={"pretrained_sha":current["pretrained_weights_sha256"]==hist_budget["pretrained_weights_sha256"],"imgsz":hist_args["imgsz"]==512,"optimizer":hist_args["optimizer"]=="auto","seed":hist_args["seed"]==42,"epochs":hist_args["epochs"]==150,"batch":hist_args["batch"]==8,"augmentations":all(hist_args[k] in (0,0.0,False) for k in ("mosaic","mixup","copy_paste","cutmix","val")),"base_schedule_hash":historical["sequence_hashes_sha256"]==seqhash}
 env={"status":"PASS" if all(checks.values()) else "HISTORICAL_COMPARATOR_ENVIRONMENT_MISMATCH","checks":checks,"current":current,"historical_training_arguments":hist_args,"historical_versions_not_recorded":{"ultralytics":True,"torch":True,"cuda_major":True},"policy":"unrecorded historical versions are reported, never fabricated"};save(a.output/"training_environment_equivalence.json",env)
 if not all(checks.values()):raise RuntimeError("HISTORICAL_COMPARATOR_ENVIRONMENT_MISMATCH")
 run=a.runs/"cpt120_s42"
 if run.exists():raise RuntimeError("STAGE13A_RUN_EXISTS: "+str(run))
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
   ds=self.build_dataset(path,mode,batch_size);names=[Path(x).stem for x in ds.im_files];state["sampler"]=Sampler(names,seq);return Loader(ds,batch_size=8,sampler=state["sampler"],num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
  def preprocess_batch(self,batch):self.consumed.extend(Path(x).stem for x in batch["im_file"]);return super().preprocess_batch(batch)
 def start(t):
  old=t.optimizer.step
  def step(self,*x,**kw):q=old(*x,**kw);state["steps"]+=1;return q
  t.optimizer.step=MethodType(step,t.optimizer)
 def es(t):state["sampler"].set_epoch(t.epoch);t.consumed=[]
 def ee(t):
  if t.consumed!=seq[t.epoch]:raise RuntimeError("STAGE13A_EXPOSURE_SEQUENCE_MISMATCH")
  state["epochs"].append({"epoch":t.epoch+1,"base_draws":260,"cpt_draws":120,"total_draws":380,"batches":len(t.train_loader),"cumulative_optimizer_steps":state["steps"],"sequence_sha256":hashes[t.epoch]});save(a.output/"epoch_exposure_cpt120.json",{"epochs":state["epochs"]})
 m=YOLO(str(a.model));m.add_callback("on_train_start",start);m.add_callback("on_train_epoch_start",es);m.add_callback("on_train_epoch_end",ee);m.train(data=str(yp.resolve()),imgsz=512,epochs=150,patience=151,batch=8,deterministic=True,seed=42,optimizer="auto",mosaic=0.,mixup=0.,copy_paste=0.,cutmix=0.,val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer)
 last=run/"weights/last.pt";best=run/"weights/best.pt";common={"data":str((a.real_dataset/"data.yaml").resolve()),"split":"val","imgsz":512,"device":a.device,"verbose":False,"plots":False,"project":str(a.runs/"final_validation"),"exist_ok":False};lm=metric(YOLO(str(last)).val(name="cpt_last",**common));bm=metric(YOLO(str(best)).val(name="cpt_best",**common));save(a.output/"primary_last_metrics.json",{"arm":"CPT120","checkpoint":"epoch150 last.pt","checkpoint_sha256":sha(last),**lm});save(a.output/"secondary_best_metrics.json",{"arm":"CPT120","role":"secondary only","checkpoint_sha256":sha(best),**bm})
 save(a.output/"validation_usage_audit.json",{"status":"FINAL_EVALUATION_ONLY","official_validation_selection_uses":0,"recipient_matching_uses":0,"transport_gate_uses":0,"hyperparameter_tuning_uses":0,"training_uses":0,"final_evaluations":{"last_pt":1,"best_pt":1},"best_pt_role":"secondary only"})
 budget={"epochs":len(state["epochs"]),"total_draws":sum(x["total_draws"] for x in state["epochs"]),"base_draws":39000,"cpt_draws":18000,"total_batches":sum(x["batches"] for x in state["epochs"]),"successful_optimizer_steps":state["steps"],"sequence_hashes_sha256":seqhash};schedule_ok=budget["epochs"]==150 and budget["total_draws"]==57000 and budget["total_batches"]==7200 and seqhash==historical["sequence_hashes_sha256"];save(a.output/"training_schedule_equivalence.json",{"status":"PASS" if schedule_ok else "FAIL","cpt":budget,"historical_uniform":historical,"successful_optimizer_steps_reported_not_equalized":True})
 if not schedule_ok:raise RuntimeError("STAGE13A_SCHEDULE_MISMATCH")
 d={"overall":{"map50_95_raw":lm["map50_95"]-uniform["map50_95"],"map50_95_pp":100*(lm["map50_95"]-uniform["map50_95"]),"map50_raw":lm["map50"]-uniform["map50"],"map50_pp":100*(lm["map50"]-uniform["map50"]),"recall_raw":lm["recall"]-uniform["recall"],"recall_pp":100*(lm["recall"]-uniform["recall"])},"per_class":{c:{"ap50_95_raw":lm["per_class"][c]["ap50_95"]-uniform["per_class"][c]["ap50_95"],"ap50_95_pp":100*(lm["per_class"][c]["ap50_95"]-uniform["per_class"][c]["ap50_95"])} for c in CLASSES}};save(a.output/"cpt_vs_uniform_deltas.json",d);cd=[d["per_class"][c]["ap50_95_raw"] for c in CLASSES];A=d["overall"]["map50_95_raw"]>=.005;B=sum(x>=0 for x in cd)>=4 and min(cd)>=-.02
 status="COUNTERFACTUAL_PAIR_TRANSPORT_SUPPORTED" if A and B else "COUNTERFACTUAL_PAIR_TRANSPORT_MIXED" if d["overall"]["map50_95_raw"]>0 or B else "CROSS_CONTEXT_RECOMBINATION_NOT_SUPPORTED";final={"status":status,"gates":{"overall_gain_at_least_0_5pp":A,"class_stability":B},"nonnegative_classes":sum(x>=0 for x in cd),"worst_class_delta":min(cd),"localization_consistency":{"map50_delta":d["overall"]["map50_raw"],"map50_95_delta":d["overall"]["map50_95_raw"]},"STAGE13B_AUTOMATICALLY_STARTED":False};save(a.output/"stage13a_status.json",final);(a.output/"STAGE13A_REPORT.md").write_text(f"# DeepPCB Stage13A Counterfactual Pair Transport\n\nStatus: **{status}**.\n\nCPT120 - UniformRepeat120 mAP50-95: {d['overall']['map50_95_pp']:+.3f} pp.\n\nNonnegative classes: {sum(x>=0 for x in cd)}/6; worst class delta: {100*min(cd):+.3f} pp.\n\nPrimary checkpoint is epoch150 `last.pt`; `best.pt` is secondary only. No generator, learned matching, donor reselection, official-validation tuning, or Stage13B auto-start was used.\n")
 print(json.dumps(final,indent=2))
if __name__=="__main__":main()
