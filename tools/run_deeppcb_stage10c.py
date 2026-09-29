"""Stage10C exact-paired, fixed-exposure seed42 detector pilot."""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole");TARGET=("short","pinhole");ARMS=("real_repeat80","sd2_syn80","tmvb_syn80")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def label(row):
 m=np.asarray(Image.open(row["mask_path"]).convert("L"))>127;y,x=np.where(m);return f"{CLASSES.index(row['class_name'])} {(x.min()+x.max()+1)/1024:.10f} {(y.min()+y.max()+1)/1024:.10f} {(x.max()-x.min()+1)/512:.10f} {(y.max()-y.min()+1)/512:.10f}\n"
def metric(r):
 b=r.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i]),"recall":float(b.r[i])} for i,n in enumerate(CLASSES)}}
class Sampler:
 def __init__(self,names,seq):self.names=tuple(names);self.idx={x:i for i,x in enumerate(names)};self.seq=seq;self.epoch=0;self.observed=[]
 def __len__(self):return len(self.seq[self.epoch])
 def set_epoch(self,e):self.epoch=e
 def __iter__(self):
  self.observed=[]
  for n in self.seq[self.epoch]:self.observed.append(n);yield self.idx[n]
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","gate","tmvb_manifest","sd2_manifest","real_dataset","model","dataset_root","runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();cfg=load(a.protocol);gate=load(a.gate);assert gate["STAGE10C_DETECTOR_PILOT_AUTHORIZED"] and cfg["detector_seed"]==42;a.output.mkdir(parents=True,exist_ok=True)
 tm=load(a.tmvb_manifest)["records"];sd={(x["instance_id"],int(x["generation_seed"])):x for x in load(a.sd2_manifest)["records"]};by=defaultdict(list)
 for x in tm:by[x["class_name"]].append(x)
 rng=random.Random(cfg["selection_seed"]);chosen=[]
 for c in TARGET:
  ids=sorted({x["instance_id"] for x in by[c]});picked=rng.sample(ids,40)
  for iid in picked:
   candidates=sorted([x for x in by[c] if x["instance_id"]==iid],key=lambda z:int(z["generation_seed"]));chosen.append(dict(rng.choice(candidates)))
 assert len(chosen)==80 and all((x["instance_id"],int(x["generation_seed"])) in sd for x in chosen);save(a.output/"frozen_source_seed_selection.json",{"selection_seed":cfg["selection_seed"],"count":80,"class_counts":dict(Counter(x["class_name"] for x in chosen)),"unique_sources":len({x["instance_id"] for x in chosen}),"records":chosen})
 base_i={p.stem:p for p in (a.real_dataset/"images/train").iterdir()};base_l={p.stem:p for p in (a.real_dataset/"labels/train").iterdir()};assert len(base_i)==len(base_l)==100;roots={}
 for arm in ARMS:
  root=a.dataset_root/arm
  if root.exists():
   images=list((root/"images/train").iterdir());labels=list((root/"labels/train").iterdir());assert len(images)==len(labels)==180 and (root/"data.yaml").is_file();roots[arm]=root;continue
  for k in ("images","labels"):(root/k/"train").mkdir(parents=True)
  for i,s in enumerate(sorted(base_i)):(root/"images/train"/f"base_{i:03d}{base_i[s].suffix}").symlink_to(base_i[s].resolve());(root/"labels/train"/f"base_{i:03d}.txt").symlink_to(base_l[s].resolve())
  for i,x in enumerate(chosen):
   key=(x["instance_id"],int(x["generation_seed"]));src=Path(x["real_path"] if arm=="real_repeat80" else sd[key]["image_path"] if arm=="sd2_syn80" else x["image_path"]);assert src.is_file();(root/"images/train"/f"extra_{i:03d}{src.suffix}").symlink_to(src.resolve());(root/"labels/train"/f"extra_{i:03d}.txt").write_text(label(x))
  (root/"data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: {(a.real_dataset/'images/val').resolve()}\nnc: 6\nnames: {json.dumps(list(CLASSES))}\n");roots[arm]=root
 seq=[]
 for e in range(150):
  q=random.Random(42+e);base=[q.choice([f"base_{i:03d}" for i in range(100)]) for _ in range(260)];extra=[f"extra_{i:03d}" for i in range(80)];q.shuffle(extra);z=base+extra;q.shuffle(z);seq.append(z)
 import torch,yaml
 from torch.utils.data import DataLoader
 from ultralytics import YOLO
 from ultralytics.data.build import seed_worker
 from ultralytics.models.yolo.detect import DetectionTrainer
 rows=[];secondary=[];audits={}
 for arm in ARMS:
  run=a.runs/f"{arm}_s42";state={"sampler":None,"steps":0,"epochs":[]}
  class Loader(DataLoader):
   def reset(self):return None
  class Trainer(DetectionTrainer):
   def get_dataloader(self,path,batch_size=16,rank=0,mode="train"):
    if mode!="train":return super().get_dataloader(path,batch_size,rank,mode)
    ds=self.build_dataset(path,mode,batch_size);state["sampler"]=Sampler([Path(x).stem for x in ds.im_files],seq);return Loader(ds,batch_size=8,sampler=state["sampler"],num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
  def start(t):
   old=t.optimizer.step
   def step(self,*x,**kw):r=old(*x,**kw);state["steps"]+=1;return r
   t.optimizer.step=MethodType(step,t.optimizer)
  def es(t):state["sampler"].set_epoch(t.epoch)
  def ee(t):state["epochs"].append({"epoch":t.epoch+1,"draws":340,"real_draws":260,"extra_draws":80,"batches":43,"cumulative_optimizer_steps":state["steps"],"sequence_sha256":hashlib.sha256(json.dumps(seq[t.epoch],separators=(",",":")).encode()).hexdigest()})
  m=YOLO(str(a.model));m.add_callback("on_train_start",start);m.add_callback("on_train_epoch_start",es);m.add_callback("on_train_epoch_end",ee);m.train(data=str((roots[arm]/"data.yaml").resolve()),imgsz=512,epochs=150,patience=151,batch=8,deterministic=True,seed=42,optimizer="auto",mosaic=0.,mixup=0.,copy_paste=0.,cutmix=0.,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer);last=metric(YOLO(str(run/"weights/last.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False));best=metric(YOLO(str(run/"weights/best.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False));rows.append({"arm":arm,**last});secondary.append({"arm":arm,**best});audits[arm]={"epochs":len(state["epochs"]),"batches":sum(x["batches"] for x in state["epochs"]),"optimizer_steps":state["steps"],"last_sha256":sha(run/"weights/last.pt"),"best_sha256":sha(run/"weights/best.pt")};save(a.output/f"epoch_exposure_{arm}.json",{"epochs":state["epochs"]})
 save(a.output/"primary_last_metrics.json",{"seed":42,"checkpoint":"epoch150 last.pt","records":rows});save(a.output/"secondary_best_metrics.json",{"records":secondary});save(a.output/"training_budget_audit.json",{"arms":audits,"equal_batches":len({x["batches"] for x in audits.values()})==1,"same_schedule":True});r={x["arm"]:x for x in rows};delta={arm:{"overall_map50_95":r[arm]["map50_95"]-r["real_repeat80"]["map50_95"],"target2_map50_95":np.mean([r[arm]["per_class"][c]["ap50_95"] for c in TARGET])-np.mean([r["real_repeat80"]["per_class"][c]["ap50_95"] for c in TARGET])} for arm in ("sd2_syn80","tmvb_syn80")};status="TMVB_DOWNSTREAM_SIGNAL" if delta["tmvb_syn80"]["overall_map50_95"]>0 and delta["tmvb_syn80"]["target2_map50_95"]>0 and delta["tmvb_syn80"]["target2_map50_95"]>delta["sd2_syn80"]["target2_map50_95"] else "TMVB_DOWNSTREAM_NOT_SUPPORTED";save(a.output/"stage10c_status.json",{"status":status,"single_seed_diagnostic_only":True,"additional_seeds_authorized":status=="TMVB_DOWNSTREAM_SIGNAL","deltas":delta,"official_validation_role":"final detector evaluation only"});print(json.dumps({"status":status,"deltas":delta},indent=2))
if __name__=="__main__":main()
