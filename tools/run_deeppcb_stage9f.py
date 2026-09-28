"""Stage9F seed-42 fixed-budget five-arm detector diagnostic."""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
from types import MethodType
CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole");TARGET=("short","pinhole");SEED=42
ARMS=("real_only","real_repeat80","pretrained_noadapt_syn80","finetuned_sd2_syn80","frozen_adapter_syn80")
FIELD={"pretrained_noadapt_syn80":"pretrained_no_adaptation","finetuned_sd2_syn80":"class_finetuned_sd2","frozen_adapter_syn80":"repaired_frozen_adapter"}
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def tree(root,split):
 rows=[]
 for k in ("images","labels"):
  for p in sorted((root/k/split).iterdir()):rows.append([k,p.name,sha(p)])
 return hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest(),len(rows)
def metrics(result):
 b=result.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"recall":float(b.r[i]),"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i])} for i,n in enumerate(CLASSES)}}
def select(rows):
 rng=random.Random(2026);selected=[]
 for cls in TARGET:
  by=defaultdict(list)
  for x in rows:
   if x["class_name"]==cls:by[x["instance_id"]].append(x)
  ids=sorted(by);chosen=rng.sample(ids,40)
  for iid in chosen:selected.append(dict(rng.choice(sorted(by[iid],key=lambda x:int(x["generation_seed"])))))
 return selected
def yolo_label(row):
 m=__import__('numpy').asarray(__import__('PIL.Image',fromlist=['Image']).open(row["mask"]).convert("L"))>127;y,x=__import__('numpy').where(m);w=h=512;xc=(x.min()+x.max()+1)/(2*w);yc=(y.min()+y.max()+1)/(2*h);bw=(x.max()-x.min()+1)/w;bh=(y.max()-y.min()+1)/h;return f"{CLASSES.index(row['class_name'])} {xc:.10f} {yc:.10f} {bw:.10f} {bh:.10f}\n"
def base_files(root):
 ims={p.stem:p for p in (root/"images/train").iterdir()};labs={p.stem:p for p in (root/"labels/train").iterdir()};assert set(ims)==set(labs) and len(ims)==100;return ims,labs
def build(a,selected):
 ims,labs=base_files(a.real_dataset);roots={};label_text=[yolo_label(x) for x in selected]
 for arm in ARMS:
  root=a.dataset_root/arm
  if root.exists():raise RuntimeError("STAGE9F_DATASET_EXISTS_"+arm)
  for k in ("images","labels"):(root/k/"train").mkdir(parents=True,exist_ok=False)
  for j,stem in enumerate(sorted(ims)):
   (root/"images/train"/f"base_{j:03d}{ims[stem].suffix}").symlink_to(ims[stem].resolve());(root/"labels/train"/f"base_{j:03d}.txt").symlink_to(labs[stem].resolve())
  if arm!="real_only":
   for j,row in enumerate(selected):
    image=Path(row["real"] if arm=="real_repeat80" else row[FIELD[arm]]);ext=image.suffix.lower();assert image.is_file();(root/"images/train"/f"extra_{j:03d}{ext}").symlink_to(image.resolve());(root/"labels/train"/f"extra_{j:03d}.txt").write_text(label_text[j])
  (root/"data.yaml").write_text("path: "+str(root.resolve())+"\ntrain: images/train\nval: "+str((a.real_dataset/"images/val").resolve())+"\nnc: 6\nnames: "+json.dumps(list(CLASSES))+"\n");roots[arm]=root
 return roots,label_text
def schedules(arm):
 names=[f"base_{i:03d}" for i in range(100)];extra=[f"extra_{i:03d}" for i in range(80)];out=[]
 for epoch in range(150):
  rng=random.Random(SEED+epoch);base=[rng.choice(names) for _ in range(260)]
  if arm=="real_only":rng.shuffle(base);out.append(base)
  else:e=extra.copy();rng.shuffle(e);seq=base+e;rng.shuffle(seq);out.append(seq)
 return out
class NameSampler:
 def __init__(self,names,seq):self.names=tuple(names);self.index={x:i for i,x in enumerate(names)};self.seq=seq;self.epoch=0;self.observed=[]
 def __len__(self):return len(self.seq[self.epoch])
 def set_epoch(self,e):self.epoch=int(e)
 def __iter__(self):
  self.observed=[]
  for n in self.seq[self.epoch]:i=self.index[n];self.observed.append(i);yield i
def target2(m):return (m["per_class"]["short"]["ap50_95"]+m["per_class"]["pinhole"]["ap50_95"])/2
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","stage9e_manifest","stage9er_status","real_dataset","repo_root","model","reference_run","dataset_root","runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.runs=a.runs.resolve();cfg=load(a.protocol);assert cfg["detector_seed"]==42 and cfg["selection_rng"]==2026 and cfg["epochs"]==150 and cfg["primary_checkpoint"]=="epoch150 last.pt";assert load(a.stage9er_status)["STAGE9F_DIAGNOSTIC_DETECTOR_PILOT_WORTH_TESTING"]
 allrows=load(a.stage9e_manifest)["records"];selected=select(allrows);assert len(selected)==80 and Counter(x["class_name"] for x in selected)==Counter({"short":40,"pinhole":40}) and len({x["instance_id"] for x in selected})==80
 frozen={"selection_rng":2026,"selection_method":"uniform random unique source per class; one uniformly random existing generation seed per source; no metric or score used","records":[{"selection_index":i,**x} for i,x in enumerate(selected)],"count":80,"class_counts":{"short":40,"pinhole":40},"unique_sources":80,"generation_rerun_count":0};save(a.output/"stage9f_frozen_selection_manifest.json",frozen)
 roots,labels=build(a,selected);save(a.output/"source_seed_pairing_audit.json",{"same_80_source_seed_keys_across_synthetic_arms":True,"real_repeat_same_sources":True,"unique_sources":80,"class_counts":dict(Counter(x["class_name"] for x in selected)),"seed_counts":dict(Counter(str(x["generation_seed"]) for x in selected)),"separate_sampling":False});save(a.output/"label_pairing_audit.json",{"synthetic_arm_label_matrix_identical":True,"real_repeat_target_label_matrix_identical":True,"label_sha256":hashlib.sha256(''.join(labels).encode()).hexdigest(),"records":80,"classes":[1,5],"image_resolution":"512x512"})
 import yaml,torch,ultralytics
 from torch.utils.data import DataLoader
 from ultralytics import YOLO
 from ultralytics.data.build import seed_worker
 from ultralytics.models.yolo.detect import DetectionTrainer
 assert ultralytics.__version__=="8.4.145";reference=yaml.safe_load((a.reference_run/"args.yaml").read_text());required={"imgsz":512,"epochs":150,"batch":8,"optimizer":"auto","deterministic":True,"mosaic":0.0,"mixup":0.0,"copy_paste":0.0,"cutmix":0.0};assert all(reference[k]==v for k,v in required.items())
 schedule={arm:schedules(arm) for arm in ARMS};rows=[];bestrows=[];run_audit={};exposure={}
 for arm in ARMS:
  run=a.runs/f"{arm}_s42"
  if run.exists():raise RuntimeError("STAGE9F_RUN_EXISTS_"+arm)
  state={"sampler":None,"epochs":[],"steps":0,"sizes":[]}
  class FiniteLoader(DataLoader):
   def reset(self):return None
  class Trainer(DetectionTrainer):
   def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
    if mode!="train":return super().get_dataloader(dataset_path,batch_size,rank,mode)
    ds=self.build_dataset(dataset_path,mode,batch_size);names=[Path(x).stem for x in ds.im_files];state["sampler"]=NameSampler(names,schedule[arm]);gen=torch.Generator().manual_seed(6148914691236517205+SEED);return FiniteLoader(ds,batch_size=8,sampler=state["sampler"],shuffle=False,num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=gen)
   def preprocess_batch(self,batch):self.consumed.extend(Path(x).stem for x in batch["im_file"]);state["sizes"].append(len(batch["im_file"]));return super().preprocess_batch(batch)
  def start(t):
   old=t.optimizer.step
   def step(self,*x,**kw):r=old(*x,**kw);state["steps"]+=1;return r
   t.optimizer.step=MethodType(step,t.optimizer)
  def estart(t):state["sampler"].set_epoch(t.epoch);t.consumed=[];state["sizes"]=[]
  def eend(t):
   expected=schedule[arm][t.epoch];observed=[state["sampler"].names[i] for i in state["sampler"].observed];batches=33 if arm=="real_only" else 43;last=4;assert t.consumed==expected and observed==expected and state["sizes"]==[8]*(batches-1)+[last];state["epochs"].append({"epoch":t.epoch+1,"draws":len(expected),"real_draws":260,"extra_draws":0 if arm=="real_only" else 80,"synthetic_draws":80 if arm in FIELD else 0,"batches":batches,"cumulative_successful_optimizer_updates_amp":state["steps"],"sequence_sha256":hashlib.sha256(json.dumps(expected,separators=(",",":")).encode()).hexdigest()})
  model=YOLO(str(a.model));model.add_callback("on_train_start",start);model.add_callback("on_train_epoch_start",estart);model.add_callback("on_train_epoch_end",eend);model.train(data=str((roots[arm]/"data.yaml").resolve()),imgsz=512,epochs=150,patience=151,batch=8,deterministic=True,seed=42,optimizer="auto",mosaic=0.,mixup=0.,copy_paste=0.,cutmix=0.,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer)
  args=yaml.safe_load((run/"args.yaml").read_text());assert all(args[k]==v for k,v in {**required,"patience":151,"seed":42}.items()) and len(state["epochs"])==150 and (run/"weights/last.pt").is_file() and (run/"weights/best.pt").is_file();last=metrics(YOLO(str(run/"weights/last.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False));best=metrics(YOLO(str(run/"weights/best.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False));last["target2_map50_95"]=target2(last);best["target2_map50_95"]=target2(best);rows.append({"arm":arm,**last});bestrows.append({"arm":arm,**best});run_audit[arm]={"epochs":150,"audited_batches":sum(x["batches"] for x in state["epochs"]),"expected_batches":4950 if arm=="real_only" else 6450,"successful_optimizer_updates_amp":state["steps"],"last_checkpoint_sha256":sha(run/"weights/last.pt"),"best_checkpoint_sha256":sha(run/"weights/best.pt"),"config":{k:args[k] for k in set(required)|{"patience","seed"}}};save(a.output/f"epoch_exposure_{arm}.json",{"epochs":state["epochs"]})
 by={x["arm"]:x for x in rows};bb={x["arm"]:x for x in bestrows};assert set(by)==set(ARMS);save(a.output/"primary_last_metrics.json",{"checkpoint":"epoch150 last.pt","seed":42,"records":rows});save(a.output/"secondary_best_metrics.json",{"checkpoint":"best.pt secondary only","seed":42,"records":bestrows});save(a.output/"training_run_audit.json",run_audit)
 freq={}
 for arm in ARMS:
  source_frequency=Counter(n for epoch in schedule[arm] for n in epoch);freq[arm]={"images_per_epoch":260 if arm=="real_only" else 340,"epochs":150,"total_batches":run_audit[arm]["audited_batches"],"real_exposures":39000,"extra_exposures":0 if arm=="real_only" else 12000,"synthetic_exposures":12000 if arm in FIELD else 0,"source_frequency":dict(sorted(source_frequency.items())),"extra_source_frequency":{} if arm=="real_only" else {f"extra_{i:03d}":150 for i in range(80)},"extra_class_frequency":{} if arm=="real_only" else {"short":6000,"pinhole":6000}}
 save(a.output/"fixed_budget_exposure_audit.json",{"arms":freq,"four_340_sample_arms_equal_exposure":True,"four_340_sample_arms_equal_batches":True,"four_augmented_arms_identical_source_frequency":len({json.dumps(freq[x]["source_frequency"],sort_keys=True) for x in ARMS[1:]})==1,"real_only_protocol":"260 real draws; 33 batches/epoch","augmented_protocol":"260 real + 80 extra; 43 batches/epoch"})
 save(a.output/"per_class_metrics.json",{"primary":{arm:by[arm]["per_class"] for arm in ARMS},"secondary_best":{arm:bb[arm]["per_class"] for arm in ARMS}});save(a.output/"target2_metrics.json",{"preregistered":True,"definition":"mean(short AP50-95, pinhole AP50-95)","primary":{arm:by[arm]["target2_map50_95"] for arm in ARMS},"secondary_best":{arm:bb[arm]["target2_map50_95"] for arm in ARMS}})
 comparisons={"synthetic_content_gain":{arm:{m:by[arm][m]-by["real_repeat80"][m] for m in ("map50_95","target2_map50_95")} for arm in FIELD},"fidelity_vs_variation":{m:by["frozen_adapter_syn80"][m]-by["finetuned_sd2_syn80"][m] for m in ("map50_95","target2_map50_95")},"semantic_adaptation_necessity":{m:by["frozen_adapter_syn80"][m]-by["pretrained_noadapt_syn80"][m] for m in ("map50_95","target2_map50_95")},"single_seed_diagnostic_only":True};save(a.output/"comparison_deltas.json",comparisons)
 f,sd,rr=by["frozen_adapter_syn80"],by["finetuned_sd2_syn80"],by["real_repeat80"];collapse=f["map50_95"]<.9*rr["map50_95"]
 if f["target2_map50_95"]>sd["target2_map50_95"] and f["target2_map50_95"]>rr["target2_map50_95"] and not collapse:status="FROZEN_VARIATION_UTILITY_SIGNAL"
 elif sd["target2_map50_95"]>f["target2_map50_95"] and sd["map50_95"]>f["map50_95"]:status="HIGH_FIDELITY_SD2_BETTER"
 elif not (sd["target2_map50_95"]>rr["target2_map50_95"] and sd["map50_95"]>rr["map50_95"]) and not (f["target2_map50_95"]>rr["target2_map50_95"] and f["map50_95"]>rr["map50_95"]):status="SYNTHETIC_GAIN_NOT_SUPPORTED"
 else:status="STAGE9F_MIXED"
 save(a.output/"stage9f_protocol.json",cfg);save(a.output/"stage9f_status.json",{"status":status,"seed":42,"primary_checkpoint":"epoch150 last.pt","obvious_overall_collapse":collapse,"additional_three_seed_confirmation_authorized":status=="FROZEN_VARIATION_UTILITY_SIGNAL","single_seed_diagnostic_only":True,"historical_stage9e_gate_modified":False,"generator_training_count":0,"selection_metric_use_count":0,"official_validation_role":"detector evaluation only"})
 direction=any((bb[a0]["target2_map50_95"]-bb[a1]["target2_map50_95"])*(by[a0]["target2_map50_95"]-by[a1]["target2_map50_95"])<0 for a0,a1 in (("frozen_adapter_syn80","finetuned_sd2_syn80"),("frozen_adapter_syn80","real_repeat80")))
 lines=["# DeepPCB Stage9F Fixed-Budget Downstream Utility Pilot","",f"Status: **{status}**.","Single detector seed42 diagnostic only; no significance or stability claim.","Primary checkpoint is epoch150 last.pt; best.pt is secondary.","", "| arm | overall mAP50-95 | Target2 mAP50-95 | short AP | pinhole AP |","|---|---:|---:|---:|---:|"]+[f"| {arm} | {by[arm]['map50_95']:.6f} | {by[arm]['target2_map50_95']:.6f} | {by[arm]['per_class']['short']['ap50_95']:.6f} | {by[arm]['per_class']['pinhole']['ap50_95']:.6f} |" for arm in ARMS]+["",f"Best.pt reverses a primary comparison direction: **{direction}**.","No selector, generator retraining, detector tuning, or validation-driven decision was used."];(a.output/"STAGE9F_REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps({"status":status,"collapse":collapse},indent=2))
if __name__=="__main__":main()
