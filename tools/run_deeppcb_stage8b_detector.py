"""Stage8B-B: authorized seed42 NoMorph detector pilot with Stage7C controls."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,subprocess,sys
from collections import defaultdict
from pathlib import Path
import yaml
from run_deeppcb_stage2_pool import CLASSES,link_or_copy,prepare_batch
from run_deeppcb_stage5b import load,metrics,sha
from run_deeppcb_stage5c import best_epoch,save,transforms_audit,walk_transforms
from run_deeppcb_stage5e import finite_metrics
from run_deeppcb_stage6b import file_population
from run_deeppcb_stage6g import schedule_for_seed
from run_deeppcb_stage6h import checkpoint_epoch
from run_deeppcb_stage7c import PairedSampler7C,frozen_selection
SEED=42
def fail(x,m):
 if not x:raise RuntimeError("STAGE8B_DETECTOR_"+m)
def npath(root,x):return root/x["class_name"]/("nomorph_"+x["candidate_id"]+".jpg")
def generate(tasks,a):
 view=a.work_root/"checkpoint_view/DeepPCB";view.mkdir(parents=True,exist_ok=True)
 for c in CLASSES:
  dst=view/c
  if not dst.exists():os.symlink((a.nomorph_models/c).resolve(),dst,target_is_directory=True)
 cfg={"experiment_name":"deeppcb_stage8b_nomorph_detector_pool","pipeline_mode":"custom","seed":42,"device":"cuda","dtype":"float16","prompt":"a photo of a sks defect","negative_prompt":None,"num_inference_steps":50,"guidance_scale":7.5,"blur_factor":0,"normal_filter":{"enabled":False},"modules":{"prompt_perturbation":{"enabled":False},"spatial_guidance":{"enabled":False},"cama":{"enabled":False},"ddim_noise":{"enabled":False},"mdap":{"enabled":False},"rda":{"enabled":False},"carf":{"enabled":False},"msdf":{"enabled":True,"root":str((a.work_root/"checkpoint_view").resolve()),"filename":"msdf.pt"}}};cfgp=a.work_root/"config.json";cfgp.write_text(json.dumps(cfg,indent=2));groups=defaultdict(list)
 for x in tasks:groups[(x["class_name"],int(x["generation_seed"]))].append(x)
 for (cls,seed),batch in groups.items():
  if all(npath(a.pool_root,x).is_file() for x in batch):continue
  br=prepare_batch(batch,cls,seed,a.work_root/"batches");gen=br/"generated";cmd=[sys.executable,"inference.py","--model_ckpt_root",str(a.work_root/"checkpoint_view"),"--ddim_scheduler_root",str(a.scheduler),"--config",str(cfgp),"--categories","DeepPCB","--base_dir",str(br/"input"),"--reference_base_dir",str(br/"references"),"--use_paired_normal","--dataset_type","mvtec","--output_name",str(gen),"--seed",str(seed)]
  with (a.logs/f"generate_{cls}_s{seed}.log").open("w") as f:rc=subprocess.run(cmd,cwd=a.repo_root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,"CUDA_VISIBLE_DEVICES":a.device}).returncode
  fail(rc==0,f"GENERATE_{cls}_{seed}");src=gen/cfg["experiment_name"]/"DeepPCB"/cls/"image"
  for i,x in enumerate(batch):link_or_copy(src/f"{i}.jpg",npath(a.pool_root,x))
def main():
 p=argparse.ArgumentParser()
 for n in ("stage8b","formal_selection","sd2_manifest","full_manifest","nomorph_models","scheduler","repo_root","real_dataset","model","stage7c_results","stage7c_runs","pool_root","work_root","dataset_root","runs","output","logs"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True);a.work_root.mkdir(parents=True,exist_ok=True)
 gate=load(a.stage8b/"stage8b_generator_gate.json");fail(gate["status"]=="NOMORPH_REPAIR_SUPPORTED" and gate["detector_pilot_authorized"],"GATE")
 matched,_,_=frozen_selection(a.formal_selection,a.sd2_manifest,a.full_manifest);pool={(x["target_instance_id"],int(x["generation_seed"])):x for x in load(a.full_manifest)["records"]};keys=[(x["source_instance_id"],int(x["generation_seed"])) for x in matched["records"]];tasks=[pool[k] for k in keys];generate(tasks,a)
 root=a.dataset_root;fail(not root.exists(),"DATASET_EXISTS");real_images,real_labels=file_population(a.real_dataset)
 for kind in ("images","labels"):(root/kind/"train").mkdir(parents=True,exist_ok=False)
 for stem,image in real_images.items():(root/"images/train"/image.name).symlink_to(image.resolve());(root/"labels/train"/real_labels[stem].name).symlink_to(real_labels[stem].resolve())
 for i,x in enumerate(tasks):
  image=npath(a.pool_root,x);label=Path(x["label_path"]);label=label if label.is_absolute() else a.repo_root/label;fail(image.is_file() and label.is_file(),"SYNTHETIC");(root/"images/train"/f"synthetic_{i:03d}.jpg").symlink_to(image.resolve());(root/"labels/train"/f"synthetic_{i:03d}.txt").symlink_to(label.resolve())
 (root/"data.yaml").write_text("path: "+str(root.resolve())+"\ntrain: images/train\nval: "+str((a.real_dataset/"images/val").resolve())+"\nnc: 6\nnames: "+json.dumps(list(CLASSES))+"\n")
 names=[f"synthetic_{i:03d}" for i in range(240)];schedule=schedule_for_seed(sorted(real_images),names,SEED);state={"sampler":None,"epochs":[],"sizes":[]}
 import torch,ultralytics
 from torch.utils.data import DataLoader
 from ultralytics import YOLO
 from ultralytics.data.build import seed_worker
 from ultralytics.models.yolo.detect import DetectionTrainer
 fail(ultralytics.__version__=="8.4.145","VERSION")
 class FiniteLoader(DataLoader):
  def reset(self):return None
 class Trainer(DetectionTrainer):
  def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
   if mode!="train":return super().get_dataloader(dataset_path,batch_size,rank,mode)
   ds=self.build_dataset(dataset_path,mode,batch_size);state["sampler"]=PairedSampler7C([Path(x).stem for x in ds.im_files],schedule);state["transforms"]=transforms_audit(ds.transforms);gen=torch.Generator().manual_seed(6148914691236517205+SEED);return FiniteLoader(ds,batch_size=8,sampler=state["sampler"],shuffle=False,num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=gen)
  def preprocess_batch(self,batch):self.consumed.extend(Path(x).stem for x in batch["im_file"]);state["sizes"].append(len(batch["im_file"]));return super().preprocess_batch(batch)
 def start(t):state["sampler"].set_epoch(t.epoch);t.consumed=[];state["sizes"]=[]
 def end(t):
  exp=schedule[t.epoch];obs=[state["sampler"].names[i] for i in state["sampler"].observed];fail(t.consumed==exp and obs==exp and state["sizes"]==[8]*42+[4],"EXPOSURE");state["epochs"].append({"epoch":t.epoch+1,"batches":43,"real_draws":260,"synthetic_draws":80,"schedule_sha256":hashlib.sha256(json.dumps(exp,separators=(",",":")).encode()).hexdigest()});save(a.output/"nomorph_epoch_exposure_s42.json",{"epochs":state["epochs"]})
 run=a.runs/"nomorph_s42";fail(not run.exists(),"RUN_EXISTS");model=YOLO(str(a.model));model.add_callback("on_train_epoch_start",start);model.add_callback("on_train_epoch_end",end);model.train(data=str((root/"data.yaml").resolve()),imgsz=512,epochs=150,patience=151,batch=8,deterministic=True,seed=42,optimizer="auto",mosaic=0.0,mixup=0.0,copy_paste=0.0,cutmix=0.0,device=a.device,project=str(a.runs.resolve()),name=run.name,exist_ok=False,trainer=Trainer)
 fail(len(state["epochs"])==150 and checkpoint_epoch(run/"weights/last.pt") in (-1,149),"BUDGET");score=metrics(YOLO(str(run/"weights/last.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False));finite_metrics(score)
 old=load(a.stage7c_results/"per_seed_metrics.json")["records"];controls={x["arm"]:x for x in old if int(x["seed"])==42};fail(set(controls)=={"sd2","msdf"},"CONTROL_RESULTS")
 reuse={"stage7c_derived_selection_sha256":sha(a.stage7c_results/"source_seed_selection_manifest.json"),"formal_selection_sha256":sha(a.formal_selection),"same_240_source_seed_keys":True,"note":"Hashes differ because one file is the derived cross-generator mapping and the other is the source selector manifest.","same_real_train":True,"same_schedule_seed42":True,"same_detector_protocol":True,"controls_reused":True,"sd2_last_checkpoint":controls["sd2"]["last_checkpoint_sha256"],"full_last_checkpoint":controls["msdf"]["last_checkpoint_sha256"]};save(a.output/"detector_control_reuse_audit.json",reuse)
 result={"training_seed":42,"primary":"epoch150 last.pt","sd2":controls["sd2"],"full_msdf":controls["msdf"],"nomorph":score};save(a.output/"detector_seed42_metrics.json",result);budget={"nomorph":{"epochs":150,"batches":6450,"real_draws":39000,"synthetic_draws":12000},"controls":"Stage7C audited identical seed42 budget","equal_compute_protocol":True};save(a.output/"detector_training_budget_audit.json",budget)
 d_full=score["map50_95"]-controls["msdf"]["map50_95"];d_sd2=score["map50_95"]-controls["sd2"]["map50_95"];status="NOMORPH_DETECTOR_SEED42_SUPPORTED" if d_full>0 and d_sd2>=0 else "NOMORPH_DETECTOR_SEED42_MIXED" if d_full>0 else "NOMORPH_DETECTOR_SEED42_NOT_SUPPORTED";authorized=status=="NOMORPH_DETECTOR_SEED42_SUPPORTED";save(a.output/"stage8b_detector_gate.json",{"status":status,"nomorph_minus_full":d_full,"nomorph_minus_sd2":d_sd2,"authorize_seeds_3407_2026":authorized});save(a.output/"stage8b_status.json",{"status":"MSDF_REPAIR_SEED42_SUPPORTED" if authorized else "MSDF_REPAIR_NOT_SUPPORTED","generator_gate":"NOMORPH_REPAIR_SUPPORTED","detector_gate":status,"further_detector_seeds_authorized":authorized,"new_module_design_authorized":False});print(json.dumps({"status":status,"d_full":d_full,"d_sd2":d_sd2},indent=2))
if __name__=="__main__":main()
