"""Run Stage14B three-seed confirmation using final epoch last.pt only."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ultralytics import YOLO
from audit_plastic_bomo_stage14b import load,save
ARMS=("real_repeat","random","high","low");SEEDS=(42,3407,2026);NAMES=("flash","black")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(r):
 b=r.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i]),"recall":float(b.r[i])} for i,n in enumerate(NAMES)}}
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--attribution",type=Path,required=True);p.add_argument("--stage14a",type=Path,required=True);p.add_argument("--dataset_root",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--runs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();cfg=load(a.protocol);definition=load(a.attribution/"matched_subset_definition.json");matched=definition["status"]=="FROZEN_MATCHED_SUBSETS";rows=[];validation=[]
 old={x["arm"]:x for x in load(a.stage14a/"subset_detector_metrics.json")["records"]}
 reuse={"real_repeat":"real_repeat","random":"random","high":"high_fidelity","low":"low_fidelity_diagnostic"}
 for seed in SEEDS:
  for arm in ARMS:
   if seed==42 and not matched:rows.append({**old[reuse[arm]],"arm":arm,"reused_stage14a":True});validation.append({"arm":arm,"seed":seed,"reused_stage14a":True,"training_validation_uses":0,"final_last_evaluation_uses":1,"best_checkpoint_used":False});continue
   run=a.runs/f"{arm}_s{seed}";last=run/"weights/last.pt"
   if not last.exists():
    if run.exists():raise RuntimeError("STAGE14B_INCOMPLETE_RUN_EXISTS: "+str(run))
    YOLO(str(a.model)).train(data=str((a.dataset_root/arm/"data.yaml").resolve()),imgsz=1536,epochs=150,patience=151,batch=1,deterministic=True,seed=seed,optimizer="auto",rect=False,augment=False,mosaic=1.,mixup=0.,copy_paste=0.,close_mosaic=10,val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False)
   result=YOLO(str(last)).val(data=str((a.dataset_root/arm/"data.yaml").resolve()),split="val",imgsz=1536,batch=1,device=a.device,verbose=False,plots=False,project=str(a.runs/"final_validation"),name=f"{arm}_s{seed}",exist_ok=True);rows.append({"arm":arm,"seed":seed,"checkpoint":"epoch150 last.pt","checkpoint_sha256":sha(last),"reused_stage14a":False,**metric(result)});validation.append({"arm":arm,"seed":seed,"reused_stage14a":False,"training_validation_uses":0,"final_last_evaluation_uses":1,"best_checkpoint_used":False})
 for seed in SEEDS:save(a.output/f"seed{seed}_metrics.json",{"records":[x for x in rows if x["seed"]==seed]})
 budget=load(a.output/"training_budget_audit.json");budget["training"]={f"{x['arm']}_s{x['seed']}":{"epochs":150,"base_images":138,"extra_images":80,"images_per_epoch":218,"scheduled_draws":32700,"scheduled_batches":32700,"primary_checkpoint":"last.pt","reused_stage14a":x["reused_stage14a"]} for x in validation};budget["equal_scheduled_compute"]=True;save(a.output/"training_budget_audit.json",budget);save(a.output/"validation_usage_audit.json",{"status":"FINAL_LAST_ONLY","records":validation,"official_validation_subset_selection_uses":0,"official_validation_threshold_tuning_uses":0});save(a.output/"all_seed_metrics.json",{"records":rows});print(json.dumps({"status":"STAGE14B_TRAINING_COMPLETE","matched":matched,"runs_trained":sum(not x['reused_stage14a'] for x in validation)},indent=2))
if __name__=="__main__":main()
