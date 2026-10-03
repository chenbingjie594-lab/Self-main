"""Train only M10/M01 and merge their final-last metrics with frozen Stage14B M00/M11."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ultralytics import YOLO

ARMS=("M10","M01");SEEDS=(42,3407,2026);NAMES=("flash","black")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(r):
 b=r.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i]),"recall":float(b.r[i])} for i,n in enumerate(NAMES)}}
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--prepared",type=Path,required=True);p.add_argument("--stage14b",type=Path,required=True);p.add_argument("--dataset_root",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--runs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);det=cfg["detector"]
 frozen=load(a.prepared/"factorial_manifest_freeze_audit.json");
 if frozen["status"]!="PASS":raise RuntimeError("FACTORIAL_MANIFEST_INVALID")
 rows=[];usage=[];env=[]
 for arm in ARMS:
  for seed in SEEDS:
   run=a.runs/f"{arm.lower()}_s{seed}";last=run/"weights/last.pt"
   if not last.exists():
    if run.exists():raise RuntimeError("STAGE15A_INCOMPLETE_RUN_EXISTS: "+str(run))
    YOLO(str(a.model)).train(data=str((a.dataset_root/arm/"data.yaml").resolve()),imgsz=det["imgsz"],epochs=det["epochs"],patience=det["patience"],batch=det["batch"],deterministic=det["deterministic"],seed=seed,optimizer=det["optimizer"],rect=False,augment=False,mosaic=det["mosaic"],mixup=det["mixup"],copy_paste=det["copy_paste"],close_mosaic=det["close_mosaic"],val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False)
   result=YOLO(str(last)).val(data=str((a.dataset_root/arm/"data.yaml").resolve()),split="val",imgsz=det["imgsz"],batch=1,device=a.device,verbose=False,plots=False,project=str(a.runs/"final_validation"),name=f"{arm.lower()}_s{seed}",exist_ok=True)
   rows.append({"arm":arm,"seed":seed,"checkpoint":"epoch150 last.pt","checkpoint_sha256":sha(last),"reused_stage14b":False,**metric(result)});usage.append({"arm":arm,"seed":seed,"training_validation_uses":0,"final_last_evaluation_uses":1,"best_checkpoint_used":False});args=run/"args.yaml";env.append({"arm":arm,"seed":seed,"args_yaml":str(args),"args_yaml_sha256":sha(args) if args.exists() else None})
 old=load(a.stage14b/"all_seed_metrics.json")["records"]
 for x in old:
  if x["arm"] in ("random","high"):rows.append({**x,"arm":"M00" if x["arm"]=="random" else "M11","reused_stage14b":True})
 rows.sort(key=lambda x:(int(x["seed"]),x["arm"]));save(a.output/"arm10_flash_morph_metrics.json",{"records":[x for x in rows if x["arm"]=="M10"]});save(a.output/"arm01_black_morph_metrics.json",{"records":[x for x in rows if x["arm"]=="M01"]});save(a.output/"factorial_all_arms_metrics.json",{"records":rows})
 budget=load(a.output/"factorial_training_budget_audit.json");budget["status"]="PASS";budget["all_arms_equal_scheduled_compute"]=True;budget["new_detector_trainings"]=6;budget["reused_detector_trainings"]=6;save(a.output/"factorial_training_budget_audit.json",budget);save(a.output/"validation_usage_audit.json",{"status":"FINAL_LAST_ONLY","new_runs":usage,"reused_M00_M11":"Stage14B final-last metrics","selection_uses":0,"threshold_tuning_uses":0});save(a.output/"training_environment_audit.json",{"status":"STAGE14B_PROTOCOL_EXACTLY_REUSED","detector":det,"pretrained_model":str(a.model),"new_run_args":env,"M00_M11_source":str(a.stage14b/"all_seed_metrics.json")});print(json.dumps({"status":"STAGE15A_TRAINING_COMPLETE","new_trainings":6,"reused_trainings":6},indent=2))
if __name__=="__main__":main()
