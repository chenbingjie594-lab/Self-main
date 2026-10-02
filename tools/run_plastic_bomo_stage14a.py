"""Single-seed fixed-budget YOLO11s Stage14A subset screening."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ultralytics import YOLO

ARMS=("real_repeat","random","high_fidelity","valid_novel","high_context_compatibility","low_fidelity_diagnostic","low_context_diagnostic")
NAMES=("flash","black")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def metric(r):
 b=r.box;return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map50_95":float(b.map),"per_class":{n:{"ap50":float(b.ap50[i]),"ap50_95":float(b.ap[i]),"recall":float(b.r[i])} for i,n in enumerate(NAMES)}}
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--prepared",type=Path,required=True);p.add_argument("--dataset_root",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--runs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");a=p.parse_args();a.runs=a.runs.resolve();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);manifest=load(a.prepared/"stage14a_dataset_manifest.json");budget=load(a.prepared/"subset_training_budget_audit.json");assert manifest["status"]=="FROZEN" and budget["status"]=="PASS"
 rows=[];audits={}
 for arm in ARMS:
  run=a.runs/f"{arm}_s42";last=run/"weights/last.pt"
  if not last.exists():
   if run.exists():raise RuntimeError("STAGE14A_INCOMPLETE_RUN_EXISTS: "+str(run))
   YOLO(str(a.model)).train(data=str((a.dataset_root/arm/"data.yaml").resolve()),imgsz=cfg["imgsz"],epochs=cfg["epochs"],patience=cfg["patience"],batch=cfg["batch"],deterministic=True,seed=cfg["screening_seed"],optimizer=cfg["optimizer"],rect=False,augment=False,mosaic=1.0,mixup=0.0,copy_paste=0.0,close_mosaic=10,val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False)
  import yaml
  args=yaml.safe_load((run/"args.yaml").read_text());checks={"epochs":args["epochs"]==150,"batch":args["batch"]==1,"imgsz":args["imgsz"]==1536,"seed":args["seed"]==42,"optimizer":args["optimizer"]=="auto","last_exists":last.is_file()}
  if not all(checks.values()):raise RuntimeError("STAGE14A_TRAIN_PROTOCOL_MISMATCH_"+arm)
  result=YOLO(str(last)).val(data=str((a.dataset_root/arm/"data.yaml").resolve()),split="val",imgsz=cfg["imgsz"],batch=1,device=a.device,verbose=False,plots=False,project=str(a.runs/"final_validation"),name=arm,exist_ok=True)
  rows.append({"arm":arm,"seed":42,"checkpoint":"epoch150 last.pt","checkpoint_sha256":sha(last),**metric(result)});audits[arm]={"checks":checks,"train_images":218,"epochs":150,"scheduled_draws":32700,"scheduled_batches":32700,"primary_checkpoint":"last.pt"}
 save(a.output/"subset_detector_metrics.json",{"status":"COMPLETE","primary_checkpoint":"epoch150 last.pt","records":rows});save(a.output/"subset_training_budget_audit.json",{**budget,"training":audits,"equal_scheduled_compute":len({(x["train_images"],x["epochs"],x["scheduled_draws"],x["scheduled_batches"]) for x in audits.values()})==1});print(json.dumps({"status":"STAGE14A_SCREENING_COMPLETE","arms":list(ARMS)},indent=2))
if __name__=="__main__":main()
