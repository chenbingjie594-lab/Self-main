"""Stage10B Gate-1: feature equivalence, radius lookup, and frozen t_band."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from dwbg_feature_extraction import DetectInputExtractor
from stage10b_differentiable_features import DifferentiableDetectInputExtractor

CLASSES=("short","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def bbox(mask):
 m=np.asarray(Image.open(mask).convert("L"))>127;y,x=np.where(m);return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","stage10ar","registry","folds","stage3r","stage3r_runs","base_model","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);gate=load(a.stage10ar/"stage10ar_status.json");assert gate["STAGE10B_GENERATOR_FEASIBILITY_AUTHORIZED"] and cfg["band"]=="medium"
 bands={x["name"]:x for x in load(a.stage10ar/"pair_independent_trust_region_candidates.json")["candidates"]};medium=bands["medium"];assert medium["lower_beta"]==cfg["beta_L"] and medium["upper_beta"]==cfg["beta_U"]
 radii=load(a.stage10ar/"pair_independent_local_radius.json")["records"];radius={x["instance_id"]:x for x in radii};registry=[x for x in load(a.registry)["instances"] if x["class_name"] in CLASSES];foldof={p:int(f["fold"]) for f in load(a.folds)["folds"] for p in f["holdout_pair_ids"]};assert len(registry)==258
 rr=[]
 for x in registry:
  r=radius[x["instance_id"]];ok=r["pair_id"]==x["pair_id"] and r["class_name"]==x["class_name"] and r["fold"]==foldof[x["pair_id"]] and r["k5_radius"]>0;rr.append({"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":x["class_name"],"fold":foldof[x["pair_id"]],"k5_radius":r["k5_radius"],"matched":ok})
 coverage={c:sum(x["class_name"]==c and x["matched"] for x in rr) for c in CLASSES};radius_ok=coverage=={"short":129,"pinhole":129};save(a.output/"radius_lookup_audit.json",{"status":"PASS" if radius_ok else "FAIL","source":"Stage10A-R only","coverage":coverage,"expected":{"short":129,"pinhole":129},"global_or_class_radius_used":False,"records":rr})
 import torch
 from diffusers import DDPMScheduler
 scheduler=DDPMScheduler.from_pretrained(a.base_model,subfolder="scheduler");alpha=scheduler.alphas_cumprod.float();snr=alpha/(1-alpha);t=int(torch.argmin(torch.abs(torch.log(snr))).item());save(a.output/"t_band_protocol.json",{"rule":cfg["t_band_rule"],"timestep":t,"alpha_bar":float(alpha[t]),"snr":float(snr[t]),"log_snr":float(torch.log(snr[t])),"prediction_type":scheduler.config.prediction_type,"manually_selected":False})
 bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");bankrec={x["instance_id"]:(int(f["fold"]),int(x["feature_index"])) for f in bankdoc["folds"] for x in f["records"]};records=[]
 for c in CLASSES:
  chosen=sorted([x for x in registry if x["class_name"]==c],key=lambda x:x["instance_id"])[:cfg["feature_equivalence_samples_per_class"]]
  byfold={}
  for x in chosen:byfold.setdefault(foldof[x["pair_id"]],[]).append(x)
  for fold,items in byfold.items():
   hist=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512);diff=DifferentiableDetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512);z=np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz");features=z["features"]
   try:
    for x in items:
     box=x["raw_bbox"];old=hist.encode(x["raw_image"],box,x["width"],x["height"]);new=diff.encode_path(x["raw_image"],box).detach().float().cpu().numpy()[0];stored=features[bankrec[x["instance_id"]][1]];cos=float(np.dot(old,new)/(np.linalg.norm(old)*np.linalg.norm(new)));stored_cos=float(np.dot(stored,new)/(np.linalg.norm(stored)*np.linalg.norm(new)));records.append({"instance_id":x["instance_id"],"class_name":c,"fold":fold,"cosine_historical_encode_vs_differentiable":cos,"cosine_stored_bank_vs_differentiable":stored_cos,"max_absolute_feature_difference":float(np.max(np.abs(old-new))),"pass":cos>=cfg["feature_equivalence_cosine_minimum"] and stored_cos>=cfg["feature_equivalence_cosine_minimum"]})
   finally:hist.close();diff.close()
 minimum=min(min(x["cosine_historical_encode_vs_differentiable"],x["cosine_stored_bank_vs_differentiable"]) for x in records);ok=len(records)==60 and all(x["pass"] for x in records);audit={"status":"DIFFERENTIABLE_FEATURE_PROTOCOL_EQUIVALENT" if ok else "DIFFERENTIABLE_FEATURE_PROTOCOL_MISMATCH","threshold":cfg["feature_equivalence_cosine_minimum"],"samples":len(records),"per_class":{c:sum(x["class_name"]==c for x in records) for c in CLASSES},"mean_cosine":float(np.mean([x["cosine_historical_encode_vs_differentiable"] for x in records])),"minimum_cosine":minimum,"max_absolute_feature_difference":max(x["max_absolute_feature_difference"] for x in records),"all_samples_pass":ok,"records":records};save(a.output/"differentiable_feature_equivalence_audit.json",audit)
 protocol={**cfg,"stage10ar_status":gate["status"],"frozen_band_record":medium,"gate1_pass":ok and radius_ok,"official_validation_use_count":0,"synthetic_generation_count":0,"detector_training_count":0,"generator_training_count":0};save(a.output/"stage10b_protocol.json",protocol);save(a.output/"stage10b_gate.json",{"status":"GATE1_PASSED_READY_FOR_GRADIENT_TEST" if ok and radius_ok else audit["status"],"GATE1_PASSED":ok and radius_ok,"FORMAL_TRAINING_AUTHORIZED":False,"STAGE10C_DETECTOR_PILOT_AUTHORIZED":False})
 print(json.dumps({"feature_status":audit["status"],"minimum_cosine":minimum,"radius_status":radius_ok,"t_band":t},indent=2))
 if not(ok and radius_ok):raise RuntimeError(audit["status"])
if __name__=="__main__":main()
