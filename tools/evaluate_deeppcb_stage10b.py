"""Evaluate Stage10B TMVB band behavior on frozen OOF features; no detector training."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from stage10b_differentiable_features import DifferentiableDetectInputExtractor

CLASSES=("short","pinhole");ARMS=("sd2","tmvb")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def summarize(rows,bl,bu):
 z=np.asarray([x["normalized_pair_gap"] for x in rows]);v=np.maximum(bl-z,0)**2+np.maximum(z-bu,0)**2
 return {"n":len(rows),"mean_normalized_pair_gap":float(z.mean()),"median_normalized_pair_gap":float(np.median(z)),"below_band_fraction":float((z<bl).mean()),"inside_band_fraction":float(((z>=bl)&(z<=bu)).mean()),"above_band_fraction":float((z>bu).mean()),"mean_squared_band_violation":float(v.mean()),"median_squared_band_violation":float(np.median(v))}
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","tmvb_manifest","sd2_manifest","folds","stage3r","stage3r_runs","stage10ar","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);bl,bu=cfg["beta_L"],cfg["beta_U"]
 tm=load(a.tmvb_manifest)["records"];sd=load(a.sd2_manifest)["records"];sdmap={(x["instance_id"],int(x["generation_seed"])):x for x in sd};assert len(tm)==774 and all((x["instance_id"],int(x["generation_seed"])) in sdmap for x in tm)
 foldof={pid:int(f["fold"]) for f in load(a.folds)["folds"] for pid in f["holdout_pair_ids"]};rad={x["instance_id"]:x["k5_radius"] for x in load(a.stage10ar/"pair_independent_local_radius.json")["records"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");pos={x["instance_id"]:(int(f["fold"]),int(x["feature_index"])) for f in bankdoc["folds"] for x in f["records"]};records=[]
 for fold in range(3):
  teacher=DifferentiableDetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512);bank=np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz")["features"]
  try:
   for x in [r for r in tm if foldof[r["pair_id"]]==fold]:
    key=(x["instance_id"],int(x["generation_seed"]));paths={"tmvb":x["image_path"],"sd2":sdmap[key]["image_path"]};target=bank[pos[x["instance_id"]][1]]
    for arm,path in paths.items():
     feat=teacher.encode_path(path,__import__('PIL.Image',fromlist=['Image']).open(x["mask_path"]).getbbox()).detach().cpu().numpy()[0];d=float(1-np.dot(feat,target)/(np.linalg.norm(feat)*np.linalg.norm(target)));z=d/rad[x["instance_id"]];records.append({"arm":arm,"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":x["class_name"],"generation_seed":int(x["generation_seed"]),"fold":fold,"distance":d,"radius":rad[x["instance_id"]],"normalized_pair_gap":z,"band_state":"below" if z<bl else "above" if z>bu else "inside"})
  finally:teacher.close()
 metrics={arm:{"overall":summarize([x for x in records if x["arm"]==arm],bl,bu),"per_class":{c:summarize([x for x in records if x["arm"]==arm and x["class_name"]==c],bl,bu) for c in CLASSES}} for arm in ARMS};delta={c:{"inside_band_fraction_delta":metrics["tmvb"]["per_class"][c]["inside_band_fraction"]-metrics["sd2"]["per_class"][c]["inside_band_fraction"],"mean_squared_violation_delta":metrics["tmvb"]["per_class"][c]["mean_squared_band_violation"]-metrics["sd2"]["per_class"][c]["mean_squared_band_violation"]} for c in CLASSES};improved={c:delta[c]["inside_band_fraction_delta"]>0 and delta[c]["mean_squared_violation_delta"]<0 for c in CLASSES};status="TMVB_BAND_CONTROL_SUPPORTED" if all(improved.values()) else "TMVB_BAND_CONTROL_MIXED" if any(improved.values()) else "TMVB_BAND_CONTROL_NOT_SUPPORTED";save(a.output/"band_metrics.json",{"band":{"lower":bl,"upper":bu},"metrics":metrics,"paired_class_delta":delta,"raw_records":records});save(a.output/"stage10b_generation_audit.json",{"status":status,"exact_source_seed_pairing":True,"records_per_arm":774,"per_class_per_arm":{"short":387,"pinhole":387},"seed_counts_per_arm":dict(Counter(str(x["generation_seed"]) for x in tm)),"selector_used":False,"official_validation_use_count":0,"detector_training_count":0});gate=load(a.output.parent/"deeppcb_stage10b_tmvb_feasibility"/"stage10b_gate.json");gate.update({"status":status,"GENERATION_EVALUATION_COMPLETE":True,"STAGE10C_DETECTOR_PILOT_AUTHORIZED":status=="TMVB_BAND_CONTROL_SUPPORTED"});save(a.output/"stage10b_gate_after_generation.json",gate);print(json.dumps({"status":status,"delta":delta},indent=2))
if __name__=="__main__":main()
