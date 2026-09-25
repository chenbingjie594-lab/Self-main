"""Stage8B-A matched three-generator gate; no detector training."""
from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import numpy as np
from PIL import Image
from dwbg_feature_extraction import DetectInputExtractor
from dwbg_v2_utils import l2_normalize
from evaluate_deeppcb_stage7b import crop_box,edge_cos,ssim_value
from evaluate_deeppcb_stage8a import boundary,mask_and_bbox,stat
from run_deeppcb_stage2_pool import CLASSES
from run_deeppcb_stage5b import load
from run_deeppcb_stage5c import save
ARMS=("sd2","full","nomorph")
def fail(x,m):
 if not x:raise RuntimeError("STAGE8B_EVAL_"+m)
def valid(values):return [float(x) for x in values if x is not None and np.isfinite(x)]
def summary(rows,fields):return {"overall":{f:{**stat(valid([x[f] for x in rows])),"missing":len(rows)-len(valid([x[f] for x in rows]))} for f in fields},"per_class":{c:{f:{**stat(valid([x[f] for x in rows if x["class_name"]==c])),"missing":sum(x["class_name"]==c for x in rows)-len(valid([x[f] for x in rows if x["class_name"]==c]))} for f in fields} for c in CLASSES}}
def main():
 p=argparse.ArgumentParser()
 for n in ("stage8b","folds","stage3r","stage3r_runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");p.add_argument("--batch",type=int,default=8);a=p.parse_args();records=load(a.stage8b/"source_pairing_audit.json")["records"];fail(len(records)==775,"MATRIX")
 import torch
 from torchvision import transforms
 from evaluation.compare_mask_crop_lpips import lpips
 dev=torch.device("cuda:"+a.device if torch.cuda.is_available() else "cpu");tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);model=lpips.LPIPS(net="alex",verbose=False).to(dev).eval();pix=[];geo=[]
 with torch.no_grad():
  for arm in ARMS:
   for start in range(0,len(records),a.batch):
    sub=records[start:start+a.batch];real=[];syn=[];meta=[]
    for x in sub:
     box,_=crop_box(x["mask"]);r=Image.open(x["real"]).convert("RGB").crop(box);s=Image.open(x[arm]).convert("RGB").crop(box);real.append(tf(r));syn.append(tf(s));meta.append((x,r,s))
    vals=model(torch.stack(real).to(dev),torch.stack(syn).to(dev)).reshape(-1).cpu().tolist()
    for (x,r,s),v in zip(meta,vals):pix.append({"arm":arm,"instance_id":x["instance_id"],"class_name":x["class_name"],"lpips":float(v),"ssim":ssim_value(r,s),"edge_cosine":edge_cos(r,s)})
   for x in records:
    mask,_=mask_and_bbox(x["mask"]);values=boundary(Image.open(x["real"]).convert("RGB"),Image.open(x[arm]).convert("RGB"),mask);values={k:(float(v) if v is not None and np.isfinite(v) else None) for k,v in values.items()};geo.append({"arm":arm,"instance_id":x["instance_id"],"class_name":x["class_name"],**values})
 del model,real,syn,meta,sub,vals;gc.collect();torch.cuda.empty_cache()
 pixels={arm:summary([x for x in pix if x["arm"]==arm],("lpips","ssim","edge_cosine")) for arm in ARMS};geos={arm:summary([x for x in geo if x["arm"]==arm],("mask_boundary_gradient_cosine","boundary_chamfer_like","edge_density_ratio","inside_outside_transition_gradient_ratio")) for arm in ARMS};save(a.output/"pixel_perceptual_metrics.json",{"metrics":pixels,"raw_records":pix,"no_weighted_score":True});save(a.output/"boundary_geometry_metrics.json",{"metrics":geos,"raw_records":geo,"no_weighted_score":True})
 folds=load(a.folds);foldof={pid:f["fold"] for f in folds["folds"] for pid in f["holdout_pair_ids"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");refs=load(a.stage3r/"manifold_reference_corrected.json")["references"];fr=[]
 for entry in bankdoc["folds"]:
  fold=entry["fold"]
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz") as z:realbank,classes=z["features"],z["class_ids"]
  source={r["instance_id"]:i for i,r in enumerate(entry["records"])};ext=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512)
  try:
   for arm in ARMS:
    for x in records:
     if foldof[x["pair_id"]]!=fold:continue
     ci=CLASSES.index(x["class_name"]);_,bbox=mask_and_bbox(x["mask"]);feat=ext.encode(x[arm],bbox,512,512);bank=realbank[classes==ci];q95=float(refs[f"fold{fold}:{x['class_name']}"]["q95_real_distance"]);gap=float(1-np.dot(l2_normalize(feat),l2_normalize(realbank[source[x["instance_id"]]])));d=1-np.clip(np.stack([l2_normalize(v) for v in bank])@l2_normalize(feat),-1,1);gi=np.where(classes==ci)[0][int(np.argmin(d))];fr.append({"arm":arm,"instance_id":x["instance_id"],"class_name":x["class_name"],"normalized_pair_gap":gap/max(q95,1e-12),"fraction_gt_real_q95":gap>q95,"same_instance_nearest":entry["records"][gi]["instance_id"]==x["instance_id"]})
  finally:ext.close()
 def fs(z):return {"normalized_pair_gap":stat([x["normalized_pair_gap"] for x in z]),"fraction_gt_real_q95":float(np.mean([x["fraction_gt_real_q95"] for x in z])),"same_instance_nearest":float(np.mean([x["same_instance_nearest"] for x in z])),"n":len(z)}
 features={arm:{"overall":fs([x for x in fr if x["arm"]==arm]),"per_class":{c:fs([x for x in fr if x["arm"]==arm and x["class_name"]==c]) for c in CLASSES}} for arm in ARMS};save(a.output/"task_feature_metrics.json",{"metrics":features,"raw_records":fr,"extractor":"frozen Stage7A/7B OOF YOLO","validation_use_count":0})
 per={c:{arm:{"pixel":pixels[arm]["per_class"][c],"boundary":geos[arm]["per_class"][c],"feature":features[arm]["per_class"][c]} for arm in ARMS} for c in CLASSES};save(a.output/"per_class_metrics.json",per);save(a.output/"short_class_analysis.json",per["short"])
 lp=pixels["nomorph"]["overall"]["lpips"]["mean"]<pixels["full"]["overall"]["lpips"]["mean"];edge=pixels["nomorph"]["overall"]["edge_cosine"]["mean"]>pixels["full"]["overall"]["edge_cosine"]["mean"];bd=geos["nomorph"]["overall"]["mask_boundary_gradient_cosine"]["mean"]>geos["full"]["overall"]["mask_boundary_gradient_cosine"]["mean"];fg=features["nomorph"]["overall"]["normalized_pair_gap"]["median"]<=1.05*features["full"]["overall"]["normalized_pair_gap"]["median"] and features["nomorph"]["overall"]["same_instance_nearest"]>=features["full"]["overall"]["same_instance_nearest"]-.02
 lpcls=sum(per[c]["nomorph"]["pixel"]["lpips"]["mean"]<per[c]["full"]["pixel"]["lpips"]["mean"] for c in CLASSES);stcls=sum(per[c]["nomorph"]["pixel"]["edge_cosine"]["mean"]>per[c]["full"]["pixel"]["edge_cosine"]["mean"] or per[c]["nomorph"]["boundary"]["mask_boundary_gradient_cosine"]["mean"]>per[c]["full"]["boundary"]["mask_boundary_gradient_cosine"]["mean"] for c in CLASSES);short=per["short"]["nomorph"]["pixel"]["lpips"]["mean"]<per["short"]["full"]["pixel"]["lpips"]["mean"] and (per["short"]["nomorph"]["pixel"]["edge_cosine"]["mean"]>per["short"]["full"]["pixel"]["edge_cosine"]["mean"] or per["short"]["nomorph"]["boundary"]["mask_boundary_gradient_cosine"]["mean"]>per["short"]["full"]["boundary"]["mask_boundary_gradient_cosine"]["mean"])
 supported=lp and (edge or bd) and fg and lpcls>=4 and stcls>=4 and short
 status="NOMORPH_REPAIR_SUPPORTED" if supported else "NOMORPH_REPAIR_NOT_SUPPORTED" if (not fg or (not lp and not(edge or bd))) else "NOMORPH_REPAIR_MIXED"
 gate={"status":status,"criteria":{"lpips_improved":lp,"edge_improved":edge,"boundary_improved":bd,"task_feature_no_collapse":fg,"lpips_improved_classes":lpcls,"structural_improved_classes":stcls,"short_consistent_improvement":short},"detector_pilot_authorized":supported,"thresholds_frozen":{"max_relative_pair_gap_increase":.05,"max_same_instance_nearest_drop":.02,"minimum_broad_classes":4},"weighted_score":False};save(a.output/"stage8b_generator_gate.json",gate)
 final="GENERATOR_GATE_PASSED_AWAITING_SEED42_DETECTOR" if supported else "MSDF_REPAIR_NOT_SUPPORTED";lines=["# DeepPCB Stage 8B morphology training ablation","",f"Generator gate: **{status}**.",f"Stage status: **{final}**.","No weighted score was used; all generator axes are reported independently.",f"LPIPS improved: {lp}; edge improved: {edge}; boundary improved: {bd}; task feature no-collapse: {fg}.",f"Class breadth: LPIPS {lpcls}/6; structural {stcls}/6; short consistent improvement: {short}."];(a.output/"STAGE8B_REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps(gate,indent=2))
if __name__=="__main__":main()
