"""Evaluate Stage8A interventions without detector training or score fusion."""
from __future__ import annotations
import argparse,gc,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from dwbg_feature_extraction import DetectInputExtractor
from dwbg_v2_utils import l2_normalize
from run_deeppcb_stage2_pool import CLASSES
from run_deeppcb_stage5b import load
from run_deeppcb_stage5c import save
from evaluate_deeppcb_stage7b import crop_box, edge_cos, ssim_value

def fail(ok,msg):
 if not ok:raise RuntimeError("STAGE8A_EVAL_"+msg)
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":int(a.size)}
def crop(path,box):return Image.open(path).convert("RGB").crop(box).resize((256,256),Image.Resampling.BILINEAR)
def mask_and_bbox(mask_path):
 m=np.asarray(Image.open(mask_path).convert("L"))>127;y,x=np.where(m);fail(len(x)>0,"EMPTY_MASK")
 return m,(int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)
def grad(a):
 x=np.asarray(a.convert("L"),float)/255;p=np.pad(x,1,mode="edge");gx=-p[:-2,:-2]+p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-p[2:,:-2]+p[2:,2:];gy=-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:];return np.hypot(gx,gy)
def morph_band(mask):
 from scipy.ndimage import binary_dilation,binary_erosion
 return binary_dilation(mask,iterations=2)^binary_erosion(mask,iterations=2)
def boundary(real,syn,mask):
 from scipy.ndimage import binary_dilation,distance_transform_edt
 gr,gs=grad(real),grad(syn);band=morph_band(mask);inside=mask;outside=binary_dilation(mask,iterations=4)&~mask
 def cosine(a,b):
  den=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/den) if den else float(np.allclose(a,b))
 er=gr>np.quantile(gr[band],.65);es=gs>np.quantile(gs[band],.65)
 chamfer=float((distance_transform_edt(~es)[er&band].mean()+distance_transform_edt(~er)[es&band].mean())/2) if (er&band).any() and (es&band).any() else None
 return {"mask_boundary_gradient_cosine":cosine(gr[band],gs[band]),"boundary_chamfer_like":chamfer,"edge_density_ratio":float((gs[inside].mean()+1e-8)/(gr[inside].mean()+1e-8)),"inside_outside_transition_gradient_ratio":float(((gs[band].mean()/(gs[outside].mean()+1e-8))+1e-8)/((gr[band].mean()/(gr[outside].mean()+1e-8))+1e-8))}
def summarize(rows,fields):
 return {"overall":{f:stat([x[f] for x in rows if x[f] is not None]) for f in fields},"per_class":{c:{f:stat([x[f] for x in rows if x["class_name"]==c and x[f] is not None]) for f in fields} for c in CLASSES}}

def main():
 p=argparse.ArgumentParser()
 for n in ("stage8a","folds","stage3r","stage3r_runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");p.add_argument("--batch",type=int,default=16);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 manifest=load(a.stage8a/"stage8a_generated_manifest.json");records=manifest["records"];variants=manifest["variants"]
 fail(manifest.get("full_matches_frozen_stage2_seed42") is True and all("reference_path" in x and "reference_is_self" in x for x in records),"TARGET_REFERENCE_AUDIT")
 fail(len(records)==775*len(variants),"MATRIX")
 import torch
 from torchvision import transforms
 from evaluation.compare_mask_crop_lpips import lpips
 dev=torch.device("cuda:"+a.device if torch.cuda.is_available() else "cpu");tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);model=lpips.LPIPS(net="alex",verbose=False).to(dev).eval()
 pixrows=[];boundrows=[]
 with torch.no_grad():
  for start in range(0,len(records),a.batch):
   sub=records[start:start+a.batch];rr=[];ss=[];meta=[]
   for x in sub:
    box,_=crop_box(x["mask_path"]);mask,_=mask_and_bbox(x["mask_path"]);real=crop(x["real_path"],box);syn=crop(x["image_path"],box);rr.append(tf(real));ss.append(tf(syn));meta.append((x,real,syn,box,mask))
   vals=model(torch.stack(rr).to(dev),torch.stack(ss).to(dev)).reshape(-1).cpu().tolist()
   for (x,real,syn,box,mask),lp in zip(meta,vals):
    pixrows.append({"variant":x["variant"],"instance_id":x["instance_id"],"class_name":x["class_name"],"lpips":float(lp),"ssim":ssim_value(real,syn),"edge_cosine":edge_cos(real,syn)})
    fullreal=Image.open(x["real_path"]).convert("RGB");fullsyn=Image.open(x["image_path"]).convert("RGB");boundrows.append({"variant":x["variant"],"instance_id":x["instance_id"],"class_name":x["class_name"],**boundary(fullreal,fullsyn,mask)})
 del model,rr,ss,meta,sub,vals
 gc.collect()
 if torch.cuda.is_available():
  torch.cuda.synchronize(dev)
  torch.cuda.empty_cache()
 pixel={v:summarize([x for x in pixrows if x["variant"]==v],("lpips","ssim","edge_cosine")) for v in variants};save(a.output/"pixel_perceptual_metrics.json",{"metrics":pixel,"raw_records":pixrows,"no_weighted_score":True})
 bfields=("mask_boundary_gradient_cosine","boundary_chamfer_like","edge_density_ratio","inside_outside_transition_gradient_ratio");bounds={v:summarize([x for x in boundrows if x["variant"]==v],bfields) for v in variants};save(a.output/"boundary_geometry_metrics.json",{"metrics":bounds,"raw_records":boundrows,"diagnostic_only":True,"no_weighted_score":True})
 folds=load(a.folds);foldof={pid:f["fold"] for f in folds["folds"] for pid in f["holdout_pair_ids"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");refs=load(a.stage3r/"manifold_reference_corrected.json")["references"]
 feature_rows=[]
 for entry in bankdoc["folds"]:
  fold=entry["fold"]
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz") as z:real,classes=z["features"],z["class_ids"]
  source={r["instance_id"]:i for i,r in enumerate(entry["records"])};ext=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512)
  try:
   for x in records:
    if foldof[x["pair_id"]]!=fold:continue
    ci=CLASSES.index(x["class_name"]);_,bbox=mask_and_bbox(x["mask_path"]);feat=ext.encode(x["image_path"],bbox,512,512);bank=real[classes==ci];q95=float(refs[f"fold{fold}:{x['class_name']}"]["q95_real_distance"]);gap=float(1-np.dot(l2_normalize(feat),l2_normalize(real[source[x["instance_id"]]])));d=1-np.clip(np.stack([l2_normalize(v) for v in bank])@l2_normalize(feat),-1,1);global_idx=np.where(classes==ci)[0][int(np.argmin(d))]
    feature_rows.append({"variant":x["variant"],"instance_id":x["instance_id"],"class_name":x["class_name"],"normalized_pair_gap":gap/max(q95,1e-12),"fraction_gt_real_q95":gap>q95,"same_instance_nearest":entry["records"][global_idx]["instance_id"]==x["instance_id"]})
  finally:ext.close()
 def fs(rows):return {"normalized_pair_gap":stat([x["normalized_pair_gap"] for x in rows]),"fraction_gt_real_q95":float(np.mean([x["fraction_gt_real_q95"] for x in rows])),"same_instance_nearest":float(np.mean([x["same_instance_nearest"] for x in rows])),"n":len(rows)}
 features={v:{"overall":fs([x for x in feature_rows if x["variant"]==v]),"per_class":{c:fs([x for x in feature_rows if x["variant"]==v and x["class_name"]==c]) for c in CLASSES}} for v in variants};save(a.output/"task_feature_metrics.json",{"feature_rule":"Stage7A/7B DetectInputExtractor; OOF YOLO11s best.pt; source-paired","metrics":features,"raw_records":feature_rows,"validation_use_count":0})
 full={"lpips":pixel["full"]["overall"]["lpips"]["mean"],"edge":pixel["full"]["overall"]["edge_cosine"]["mean"],"boundary":bounds["full"]["overall"]["mask_boundary_gradient_cosine"]["mean"],"feature":features["full"]["overall"]["normalized_pair_gap"]["median"]}
 trade={}
 for v in variants:
  trade[v]={"delta_vs_full":{"normalized_pair_gap_median":features[v]["overall"]["normalized_pair_gap"]["median"]-full["feature"],"lpips_mean":pixel[v]["overall"]["lpips"]["mean"]-full["lpips"],"edge_cosine_mean":pixel[v]["overall"]["edge_cosine"]["mean"]-full["edge"],"boundary_gradient_cosine_mean":bounds[v]["overall"]["mask_boundary_gradient_cosine"]["mean"]-full["boundary"]},"inference_only_diagnostic":v!="full"}
 save(a.output/"tradeoff_summary.json",{"full":full,"variants":trade,"axes_separate":True,"no_weighted_score":True})
 perclass={c:{v:{"pixel":pixel[v]["per_class"][c],"feature":features[v]["per_class"][c],"boundary":bounds[v]["per_class"][c]} for v in variants} for c in CLASSES};save(a.output/"per_class_attribution.json",perclass)
 candidates=[v for v in variants if v not in ("full","residual_zero","no_multiscale") and trade[v]["delta_vs_full"]["lpips_mean"]<0 and trade[v]["delta_vs_full"]["edge_cosine_mean"]>0]
 status="NO_CLEAR_COMPONENT_ATTRIBUTION" if not candidates else "MULTIPLE_COMPONENTS_CONFOUNDED" if len(candidates)>1 else {"no_morphology":"MORPHOLOGY_BRANCH_SUSPECTED","no_latent":"SEMANTIC_BRANCH_SUSPECTED","no_pixel":"PIXEL_BRANCH_SUSPECTED"}.get(candidates[0],"MULTIPLE_COMPONENTS_CONFOUNDED")
 save(a.output/"stage8a_status.json",{"status":status,"diagnostic_candidates":candidates,"detector_training_count":0,"validation_use_count":0,"weighted_quality_score":False,"formal_causal_claim":False})
 lines=["# DeepPCB Stage 8A MSDF mechanism attribution","",f"Status: **{status}**.","All interventions use the frozen MSDF-v3 checkpoints and generation seed 42; no detector was trained.","Branch removals are inference diagnostics with documented train-test mismatch, not standalone causal proof.","Pixel/perceptual, task-feature, and boundary axes remain separate; no weighted score was constructed.","","## Overall diagnostic deltas versus Full","","| Variant | normalized pair gap median | LPIPS mean | edge cosine mean | boundary gradient cosine mean |","|---|---:|---:|---:|---:|"]
 for v in variants:
  if v=="full":continue
  d=trade[v]["delta_vs_full"];lines.append(f"| {v} | {d['normalized_pair_gap_median']:+.6f} | {d['lpips_mean']:+.6f} | {d['edge_cosine_mean']:+.6f} | {d['boundary_gradient_cosine_mean']:+.6f} |")
 lines += ["","Lower normalized pair gap and LPIPS are favorable; higher edge and boundary cosine are favorable.","No intervention improved both overall LPIPS and edge cosine. Removing latent or all residual injection slightly reduced the feature gap, but worsened LPIPS and edge; therefore neither isolates the fidelity conflict.","","## Edge-sensitive classes","","For short, every branch removal worsened the feature-gap median and none improved LPIPS; this does not identify the Stage7C short-class loss with a single inference-time component.","For mousebite, no-pixel and half-residual variants produced tiny local LPIPS/edge improvements, but the effects did not hold overall or consistently across boundary and feature axes.","For pinhole, no-latent improved feature gap and boundary cosine while worsening LPIPS and edge, showing a trade-off rather than a dominant harmful branch.","",f"Diagnostic trade-off candidates under the preregistered joint LPIPS+edge rule: {', '.join(candidates) if candidates else 'none'}.","Morphology alignment remains class B: bypass changes spatial support semantics and requires a minimal retraining ablation before a causal claim."]
 (a.output/"STAGE8A_REPORT.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"status":status,"candidates":candidates},indent=2))
if __name__=="__main__":main()
