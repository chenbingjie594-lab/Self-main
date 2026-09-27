"""Stage9C canonical fidelity, feature probes, and failure attribution."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms
import lpips
from scipy.ndimage import binary_dilation,binary_erosion
from scipy.stats import spearmanr
from evaluate_deeppcb_stage7b import crop_box,edge_cos,ssim_value
from dwbg_feature_extraction import DetectInputExtractor
from dwbg_v2_utils import l2_normalize

CLASSES=("short","pinhole");STAGE9_ARMS=("sd2","full_msdf","frozen_adapter");PROBE_ARMS=("pretrained_off","adapter_on","class_finetuned_sd2")
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":int(a.size)}
def mask(path):return np.asarray(Image.open(path).convert("L"))>127
def bbox(path):
 m=mask(path);y,x=np.where(m)
 if not len(x):raise RuntimeError("STAGE9C_EMPTY_MASK "+str(path))
 return int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1
def grad(im):
 x=np.asarray(im.convert("L"),float)/255;p=np.pad(x,1,mode="edge");gx=-p[:-2,:-2]+p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-p[2:,:-2]+p[2:,2:];gy=-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:];return np.hypot(gx,gy)
def cos(x,y):
 x=np.asarray(x).reshape(-1);y=np.asarray(y).reshape(-1);d=np.linalg.norm(x)*np.linalg.norm(y);return float(np.dot(x,y)/d) if d else float(np.allclose(x,y))
def summary(rows):return {k:stat([x[k] for x in rows]) for k in ("lpips","ssim","edge_cosine","boundary_gradient_cosine")}
def canonical(records,arms,dev):
 net=lpips.LPIPS(net="alex",verbose=False).to(dev).eval();tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);out=[]
 with torch.no_grad():
  for x in records:
   box,_=crop_box(x["mask"]);real=Image.open(x["real"]).convert("RGB").crop(box);m=mask(x["mask"]);band=binary_dilation(m,iterations=2)^binary_erosion(m,iterations=2);gf=grad(Image.open(x["real"]).convert("RGB"))
   for arm in arms:
    syn=Image.open(x[arm]).convert("RGB");local=syn.crop(box);lp=float(net(tf(real)[None].to(dev),tf(local)[None].to(dev)).item());out.append({"instance_id":x["instance_id"],"class_name":x["class_name"],"seed":int(x.get("generation_seed",42)),"arm":arm,"lpips":lp,"ssim":ssim_value(real,local),"edge_cosine":edge_cos(real,local),"boundary_gradient_cosine":cos(gf[band],grad(syn)[band])})
 del net;torch.cuda.empty_cache();return out
def aggregate(rows,arms):return {a:{"overall":summary([x for x in rows if x["arm"]==a]),"per_class":{c:summary([x for x in rows if x["arm"]==a and x["class_name"]==c]) for c in CLASSES}} for a in arms}
def stage9_records(a):
 fra=load(a.stage9a_manifest)["records_detail"];keys={(x["instance_id"],int(x["generation_seed"])) for x in fra};reg={x["instance_id"]:x for x in load(a.registry)["instances"]};sd={(x["instance_id"],int(x["generation_seed"])):x for x in load(a.sd2_manifest)["records"]};ms={(x["target_instance_id"],int(x["generation_seed"])):x for x in load(a.msdf_manifest)["records"]};fa={(x["instance_id"],int(x["generation_seed"])):x for x in fra};rows=[]
 for key in sorted(keys):
  iid,seed=key;r=reg[iid];rows.append({"instance_id":iid,"class_name":r["class_name"],"generation_seed":seed,"real":r["defect_image"],"mask":r["instance_mask_path"],"sd2":sd[key]["image_path"],"full_msdf":ms[key]["image_path"],"frozen_adapter":fa[key]["image_path"]})
 return rows
def feature_probe(records,a):
 folds=load(a.folds);foldof={pid:f["fold"] for f in folds["folds"] for pid in f["holdout_pair_ids"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");refs=load(a.stage3r/"manifold_reference_corrected.json")["references"];out=[]
 for entry in bankdoc["folds"]:
  fold=entry["fold"]
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz") as z:real,ids=z["features"],z["class_ids"]
  source={x["instance_id"]:i for i,x in enumerate(entry["records"])};ext=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512)
  try:
   for x in records:
    if foldof[x["pair_id"]]!=fold:continue
    box=bbox(x["mask"]);ci=(1 if x["class_name"]=="short" else 5);bank_idx=np.where(ids==ci)[0];bank=np.stack([l2_normalize(real[i]) for i in bank_idx]);q95=float(refs[f"fold{fold}:{x['class_name']}"]["q95_real_distance"])
    for arm in PROBE_ARMS:
     feat=l2_normalize(ext.encode(x[arm],box,512,512));gap=float(1-np.dot(feat,l2_normalize(real[source[x["instance_id"]]])));order=np.argsort(1-np.clip(bank@feat,-1,1));retrieved=[entry["records"][bank_idx[i]]["instance_id"] for i in order[:5]];out.append({"instance_id":x["instance_id"],"class_name":x["class_name"],"arm":arm,"normalized_pair_gap":gap/max(q95,1e-12),"fraction_gt_real_q95":bool(gap>q95),"retrieval_top1":retrieved[0]==x["instance_id"],"retrieval_top5":x["instance_id"] in retrieved})
  finally:ext.close()
 def agg(rows):return {"normalized_pair_gap":stat([x["normalized_pair_gap"] for x in rows]),"fraction_gt_real_q95":float(np.mean([x["fraction_gt_real_q95"] for x in rows])),"retrieval_top1":float(np.mean([x["retrieval_top1"] for x in rows])),"retrieval_top5":float(np.mean([x["retrieval_top5"] for x in rows])),"n":len(rows)}
 return out,{arm:{"overall":agg([x for x in out if x["arm"]==arm]),"per_class":{c:agg([x for x in out if x["arm"]==arm and x["class_name"]==c]) for c in CLASSES}} for arm in PROBE_ARMS}
def residual(raw,probe_sources):
 rows=raw["records"];fields=("global_residual_to_hidden_ratio","masked_residual_to_hidden_ratio","raw_masked_residual_to_hidden_ratio","scale_multiplier","bounded_outside_residual_rms");summary={}
 for c in CLASSES:
  summary[c]={}
  for block in (2,3):
   z=[x for x in rows if x["class_name"]==c and x["block"]==block and x["branch"]=="conditional"];summary[c][f"block{block}"]={**{f:stat([x[f] for x in z]) for f in fields},"bound_active_fraction":float(np.mean([x["global_rms_bound_active"] for x in z]))}
 area={x["instance_id"]:float(x["area_fraction"]) for x in probe_sources};corr={}
 for c in CLASSES:
  corr[c]={}
  for block in (2,3):
   z=[x for x in rows if x["class_name"]==c and x["block"]==block and x["branch"]=="conditional"];by=defaultdict(list)
   for x in z:by[x["instance_id"]].append(float(x["masked_residual_to_hidden_ratio"]))
   source_ids=sorted(by);xx=np.asarray([area[i] for i in source_ids]);yy=np.asarray([np.mean(by[i]) for i in source_ids]);undefined=None
   if np.allclose(xx,xx[0]):rho=p=None;undefined="original mask area is constant"
   elif np.allclose(yy,yy[0]):rho=p=None;undefined="source-mean local residual ratio is constant"
   else:
    rv,pv=spearmanr(xx,yy);rho,p=(float(rv),float(pv)) if np.isfinite(rv) and np.isfinite(pv) else (None,None);undefined=None if rho is not None else "Spearman returned non-finite value"
   corr[c][f"block{block}"]={"spearman_rho":rho,"p_value":p,"n_sources":len(source_ids),"unit_of_analysis":"source; conditional branch averaged across 50 timesteps","mask_area_definition":"original 512x512 frozen source mask area_fraction","undefined_reason":undefined}
 return summary,corr
def main():
 p=argparse.ArgumentParser()
 for n in ("stage9a","stage9a_manifest","sd2_manifest","msdf_manifest","registry","probe_manifest","residual_raw","folds","stage3r","stage3r_runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev=torch.device("cuda:"+a.device);stage=stage9_records(a);canonical_rows=canonical(stage,STAGE9_ARMS,dev);canonical_metrics=aggregate(canonical_rows,STAGE9_ARMS);save(a.output/"canonical_stage9a_fidelity_metrics.json",{"protocol":{"crop":"Stage7B adaptive padded crop: max(64,4x bbox extent), then LPIPS resize 256","lpips":"lpips AlexNet, input normalized [-1,1]","reference":"matched frozen real training instance","seed_aggregation":"all three seeds as paired records"},"metrics":canonical_metrics,"raw_records":canonical_rows})
 seed42=[x for x in stage if x["generation_seed"]==42];reaudit=aggregate([x for x in canonical_rows if x["seed"]==42],STAGE9_ARMS);historical=load(a.stage9a/"fidelity_metrics.json")["metrics"];reconfirmed=all(canonical_metrics["frozen_adapter"]["per_class"][c]["lpips"]["mean"]>canonical_metrics["sd2"]["per_class"][c]["lpips"]["mean"] and canonical_metrics["frozen_adapter"]["per_class"][c]["edge_cosine"]["mean"]<canonical_metrics["sd2"]["per_class"][c]["edge_cosine"]["mean"] for c in CLASSES);save(a.output/"cross_stage_fidelity_reaudit.json",{"audit":{"Stage7B":{"crop":"adaptive padded crop, minimum 64, context scale 4; crop then transform resize 256","mask":"frozen target instance mask","reference":"matched frozen real training instance","seed_aggregation":"six generation seeds"},"Stage8A":{"crop":"same Stage7B adaptive padded crop; explicit resize 256 before LPIPS","mask":"frozen target instance mask","reference":"matched frozen real training instance","seed_aggregation":"seed42"},"Stage8B":{"crop":"same Stage7B adaptive padded crop; LPIPS transform resize 256","mask":"frozen target instance mask","reference":"matched frozen real training instance","seed_aggregation":"seed42"},"Stage9A_historical":{"crop":"tight nonzero mask bbox, then explicit resize 256","mask":"frozen target instance mask","reference":"matched frozen real training instance","seed_aggregation":"three seeds"},"shared_lpips":"AlexNet LPIPS with RGB normalized to [-1,1]","source_matching":"same instance within every comparison","direct_historical_values_not_cross_stage_comparable":True},"canonical_seed42":reaudit,"historical_stage9a":historical,"status":"FIDELITY_FAILURE_RECONFIRMED" if reconfirmed else "FIDELITY_FAILURE_EVALUATOR_DEPENDENT","stage9a_gate_unchanged":True})
 probes=load(a.probe_manifest)["records"];probe_pix=canonical(probes,PROBE_ARMS,dev);probe_metrics=aggregate(probe_pix,PROBE_ARMS);feature_rows,feature_metrics=feature_probe(probes,a);save(a.output/"adapter_on_off_probe.json",{"fidelity":{k:probe_metrics[k] for k in ("pretrained_off","adapter_on")},"task_feature":{k:feature_metrics[k] for k in ("pretrained_off","adapter_on")},"raw_fidelity":probe_pix,"raw_task_feature":feature_rows});save(a.output/"pretrained_vs_finetuned_probe.json",{"fidelity":probe_metrics,"task_feature":feature_metrics,"same_probe_sources":60,"seed":42})
 rsummary,corr=residual(load(a.residual_raw),load(a.output/"probe_source_manifest.json")["sources"]);save(a.output/"residual_magnitude_summary.json",{"metrics":rsummary,"projection_bias_outside_mask_observed":any(x["bounded_outside_residual_rms"]>0 for x in load(a.residual_raw)["records"])});save(a.output/"mask_area_residual_correlation.json",corr)
 # Preregistered interpretation thresholds for this audit (not a model gate).
 overshoot=all(rsummary[c][f"block{b}"]["masked_residual_to_hidden_ratio"]["median"]>.15 for c in CLASSES for b in (2,3)) and all(corr[c][f"block{b}"]["spearman_rho"] is not None and corr[c][f"block{b}"]["spearman_rho"]<=-.30 and corr[c][f"block{b}"]["p_value"]<.05 for c in CLASSES for b in (2,3)) and all(feature_metrics["adapter_on"]["per_class"][c]["normalized_pair_gap"]["mean"]>1.10*feature_metrics["pretrained_off"]["per_class"][c]["normalized_pair_gap"]["mean"] for c in CLASSES)
 semantic=all(feature_metrics["pretrained_off"]["per_class"][c]["normalized_pair_gap"]["mean"]>1.10*feature_metrics["class_finetuned_sd2"]["per_class"][c]["normalized_pair_gap"]["mean"] and feature_metrics["pretrained_off"]["per_class"][c]["fraction_gt_real_q95"]-feature_metrics["class_finetuned_sd2"]["per_class"][c]["fraction_gt_real_q95"]>.10 for c in CLASSES)
 status="BOTH_CONTRIBUTE" if overshoot and semantic else "LOCAL_INJECTION_OVERSHOOT_SUPPORTED" if overshoot else "SEMANTIC_ADAPTATION_DEFICIT_SUPPORTED" if semantic else "NO_CLEAR_FAILURE_ATTRIBUTION";probe_doc=load(a.probe_manifest);identical=sum(x["adapter_on_sha256"]==x["pretrained_off_sha256"] for x in probe_doc["records"]);raw_doc=load(a.residual_raw);zero=sum(x["raw_global_residual_rms"]==0 and x["bounded_global_residual_rms"]==0 for x in raw_doc["records"]);save(a.output/"stage9c_status.json",{"status":status,"local_injection_overshoot_supported":overshoot,"semantic_adaptation_deficit_supported":semantic,"effective_adapter_noop_observed":identical==len(probe_doc["records"]) and zero==len(raw_doc["records"]),"effective_adapter_noop_evidence":{"adapter_on_off_identical_image_hashes":f"{identical}/{len(probe_doc['records'])}","residual_records_exactly_zero":f"{zero}/{len(raw_doc['records'])}","blocks":[2,3],"cfg_branches":["unconditional","conditional"],"diffusion_timesteps":50},"thresholds":{"local_ratio_median_min":.15,"mask_area_spearman_max":-.30,"spearman_p_max":.05,"adapter_on_gap_relative_worsening_min":.10,"pretrained_gap_relative_to_finetuned_min":.10,"q95_fraction_absolute_worsening_min":.10},"stage9a_gate_unchanged":True,"detector_training_count":0,"official_validation_use_count":0,"stage9d_authorized":False})
 print(json.dumps({"status":status,"overshoot":overshoot,"semantic":semantic},indent=2))
if __name__=="__main__":main()
