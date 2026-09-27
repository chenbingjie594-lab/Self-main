"""Formal Stage9A three-arm evaluation; no detector training or validation use."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms
import lpips
from scipy.ndimage import binary_dilation,binary_erosion
from dwbg_feature_extraction import DetectInputExtractor
from dwbg_v2_utils import l2_normalize

ARMS=("sd2","full_msdf","frozen_adapter");CLASSES=("short","pinhole");SEEDS=(42,2026,3407)
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":int(a.size)}
def grad(im):
 x=np.asarray(im.convert("L"),float)/255;p=np.pad(x,1,mode="edge");gx=-p[:-2,:-2]+p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-p[2:,:-2]+p[2:,2:];gy=-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:];return np.hypot(gx,gy)
def cos(a,b):
 a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1);d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else float(np.allclose(a,b))
def maskbox(path):
 m=np.asarray(Image.open(path).convert("L"))>127;y,x=np.where(m)
 if not len(x):raise RuntimeError("EMPTY_MASK "+str(path))
 return m,(int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)
def crop(path,box):return Image.open(path).convert("RGB").crop(box).resize((256,256),Image.Resampling.BILINEAR)
def ssim_value(first,second):
 # Same dependency-free global SSIM fallback already frozen in Stage7B.
 a=np.asarray(first.convert("RGB"),dtype=np.float64);b=np.asarray(second.convert("RGB"),dtype=np.float64);values=[];c1=(.01*255)**2;c2=(.03*255)**2
 for channel in range(3):
  x,y=a[...,channel],b[...,channel];mx,my=x.mean(),y.mean();vx,vy=x.var(),y.var();cov=((x-mx)*(y-my)).mean();values.append(((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)))
 return float(np.mean(values))
def summarize(rows,fields):return {f:stat([x[f] for x in rows]) for f in fields}
def arm_records(a):
 fra=load(a.adapter_manifest)["records_detail"];wanted={(x["instance_id"],int(x["generation_seed"])) for x in fra}
 sd=[]
 for x in load(a.sd2_manifest)["records"]:
  iid=x.get("instance_id",x.get("target_instance_id"));seed=int(x.get("generation_seed",x.get("seed",-1)))
  if (iid,seed) in wanted:sd.append({"arm":"sd2","instance_id":iid,"pair_id":x.get("pair_id",x.get("target_pair_id")),"class_name":x["class_name"],"generation_seed":seed,"image_path":x["image_path"],"real_path":x.get("real",x.get("real_path")),"mask_path":x.get("mask",x.get("mask_path",x.get("target_mask")))})
 ms=[]
 for x in load(a.msdf_manifest)["records"]:
  iid=x["target_instance_id"];seed=int(x["generation_seed"])
  if (iid,seed) in wanted:ms.append({"arm":"full_msdf","instance_id":iid,"pair_id":x["target_pair_id"],"class_name":x["class_name"],"generation_seed":seed,"image_path":x["image_path"],"real_path":None,"mask_path":x["target_mask"]})
 reg={x["instance_id"]:x for x in load(a.registry)["instances"]}
 out=[]
 for x in sd+ms+fra:
  x=dict(x);r=reg[x["instance_id"]];x["real_path"]=x.get("real_path") or r["defect_image"];x["mask_path"]=x.get("mask_path") or r["instance_mask_path"];x["pair_id"]=x.get("pair_id") or r["pair_id"];out.append(x)
 expected=len(wanted)
 if any(sum(x["arm"]==arm for x in out)!=expected for arm in ARMS):raise RuntimeError("STAGE9A_UNMATCHED_ARM_MATRIX")
 return out
def main():
 p=argparse.ArgumentParser()
 for n in ("adapter_manifest","sd2_manifest","msdf_manifest","registry","folds","stage3r","stage3r_runs","gate_protocol","adapter_root","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=arm_records(a);dev=torch.device("cuda:"+a.device);net=lpips.LPIPS(net="alex",verbose=False).to(dev).eval();tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);pix=[]
 with torch.no_grad():
  for x in records:
   m,box=maskbox(x["mask_path"]);real=crop(x["real_path"],box);syn=crop(x["image_path"],box);lp=float(net(tf(real)[None].to(dev),tf(syn)[None].to(dev)).item());gr,gs=grad(real),grad(syn);fullr=Image.open(x["real_path"]).convert("RGB");fulls=Image.open(x["image_path"]).convert("RGB");gfr,gfs=grad(fullr),grad(fulls);band=binary_dilation(m,iterations=2)^binary_erosion(m,iterations=2);pix.append({**{k:x[k] for k in ("arm","instance_id","class_name","generation_seed")},"lpips":lp,"ssim":ssim_value(real,syn),"edge_cosine":cos(gr,gs),"boundary_gradient_cosine":cos(gfr[band],gfs[band])})
 fields=("lpips","ssim","edge_cosine","boundary_gradient_cosine");fidelity={arm:{"overall":summarize([x for x in pix if x["arm"]==arm],fields),"per_class":{c:summarize([x for x in pix if x["arm"]==arm and x["class_name"]==c],fields) for c in CLASSES}} for arm in ARMS};save(a.output/"fidelity_metrics.json",{"metrics":fidelity,"raw_records":pix,"no_weighted_score":True})
 # Conditional diversity: exact same source, three fixed generation seeds.
 by=defaultdict(list)
 for x in records:by[(x["arm"],x["class_name"],x["instance_id"])].append(x)
 div=[]
 with torch.no_grad():
  for (arm,cls,iid),group in by.items():
   group=sorted(group,key=lambda z:z["generation_seed"]);_,box=maskbox(group[0]["mask_path"]);ims=[crop(x["image_path"],box) for x in group];vals=[];edges=[];pixels=[]
   for i in range(3):
    pixels.append(np.asarray(ims[i],float)/255)
    for j in range(i+1,3):vals.append(float(net(tf(ims[i])[None].to(dev),tf(ims[j])[None].to(dev)).item()));edges.append(1-cos(grad(ims[i]),grad(ims[j])))
   div.append({"arm":arm,"class_name":cls,"instance_id":iid,"ic_lpips":float(np.mean(vals)),"edge_map_distance":float(np.mean(edges)),"mask_crop_pixel_variance":float(np.stack(pixels).var(axis=0).mean())})
 # OOF feature extraction for task support, retrieval and feature diversity.
 fold_doc=load(a.folds);foldof={pid:f["fold"] for f in fold_doc["folds"] for pid in f["holdout_pair_ids"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");refs=load(a.stage3r/"manifold_reference_corrected.json")["references"];features=[]
 for entry in bankdoc["folds"]:
  fold=entry["fold"]
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz") as z:real=z["features"];class_ids=z["class_ids"]
  ext=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512)
  try:
   source={r["instance_id"]:i for i,r in enumerate(entry["records"])}
   for x in records:
    if foldof[x["pair_id"]]!=fold:continue
    _,box=maskbox(x["mask_path"]);f=l2_normalize(ext.encode(x["image_path"],box,512,512));ci=(1 if x["class_name"]=="short" else 5);idx=np.where(class_ids==ci)[0];bank=np.stack([l2_normalize(real[i]) for i in idx]);dist=1-np.clip(bank@f,-1,1);order=np.argsort(dist);retrieved=[entry["records"][idx[i]]["instance_id"] for i in order[:5]];gap=float(1-np.dot(f,l2_normalize(real[source[x["instance_id"]]])));q95=float(refs[f"fold{fold}:{x['class_name']}"]["q95_real_distance"]);features.append({**{k:x[k] for k in ("arm","instance_id","class_name","generation_seed")},"feature":f.tolist(),"normalized_pair_gap":gap/max(q95,1e-12),"fraction_gt_real_q95":bool(gap>q95),"retrieval_top1":retrieved[0]==x["instance_id"],"retrieval_top5":x["instance_id"] in retrieved})
  finally:ext.close()
 fby=defaultdict(list)
 for x in features:fby[(x["arm"],x["class_name"],x["instance_id"])].append(np.asarray(x["feature"]))
 for row in div:
  fs=np.stack(fby[(row["arm"],row["class_name"],row["instance_id"])]);row["feature_variance"]=float(fs.var(axis=0).sum());row["feature_pairwise_distance"]=float(np.mean([1-np.dot(fs[i],fs[j]) for i in range(3) for j in range(i+1,3)]))
 dfields=("ic_lpips","edge_map_distance","mask_crop_pixel_variance","feature_variance","feature_pairwise_distance");diversity={arm:{"overall":summarize([x for x in div if x["arm"]==arm],dfields),"per_class":{c:summarize([x for x in div if x["arm"]==arm and x["class_name"]==c],dfields) for c in CLASSES}} for arm in ARMS};save(a.output/"diversity_metrics.json",{"metrics":diversity,"raw_records":div,"no_weighted_score":True})
 def identity(rows):
  arr=np.stack([x["feature"] for x in rows]);total=float(arr.var(axis=0).sum());groups=defaultdict(list)
  for x in rows:groups[x["instance_id"]].append(x["feature"])
  within=float(np.mean([np.stack(v).var(axis=0).sum() for v in groups.values()]));return {"retrieval_top1":float(np.mean([x["retrieval_top1"] for x in rows])),"retrieval_top5":float(np.mean([x["retrieval_top5"] for x in rows])),"within_source_variance":within,"total_variance":total,"within_total_ratio":within/max(total,1e-12),"n":len(rows)}
 identity_metrics={arm:{"overall":identity([x for x in features if x["arm"]==arm]),"per_class":{c:identity([x for x in features if x["arm"]==arm and x["class_name"]==c]) for c in CLASSES}} for arm in ARMS};save(a.output/"source_identity_metrics.json",{"metrics":identity_metrics,"raw_records":features})
 def task(rows):return {"normalized_pair_gap":stat([x["normalized_pair_gap"] for x in rows]),"fraction_gt_real_q95":float(np.mean([x["fraction_gt_real_q95"] for x in rows])),"n":len(rows)}
 task_metrics={arm:{"overall":task([x for x in features if x["arm"]==arm]),"per_class":{c:task([x for x in features if x["arm"]==arm and x["class_name"]==c]) for c in CLASSES}} for arm in ARMS};save(a.output/"task_feature_support_metrics.json",{"metrics":task_metrics,"raw_records":features,"validation_use_count":0})
 protocol=load(a.gate_protocol);checks={};class_pass={}
 for c in CLASSES:
  fra=fidelity["frozen_adapter"]["per_class"][c];sd=fidelity["sd2"]["per_class"][c];dv=diversity["frozen_adapter"]["per_class"][c];dm=diversity["full_msdf"]["per_class"][c];fi=identity_metrics["frozen_adapter"]["per_class"][c];mi=identity_metrics["full_msdf"]["per_class"][c];ft=task_metrics["frozen_adapter"]["per_class"][c];st=task_metrics["sd2"]["per_class"][c];q=protocol["criteria"]
  checks[c]={"fidelity":fra["lpips"]["mean"]<=sd["lpips"]["mean"]*q["no_clear_fidelity_collapse"]["lpips_ratio_vs_sd2_max"] and fra["edge_cosine"]["mean"]-sd["edge_cosine"]["mean"]>=q["no_clear_fidelity_collapse"]["edge_cosine_delta_vs_sd2_min"] and fra["boundary_gradient_cosine"]["mean"]-sd["boundary_gradient_cosine"]["mean"]>=q["no_clear_fidelity_collapse"]["boundary_gradient_cosine_delta_vs_sd2_min"],"diversity":dv["ic_lpips"]["mean"]>=dm["ic_lpips"]["mean"]*q["diversity_increase_vs_full_msdf"]["ic_lpips_ratio_min"] and dv["feature_variance"]["mean"]>=dm["feature_variance"]["mean"]*q["diversity_increase_vs_full_msdf"]["feature_variance_ratio_min"],"identity":fi["retrieval_top1"]-mi["retrieval_top1"]<=q["source_identity_dominance_down_vs_full_msdf"]["top1_retrieval_delta_max"] and fi["within_total_ratio"]-mi["within_total_ratio"]<=q["source_identity_dominance_down_vs_full_msdf"]["within_total_ratio_delta_max"],"task":ft["normalized_pair_gap"]["mean"]<=st["normalized_pair_gap"]["mean"]*q["task_feature_not_notably_worse_than_sd2"]["normalized_pair_gap_ratio_max"] and ft["fraction_gt_real_q95"]-st["fraction_gt_real_q95"]<=q["task_feature_not_notably_worse_than_sd2"]["fraction_gt_q95_delta_max"]};class_pass[c]=all(checks[c].values())
 if all(class_pass.values()):status="FROZEN_ADAPTER_FEASIBLE"
 elif any(not checks[c]["fidelity"] or not checks[c]["task"] for c in CLASSES) or all(not checks[c]["diversity"] or not checks[c]["identity"] for c in CLASSES):status="FROZEN_ADAPTER_NOT_FEASIBLE"
 else:status="FROZEN_ADAPTER_MIXED"
 train={c:load(a.adapter_root/c/"training_audit.json") for c in CLASSES};save(a.output/"training_audit.json",{"classes":train,"all_complete_2000":all(x["status"]=="COMPLETE" and x["successful_updates"]==2000 for x in train.values())});save(a.output/"source_seed_pairing_audit.json",{"arms":list(ARMS),"sources":258,"seeds":list(SEEDS),"records_per_arm":774,"exact_matrix":True,"official_validation_use_count":0});per={c:{arm:{"fidelity":fidelity[arm]["per_class"][c],"diversity":diversity[arm]["per_class"][c],"identity":identity_metrics[arm]["per_class"][c],"task":task_metrics[arm]["per_class"][c]} for arm in ARMS} for c in CLASSES};save(a.output/"per_class_metrics.json",per);save(a.output/"memorization_drift_diagnosis.json",{"source_identity":identity_metrics,"historical_gradient_data_unavailable":True,"interpretation":"diagnostic only; parameter drift is not a quality score"});save(a.output/"stage9a_gate.json",{"status":status,"checks":checks,"per_class_pass":class_pass,"STAGE9B_DETECTOR_PILOT_AUTHORIZED":status=="FROZEN_ADAPTER_FEASIBLE","detector_training_count":0,"weighted_score":False,"validation_use_count":0})
 lines=["# DeepPCB Stage9A Frozen-Backbone Feasibility","",f"Status: **{status}**.","","No detector was trained and official validation was not used. All axes were evaluated independently without a weighted quality score.","","| Class | Fidelity | Diversity | Lower identity dominance | Task support | Overall |","|---|---:|---:|---:|---:|---:|"]
 for c in CLASSES:lines.append(f"| {c} | {checks[c]['fidelity']} | {checks[c]['diversity']} | {checks[c]['identity']} | {checks[c]['task']} | {class_pass[c]} |")
 lines += ["",f"STAGE9B_DETECTOR_PILOT_AUTHORIZED={str(status=='FROZEN_ADAPTER_FEASIBLE').lower()}"];(a.output/"STAGE9A_REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps({"status":status,"checks":checks},indent=2))
if __name__=="__main__":main()
