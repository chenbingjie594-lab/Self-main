"""Canonical Stage9E fidelity, diversity, identity, task-support, and frozen gate."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms
import lpips
from scipy.ndimage import binary_dilation,binary_erosion
from evaluate_deeppcb_stage7b import crop_box,edge_cos,ssim_value,sobel
from dwbg_feature_extraction import DetectInputExtractor
from dwbg_v2_utils import l2_normalize
CLASSES=("short","pinhole");ARMS=("class_finetuned_sd2","repaired_frozen_adapter","pretrained_no_adaptation")
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":len(a)}
def grad(im):return sobel(im).reshape(np.asarray(im.convert("L")).shape)
def cos(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else float(np.allclose(a,b))
def summarize(rows,fields):return {k:stat([x[k] for x in rows]) for k in fields}
def fidelity(rows,dev,batch):
 tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);net=lpips.LPIPS(net="alex",verbose=False).to(dev).eval();raw=[]
 with torch.no_grad():
  for x in rows:
   box,_=crop_box(x["mask"]);real_full=Image.open(x["real"]).convert("RGB");real=real_full.crop(box);m=np.asarray(Image.open(x["mask"]).convert("L"))>127;band=binary_dilation(m,iterations=2)^binary_erosion(m,iterations=2);rg=grad(real_full)
   for arm in ARMS:
    syn_full=Image.open(x[arm]).convert("RGB");syn=syn_full.crop(box);lp=float(net(tf(real)[None].to(dev),tf(syn)[None].to(dev)).item());raw.append({"instance_id":x["instance_id"],"class_name":x["class_name"],"seed":x["generation_seed"],"arm":arm,"lpips":lp,"ssim":ssim_value(real,syn),"edge_cosine":edge_cos(real,syn),"boundary_gradient_cosine":cos(rg[band],grad(syn_full)[band])})
 del net;torch.cuda.empty_cache();fields=("lpips","ssim","edge_cosine","boundary_gradient_cosine");return raw,{a:{"overall":summarize([x for x in raw if x["arm"]==a],fields),"per_class":{c:summarize([x for x in raw if x["arm"]==a and x["class_name"]==c],fields) for c in CLASSES}} for a in ARMS}
def feature_rows(rows,a):
 folds=load(a.folds);foldof={pid:f["fold"] for f in folds["folds"] for pid in f["holdout_pair_ids"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");refs=load(a.stage3r/"manifold_reference_corrected.json")["references"];out=[]
 for entry in bankdoc["folds"]:
  fold=entry["fold"]
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz") as z:real,ids=z["features"],z["class_ids"]
  source={x["instance_id"]:i for i,x in enumerate(entry["records"])};ext=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512)
  try:
   for x in rows:
    if foldof[x["pair_id"]]!=fold:continue
    box=crop_box(x["mask"])[0];ci=1 if x["class_name"]=="short" else 5;q95=float(refs[f"fold{fold}:{x['class_name']}"]["q95_real_distance"]);rf=l2_normalize(real[source[x["instance_id"]]])
    for arm in ARMS:
     f=l2_normalize(ext.encode(x[arm],box,512,512));gap=float(1-np.dot(f,rf));out.append({"instance_id":x["instance_id"],"class_name":x["class_name"],"seed":x["generation_seed"],"arm":arm,"feature":f.tolist(),"normalized_pair_gap":gap/max(q95,1e-12),"fraction_gt_real_q95":gap>q95})
  finally:ext.close()
 return out
def task_summary(rows):
 def one(z):return {"normalized_pair_gap":stat([x["normalized_pair_gap"] for x in z]),"fraction_gt_real_q95":float(np.mean([x["fraction_gt_real_q95"] for x in z])),"n":len(z)}
 return {a:{"overall":one([x for x in rows if x["arm"]==a]),"per_class":{c:one([x for x in rows if x["arm"]==a and x["class_name"]==c]) for c in CLASSES}} for a in ARMS}
def identity(features):
 out={}
 for arm in ARMS:
  out[arm]={}
  for label,cls in [("overall",None)]+[(c,c) for c in CLASSES]:
   z=[x for x in features if x["arm"]==arm and (cls is None or x["class_name"]==cls)];groups=defaultdict(list)
   for x in z:groups[x["instance_id"]].append(np.asarray(x["feature"]))
   top1=[];top5=[]
   for x in z:
    f=np.asarray(x["feature"]);cent=[]
    for iid,vals in groups.items():
     use=[v for v in vals if not (iid==x["instance_id"] and np.array_equal(v,f))] if iid==x["instance_id"] else vals
     cent.append((iid,l2_normalize(np.mean(use or vals,axis=0))))
    order=sorted(cent,key=lambda q:1-float(np.dot(f,q[1])));top1.append(order[0][0]==x["instance_id"]);top5.append(x["instance_id"] in [q[0] for q in order[:5]])
   allf=np.stack([np.asarray(x["feature"]) for x in z]);grand=allf.mean(0);within=sum(float(((np.stack(v)-np.stack(v).mean(0))**2).sum()) for v in groups.values());between=sum(len(v)*float(((np.stack(v).mean(0)-grand)**2).sum()) for v in groups.values());total=within+between;out[arm][label]={"retrieval_top1":float(np.mean(top1)),"retrieval_top5":float(np.mean(top5)),"within_over_total":within/total,"between_over_total":between/total,"source_identity_R2":between/total,"n":len(z)}
 return out
def diversity(rows,features,dev):
 tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);net=lpips.LPIPS(net="alex",verbose=False).to(dev).eval();raw=[];groups=defaultdict(list)
 for x in rows:groups[(x["instance_id"],x["class_name"])].append(x)
 fmap={(x["instance_id"],x["seed"],x["arm"]):np.asarray(x["feature"]) for x in features}
 with torch.no_grad():
  for (iid,cls),g in groups.items():
   g=sorted(g,key=lambda x:x["generation_seed"])
   for arm in ARMS:
    ims=[];fs=[];edges=[]
    for x in g:
     box=crop_box(x["mask"])[0];im=Image.open(x[arm]).convert("RGB").crop(box);ims.append(tf(im));fs.append(fmap[(iid,x["generation_seed"],arm)]);edges.append(sobel(im))
    pairs=[(0,1),(0,2),(1,2)];ic=[float(net(ims[i][None].to(dev),ims[j][None].to(dev)).item()) for i,j in pairs];fd=[float(1-np.dot(fs[i],fs[j])) for i,j in pairs];ed=[float(np.mean(np.abs(edges[i]-edges[j]))) for i,j in pairs];raw.append({"instance_id":iid,"class_name":cls,"arm":arm,"ic_lpips":float(np.mean(ic)),"feature_variance":float(np.stack(fs).var(0).mean()),"pairwise_feature_distance":float(np.mean(fd)),"edge_diversity":float(np.mean(ed))})
 fields=("ic_lpips","feature_variance","pairwise_feature_distance","edge_diversity");return raw,{a:{"overall":summarize([x for x in raw if x["arm"]==a],fields),"per_class":{c:summarize([x for x in raw if x["arm"]==a and x["class_name"]==c],fields) for c in CLASSES}} for a in ARMS}
def main():
 p=argparse.ArgumentParser();
 for n in ("manifest","adapter_root","folds","stage3r","stage3r_runs","gate_protocol","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");p.add_argument("--batch",type=int,default=16);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=load(a.manifest)["records"];dev=torch.device("cuda:"+a.device)
 assert len(rows)==258*3 and len({(x["instance_id"],x["generation_seed"]) for x in rows})==len(rows);save(a.output/"source_seed_pairing_audit.json",{"exact_pairing":True,"records":len(rows),"sources":len({x["instance_id"] for x in rows}),"per_class":{c:sum(x["class_name"]==c for x in rows) for c in CLASSES},"per_seed":{str(s):sum(x["generation_seed"]==s for x in rows) for s in (42,2026,3407)},"official_validation_use_count":0})
 fr,fm=fidelity(rows,dev,a.batch);save(a.output/"canonical_fidelity_metrics.json",{"protocol":{"crop":"adaptive padded max(64,4x bbox), crop then resize 256","lpips":"AlexNet RGB [-1,1]"},"metrics":fm,"raw_records":fr});features=feature_rows(rows,a);tm=task_summary(features);save(a.output/"task_feature_support_metrics.json",{"metrics":tm,"raw_records":[{k:v for k,v in x.items() if k!="feature"} for x in features]});im=identity(features);save(a.output/"source_identity_metrics.json",{"metrics":im,"direction":{"source_identity_R2":"higher means stronger source identity","within_over_total":"lower means tighter same-source clusters"}});dr,dm=diversity(rows,features,dev);save(a.output/"diversity_metrics.json",{"metrics":dm,"raw_records":dr})
 audits={};changed=True
 for c in CLASSES:
  t=load(a.adapter_root/c/"training_audit.json");audits[c]=t;changed&=t["successful_updates"]==t["actual_parameter_update_count"]==2000 and t["finite_fraction"]==1 and t["total_adapter_delta_vs_initialization"]>0 and t["projection_parameter_norm"]>0
 save(a.output/"formal_training_audit.json",audits);save(a.output/"checkpoint_parameter_audit.json",{"checkpoint_changed_from_initialization":changed,"per_class":{c:{k:audits[c][k] for k in ("adapter_parameters","projection_parameter_norm","total_adapter_delta_vs_initialization","finite_fraction")} for c in CLASSES}})
 task_ok=all(tm["repaired_frozen_adapter"]["per_class"][c]["normalized_pair_gap"]["mean"]<tm["pretrained_no_adaptation"]["per_class"][c]["normalized_pair_gap"]["mean"] and tm["repaired_frozen_adapter"]["per_class"][c]["fraction_gt_real_q95"]<tm["pretrained_no_adaptation"]["per_class"][c]["fraction_gt_real_q95"] for c in CLASSES);manifold_ok=all(tm["repaired_frozen_adapter"]["per_class"][c]["fraction_gt_real_q95"]<1 for c in CLASSES);fidelity_ok=fm["repaired_frozen_adapter"]["overall"]["lpips"]["mean"]<=fm["pretrained_no_adaptation"]["overall"]["lpips"]["mean"] and fm["repaired_frozen_adapter"]["overall"]["edge_cosine"]["mean"]>=fm["pretrained_no_adaptation"]["overall"]["edge_cosine"]["mean"]
 status="REPAIRED_FROZEN_ADAPTER_FEASIBLE" if changed and task_ok and manifold_ok and fidelity_ok else "REPAIRED_FROZEN_ADAPTER_MIXED" if changed and (task_ok or fidelity_ok) else "REPAIRED_FROZEN_ADAPTER_NOT_FEASIBLE";gate={"status":status,"training_valid":changed,"task_support_both_classes":task_ok,"real_manifold_sanity":manifold_ok,"fidelity_not_overall_worse":fidelity_ok,"STAGE9F_DETECTOR_PILOT_AUTHORIZED":status=="REPAIRED_FROZEN_ADAPTER_FEASIBLE","detector_training_count":0,"official_validation_use_count":0};save(a.output/"stage9e_gate.json",gate);protocol=load(a.gate_protocol);assert protocol["frozen_before_metrics"] and protocol["official_validation_use_count"]==0;save(a.output/"stage9e_gate_protocol.json",protocol);save(a.output/"per_class_metrics.json",{c:{"fidelity":{arm:fm[arm]["per_class"][c] for arm in ARMS},"task":{arm:tm[arm]["per_class"][c] for arm in ARMS},"identity":{arm:im[arm][c] for arm in ARMS},"diversity":{arm:dm[arm]["per_class"][c] for arm in ARMS}} for c in CLASSES});(a.output/"STAGE9E_REPORT.md").write_text(f"# DeepPCB Stage9E Numerical Repair and True Frozen-Adapter Feasibility\n\nStatus: **{status}**.\n\nStage9A was invalid as a test of the lightweight-adaptation hypothesis because a numerical backward singularity kept the residual adapter at initialization. Stage9E provides the first valid empirical test after a numerical-only repair.\n\n- Training valid: {changed}\n- Task support improves in both classes: {task_ok}\n- Real-manifold sanity: {manifold_ok}\n- Fidelity not overall worse: {fidelity_ok}\n- Stage9F detector authorized: {gate['STAGE9F_DETECTOR_PILOT_AUTHORIZED']}\n\nNo detector or official validation was used.\n");print(json.dumps(gate,indent=2))
if __name__=="__main__":main()
