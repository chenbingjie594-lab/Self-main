"""Image-property audit and frozen Stage14A subset construction (no detector training)."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES={0:"flash",1:"black"}
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def save(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+"\n",encoding="utf-8")
def q(v,p):return float(np.quantile(np.asarray(v,float),p)) if v else None
def summary(v):return {"n":len(v),"mean":float(np.mean(v)),"median":q(v,.5),"q90":q(v,.9),"std":float(np.std(v))} if v else {"n":0,"mean":None,"median":None,"q90":None,"std":None}
def gray(a):return (.299*a[...,0]+.587*a[...,1]+.114*a[...,2]).astype(np.float32)
def grad(g):
 gy,gx=np.gradient(g);return np.hypot(gx,gy)
def metrics(path,box):
 a=np.asarray(Image.open(path).convert("RGB"),dtype=np.float32);h,w=a.shape[:2];x0,y0,x1,y1=map(int,box);x0=max(0,x0);y0=max(0,y0);x1=min(w,max(x0+1,x1));y1=min(h,max(y0+1,y1));bw=x1-x0;bh=y1-y0
 cx0=max(0,int(x0-.75*bw));cy0=max(0,int(y0-.75*bh));cx1=min(w,int(x1+.75*bw));cy1=min(h,int(y1+.75*bh));g=gray(a);inside=g[y0:y1,x0:x1];ctx=g[cy0:cy1,cx0:cx1];ring=np.ones(ctx.shape,bool);ring[y0-cy0:y1-cy0,x0-cx0:x1-cx0]=False;r=ctx[ring];gm=grad(g);band=np.zeros((h,w),bool);band[max(0,y0-2):min(h,y1+2),max(0,x0-2):min(w,x1+2)]=1;band[y0+2:max(y0+2,y1-2),x0+2:max(x0+2,x1-2)]=0
 f=np.abs(np.fft.rfft2(inside-inside.mean()))**2 if inside.size>3 else np.zeros((1,1));cut=max(1,min(f.shape)//4);lf=float(f[:cut,:cut].mean());hf=float(f.mean());rgb_in=a[y0:y1,x0:x1].mean((0,1));rgb_bg=a[cy0:cy1,cx0:cx1][ring].mean(0)
 return {"brightness":float(r.mean()),"texture":float(r.std()),"contrast_ratio":float(abs(inside.mean()-r.mean())/(r.std()+1e-6)),"boundary_gradient":float(gm[band].mean()),"hf_lf_ratio":float(hf/(lf+1e-6)),"color_residual":float(np.linalg.norm(rgb_in-rgb_bg)/255),"width":bw,"height":bh,"area":bw*bh,"aspect_ratio":bw/max(bh,1),"edge_density":float((grad(inside)>np.percentile(grad(inside),75)).mean())}
def canonical(name):
 s=str(name).lower().replace("_"," ").replace("-"," ")
 if "flash" in s:return "flash"
 if "black" in s:return "black"
 return None
def yaml_doc(p):
 try:
  import yaml
  return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
 except Exception as e:raise RuntimeError(f"STAGE14A_DATA_YAML_INVALID: {p}: {e}") from e
def dataset_names(root):
 p=root if Path(root).is_file() else Path(root)/"data.yaml"
 if not p.exists():return dict(CLASSES)
 names=yaml_doc(p).get("names",{})
 if isinstance(names,list):return {i:canonical(x) for i,x in enumerate(names)}
 return {int(i):canonical(x) for i,x in names.items()}
def yaml_layout(p):
 p=Path(p);d=yaml_doc(p);base=Path(d.get("path",p.parent))
 if not base.is_absolute():base=(p.parent/base).resolve()
 train=d.get("train","images/train");train=train[0] if isinstance(train,list) else train
 image_dir=Path(train);image_dir=image_dir if image_dir.is_absolute() else base/image_dir
 label_dir=Path(str(image_dir).replace("/images/","/labels/").replace("\\images\\","\\labels\\"))
 return p.parent,image_dir,label_dir
def resolve_layout(requested,historical_args=None):
 requested=Path(requested)
 if requested.is_file():return yaml_layout(requested)
 if (requested/"images/train").is_dir() and (requested/"labels/train").is_dir():return requested,requested/"images/train",requested/"labels/train"
 if (requested/"data.yaml").is_file():
  z=yaml_layout(requested/"data.yaml")
  if z[1].is_dir() and z[2].is_dir():return z
 if historical_args and Path(historical_args).is_file():
  d=yaml_doc(historical_args);recorded=Path(str(d.get("data","")))
  if not recorded.is_absolute():recorded=(Path(historical_args).parent/recorded).resolve()
  if recorded.is_file():
   z=yaml_layout(recorded)
   if z[1].is_dir() and z[2].is_dir():return z
 raise RuntimeError("STAGE14A_REAL_DATASET_UNRESOLVED: "+json.dumps({"requested":str(requested),"historical_args":str(historical_args) if historical_args else None,"policy":"No filesystem-wide fallback is allowed because it could select DeepPCB."}))
def yolo_real(root,historical_args=None):
 root,image_dir,label_dir=resolve_layout(root,historical_args);names=dataset_names(root)
 out=[]
 for ip in sorted(image_dir.glob("*")):
  if not ip.is_file():continue
  with Image.open(ip) as im:w,h=im.size
  lp=label_dir/(ip.stem+".txt")
  if not lp.exists():raise RuntimeError(f"STAGE14A_REAL_LABEL_MISSING: {lp}")
  for j,line in enumerate(lp.read_text().splitlines()):
   z=line.split();c=int(float(z[0]));class_name=names.get(c)
   if class_name not in CLASSES.values():continue
   cx,cy,bw,bh=map(float,z[1:5]);box=((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h);out.append({"id":f"{ip.stem}:{j}","class_id":c,"class_name":class_name,"dataset_class_name":class_name,"image_path":str(ip),"bbox_xyxy":box,**metrics(ip,box)})
 return out
def rank_pick(rows,key,n=40):return sorted(rows,key=lambda x:(key(x),x["candidate_id"]))[:n]
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--candidate_pool",type=Path,required=True);p.add_argument("--random_manifest",type=Path,required=True);p.add_argument("--real_dataset",type=Path,required=True);p.add_argument("--historical_args",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);pool=load(a.candidate_pool);syn=pool.get("candidates",pool.get("selected",[]));real=yolo_real(a.real_dataset,a.historical_args)
 for x in syn:x["class_name"]=CLASSES[int(x["class_id"])];x.update(metrics(x["image_path"],x["bbox_xyxy"]))
 bys=defaultdict(list);byr=defaultdict(list)
 for x in syn:bys[x["class_name"]].append(x)
 for x in real:byr[x["class_name"]].append(x)
 resolved_root,resolved_images,resolved_labels=resolve_layout(a.real_dataset,a.historical_args);real_counts={c:len(byr[c]) for c in CLASSES.values()}
 if any(real_counts[c]==0 for c in CLASSES.values()):
  save(a.output/"real_training_parse_audit.json",{"status":"FAILED_MISSING_TARGET_CLASS","data_yaml_names":dataset_names(resolved_root),"parsed_target_objects":real_counts,"requested_real_dataset":str(a.real_dataset),"resolved_root":str(resolved_root),"resolved_images":str(resolved_images),"resolved_labels":str(resolved_labels)})
  raise RuntimeError("STAGE14A_REAL_TARGET_CLASS_MISSING: "+json.dumps(real_counts))
 save(a.output/"real_training_parse_audit.json",{"status":"PASS","data_yaml_names":dataset_names(resolved_root),"parsed_target_objects":real_counts,"requested_real_dataset":str(a.real_dataset),"resolved_root":str(resolved_root),"resolved_images":str(resolved_images),"resolved_labels":str(resolved_labels)})
 fields=("contrast_ratio","boundary_gradient","hf_lf_ratio","color_residual","width","height","area","aspect_ratio","edge_density")
 fidelity={"status":"COMPLETE","crop_kid":{"status":"NOT_COMPUTED_NO_CANONICAL_FEATURE_EXTRACTOR","reason":"No repository-frozen Plastic_Bomo crop feature extractor exists; not replaced by an ad-hoc network."},"real_nearest_lpips":{"status":"NOT_COMPUTED_NO_FROZEN_PAIR_PROTOCOL","reason":"No repository-frozen crop alignment/LPIPS protocol exists; no proxy is mislabeled LPIPS."},"metrics":{},"raw_records":syn}
 context={"status":"COMPLETE","context_scale":2.5,"metrics":{},"context_conditioning":{},"raw_records":[]}
 subsets={"random":[x["candidate_id"] for x in load(a.random_manifest)["selected"]],"high_fidelity":[],"valid_novel":[],"high_context_compatibility":[],"low_fidelity_diagnostic":[],"duplicate_like_diagnostic":[],"off_manifold_diagnostic":[],"low_context_diagnostic":[]};capacity={}
 for c in CLASSES.values():
  rr,ss=byr[c],bys[c];fidelity["metrics"][c]={k:{"real":summary([x[k] for x in rr]),"synthetic":summary([x[k] for x in ss])} for k in fields}
  med={k:q([x[k] for x in rr],.5) for k in fields};iqr={k:max(q([x[k] for x in rr],.75)-q([x[k] for x in rr],.25),1e-6) for k in fields}
  for x in ss:
   x["morphology_distance"]=max(abs(x[k]-med[k])/iqr[k] for k in ("width","height","aspect_ratio","edge_density"));x["context_distance"]=max(abs(x[k]-med[k])/iqr[k] for k in ("contrast_ratio","boundary_gradient","hf_lf_ratio","color_residual"))
  hi=rank_pick(ss,lambda x:x["morphology_distance"]);lo=rank_pick(ss,lambda x:-x["morphology_distance"]);hc=rank_pick(ss,lambda x:x["context_distance"]);lc=rank_pick(ss,lambda x:-x["context_distance"])
  bins={"duplicate_like_diagnostic":[x for x in ss if x["median_normalized_manifold_distance"]<=.25],"valid_novel":[x for x in ss if .25<x["median_normalized_manifold_distance"]<=.95],"off_manifold_diagnostic":[x for x in ss if x["median_normalized_manifold_distance"]>.95]}
  for name,z in (("high_fidelity",hi),("low_fidelity_diagnostic",lo),("high_context_compatibility",hc),("low_context_diagnostic",lc)):subsets[name]+=[x["candidate_id"] for x in z[:40]]
  for name,z in bins.items():subsets[name]+=[x["candidate_id"] for x in sorted(z,key=lambda x:x["candidate_id"])[:40]];capacity.setdefault(name,{})[c]=len(z)
  context["metrics"][c]={k:{"real":summary([x[k] for x in rr]),"synthetic":summary([x[k] for x in ss])} for k in ("contrast_ratio","boundary_gradient","hf_lf_ratio","color_residual")}
  for origin,z in (("real",rr),("synthetic",ss)):
   br=np.quantile([x["brightness"] for x in z],[1/3,2/3]);tr=np.quantile([x["texture"] for x in z],[1/3,2/3]);context["context_conditioning"].setdefault(c,{})[origin]={"brightness_terciles":br.tolist(),"texture_terciles":tr.tolist()}
 task={c:{k:summary([float(x[k]) for x in bys[c]]) for k in ("median_confidence","median_iou","median_normalized_manifold_distance","consensus_score")} for c in CLASSES.values()}
 diversity={"status":"COMPLETE","thresholds_frozen_before_screening":{"duplicate_like_max":.25,"valid_novel_max":.95},"capacity":capacity,"per_class":{c:{"normalized_nearest_real_distance":summary([x["median_normalized_manifold_distance"] for x in bys[c]]),"bin_counts":dict(Counter("duplicate_like" if x["median_normalized_manifold_distance"]<=.25 else "valid_novel" if x["median_normalized_manifold_distance"]<=.95 else "off_manifold" for x in bys[c]))} for c in CLASSES.values()},"ic_lpips":{"status":"NOT_COMPUTED_NO_FROZEN_PAIR_PROTOCOL"},"feature_pairwise_distance":{"status":"NOT_AVAILABLE_POOL_STORES_ONLY_NEAREST_REAL_DISTANCE"}}
 eligible={k:(len(v)==80 and all(sum(next(x for x in syn if x["candidate_id"]==i)["class_name"]==c for i in v)==40 for c in CLASSES.values())) for k,v in subsets.items()};save(a.output/"fidelity_audit.json",fidelity);save(a.output/"context_compatibility_audit.json",context);save(a.output/"effective_diversity_audit.json",diversity);save(a.output/"synthetic_task_response.json",{"status":"COMPLETE_OOF_VALIDATION_INDEPENDENT","metrics":task,"official_validation_used":False});save(a.output/"subset_definition.json",{"status":"FROZEN_BEFORE_STAGE14A_SCREENING","selection_uses_downstream_results":False,"weighted_quality_score_used":False,"definitions":cfg["selection_policy"],"subsets":subsets,"capacity":capacity,"eligible_equal_budget":eligible});save(a.output/"subset_overlap_audit.json",{x:{y:len(set(subsets[x])&set(subsets[y])) for y in subsets} for x in subsets});print(json.dumps({"status":"STAGE14A_IMAGE_AUDIT_COMPLETE","eligible_equal_budget":eligible},indent=2))
if __name__=="__main__":main()
