"""Freeze RCCRepeat120, prove coverage manipulation, and build one replay dataset."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_deeppcb_stage11b import isolate,link,sha,tree_digest

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def frozen(p,x):
 data=(json.dumps(x,indent=2,allow_nan=False)+"\n").encode();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=data:raise RuntimeError("STAGE12B_FROZEN_ARTIFACT_CONFLICT: "+str(p))
 p.write_bytes(data)
def quota(counts,total=20):
 raw={f:total*counts[f]/sum(counts.values()) for f in counts};q={f:max(1,int(math.floor(raw[f]))) for f in counts}
 while sum(q.values())<total:q[max(q,key=lambda f:(raw[f]-q[f],-f))]+=1
 while sum(q.values())>total:
  choices=[f for f in q if q[f]>1];q[min(choices,key=lambda f:(raw[f]-q[f],f))]-=1
 return q,raw
def stats(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"q75":float(np.quantile(a,.75)),"q90":float(np.quantile(a,.9)),"fraction_z_le_1":float(np.mean(a<=1)),"fraction_z_le_1_5":float(np.mean(a<=1.5)),"n":len(a)}
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","registry","stage3r","stage10ar","stage11b","real_dataset","dataset_root","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();cfg=load(a.protocol);a.output.mkdir(parents=True,exist_ok=True);save(a.output/"stage12b_protocol.json",cfg);reg=load(a.registry)["instances"];idx={x["instance_id"]:x for x in reg};rad={x["instance_id"]:x for x in load(a.stage10ar/"pair_independent_local_radius.json")["records"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");groups={};budget=[]
 for e in bankdoc["folds"]:
  f=int(e["fold"])
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{f}.npz") as z:features=z["features"].astype(float)
  features/=np.linalg.norm(features,axis=1,keepdims=True).clip(1e-12)
  for ci,c in enumerate(CLASSES):
   jj=[i for i,r in enumerate(e["records"]) if int(r["class_id"])==ci];rr=[e["records"][i] for i in jj];groups[(f,ci)]={"records":rr,"features":features[jj]}
 for ci,c in enumerate(CLASSES):
  counts={f:len(groups[(f,ci)]["records"]) for f in range(3)};qq,raw=quota(counts)
  for f in range(3):budget.append({"class_name":c,"class_id":ci,"fold":f,"real_instance_count":counts[f],"raw_quota":raw[f],"quota":qq[f]})
 frozen(a.output/"fold_class_budget.json",{"method":"largest remainder with minimum one per fold-class","class_total":20,"total":120,"records":budget})
 selected=[]
 for b in budget:
  f,ci,n=b["fold"],b["class_id"],b["quota"];g=groups[(f,ci)];R=g["records"];X=g["features"];pairs=np.asarray([x["pair_id"] for x in R]);rho=np.asarray([rad[x["instance_id"]]["k5_radius"] for x in R]);D=(1-np.clip(X@X.T,-1,1))/rho[:,None];chosen=[];current=None
  for step in range(n):
   candidates=[j for j in range(len(R)) if j not in chosen and R[j]["pair_id"] not in {R[k]["pair_id"] for k in chosen}];scores=[]
   before=None if current is None else stats(current)
   for j in candidates:
    col=D[:,j].copy();col[pairs==pairs[j]]=np.inf;after=col if current is None else np.minimum(current,col);finite=after[np.isfinite(after)];s=stats(finite);scores.append((s["q90"],s["median"],s["mean"],R[j]["instance_id"],j,after,s))
   best=min(scores,key=lambda x:x[:4]);j,current,s=best[4],best[5],best[6];chosen.append(j);selected.append({"instance_id":R[j]["instance_id"],"pair_id":R[j]["pair_id"],"class_name":b["class_name"],"class_id":ci,"fold":f,"quota_group":f"fold{f}:{b['class_name']}","selection_step":step+1,"q90_before":None if before is None else before["q90"],"q90_after":s["q90"],"median_after":s["median"],"mean_after":s["mean"]})
 if len(selected)!=120 or Counter(x["class_name"] for x in selected)!=Counter({c:20 for c in CLASSES}) or any(len({x["pair_id"] for x in selected if x["class_name"]==c})!=20 for c in CLASSES):raise RuntimeError("STAGE12B_SELECTION_INVALID")
 doc={"arm":"rcc","objective":"minimize q90 only","tie_break":cfg["tie_break"],"count":120,"class_counts":dict(Counter(x["class_name"] for x in selected)),"physical_pair_cap_per_class":1,"records":selected};frozen(a.output/"rcc_selection.json",doc);frozen(a.output/"selection_freeze_audit.json",{"status":"RCC_SELECTION_FROZEN_BEFORE_TRAINING","selection_sha256":sha(a.output/"rcc_selection.json"),"validation_used":False,"support_ess_gradient_inspected_during_selection":False})
 uniform=load(a.stage11b/"uniform_selection.json")["records"]
 def coverage(sel):
  by=defaultdict(list)
  for x in sel:by[(int(x["fold"]),int(x["class_id"]))].append(x)
  allz=[];per={}
  for ci,c in enumerate(CLASSES):
   cz=[]
   for f in range(3):
    g=groups[(f,ci)];R,X=g["records"],g["features"];lookup={x["instance_id"]:i for i,x in enumerate(R)};S=by[(f,ci)];J=[lookup[x["instance_id"]] for x in S];sp=np.asarray([R[j]["pair_id"] for j in J]);rp=np.asarray([x["pair_id"] for x in R]);rho=np.asarray([rad[x["instance_id"]]["k5_radius"] for x in R]);D=(1-np.clip(X@X[J].T,-1,1))/rho[:,None];D[rp[:,None]==sp[None,:]]=np.inf;z=D.min(1);cz.extend(z.tolist())
   per[c]=stats(cz);allz.extend(cz)
  return {"overall":stats(allz),"per_class":per}
 cu,cr=coverage(uniform),coverage(selected);imp=(cu["overall"]["q90"]-cr["overall"]["q90"])/cu["overall"]["q90"];wins=sum(cr["per_class"][c]["q90"]<cu["per_class"][c]["q90"] for c in CLASSES);passed=cr["overall"]["q90"]<cu["overall"]["q90"] and wins>=cfg["coverage_improved_classes_min"] and imp>=cfg["coverage_relative_improvement_min"]
 manipulation={"status":"RCC_MANIPULATION_PASS" if passed else "RCC_MANIPULATION_INSUFFICIENT","uniform":cu,"rcc":cr,"relative_q90_improvement":imp,"improved_class_count":wins,"requirements":{"overall_strictly_lower":True,"minimum_improved_classes":5,"minimum_relative_improvement":.05}};save(a.output/"coverage_manipulation_audit.json",manipulation)
 if not passed:save(a.output/"stage12b_status.json",{"status":"RCC_MANIPULATION_INSUFFICIENT","STAGE12C_SYNTHETIC_COVERAGE_SELECTION_AUTHORIZED":False,"detector_training_count":0});raise RuntimeError("RCC_MANIPULATION_INSUFFICIENT")
 old={x["instance_id"]:x for x in load(a.stage11b/"replay_isolation_audit.json")["records"]};by_pair=defaultdict(list)
 for x in reg:by_pair[x["pair_id"]].append(x)
 root=a.dataset_root/"isolated";aud=[];reused=0;created=0
 for x in selected:
  iid=x["instance_id"]
  if iid in old and Path(old[iid]["image_path"]).is_file() and Path(old[iid]["label_path"]).is_file():r=old[iid];ip,lp=Path(r["image_path"]),Path(r["label_path"]);reused+=1
  else:
   row=idx[iid];image,r=isolate(row,by_pair);ip=root/"images"/(iid+".png");lp=root/"labels"/(iid+".txt");ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(image).save(ip);w,h=image.shape[1],image.shape[0];x0,y0,x1,y1=row["bbox_xyxy"];lp.write_text(f"{row['class_id']} {(x0+x1)/(2*w):.10f} {(y0+y1)/(2*h):.10f} {(x1-x0)/w:.10f} {(y1-y0)/h:.10f}\n");created+=1
  aud.append({"instance_id":iid,"image_path":str(ip.resolve()),"label_path":str(lp.resolve()),"reused_from_stage11b":iid in old,"image_sha256":sha(ip),"label_sha256":sha(lp)})
 save(a.output/"replay_image_reuse_audit.json",{"status":"PASS","selected":120,"reused":reused,"created_by_frozen_stage11b_isolation":created,"synthetic_generation_count":0,"records":aud});am={x["instance_id"]:x for x in aud};base_i={p.stem:p for p in (a.real_dataset/"images/train").iterdir()};base_l={p.stem:p for p in (a.real_dataset/"labels/train").iterdir()};ds=a.dataset_root/"rcc"
 for i,s in enumerate(sorted(base_i)):link(base_i[s],ds/"images/train"/f"base_{i:03d}{base_i[s].suffix}");link(base_l[s],ds/"labels/train"/f"base_{i:03d}.txt")
 for i,x in enumerate(selected):link(am[x["instance_id"]]["image_path"],ds/"images/train"/f"extra_{i:03d}.png");link(am[x["instance_id"]]["label_path"],ds/"labels/train"/f"extra_{i:03d}.txt")
 yaml=f"path: {ds.resolve()}\ntrain: images/train\nval: images/train\nnc: 6\nnames: {json.dumps(list(CLASSES))}\n";(ds/"data.yaml").write_text(yaml);print(json.dumps({"status":"STAGE12B_PREPARATION_COMPLETE","coverage":manipulation,"selection_sha256":sha(a.output/"rcc_selection.json")},indent=2))
if __name__=="__main__":main()
