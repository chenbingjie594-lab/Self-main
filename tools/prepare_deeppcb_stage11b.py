"""Freeze Stage11B allocations and build target-isolated real replay datasets."""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
ARMS=("uniform","feature","hardness","gradient")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def encoded(x):return (json.dumps(x,indent=2,allow_nan=False)+"\n").encode()
def frozen_write(p,x):
 data=encoded(x);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=data:raise RuntimeError("STAGE11B_FROZEN_ARTIFACT_CONFLICT: "+str(p))
 p.write_bytes(data)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def tree_digest(paths):
 h=hashlib.sha256()
 for p in sorted(map(Path,paths),key=lambda x:str(x)):
  h.update(p.name.encode());h.update(sha(p).encode())
 return h.hexdigest()
def mask(p):return np.asarray(Image.open(p).convert("L"))>127
def tight(m):
 y,x=np.where(m)
 if not len(x):raise RuntimeError("STAGE11B_EMPTY_MASK")
 return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
def percentiles(values):
 a=np.asarray(values,float);order=np.argsort(a,kind="stable");out=np.empty(len(a),float);i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]:j+=1
  rank=(i+j-1)/2;out[order[i]]=rank/max(1,len(a)-1);i=j
 return out
def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():
  if dst.resolve()!=src:raise RuntimeError("STAGE11B_EXISTING_LINK_CONFLICT: "+str(dst))
 else:dst.symlink_to(src)
def select_ranked(rows,key):
 out=[]
 for c in CLASSES:
  candidates=sorted((x for x in rows if x["class_name"]==c),key=lambda x:(-x[key],x["pair_id"],x["instance_id"]));best={}
  for x in candidates:best.setdefault(x["pair_id"],x)
  picked=sorted(best.values(),key=lambda x:(-x[key],x["pair_id"],x["instance_id"]))[:20]
  if len(picked)!=20:raise RuntimeError("STAGE11B_PAIR_DIVERSITY_INFEASIBLE: "+c)
  out.extend(picked)
 return out
def select_uniform(rows,seed):
 rng=random.Random(seed);out=[]
 for c in CLASSES:
  by=defaultdict(list)
  for x in rows:
   if x["class_name"]==c:by[x["pair_id"]].append(x)
  chosen_pairs=rng.sample(sorted(by),20)
  for pid in chosen_pairs:out.append(rng.choice(sorted(by[pid],key=lambda x:x["instance_id"])))
 return out
def selection_doc(arm,rows):
 keep=("instance_id","pair_id","class_name","class_id","fold","gradient_percentile","feature_percentile","hardness_percentile")
 records=[{k:x[k] for k in keep} for x in rows];counts=Counter(x["class_name"] for x in records)
 pair_cap=all(len({x["pair_id"] for x in records if x["class_name"]==c})==20 for c in CLASSES)
 valid=len(records)==120 and counts==Counter({c:20 for c in CLASSES}) and len({x["instance_id"] for x in records})==120 and pair_cap
 if not valid:raise RuntimeError("STAGE11B_SELECTION_INVALID: "+arm)
 return {"arm":arm,"count":120,"class_counts":dict(counts),"duplicate_instances":0,"physical_pair_cap_per_class":1,"pair_cap_satisfied":pair_cap,"records":records}
def isolate(row,by_pair):
 defect=np.asarray(Image.open(row["defect_image"]).convert("RGB"));normal=np.asarray(Image.open(row["paired_normal_image"]).convert("RGB"));tm=mask(row["instance_mask_path"]);missing=[];other=np.zeros_like(tm)
 if defect.shape!=normal.shape or tm.shape!=defect.shape[:2]:raise RuntimeError("STAGE11B_PAIRED_SIZE_MISMATCH: "+row["instance_id"])
 for q in by_pair[row["pair_id"]]:
  if q["instance_id"]==row["instance_id"]:continue
  p=Path(q["instance_mask_path"])
  if not p.is_file():missing.append(q["instance_id"]);continue
  om=mask(p)
  if om.shape!=tm.shape:missing.append(q["instance_id"]);continue
  other|=om
 replace=other&~tm;out=defect.copy();out[replace]=normal[replace];changed=np.any(out!=defect,axis=2)
 audit={"instance_id":row["instance_id"],"pair_id":row["pair_id"],"class_name":row["class_name"],"target_pixels_unchanged":bool(np.all(out[tm]==defect[tm])),"other_known_defects_restored":bool(np.all(out[replace]==normal[replace])) if replace.any() else True,"changed_pixels_outside_other_defect_masks":int((changed&~replace).sum()),"paired_normal_size_match":defect.shape==normal.shape,"target_bbox_unchanged":tight(tm)==row["bbox_xyxy"],"missing_masks":missing,"full_resolution":[int(defect.shape[1]),int(defect.shape[0])],"bbox_xyxy":row["bbox_xyxy"]}
 return out,audit
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","stage11a","stage10ar","registry","real_dataset","dataset_root","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();cfg=load(a.protocol);status=load(a.stage11a/"stage11a_status.json");assert status["status"]==cfg["status_prerequisite"] and status["STAGE11B_ALLOCATION_CALIBRATION_AUTHORIZED"]
 a.output.mkdir(parents=True,exist_ok=True);registry=load(a.registry)["instances"];assert len(registry)==775;idx={x["instance_id"]:dict(x) for x in registry};grad=load(a.stage11a/"gradient_local_radius.json")["records"];feat=load(a.stage10ar/"pair_independent_local_radius.json")["records"];loss=load(a.stage11a/"raw_gradient_diagnostics.json")["records"]
 distribution=load(a.stage11a/"gradient_coverage_distribution.json");bank=np.load(a.stage11a/"real_oof_gradient_embeddings.npz");bank_ids=bank["instance_ids"].astype(str)
 if distribution["status"]!="GRADIENT_COVERAGE_HETEROGENEITY_SUPPORTED" or bank["embeddings"].shape!=(775,1024) or set(bank_ids)!=set(idx):raise RuntimeError("STAGE11B_FROZEN_STAGE11A_INPUT_INVALID")
 if not (len(grad)==len(feat)==len(loss)==775):raise RuntimeError("STAGE11B_INPUT_CARDINALITY_MISMATCH")
 gm={x["instance_id"]:x for x in grad};fm={x["instance_id"]:x for x in feat};hm={x["instance_id"]:x for x in loss};groups=defaultdict(list)
 for iid,x in idx.items():
  if iid not in gm or iid not in fm or iid not in hm:raise RuntimeError("STAGE11B_INPUT_ID_MISMATCH")
  if int(gm[iid]["fold"])!=int(fm[iid]["fold"]) or int(gm[iid]["class_id"])!=int(fm[iid]["class_id"]) or int(gm[iid]["fold"])!=int(hm[iid]["fold"]):raise RuntimeError("STAGE11B_FOLD_CLASS_IDENTITY_MISMATCH")
  x["fold"]=int(gm[iid]["fold"]);x["gradient_radius"]=float(gm[iid]["k5_radius"]);x["feature_radius"]=float(fm[iid]["k5_radius"]);x["total_oof_loss"]=float(hm[iid]["total_loss"]);groups[(x["fold"],x["class_id"])].append(x)
 if len(groups)!=18:raise RuntimeError("STAGE11B_FOLD_CLASS_GROUP_MISMATCH")
 for group in groups.values():
  for field,out in (("gradient_radius","gradient_percentile"),("feature_radius","feature_percentile"),("total_oof_loss","hardness_percentile")):
   q=percentiles([x[field] for x in group])
   for x,v in zip(group,q):x[out]=float(v)
 table=[]
 for iid in sorted(idx):
  x=idx[iid];table.append({k:x[k] for k in ("instance_id","pair_id","class_name","class_id","fold","gradient_radius","feature_radius","total_oof_loss","gradient_percentile","feature_percentile","hardness_percentile")})
 frozen_write(a.output/"allocation_percentile_table.json",{"normalization":"independent empirical average-rank percentile within each fold x class group","group_count":18,"count":775,"records":table})
 selections={"uniform":select_uniform(list(idx.values()),cfg["selection_seed"]),"feature":select_ranked(list(idx.values()),"feature_percentile"),"hardness":select_ranked(list(idx.values()),"hardness_percentile"),"gradient":select_ranked(list(idx.values()),"gradient_percentile")};docs={}
 for arm,rows in selections.items():docs[arm]=selection_doc(arm,rows);frozen_write(a.output/f"{arm}_selection.json",docs[arm])
 selection_hashes={arm:sha(a.output/f"{arm}_selection.json") for arm in ARMS};frozen_write(a.output/"selection_freeze_audit.json",{"status":"SELECTIONS_FROZEN_BEFORE_DETECTOR_TRAINING","selection_seed":cfg["selection_seed"],"selection_sha256":selection_hashes,"all_counts_120":all(docs[x]["count"]==120 for x in ARMS),"all_exactly_20_per_class":all(docs[x]["class_counts"]=={c:20 for c in CLASSES} for x in ARMS),"all_pair_caps_satisfied":all(docs[x]["pair_cap_satisfied"] for x in ARMS),"validation_used":False})
 by_pair=defaultdict(list)
 for x in registry:by_pair[x["pair_id"]].append(x)
 union=sorted({x["instance_id"] for arm in ARMS for x in selections[arm]});isolated_dir=a.dataset_root/"isolated";audits=[]
 for iid in union:
  row=idx[iid];image,audit=isolate(row,by_pair);ip=isolated_dir/"images"/(iid+".png");lp=isolated_dir/"labels"/(iid+".txt");ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
  if ip.exists():
   if not np.array_equal(np.asarray(Image.open(ip).convert("RGB")),image):raise RuntimeError("STAGE11B_ISOLATED_IMAGE_CONFLICT: "+iid)
  else:Image.fromarray(image).save(ip)
  w,h=image.shape[1],image.shape[0];x0,y0,x1,y1=row["bbox_xyxy"];label=f"{row['class_id']} {(x0+x1)/(2*w):.10f} {(y0+y1)/(2*h):.10f} {(x1-x0)/w:.10f} {(y1-y0)/h:.10f}\n"
  if lp.exists() and lp.read_text()!=label:raise RuntimeError("STAGE11B_ISOLATED_LABEL_CONFLICT: "+iid)
  lp.write_text(label);audit.update({"image_path":str(ip.resolve()),"label_path":str(lp.resolve()),"image_sha256":sha(ip),"label_sha256":sha(lp)});audits.append(audit)
 iso_ok=all(x["target_pixels_unchanged"] and x["other_known_defects_restored"] and x["changed_pixels_outside_other_defect_masks"]==0 and x["paired_normal_size_match"] and x["target_bbox_unchanged"] and not x["missing_masks"] for x in audits);frozen_write(a.output/"replay_isolation_audit.json",{"status":"REPLAY_ISOLATION_PASS" if iso_ok else "REPLAY_ISOLATION_FAILED","unique_selected_instances":len(union),"arm_instance_uses":480,"same_instance_reuse_allowed":True,"records":audits})
 if not iso_ok:raise RuntimeError("STAGE11B_REPLAY_ISOLATION_FAILED")
 base_i={p.stem:p for p in (a.real_dataset/"images/train").iterdir()};base_l={p.stem:p for p in (a.real_dataset/"labels/train").iterdir()}
 if len(base_i)!=100 or set(base_i)!=set(base_l):raise RuntimeError("STAGE11B_BASE_DATASET_INVALID")
 dataset_audit={"base_image_count":100,"base_label_count":100,"base_image_tree_sha256":tree_digest(base_i.values()),"base_label_tree_sha256":tree_digest(base_l.values()),"arms":{}}
 for arm in ARMS:
  root=a.dataset_root/arm
  for i,stem in enumerate(sorted(base_i)):link(base_i[stem],root/"images/train"/f"base_{i:03d}{base_i[stem].suffix}");link(base_l[stem],root/"labels/train"/f"base_{i:03d}.txt")
  for i,row in enumerate(selections[arm]):iid=row["instance_id"];link(isolated_dir/"images"/(iid+".png"),root/"images/train"/f"extra_{i:03d}.png");link(isolated_dir/"labels"/(iid+".txt"),root/"labels/train"/f"extra_{i:03d}.txt")
  yaml=f"path: {root.resolve()}\ntrain: images/train\nval: images/train\nnc: 6\nnames: {json.dumps(list(CLASSES))}\n";yp=root/"data.yaml";yp.parent.mkdir(parents=True,exist_ok=True)
  if yp.exists() and yp.read_text()!=yaml:raise RuntimeError("STAGE11B_DATA_YAML_CONFLICT: "+arm)
  yp.write_text(yaml);images=list((root/"images/train").iterdir());labels=list((root/"labels/train").iterdir());dataset_audit["arms"][arm]={"images":len(images),"labels":len(labels),"base":100,"extra":120,"training_internal_val":"same arm train set; official validation excluded","image_tree_sha256":tree_digest(images),"label_tree_sha256":tree_digest(labels),"selection_sha256":selection_hashes[arm]}
 frozen_write(a.output/"dataset_build_audit.json",dataset_audit);frozen_write(a.output/"stage11b_protocol.json",{**cfg,"arms":["UniformRepeat120","FeatureRepeat120","HardnessRepeat120","GradientRepeat120"],"percentile_source":"frozen Stage11A/Stage10A-R values; normalized within fold x class","physical_pair_cap_per_class":1,"replay":"target-isolated full-resolution real image; other known defects restored from paired normal","synthetic_count":0,"selection_completed_before_detector_training":True,"official_validation_used_for_selection":False,"frozen_input_sha256":{"stage11a_status":sha(a.stage11a/"stage11a_status.json"),"gradient_embeddings":sha(a.stage11a/"real_oof_gradient_embeddings.npz"),"gradient_local_radius":sha(a.stage11a/"gradient_local_radius.json"),"gradient_coverage_distribution":sha(a.stage11a/"gradient_coverage_distribution.json"),"raw_gradient_diagnostics":sha(a.stage11a/"raw_gradient_diagnostics.json"),"feature_local_radius":sha(a.stage10ar/"pair_independent_local_radius.json"),"instance_registry":sha(a.registry)}})
 print(json.dumps({"status":"STAGE11B_PREPARATION_COMPLETE","selection_sha256":selection_hashes,"unique_isolated_instances":len(union)},indent=2))
if __name__=="__main__":main()
