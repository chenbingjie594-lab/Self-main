"""Stage10A real-only OOF task-manifold calibration."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
Q=(.05,.10,.25,.50,.75,.90,.95)
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def stats(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"std":float(a.std(ddof=1)),**{f"q{int(q*100):02d}":float(np.quantile(a,q)) for q in Q},"n":len(a)}
def quant(v):
 a=np.asarray(v,float);return {f"q{int(q*100):02d}":float(np.quantile(a,q)) for q in (.05,.25,.50,.75,.90,.95)}|{"n":len(a)}
def cv(v):
 a=np.asarray(v,float);return float(a.std(ddof=1)/a.mean())
def wasserstein_1d(a,b):
 a=np.sort(np.asarray(a,float));b=np.sort(np.asarray(b,float));grid=np.sort(np.concatenate((a,b)));return float(np.sum(np.abs(np.searchsorted(a,grid[:-1],side="right")/len(a)-np.searchsorted(b,grid[:-1],side="right")/len(b))*np.diff(grid)))
def ks_1d(a,b):
 a=np.sort(np.asarray(a,float));b=np.sort(np.asarray(b,float));grid=np.sort(np.concatenate((a,b)));return float(np.max(np.abs(np.searchsorted(a,grid,side="right")/len(a)-np.searchsorted(b,grid,side="right")/len(b))))
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--stage3r",type=Path,required=True);p.add_argument("--folds",type=Path,required=True);p.add_argument("--lowdata",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol)
 assert cfg["real_only"] and not any(x.lower().startswith("stage") for x in (str(a.stage3r),str(a.folds),str(a.lowdata)) if "stage3" not in x.lower())
 manifest=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");folds=load(a.folds);low=load(a.lowdata);assert manifest["official_val_use_count"]==folds["official_val_use_count"]==0 and folds["split_unit"]=="physical_pair_id" and folds["selected_ids_sha256"]==low["selected_ids_sha256"]
 folddef={int(x["fold"]):x for x in folds["folds"]};rows=[];npz_audit=[]
 for fm in manifest["folds"]:
  fold=int(fm["fold"]);path=a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz";z=np.load(path,allow_pickle=False);features=z["features"].astype(np.float64);class_ids=z["class_ids"].astype(int);records=fm["records"];assert len(records)==len(features)==len(class_ids)
  norms=np.linalg.norm(features,axis=1,keepdims=True);assert np.all(norms>0);features/=norms
  for r,f,c in zip(records,features,class_ids):
   assert int(r["class_id"])==int(c) and int(r["fold"])==fold;assert r["pair_id"] in folddef[fold]["holdout_pair_ids"] and r["pair_id"] not in folddef[fold]["train_pair_ids"]
   rows.append({**r,"class_name":CLASSES[int(c)],"feature":f})
  npz_audit.append({"fold":fold,"path":str(path),"sha256":sha(path),"records":len(records),"feature_shape":list(z["features"].shape),"manifest_counts":fm["counts"]})
 assert len(rows)==775 and len({r["instance_id"] for r in rows})==775 and {r["pair_id"] for r in rows}==set(low["selected_image_ids"])
 pair_folds=defaultdict(set)
 for r in rows:pair_folds[r["pair_id"]].add(r["fold"])
 audit={"status":"PASS","feature_source":"corrected Stage3R OOF holdout banks only","instances":775,"physical_pairs":len(pair_folds),"classes":dict(Counter(r["class_name"] for r in rows)),"pair_single_holdout_fold":all(len(x)==1 for x in pair_folds.values()),"every_record_from_holdout_pair":True,"no_pair_in_own_teacher_train_set":True,"no_self_trained_teacher":True,"official_validation_use_count":0,"synthetic_use_count":0,"downstream_metric_use_count":0,"fold_split_unit":folds["split_unit"],"lowshot_selected_ids_sha256":low["selected_ids_sha256"],"manifest_sha256":sha(a.stage3r/"oof_feature_bank_manifest_corrected.json"),"npz_banks":npz_audit};save(a.output/"real_oof_feature_audit.json",audit)
 # Distances are valid only inside one teacher coordinate system and class.
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[(r["fold"],r["class_id"])].append(i)
 local=[];neighbor_rel=[];random_rel=[];rng=np.random.default_rng(cfg["random_neighbor_seed"])
 for (fold,c),idx in groups.items():
  F=np.stack([rows[i]["feature"] for i in idx]);D=np.maximum(0.,1-F@F.T);np.fill_diagonal(D,np.inf);order=np.argsort(D,axis=1)
  assert len(idx)>=11
  for u,i in enumerate(idx):
   ds=D[u,order[u,:10]];radius=float(ds[:5].mean());assert radius>0
   rec={"instance_id":rows[i]["instance_id"],"pair_id":rows[i]["pair_id"],"fold":fold,"class_id":int(c),"class_name":CLASSES[c],"d1":float(ds[0]),"k3_radius":float(ds[:3].mean()),"k5_radius":radius,"k10_radius":float(ds.mean()),"neighbor_instance_ids":[rows[idx[j]]["instance_id"] for j in order[u,:10]]};local.append(rec)
   for rank,j in enumerate(order[u,:5],1):neighbor_rel.append({"instance_id":rows[i]["instance_id"],"class_name":CLASSES[c],"fold":fold,"neighbor_instance_id":rows[idx[j]]["instance_id"],"rank":rank,"distance":float(D[u,j]),"normalized_by_instance_k5":float(D[u,j]/radius)})
   candidates=[j for j in range(len(idx)) if j!=u];j=int(rng.choice(candidates));random_rel.append({"instance_id":rows[i]["instance_id"],"class_name":CLASSES[c],"fold":fold,"neighbor_instance_id":rows[idx[j]]["instance_id"],"distance":float(D[u,j]),"normalized_by_instance_k5":float(D[u,j]/radius)})
 save(a.output/"local_manifold_radius.json",{"primary_candidate":"k5_radius","primary_not_claimed_optimum":True,"distance":"cosine","comparison_scope":"same corrected OOF fold and same class; self excluded","records":local})
 dist={}
 for scope in ("overall",)+CLASSES:
  z=[x for x in local if scope=="overall" or x["class_name"]==scope];dist[scope]={m:stats([x[m] for x in z]) for m in ("d1","k3_radius","k5_radius","k10_radius")}
 save(a.output/"real_radius_distribution.json",{"metrics":dist})
 norm={}
 for scope in ("overall",)+CLASSES:
  lr=[x for x in local if scope=="overall" or x["class_name"]==scope];nr=[x for x in neighbor_rel if scope=="overall" or x["class_name"]==scope];rr=[x for x in random_rel if scope=="overall" or x["class_name"]==scope]
  byid={x["instance_id"]:x for x in lr};norm[scope]={"nearest_neighbor":quant([x["d1"]/x["k5_radius"] for x in lr]),"top5_neighbor_relations":quant([x["normalized_by_instance_k5"] for x in nr]),"top5_mean_per_source":quant([np.mean([y["normalized_by_instance_k5"] for y in nr if y["instance_id"]==x["instance_id"]]) for x in lr]),"random_same_fold_same_class":quant([x["normalized_by_instance_k5"] for x in rr])}
 save(a.output/"normalized_real_variation.json",{"normalizer":"source instance k5 local radius","metrics":norm,"top5_relation_count":len(neighbor_rel),"random_relation_seed":cfg["random_neighbor_seed"]})
 nearest=[x["d1"]/x["k5_radius"] for x in local];top5=[x["normalized_by_instance_k5"] for x in neighbor_rel];lower=float(np.quantile(nearest,.05));upper_q=(("conservative",.75),("medium",.90),("wide",.95));bands=[]
 for name,q in upper_q:
  upper=float(np.quantile(top5,q));covered=float(np.mean([(lower<=x<=upper) for x in top5]));bands.append({"name":name,"lower_beta":lower,"upper_beta":upper,"lower_real_only_basis":"q05 of nearest-neighbor normalized variation","upper_real_only_basis":f"q{int(q*100)} of top5-neighbor normalized relations","real_top5_relation_coverage":covered,"relations":len(top5)})
 save(a.output/"trust_region_candidates.json",{"frozen_before_any_new_synthetic_result":True,"radius":"instance k5 local radius r_i","distance":"1-cos(real source feature, candidate feature)","candidates":bands,"synthetic_inputs_used":0})
 nq05=float(np.quantile(nearest,.05));nmed=float(np.median(nearest));floor="NONZERO_VARIATION_FLOOR_SUPPORTED" if nq05>=.10 and nmed>=.25 else "VARIATION_FLOOR_WEAK" if nq05>0 and nmed>=.10 else "NO_VARIATION_FLOOR_SUPPORTED";save(a.output/"lower_bound_feasibility.json",{"status":floor,"nearest_normalized_q05":nq05,"nearest_normalized_median":nmed,"absolute_d1":stats([x["d1"] for x in local]),"top5_absolute_distance":stats([x["distance"] for x in neighbor_rel]),"rule":cfg["variation_floor_rules"],"selected_lower_beta":lower})
 # Compare cross-class consistency of top5 relation distances under three normalizers.
 global_r=float(np.median([x["k5_radius"] for x in local]));class_r={c:float(np.median([x["k5_radius"] for x in local if x["class_name"]==c])) for c in CLASSES};ri={x["instance_id"]:x["k5_radius"] for x in local};definitions={"global":lambda x:global_r,"class-adaptive":lambda x:class_r[x["class_name"]],"instance-adaptive":lambda x:ri[x["instance_id"]]};comparison={}
 for name,den in definitions.items():
  vals={c:np.asarray([x["distance"]/den(x) for x in neighbor_rel if x["class_name"]==c]) for c in CLASSES};med=[float(np.median(vals[c])) for c in CLASSES];pairs=[(a0,b0) for i,a0 in enumerate(CLASSES) for b0 in CLASSES[i+1:]];comparison[name]={"class_coefficient_of_variation":{c:cv(vals[c]) for c in CLASSES},"mean_class_coefficient_of_variation":float(np.mean([cv(vals[c]) for c in CLASSES])),"class_medians":dict(zip(CLASSES,med)),"class_median_spread":float(max(med)-min(med)),"mean_pairwise_wasserstein":float(np.mean([wasserstein_1d(vals[x],vals[y]) for x,y in pairs])),"mean_pairwise_ks_statistic":float(np.mean([ks_1d(vals[x],vals[y]) for x,y in pairs]))}
 best=min(x["mean_pairwise_wasserstein"] for x in comparison.values());tol=1.10*best
 if comparison["global"]["mean_pairwise_wasserstein"]<=tol:choice="global"
 elif comparison["class-adaptive"]["mean_pairwise_wasserstein"]<=1.10*comparison["instance-adaptive"]["mean_pairwise_wasserstein"]:choice="class-adaptive"
 else:choice="instance-adaptive"
 unstable=sum(dist[c]["k5_radius"]["std"]/dist[c]["k5_radius"]["mean"]>1 for c in CLASSES)>=4
 status="REAL_MANIFOLD_TOO_UNSTABLE" if unstable else {"global":"GLOBAL_TRUST_REGION_SUFFICIENT","class-adaptive":"CLASS_ADAPTIVE_TRUST_REGION_SUPPORTED","instance-adaptive":"INSTANCE_ADAPTIVE_TRUST_REGION_SUPPORTED"}[choice]
 save(a.output/"radius_normalization_comparison.json",{"selection":choice,"selection_rule":cfg["normalization_selection_rule"],"global_radius_median_k5":global_r,"class_radius_median_k5":class_r,"metrics":comparison,"real_manifold_unstable":unstable})
 formulation={"distance":"d_i = 1 - cos(f_i_real, f_i_syn)","constraint":"beta_L * r_i <= d_i <= beta_U * r_i","future_band_loss":"max(0, beta_L*r_i-d_i)^2 + max(0, d_i-beta_U*r_i)^2","implemented":False,"trained":False};st={"status":status,"frozen_future_protocol":choice,"primary_radius":"same-fold same-class k5 mean cosine distance","variation_floor_status":floor,"trust_region_candidates_frozen":True,"synthetic_inputs_used":0,"official_validation_use_count":0,"generator_training_count":0,"detector_training_count":0,"future_candidate":formulation,"frozen_residual_adapter_status":"RETIRED_AS_PAPER_INNOVATION_CANDIDATE"};save(a.output/"stage10a_status.json",st)
 lines=["# DeepPCB Stage10A Real-Manifold Calibration","",f"Status: **{status}**",f"Frozen normalization protocol for Stage10B: **{choice}**",f"Variation floor: **{floor}**","","This calibration used 775 corrected Stage3R OOF real instances only. Every feature came from a teacher whose training split excluded the instance's physical pair. Official validation, synthetic samples, downstream metrics, Stage9E, Stage9F, and Stage9G were not used for numerical calibration.","","## Frozen trust-region candidates","","| band | lower beta | upper beta | real top5 coverage |","|---|---:|---:|---:|"]+[f"| {x['name']} | {x['lower_beta']:.6f} | {x['upper_beta']:.6f} | {x['real_top5_relation_coverage']:.4f} |" for x in bands]+["","The future Task-Manifold Variation Band is a per-generated-sample generator constraint. BootstrapGuard remains a separate selected-dataset/set-composition mechanism; the two innovations are not merged.","",f"Future candidate only (not implemented): `{formulation['constraint']}` with band loss `{formulation['future_band_loss']}`."]
 (a.output/"STAGE10A_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(st,indent=2))
if __name__=="__main__":main()
