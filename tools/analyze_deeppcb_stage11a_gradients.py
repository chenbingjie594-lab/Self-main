"""Stage11A-2: pair-independent real-gradient coverage geometry and final gate."""
from __future__ import annotations
import argparse,json,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr,pearsonr

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def jsonable(x):
 if isinstance(x,np.ndarray):return [jsonable(v) for v in x.tolist()]
 if isinstance(x,np.generic):return jsonable(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 return x
def save(p,x):Path(p).write_text(json.dumps(jsonable(x),indent=2,allow_nan=False),encoding="utf-8")
def pct(v):
 a=np.asarray(v,float);return (rankdata(a,method="average")-1)/max(1,len(a)-1)
def corr(x,y):
 return {"spearman":float(spearmanr(x,y).statistic),"pearson":float(pearsonr(x,y).statistic),"n":len(x)}
def main():
 p=argparse.ArgumentParser()
 for n in ("gradient_npz","raw_diagnostics","projection_audit","stage10ar","stage3r","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);projection=load(a.projection_audit);assert projection["status"]=="GRADIENT_PROJECTION_PASS"
 z=np.load(a.gradient_npz);E=z["embeddings"].astype(float);ids=z["instance_ids"].astype(str);pairs=z["pair_ids"].astype(str);folds=z["folds"].astype(int);class_ids=z["class_ids"].astype(int);diag={x["instance_id"]:x for x in load(a.raw_diagnostics)["records"]};n=len(ids);assert E.shape==(775,1024)
 groups=defaultdict(list)
 for i in range(n):groups[(folds[i],class_ids[i])].append(i)
 local=[];gp3=np.zeros(n);gp5=np.zeros(n);gp10=np.zeros(n)
 for (f,c),ix in groups.items():
  vals=[]
  for i in ix:
   cand=[j for j in ix if pairs[j]!=pairs[i]];d=1-np.clip(E[cand]@E[i],-1,1);order=np.argsort(d);near=[cand[k] for k in order[:10]];ds=d[order];vals.append((i,float(ds[0]),float(ds[:3].mean()),float(ds[:5].mean()),float(ds[:10].mean()),near))
  p3=pct([x[2] for x in vals]);p5=pct([x[3] for x in vals]);p10=pct([x[4] for x in vals])
  for q,(i,d1,k3,k5,k10,near) in enumerate(vals):gp3[i]=p3[q];gp5[i]=p5[q];gp10[i]=p10[q];local.append({"instance_id":ids[i],"pair_id":pairs[i],"fold":f,"class_id":c,"class_name":CLASSES[c],"candidate_neighbor_count":len([j for j in ix if pairs[j]!=pairs[i]]),"d1":d1,"k3_radius":k3,"k5_radius":k5,"k10_radius":k10,"k3_percentile":float(p3[q]),"k5_percentile":float(p5[q]),"k10_percentile":float(p10[q]),"neighbor_instance_ids":[ids[j] for j in near],"neighbor_pair_ids":[pairs[j] for j in near]})
 save(a.output/"gradient_local_radius.json",{"primary_candidate":"same-fold same-class different-pair k5 mean gradient cosine distance","cross_fold_raw_distance_used":False,"records":local})
 # Fold-class heterogeneity.
 dist={};ratios=[]
 for (f,c),ix in sorted(groups.items()):
  v=np.asarray([next(x["k5_radius"] for x in local if x["instance_id"]==ids[i]) for i in ix]);q=np.quantile(v,[.1,.25,.5,.75,.9,.95]);ratio=float(q[4]/q[2]);ratios.append(ratio);dist[f"fold{f}:{CLASSES[c]}"]={"n":len(v),"mean":float(v.mean()),"std":float(v.std()),"cv":float(v.std()/v.mean()),"q10":float(q[0]),"q25":float(q[1]),"median":float(q[2]),"q75":float(q[3]),"q90":float(q[4]),"q95":float(q[5]),"q90_over_q50":ratio}
 hetero=bool(np.median(ratios)>=1.30);save(a.output/"gradient_coverage_distribution.json",{"status":"GRADIENT_COVERAGE_HETEROGENEITY_SUPPORTED" if hetero else "GRADIENT_COVERAGE_TOO_UNIFORM","threshold_median_q90_q50":1.30,"median_fold_class_q90_q50":float(np.median(ratios)),"minimum":float(min(ratios)),"maximum":float(max(ratios)),"groups":dist})
 # k robustness uses within-group percentiles only.
 overall={"k3_vs_k5":float(spearmanr(gp3,gp5).statistic),"k5_vs_k10":float(spearmanr(gp5,gp10).statistic)};per={}
 for c in range(6):
  ix=np.where(class_ids==c)[0];per[CLASSES[c]]={"k3_vs_k5":float(spearmanr(gp3[ix],gp5[ix]).statistic),"k5_vs_k10":float(spearmanr(gp5[ix],gp10[ix]).statistic)}
 class_pass=sum(v["k3_vs_k5"]>=.8 and v["k5_vs_k10"]>=.8 for v in per.values());stable=overall["k3_vs_k5"]>=.8 and overall["k5_vs_k10"]>=.8 and class_pass>=5;save(a.output/"gradient_k_robustness.json",{"status":"GRADIENT_COVERAGE_STABLE" if stable else "GRADIENT_COVERAGE_UNSTABLE","threshold":.8,"required_classes":5,"passing_classes":class_pass,"overall":overall,"per_class":per})
 # Frozen Stage10A-R feature geometry.
 fr=load(a.stage10ar/"pair_independent_local_radius.json")["records"];fm={x["instance_id"]:x for x in fr};fp=np.zeros(n)
 for key,ix in groups.items():fp[ix]=pct([fm[ids[i]]["k5_radius"] for i in ix])
 radius_corr=float(spearmanr(gp5,fp).statistic);j5=[];j10=[]
 for i in range(n):
  gr=next(x for x in local if x["instance_id"]==ids[i]);g5=set(gr["neighbor_instance_ids"][:5]);g10=set(gr["neighbor_instance_ids"][:10]);f5=set(fm[ids[i]]["neighbor_instance_ids"][:5]);f10=set(fm[ids[i]]["neighbor_instance_ids"][:10]);j5.append(len(g5&f5)/len(g5|f5));j10.append(len(g10&f10)/len(g10|f10))
 # Feature vectors are only compared inside each fold/class.
 bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");F={}
 for entry in bankdoc["folds"]:
  bank=np.load(a.stage3r/f"oof_real_bank_corrected_fold{entry['fold']}.npz")["features"]
  for r in entry["records"]:q=bank[int(r["feature_index"])].astype(float);F[r["instance_id"]]=q/np.linalg.norm(q)
 pair_metrics={}
 for (f,c),ix in sorted(groups.items()):
  feasible=[(ix[u],ix[v]) for u in range(len(ix)) for v in range(u+1,len(ix)) if pairs[ix[u]]!=pairs[ix[v]]];rng=random.Random(2026+f*10+c);chosen=rng.sample(feasible,min(500,len(feasible)));fd=[float(1-np.dot(F[ids[i]],F[ids[j]])) for i,j in chosen];gd=[float(1-np.dot(E[i],E[j])) for i,j in chosen];pair_metrics[f"fold{f}:{CLASSES[c]}"]=corr(fd,gd)
 group_rho=[x["spearman"] for x in pair_metrics.values() if x["spearman"] is not None];group_pearson=[x["pearson"] for x in pair_metrics.values() if x["pearson"] is not None]
 nonred=abs(radius_corr)<=.70 and float(np.mean(j5))<=.60;save(a.output/"gradient_vs_feature_geometry.json",{"status":"GRADIENT_GEOMETRY_NONREDUNDANT_WITH_FEATURES" if nonred else "GRADIENT_GEOMETRY_LARGELY_REDUNDANT","radius_percentile_spearman":radius_corr,"mean_jaccard_at5":float(np.mean(j5)),"mean_jaccard_at10":float(np.mean(j10)),"pairwise_distance":{"per_fold_class":pair_metrics,"cross_fold_summary":{"median_group_spearman":float(np.median(group_rho)) if group_rho else None,"median_group_pearson":float(np.median(group_pearson)) if group_pearson else None},"seed":2026,"max_pairs_per_group":500},"gate":{"absolute_radius_spearman_max":.70,"mean_jaccard_at5_max":.60},"cross_fold_raw_gradient_distance_used":False})
 grad_only=(gp5>=.8)&(fp<.8);save(a.output/"gradient_unique_coverage_subset.json",{"definition":{"gradient_undercovered_percentile_min":.8,"feature_undercovered_percentile_min":.8},"gradient_only_count":int(grad_only.sum()),"gradient_only_percentage":float(grad_only.mean()),"per_class":{CLASSES[c]:int(np.sum(grad_only&(class_ids==c))) for c in range(6)},"per_fold":{str(f):int(np.sum(grad_only&(folds==f))) for f in range(3)},"instance_ids":ids[grad_only].tolist()})
 # Hardness confound.
 fields=("total_loss","cls_loss","box_loss","dfl_loss","raw_gradient_l2_norm");hc={k:float(spearmanr(gp5,[diag[i][k] for i in ids]).statistic) for k in fields};hard_n=max(1,int(np.ceil(.2*len(ids))));topg=set(np.asarray(ids)[np.argsort(gp5)[-hard_n:]]);lossv=np.asarray([diag[i]["total_loss"] for i in ids]);topl=set(np.asarray(ids)[np.argsort(lossv)[-hard_n:]]);overlap=len(topg&topl)/len(topg|topl);hard=abs(hc["total_loss"])<.8;save(a.output/"gradient_hardness_confound.json",{"status":"GRADIENT_COVERAGE_NOT_TRIVIALLY_HARDNESS" if hard else "GRADIENT_COVERAGE_DOMINATED_BY_HARDNESS","spearman":hc,"top20_count_each":hard_n,"top20_selection":"exact descending rank with deterministic index tie break","top20_jaccard_gradient_undercoverage_vs_total_loss":overlap,"gate_absolute_total_loss_rho_max":.8})
 meta=[]
 for i in range(n):
  d=diag[ids[i]];meta.append({"instance_id":ids[i],"pair_id":pairs[i],"class_name":CLASSES[class_ids[i]],"fold":int(folds[i]),"gradient_percentile":float(gp5[i]),"feature_percentile":float(fp[i]),"oof_loss":d["total_loss"],"gradient_norm":d["raw_gradient_l2_norm"],"k5_gradient_radius":next(x["k5_radius"] for x in local if x["instance_id"]==ids[i]),"k5_feature_radius":fm[ids[i]]["k5_radius"],"real_path":d["real_path"],"mask_path":d["mask_path"]})
 save(a.output/"gradient_coverage_extremes.json",{"top20":sorted(meta,key=lambda x:x["gradient_percentile"],reverse=True)[:20],"bottom20":sorted(meta,key=lambda x:x["gradient_percentile"])[:20],"threshold_tuning_use":False})
 A=True;B=hetero;C=stable;D=nonred;Ehard=hard
 if A and B and C and D and Ehard:status="REAL_GRADIENT_COVERAGE_STRUCTURE_SUPPORTED"
 elif not(A and B and C):status="REAL_GRADIENT_GEOMETRY_NOT_SUPPORTED"
 elif not D:status="GRADIENT_GEOMETRY_REDUNDANT_WITH_FEATURES"
 else:status="GRADIENT_COVERAGE_MAINLY_HARDNESS"
 final={"status":status,"gates":{"A_projection":A,"B_heterogeneity":B,"C_k_robustness":C,"D_nonredundancy":D,"E_not_hardness":Ehard},"STAGE11B_ALLOCATION_CALIBRATION_AUTHORIZED":status=="REAL_GRADIENT_COVERAGE_STRUCTURE_SUPPORTED","real_only_calibration":True,"synthetic_inputs_used":0,"generator_training_count":0,"detector_training_count":0,"optimizer_step_count":0,"official_validation_use_count":0};save(a.output/"stage11a_status.json",final)
 lines=["# DeepPCB Stage11A Real Gradient Geometry Audit","",f"Status: **{status}**.","","## Frozen gates",""]+[f"- {k}: **{v}**" for k,v in final["gates"].items()]+["",f"Stage11B allocation calibration authorized: **{final['STAGE11B_ALLOCATION_CALIBRATION_AUTHORIZED']}**.","","Only real-only OOF data were used. No generator, synthetic image, detector training, optimizer update, official validation, BootstrapGuard, or downstream result participated in calibration."];(a.output/"STAGE11A_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(final,indent=2))
if __name__=="__main__":main()
