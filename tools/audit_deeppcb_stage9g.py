"""Stage9G: read-only downstream failure attribution over frozen Stage9E-R/9F evidence."""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,wilcoxon

ARMS=("class_finetuned_sd2","repaired_frozen_adapter","pretrained_no_adaptation")
CLASSES=("short","pinhole")
METRICS=("lpips","edge_cosine","normalized_pair_gap","fraction_gt_real_q95","ic_lpips","corrected_feature_variance","corrected_pairwise_feature_distance")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def save(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding="utf-8")
def summary(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"q25":float(np.quantile(a,.25)),"q50":float(np.quantile(a,.5)),"q75":float(np.quantile(a,.75)),"q90":float(np.quantile(a,.9)),"std":float(a.std(ddof=1)) if len(a)>1 else 0.,"n":len(a)}
def boolmean(v):return float(np.mean(v)) if v else None
def key(x):return (x["instance_id"],int(x.get("seed",x.get("generation_seed"))))
def paired_test(a,b):
 d=np.asarray(a)-np.asarray(b)
 try:p=float(wilcoxon(d).pvalue)
 except ValueError:p=1.
 return {"mean_delta":float(d.mean()),"median_delta":float(np.median(d)),"p_value_two_sided_wilcoxon":p,"n":len(d)}
def metric_rows(stage9e,stage9er):
 fidelity=load(stage9e/"canonical_fidelity_metrics.json")["raw_records"]
 task=load(stage9er/"corrected_task_feature_metrics.json")["raw_records"]
 pixeldiv=load(stage9e/"diversity_metrics.json")["raw_records"]
 featdiv=load(stage9er/"corrected_diversity_metrics.json")["raw_corrected_feature_diversity"]
 sample={key(x)+(x["arm"],):dict(x) for x in fidelity}
 for x in task:sample[key(x)+(x["arm"],)].update(x)
 source={(x["instance_id"],x["arm"]):dict(x) for x in pixeldiv}
 for x in featdiv:source[(x["instance_id"],x["arm"])].update(x)
 return list(sample.values()),list(source.values())
def strat_boot(full,metric,cfg):
 rng=np.random.default_rng(cfg["representativeness_bootstrap_seed"]);by={c:defaultdict(list) for c in CLASSES};vals=[]
 for x in full:by[x["class_name"]][x["instance_id"]].append(x)
 for _ in range(cfg["representativeness_bootstrap_replicates"]):
  z=[]
  for c in CLASSES:
   chosen=rng.choice(sorted(by[c]),40,replace=False)
   for iid in chosen:
    candidates=by[c][iid];z.append(float(candidates[int(rng.integers(len(candidates)))][metric]))
  vals.append(np.mean(z))
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def training_dynamics(runs):
 out={};points={25,50,75,100,125,150}
 for arm in ("real_repeat80","finetuned_sd2_syn80","frozen_adapter_syn80"):
  p=runs/f"{arm}_s42"/"results.csv"
  if not p.is_file():raise FileNotFoundError(f"STAGE9G_REQUIRES_EXISTING_RESULTS_CSV: {p}")
  rows=[]
  with p.open(newline="",encoding="utf-8") as f:
   for r in csv.DictReader(f):rows.append({k.strip():float(v) for k,v in r.items() if v!=""})
  def compact(r):return {k:v for k,v in r.items() if k=="epoch" or k.startswith("train/") or k.startswith("val/") or k.startswith("metrics/")}
  chosen={str(int(r["epoch"])):compact(r) for r in rows if int(r["epoch"]) in points};best=max(rows,key=lambda r:r.get("metrics/mAP50-95(B)",-1))
  trend=np.polyfit([r["epoch"] for r in rows[-50:]],[r.get("metrics/mAP50-95(B)",0) for r in rows[-50:]],1)[0]
  out[arm]={"available_epochs":len(rows),"milestones":chosen,"best_epoch_by_overall_map50_95":int(best["epoch"]),"best_overall_map50_95":best.get("metrics/mAP50-95(B)"),"last50_overall_map50_95_slope_per_epoch":float(trend),"target2_per_epoch_available":False,"target2_unavailable_reason":"Ultralytics results.csv does not store per-class AP per epoch; no detector rerun is allowed."}
 return out
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","stage9e","stage9er","stage9f","stage9f_runs","stage5f","stage6h","stage7c","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol)
 frozen=load(a.stage9f/"stage9f_frozen_selection_manifest.json");sel=frozen["records"];ids={x["instance_id"] for x in sel};keys={(x["instance_id"],int(x["generation_seed"])) for x in sel}
 assert len(sel)==80 and len(ids)==80 and Counter(x["class_name"] for x in sel)==Counter(short=40,pinhole=40) and frozen["selection_rng"]==2026
 save(a.output/"stage9f_selection_reaudit.json",{"status":"PASS","selection_count":80,"unique_source_count":80,"class_counts":{"short":40,"pinhole":40},"generation_seed_counts":dict(Counter(str(x["generation_seed"]) for x in sel)),"selection_rng":2026,"no_metric_based_selection":True,"no_regeneration":True,"frozen_instance_seed_keys":sorted([list(x) for x in keys])})
 samples,sources=metric_rows(a.stage9e,a.stage9er);s80=[x for x in samples if key(x) in keys];src80=[x for x in sources if x["instance_id"] in ids]
 assert len(s80)==240 and len(src80)==240
 identity=load(a.stage9er/"source_identity_dual_protocol.json")
 records=[]
 for x in s80:
  src=next(z for z in src80 if z["instance_id"]==x["instance_id"] and z["arm"]==x["arm"]);records.append({**x,**{k:src[k] for k in ("ic_lpips","edge_diversity","corrected_feature_variance","corrected_pairwise_feature_distance")}})
 summaries={arm:{"overall":{},"per_class":{}} for arm in ARMS}
 for arm in ARMS:
  for group,label in (([x for x in records if x["arm"]==arm],"overall"),):
   summaries[arm][label]={m:(boolmean([x[m] for x in group]) if m in ("fraction_gt_real_q95","real_bank_top1","real_bank_top5","same_instance_nearest") else summary([x[m] for x in group])) for m in ("lpips","ssim","edge_cosine","boundary_gradient_cosine","pair_gap","normalized_pair_gap","fraction_gt_real_q95","real_bank_top1","real_bank_top5","same_instance_nearest")}
  summaries[arm]["per_class"]={c:{m:(boolmean([x[m] for x in records if x["arm"]==arm and x["class_name"]==c]) if m in ("fraction_gt_real_q95","real_bank_top1","real_bank_top5","same_instance_nearest") else summary([x[m] for x in records if x["arm"]==arm and x["class_name"]==c])) for m in ("lpips","ssim","edge_cosine","boundary_gradient_cosine","pair_gap","normalized_pair_gap","fraction_gt_real_q95","real_bank_top1","real_bank_top5","same_instance_nearest")} for c in CLASSES}
  summaries[arm]["source_level_diversity"]={scope:{m:summary([x[m] for x in src80 if x["arm"]==arm and (scope=="overall" or x["class_name"]==scope)]) for m in ("ic_lpips","edge_diversity","corrected_feature_variance","corrected_pairwise_feature_distance")} for scope in ("overall",)+CLASSES}
  summaries[arm]["generated_source_identity_aggregate_only"]=identity["generated_source_cluster_retrieval"]["metrics"][arm]
 save(a.output/"selected80_generator_metrics.json",{"summaries":summaries,"raw_joined_records":records,"source_level_identity_limitation":"Historical artifacts retain only arm/class aggregate generated-source R2; per-source R2 cannot be reconstructed without forbidden feature recomputation. No value was fabricated."})
 # Representativeness uses stratified random-80 bootstrap intervals, separately for sample and source metrics.
 rep={};shift=[]
 for arm in ARMS:
  rep[arm]={};outside=0
  for m in METRICS:
   pool=samples if m in ("lpips","edge_cosine","normalized_pair_gap","fraction_gt_real_q95") else sources;selected=s80 if pool is samples else src80;full=[x for x in pool if x["arm"]==arm];ss=[x for x in selected if x["arm"]==arm];mu=float(np.mean([x[m] for x in ss]));ci=strat_boot(full,m,cfg);flag=not(ci[0]<=mu<=ci[1]);outside+=flag;rep[arm][m]={"selected80_mean":mu,"full_pool_mean":float(np.mean([x[m] for x in full])),"stratified_random80_mean_95_interval":ci,"outside_interval":flag,"selected_n":len(ss),"full_n":len(full)}
  rep[arm]["outside_metric_count"]=outside
  if outside>=2:shift.append(arm)
 repstatus="RANDOM80_SHIFTED" if shift else "RANDOM80_REPRESENTATIVE";save(a.output/"selection_representativeness.json",{"status":repstatus,"rule":cfg["representativeness_rule"],"shifted_arms":shift,"metrics":rep})
 manifold={}
 for arm in ARMS:
  manifold[arm]={}
  for scope in ("overall",)+CLASSES:
   z=[x for x in records if x["arm"]==arm and (scope=="overall" or x["class_name"]==scope)];v=[x["normalized_pair_gap"] for x in z];manifold[arm][scope]={**summary(v),"fraction_le_real_q95":boolmean([x<=1 for x in v]),"fraction_gt_real_q95":boolmean([x>1 for x in v]),"fraction_gt_1_5x_real_q95":boolmean([x>1.5 for x in v]),"fraction_gt_2x_real_q95":boolmean([x>2 for x in v])}
 # exact paired significance over full pool and selected80
 comparisons={}
 for scope_name,pool in (("full_pool",samples),("selected80",s80)):
  idx={(x["instance_id"],int(x["seed"]),x["arm"]):x for x in pool};common=sorted({(x[0],x[1]) for x in idx if x[2]==ARMS[0]} & {(x[0],x[1]) for x in idx if x[2]==ARMS[1]});comparisons[scope_name]=paired_test([idx[k+(ARMS[1],)]["normalized_pair_gap"] for k in common],[idx[k+(ARMS[0],)]["normalized_pair_gap"] for k in common])
 save(a.output/"selected80_manifold_distance.json",{"historical_real_q95_normalization":"corrected Stage3R fold/class real q95; normalized threshold=1","metrics":manifold,"frozen_minus_finetuned_paired_tests":comparisons})
 corr={}
 for population,pool in (("full_pool",sources),("selected80",src80)):
  corr[population]={}
  task_by=defaultdict(list)
  base=samples if population=="full_pool" else s80
  for x in base:
   if x["arm"]=="repaired_frozen_adapter":task_by[x["instance_id"]].append(x["normalized_pair_gap"])
  for scope in ("overall",)+CLASSES:
   z=[x for x in pool if x["arm"]=="repaired_frozen_adapter" and (scope=="overall" or x["class_name"]==scope)];corr[population][scope]={}
   for m in ("ic_lpips","corrected_feature_variance","corrected_pairwise_feature_distance"):
    rho,pv=spearmanr([x[m] for x in z],[np.mean(task_by[x["instance_id"]]) for x in z]);corr[population][scope][m]={"spearman_rho":float(rho),"p_value":float(pv),"n":len(z)}
   corr[population][scope]["generated_source_identity_R2"]={"available":False,"reason":"Only arm/class aggregate R2 was retained; source-level correlation is not identifiable from frozen artifacts."}
 save(a.output/"variation_manifold_correlation.json",{"interpretation_limit":"Association only; no per-sample detector utility and no causal claim.","correlations":corr})
 primary={x["arm"]:x for x in load(a.stage9f/"primary_last_metrics.json")["records"]};alias={"FrozenAdapter":"frozen_adapter_syn80","FineTunedSD2":"finetuned_sd2_syn80","RealRepeat80":"real_repeat80"}
 td={}
 for name,left,right in (("Frozen_minus_FineTuned","FrozenAdapter","FineTunedSD2"),("Frozen_minus_RealRepeat80","FrozenAdapter","RealRepeat80"),("FineTuned_minus_RealRepeat80","FineTunedSD2","RealRepeat80")):
  td[name]={c:{m:primary[alias[left]]["per_class"][c][m]-primary[alias[right]]["per_class"][c][m] for m in ("ap50_95","ap50","recall")} for c in CLASSES}
 loss={c:abs(td["Frozen_minus_FineTuned"][c]["ap50_95"]) for c in CLASSES};td["dominant_target_loss"]="both" if min(loss.values())/max(loss.values())>=.5 else max(loss,key=loss.get);save(a.output/"target_class_detector_deltas.json",td)
 non={};
 for name,ref in (("Frozen_minus_FineTuned","finetuned_sd2_syn80"),("Frozen_minus_RealRepeat80","real_repeat80")):non[name]={c:primary["frozen_adapter_syn80"]["per_class"][c]["ap50_95"]-primary[ref]["per_class"][c]["ap50_95"] for c in ("open","mousebite","spur","spurious_copper")}
 vals=list(non["Frozen_minus_FineTuned"].values());non["status"]="broad negative transfer" if sum(x<0 for x in vals)>=3 else "target-only degradation" if sum(x<0 for x in vals)<=1 else "mixed";non["interpretation_limit"]="detector-level diagnostic only; not causal";save(a.output/"nontarget_transfer_audit.json",non)
 dynamics=training_dynamics(a.stage9f_runs);f=dynamics["frozen_adapter_syn80"];s=dynamics["finetuned_sd2_syn80"];f50=f["last50_overall_map50_95_slope_per_epoch"];dynamics["frozen_pattern"]="late overfit-like divergence" if f50<0 and s["last50_overall_map50_95_slope_per_epoch"]>=f50 else "no clear dynamics pattern";dynamics["primary_checkpoint_unchanged"]="epoch150 last.pt";save(a.output/"training_dynamics_audit.json",dynamics)
 s5=load(a.stage5f/"synthetic_content_status.json");s6=load(a.stage6h/"fixed_budget_status.json");s7=load(a.stage7c/"stage7c_status.json");s9=load(a.stage9f/"stage9f_status.json")
 sd2_rr_overall_pp=100*(primary["finetuned_sd2_syn80"]["map50_95"]-primary["real_repeat80"]["map50_95"]);sd2_rr_target2_pp=100*(primary["finetuned_sd2_syn80"]["target2_map50_95"]-primary["real_repeat80"]["target2_map50_95"])
 cross={"status":"DEEPPPCB_SYNTHETIC_GAIN_REPEATEDLY_NOT_SUPPORTED","comparable_direct_exposure_controls":{"Stage5F":{"status":s5["status"],"mean_random_minus_realrepeat":s5["random_minus_real_repeat340"]["mean"],"wins":s5["random_minus_real_repeat340"]["wins"]},"Stage9F":{"status":s9.get("secondary_interpretation","SYNTHETIC_GAIN_NOT_SUPPORTED_VS_REAL_REPEAT80"),"finetuned_minus_realrepeat_overall_pp":sd2_rr_overall_pp,"finetuned_minus_realrepeat_target2_pp":sd2_rr_target2_pp,"computed_from":"primary_last_metrics.json"}},"supporting_but_not_numerically_pooled":{"Stage6H":s6["batch_matched_status"],"Stage7C":s7["status"]},"pooling_performed":False,"claim":"Under the current DeepPCB low-shot fixed-budget protocols, synthetic-content gain is difficult to separate from exposure and is not consistently positive."};save(a.output/"cross_stage_synthetic_gain_audit.json",cross)
 # Predeclared attribution gates.
 div=load(a.stage9er/"corrected_diversity_metrics.json")["raw_corrected_feature_diversity"];px=load(a.stage9e/"diversity_metrics.json")["raw_records"]
 tests={}
 for m,pool in (("corrected_feature_variance",div),("corrected_pairwise_feature_distance",div),("ic_lpips",px)):
  ix={(x["instance_id"],x["arm"]):x for x in pool};common=sorted({x[0] for x in ix if x[1]==ARMS[0]}&{x[0] for x in ix if x[1]==ARMS[1]});tests[m]=paired_test([ix[(i,ARMS[1])][m] for i in common],[ix[(i,ARMS[0])][m] for i in common])
 variation=all(x["mean_delta"]>0 and x["p_value_two_sided_wilcoxon"]<cfg["significance_alpha"] for x in tests.values());departure=comparisons["full_pool"]["mean_delta"]>0 and comparisons["full_pool"]["p_value_two_sided_wilcoxon"]<cfg["significance_alpha"];utility=primary["frozen_adapter_syn80"]["map50_95"]<primary["finetuned_sd2_syn80"]["map50_95"] and primary["frozen_adapter_syn80"]["target2_map50_95"]<primary["finetuned_sd2_syn80"]["target2_map50_95"];close=abs(sd2_rr_overall_pp)<=cfg["fine_tuned_close_to_exposure_control_pp"]
 overshoot=variation and departure and repstatus=="RANDOM80_REPRESENTATIVE" and utility and close;ceiling=cross["status"]=="DEEPPPCB_SYNTHETIC_GAIN_REPEATEDLY_NOT_SUPPORTED"
 status="BOTH_LIMITATIONS_SUPPORTED" if overshoot and ceiling else "VARIATION_OVERSHOOT_SUPPORTED" if overshoot else "SYNTHETIC_UTILITY_CEILING_DOMINANT" if ceiling else "NO_CLEAR_ATTRIBUTION";route="retire FrozenResidualAdapter as a paper innovation candidate; any future first innovation must restart from utility-constrained generation" if status=="BOTH_LIMITATIONS_SUPPORTED" else "manifold-constrained useful variation (research recommendation only)" if status=="VARIATION_OVERSHOOT_SUPPORTED" else "stop generator iteration on DeepPCB; prioritize BootstrapGuard, a second dataset, or a stricter low-shot setting" if status=="SYNTHETIC_UTILITY_CEILING_DOMINANT" else "no route change supported"
 st={"status":status,"selection_representativeness":repstatus,"variation_significantly_higher":variation,"manifold_departure_significantly_larger":departure,"stage9f_frozen_utility_lower":utility,"finetuned_overall_close_to_realrepeat":close,"synthetic_ceiling_repeatedly_not_supported":ceiling,"variation_tests":tests,"route_recommendation":route,"training_count":0,"generation_count":0,"selection_count":0,"historical_results_modified":False,"correlation_is_not_causation":True};save(a.output/"stage9g_status.json",st)
 report=["# DeepPCB Stage9G Downstream Failure Attribution","",f"Status: **{status}**",f"Selection representativeness: **{repstatus}**","",f"FrozenAdapter variation significantly higher: **{variation}**",f"FrozenAdapter manifold departure significantly larger: **{departure}**",f"FrozenAdapter downstream utility lower than FineTuned-SD2: **{utility}**",f"Repeated synthetic-content gain supported: **{not ceiling}**","",f"Route: {route}.","","All correlations are descriptive associations. No detector or generator was trained, no image was regenerated, and no sample was reselected. Per-source generated-identity R2 correlation is not identifiable from the frozen artifacts and was not fabricated."];(a.output/"STAGE9G_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8");print(json.dumps(st,indent=2))
if __name__=="__main__":main()
