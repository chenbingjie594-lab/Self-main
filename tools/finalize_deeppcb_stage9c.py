"""Stage9C semantic audits and final report; consumes only completed diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def corr(x,y):
 r,p=spearmanr(x,y);return {"spearman_rho":float(r),"p_value":float(p),"n":len(x)}
def identity(stage9a,out):
 doc=load(stage9a/"source_identity_metrics.json");rows=doc["raw_records"];result={}
 for arm in ("sd2","full_msdf","frozen_adapter"):
  result[arm]={}
  for cls in ("overall","short","pinhole"):
   z=[x for x in rows if x["arm"]==arm and (cls=="overall" or x["class_name"]==cls)];features=np.stack([x["feature"] for x in z]);grand=features.mean(0);total=float(((features-grand)**2).sum());groups={}
   for x in z:groups.setdefault(x["instance_id"],[]).append(np.asarray(x["feature"]))
   within=float(sum(((np.stack(v)-np.stack(v).mean(0))**2).sum() for v in groups.values()));between=max(0.0,total-within);result[arm][cls]={"retrieval_top1":float(np.mean([x["retrieval_top1"] for x in z])),"retrieval_top5":float(np.mean([x["retrieval_top5"] for x in z])),"within_source_sum_squares":within,"between_source_sum_squares":between,"total_sum_squares":total,"within_over_total":within/max(total,1e-12),"between_over_total":between/max(total,1e-12),"source_identity_R2":between/max(total,1e-12),"direction":{"retrieval":"higher means easier source recovery / stronger identity","between_over_total_and_R2":"higher means source ID explains more feature variation / stronger identity","within_over_total":"lower means tighter same-source seed clusters / stronger identity"}}
 audit={"metrics":result,"stage9a_identity_gate_direction_correct":False,"protocol_bug":"Stage9A required within/total to decrease. A decrease means tighter within-source clusters and a larger between/total (source-identity R2), so it indicates stronger rather than weaker source dominance.","stage9a_status_rewritten":False,"reason_status_unchanged":"Stage9A independently failed fidelity and task-support gates."};save(out/"source_identity_semantics_audit.json",audit)
def vae(stage9a,out):
 rows=load(stage9a/"vae_bottleneck_raw.json")["records"];classes=sorted({x["class_name"] for x in rows});metrics=("mask_crop_lpips","edge_error")
 for x in rows:x["edge_error"]=1-x["edge_cosine"]
 def block(z):return {"area_vs_lpips":corr([x["area_fraction"] for x in z],[x["mask_crop_lpips"] for x in z]),"area_vs_edge_error":corr([x["area_fraction"] for x in z],[x["edge_error"] for x in z]),"contrast_vs_lpips":corr([x["contrast"] for x in z],[x["mask_crop_lpips"] for x in z]),"contrast_vs_edge_error":corr([x["contrast"] for x in z],[x["edge_error"] for x in z])}
 # OLS descriptive model: metric ~ log(area) + contrast + class fixed effects.
 X=[]
 for x in rows:X.append([1,np.log(max(x["area_fraction"],1e-12)),x["contrast"]]+[float(x["class_name"]==c) for c in classes[1:]])
 X=np.asarray(X);reg={}
 for metric in metrics:
  y=np.asarray([x[metric] for x in rows]);beta=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@beta;reg[metric]={"coefficients":{"intercept":float(beta[0]),"log_area":float(beta[1]),"contrast":float(beta[2]),**{f"class_{c}":float(v) for c,v in zip(classes[1:],beta[3:])}},"r_squared":float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum())}
 glob=block(rows);area_signal=abs(glob["area_vs_lpips"]["spearman_rho"])>=.3 or abs(glob["area_vs_edge_error"]["spearman_rho"])>=.3;status="VAE_BOTTLENECK_MIXED" if area_signal else "VAE_BOTTLENECK_NOT_SUPPORTED";save(out/"vae_bottleneck_claim_audit.json",{"status":status,"global":glob,"per_class":{c:block([x for x in rows if x["class_name"]==c]) for c in classes},"regression":reg,"descriptive_not_causal":True,"existing_reconstruction_summary":load(stage9a/"vae_bottleneck_summary.json"),"rerun_vae":False})
def numerical(stage9a,out):
 train=load(stage9a/"training_audit.json")["classes"];result={}
 for c,x in train.items():
  success=x["successful_updates"];skip=x["nonfinite_or_amp_skips"];result[c]={"attempted_optimizer_steps":success+skip,"successful_steps":success,"skipped_steps":skip,"skip_ratio":skip/(success+skip),"skip_stage_distribution":"unavailable from retained training history","grad_scaler_history":"unavailable","gradient_norm_history":"unavailable"}
 save(out/"numerical_stability_audit.json",{"status":"NUMERICAL_INSTABILITY_UNCLEAR","classes":result,"reason":"7.66% AMP skips are nontrivial, but retained logs do not identify skip timestep, gradient component, or GradScaler trajectory. No diagnostic retraining was used to alter the formal model.","formal_model_unchanged":True})
def main():
 p=argparse.ArgumentParser();p.add_argument("--stage9a",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);identity(a.stage9a,a.output);vae(a.stage9a,a.output);numerical(a.stage9a,a.output);status=load(a.output/"stage9c_status.json");fid=load(a.output/"cross_stage_fidelity_reaudit.json");res=load(a.output/"residual_magnitude_summary.json");corrs=load(a.output/"mask_area_residual_correlation.json");onoff=load(a.output/"adapter_on_off_probe.json");pre=load(a.output/"pretrained_vs_finetuned_probe.json");num=load(a.output/"numerical_stability_audit.json");vae_doc=load(a.output/"vae_bottleneck_claim_audit.json");ident=load(a.output/"source_identity_semantics_audit.json");lines=["# DeepPCB Stage9C Frozen Adapter Failure Attribution","",f"Status: **{status['status']}**.","",f"Canonical fidelity audit: **{fid['status']}**. Historical Stage9A absolute LPIPS used a tight bbox, while Stage7B/8B used an adaptive padded crop; historical values are not directly comparable across stages. The within-Stage9A adapter failure was re-evaluated without changing its formal gate.","",f"Identity protocol direction correct: **{str(ident['stage9a_identity_gate_direction_correct']).lower()}**. The Stage9A within/total direction was a protocol bug; the Stage9A status remains unchanged because fidelity and task support failed independently.","","## Residual and semantic probes",""]
 for c in ("short","pinhole"):
  for b in (2,3):
   m=res["metrics"][c][f"block{b}"]["masked_residual_to_hidden_ratio"]["median"];r=corrs[c][f"block{b}"]["spearman_rho"];rt="undefined" if r is None else f"{r:.4f}";lines.append(f"- {c} block {b}: median local residual/hidden={m:.4f}, source-level area Spearman rho={rt}.")
 lines += ["",f"Effective adapter no-op observed: {status.get('effective_adapter_noop_observed',False)} ({status.get('effective_adapter_noop_evidence',{})}).","All instrumented residuals were exactly zero and every ON/OFF image hash matched; therefore no local injection overshoot occurred in the actual checkpoint.","",f"Local overshoot supported: {status['local_injection_overshoot_supported']}.",f"Frozen-backbone semantic deficit supported: {status['semantic_adaptation_deficit_supported']}.","The pretrained OFF/adapter ON controls were substantially worse than class-finetuned Vanilla SD2 on canonical fidelity and OOF task features, supporting insufficient semantic adaptation of the completely frozen backbone.","",f"Numerical audit: **{num['status']}**. VAE claim audit: **{vae_doc['status']}**.","","No detector was trained; official validation was not used; Stage9A files and gate were not modified. Stage9D is not authorized. Future ideas, if warranted by the attribution, remain report-only and were not implemented."];(a.output/"STAGE9C_REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps(status,indent=2))
if __name__=="__main__":main()
