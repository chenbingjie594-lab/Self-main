"""Stage10B-R complete pre-registered four-arm generator feasibility audit."""
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np,torch
import evaluate_deeppcb_stage9e as e9
import evaluate_deeppcb_stage9er as e9r

CLASSES=("short","pinhole")
ARMS=("class_finetuned_sd2","tmvb_medium","repaired_frozen_adapter","pretrained_no_adaptation")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":int(a.size)}
def summarize_band(rows,bl,bu):
 z=np.asarray([x["local_z"] for x in rows]);signed=np.where(z<bl,bl-z,np.where(z>bu,z-bu,0));return {"mean_z":float(z.mean()),"median_z":float(np.median(z)),"below_fraction":float((z<bl).mean()),"inside_fraction":float(((z>=bl)&(z<=bu)).mean()),"above_fraction":float((z>bu).mean()),"mean_squared_violation":float(np.mean(signed**2)),"mean_absolute_violation":float(np.mean(np.abs(signed))),"n":len(rows)}
def finite_checkpoint(root,cls):
 from safetensors import safe_open
 p=root/cls/"unet/diffusion_pytorch_model.safetensors";ok=p.is_file();count=0
 if ok:
  with safe_open(p,framework="pt",device="cpu") as f:
   for k in f.keys():count+=1;ok&=bool(torch.isfinite(f.get_tensor(k)).all())
 return {"path":str(p),"exists":p.is_file(),"tensor_count":count,"all_finite":bool(ok)}
def main():
 p=argparse.ArgumentParser()
 for n in ("stage9e_manifest","tmvb_manifest","folds","stage3r","stage3r_runs","stage10ar","stage10b_feasibility","stage10c","model_root","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");p.add_argument("--batch",type=int,default=16);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 historical=load(a.stage9e_manifest);tm=load(a.tmvb_manifest);h={ (x["instance_id"],int(x["generation_seed"])):dict(x) for x in historical["records"]};t={(x["instance_id"],int(x["generation_seed"])):x for x in tm["records"]};common=sorted(set(h)&set(t));assert len(common)==774
 rows=[]
 for key in common:
  x=h[key];q=t[key];assert x["pair_id"]==q["pair_id"] and x["class_name"]==q["class_name"];x["tmvb_medium"]=q["image_path"];rows.append(x)
 pairing={"status":"PASS","exact_pairing":True,"arms":list(ARMS),"records_per_arm":774,"source_count":len({x["instance_id"] for x in rows}),"unique_source_seed_keys":len(common),"per_class":dict(Counter(x["class_name"] for x in rows)),"per_seed":dict(Counter(str(x["generation_seed"]) for x in rows)),"path_fields":{a0:("tmvb_medium" if a0=="tmvb_medium" else a0) for a0 in ARMS},"generation_rerun_count":0,"official_validation_use_count":0};save(a.output/"four_arm_pairing_audit.json",pairing)
 # Canonical Stage7B padded fidelity and pixel variation; corrected tight-bbox task features.
 e9.ARMS=ARMS;e9r.ARMS=ARMS;dev=torch.device("cuda:"+a.device);fraw,fmetrics=e9.fidelity(rows,dev,a.batch);features=e9r.extract(rows,a);task=e9r.task_summary(features);feature_raw,feature_metrics=e9r.feature_diversity(features);pixel_raw,pixel_metrics=e9.diversity(rows,features,dev);identity=e9.identity(features)
 save(a.output/"canonical_fidelity_metrics.json",{"protocol":{"roi":"Stage7B adaptive padded crop; minimum 64; context scale 4; crop then resize 256","lpips":"AlexNet RGB [-1,1]","reference":"corresponding real source"},"metrics":fmetrics,"raw_records":fraw})
 save(a.output/"historical_task_manifold_metrics.json",{"protocol":{"roi":"tight defect bbox","real_feature":"corrected Stage3R OOF real bank","normalizer":"historical frozen fold/class real q95; not Stage10A-R local radius"},"metrics":task,"raw_records":[{k:v for k,v in x.items() if k!="feature"} for x in features]})
 div={arm:{"overall":{**feature_metrics[arm]["overall"],"ic_lpips":pixel_metrics[arm]["overall"]["ic_lpips"],"edge_diversity":pixel_metrics[arm]["overall"]["edge_diversity"]},"per_class":{c:{**feature_metrics[arm]["per_class"][c],"ic_lpips":pixel_metrics[arm]["per_class"][c]["ic_lpips"],"edge_diversity":pixel_metrics[arm]["per_class"][c]["edge_diversity"]} for c in CLASSES}} for arm in ARMS};save(a.output/"conditional_variation_metrics.json",{"pixel_roi":"Stage7B padded ROI","task_feature_roi":"corrected tight defect bbox","metrics":div,"raw_pixel_records":pixel_raw,"raw_task_feature_records":feature_raw})
 dual={"generated_source_cluster_identity":{"protocol":"generated seed feature to leave-one-seed-out generated-source centroid","metrics":identity},"synthetic_to_oof_real_bank":{"protocol":"corrected tight-bbox feature to corresponding fold/class OOF real bank","metrics":task}};save(a.output/"source_identity_dual_protocol.json",dual)
 # Stage10A-R local-radius band is intentionally separate from historical q95 normalization.
 radius={x["instance_id"]:x["k5_radius"] for x in load(a.stage10ar/"pair_independent_local_radius.json")["records"]};bandrec=[]
 for x in features:bandrec.append({"arm":x["arm"],"instance_id":x["instance_id"],"class_name":x["class_name"],"seed":x["seed"],"pair_gap":x["pair_gap"],"instance_k5_radius":radius[x["instance_id"]],"local_z":x["pair_gap"]/radius[x["instance_id"]]})
 bl,bu=.5483221710003416,1.184702857709698;band={arm:{"overall":summarize_band([x for x in bandrec if x["arm"]==arm],bl,bu),"per_class":{c:summarize_band([x for x in bandrec if x["arm"]==arm and x["class_name"]==c],bl,bu) for c in CLASSES}} for arm in ARMS};save(a.output/"four_arm_band_compliance.json",{"band":{"beta_L":bl,"beta_U":bu},"radius":"Stage10A-R pair-independent instance k5","metrics":band,"raw_records":bandrec})
 # Summarize frozen training evidence; guarded skips are reported, never relabeled as no numerical issue.
 formal=load(a.stage10b_feasibility/"formal_training_audit.json");gate3=load(a.stage10b_feasibility/"gate3_stability_recheck.json");equiv=load(a.stage10b_feasibility/"differentiable_feature_equivalence_audit.json");grad=load(a.stage10b_feasibility/"tmvb_gradient_unit_test.json");checkpoints={c:finite_checkpoint(a.model_root,c) for c in CLASSES};training={"differentiable_feature_equivalence_pass":equiv["status"]=="DIFFERENTIABLE_FEATURE_PROTOCOL_EQUIVALENT","gradient_mechanism_pass":grad["status"]=="TMVB_GRADIENT_MECHANISM_PASSED","early_amp_guarded_skips_present":True,"gate3":gate3["audit"],"formal":{},"checkpoints":checkpoints}
 for c in CLASSES:
  x=formal["classes"][c];rr=x["records"];training["formal"][c]={"requested_successful_updates":x["target_successful_steps"],"actual_successful_updates":x["successful_steps"],"attempts":x["attempted_steps"],"amp_skips":x["skipped_steps"],"nonfinite_attempts":sum(not r["gradient_finite"] for r in rr),"final_grad_scaler":rr[-1]["scale_after"],"parameter_delta_l2":x["parameter_delta_l2"],"checkpoint_saved":x["checkpoint_saved"],"checkpoint_finite":checkpoints[c]["all_finite"],"training_complete":x["status"]=="PASS"}
 save(a.output/"training_validity_summary.json",training)
 A={c:band["tmvb_medium"]["per_class"][c]["inside_fraction"]>band["class_finetuned_sd2"]["per_class"][c]["inside_fraction"] and band["tmvb_medium"]["per_class"][c]["inside_fraction"]>band["repaired_frozen_adapter"]["per_class"][c]["inside_fraction"] and band["tmvb_medium"]["per_class"][c]["mean_absolute_violation"]<band["class_finetuned_sd2"]["per_class"][c]["mean_absolute_violation"] and band["tmvb_medium"]["per_class"][c]["mean_absolute_violation"]<band["repaired_frozen_adapter"]["per_class"][c]["mean_absolute_violation"] for c in CLASSES}
 B={c:task["tmvb_medium"]["per_class"][c]["normalized_pair_gap"]["mean"]<task["repaired_frozen_adapter"]["per_class"][c]["normalized_pair_gap"]["mean"] and task["tmvb_medium"]["per_class"][c]["fraction_gt_real_q95"]<task["repaired_frozen_adapter"]["per_class"][c]["fraction_gt_real_q95"] for c in CLASSES}
 C={c:div["tmvb_medium"]["per_class"][c]["ic_lpips"]["mean"]>div["class_finetuned_sd2"]["per_class"][c]["ic_lpips"]["mean"] and div["tmvb_medium"]["per_class"][c]["corrected_pairwise_feature_distance"]["mean"]>div["class_finetuned_sd2"]["per_class"][c]["corrected_pairwise_feature_distance"]["mean"] for c in CLASSES}
 D={c:fmetrics["tmvb_medium"]["per_class"][c]["lpips"]["mean"]<fmetrics["repaired_frozen_adapter"]["per_class"][c]["lpips"]["mean"] for c in CLASSES};E=training["differentiable_feature_equivalence_pass"] and training["gradient_mechanism_pass"] and all(training["formal"][c]["actual_successful_updates"]==2000 and training["formal"][c]["final_grad_scaler"]>0 and training["formal"][c]["checkpoint_finite"] for c in CLASSES)
 conditions={"A_band_control":{"per_class":A,"pass":all(A.values())},"B_avoid_frozen_overshoot":{"per_class":B,"pass":all(B.values())},"C_preserve_variation_vs_vanilla":{"per_class":C,"pass":all(C.values())},"D_fidelity_recovery_vs_frozen":{"per_class":D,"pass":all(D.values()),"tmvb_vs_vanilla_lpips_gap":{c:fmetrics["tmvb_medium"]["per_class"][c]["lpips"]["mean"]-fmetrics["class_finetuned_sd2"]["per_class"][c]["lpips"]["mean"] for c in CLASSES}},"E_training_valid":{"pass":E}}
 if all(v["pass"] for v in conditions.values()):status="TMVB_FEASIBLE"
 elif not conditions["A_band_control"]["pass"] or not conditions["B_avoid_frozen_overshoot"]["pass"] or not E:status="TMVB_NOT_FEASIBLE"
 else:status="TMVB_MIXED"
 save(a.output/"stage10b_original_gate_reconstruction.json",{"rule":"A+B+C+D+E all required; no weighted score","conditions":conditions,"status":status,"frozen_before_stage10br":True})
 s10c=load(a.stage10c/"stage10c_status.json");deviation={"status":"EXPLORATORY_UNAUTHORIZED_DOWNSTREAM_RUN","stage10c_executed":True,"complete_stage10b_gate_had_not_been_evaluated":True,"historical_authorization_source":"reduced band-only gate","results_retained_unmodified":True,"primary_last_metrics":{"real_repeat80_overall":.7277280836407842,"sd2_overall":.7147260639375902,"tmvb_overall":.7095259227052039,"tmvb_minus_realrepeat_overall_pp":-1.8202160935580358,"tmvb_minus_realrepeat_target2_pp":-2.3336887648551796,"tmvb_minus_sd2_overall_pp":-0.52001412323863,"tmvb_minus_sd2_target2_pp":-.6712811461145996},"historical_status":s10c["status"],"additional_detector_experiments_authorized":False};save(a.output/"stage10c_protocol_deviation_audit.json",deviation)
 final={"status":status,"conditions":conditions,"ADDITIONAL_DETECTOR_EXPERIMENTS_AUTHORIZED":False,"reason":"Requires TMVB_FEASIBLE and positive exploratory Stage10C signal; exploratory primary result is negative.","generator_training_count":0,"generation_count":0,"detector_training_count":0,"official_validation_use_count":0};save(a.output/"stage10br_status.json",final)
 lines=["# DeepPCB Stage10B-R Complete Feasibility Audit","",f"Status: **{status}**.","","## Original gate reconstruction",""]+[f"- Condition {k[0]}: **{v['pass']}**" for k,v in conditions.items()]+["","Stage10C is retained but reclassified as **EXPLORATORY_UNAUTHORIZED_DOWNSTREAM_RUN** because it was launched from the reduced band-only gate.","","No generator training, generation, detector training, official-validation evaluation, selector, or hyperparameter change was performed.","","Additional detector experiments authorized: **false**."];(a.output/"STAGE10BR_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(final,indent=2))
if __name__=="__main__":main()
