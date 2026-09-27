"""Stage9E-R: restore tight-mask-bbox OOF feature protocol without regeneration."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from dwbg_feature_extraction import DetectInputExtractor
from dwbg_v2_utils import l2_normalize

CLASSES=("short","pinhole");ARMS=("class_finetuned_sd2","repaired_frozen_adapter","pretrained_no_adaptation")
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":int(a.size)}
def bbox(mask_path):
 m=np.asarray(Image.open(mask_path).convert("L"))>127;y,x=np.where(m)
 if not len(x):raise RuntimeError("STAGE9ER_EMPTY_MASK "+str(mask_path))
 return int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1
def extract(rows,a):
 folds=load(a.folds);foldof={pid:f["fold"] for f in folds["folds"] for pid in f["holdout_pair_ids"]};bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");refs=load(a.stage3r/"manifold_reference_corrected.json")["references"];out=[]
 for entry in bankdoc["folds"]:
  fold=entry["fold"]
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz") as z:real,ids=z["features"],z["class_ids"]
  source={x["instance_id"]:i for i,x in enumerate(entry["records"])};ext=DetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512)
  try:
   for x in rows:
    if foldof[x["pair_id"]]!=fold:continue
    box=bbox(x["mask"]);ci=1 if x["class_name"]=="short" else 5;bank_idx=np.where(ids==ci)[0];bank=np.stack([l2_normalize(real[i]) for i in bank_idx]);q95=float(refs[f"fold{fold}:{x['class_name']}"]["q95_real_distance"]);rf=l2_normalize(real[source[x["instance_id"]]])
    for arm in ARMS:
     f=l2_normalize(ext.encode(x[arm],box,512,512));gap=float(1-np.dot(f,rf));order=np.argsort(1-np.clip(bank@f,-1,1));retrieved=[entry["records"][bank_idx[i]]["instance_id"] for i in order[:5]];out.append({"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":x["class_name"],"seed":int(x["generation_seed"]),"fold":fold,"arm":arm,"feature":f.tolist(),"pair_gap":gap,"real_q95":q95,"normalized_pair_gap":gap/max(q95,1e-12),"fraction_gt_real_q95":gap>q95,"real_bank_top1":retrieved[0]==x["instance_id"],"real_bank_top5":x["instance_id"] in retrieved,"same_instance_nearest":retrieved[0]==x["instance_id"]})
  finally:ext.close()
 return out
def task_summary(rows):
 def one(z):return {"pair_gap":stat([x["pair_gap"] for x in z]),"normalized_pair_gap":stat([x["normalized_pair_gap"] for x in z]),"fraction_gt_real_q95":float(np.mean([x["fraction_gt_real_q95"] for x in z])),"real_bank_top1":float(np.mean([x["real_bank_top1"] for x in z])),"real_bank_top5":float(np.mean([x["real_bank_top5"] for x in z])),"same_instance_nearest":float(np.mean([x["same_instance_nearest"] for x in z])),"n":len(z)}
 return {a:{"overall":one([x for x in rows if x["arm"]==a]),"per_class":{c:one([x for x in rows if x["arm"]==a and x["class_name"]==c]) for c in CLASSES}} for a in ARMS}
def feature_diversity(features):
 groups=defaultdict(list)
 for x in features:groups[(x["instance_id"],x["class_name"],x["arm"])].append(x)
 raw=[]
 for (iid,c,a),z in groups.items():
  z=sorted(z,key=lambda x:x["seed"]);f=np.stack([np.asarray(x["feature"]) for x in z]);pairs=[1-float(np.dot(f[i],f[j])) for i,j in ((0,1),(0,2),(1,2))];raw.append({"instance_id":iid,"class_name":c,"arm":a,"corrected_feature_variance":float(f.var(0).mean()),"corrected_pairwise_feature_distance":float(np.mean(pairs))})
 fields=("corrected_feature_variance","corrected_pairwise_feature_distance")
 def summ(z):return {k:stat([x[k] for x in z]) for k in fields}
 return raw,{a:{"overall":summ([x for x in raw if x["arm"]==a]),"per_class":{c:summ([x for x in raw if x["arm"]==a and x["class_name"]==c]) for c in CLASSES}} for a in ARMS}
def generated_cluster_renamed(old):
 out={}
 for arm,groups in old["metrics"].items():
  out[arm]={}
  for group,v in groups.items():out[arm][group]={"generated_cluster_top1":v["retrieval_top1"],"generated_cluster_top5":v["retrieval_top5"],"generated_within_over_total":v["within_over_total"],"generated_source_identity_R2":v["source_identity_R2"],"n":v["n"]}
 return out
def reproduction(features,a):
 probe=load(a.stage9c_probe_sources);ids={x["instance_id"] for x in probe["sources"]};z=[x for x in features if x["arm"]=="class_finetuned_sd2" and x["seed"]==42 and x["instance_id"] in ids];got=task_summary(z)["class_finetuned_sd2"];expected=load(a.stage9c_metrics)["task_feature"]["class_finetuned_sd2"];rows={};ok=True
 for c in CLASSES:
  g=got["per_class"][c];e=expected["per_class"][c];dg=abs(g["normalized_pair_gap"]["mean"]-e["normalized_pair_gap"]["mean"]);dq=abs(g["fraction_gt_real_q95"]-e["fraction_gt_real_q95"]);rows[c]={"sources":g["n"],"corrected_normalized_pair_gap_mean":g["normalized_pair_gap"]["mean"],"stage9c_normalized_pair_gap_mean":e["normalized_pair_gap"]["mean"],"absolute_gap_difference":dg,"corrected_fraction_gt_real_q95":g["fraction_gt_real_q95"],"stage9c_fraction_gt_real_q95":e["fraction_gt_real_q95"],"absolute_fraction_difference":dq,"pass":g["n"]==30 and dg<=1e-6 and dq<=1e-12};ok&=rows[c]["pass"]
 return {"status":"FEATURE_PROTOCOL_REPRODUCTION_PASSED" if ok else "FEATURE_PROTOCOL_REPRODUCTION_FAILED","tolerance":{"normalized_pair_gap_mean":1e-6,"fraction_gt_real_q95":1e-12},"per_class":rows,"official_validation_use_count":0},ok
def main():
 p=argparse.ArgumentParser();
 for n in ("stage9e_manifest","stage9e_diversity","stage9e_identity","stage9e_fidelity","stage9e_training","stage9e_gate","stage9c_probe_sources","stage9c_metrics","folds","stage3r","stage3r_runs","gate_protocol","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);manifest=load(a.stage9e_manifest);rows=manifest["records"]
 pairing={"generation_rerun_count":0,"record_count":len(rows),"arm_count":3,"source_count":len({x["instance_id"] for x in rows}),"unique_source_seed_keys":len({(x["instance_id"],x["generation_seed"]) for x in rows}),"per_class_records":{c:sum(x["class_name"]==c for x in rows) for c in CLASSES},"per_seed_records":{str(s):sum(x["generation_seed"]==s for x in rows) for s in (42,2026,3407)},"exact_matrix":len(rows)==774 and len({(x["instance_id"],x["generation_seed"]) for x in rows})==774,"official_validation_use_count":0};save(a.output/"source_seed_pairing_reaudit.json",pairing);assert pairing["exact_matrix"]
 protocol={"fidelity_and_pixel_diversity_roi":"Stage7B adaptive padded max(64,4x bbox), crop then resize 256","task_feature_roi":"tight nonzero defect mask bbox, identical to Stage9C and historical OOF q95","oof_real_bank":"corrected Stage3R fold/class bank","q95":"frozen manifold_reference_corrected.json","real_bank_retrieval":"synthetic to corresponding fold/class OOF real bank","generated_cluster_retrieval":"generated seeds to generated source centroids; retained Stage9E metric and explicitly separate","generation_rerun_count":0,"detector_training_count":0,"official_validation_use_count":0};save(a.output/"feature_protocol_audit.json",protocol)
 features=extract(rows,a);repro,repro_ok=reproduction(features,a);save(a.output/"stage9c_reproduction_audit.json",repro)
 if not repro_ok:save(a.output/"stage9er_status.json",{"status":"FEATURE_PROTOCOL_REPRODUCTION_FAILED","STAGE9F_DIAGNOSTIC_DETECTOR_PILOT_WORTH_TESTING":False,"historical_stage9e_detector_authorized":False});raise RuntimeError("STAGE9ER_FEATURE_PROTOCOL_REPRODUCTION_FAILED")
 task=task_summary(features);save(a.output/"corrected_task_feature_metrics.json",{"metrics":task,"raw_records":[{k:v for k,v in x.items() if k!="feature"} for x in features]});raw,fd=feature_diversity(features);pixel=load(a.stage9e_diversity)["metrics"];div={a0:{"overall":{**fd[a0]["overall"],"ic_lpips":pixel[a0]["overall"]["ic_lpips"],"edge_diversity":pixel[a0]["overall"]["edge_diversity"]},"per_class":{c:{**fd[a0]["per_class"][c],"ic_lpips":pixel[a0]["per_class"][c]["ic_lpips"],"edge_diversity":pixel[a0]["per_class"][c]["edge_diversity"]} for c in CLASSES}} for a0 in ARMS};save(a.output/"corrected_diversity_metrics.json",{"pixel_diversity_protocol":"Stage7B padded crop (reused without recomputation)","task_feature_diversity_protocol":"tight defect bbox (recomputed)","metrics":div,"raw_corrected_feature_diversity":raw})
 generated=generated_cluster_renamed(load(a.stage9e_identity));dual={"generated_source_cluster_retrieval":{"protocol":"generated seeds to generated source centroid; Stage9E values retained and renamed","metrics":generated},"real_bank_source_retrieval":{"protocol":"tight-bbox synthetic feature to fold/class OOF real bank","metrics":task}};save(a.output/"source_identity_dual_protocol.json",dual)
 training=all(v["successful_updates"]==v["actual_parameter_update_count"]==2000 and v["finite_fraction"]==1 for v in load(a.stage9e_training).values());improve={c:task["repaired_frozen_adapter"]["per_class"][c]["normalized_pair_gap"]["mean"]<task["pretrained_no_adaptation"]["per_class"][c]["normalized_pair_gap"]["mean"] and task["repaired_frozen_adapter"]["per_class"][c]["fraction_gt_real_q95"]<task["pretrained_no_adaptation"]["per_class"][c]["fraction_gt_real_q95"] for c in CLASSES};collapse=any(task["repaired_frozen_adapter"]["per_class"][c]["fraction_gt_real_q95"]>=1 or not improve[c] for c in CLASSES);task_status="TASK_ADAPTATION_SUPPORTED" if all(improve.values()) and not collapse else "TASK_ADAPTATION_MIXED" if any(improve.values()) else "TASK_ADAPTATION_NOT_SUPPORTED";useful=all(div["repaired_frozen_adapter"]["per_class"][c][k]["mean"]>div["class_finetuned_sd2"]["per_class"][c][k]["mean"] for c in CLASSES for k in ("ic_lpips","corrected_feature_variance","corrected_pairwise_feature_distance")) and all(generated["repaired_frozen_adapter"][c]["generated_source_identity_R2"]<generated["class_finetuned_sd2"][c]["generated_source_identity_R2"] for c in CLASSES) and not collapse;worth=training and task_status=="TASK_ADAPTATION_SUPPORTED" and useful and not collapse;gate=load(a.gate_protocol);assert gate["frozen_before_corrected_metrics"] and load(a.stage9e_gate)["STAGE9F_DETECTOR_PILOT_AUTHORIZED"] is False
 status={"status":task_status,"useful_variation_status":"USEFUL_VARIATION_SUPPORTED" if useful else "USEFUL_VARIATION_NOT_SUPPORTED","stage9e_training_valid":training,"catastrophic_real_manifold_collapse":collapse,"class_task_improvement":improve,"STAGE9F_DIAGNOSTIC_DETECTOR_PILOT_WORTH_TESTING":worth,"historical_stage9e_detector_authorized":False,"historical_stage9e_gate_modified":False,"generation_rerun_count":0,"detector_training_count":0,"official_validation_use_count":0};save(a.output/"stage9er_status.json",status);save(a.output/"per_class_corrected_metrics.json",{c:{"task":{arm:task[arm]["per_class"][c] for arm in ARMS},"diversity":{arm:div[arm]["per_class"][c] for arm in ARMS},"generated_cluster_identity":{arm:generated[arm][c] for arm in ARMS}} for c in CLASSES});fidelity=load(a.stage9e_fidelity)["metrics"]
 (a.output/"STAGE9ER_REPORT.md").write_text(f"# DeepPCB Stage9E-R Task Feature Protocol Correction\n\nTask status: **{task_status}**.  \nUseful variation: **{status['useful_variation_status']}**.\n\n- Stage9C reproduction: {repro['status']}\n- Stage9E training valid: {training}\n- Catastrophic real-manifold collapse: {collapse}\n- Stage9F diagnostic detector pilot worth testing: {worth}\n- Historical Stage9E detector authorization remains false.\n\nTask features were recomputed with the historical tight defect bbox. Fidelity and pixel diversity retain the Stage7B padded crop.\n\nFrozen Stage9E overall fidelity (not recomputed):\n\n- class-finetuned SD2: LPIPS {fidelity['class_finetuned_sd2']['overall']['lpips']['mean']:.6f}; edge {fidelity['class_finetuned_sd2']['overall']['edge_cosine']['mean']:.6f}\n- repaired Frozen Adapter: LPIPS {fidelity['repaired_frozen_adapter']['overall']['lpips']['mean']:.6f}; edge {fidelity['repaired_frozen_adapter']['overall']['edge_cosine']['mean']:.6f}\n- pretrained no-adaptation: LPIPS {fidelity['pretrained_no_adaptation']['overall']['lpips']['mean']:.6f}; edge {fidelity['pretrained_no_adaptation']['overall']['edge_cosine']['mean']:.6f}\n\nThe historical Stage9E fidelity gate remains false. No image generation, model training, detector training, or official validation was performed.\n")
 print(json.dumps(status,indent=2))
if __name__=="__main__":main()
