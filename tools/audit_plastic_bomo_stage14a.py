"""Restore frozen Plastic_Bomo definitions/history before Stage14A image diagnostics."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path

def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def present(p):return str(p) if Path(p).exists() else "NOT_AVAILABLE"
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","baseline_experiment","baseline_config","random_manifest","dwbg_manifest","candidate_pool","downstream_results","downstream_audit","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);base=load(a.baseline_experiment);pool=load(a.candidate_pool);candidates=pool.get("candidates",pool.get("selected",[]));random=load(a.random_manifest);dwbg=load(a.dwbg_manifest);down=load(a.downstream_results);audit=load(a.downstream_audit)
 split=Path(base["dataset"]["manifest"]);split_doc=load(split) if split.exists() else None
 dataset={"status":"RESTORED_FROM_FROZEN_PROJECT_FILES","dataset":"Plastic_Bomo","dataset_root":base["dataset"]["source"],"train_root":base["dataset"]["train_root"],"validation_root":base["dataset"]["eval_root"],"test_root":"NOT_AVAILABLE","classes":[{"id":0,"name":"flash","source_name":"01_Flash_point"},{"id":1,"name":"black","source_name":"02_Big_black_spots"}],"split_seed":base["dataset"]["split_seed"],"train_ratio":base["dataset"]["train_ratio"],"low_shot_definition":"frozen Plastic_Bomo split70 seed42 training split","low_shot_manifest":str(split),"validation_manifest":str(split),"test_manifest":"NOT_AVAILABLE","split_manifest_sha256":sha(split) if split.exists() else "NOT_AVAILABLE","split_manifest":split_doc if split_doc else "NOT_AVAILABLE","historical_detector_dataset_audit":audit,"no_resplit":True,"deep_pcb_definitions_imported":False}
 save(a.output/"plastic_bomo_dataset_definition.json",dataset)
 methods=[{"method":"Vanilla_SD2","checkpoint":base["training"]["checkpoint_root"],"training_config":str(a.baseline_config),"source_real_images":base["training"].get("train_root",base["dataset"]["train_root"]),"generated_image_directory":"NOT_AVAILABLE","generated_sample_count":"NOT_AVAILABLE","per_class_sample_count":"NOT_AVAILABLE","generation_seed":base["inference"]["seed"],"mask_bbox_prompt_protocol":load(a.baseline_config),"detector_result":"NOT_AVAILABLE"},{"method":"MSDF_v3_DWBG_pool","checkpoint":"NOT_AVAILABLE","training_config":"configs/dwbg_v2.json","source_real_images":base["dataset"]["train_root"],"generated_image_directory":"paths recorded per candidate","generated_sample_count":len(candidates),"per_class_sample_count":dict(Counter(x["class_name"] for x in candidates)),"generation_seed":sorted({int(x["seed"]) for x in candidates}),"mask_bbox_prompt_protocol":"recorded per candidate","detector_result":"results_for_gpt/bootstrap_guard_downstream/validated_per_seed.json"},{"method":"AnomalyDiffusion","checkpoint":"NOT_AVAILABLE","training_config":"NOT_AVAILABLE","source_real_images":"NOT_AVAILABLE","generated_image_directory":"NOT_AVAILABLE","generated_sample_count":"NOT_AVAILABLE","per_class_sample_count":"NOT_AVAILABLE","generation_seed":"NOT_AVAILABLE","mask_bbox_prompt_protocol":"NOT_AVAILABLE","detector_result":"NOT_AVAILABLE"},{"method":"DefectFill","checkpoint":"NOT_AVAILABLE","training_config":"NOT_AVAILABLE","source_real_images":"NOT_AVAILABLE","generated_image_directory":"NOT_AVAILABLE","generated_sample_count":"NOT_AVAILABLE","per_class_sample_count":"NOT_AVAILABLE","generation_seed":"NOT_AVAILABLE","mask_bbox_prompt_protocol":"NOT_AVAILABLE","detector_result":"NOT_AVAILABLE"}]
 save(a.output/"plastic_bomo_historical_experiment_registry.json",{"status":"REPOSITORY_EVIDENCE_ONLY","methods":methods,"candidate_pool_sha256":sha(a.candidate_pool),"no_memory_imputation":True})
 records=down["records"];save(a.output/"plastic_bomo_detector_baselines.json",{"status":"RESTORED","records":records,"requested_arms":{"Real":"real_only","RealRepeat":"NOT_AVAILABLE_STRICT_COMPUTE_MATCH","Random Synthetic":"random","current best synthetic generator":"bootstrap_guard by historical mAP50-95; kept diagnostic-only because BootstrapGuard is second innovation","BootstrapGuard":"bootstrap_guard"},"primary_checkpoint_caveat":"historical validated_per_seed records use best checkpoints/early stopping; Stage14A screening freezes final-epoch last.pt"})
 frozen={"status":"FROZEN_FOR_STAGE14A_SCREENING","detector":"YOLO11s","pretrained":"yolo11s.pt","seed":42,"epochs":150,"patience":40,"batch":1,"imgsz":1536,"optimizer":"auto","deterministic":True,"primary_metric":"mAP50-95","primary_checkpoint":"final epoch last.pt","augmentations":{"rect":False,"augment":False,"mosaic":1.0,"mixup":0.0,"close_mosaic":10},"real_train_images":audit["arms"]["real_only"]["real_train_images"],"real_instances":audit["arms"]["real_only"]["real_instances"],"synthetic_budget":80,"per_class_synthetic_budget":{"flash":40,"black":40},"validation_hash":audit["real_source"]["val_hash"],"deep_pcb_training_count":0,"deep_pcb_generation_count":0}
 save(a.output/"plastic_bomo_frozen_detector_protocol.json",frozen)
 # Freeze metadata thresholds before any new screening detector is trained.
 by=defaultdict(list)
 for x in candidates:by[0 if x["class_id"]==0 else 1].append(x)
 thresholds={}
 for c,z in by.items():
  def q(field,p):
   v=sorted(float(x[field]) for x in z);return v[min(len(v)-1,int(p*(len(v)-1)))]
  thresholds[str(c)]={"fidelity_consensus_q50":q("consensus_score",.5),"context_proxy_q50":q("local_contrast",.5),"real_support_normalized_q25":.25,"real_support_normalized_q95":.95}
 subsets={"random":[x["candidate_id"] for x in random["selected"]],"high_fidelity":[],"valid_novel":[],"high_context_compatibility":[],"low_fidelity_diagnostic":[],"duplicate_like_diagnostic":[],"off_manifold_diagnostic":[],"low_context_diagnostic":[]}
 for c,z in by.items():
  t=thresholds[str(c)];ordered=sorted(z,key=lambda x:x["candidate_id"])
  groups={"high_fidelity":[x for x in ordered if x["consensus_score"]>=t["fidelity_consensus_q50"]],"low_fidelity_diagnostic":[x for x in ordered if x["consensus_score"]<t["fidelity_consensus_q50"]],"duplicate_like_diagnostic":[x for x in ordered if x["median_normalized_manifold_distance"]<=.25],"valid_novel":[x for x in ordered if .25<x["median_normalized_manifold_distance"]<=.95],"off_manifold_diagnostic":[x for x in ordered if x["median_normalized_manifold_distance"]>.95],"high_context_compatibility":[x for x in ordered if x["local_contrast"]>=t["context_proxy_q50"]],"low_context_diagnostic":[x for x in ordered if x["local_contrast"]<t["context_proxy_q50"]]}
  for name,g in groups.items():subsets[name].extend(x["candidate_id"] for x in g[:40])
 save(a.output/"subset_definition.json",{"status":"PROVISIONAL_METADATA_SCREENING_FREEZE","warning":"fidelity/context proxies must be replaced/confirmed by image-level Stage14A audits before detector training","thresholds_frozen_before_downstream":thresholds,"subsets":subsets,"equal_budget_required":80,"random_manifest_sha256":sha(a.random_manifest)})
 overlap={a0:{b0:len(set(subsets[a0])&set(subsets[b0])) for b0 in subsets} for a0 in subsets};save(a.output/"subset_overlap_audit.json",overlap)
 for name in ("fidelity_audit.json","effective_diversity_audit.json","context_compatibility_audit.json","synthetic_task_response.json","subset_training_budget_audit.json","subset_detector_metrics.json","bottleneck_comparison.json"):
  save(a.output/name,{"status":"PENDING_SERVER_IMAGE_OR_DETECTOR_STAGE","no_conclusion":True})
 save(a.output/"stage14a_status.json",{"status":"AUDIT_PREPARATION_COMPLETE_SCREENING_NOT_AUTHORIZED","STAGE14B_AUTO_START":False,"deep_pcb_training_count":0,"deep_pcb_generation_count":0})
 print(json.dumps({"status":"STAGE14A_RESTORATION_COMPLETE","candidate_count":len(candidates),"realrepeat_required":True},indent=2))
if __name__=="__main__":main()
