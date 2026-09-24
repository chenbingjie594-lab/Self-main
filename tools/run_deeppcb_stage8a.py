"""Stage8A-1: frozen-checkpoint, seed42 MSDF mechanism interventions."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

from run_deeppcb_stage2_pool import CLASSES, link_or_copy, prepare_batch
from run_deeppcb_stage5b import load, sha
from run_deeppcb_stage5c import save

VARIANTS={
 "full":{},
 "no_latent":{"latent_branch":False},
 "no_pixel":{"pixel_branch":False},
 "no_multiscale":{"up_block_mask":[False,False,False,False]},
 "residual_half":{"residual_scale":0.5},
 "residual_zero":{"residual_scale":0.0},
}

def fail(ok,msg):
 if not ok: raise RuntimeError("STAGE8A_"+msg)

def audits(repo,output):
 src=repo/"diffusers/pipelines/stable_diffusion/msdf_guidance.py"
 graph={"source":str(src),"source_sha256":sha(src),"format_version":3,"components":[
  {"id":"pixel_residual_branch","input":["reference_pixels Bx3x512x512","reference_mask Bx1x512x512"],"output":"pixel_encoded Bx64x64x64; pixel_support Bx1x64x64","injection":"fusion input and reference_support","residual_scale":None,"timestep_dependency":False,"trainable_parameters":["pixel_encoder.*","pixel_support_head.*","fusion.*"],"training_enabled":True,"inference_enabled":True,"depends_on":["reference mask","local pixel background"]},
  {"id":"latent_semantic_branch","input":["reference_latents Bx4x64x64","reference_mask Bx1x64x64"],"output":"latent_encoded Bx64x64x64","injection":"fusion input","residual_scale":None,"timestep_dependency":False,"trainable_parameters":["encoder.*","fusion.*"],"training_enabled":True,"inference_enabled":True,"depends_on":["VAE mode latent","local latent background"]},
  {"id":"reference_morphology_alignment","input":["fused feature Bx64x64x64","reference_support Bx1x64x64","target_prior Bx1x64x64"],"output":"aligned feature/support at target centre","injection":"before support masking and pyramid hooks","residual_scale":None,"timestep_dependency":False,"trainable_parameters":[],"training_enabled":True,"inference_enabled":True,"depends_on":["pixel branch","latent branch","support heads","context_scale"]},
  {"id":"multiscale_up_block_injection","input":["aligned feature/support","four UNet up-block outputs"],"output":"four residual-updated hidden tensors","shape":"B x [1280,1280,640,320] x stage spatial size (checkpoint verified at load)","injection":"forward hook after each unet.up_blocks[i]","residual_scale":"learned projection x dynamic gate x diagnostic residual_scale","timestep_dependency":True,"trainable_parameters":["projections.0..3.*"],"training_enabled":True,"inference_enabled":True,"depends_on":["all upstream components","dynamic gate","bounded injection"]},
  {"id":"scale_time_dynamic_gate","input":["log target-mask area","normalized diffusion timestep","aligned-feature RMS"],"output":"Bx4 gates in [0,max_injection]","injection":"per up-block residual multiplier","residual_scale":"max_injection=checkpoint value; high-resolution small-mask bias","timestep_dependency":True,"trainable_parameters":["gate.*"],"training_enabled":True,"inference_enabled":True,"depends_on":["alignment response","target mask"]},
  {"id":"bounded_residual_injection","input":["projected gated residual","current UNet hidden RMS"],"output":"hidden + limit*tanh(residual/limit)","injection":"end of every hooked up block","residual_scale":"limit=max_residual_ratio*hidden_rms","timestep_dependency":"indirect through hidden and gate","trainable_parameters":[],"training_enabled":True,"inference_enabled":True,"depends_on":["multiscale projection","dynamic gate"]},
 ]}
 feasibility={"checkpoint_retrained":False,"official_claim_scope":"diagnostic intervention only","components":{
  "residual_scale":{"class":"A","tensor_graph_safe":True,"fair_causal_ablation":False,"reason":"continuous multiplier is graph-safe, but checkpoint and UNet were jointly trained at scale 1"},
  "multiscale_up_block_injection":{"class":"A","tensor_graph_safe":True,"fair_causal_ablation":False,"reason":"hooks can be gated independently; joint training creates train-test mismatch"},
  "pixel_residual_branch":{"class":"A","tensor_graph_safe":True,"fair_causal_ablation":False,"reason":"training modality dropout exposed the fusion to missing pixel features, but complete removal remains diagnostic"},
  "latent_semantic_branch":{"class":"A","tensor_graph_safe":True,"fair_causal_ablation":False,"reason":"training modality dropout exposed the fusion to missing latent features, but complete removal remains diagnostic"},
  "reference_morphology_alignment":{"class":"B","tensor_graph_safe":True,"fair_causal_ablation":False,"reason":"bypass changes spatial semantics and support distribution"},
  "scale_time_dynamic_gate":{"class":"B","tensor_graph_safe":True,"fair_causal_ablation":False,"reason":"learned jointly with projections; zero is equivalent to removing all adapter injection"},
  "bounded_residual_injection":{"class":"B","tensor_graph_safe":False,"fair_causal_ablation":False,"reason":"removing the bound changes activation range and can be numerically unsafe"}},
  "warning":"A denotes directly executable inference intervention, not retraining-free causal fairness."}
 configs={"generation_seed":42,"sources":775,"detector_training_count":0,"variants":[{"name":k,"ablation":v,"feasibility":"A-diagnostic-inference"} for k,v in VARIANTS.items()],"excluded":{"no_morphology":"B: tensor-valid bypass but not safe/fair without retraining","unbounded_residual":"B: numerically unsafe and requires retraining"},"shared":{"prompt":"a photo of a sks defect","steps":50,"guidance_scale":7.5,"source_mask_reference":"frozen Stage2 seed42","checkpoint":"frozen MSDF-v3"}}
 save(output/"msdf_component_graph.json",graph);save(output/"ablation_feasibility.json",feasibility);save(output/"ablation_configs.json",configs)

def paths(root,variant,task):
 base=root/variant/task["class_name"]
 return base/(task["candidate_id"]+".jpg"),base/(task["candidate_id"]+".png")

def main():
 p=argparse.ArgumentParser()
 for n in ("pool_manifest","registry","model_root","scheduler","repo_root","pool_root","work_root","output","logs"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.work_root.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
 audits(a.repo_root,a.output)
 tasks=[x for x in load(a.pool_manifest)["records"] if int(x["generation_seed"])==42]
 registry={x["instance_id"]:x for x in load(a.registry)["instances"]}
 fail(len(tasks)==775 and Counter(x["class_name"] for x in tasks)==Counter({"open":130,**{c:129 for c in CLASSES[1:]}}),"SOURCE_MATRIX")
 fail(len(registry)==775 and {x["target_instance_id"] for x in tasks}==set(registry),"REGISTRY_IDENTITY")
 view=a.work_root/"checkpoint_view/DeepPCB";view.mkdir(parents=True,exist_ok=True)
 for c in CLASSES:
  dst=view/c
  if not dst.exists():os.symlink((a.model_root/c).resolve(),dst,target_is_directory=True)
 grouped=defaultdict(list)
 for x in tasks:grouped[x["class_name"]].append(x)
 records=[]; full_identity=[]
 for variant,control in VARIANTS.items():
  cfg={"experiment_name":"deeppcb_stage8a_"+variant,"pipeline_mode":"custom","seed":42,"device":"cuda","dtype":"float16","prompt":"a photo of a sks defect","negative_prompt":None,"num_inference_steps":50,"guidance_scale":7.5,"blur_factor":0,"normal_filter":{"enabled":False},"modules":{"prompt_perturbation":{"enabled":False},"spatial_guidance":{"enabled":False},"cama":{"enabled":False},"ddim_noise":{"enabled":False},"mdap":{"enabled":False},"rda":{"enabled":False},"carf":{"enabled":False},"msdf":{"enabled":True,"root":str((a.work_root/"checkpoint_view").resolve()),"filename":"msdf.pt","ablation":control}}}
  cfgp=a.work_root/f"config_{variant}.json";cfgp.write_text(json.dumps(cfg,indent=2))
  for cls,batch in grouped.items():
   if all(paths(a.pool_root,variant,x)[0].is_file() for x in batch):continue
   br=prepare_batch(batch,cls,42,a.work_root/variant);gen=br/"generated";log=a.logs/f"{variant}_{cls}.log"
   cmd=[sys.executable,"inference.py","--model_ckpt_root",str(a.work_root/"checkpoint_view"),"--ddim_scheduler_root",str(a.scheduler),"--config",str(cfgp),"--categories","DeepPCB","--base_dir",str(br/"input"),"--reference_base_dir",str(br/"references"),"--use_paired_normal","--dataset_type","mvtec","--output_name",str(gen),"--seed","42"]
   with log.open("w") as f:rc=subprocess.run(cmd,cwd=a.repo_root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,"CUDA_VISIBLE_DEVICES":a.device}).returncode
   fail(rc==0,f"GENERATION_{variant}_{cls}")
   source=gen/cfg["experiment_name"]/"DeepPCB"/cls
   for i,x in enumerate(batch):
    image,mask=paths(a.pool_root,variant,x);link_or_copy(source/"image"/f"{i}.jpg",image);link_or_copy(source/"masks"/f"{i}.jpg",mask)
  for x in tasks:
   image,mask=paths(a.pool_root,variant,x);fail(image.is_file() and mask.is_file(),"OUTPUT_MISSING")
   if variant=="full":
    original=Path(x["image_path"])
    if not original.is_absolute():original=a.repo_root/original
    full_identity.append(sha(image)==sha(original))
   records.append({"variant":variant,"instance_id":x["target_instance_id"],"class_name":x["class_name"],"generation_seed":42,"image_path":str(image),"output_mask_path":str(mask),"real_path":registry[x["target_instance_id"]]["defect_image"],"reference_path":x["reference_image"],"reference_instance_id":x["reference_instance_id"],"reference_is_self":x["reference_is_self"],"normal_path":x["background_path"],"mask_path":x["target_mask"],"pair_id":x["target_pair_id"],"image_sha256":sha(image)})
 fail(len(full_identity)==775 and all(full_identity),"FULL_NOT_IDENTICAL_TO_FROZEN_STAGE2")
 save(a.output/"stage8a_generated_manifest.json",{"records":records,"variants":list(VARIANTS),"sources":775,"generation_seed":42,"full_matches_frozen_stage2_seed42":True,"target_real_registry":str(a.registry),"official_validation_use_count":0})
 print(json.dumps({"generated":len(records),"variants":list(VARIANTS)},indent=2))
if __name__=="__main__":main()
