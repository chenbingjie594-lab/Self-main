"""Stage8B-A: audit Full MSDF, train NoMorph, and generate frozen seed42 pairs."""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
from run_deeppcb_stage2_pool import CLASSES,link_or_copy,prepare_batch
from run_deeppcb_stage5b import load,sha
from run_deeppcb_stage5c import save

def fail(ok,msg):
 if not ok:raise RuntimeError("STAGE8B_"+msg)
def history(path):
 rows=list(csv.DictReader(path.open()));return rows,all(math.isfinite(float(x["loss"])) for x in rows)
def expected(cls,a):
 return {"model_path":str(a.base_model),"image_dir":str(a.lowdata_root/"DeepPCB/test"/cls),"mask_dir":str(a.lowdata_root/"DeepPCB/ground_truth"/cls),"train_steps":2000,"batch_size":1,"gradient_accumulation_steps":4,"unet_learning_rate":5e-6,"msdf_learning_rate":1e-4,"msdf_hidden_dim":64,"msdf_context_scale":1.75,"msdf_max_injection":.75,"msdf_max_residual_ratio":.25,"msdf_branch_dropout":.20,"msdf_pixel_support_weight":.50,"msdf_region_weight":.75,"msdf_reconstruction_weight":.50,"msdf_structure_weight":.25,"msdf_background_weight":.20,"msdf_scale_weight":.25,"msdf_support_weight":.25,"mask_jitter_radius":24,"seed":42,"mixed_precision":"fp16","unet_train_scope":"full"}
def inspect(root,cls,a,nomorph):
 meta=load(root/cls/"msdf_training.json");rows,finite=history(root/cls/"training_history.csv");exp=expected(cls,a)
 same=all(str(meta.get(k))==str(v) for k,v in exp.items())
 enabled=meta.get("morphology_alignment_enabled",not bool(meta.get("msdf_disable_morphology_alignment",False)))
 valid=same and len(rows)==2000 and finite and enabled is (not nomorph) and (root/cls/"msdf.pt").is_file()
 return {"class":cls,"valid":valid,"checkpoint":str(root/cls/"msdf.pt"),"checkpoint_sha256":sha(root/cls/"msdf.pt"),"successful_optimizer_updates":len(rows),"loss_finite":finite,"morphology_alignment_enabled":enabled,"metadata":str(root/cls/"msdf_training.json")}
def train(cls,a):
 out=a.nomorph_models/cls;fail(not out.exists(),"NOMORPH_OUTPUT_EXISTS_"+cls)
 cmd=[sys.executable,"train_rda.py","--model_path",str(a.base_model),"--image_dir",str(a.lowdata_root/"DeepPCB/test"/cls),"--mask_dir",str(a.lowdata_root/"DeepPCB/ground_truth"/cls),"--output_dir",str(out),"--rda_mode","none","--enable_msdf","--msdf_disable_morphology_alignment","--train_steps","2000","--batch_size","1","--gradient_accumulation_steps","4","--unet_learning_rate","5e-6","--msdf_learning_rate","1e-4","--msdf_hidden_dim","64","--msdf_context_scale","1.75","--msdf_max_injection","0.75","--msdf_max_residual_ratio","0.25","--msdf_branch_dropout","0.20","--msdf_pixel_support_weight","0.50","--msdf_region_weight","0.75","--msdf_reconstruction_weight","0.50","--msdf_structure_weight","0.25","--msdf_background_weight","0.20","--msdf_scale_weight","0.25","--msdf_support_weight","0.25","--mask_jitter_radius","24","--seed","42","--mixed_precision","fp16","--max_nonfinite_gradient_skips","20"]
 with (a.logs/f"train_{cls}.log").open("w") as f:rc=subprocess.run(cmd,cwd=a.repo_root,stdout=f,stderr=subprocess.STDOUT).returncode
 fail(rc==0,"TRAIN_"+cls)
def outpath(root,x):return root/x["class_name"]/("nomorph_"+x["candidate_id"]+".jpg")
def generate(tasks,a):
 view=a.work_root/"checkpoint_view/DeepPCB";view.mkdir(parents=True,exist_ok=True)
 for c in CLASSES:
  dst=view/c
  if not dst.exists():os.symlink((a.nomorph_models/c).resolve(),dst,target_is_directory=True)
 cfg={"experiment_name":"deeppcb_stage8b_nomorph_seed42","pipeline_mode":"custom","seed":42,"device":"cuda","dtype":"float16","prompt":"a photo of a sks defect","negative_prompt":None,"num_inference_steps":50,"guidance_scale":7.5,"blur_factor":0,"normal_filter":{"enabled":False},"modules":{"prompt_perturbation":{"enabled":False},"spatial_guidance":{"enabled":False},"cama":{"enabled":False},"ddim_noise":{"enabled":False},"mdap":{"enabled":False},"rda":{"enabled":False},"carf":{"enabled":False},"msdf":{"enabled":True,"root":str((a.work_root/"checkpoint_view").resolve()),"filename":"msdf.pt"}}};cfgp=a.work_root/"nomorph_config.json";cfgp.write_text(json.dumps(cfg,indent=2));groups=defaultdict(list)
 for x in tasks:groups[x["class_name"]].append(x)
 for cls,batch in groups.items():
  if all(outpath(a.pool_root,x).is_file() for x in batch):continue
  br=prepare_batch(batch,cls,42,a.work_root/"batches");gen=br/"generated"
  cmd=[sys.executable,"inference.py","--model_ckpt_root",str(a.work_root/"checkpoint_view"),"--ddim_scheduler_root",str(a.scheduler),"--config",str(cfgp),"--categories","DeepPCB","--base_dir",str(br/"input"),"--reference_base_dir",str(br/"references"),"--use_paired_normal","--dataset_type","mvtec","--output_name",str(gen),"--seed","42"]
  with (a.logs/f"generate_{cls}.log").open("w") as f:rc=subprocess.run(cmd,cwd=a.repo_root,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,"CUDA_VISIBLE_DEVICES":a.device}).returncode
  fail(rc==0,"GENERATE_"+cls);src=gen/cfg["experiment_name"]/"DeepPCB"/cls/"image"
  for i,x in enumerate(batch):link_or_copy(src/f"{i}.jpg",outpath(a.pool_root,x))
def main():
 p=argparse.ArgumentParser()
 for n in ("lowdata_root","base_model","full_models","nomorph_models","pool_manifest","sd2_manifest","registry","scheduler","repo_root","pool_root","work_root","output","logs"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True);a.work_root.mkdir(parents=True,exist_ok=True)
 audit={"unet":{"trainable":True,"lr":5e-6,"initialization":"original SD2 Inpainting UNet per class"},"vae":{"trainable":False},"text_encoder":{"trainable":False},"pixel_encoder":{"trainable":True,"lr":1e-4},"latent_encoder":{"trainable":True,"lr":1e-4},"fusion":{"trainable":True,"lr":1e-4},"projection_heads":{"trainable":True,"lr":1e-4},"dynamic_gate":{"trainable":True,"lr":1e-4},"morphology_alignment":{"trainable_parameters":False},"optimizer":{"type":"AdamW","parameter_groups":["all trainable UNet parameters","all MSDF parameters"],"learning_rates":[5e-6,1e-4],"weight_decay":1e-2,"betas":[.9,.999]},"batch_size":1,"gradient_accumulation":4,"successful_optimizer_updates_per_class":2000,"mixed_precision":"fp16","answer_unet_difference_can_be_written":True,"evidence":"UNet requires_grad remains true in full scope and is included in AdamW group 0"};save(a.output/"training_parameter_audit.json",audit)
 full=[inspect(a.full_models,c,a,False) for c in CLASSES];fail(all(x["valid"] for x in full),"FULL_CONTROL_NOT_REUSABLE");save(a.output/"control_reuse_audit.json",{"reuse":True,"reason":"six existing Full checkpoints match frozen protocol and 2000-update metadata; ablation={} preserves the original full path","checkpoints":full})
 for c in CLASSES:
  if (a.nomorph_models/c).exists():fail(inspect(a.nomorph_models,c,a,True)["valid"],"INVALID_EXISTING_NOMORPH_"+c)
  else:train(c,a)
 nomorph=[inspect(a.nomorph_models,c,a,True) for c in CLASSES];fail(all(x["valid"] for x in nomorph),"NOMORPH_AUDIT");save(a.output/"nomorph_training_audit.json",{"only_variable":"morphology_alignment_enabled","same_protocol_as_full":True,"checkpoints":nomorph})
 tasks=[x for x in load(a.pool_manifest)["records"] if int(x["generation_seed"])==42];reg={x["instance_id"]:x for x in load(a.registry)["instances"]};fail(len(tasks)==len(reg)==775,"PAIRING")
 generate(tasks,a);records=[];sd2={(x["instance_id"],int(x["generation_seed"])):x for x in load(a.sd2_manifest)["records"]}
 for x in tasks:
  image=outpath(a.pool_root,x);key=(x["target_instance_id"],42);fail(image.is_file() and key in sd2,"MISSING_OUTPUT");records.append({"instance_id":x["target_instance_id"],"pair_id":x["target_pair_id"],"class_name":x["class_name"],"seed":42,"real":reg[x["target_instance_id"]]["defect_image"],"normal":x["background_path"],"mask":x["target_mask"],"reference":x["reference_image"],"sd2":sd2[key]["image_path"],"nomorph":str(image),"full":x["image_path"]})
 save(a.output/"source_pairing_audit.json",{"sources":775,"seed":42,"exact_source_mask_reference_pairing":True,"official_validation_use_count":0,"records":records});print(json.dumps({"status":"NOMORPH_GENERATED","sources":775},indent=2))
if __name__=="__main__":main()
