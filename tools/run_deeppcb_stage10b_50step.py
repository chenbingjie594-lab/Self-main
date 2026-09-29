"""Stage10B Gate-3: 50-successful-step TMVB training stability test."""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES=("short","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def bbox(mask):
 m=np.asarray(Image.open(mask).convert("L"))>127;y,x=np.where(m);return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
def finite(x):return bool(math.isfinite(float(x)))

def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","gate","stage10ar","instance_registry","folds","stage3r","stage3r_runs","base_model","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");p.add_argument("--steps",type=int,default=50);p.add_argument("--formal",action="store_true");p.add_argument("--model_output",type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cfg=load(a.protocol);gate=load(a.gate/"stage10b_gate.json");assert gate["GATE1_PASSED"] and gate["GATE2_PASSED"]
 if a.formal:
  assert gate.get("GATE3_PASSED") and gate.get("FORMAL_TRAINING_AUTHORIZED") and a.steps==cfg["formal_training"]["successful_updates_per_class"] and a.model_output is not None
  if a.model_output.exists():raise FileExistsError(f"STAGE10B_FORMAL_MODEL_OUTPUT_EXISTS: {a.model_output}")
  a.model_output.mkdir(parents=True)
 else:assert a.steps==50
 import torch
 import torch.nn.functional as F
 from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
 from transformers import CLIPTextModel,CLIPTokenizer
 from torchvision.transforms.functional import pil_to_tensor
 from stage10b_differentiable_features import DifferentiableDetectInputExtractor
 dev=torch.device(f"cuda:{a.device}");dtype=torch.float16;torch.manual_seed(42);random.seed(42);np.random.seed(42)
 tok=CLIPTokenizer.from_pretrained(a.base_model,subfolder="tokenizer");text=CLIPTextModel.from_pretrained(a.base_model,subfolder="text_encoder",torch_dtype=dtype).to(dev).eval();vae=AutoencoderKL.from_pretrained(a.base_model,subfolder="vae",torch_dtype=dtype).to(dev).eval();sched=DDPMScheduler.from_pretrained(a.base_model,subfolder="scheduler")
 text.requires_grad_(False);vae.requires_grad_(False);ids=tok("a photo of a sks defect",padding="max_length",max_length=tok.model_max_length,truncation=True,return_tensors="pt").input_ids.to(dev)
 with torch.no_grad():emb=text(ids)[0]
 t=int(load(a.gate/"t_band_protocol.json")["timestep"]);alpha=sched.alphas_cumprod[t].to(dev).float();bl,bu,lam=cfg["beta_L"],cfg["beta_U"],cfg["lambda_TMVB"]
 foldof={pid:int(f["fold"]) for f in load(a.folds)["folds"] for pid in f["holdout_pair_ids"]};radii={x["instance_id"]:x["k5_radius"] for x in load(a.stage10ar/"pair_independent_local_radius.json")["records"]};instances=load(a.instance_registry)["instances"]
 bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");bankpos={x["instance_id"]:(int(f["fold"]),int(x["feature_index"])) for f in bankdoc["folds"] for x in f["records"]};banks={f:np.load(a.stage3r/f"oof_real_bank_corrected_fold{f}.npz")["features"] for f in range(3)}
 teachers={f:DifferentiableDetectInputExtractor(a.stage3r_runs/f"fold_{f}/weights/best.pt",a.device,512) for f in range(3)};results={}
 try:
  for ci,cls in enumerate(CLASSES):
   # Trainable parameters remain FP32; autocast supplies FP16 compute. GradScaler
   # intentionally rejects FP16 parameter gradients during unscale_.
   torch.manual_seed(42);unet=UNet2DConditionModel.from_pretrained(a.base_model,subfolder="unet").to(dev).train();unet.enable_gradient_checkpointing();unet.set_attention_slice("auto");opt=torch.optim.AdamW(unet.parameters(),lr=cfg["formal_training"]["learning_rate"],betas=(.9,.999),weight_decay=.01,foreach=False);scaler=torch.amp.GradScaler("cuda");rows=[];success=attempt=0;data=sorted([x for x in instances if x["class_name"]==cls],key=lambda x:x["instance_id"]);probe=next(unet.parameters());initial=probe.detach().clone();rng=random.Random(42+ci)
   while success<a.steps and attempt<a.steps*3:
    opt.zero_grad(set_to_none=True);micro=[];attempt+=1
    for mi in range(cfg["formal_training"]["gradient_accumulation"]):
     x=data[rng.randrange(len(data))];image=pil_to_tensor(Image.open(x["defect_image"]).convert("RGB").resize((512,512),Image.Resampling.BILINEAR)).to(dev,dtype).unsqueeze(0)/127.5-1;mask=pil_to_tensor(Image.open(x["instance_mask_path"]).convert("L").resize((512,512),Image.Resampling.NEAREST)).to(dev,dtype).unsqueeze(0)/255;mask=(mask>=.5).to(dtype);masked=image*(mask<.5);fold=foldof[x["pair_id"]];target=torch.from_numpy(banks[fold][bankpos[x["instance_id"]][1]]).to(dev).float();target=F.normalize(target,dim=0);g=torch.Generator(device=dev).manual_seed(420000+attempt*10+mi)
     with torch.no_grad(),torch.autocast("cuda",dtype=dtype):lat=vae.encode(image).latent_dist.sample(generator=g)*vae.config.scaling_factor;mlat=vae.encode(masked).latent_dist.sample(generator=g)*vae.config.scaling_factor
     noise=torch.randn(lat.shape,generator=g,device=dev,dtype=dtype);tt=torch.tensor([t],device=dev);noisy=sched.add_noise(lat,noise,tt);model_in=torch.cat((noisy,F.interpolate(mask,size=lat.shape[-2:],mode="nearest"),mlat),1)
     with torch.autocast("cuda",dtype=dtype):pred=unet(model_in,tt,encoder_hidden_states=emb).sample
     denoise=F.mse_loss(pred.float(),noise.float());x0=(noisy.float()-(1-alpha).sqrt()*pred.float())/alpha.sqrt();rgb=(vae.decode((x0/vae.config.scaling_factor).to(dtype)).sample.float().clamp(-1,1)+1)/2;feat=teachers[fold].encode_tensor(rgb,bbox(x["instance_mask_path"]),512,512)[0].float();z=(1-F.cosine_similarity(feat,target,dim=0))/radii[x["instance_id"]];tmvb=F.relu(bl-z).square()+F.relu(z-bu).square();loss=denoise+lam*tmvb;scaler.scale(loss/4).backward();micro.append({"instance_id":x["instance_id"],"fold":fold,"z":float(z.detach()),"denoise_loss":float(denoise.detach()),"tmvb_loss":float(tmvb.detach()),"total_loss":float(loss.detach())})
    before=probe.detach().clone();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(unet.parameters(),1.0);grad_ok=finite(gn) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in unet.parameters());scale0=scaler.get_scale()
    if grad_ok:scaler.step(opt)
    scaler.update();changed=not torch.equal(before,probe.detach());skipped=not(grad_ok and changed);success+=0 if skipped else 1;rows.append({"attempt":attempt,"successful_step":success,"skipped":skipped,"gradient_finite":grad_ok,"gradient_norm_before_clip":float(gn) if finite(gn) else None,"scale_before":scale0,"scale_after":scaler.get_scale(),"parameter_changed":changed,"micro_batches":micro})
   delta=float((probe.detach()-initial).float().norm());tail=rows[-20:];ok=success==a.steps and delta>0 and rows[-1]["scale_after"]>0 and len(tail)==20 and all(not r["skipped"] and r["gradient_finite"] and r["parameter_changed"] for r in tail);checkpoint=None
   if a.formal and ok:
    checkpoint=a.model_output/cls/"unet";checkpoint.parent.mkdir(parents=True);unet.save_pretrained(checkpoint,safe_serialization=True)
   results[cls]={"status":"PASS" if ok else "FAIL","target_successful_steps":a.steps,"successful_steps":success,"attempted_steps":attempt,"skipped_steps":sum(r["skipped"] for r in rows),"gradient_accumulation":4,"learning_rate":cfg["formal_training"]["learning_rate"],"stable_tail_required":20,"stable_tail_passed":len(tail)==20 and all(not r["skipped"] and r["gradient_finite"] and r["parameter_changed"] for r in tail),"parameter_delta_l2":delta,"checkpoint_saved":checkpoint is not None,"checkpoint_path":str(checkpoint) if checkpoint else None,"records":rows};del opt,unet;torch.cuda.empty_cache()
 finally:
  for teacher in teachers.values():teacher.close()
 ok=all(x["status"]=="PASS" for x in results.values());name="formal_training_audit.json" if a.formal else "tmvb_50step_stability.json";status=("TMVB_FORMAL_TRAINING_COMPLETE" if ok else "TMVB_FORMAL_TRAINING_FAILED") if a.formal else ("TMVB_50STEP_STABLE" if ok else "TMVB_50STEP_UNSTABLE");save(a.output/name,{"status":status,"classes":results,"formal_checkpoint_saved":a.formal and ok,"official_validation_use_count":0,"detector_training_count":0});gate.update({"status":status if a.formal else ("GATE3_PASSED_READY_FOR_FORMAL_TRAINING" if ok else status),"GATE3_PASSED":gate.get("GATE3_PASSED",False) or (ok and not a.formal),"FORMAL_TRAINING_AUTHORIZED":gate.get("FORMAL_TRAINING_AUTHORIZED",False),"FORMAL_TRAINING_COMPLETE":bool(a.formal and ok),"STAGE10C_DETECTOR_PILOT_AUTHORIZED":False});save(a.output/"stage10b_gate.json",gate);print(json.dumps({"status":gate["status"],"classes":{c:{k:v for k,v in x.items() if k!="records"} for c,x in results.items()}},indent=2));
 if not ok:raise RuntimeError(status)
if __name__=="__main__":main()
