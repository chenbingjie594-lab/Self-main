"""Stage10B Gate-2: TMVB scalar direction and end-to-end gradient probe."""
from __future__ import annotations
import argparse,json,random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES=("short","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def box(mask):
 m=np.asarray(Image.open(mask).convert("L"))>127;y,x=np.where(m);return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","gate1","stage10ar","instance_registry","folds","stage3r","stage3r_runs","base_model","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);assert load(a.gate1/"stage10b_gate.json")["GATE1_PASSED"]
 import torch
 import torch.nn.functional as F
 from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
 from transformers import CLIPTextModel,CLIPTokenizer
 from torchvision.transforms.functional import pil_to_tensor
 from stage10b_differentiable_features import DifferentiableDetectInputExtractor
 device=torch.device(f"cuda:{a.device}");torch.manual_seed(42);random.seed(42);beta_l,beta_u=cfg["beta_L"],cfg["beta_U"]
 # Exact scalar loss-direction audit for all three states.
 scalar=[]
 for state,z0 in (("below",beta_l*.5),("inside",(beta_l+beta_u)/2),("above",beta_u*1.5)):
  z=torch.tensor(z0,requires_grad=True);loss=F.relu(beta_l-z).square()+F.relu(z-beta_u).square();g=torch.autograd.grad(loss,z,allow_unused=True)[0];gval=0. if g is None else float(g);z1=float(z.detach()-0.05*gval);correct=(state=="below" and z1>z0) or (state=="inside" and float(loss)==0 and gval==0) or (state=="above" and z1<z0);scalar.append({"state":state,"z_before":z0,"loss":float(loss),"gradient":gval,"z_after_tiny_step":z1,"direction_correct":correct})
 tokenizer=CLIPTokenizer.from_pretrained(a.base_model,subfolder="tokenizer");text=CLIPTextModel.from_pretrained(a.base_model,subfolder="text_encoder").to(device).eval();vae=AutoencoderKL.from_pretrained(a.base_model,subfolder="vae").to(device).eval();unet=UNet2DConditionModel.from_pretrained(a.base_model,subfolder="unet").to(device).train();scheduler=DDPMScheduler.from_pretrained(a.base_model,subfolder="scheduler")
 text.requires_grad_(False);vae.requires_grad_(False);unet.enable_gradient_checkpointing();dtype=torch.float16
 ids=tokenizer("a photo of a sks defect",padding="max_length",max_length=tokenizer.model_max_length,truncation=True,return_tensors="pt").input_ids.to(device)
 with torch.no_grad():emb=text(ids)[0]
 t=int(load(a.gate1/"t_band_protocol.json")["timestep"]);alpha=scheduler.alphas_cumprod[t].to(device).float();foldof={pid:int(f["fold"]) for f in load(a.folds)["folds"] for pid in f["holdout_pair_ids"]};radius={x["instance_id"]:x["k5_radius"] for x in load(a.stage10ar/"pair_independent_local_radius.json")["records"]};instances=[x for x in load(a.instance_registry)["instances"] if x["class_name"] in CLASSES];chosen=sum([sorted([x for x in instances if x["class_name"]==c],key=lambda x:x["instance_id"])[:8] for c in CLASSES],[])
 bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");bankpos={x["instance_id"]:(int(f["fold"]),int(x["feature_index"])) for f in bankdoc["folds"] for x in f["records"]};byfold={}
 for x in chosen:byfold.setdefault(foldof[x["pair_id"]],[]).append(x)
 records=[]
 for fold,items in byfold.items():
  teacher=DifferentiableDetectInputExtractor(a.stage3r_runs/f"fold_{fold}/weights/best.pt",a.device,512);bank=np.load(a.stage3r/f"oof_real_bank_corrected_fold{fold}.npz")["features"]
  try:
   for n,x in enumerate(items):
    image=pil_to_tensor(Image.open(x["defect_image"]).convert("RGB").resize((512,512),Image.Resampling.BILINEAR)).float().unsqueeze(0).to(device)/127.5-1;mask=pil_to_tensor(Image.open(x["instance_mask_path"]).convert("L").resize((512,512),Image.Resampling.NEAREST)).float().unsqueeze(0).to(device)/255.;mask=(mask>=.5).float();masked=image*(mask<.5);bbox=box(x["instance_mask_path"]);target=torch.from_numpy(bank[bankpos[x["instance_id"]][1]]).to(device).float();target=F.normalize(target,dim=0).detach();noise_gen=torch.Generator(device=device).manual_seed(42000+len(records));
    with torch.no_grad(),torch.autocast("cuda",dtype=dtype):lat=vae.encode(image).latent_dist.sample(generator=noise_gen)*vae.config.scaling_factor;mlat=vae.encode(masked).latent_dist.sample(generator=noise_gen)*vae.config.scaling_factor
    noise=torch.randn(lat.shape,generator=noise_gen,device=device,dtype=lat.dtype);tt=torch.tensor([t],device=device);noisy=scheduler.add_noise(lat,noise,tt);m=F.interpolate(mask,size=lat.shape[-2:],mode="nearest");model_in=torch.cat((noisy,m.to(noisy.dtype),mlat),1);unet.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=dtype):pred=unet(model_in,tt,encoder_hidden_states=emb).sample
    x0=(noisy.float()-(1-alpha).sqrt()*pred.float())/alpha.sqrt();decoded=vae.decode(x0/vae.config.scaling_factor).sample.float().clamp(-1,1);rgb=(decoded+1)/2;feat=teacher.encode_tensor(rgb,bbox,512,512)[0].float();d=1-F.cosine_similarity(feat,target,dim=0);z=d/radius[x["instance_id"]];loss=F.relu(beta_l-z).square()+F.relu(z-beta_u).square();loss.backward();grads=[p.grad for p in unet.parameters() if p.grad is not None];finite=bool(grads) and all(torch.isfinite(g).all() for g in grads);gn=float(torch.sqrt(sum(g.float().square().sum() for g in grads))) if grads else 0.
    # Isolated predicted-clean latent step; no model parameter/checkpoint is modified.
    leaf=x0.detach().requires_grad_(True);rgb2=(vae.decode(leaf/vae.config.scaling_factor).sample.float().clamp(-1,1)+1)/2;f2=teacher.encode_tensor(rgb2,bbox,512,512)[0].float();z_before=(1-F.cosine_similarity(f2,target,dim=0))/radius[x["instance_id"]];l2=F.relu(beta_l-z_before).square()+F.relu(z_before-beta_u).square();gz=torch.autograd.grad(z_before,leaf,retain_graph=True)[0];g=torch.autograd.grad(l2,leaf,allow_unused=True)[0]
    directional_derivative=0. if g is None else float((gz*(-g)).sum())
    if g is None or float(l2)==0:z_after=float(z_before);direction=True;finite_step_direction=True
    else:
     step=1e-4*g/(g.norm()+1e-12);new=leaf.detach()-step;f3=teacher.encode_tensor((vae.decode(new/vae.config.scaling_factor).sample.float().clamp(-1,1)+1)/2,bbox,512,512)[0].float();z_after=float((1-F.cosine_similarity(f3,target,dim=0))/radius[x["instance_id"]]);finite_step_direction=(float(z_before)<beta_l and z_after>float(z_before)) or (float(z_before)>beta_u and z_after<float(z_before));direction=(float(z_before)<beta_l and directional_derivative>0) or (float(z_before)>beta_u and directional_derivative<0)
    state="below" if float(z)<beta_l else "above" if float(z)>beta_u else "inside";records.append({"instance_id":x["instance_id"],"class_name":x["class_name"],"fold":fold,"z_before":float(z),"violation_type":state,"tmvb_loss":float(loss),"gradient_finite":finite,"unet_receives_gradient":bool(grads) and gn>0,"unet_gradient_norm":gn,"local_z_directional_derivative":directional_derivative,"direction_correct":direction,"diagnostic_latent_z_after":z_after,"diagnostic_finite_step_direction_correct":finite_step_direction,"checkpoint_modified":False});unet.zero_grad(set_to_none=True)
  finally:teacher.close()
 violation=[x for x in records if x["violation_type"]!="inside"];ok=len(records)==16 and all(x["gradient_finite"] for x in records) and all(x["unet_receives_gradient"] for x in violation) and all(x["direction_correct"] for x in violation) and all(x["direction_correct"] for x in scalar);status="TMVB_GRADIENT_MECHANISM_PASSED" if ok else "TMVB_GRADIENT_MECHANISM_FAILED";out={"status":status,"scalar_band_direction_tests":scalar,"source_count":len(records),"per_class":dict(Counter(x["class_name"] for x in records)),"violation_state_counts":dict(Counter(x["violation_type"] for x in records)),"all_gradients_finite":all(x["gradient_finite"] for x in records),"violation_direction_correct_fraction":float(np.mean([x["direction_correct"] for x in violation])) if violation else None,"formal_checkpoint_modified":False,"records":records};save(a.output/"tmvb_gradient_unit_test.json",out);gate=load(a.gate1/"stage10b_gate.json");gate.update({"status":"GATE2_PASSED_READY_FOR_50STEP" if ok else status,"GATE2_PASSED":ok,"FORMAL_TRAINING_AUTHORIZED":False,"STAGE10C_DETECTOR_PILOT_AUTHORIZED":False});save(a.output/"stage10b_gate.json",gate);print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2));
 if not ok:raise RuntimeError(status)
if __name__=="__main__":main()
