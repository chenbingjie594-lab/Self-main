"""Formal Stage9E repaired adapter training with truthful optimizer accounting."""
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
from train_deeppcb_stage9a_adapter import Data,prompt
from stage9e_frozen_adapter import RepairedFrozenResidualAdapter

def norm_delta(model,initial):return float(sum((p.detach()-initial[n]).float().square().sum() for n,p in model.named_parameters()).sqrt())
def main():
 p=argparse.ArgumentParser();p.add_argument("--model_path",type=Path,required=True);p.add_argument("--lowdata_root",type=Path,required=True);p.add_argument("--class_name",choices=("short","pinhole"),required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--gate",type=Path,required=True);p.add_argument("--steps",type=int,default=2000);p.add_argument("--accum",type=int,default=4);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42);p.add_argument("--max_consecutive_nonfinite",type=int,default=32);a=p.parse_args()
 gate=json.loads(a.gate.read_text());assert gate["status"]=="REPAIR_STABLE","STAGE9E_REPAIR_GATE_NOT_STABLE";a.output.mkdir(parents=True,exist_ok=False)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);d=torch.device("cuda");start=time.time();torch.cuda.reset_peak_memory_stats()
 tok=CLIPTokenizer.from_pretrained(a.model_path,subfolder="tokenizer");te=CLIPTextModel.from_pretrained(a.model_path,subfolder="text_encoder").to(d);vae=AutoencoderKL.from_pretrained(a.model_path,subfolder="vae").to(d);unet=UNet2DConditionModel.from_pretrained(a.model_path,subfolder="unet").to(d);sched=DDPMScheduler.from_pretrained(a.model_path,subfolder="scheduler")
 for m in (te,vae,unet):m.requires_grad_(False).eval()
 adapter=RepairedFrozenResidualAdapter().to(d);adapter.attach(unet);adapter.train();initial={n:p.detach().clone() for n,p in adapter.named_parameters()};trainable=sum(p.numel() for p in adapter.parameters());assert trainable==108928,trainable
 opt=torch.optim.AdamW(adapter.parameters(),lr=a.lr,betas=(.9,.999),weight_decay=1e-2,foreach=False);scaler=torch.amp.GradScaler("cuda");loader=DataLoader(Data(a.lowdata_root,a.class_name),batch_size=1,shuffle=True,num_workers=0);it=iter(loader);emb=prompt(tok,te,"a photo of a sks defect",d);attempt=success=skips=nonfinite=actual=consecutive=0;history=[];scales=[]
 while success<a.steps:
  opt.zero_grad(set_to_none=True);loss_sum=0.;sources=[]
  for _ in range(a.accum):
   try:image,mask,normal,ids=next(it)
   except StopIteration:it=iter(loader);image,mask,normal,ids=next(it)
   image,mask,normal=image.to(d),mask.to(d),normal.to(d)
   with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):z=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor;mz=vae.encode(normal*(mask<.5)).latent_dist.sample()*vae.config.scaling_factor
   noise=torch.randn_like(z);t=torch.randint(0,sched.config.num_train_timesteps,(1,),device=d);inp=torch.cat((sched.add_noise(z,noise,t),torch.nn.functional.interpolate(mask,size=z.shape[-2:],mode="nearest"),mz),1);adapter.set_condition(mask,normal,t)
   with torch.autocast("cuda",dtype=torch.float16):pred=unet(inp,t,encoder_hidden_states=emb).sample;loss=torch.nn.functional.mse_loss(pred.float(),noise.float())/a.accum
   scaler.scale(loss).backward();loss_sum+=float(loss)*a.accum/a.accum;sources.append({"source_id":ids[0],"timestep":int(t)})
  attempt+=1;scaler.unscale_(opt);grad_exists=all(p.grad is not None for p in adapter.parameters());finite=grad_exists and all(bool(torch.isfinite(p.grad).all()) for p in adapter.parameters());old=float(scaler.get_scale());before={n:p.detach().clone() for n,p in adapter.named_parameters()}
  if finite:scaler.step(opt)
  scaler.update();new=float(scaler.get_scale());did=finite and any(not torch.equal(before[n],p) for n,p in adapter.named_parameters());scales.append({"attempt":attempt,"before":old,"after":new,"gradient_finite":finite,"parameter_update":did})
  if not did:
   skips+=1;nonfinite+=int(not finite);consecutive+=1
   if new<=0 or consecutive>=a.max_consecutive_nonfinite:raise RuntimeError(f"STAGE9E_FAIL_FAST_NONFINITE attempt={attempt} scale={new} consecutive={consecutive}")
   continue
  consecutive=0;success+=1;actual+=1;history.append({"successful_update":success,"attempt":attempt,"loss":loss_sum,"scale":new,"sources":sources})
 assert success==actual==a.steps;finite_fraction=float(sum(torch.isfinite(p).sum().item() for p in adapter.parameters())/sum(p.numel() for p in adapter.parameters()));projection_norm=float(sum(p.float().square().sum() for n,p in adapter.named_parameters() if n.startswith("projections")).sqrt());delta=norm_delta(adapter,initial);inference_residual=[]
 with torch.no_grad():
  for i,(ch,size) in enumerate(((640,64),(320,128))):
   r=adapter._residual(torch.zeros((1,ch,size,size),device=d,dtype=torch.float16),i);inference_residual.append({"block":i+2,"finite":bool(torch.isfinite(r).all()),"rms":float(r.float().square().mean().sqrt()),"nonzero":bool((r!=0).any())})
 assert finite_fraction==1 and projection_norm>0 and delta>0 and all(x["finite"] and x["nonzero"] for x in inference_residual)
 arch={"channels":[640,320],"hidden":64,"max_rms_ratio":.10};meta={"status":"COMPLETE","class_name":a.class_name,"attempted_updates":attempt,"successful_updates":success,"skipped_updates":skips,"nonfinite_updates":nonfinite,"actual_parameter_update_count":actual,"successful_equals_actual":success==actual,"final_grad_scaler_scale":new,"seed":a.seed,"batch_size":1,"gradient_accumulation":a.accum,"learning_rate":a.lr,"optimizer":"AdamW","mixed_precision":"fp16","trainable_components":["RepairedFrozenResidualAdapter"],"adapter_parameters":trainable,"projection_parameter_norm":projection_norm,"total_adapter_delta_vs_initialization":delta,"finite_fraction":finite_fraction,"inference_residual":inference_residual,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"wall_seconds":time.time()-start,"architecture":arch,"architecture_changed":False,"numerical_change_only":"sqrt(mean(r^2)+eps^2)","forbidden_inputs":{"defect_reference":False,"detector_feature":False,"morphology_template":False,"source_identity":False,"teacher_score":False}}
 adapter.save(a.output/"frozen_adapter.pt",meta);(a.output/"training_audit.json").write_text(json.dumps(meta,indent=2));(a.output/"scale_trajectory.json").write_text(json.dumps(scales,indent=2));
 with (a.output/"training_history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=("successful_update","attempt","loss","scale","sources"));w.writeheader();w.writerows(history)
 print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
