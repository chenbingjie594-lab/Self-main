"""Stage9E RMS unit test, repaired one-step audit, and 50-update stability gate."""
from __future__ import annotations
import argparse,json,math,random,types
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
from train_deeppcb_stage9a_adapter import Data,prompt
from stage9e_frozen_adapter import RepairedFrozenResidualAdapter

CLASSES=("short","pinhole")
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def finite_params(m):return all(bool(torch.isfinite(p).all()) for p in m.parameters())
def param_snapshot(m):return {n:p.detach().clone() for n,p in m.named_parameters()}
def changed(before,m):return any(not torch.equal(before[n],p) for n,p in m.named_parameters())
def proj_nonzero(m):return any(bool((p!=0).any()) for n,p in m.named_parameters() if n.startswith("projections"))
def rms_unit():
 rows=[]
 for value in (0.,1e-8,1e-6,1e-4):
  for variant in ("OLD","STABLE","STABLE_DETACHED_SCALE"):
   r=torch.full((1,2,4,4),value,requires_grad=True);h=torch.ones_like(r);eps=1e-6
   rr=r.square().mean().sqrt().clamp_min(eps) if variant=="OLD" else torch.sqrt(r.square().mean()+eps**2)
   scale=(.1*torch.sqrt(h.square().mean()+eps**2)/rr).clamp(max=1.)
   if variant=="STABLE_DETACHED_SCALE":scale=scale.detach()
   out=r*scale;out.sum().backward();g=r.grad
   rows.append({"variant":variant,"residual_value":value,"forward_finite":bool(torch.isfinite(out).all()),"backward_finite":bool(torch.isfinite(g).all()),"gradient_mean":float(g.mean()) if torch.isfinite(g).all() else None,"gradient_nonzero":bool((g!=0).any()) if torch.isfinite(g).all() else False})
 return rows
def setup(a):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);d=torch.device("cuda:"+a.device)
 tok=CLIPTokenizer.from_pretrained(a.base_model,subfolder="tokenizer");te=CLIPTextModel.from_pretrained(a.base_model,subfolder="text_encoder").to(d);vae=AutoencoderKL.from_pretrained(a.base_model,subfolder="vae").to(d);unet=UNet2DConditionModel.from_pretrained(a.base_model,subfolder="unet").to(d);sched=DDPMScheduler.from_pretrained(a.base_model,subfolder="scheduler")
 for m in (te,vae,unet):m.requires_grad_(False).eval()
 return d,vae,unet,sched,prompt(tok,te,"a photo of a sks defect",d)
def forward(sample,adapter,o,timestep=None,return_pred=False):
 d,vae,unet,sched,emb=o;image,mask,normal,iid=sample;image=image[None].to(d);mask=mask[None].to(d);normal=normal[None].to(d)
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):z=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor;mz=vae.encode(normal*(mask<.5)).latent_dist.sample()*vae.config.scaling_factor
 noise=torch.randn_like(z);t=torch.tensor([timestep],device=d) if timestep is not None else torch.randint(0,sched.config.num_train_timesteps,(1,),device=d);inp=torch.cat((sched.add_noise(z,noise,t),F.interpolate(mask,size=z.shape[-2:],mode="nearest"),mz),1);adapter.set_condition(mask,normal,t)
 with torch.autocast("cuda",dtype=torch.float16):pred=unet(inp,t,encoder_hidden_states=emb).sample;loss=F.mse_loss(pred.float(),noise.float())
 meta={"instance_id":iid,"timestep":int(t),"mask_area":float(mask.mean())}
 return (loss,meta,pred.detach()) if return_pred else (loss,meta)
def grads_finite(m):
 rows={n:{"exists":p.grad is not None,"finite":bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,"l2":float(p.grad.float().square().sum().sqrt()) if p.grad is not None and torch.isfinite(p.grad).all() else None} for n,p in m.named_parameters()}
 return all(x["exists"] and x["finite"] for x in rows.values()),rows
def residual_probe(adapter,sample,o):
 records=[];orig=adapter._residual
 def measured(self,h,i):
  delta=orig(h,i);mask=F.interpolate(self._mask.float(),size=h.shape[-2:],mode="nearest");m=mask.expand(-1,h.shape[1],-1,-1);records.append({"block":i+2,"finite":bool(torch.isfinite(delta).all()),"residual_rms":float(delta.float().square().mean().sqrt()),"masked_residual_rms":float((delta.float().square()*m).sum().div(m.sum().clamp_min(1)).sqrt()),"global_residual_hidden_ratio":float(delta.float().square().mean().sqrt()/h.float().square().mean().sqrt().clamp_min(1e-12)),"resized_mask_pixels":int(mask.sum())});return delta
 adapter._residual=types.MethodType(measured,adapter)
 try:
  torch.manual_seed(20260927)
  with torch.no_grad():forward(sample,adapter,o,500)
 finally:adapter._residual=orig
 return records
def one_step(cls,a,o):
 data=Data(a.lowdata_root,cls);sample=data[len(data)//2];m=RepairedFrozenResidualAdapter().to(o[0]);m.attach(o[2]);before=param_snapshot(m);pre=residual_probe(m,sample,o);loss,meta=forward(sample,m,o);loss.backward();ok,g=grads_finite(m);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-2,foreach=False)
 if ok:opt.step()
 post=residual_probe(m,sample,o);row={"sample":meta,"before_projection_exact_zero":not proj_nonzero(m) if not ok else all(bool((before[n]==0).all()) for n in before if n.startswith("projections")),"before_residual_exact_zero":all(x["residual_rms"]==0 for x in pre),"loss":float(loss),"gradients":g,"all_trainable_gradients_finite":ok,"projection_has_nonzero_finite_gradient":any(x["l2"] and x["l2"]>0 for n,x in g.items() if n.startswith("projections")),"optimizer_step_executed":ok,"parameters_changed":changed(before,m) if ok else False,"parameters_finite":finite_params(m),"projection_nonzero":proj_nonzero(m),"post_residual":post,"post_residual_finite_nonzero":bool(post) and all(x["finite"] and x["residual_rms"]>0 for x in post)};row["passed"]=all(row[k] for k in ("before_projection_exact_zero","before_residual_exact_zero","all_trainable_gradients_finite","projection_has_nonzero_finite_gradient","optimizer_step_executed","parameters_changed","parameters_finite","projection_nonzero","post_residual_finite_nonzero"));m.detach();return row
def stability(a,o):
 data=Data(a.lowdata_root,"short");loader=torch.utils.data.DataLoader(data,batch_size=1,shuffle=True,num_workers=0);it=iter(loader);m=RepairedFrozenResidualAdapter().to(o[0]);m.attach(o[2]);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-2,foreach=False);scaler=torch.amp.GradScaler("cuda");initial=param_snapshot(m);attempt=success=skip=actual=0;rows=[];scale=[];probe_source=data[len(data)//2];probes={"0":residual_probe(m,probe_source,o)}
 while success<50 and attempt<256:
  opt.zero_grad(set_to_none=True);loss_sum=0.;metas=[]
  for _ in range(4):
   try:b=next(it)
   except StopIteration:it=iter(loader);b=next(it)
   sample=(b[0][0],b[1][0],b[2][0],b[3][0]);loss,meta=forward(sample,m,o);scaler.scale(loss/4).backward();loss_sum+=float(loss)/4;metas.append(meta)
  attempt+=1;scaler.unscale_(opt);finite,g=grads_finite(m);before=param_snapshot(m);old=float(scaler.get_scale())
  if finite:scaler.step(opt)
  scaler.update();new=float(scaler.get_scale());did=finite and changed(before,m);actual+=int(did);success+=int(did);skip+=int(not did);scale.append({"attempt":attempt,"before":old,"after":new,"finite":finite,"actual_parameter_update":did})
  if did:
   p0=float(m.projections[0].weight.float().square().sum().sqrt());p1=float(m.projections[1].weight.float().square().sum().sqrt());bias=float(sum(p.bias.float().square().sum() for p in m.projections).sqrt());rows.append({"successful_update":success,"attempt":attempt,"loss":loss_sum,"gradients_finite":finite,"scale":new,"projection0_l2":p0,"projection1_l2":p1,"projection_bias_l2":bias,"mask_encoder_delta":float(sum((p-initial[n]).float().square().sum() for n,p in m.named_parameters() if n.startswith("mask_encoder")).sqrt()),"context_encoder_delta":float(sum((p-initial[n]).float().square().sum() for n,p in m.named_parameters() if n.startswith("context_encoder")).sqrt()),"time_mlp_delta":float(sum((p-initial[n]).float().square().sum() for n,p in m.named_parameters() if n.startswith("time_mlp")).sqrt()),"selected_sources":metas})
   if success in (1,5,10,25,50):probes[str(success)]=residual_probe(m,probe_source,o)
  if new<=0 or skip>=32 and success==0:break
 torch.manual_seed(20260927);m.detach();_,_,off=forward(probe_source,m,o,500,True);torch.manual_seed(20260927);m.attach(o[2]);_,_,on=forward(probe_source,m,o,500,True);on_off_different=not torch.equal(on,off)
 step0_exact_zero=all(x["finite"] and x["residual_rms"]==0 for x in probes["0"]);post_update_stable=all(x["finite"] and x["residual_rms"]>0 and x["global_residual_hidden_ratio"]<=.1001 for key,z in probes.items() if key!="0" for x in z)
 stable=success==50 and actual==50 and finite_params(m) and proj_nonzero(m) and new>0 and on_off_different and step0_exact_zero and post_update_stable
 m.detach();return {"status":"REPAIR_STABLE" if stable else "REPAIR_UNSTABLE","attempted_updates":attempt,"successful_updates":success,"skipped_updates":skip,"actual_parameter_update_count":actual,"successful_equals_actual":success==actual,"final_scale":new,"parameters_finite":finite_params(m),"projection_nonzero":proj_nonzero(m),"step0_residual_exact_zero":step0_exact_zero,"post_update_residual_stable":post_update_stable,"adapter_on_off_output_identical":not on_off_different,"adapter_on_off_max_abs_delta":float((on-off).abs().max()),"scale_trajectory":scale,"successful_records":rows,"fixed_probe_source":probe_source[3],"fixed_probe_records":probes}
def main():
 p=argparse.ArgumentParser();p.add_argument("--base_model",type=Path,required=True);p.add_argument("--lowdata_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 unit=rms_unit();save(a.output/"rms_bound_gradient_unit_test.json",{"eps":1e-6,"max_ratio":.1,"rows":unit});stable_ok=all(x["forward_finite"] and x["backward_finite"] for x in unit if x["variant"]=="STABLE");selected="STABLE" if stable_ok else "STABLE_DETACHED_SCALE";save(a.output/"numerical_fix_selection.json",{"selected_variant":selected,"reason":"minimal stable RMS repair passed all tensor-level probes" if stable_ok else "stable RMS remained nonfinite; detached scale fallback required","architecture_changed":False,"parameter_count_changed":False,"max_rms_ratio_changed":False,"max_rms_ratio":.10})
 if selected!="STABLE":save(a.output/"stage9e_gate.json",{"status":"NUMERICAL_REPAIR_FAILED","STAGE9F_DETECTOR_PILOT_AUTHORIZED":False});return
 o=setup(a);audit={c:one_step(c,a,o) for c in CLASSES};save(a.output/"repaired_one_step_audit.json",audit)
 if not all(x["passed"] for x in audit.values()):save(a.output/"stage9e_gate.json",{"status":"NUMERICAL_REPAIR_FAILED","STAGE9F_DETECTOR_PILOT_AUTHORIZED":False});return
 st=stability(a,o);save(a.output/"repaired_50step_stability.json",st);save(a.output/"optimizer_step_accounting_audit.json",{"protocol":"finite gradients required; success counted only when parameters change","diagnostic":{k:st[k] for k in ("attempted_updates","successful_updates","skipped_updates","actual_parameter_update_count","successful_equals_actual","final_scale")}});print(json.dumps({"one_step":{c:audit[c]["passed"] for c in CLASSES},"stability":st["status"]},indent=2))
if __name__=="__main__":main()
