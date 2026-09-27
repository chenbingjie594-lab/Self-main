"""Stage9D single-batch, one-step, and 50-successful-step diagnostics."""
from __future__ import annotations
import argparse,io,json,math,random,types
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
from train_deeppcb_stage9a_adapter import Data,prompt
from stage9a_frozen_adapter import FrozenResidualAdapter,timestep_embedding

CLASSES=("short","pinhole")
def json_safe(value,path="$",nonfinite=None):
 if nonfinite is None:nonfinite=[]
 if isinstance(value,dict):return {k:json_safe(v,f"{path}.{k}",nonfinite) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_safe(v,f"{path}[{i}]",nonfinite) for i,v in enumerate(value)]
 if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)):
  nonfinite.append({"path":path,"value":str(value)});return None
 return value
def save(p,x):
 nonfinite=[];safe=json_safe(x,nonfinite=nonfinite)
 if isinstance(safe,dict) and nonfinite:safe["serialization_audit"]={"nonfinite_values_replaced_with_null":len(nonfinite),"locations":nonfinite}
 Path(p).write_text(json.dumps(safe,indent=2,allow_nan=False))
def tensor_stats(x):
 if x is None:return {"exists":False,"finite":None,"l1":0.,"l2":0.,"abs_max":0.,"exact_zero_fraction":None}
 y=x.detach().float();return {"exists":True,"finite":bool(torch.isfinite(y).all()),"l1":float(y.abs().sum()),"l2":float(y.square().sum().sqrt()),"abs_max":float(y.abs().max()),"exact_zero_fraction":float((y==0).float().mean())}
def grouped(adapter,attribute="grad"):
 out={}
 for name,p in adapter.named_parameters():
  key="projections.0" if name.startswith("projections.0") else "projections.1" if name.startswith("projections.1") else name.split('.')[0];value=p.grad if attribute=="grad" else p.data;out.setdefault(key,[]).append(value)
 result={}
 for key,vals in out.items():
  if attribute=="grad" and any(v is None for v in vals):result[key]={"all_grad_exist":False,"parameters":[tensor_stats(v) for v in vals]}
  else:
   z=torch.cat([v.detach().float().reshape(-1) for v in vals]);result[key]={"all_grad_exist":True,**tensor_stats(z)}
 return result
def state_finite(state):return all(bool(torch.isfinite(v).all()) for v in state.values())
def tensor_exact_equal_nan(a,b):return bool(torch.all((a==b)|(torch.isnan(a)&torch.isnan(b))))
def setup(a):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);dev=torch.device("cuda:"+a.device);tok=CLIPTokenizer.from_pretrained(a.base_model,subfolder="tokenizer");text=CLIPTextModel.from_pretrained(a.base_model,subfolder="text_encoder").to(dev);vae=AutoencoderKL.from_pretrained(a.base_model,subfolder="vae").to(dev);unet=UNet2DConditionModel.from_pretrained(a.base_model,subfolder="unet").to(dev);sched=DDPMScheduler.from_pretrained(a.base_model,subfolder="scheduler")
 for m in (text,vae,unet):m.requires_grad_(False).eval()
 return dev,tok,text,vae,unet,sched,prompt(tok,text,"a photo of a sks defect",dev)
def forward(sample,adapter,objects,timestep=None):
 dev,tok,text,vae,unet,sched,emb=objects;image,mask,normal,iid=sample;image=image[None].to(dev);mask=mask[None].to(dev);normal=normal[None].to(dev)
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):latent=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor;masked=vae.encode(normal*(mask<.5)).latent_dist.sample()*vae.config.scaling_factor
 noise=torch.randn_like(latent);t=torch.tensor([timestep],device=dev) if timestep is not None else torch.randint(0,sched.config.num_train_timesteps,(1,),device=dev);inp=torch.cat((sched.add_noise(latent,noise,t),F.interpolate(mask,size=latent.shape[-2:],mode="nearest"),masked),1);adapter.set_condition(mask,normal,t)
 with torch.autocast("cuda",dtype=torch.float16):pred=unet(inp,t,encoder_hidden_states=emb).sample;loss=F.mse_loss(pred.float(),noise.float())
 return loss,{"instance_id":iid,"timestep":int(t),"mask_area":float(mask.mean())}
def residual_probe(adapter,sample,objects,step):
 records=[];original=adapter._residual
 def measured(self,hidden,index):
  size=hidden.shape[-2:];mask=F.interpolate(self._mask.float(),size=size,mode="nearest");context=F.interpolate(self._context.float(),size=size,mode="bilinear",align_corners=False);feature=F.silu(self.mask_encoder(mask)+self.context_encoder(context)+self.time_mlp(timestep_embedding(self._timesteps,self.hidden))[:,:,None,None])*mask;raw=self.projections[index](feature).to(hidden.dtype);hr=hidden.float().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6);rr=raw.float().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6);scale=(self.max_rms_ratio*hr/rr).clamp(max=1.0).to(raw.dtype);bounded=raw*scale;m=mask.expand(-1,raw.shape[1],-1,-1);records.append({"step":step,"block":index+2,"raw_residual_rms":float(raw.float().square().mean().sqrt()),"masked_residual_rms":float((raw.float().square()*m).sum().div(m.sum().clamp_min(1)).sqrt()),"bounded_residual_rms":float(bounded.float().square().mean().sqrt()),"hidden_delta_abs_max":float(bounded.abs().max()),"resized_mask_nonzero_pixels":int(mask.sum())});return bounded
 adapter._residual=types.MethodType(measured,adapter)
 try:
  torch.manual_seed(20260927)
  with torch.no_grad():forward(sample,adapter,objects,timestep=500)
 finally:adapter._residual=original
 return records
def one_class(cls,a,objects):
 data=Data(a.lowdata_root,cls);# Prefer a source surviving both actual injection resolutions (audited resolutions 64/128).
 eligible=[]
 for i in range(len(data)):
  _,m,_,_=data[i]
  if all(int(F.interpolate(m[None],size=(s,s),mode="nearest").sum())>0 for s in (64,128)):eligible.append(i)
 index=eligible[len(eligible)//2] if eligible else 0;sample=data[index];adapter=FrozenResidualAdapter().to(objects[0]);adapter.attach(objects[4]);loss,meta=forward(sample,adapter,objects);loss.backward();grad=grouped(adapter);before={n:p.detach().clone() for n,p in adapter.named_parameters()};opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-2,foreach=False);opt.step();after={n:p.detach().clone() for n,p in adapter.named_parameters()};changes={n:{"changed":not torch.equal(before[n],after[n]),"max_abs_delta":float((after[n]-before[n]).abs().max()),"relative_l2_delta":float((after[n]-before[n]).float().square().sum().sqrt()/before[n].float().square().sum().sqrt().clamp_min(1e-30))} for n in before};adapter.zero_grad(set_to_none=True);probe=[]
 buffer=io.BytesIO();torch.save(adapter.state_dict(),buffer);buffer.seek(0);roundtrip=torch.load(buffer,map_location=objects[0],weights_only=True);roundtrip_equal=all(tensor_exact_equal_nan(after[n],roundtrip[n]) for n in after);original=adapter._residual
 def measured(self,hidden,index):
  delta=original(hidden,index);probe.append({"block":index+2,"returned_delta_rms":float(delta.float().square().mean().sqrt()),"returned_delta_abs_max":float(delta.abs().max())});return delta
 adapter._residual=types.MethodType(measured,adapter);post_loss,_=forward(sample,adapter,objects,timestep=meta["timestep"]);adapter.detach();return {"sample":meta,"loss":float(loss),"gradient_groups":grad,"projection_gradient_status":"NONFINITE_GRADIENT" if any(v.get("finite") is False for v in grad.values()) else "GRADIENT_REACHES_PROJECTION" if all(grad[k].get("l2",0)>0 for k in ("projections.0","projections.1")) else "PROJECTION_GRADIENT_ZERO","parameter_changes":changes,"post_step_parameters_finite":state_finite(after),"projection_weight_changed":any(changes[n]["changed"] for n in changes if n.startswith("projections") and n.endswith("weight")),"projection_bias_changed":any(changes[n]["changed"] for n in changes if n.startswith("projections") and n.endswith("bias")),"encoder_changed":any(changes[n]["changed"] for n in changes if n.startswith(("mask_encoder","context_encoder"))),"time_mlp_changed":any(changes[n]["changed"] for n in changes if n.startswith("time_mlp")),"state_dict_roundtrip_exact_equal":roundtrip_equal,"state_dict_roundtrip_finite":state_finite(roundtrip),"post_step_probe":probe,"post_step_output_still_exact_zero":all(x["returned_delta_rms"]==0 for x in probe)}
def trajectory(a,objects):
 data=Data(a.lowdata_root,"short");adapter=FrozenResidualAdapter().to(objects[0]);adapter.attach(objects[4]);opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-2,foreach=False);scaler=torch.amp.GradScaler("cuda");it=iter(torch.utils.data.DataLoader(data,batch_size=1,shuffle=True,num_workers=0));success=attempt=0;rows=[];probe_steps={0,1,5,10,25,50};probe_sample=data[len(data)//2];probe_records=residual_probe(adapter,probe_sample,objects,0);initial={n:p.detach().clone() for n,p in adapter.named_parameters()}
 max_attempts=256
 while success<50 and attempt<max_attempts:
  opt.zero_grad(set_to_none=True);micro=[];loss_value=0.
  for _ in range(4):
   try:batch=next(it);sample=(batch[0][0],batch[1][0],batch[2][0],batch[3][0])
   except StopIteration:it=iter(torch.utils.data.DataLoader(data,batch_size=1,shuffle=True,num_workers=0));batch=next(it);sample=(batch[0][0],batch[1][0],batch[2][0],batch[3][0])
   loss,meta=forward(sample,adapter,objects);loss_value+=float(loss)/4;micro.append(meta);scaler.scale(loss/4).backward()
  before_scale=scaler.get_scale();scaler.unscale_(opt);grads=grouped(adapter);finite=all(v.get("finite",True) is not False for v in grads.values());before_step={n:p.detach().clone() for n,p in adapter.named_parameters()};scaler.step(opt);scaler.update();after_scale=scaler.get_scale();changed=any(not torch.equal(before_step[n],p) for n,p in adapter.named_parameters());skipped=(not finite) or (not changed);attempt+=1
  if skipped:
   rows.append({"attempt":attempt,"successful_step":success,"skip":True,"scale_before":before_scale,"scale_after":after_scale,"gradient_finite":finite,"parameters_changed":changed,"sample":meta})
   if after_scale==0:break
   continue
  success+=1;params=grouped(adapter,attribute="data");delta={k:float(torch.cat([(p-initial[n]).float().reshape(-1) for n,p in adapter.named_parameters() if n.startswith(k)]).square().sum().sqrt()) for k in ("projections.0","projections.1","mask_encoder","context_encoder","time_mlp")};row={"attempt":attempt,"successful_step":success,"skip":False,"loss":loss_value,"scale_before":before_scale,"scale_after":after_scale,"gradient_finite":finite,"gradient_groups":grads,"parameter_groups":params,"parameter_delta_l2_from_step0":delta,"gradient_accumulation":4,"micro_batches":micro}
  if success in probe_steps:probe_records.extend(residual_probe(adapter,probe_sample,objects,success))
  rows.append(row)
 adapter.detach();skips=sum(x["skip"] for x in rows);scale_zero=bool(rows and rows[-1]["scale_after"]==0);return {"class_name":"short","formal_checkpoint":False,"checkpoint_saved":False,"target_successful_updates":50,"successful_updates":success,"attempted_updates":len(rows),"max_attempts":max_attempts,"stopped_on_zero_scale":scale_zero,"skipped_updates":skips,"skip_ratio":skips/len(rows),"fixed_probe_source":probe_sample[3],"fixed_probe_records":probe_records,"records":rows,"amp_skip_status":"AMP_SCALE_COLLAPSED_TO_ZERO" if scale_zero else "ALL_UPDATES_SKIPPED_NONFINITE" if success==0 and skips else "AMP_PARTIAL_PROGRESS" if skips else "NO_AMP_SKIPS"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--base_model",type=Path,required=True);p.add_argument("--lowdata_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);objects=setup(a);audit={c:one_class(c,a,objects) for c in CLASSES};save(a.output/"single_batch_gradient_audit.json",{c:{"sample":x["sample"],"loss":x["loss"],"gradient_groups":x["gradient_groups"],"status":x["projection_gradient_status"]} for c,x in audit.items()});save(a.output/"one_step_update_audit.json",audit);traj=trajectory(a,objects);save(a.output/"diagnostic_50step_trajectory.json",traj);save(a.output/"amp_skip_audit.json",{"status":traj["amp_skip_status"],"diagnostic":{k:traj[k] for k in ("attempted_updates","successful_updates","skipped_updates","skip_ratio")},"historical":{"successful":2000,"skipped":166,"attempted":2166,"skip_ratio":166/2166},"formal_training_changed":False});print(json.dumps({"gradient":{c:audit[c]["projection_gradient_status"] for c in CLASSES},"one_step_zero":{c:audit[c]["post_step_output_still_exact_zero"] for c in CLASSES},"trajectory_skips":traj["skipped_updates"]},indent=2))
if __name__=="__main__":main()
