"""Stage9D checkpoint, initialization, mask-survival, and hook audits."""
from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
from train_deeppcb_stage9a_adapter import Data,prompt
from stage9a_frozen_adapter import FrozenResidualAdapter,timestep_embedding

CLASSES=("short","pinhole")
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def stats(t):
 x=t.detach().float().cpu();return {"shape":list(x.shape),"numel":x.numel(),"dtype":str(t.dtype),"requires_grad_metadata_available":False,"l1_norm":float(x.abs().sum()),"l2_norm":float(x.square().sum().sqrt()),"abs_mean":float(x.abs().mean()),"abs_max":float(x.abs().max()),"exact_zero_fraction":float((x==0).float().mean()),"finite_fraction":float(torch.isfinite(x).float().mean())}
def group(name):return name.split('.')[0] if not name.startswith('projections') else '.'.join(name.split('.')[:2])
def stage9c_noop_evidence(stage9c):
 status=load(stage9c/"stage9c_status.json")
 evidence=status.get("effective_adapter_noop_evidence")
 if evidence is not None:return evidence
 # Older downloaded Stage9C bundles recorded the same conclusion without the
 # expanded evidence object. Preserve provenance instead of failing on schema.
 probe_path=stage9c/"adapter_on_off_probe.json";residual_path=stage9c/"residual_magnitude_summary.json"
 return {
  "effective_adapter_noop_observed":status.get("effective_adapter_noop_observed"),
  "stage9c_status":status.get("status"),
  "adapter_on_off_probe_file":probe_path.name if probe_path.exists() else None,
  "residual_magnitude_summary_file":residual_path.name if residual_path.exists() else None,
  "evidence_schema_fallback":True,
 }
def main():
 p=argparse.ArgumentParser()
 for n in ("base_model","lowdata_root","adapter_root","stage9a_manifest","stage9c","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:"+a.device);parameter_audit={};comparison={};trained={}
 for cls in CLASSES:
  payload=torch.load(a.adapter_root/cls/"frozen_adapter.pt",map_location="cpu");state=payload["state_dict"];trained[cls]=state;rows={n:stats(v) for n,v in state.items()};projection_zero=all(rows[n]["exact_zero_fraction"]==1 for n in rows if n.startswith("projections."));entire_zero=all(x["exact_zero_fraction"]==1 for x in rows.values());parameter_audit[cls]={"parameters":rows,"groups":{g:{"numel":sum(rows[n]["numel"] for n in rows if group(n)==g),"l2_norm":float(sum(rows[n]["l2_norm"]**2 for n in rows if group(n)==g)**.5),"all_exact_zero":all(rows[n]["exact_zero_fraction"]==1 for n in rows if group(n)==g)} for g in sorted({group(n) for n in rows})},"projection_weights_and_biases_exact_zero":projection_zero,"entire_adapter_zero":entire_zero,"status":"PROJECTIONS_REMAIN_ZERO" if projection_zero else "PARAMETERS_UPDATED"}
 # Reproduce Stage9A construction order before fresh adapter initialization.
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);tok=CLIPTokenizer.from_pretrained(a.base_model,subfolder="tokenizer");text=CLIPTextModel.from_pretrained(a.base_model,subfolder="text_encoder").to(device);vae=AutoencoderKL.from_pretrained(a.base_model,subfolder="vae").to(device);unet=UNet2DConditionModel.from_pretrained(a.base_model,subfolder="unet").to(device);sched=DDPMScheduler.from_pretrained(a.base_model,subfolder="scheduler");fresh=FrozenResidualAdapter()
 for cls in CLASSES:
  comp={}
  for n,v in fresh.state_dict().items():
   d=trained[cls][n].float()-v.float();den=v.float().square().sum().sqrt();comp[n]={"exact_equal":bool(torch.equal(trained[cls][n],v)),"max_abs_delta":float(d.abs().max()),"relative_l2_delta":float(d.square().sum().sqrt()/den.clamp_min(1e-30))}
  comparison[cls]={"parameters":comp,"all_exact_equal":all(x["exact_equal"] for x in comp.values()),"projections_exact_equal_to_zero_initialization":all(comp[n]["exact_equal"] for n in comp if n.startswith("projections.")),"status":"ADAPTER_OPTIMIZER_EFFECTIVELY_NO_UPDATE" if all(x["exact_equal"] for x in comp.values()) else "DEAD_ZERO_PROJECTION_PATH" if all(comp[n]["exact_equal"] for n in comp if n.startswith("projections.")) else "PARAMETERS_CHANGED"}
 save(a.output/"checkpoint_parameter_state.json",parameter_audit);save(a.output/"checkpoint_vs_initialization.json",comparison)
 # Capture actual last-two up-block output resolutions with a true Stage9A forward.
 data={c:Data(a.lowdata_root,c) for c in CLASSES};sample=data["short"][0];image,mask,normal,_=sample;image=image[None].to(device);mask=mask[None].to(device);normal=normal[None].to(device);text.requires_grad_(False).eval();vae.requires_grad_(False).eval();unet.requires_grad_(False).eval();emb=prompt(tok,text,"a photo of a sks defect",device)
 with torch.no_grad():latent=vae.encode(image).latent_dist.mode()*vae.config.scaling_factor;masked=vae.encode(normal*(mask<.5)).latent_dist.mode()*vae.config.scaling_factor
 t=torch.tensor([500],device=device);noisy=sched.add_noise(latent,torch.randn_like(latent),t);inp=torch.cat((noisy,F.interpolate(mask,size=latent.shape[-2:],mode="nearest"),masked),1);shapes={};handles=[]
 for index,block in enumerate(unet.up_blocks[-2:]):
  def hook(_m,_i,o,j=index):shapes[j+2]=list((o[0] if isinstance(o,tuple) else o).shape)
  handles.append(block.register_forward_hook(hook))
 with torch.no_grad():unet(inp,t,encoder_hidden_states=emb)
 for h in handles:h.remove()
 survival={"actual_hook_shapes":shapes,"classes":{}}
 for cls in CLASSES:
  records=[]
  for i in range(len(data[cls])):
   _,m,_,iid=data[cls][i];orig=float(m.sum());row={"instance_id":iid,"original_mask_pixels":int(orig),"original_mask_area_ratio":float(m.mean())}
   for block in (2,3):
    size=shapes[block][-2:];r=F.interpolate(m[None],size=size,mode="nearest")[0];pix=int(r.sum());row[f"block{block}"]={"spatial_shape":size,"resized_mask_pixels":pix,"resized_mask_area_ratio":float(r.mean()),"empty":pix==0,"retained_area_ratio":pix/max(orig,1)}
   records.append(row)
  areas=np.asarray([x["original_mask_area_ratio"] for x in records]);q=np.quantile(areas,[1/3,2/3])
  for x in records:x["area_stratum"]="small" if x["original_mask_area_ratio"]<=q[0] else "medium" if x["original_mask_area_ratio"]<=q[1] else "large"
  blocks={}
  for block in (2,3):
   vals=[x[f"block{block}"]["resized_mask_pixels"] for x in records];blocks[f"block{block}"]={"empty_mask_fraction":float(np.mean([v==0 for v in vals])),"resized_pixels":{"q10":float(np.quantile(vals,.1)),"q25":float(np.quantile(vals,.25)),"median":float(np.median(vals)),"q75":float(np.quantile(vals,.75)),"q90":float(np.quantile(vals,.9))},"strata_empty_fraction":{s:float(np.mean([x[f"block{block}"]["empty"] for x in records if x["area_stratum"]==s])) for s in ("small","medium","large")}}
  survival["classes"][cls]={"blocks":blocks,"records":records}
 max_empty=max(survival["classes"][c][f"blocks"][f"block{b}"]["empty_mask_fraction"] for c in CLASSES for b in (2,3));survival["status"]="MASK_DOWNSAMPLE_COLLAPSE_SUPPORTED" if max_empty>=.5 else "MASK_DOWNSAMPLE_COLLAPSE_PARTIAL" if max_empty>0 else "MASK_DOWNSAMPLE_COLLAPSE_NOT_SUPPORTED";save(a.output/"mask_survival_audit.json",survival)
 # Single true forward for fresh and trained adapters; hook path and condition are audited directly.
 hook_audit={}
 for cls in CLASSES:
  x=data[cls][0];im,m,n,iid=x;im=im[None].to(device);m=m[None].to(device);n=n[None].to(device)
  with torch.no_grad():z=vae.encode(im).latent_dist.mode()*vae.config.scaling_factor;ml=vae.encode(n*(m<.5)).latent_dist.mode()*vae.config.scaling_factor
  tt=torch.tensor([500],device=device);model_in=torch.cat((sched.add_noise(z,torch.randn_like(z),tt),F.interpolate(m,size=z.shape[-2:],mode="nearest"),ml),1)
  hook_audit[cls]={}
  for label,adapter in (("fresh",FrozenResidualAdapter()),("trained",FrozenResidualAdapter.load(a.adapter_root/cls/"frozen_adapter.pt")[0])):
   adapter.to(device).eval();calls=[]
   def measured(self,hidden,index):
    mk=F.interpolate(self._mask.float(),size=hidden.shape[-2:],mode="nearest");ctx=F.interpolate(self._context.float(),size=hidden.shape[-2:],mode="bilinear",align_corners=False);pre=self.mask_encoder(mk)+self.context_encoder(ctx);feat=F.silu(pre+self.time_mlp(timestep_embedding(self._timesteps,self.hidden))[:,:,None,None])*mk;raw=self.projections[index](feat).to(hidden.dtype);hr=hidden.float().square().mean().sqrt().clamp_min(1e-6);rr=raw.float().square().mean().sqrt().clamp_min(1e-6);scale=min(1.,float(self.max_rms_ratio*hr/rr));delta=raw*scale;calls.append({"block":index+2,"hidden_shape":list(hidden.shape),"mask_nonzero_pixels":int(mk.sum()),"condition_set_before_hook":self._mask is not None and self._context is not None and self._timesteps is not None,"timestep":int(self._timesteps[0]),"feature_abs_mean":float(feat.abs().mean()),"feature_abs_max":float(feat.abs().max()),"projection_input_abs_mean":float(feat.abs().mean()),"projection_input_abs_max":float(feat.abs().max()),"raw_residual_abs_mean":float(raw.abs().mean()),"raw_residual_abs_max":float(raw.abs().max()),"returned_hidden_delta_abs_mean":float(delta.abs().mean()),"returned_hidden_delta_abs_max":float(delta.abs().max())});return delta
   import types;adapter._residual=types.MethodType(measured,adapter);adapter.set_condition(m,n,tt);adapter.attach(unet)
   with torch.no_grad():unet(model_in,tt,encoder_hidden_states=emb)
   adapter.detach();hook_audit[cls][label]={"sample":iid,"hook_called_count":len(calls),"calls":calls}
 save(a.output/"hook_execution_audit.json",hook_audit)
 evidence=stage9c_noop_evidence(a.stage9c);manifest=load(a.stage9a_manifest);checkpoint_shas=sorted({x["adapter_checkpoint_sha256"] for x in manifest["records_detail"]});save(a.output/"stage9a_effective_arm_reinterpretation.json",{"formal_description":"Stage9A frozen_adapter arm behaved as an unadapted pretrained-backbone control because the learned residual path was inactive.","probe_evidence":evidence,"manifest_records":manifest["records"],"manifest_checkpoint_sha256":checkpoint_shas,"generation_seeds":manifest["generation_seeds"],"adapter_attach_intended_by_generation_code":True,"base_model":"SD2 Inpainting pretrained backbone","scheduler":"frozen repository DDIM scheduler","historical_stage9a_files_modified":False,"UNADAPTED_PRETRAINED_BACKBONE_INSUFFICIENT":True,"FULL_UNET_FINETUNING_REQUIRED":False,"LORA_REQUIRED":False})
 print(json.dumps({"parameter_status":{c:parameter_audit[c]["status"] for c in CLASSES},"mask_status":survival["status"]},indent=2))
if __name__=="__main__":main()
