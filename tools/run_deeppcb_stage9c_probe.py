"""Stage9C fixed 60-source ON/OFF probe with exact residual instrumentation."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,types
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) in sys.path:sys.path.remove(str(REPO))
sys.path.insert(0,str(REPO))
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import StableDiffusionInpaintPipeline_dynamic
from magic_ddim import DDIMScheduler
from stage9a_frozen_adapter import FrozenResidualAdapter,timestep_embedding

CLASSES=("short","pinhole");SEED=42
resize=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512)])
resize_mask=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.NEAREST),transforms.CenterCrop(512)])
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def choose(rows):
 out=[]
 for cls in CLASSES:
  group=sorted([x for x in rows if x["class_name"]==cls],key=lambda x:(x.get("area_fraction",0),x["instance_id"]));bins=np.array_split(group,3)
  for label,items in zip(("small","medium","large"),bins):
   # Deterministic quantile-spread selection; depends only on frozen mask area.
   ids=np.linspace(0,len(items)-1,10,dtype=int)
   for i in ids:
    x=dict(items[int(i)]);x["area_stratum"]=label;out.append(x)
 return out
def rms(x,mask=None):
 y=x.float()
 if mask is None:return float(y.square().mean().sqrt())
 m=mask.expand(-1,y.shape[1],-1,-1);return float((y.square()*m).sum().div(m.sum().clamp_min(1)).sqrt())
def generate(pipe,normal,mask):
 torch.manual_seed(SEED)
 return pipe(prompt=["a photo of a sks defect"],image=normal,mask_image=mask.convert("RGB"),num_inference_steps=50,guidance_scale=7.5,mdap_prior_image=None,mdap_strength=0.0,mdap_schedule="cosine",mdap_end_fraction=.7,rda_enabled=False,rda_path=None,rda_reference_image=None,rda_reference_mask=None,carf_enabled=False,carf_path=None,msdf_enabled=False,msdf_path=None,msdf_reference_image=None,msdf_reference_mask=None,msdf_ablation=None,anomaly_strength=0.0,anomaly_stop_step=999999,eta_mask_stop_step=999999,use_random_mask=False,eta=0.0,eta_mask=0.0,guidance_scale_inside=None,guidance_scale_outside=None,gsi_use_schedule=False,gsi_schedule="linear",guidance_scale_inside_min=None,guidance_scale_inside_max=None,gsi_power=2.0,gsi_exp_k=3.0,gsi_sigmoid_k=8.0,gsi_sample_per_step=False).images[0]
def main():
 p=argparse.ArgumentParser()
 for n in ("registry","base_model","adapter_root","scheduler","stage9a_manifest","sd2_manifest","output","image_root"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.image_root.mkdir(parents=True,exist_ok=True);rows=choose(load(a.registry)["instances"]);old={(x["instance_id"],int(x["generation_seed"])):x for x in load(a.stage9a_manifest)["records_detail"]};sd2={(x["instance_id"],int(x["generation_seed"])):x for x in load(a.sd2_manifest)["records"]};manifest=[];raw=[];device="cuda:"+a.device
 save(a.output/"probe_source_manifest.json",{"selection_before_probe_outputs":True,"selection_variables":["class_name","mask_area_fraction","instance_id_tiebreak"],"generation_seed":SEED,"classes":list(CLASSES),"per_class":30,"strata":{"small":10,"medium":10,"large":10},"official_validation_use_count":0,"sources":rows})
 for cls in CLASSES:
  pipe=StableDiffusionInpaintPipeline_dynamic.from_pretrained(a.base_model,torch_dtype=torch.float16);pipe.scheduler=DDIMScheduler.from_pretrained(a.scheduler);pipe.to(device);adapter,meta=FrozenResidualAdapter.load(a.adapter_root/cls/"frozen_adapter.pt",device);adapter.to(device).eval();adapter.attach(pipe.unet);static={"source":None,"timestep":None};original=adapter._residual
  def measured(self,hidden,index):
   size=hidden.shape[-2:];mask=F.interpolate(self._mask.float(),size=size,mode="nearest");context=F.interpolate(self._context.float(),size=size,mode="bilinear",align_corners=False);feature=self.mask_encoder(mask)+self.context_encoder(context);time=self.time_mlp(timestep_embedding(self._timesteps,self.hidden)).to(feature.dtype);feature=F.silu(feature+time[:,:,None,None])*mask;residual=self.projections[index](feature).to(hidden.dtype);hr=hidden.float().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6);rr=residual.float().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6);scale=(self.max_rms_ratio*hr/rr).clamp(max=1.0).to(residual.dtype);bounded=residual*scale;local_mask=mask
   for bi in range(hidden.shape[0]):
    branch="conditional" if bi==hidden.shape[0]-1 else "unconditional";hm=local_mask[bi:bi+1];outside=1-hm;absval=bounded[bi].float().abs().reshape(-1);raw.append({"instance_id":static["source"]["instance_id"],"class_name":static["source"]["class_name"],"area_stratum":static["source"]["area_stratum"],"block":index+2,"branch":branch,"timestep":int(self._timesteps[bi]),"mask_area_pixels":int(hm.sum()),"mask_area_ratio":float(hm.mean()),"global_hidden_rms":rms(hidden[bi:bi+1]),"masked_hidden_rms":rms(hidden[bi:bi+1],hm),"raw_global_residual_rms":rms(residual[bi:bi+1]),"raw_masked_residual_rms":rms(residual[bi:bi+1],hm),"raw_outside_residual_rms":rms(residual[bi:bi+1],outside),"bounded_global_residual_rms":rms(bounded[bi:bi+1]),"bounded_masked_residual_rms":rms(bounded[bi:bi+1],hm),"bounded_outside_residual_rms":rms(bounded[bi:bi+1],outside),"global_residual_to_hidden_ratio":rms(bounded[bi:bi+1])/max(rms(hidden[bi:bi+1]),1e-12),"masked_residual_to_hidden_ratio":rms(bounded[bi:bi+1],hm)/max(rms(hidden[bi:bi+1],hm),1e-12),"raw_masked_residual_to_hidden_ratio":rms(residual[bi:bi+1],hm)/max(rms(hidden[bi:bi+1],hm),1e-12),"scale_multiplier":float(scale[bi]),"global_rms_bound_active":bool(scale[bi]<.999999),"residual_abs_p50":float(torch.quantile(absval,.5)),"residual_abs_p95":float(torch.quantile(absval,.95)),"residual_abs_p99":float(torch.quantile(absval,.99)),"residual_abs_max":float(absval.max())})
   return bounded
  adapter._residual=types.MethodType(measured,adapter)
  def before(_module,args):
   t=args[1];t=t.reshape(-1) if torch.is_tensor(t) else torch.tensor([t],device=device);batch=args[0].shape[0];adapter.set_condition(static["mask"].expand(batch,-1,-1,-1),static["normal"].expand(batch,-1,-1,-1),t.expand(batch))
  pre=pipe.unet.register_forward_pre_hook(before)
  try:
   for x in [z for z in rows if z["class_name"]==cls]:
    normal=resize(Image.open(x["paired_normal_image"]).convert("RGB"));mask=resize_mask(Image.open(x["instance_mask_path"]).convert("L"));static.update(source=x,normal=torch.from_numpy(np.asarray(normal).copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/127.5-1,mask=torch.from_numpy((np.asarray(mask)>127).astype("float32"))[None,None].to(device));off=a.image_root/cls/f"{x['instance_id']}_off_s42.jpg";on=a.image_root/cls/f"{x['instance_id']}_on_s42.jpg"
    adapter.detach();off_image=generate(pipe,normal,mask);off.parent.mkdir(parents=True,exist_ok=True);off_image.save(off,quality=95);adapter.attach(pipe.unet);on_image=generate(pipe,normal,mask);on_image.save(on,quality=95);key=(x["instance_id"],SEED)
    if key not in sd2 or key not in old:raise RuntimeError("STAGE9C_MISSING_MATCHED_CONTROL "+str(key))
    manifest.append({"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":cls,"area_stratum":x["area_stratum"],"generation_seed":SEED,"real":x["defect_image"],"normal":x["paired_normal_image"],"mask":x["instance_mask_path"],"pretrained_off":str(off.resolve()),"adapter_on":str(on.resolve()),"stage9a_adapter_on":old[key]["image_path"],"class_finetuned_sd2":sd2[key]["image_path"],"pretrained_off_sha256":sha(off),"adapter_on_sha256":sha(on)})
  finally:pre.remove();adapter.detach();del pipe,adapter;torch.cuda.empty_cache()
 save(a.output/"stage9c_probe_generated_manifest.json",{"same_source_mask_normal_prompt_seed_scheduler_steps_guidance":True,"records":manifest});save(a.output/"residual_magnitude_raw.json",{"architecture_unchanged":True,"checkpoint_unchanged":True,"projection_bias_means_residual_not_guaranteed_zero_outside_mask":True,"records":raw});print(json.dumps({"sources":len(manifest),"residual_records":len(raw)},indent=2))
if __name__=="__main__":main()
