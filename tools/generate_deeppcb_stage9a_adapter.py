"""Generate the fixed Stage9A short/pinhole source x seed matrix."""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path

# Executing ``python tools/<script>.py`` otherwise puts ``tools`` before the
# repository root and resolves the pip-installed Diffusers package.  Stage9A
# must use this repository's audited custom inpainting pipeline and scheduler.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
 sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np,torch
from PIL import Image
from torchvision import transforms
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import StableDiffusionInpaintPipeline_dynamic
from magic_ddim import DDIMScheduler
from stage9a_frozen_adapter import FrozenResidualAdapter

CLASSES=("short","pinhole");SEEDS=(42,2026,3407)
image_resize_center_crop=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512)])
mask_resize_center_crop=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.NEAREST),transforms.CenterCrop(512)])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();
 for n in ("registry","base_model","adapter_root","scheduler","output","manifest"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);device="cuda:"+a.device;rows=json.loads(a.registry.read_text())["instances"];rows=[x for x in rows if x["class_name"] in CLASSES];out=[]
 for cls in CLASSES:
  pipe=StableDiffusionInpaintPipeline_dynamic.from_pretrained(a.base_model,torch_dtype=torch.float16);pipe.scheduler=DDIMScheduler.from_pretrained(a.scheduler);pipe.to(device);adapter,meta=FrozenResidualAdapter.load(a.adapter_root/cls/"frozen_adapter.pt",device);adapter.to(device).eval();adapter.attach(pipe.unet);static={}
  def before(_module,args):
   timestep=args[1];t=timestep.reshape(-1) if torch.is_tensor(timestep) else torch.tensor([timestep],device=device);batch=args[0].shape[0];t=t.expand(batch);adapter.set_condition(static["mask"].expand(batch,-1,-1,-1),static["normal"].expand(batch,-1,-1,-1),t)
  handle=pipe.unet.register_forward_pre_hook(before)
  try:
   for x in [r for r in rows if r["class_name"]==cls]:
    normal=image_resize_center_crop(Image.open(x["paired_normal_image"]).convert("RGB"));mask=mask_resize_center_crop(Image.open(x["instance_mask_path"]).convert("L"));normal_t=torch.from_numpy(np.asarray(normal).copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/127.5-1;mask_t=torch.from_numpy((np.asarray(mask)>127).astype("float32"))[None,None].to(device);static.update(normal=normal_t,mask=mask_t)
    for seed in SEEDS:
     torch.manual_seed(seed)
     image=pipe(prompt=["a photo of a sks defect"],image=normal,mask_image=mask.convert("RGB"),num_inference_steps=50,guidance_scale=7.5,mdap_prior_image=None,mdap_strength=0.0,mdap_schedule="cosine",mdap_end_fraction=.7,rda_enabled=False,rda_path=None,rda_reference_image=None,rda_reference_mask=None,carf_enabled=False,carf_path=None,msdf_enabled=False,msdf_path=None,msdf_reference_image=None,msdf_reference_mask=None,msdf_ablation=None,anomaly_strength=0.0,anomaly_stop_step=999999,eta_mask_stop_step=999999,use_random_mask=False,eta=0.0,eta_mask=0.0,guidance_scale_inside=None,guidance_scale_outside=None,gsi_use_schedule=False,gsi_schedule="linear",guidance_scale_inside_min=None,guidance_scale_inside_max=None,gsi_power=2.0,gsi_exp_k=3.0,gsi_sigmoid_k=8.0,gsi_sample_per_step=False).images[0]
     path=a.output/cls/f"{x['instance_id']}_s{seed}.jpg";path.parent.mkdir(parents=True,exist_ok=True);image.save(path,quality=95);out.append({"arm":"frozen_adapter","instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":cls,"generation_seed":seed,"image_path":str(path.resolve()),"image_sha256":sha(path),"real_path":x["defect_image"],"normal_path":x["paired_normal_image"],"mask_path":x["instance_mask_path"],"adapter_checkpoint":str((a.adapter_root/cls/"frozen_adapter.pt").resolve()),"adapter_checkpoint_sha256":sha(a.adapter_root/cls/"frozen_adapter.pt")})
  finally:handle.remove();adapter.detach();del pipe,adapter;torch.cuda.empty_cache()
 audit={"classes":list(CLASSES),"generation_seeds":list(SEEDS),"records":len(out),"sources":len(rows),"same_source_mask_normal_prompt_steps_guidance_scheduler":True,"defect_reference_use_count":0,"official_validation_use_count":0,"records_detail":out};a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps(audit,indent=2));print(json.dumps({"records":len(out),"sources":len(rows)},indent=2))
if __name__=="__main__":main()
