"""Generate exact-paired repaired-adapter and pretrained controls for Stage9E."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO)) if str(REPO) not in sys.path else None
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import StableDiffusionInpaintPipeline_dynamic
from magic_ddim import DDIMScheduler
from stage9e_frozen_adapter import RepairedFrozenResidualAdapter
CLASSES=("short","pinhole");SEEDS=(42,2026,3407);resize=transforms.Compose([transforms.Resize(512),transforms.CenterCrop(512)]);resize_mask=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.NEAREST),transforms.CenterCrop(512)])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(pipe,normal,mask,seed):
 torch.manual_seed(seed);return pipe(prompt=["a photo of a sks defect"],image=normal,mask_image=mask.convert("RGB"),num_inference_steps=50,guidance_scale=7.5,mdap_prior_image=None,mdap_strength=0.,mdap_schedule="cosine",mdap_end_fraction=.7,rda_enabled=False,rda_path=None,rda_reference_image=None,rda_reference_mask=None,carf_enabled=False,carf_path=None,msdf_enabled=False,msdf_path=None,msdf_reference_image=None,msdf_reference_mask=None,msdf_ablation=None,anomaly_strength=0.,anomaly_stop_step=999999,eta_mask_stop_step=999999,use_random_mask=False,eta=0.,eta_mask=0.,guidance_scale_inside=None,guidance_scale_outside=None,gsi_use_schedule=False,gsi_schedule="linear",guidance_scale_inside_min=None,guidance_scale_inside_max=None,gsi_power=2.,gsi_exp_k=3.,gsi_sigmoid_k=8.,gsi_sample_per_step=False).images[0]
def main():
 p=argparse.ArgumentParser();
 for n in ("registry","base_model","adapter_root","scheduler","sd2_manifest","output","manifest"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[x for x in json.loads(a.registry.read_text())["instances"] if x["class_name"] in CLASSES];sd={(x["instance_id"],int(x["generation_seed"])):x for x in json.loads(a.sd2_manifest.read_text())["records"]};out=[];dev="cuda:"+a.device
 for cls in CLASSES:
  pipe=StableDiffusionInpaintPipeline_dynamic.from_pretrained(a.base_model,torch_dtype=torch.float16);pipe.scheduler=DDIMScheduler.from_pretrained(a.scheduler);pipe.to(dev);adapter,meta=RepairedFrozenResidualAdapter.load(a.adapter_root/cls/"frozen_adapter.pt",dev);assert meta["successful_updates"]==meta["actual_parameter_update_count"]==2000;adapter.to(dev).eval();static={}
  def before(_m,args):
   t=args[1].reshape(-1) if torch.is_tensor(args[1]) else torch.tensor([args[1]],device=dev);b=args[0].shape[0];adapter.set_condition(static["mask"].expand(b,-1,-1,-1),static["normal"].expand(b,-1,-1,-1),t.expand(b))
  pre=pipe.unet.register_forward_pre_hook(before)
  try:
   for x in [z for z in rows if z["class_name"]==cls]:
    normal=resize(Image.open(x["paired_normal_image"]).convert("RGB"));mask=resize_mask(Image.open(x["instance_mask_path"]).convert("L"));static.update(normal=torch.from_numpy(np.asarray(normal).copy()).permute(2,0,1)[None].to(dev,dtype=torch.float32)/127.5-1,mask=torch.from_numpy((np.asarray(mask)>127).astype("float32"))[None,None].to(dev))
    for seed in SEEDS:
     key=(x["instance_id"],seed);assert key in sd;off=a.output/"pretrained"/cls/f"{x['instance_id']}_s{seed}.jpg";on=a.output/"repaired"/cls/f"{x['instance_id']}_s{seed}.jpg";off.parent.mkdir(parents=True,exist_ok=True);on.parent.mkdir(parents=True,exist_ok=True)
     adapter.detach();call(pipe,normal,mask,seed).save(off,quality=95);adapter.attach(pipe.unet);call(pipe,normal,mask,seed).save(on,quality=95);out.append({"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":cls,"generation_seed":seed,"real":x["defect_image"],"normal":x["paired_normal_image"],"mask":x["instance_mask_path"],"class_finetuned_sd2":sd[key]["image_path"],"pretrained_no_adaptation":str(off.resolve()),"repaired_frozen_adapter":str(on.resolve()),"class_finetuned_sha256":sha(sd[key]["image_path"]),"pretrained_sha256":sha(off),"repaired_sha256":sha(on),"adapter_checkpoint_sha256":sha(a.adapter_root/cls/"frozen_adapter.pt")})
  finally:pre.remove();adapter.detach();del pipe,adapter;torch.cuda.empty_cache()
 expected=len(rows)*len(SEEDS);assert len(out)==expected and len({(x["instance_id"],x["generation_seed"]) for x in out})==expected;a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps({"records":out,"record_count":len(out),"source_count":len(rows),"classes":list(CLASSES),"generation_seeds":list(SEEDS),"exact_pairing":True,"same_source_mask_normal_prompt_scheduler_steps_guidance":True,"official_validation_use_count":0},indent=2));print(json.dumps({"records":len(out),"sources":len(rows)},indent=2))
if __name__=="__main__":main()
