"""Generate the frozen Stage10B source/seed matrix without selection."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import torch
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import StableDiffusionInpaintPipeline_dynamic
from magic_ddim import DDIMScheduler
from generate_deeppcb_stage9e import call,resize,resize_mask

CLASSES=("short","pinhole");SEEDS=(42,2026,3407)
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for n in ("gate","registry","base_model","model_root","scheduler","output","manifest"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();gate=load(a.gate/"stage10b_gate.json");audit=load(a.gate/"formal_training_audit.json");trained=audit.get("status")=="TMVB_FORMAL_TRAINING_COMPLETE" and all(audit["classes"][c]["successful_steps"]==audit["classes"][c]["target_successful_steps"]==2000 and audit["classes"][c]["checkpoint_saved"] for c in CLASSES);assert trained and not gate.get("STAGE10C_DETECTOR_PILOT_AUTHORIZED",False);rows=[x for x in load(a.registry)["instances"] if x["class_name"] in CLASSES];a.output.mkdir(parents=True,exist_ok=True);records=[];dev="cuda:"+a.device
 for cls in CLASSES:
  pipe=StableDiffusionInpaintPipeline_dynamic.from_pretrained(a.base_model,torch_dtype=torch.float16);pipe.unet=UNet2DConditionModel.from_pretrained(a.model_root/cls/"unet",torch_dtype=torch.float16);pipe.scheduler=DDIMScheduler.from_pretrained(a.scheduler);pipe.to(dev);pipe.set_progress_bar_config(disable=True)
  try:
   for x in [r for r in rows if r["class_name"]==cls]:
    normal=resize(Image.open(x["paired_normal_image"]).convert("RGB"));mask=resize_mask(Image.open(x["instance_mask_path"]).convert("L"))
    for seed in SEEDS:
     dst=a.output/cls/f"{x['instance_id']}_s{seed}.jpg";dst.parent.mkdir(parents=True,exist_ok=True);call(pipe,normal,mask,seed).save(dst,quality=95);records.append({"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":cls,"generation_seed":seed,"image_path":str(dst.resolve()),"real_path":x["defect_image"],"normal_path":x["paired_normal_image"],"mask_path":x["instance_mask_path"],"image_sha256":sha(dst)})
  finally:del pipe;torch.cuda.empty_cache()
 expected=len(rows)*len(SEEDS);assert len(rows)==258 and len(records)==expected and len({(x["instance_id"],x["generation_seed"]) for x in records})==expected;a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps({"status":"COMPLETE","record_count":len(records),"source_count":len(rows),"classes":list(CLASSES),"generation_seeds":list(SEEDS),"exact_source_seed_matrix":True,"selector_used":False,"official_validation_use_count":0,"records":records},indent=2),encoding="utf-8");print(json.dumps({"records":len(records),"sources":len(rows),"seeds":list(SEEDS)},indent=2))
if __name__=="__main__":main()
