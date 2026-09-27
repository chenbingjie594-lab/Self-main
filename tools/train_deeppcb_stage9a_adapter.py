"""Train the Stage9A adapter while keeping SD2 UNet/VAE/text encoder frozen."""
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL,DDPMScheduler,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
from stage9a_frozen_adapter import FrozenResidualAdapter

SUFFIX={".jpg",".jpeg",".png",".bmp"}
class Data(Dataset):
 def __init__(self,root,cls,res=512):
  self.rows=[];ims={p.stem:p for p in (root/"DeepPCB/test"/cls).iterdir() if p.suffix.lower() in SUFFIX};masks={p.stem:p for p in (root/"DeepPCB/ground_truth"/cls).iterdir() if p.suffix.lower() in SUFFIX};norm={p.stem:p for p in (root/"DeepPCB/paired_normal"/cls).iterdir() if p.suffix.lower() in SUFFIX}
  for key in sorted(set(ims)&set(masks)&set(norm)):self.rows.append((ims[key],masks[key],norm[key]))
  if not self.rows:raise RuntimeError("STAGE9A_EMPTY_TRAIN_SET")
  self.rgb=transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);self.mask=transforms.Compose([transforms.Resize(res,interpolation=transforms.InterpolationMode.NEAREST),transforms.CenterCrop(res),transforms.ToTensor()])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  image,mask,normal=self.rows[i];return self.rgb(Image.open(image).convert("RGB")),(self.mask(Image.open(mask).convert("L"))>=.5).float(),self.rgb(Image.open(normal).convert("RGB")),image.stem
def prompt(tok,enc,text,device):
 ids=tok(text,padding="max_length",max_length=tok.model_max_length,truncation=True,return_tensors="pt").input_ids.to(device)
 with torch.no_grad():return enc(ids)[0]
def main():
 p=argparse.ArgumentParser();p.add_argument("--model_path",type=Path,required=True);p.add_argument("--lowdata_root",type=Path,required=True);p.add_argument("--class_name",choices=("short","pinhole"),required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=2000);p.add_argument("--accum",type=int,default=4);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
 # The parent/class directory may be created by deployment or orchestration.
 # Refuse only a completed/partial formal artifact, and do so before spending
 # GPU time rather than after all 2,000 optimizer updates.
 formal_artifacts=(a.output/"frozen_adapter.pt",a.output/"training_audit.json",a.output/"training_history.csv")
 if any(path.exists() for path in formal_artifacts):
  raise FileExistsError("STAGE9A_OUTPUT_ARTIFACT_EXISTS: "+", ".join(str(x) for x in formal_artifacts if x.exists()))
 a.output.mkdir(parents=True,exist_ok=True)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device("cuda");dtype=torch.float16
 tok=CLIPTokenizer.from_pretrained(a.model_path,subfolder="tokenizer");text=CLIPTextModel.from_pretrained(a.model_path,subfolder="text_encoder").to(device);vae=AutoencoderKL.from_pretrained(a.model_path,subfolder="vae").to(device);unet=UNet2DConditionModel.from_pretrained(a.model_path,subfolder="unet").to(device);sched=DDPMScheduler.from_pretrained(a.model_path,subfolder="scheduler")
 for module in (text,vae,unet):module.requires_grad_(False).eval()
 adapter=FrozenResidualAdapter().to(device);adapter.attach(unet);adapter.train();trainable=sum(x.numel() for x in adapter.parameters());backbone=sum(x.numel() for x in unet.parameters());opt=torch.optim.AdamW(adapter.parameters(),lr=a.lr,betas=(.9,.999),weight_decay=1e-2,foreach=False);scaler=torch.amp.GradScaler("cuda");loader=DataLoader(Data(a.lowdata_root,a.class_name),batch_size=1,shuffle=True,num_workers=0);it=iter(loader);emb=prompt(tok,text,"a photo of a sks defect",device);history=[];updates=skips=micro=0;started=time.time();torch.cuda.reset_peak_memory_stats()
 while updates<a.steps:
  try:image,mask,normal,ids=next(it)
  except StopIteration:it=iter(loader);image,mask,normal,ids=next(it)
  image=image.to(device);mask=mask.to(device);normal=normal.to(device);masked=normal*(mask<.5)
  with torch.no_grad(),torch.autocast("cuda",dtype=dtype):
   latent=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor;masked_latent=vae.encode(masked).latent_dist.sample()*vae.config.scaling_factor
  noise=torch.randn_like(latent);t=torch.randint(0,sched.config.num_train_timesteps,(1,),device=device);noisy=sched.add_noise(latent,noise,t);latent_mask=torch.nn.functional.interpolate(mask,size=latent.shape[-2:],mode="nearest");model_input=torch.cat((noisy,latent_mask,masked_latent),1);adapter.set_condition(mask,normal,t)
  with torch.autocast("cuda",dtype=dtype):pred=unet(model_input,t,encoder_hidden_states=emb).sample;loss=torch.nn.functional.mse_loss(pred.float(),noise.float())/a.accum
  scaler.scale(loss).backward();micro+=1
  if micro%a.accum:continue
  old=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
  if scaler.get_scale()<old:skips+=1;continue
  updates+=1;history.append({"step":updates,"loss":float(loss.detach()*a.accum),"source_id":ids[0]})
 arch={"channels":[640,320],"hidden":64,"max_rms_ratio":.10};meta={"status":"COMPLETE","class_name":a.class_name,"successful_updates":updates,"nonfinite_or_amp_skips":skips,"seed":a.seed,"batch_size":1,"gradient_accumulation":a.accum,"learning_rate":a.lr,"optimizer":"AdamW","mixed_precision":"fp16","trainable_components":["FrozenResidualAdapter"],"unet_trainable_parameters":0,"vae_trainable_parameters":0,"text_encoder_trainable_parameters":0,"adapter_parameters":trainable,"unet_parameters":backbone,"adapter_to_unet_ratio":trainable/backbone,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"wall_seconds":time.time()-started,"final_loss":history[-1]["loss"],"architecture":arch,"forbidden_inputs":{"defect_reference":False,"detector_feature":False,"morphology_template":False,"source_identity":False,"teacher_score":False}}
 adapter.save(a.output/"frozen_adapter.pt",meta);(a.output/"training_audit.json").write_text(json.dumps(meta,indent=2));
 with (a.output/"training_history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
