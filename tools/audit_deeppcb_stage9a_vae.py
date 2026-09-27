"""Stage9A VAE encode/decode bottleneck audit on frozen low-shot train data."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
import lpips
from scipy.ndimage import binary_dilation,binary_erosion

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def gradient(image):
 x=np.asarray(image.convert("L"),dtype=np.float32)/255.0;p=np.pad(x,1,mode="edge")
 gx=-p[:-2,:-2]+p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-p[2:,:-2]+p[2:,2:]
 gy=-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:]
 return np.hypot(gx,gy)
def cosine(x,y):
 x=np.asarray(x,float).reshape(-1);y=np.asarray(y,float).reshape(-1);den=np.linalg.norm(x)*np.linalg.norm(y)
 return float(np.dot(x,y)/den) if den else float(np.allclose(x,y))
def edge_cos(real,reconstructed):return cosine(gradient(real),gradient(reconstructed))
def boundary_gradient_cosine(real,reconstructed,mask):
 band=binary_dilation(mask,iterations=2)^binary_erosion(mask,iterations=2)
 return cosine(gradient(real)[band],gradient(reconstructed)[band])
def stat(v):
 a=np.asarray(v,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std()),"n":len(a)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--lowdata_root",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev=torch.device("cuda:"+a.device);vae=AutoencoderKL.from_pretrained(a.model,subfolder="vae").to(dev).eval();vae.requires_grad_(False);net=lpips.LPIPS(net="alex",verbose=False).to(dev).eval();tf=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()]);rows=[]
 for cls in CLASSES:
  masks={x.stem:x for x in (a.lowdata_root/"DeepPCB/ground_truth"/cls).iterdir() if x.suffix.lower() in (".jpg",".png",".jpeg")}
  for path in sorted((a.lowdata_root/"DeepPCB/test"/cls).iterdir()):
   if path.stem not in masks:continue
   pil=Image.open(path).convert("RGB").resize((512,512));mask=np.asarray(Image.open(masks[path.stem]).convert("L").resize((512,512),Image.Resampling.NEAREST))>127;x=tf(pil)[None].to(dev);xn=x*2-1
   with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):z=vae.encode(xn).latent_dist.mode();rec=vae.decode(z).sample
   rec=((rec[0].float().cpu().clamp(-1,1)+1)/2);rp=transforms.ToPILImage()(rec);mt=torch.from_numpy(mask)[None,None].to(dev);den=mt.sum().clamp_min(1);diff=(rec.to(dev)-x[0]).float();mse=float((diff.square()*mt).sum()/(den*3));mae=float((diff.abs()*mt).sum()/(den*3));ys,xs=np.where(mask);box=(xs.min(),ys.min(),xs.max()+1,ys.max()+1);realc=pil.crop(box).resize((256,256));recc=rp.crop(box).resize((256,256));tt=transforms.ToTensor();lp=float(net((tt(realc)[None].to(dev)*2-1),(tt(recc)[None].to(dev)*2-1)).item());inside=np.asarray(pil,float)[mask].mean();outside=np.asarray(pil,float)[~mask].mean();rows.append({"instance_id":path.stem,"class_name":cls,"mask_area":int(mask.sum()),"area_fraction":float(mask.mean()),"aspect_ratio":float((box[2]-box[0])/max(box[3]-box[1],1)),"contrast":float(abs(inside-outside)/255),"masked_mae":mae,"masked_mse":mse,"mask_crop_lpips":lp,"edge_cosine":edge_cos(realc,recc),"boundary_gradient_cosine":boundary_gradient_cosine(pil,rp,mask)})
 areas=np.asarray([x["area_fraction"] for x in rows]);q=np.quantile(areas,[1/3,2/3]);contrasts={c:np.median([x["contrast"] for x in rows if x["class_name"]==c]) for c in CLASSES}
 for x in rows:x["area_tertile"]=("small" if x["area_fraction"]<=q[0] else "medium" if x["area_fraction"]<=q[1] else "large");x["contrast_half"]="low" if x["contrast"]<=contrasts[x["class_name"]] else "high"
 fields=("masked_mae","masked_mse","mask_crop_lpips","edge_cosine","boundary_gradient_cosine");groups=defaultdict(list)
 for x in rows:groups[(x["class_name"],x["area_tertile"],x["contrast_half"])].append(x)
 summary={"overall":{f:stat([x[f] for x in rows]) for f in fields},"per_class":{c:{f:stat([x[f] for x in rows if x["class_name"]==c]) for f in fields} for c in CLASSES},"strata":{"|".join(k):{f:stat([x[f] for x in v]) for f in fields} for k,v in groups.items()},"no_weighted_score":True,"official_validation_use_count":0}
 (a.output/"vae_bottleneck_raw.json").write_text(json.dumps({"records":rows},indent=2));(a.output/"vae_bottleneck_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps({"instances":len(rows)},indent=2))
if __name__=="__main__":main()
