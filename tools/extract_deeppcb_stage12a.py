"""Extract frozen Stage3R tight-box features and Stage11A gradients for historical sets."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch
from PIL import Image
from dwbg_feature_extraction import DetectInputExtractor
from extract_deeppcb_stage11a_gradients import selected_head,prepare_loss_config,crop_protocol

def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser()
 for n in ("manifest","stage3r_runs","stage11a","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();rows=load(a.manifest)["records"];a.output.mkdir(parents=True,exist_ok=True);missing=[x["image_path"] for x in rows if not Path(x["image_path"]).is_file()]
 save(a.output/"historical_input_availability_audit.json",{"records":len(rows),"missing_count":len(missing),"missing_fraction":len(missing)/max(1,len(rows)),"missing_paths":missing[:200]})
 if missing:raise RuntimeError(f"STAGE12A_MISSING_HISTORICAL_IMAGES_{len(missing)}")
 from ultralytics import YOLO
 dev="cuda:"+a.device;models={};selected={};extractors={};before={};checkpoints={f:a.stage3r_runs/f"fold_{f}/weights/best.pt" for f in range(3)}
 for f in range(3):
  extractors[f]=DetectInputExtractor(checkpoints[f],a.device,512);m=YOLO(str(checkpoints[f])).model.to(dev);prepare_loss_config(m);m.train()
  for q in m.parameters():q.requires_grad_(False)
  for bn in m.modules():
   if isinstance(bn,torch.nn.modules.batchnorm._BatchNorm):bn.eval()
  _,s=selected_head(m)
  for _,q in s:q.requires_grad_(True)
  models[f]=m;selected[f]=s;before[f]=[q.detach().cpu().clone() for _,q in s]
 total=sum(q.numel() for _,q in selected[0]);rng=np.random.default_rng(2026);bucket=torch.from_numpy(rng.integers(0,1024,total,dtype=np.int64)).to(dev);sign=torch.from_numpy(rng.choice(np.asarray([-1.,1.],np.float32),total)).to(dev)
 feats=[];grads=[];meta=[]
 try:
  for i,x in enumerate(rows):
   f=int(x["fold"]);path=x["image_path"];im=Image.open(path).convert("RGB");w,h=im.size;box=x["bbox_xyxy"];v=extractors[f].encode(path,box,w,h);v=v/max(np.linalg.norm(v),1e-12)
   m=models[f];m.zero_grad(set_to_none=True);crop,b2=crop_protocol(box,w,h);patch=np.asarray(im.crop(crop).resize((512,512),Image.Resampling.BILINEAR));image=torch.from_numpy(patch.copy()).permute(2,0,1).unsqueeze(0).to(dev).float()/255.;x0,y0,x1,y1=b2;target={"img":image,"batch_idx":torch.zeros(1,device=dev,dtype=torch.long),"cls":torch.tensor([[x["class_id"]]],device=dev,dtype=torch.float32),"bboxes":torch.tensor([[(x0+x1)/1024,(y0+y1)/1024,(x1-x0)/512,(y1-y0)/512]],device=dev,dtype=torch.float32)};loss,_=m(target);loss.sum().backward();flat=torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).detach().reshape(-1).float() for _,q in selected[f]]);proj=torch.zeros(1024,device=dev);proj.scatter_add_(0,bucket,flat*sign);g=(proj/proj.norm().clamp_min(1e-30)).cpu().numpy();feats.append(v);grads.append(g);meta.append({k:x[k] for k in ("family_id","arm","sample_id","source_instance_id","physical_pair_id","class_name","class_id","fold")}|{"image_sha256":sha(path),"gradient_norm":float(flat.norm()),"gradient_finite":bool(torch.isfinite(flat).all())})
   m.zero_grad(set_to_none=True)
 finally:
  for e in extractors.values():e.close()
 unchanged=all(all(torch.equal(q.detach().cpu(),b) for (_,q),b in zip(selected[f],before[f])) for f in range(3));np.savez_compressed(a.output/"historical_extra_set_embeddings.npz",features=np.asarray(feats,np.float32),gradients=np.asarray(grads,np.float32));save(a.output/"historical_embedding_manifest.json",{"records":meta});save(a.output/"historical_extraction_audit.json",{"status":"PASS" if unchanged and all(x["gradient_finite"] for x in meta) else "FAIL","record_count":len(meta),"feature_dim":int(len(feats[0])),"gradient_dim":1024,"teacher_checkpoints":{str(f):sha(checkpoints[f]) for f in range(3)},"teacher_parameters_unchanged":unchanged,"feature_protocol":"tight bbox Stage9E-R","gradient_protocol":"Stage11A scale4/min64 target-only loss CountSketch seed2026","optimizer_step_count":0,"official_validation_use_count":0})
 if not unchanged:raise RuntimeError("STAGE12A_TEACHER_MUTATED")
 print(json.dumps({"status":"STAGE12A_EXTRACTION_COMPLETE","records":len(meta)},indent=2))
if __name__=="__main__":main()
