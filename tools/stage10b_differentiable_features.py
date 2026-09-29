"""Differentiable equivalent of historical DetectInputExtractor."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

class DifferentiableDetectInputExtractor:
 def __init__(self,weights,device,imgsz=512):
  import torch
  from ultralytics import YOLO
  self.torch=torch;self.device=torch.device(f"cuda:{device}" if str(device).isdigit() else device);self.imgsz=int(imgsz);self.yolo=YOLO(str(weights));self.model=self.yolo.model.to(self.device).eval();self.model.fuse()
  for p in self.model.parameters():p.requires_grad_(False)
  detect=next((m for m in reversed(list(self.model.modules())) if m.__class__.__name__.lower()=="detect"),None)
  if detect is None:raise RuntimeError("STAGE10B_DETECT_HEAD_NOT_FOUND")
  self.features=None
  def capture(module,inputs):
   values=inputs[0] if inputs and isinstance(inputs[0],(list,tuple)) else inputs;self.features=list(values)
  self.handle=detect.register_forward_pre_hook(capture)
 def close(self):self.handle.remove()
 def preprocess_path(self,path):
  import cv2,torch
  from ultralytics.data.augment import LetterBox
  bgr=cv2.imread(str(path));
  if bgr is None:raise FileNotFoundError(path)
  h,w=bgr.shape[:2];rgb=LetterBox(new_shape=(self.imgsz,self.imgsz),auto=False,stride=32)(image=bgr)[...,::-1].copy();return torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device).float()/255.,w,h
 def encode_tensor(self,image,bbox_xyxy,image_width,image_height):
  import torch
  import torch.nn.functional as F
  if image.ndim==3:image=image.unsqueeze(0)
  image=image.to(self.device)
  if image.shape[-2:]!=(self.imgsz,self.imgsz):raise ValueError("Tensor input must already be letterboxed to imgsz")
  self.features=None;self.model(image)
  if not self.features or len(self.features)!=3:raise RuntimeError("STAGE10B_FEATURE_HOOK_FAILED")
  ratio=min(self.imgsz/float(image_width),self.imgsz/float(image_height));pad_x=(self.imgsz-image_width*ratio)/2;pad_y=(self.imgsz-image_height*ratio)/2;x0,y0,x1,y1=map(float,bbox_xyxy);box=(x0*ratio+pad_x,y0*ratio+pad_y,x1*ratio+pad_x,y1*ratio+pad_y);vectors=[]
  for fmap in self.features:
   _,_,mh,mw=fmap.shape;sx,sy=mw/self.imgsz,mh/self.imgsz;xa=max(0,min(mw-1,math.floor(box[0]*sx)));ya=max(0,min(mh-1,math.floor(box[1]*sy)));xb=max(xa+1,min(mw,math.ceil(box[2]*sx)));yb=max(ya+1,min(mh,math.ceil(box[3]*sy)));v=F.adaptive_avg_pool2d(fmap[:,:,ya:yb,xa:xb],(1,1)).flatten(1);vectors.append(F.normalize(v,p=2,dim=1,eps=1e-8))
  return torch.cat(vectors,dim=1)
 def encode_path(self,path,bbox_xyxy):
  image,w,h=self.preprocess_path(path);return self.encode_tensor(image,bbox_xyxy,w,h)
