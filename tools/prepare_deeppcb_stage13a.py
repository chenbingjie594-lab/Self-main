"""Build the frozen Stage13A counterfactual-pair-transport set and all pre-training audits."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion,gaussian_filter

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert("RGB"))
def gray(x):return cv2.cvtColor(x.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
def edge(x):
 g=cv2.GaussianBlur(gray(x),(0,0),1);return cv2.magnitude(cv2.Sobel(g,cv2.CV_32F,1,0),cv2.Sobel(g,cv2.CV_32F,0,1))
def cos(a,b):
 a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel();d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else 1.0
def ncc(a,b):
 a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel();a-=a.mean();b-=b.mean();d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else 0.0
def ssim(a,b):
 """Dependency-free global SSIM, averaged over channels (frozen Stage7B fallback)."""
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if a.ndim==2:a=a[...,None];b=b[...,None]
 c1=(.01*255)**2;c2=(.03*255)**2;values=[]
 for channel in range(a.shape[2]):
  x,y=a[...,channel],b[...,channel];mx,my=x.mean(),y.mean();vx,vy=x.var(),y.var();cov=((x-mx)*(y-my)).mean();values.append(((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)))
 return float(np.mean(values))
def crop_reflect(a,cx,cy,side):
 side=int(side);left=side//2;right=side-left;p=cv2.copyMakeBorder(a,left,right,left,right,cv2.BORDER_REFLECT_101);x=int(round(cx));y=int(round(cy));return p[y:y+side,x:x+side]
def isolate(row,by_pair):
 d=rgb(row["defect_image"]).copy();n=rgb(row["paired_normal_image"]);tm=np.asarray(Image.open(row["instance_mask_path"]).convert("L"))>0
 for z in by_pair[row["pair_id"]]:
  if z["instance_id"]!=row["instance_id"]:
   m=np.asarray(Image.open(z["instance_mask_path"]).convert("L"))>0;d[m]=n[m]
 return d,n,tm
def phase(a,b,valid):
 aa=a.copy();bb=b.copy();aa[~valid]=0;bb[~valid]=0;s,_=cv2.phaseCorrelate(aa.astype(np.float32),bb.astype(np.float32));return [float(s[0]),float(s[1])]
def shift_image(a,dx,dy):return cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
def summ(v):return {"median":float(np.median(v)),"q90":float(np.quantile(v,.9)),"q95":float(np.quantile(v,.95)),"max":float(np.max(v))}
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","registry","uniform_selection","lowdata_split","dataset_root","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);save(a.output/"stage13a_protocol.json",{**cfg,"donor_source":"frozen Stage11B UniformRepeat120","recipient_source":"paired-normal images of frozen 100-pair low-shot training pool","translation_only":True,"learned_matching":False,"generator_training":False,"official_validation_use_before_final_evaluation":0});reg=load(a.registry)["instances"];sel=load(a.uniform_selection);donor_ids=[x["instance_id"] for x in sel["records"]];idx={x["instance_id"]:x for x in reg};by_pair=defaultdict(list)
 for x in reg:by_pair[x["pair_id"]].append(x)
 allowed=set(load(a.lowdata_split)["selected_image_ids"]);assert len(donor_ids)==120 and len(set(donor_ids))==120 and all(x in idx for x in donor_ids) and Counter(idx[x]["class_name"] for x in donor_ids)==Counter({c:20 for c in CLASSES})
 pair_normal={q:sorted({str(x["paired_normal_image"]) for x in rows})[0] for q,rows in by_pair.items() if q in allowed};assert set(pair_normal)==allowed and len(allowed)==100
 try:
  import torch,lpips
  net=lpips.LPIPS(net="alex").to("cuda:"+a.device).eval()
  def lp(a0,b0):
   def t(x):return torch.from_numpy(x.copy()).permute(2,0,1)[None].float().to("cuda:"+a.device)/127.5-1
   with torch.no_grad():return float(net(t(a0),t(b0)).item())
 except Exception as e:raise RuntimeError("STAGE13A_LPIPS_UNAVAILABLE") from e
 donors=[];registration=[];recon=[]
 for iid in donor_ids:
  r=idx[iid];d,n,m=isolate(r,by_pair);valid=~binary_dilation(m,iterations=2);gs=phase(gray(d),gray(n),valid);es=phase(edge(d),edge(n),valid);corrected=False
  if max(map(abs,gs))>1 or max(map(abs,es))>1:
   candidates=[]
   for sign in (-1,1):
    nn=shift_image(n,sign*round(gs[0]),sign*round(gs[1]));q=phase(gray(d),gray(nn),valid);candidates.append((max(map(abs,q)),nn,q,sign))
   _,n,_,_=min(candidates,key=lambda z:z[0]);corrected=True
  gs2=phase(gray(d),gray(n),valid);es2=phase(edge(d),edge(n),valid);gd,gn=gray(d),gray(n);fill=float(np.mean(np.r_[gd[valid],gn[valid]]));gd2,gn2=gd.copy(),gn.copy();gd2[~valid]=fill;gn2[~valid]=fill;registration.append({"instance_id":iid,"initial_gray_shift":gs,"initial_edge_shift":es,"integer_translation_corrected":corrected,"residual_gray_shift":gs2,"residual_edge_shift":es2,"outside_ssim":ssim(gd2,gn2),"outside_ncc":ncc(gd[valid],gn[valid]),"outside_mae":float(np.mean(np.abs(d.astype(float)[valid]-n.astype(float)[valid])))})
  if max(map(abs,gs2))>1 or max(map(abs,es2))>1:save(a.output/"paired_registration_audit.json",{"status":"DONOR_PAIR_ALIGNMENT_FAIL","records":registration});raise RuntimeError("DONOR_PAIR_ALIGNMENT_FAIL")
  soft=gaussian_filter(binary_dilation(m,iterations=2).astype(np.float32),1);soft/=max(float(soft.max()),1e-12);res=soft[...,None]*(d.astype(np.float32)-n.astype(np.float32));rec=np.clip(n.astype(float)+res,0,255).astype(np.uint8);x0,y0,x1,y1=map(int,r["bbox_xyxy"]);pad=8;sl=np.s_[max(0,y0-pad):min(d.shape[0],y1+pad),max(0,x0-pad):min(d.shape[1],x1+pad)];rr,dd=rec[sl],d[sl];band=binary_dilation(m,iterations=4)^binary_erosion(m,iterations=2)
  z={"instance_id":iid,"lpips":lp(rr,dd),"ssim":ssim(rr,dd),"edge_cosine":cos(edge(rr),edge(dd)),"boundary_gradient_cosine":cos(edge(rec)[band],edge(d)[band]),"target_mask_mae":float(np.mean(np.abs(rec.astype(float)[m]-d.astype(float)[m]))),"clipping_fraction":float(np.mean((n.astype(float)+res<=0)|(n.astype(float)+res>=255))),"finite":bool(np.isfinite(res).all())};recon.append(z);donors.append({"row":r,"d":d,"n":n,"mask":m,"soft":soft,"res":res})
 save(a.output/"paired_registration_audit.json",{"status":"PASS","policy":"outside dilated target; global integer translation only","records":registration});rs={k:summ([x[k] for x in recon]) for k in ("lpips","ssim","edge_cosine","boundary_gradient_cosine","target_mask_mae")};rok=rs["ssim"]["median"]>=.98 and rs["edge_cosine"]["median"]>=.95 and rs["lpips"]["median"]<=.03 and all(x["finite"] for x in recon);save(a.output/"donor_reconstruction_audit.json",{"status":"PASS" if rok else "RESIDUAL_EXTRACTION_NOT_VALID","summary":rs,"records":recon});
 if not rok:raise RuntimeError("RESIDUAL_EXTRACTION_NOT_VALID")
 candidates={}
 for z in donors:
  r=z["row"];x0,y0,x1,y1=map(int,r["bbox_xyxy"]);bw,bh=x1-x0,y1-y0;side=max(64,int(math.ceil(3*max(bw,bh))));cx=(x0+x1)/2;cy=(y0+y1)/2;dt=crop_reflect(z["n"],cx,cy,side);de=edge(dt);dg=gray(dt);cs=[]
  for q,npth in sorted(pair_normal.items()):
   if q==r["pair_id"]:continue
   im=rgb(npth);H,W=im.shape[:2];gim=gray(im);eim=edge(im);left=side//2;right=side-left;gp=cv2.copyMakeBorder(gim,left,right,left,right,cv2.BORDER_REFLECT_101);ep=cv2.copyMakeBorder(eim,left,right,left,right,cv2.BORDER_REFLECT_101);pm=cv2.matchTemplate(gp,dg.astype(np.float32),cv2.TM_CCOEFF_NORMED);em=cv2.matchTemplate(ep,de.astype(np.float32),cv2.TM_CCOEFF_NORMED);ys=np.arange(0,H-bh+1,4,dtype=int);xs=np.arange(0,W-bw+1,4,dtype=int);yy,xx=np.meshgrid(ys,xs,indexing="ij");cy=np.rint(yy+bh/2).astype(int);cx=np.rint(xx+bw/2).astype(int);pv=pm[cy,cx];ev=em[cy,cx];ok=pv<cfg["pixel_ncc_reject"]
   best=None
   if np.any(ok):
    flat=np.flatnonzero(ok);order=np.lexsort((xx.ravel()[flat],yy.ravel()[flat],pv.ravel()[flat],-ev.ravel()[flat]));k=flat[order[0]];y0i=int(yy.ravel()[k]);x0i=int(xx.ravel()[k]);best=(( -float(ev.ravel()[k]),float(pv.ravel()[k]),q,y0i,x0i),{"recipient_pair_id":q,"recipient_image":npth,"x":x0i,"y":y0i,"edge_ncc":float(ev.ravel()[k]),"pixel_ncc":float(pv.ravel()[k]),"context_side":side})
   if best:cs.append(best[1])
  cs.sort(key=lambda v:(-v["edge_ncc"],v["pixel_ncc"],v["recipient_pair_id"],v["y"],v["x"]));candidates[r["instance_id"]]=cs
  if len(cs)<10:save(a.output/"recipient_matching_audit.json",{"status":"RECIPIENT_MATCHING_INSUFFICIENT","candidate_counts":{k:len(v) for k,v in candidates.items()}});raise RuntimeError("RECIPIENT_MATCHING_INSUFFICIENT")
 reuse=Counter();assigned=[]
 for z in sorted(donors,key=lambda q:(q["row"]["class_name"],q["row"]["instance_id"])):
  iid=z["row"]["instance_id"];choice=next((x for x in candidates[iid] if reuse[x["recipient_pair_id"]]<2),None)
  if choice is None:raise RuntimeError("RECIPIENT_MATCHING_INSUFFICIENT")
  reuse[choice["recipient_pair_id"]]+=1;assigned.append((z,choice))
 records=[];identity=[];context=[];real_seam=[];cpt_seam=[];out_img=a.dataset_root/"cpt120/images";out_lab=a.dataset_root/"cpt120/labels";out_img.mkdir(parents=True,exist_ok=True);out_lab.mkdir(parents=True,exist_ok=True)
 def seam(im,m):
  outer=binary_dilation(m,iterations=4)^binary_dilation(m,iterations=2);inner=binary_dilation(m,iterations=2)^binary_erosion(m,iterations=2);g=edge(im);lap=np.abs(cv2.Laplacian(gray(im),cv2.CV_32F));return {"boundary_gradient_jump":float(abs(g[inner].mean()-g[outer].mean())),"boundary_rgb_jump":float(abs(im[inner].astype(float).mean()-im[outer].astype(float).mean())),"laplacian_energy":float(lap[inner|outer].mean())}
 for z,ch in assigned:
  r=z["row"];recip=rgb(ch["recipient_image"]);x0,y0,x1,y1=map(int,r["bbox_xyxy"]);bw,bh=x1-x0,y1-y0;dx=ch["x"]-x0;dy=ch["y"]-y0;side=ch["context_side"];dc=crop_reflect(z["n"],(x0+x1)/2,(y0+y1)/2,side);rc=crop_reflect(recip,ch["x"]+bw/2,ch["y"]+bh/2,side);ds=(np.subtract(*np.percentile(dc,[75,25],axis=(0,1)))/1.349);rsd=(np.subtract(*np.percentile(rc,[75,25],axis=(0,1)))/1.349);scale=np.clip((rsd+1e-6)/(ds+1e-6),.75,1.25);tr=shift_image(z["res"]*scale,dx,dy);tm=shift_image(z["soft"],dx,dy);im=np.clip(recip.astype(float)+tr,0,255).astype(np.uint8);mask=tm>1e-6;tb=[ch["x"],ch["y"],ch["x"]+bw,ch["y"]+bh];ip=out_img/(r["instance_id"]+".png");lp0=out_lab/(r["instance_id"]+".txt");Image.fromarray(im).save(ip);H,W=im.shape[:2];lp0.write_text(f"{r['class_id']} {(tb[0]+tb[2])/(2*W):.10f} {(tb[1]+tb[3])/(2*H):.10f} {(tb[2]-tb[0])/W:.10f} {(tb[3]-tb[1])/H:.10f}\n")
  eff=im.astype(float)-recip.astype(float);identity.append({"instance_id":r["instance_id"],"residual_cosine":cos(z["res"],shift_image(eff,-dx,-dy)),"edge_cosine":cos(edge(np.clip(z["n"].astype(float)+z["res"],0,255)),edge(np.clip(z["n"].astype(float)+shift_image(eff,-dx,-dy),0,255))),"mask_iou":float(np.sum(mask&(tm>0))/np.sum(mask|(tm>0))),"residual_norm_ratio":float(np.linalg.norm(eff)/(np.linalg.norm(z["res"])+1e-12))});context.append({"instance_id":r["instance_id"],"lpips":lp(dc,rc),"ssim":ssim(dc,rc),"pixel_ncc":ch["pixel_ncc"],"edge_ncc":ch["edge_ncc"]});real_seam.append(seam(z["d"],z["mask"]));cpt_seam.append(seam(im,mask));records.append({"donor_instance_id":r["instance_id"],"donor_pair_id":r["pair_id"],"class_name":r["class_name"],"class_id":r["class_id"],"recipient_pair_id":ch["recipient_pair_id"],"recipient_image":ch["recipient_image"],"recipient_x":ch["x"],"recipient_y":ch["y"],"donor_bbox":r["bbox_xyxy"],"transported_bbox":tb,"structural_edge_ncc":ch["edge_ncc"],"pixel_ncc":ch["pixel_ncc"],"photometric_scale_rgb":[float(x) for x in scale],"image_path":str(ip.resolve()),"label_path":str(lp0.resolve()),"image_sha256":sha(ip),"label_sha256":sha(lp0)})
 match=[{"donor_instance_id":z["row"]["instance_id"],**ch,"candidate_pair_count":len(candidates[z["row"]["instance_id"]]),"recipient_reuse_count":reuse[ch["recipient_pair_id"]]} for z,ch in assigned];save(a.output/"recipient_matching_audit.json",{"status":"PASS","reuse_counts":dict(reuse),"records":match})
 ident={k:summ([x[k] for x in identity]) for k in ("residual_cosine","edge_cosine","mask_iou","residual_norm_ratio")};iok=ident["residual_cosine"]["median"]>=.95 and ident["edge_cosine"]["median"]>=.90 and ident["mask_iou"]["median"]==1;save(a.output/"defect_identity_preservation.json",{"status":"PASS" if iok else "FAIL","summary":ident,"records":identity});save(a.output/"context_intervention_audit.json",{"role":"audit only; never selection","records":context})
 rg=[x["boundary_gradient_jump"] for x in real_seam];cg=[x["boundary_gradient_jump"] for x in cpt_seam];sok=np.median(cg)<=np.quantile(rg,.95) and np.mean(np.asarray(cg)<=max(rg))>=.9;save(a.output/"boundary_seam_audit.json",{"status":"PASS" if sok else "TRANSPORT_ARTIFACT_TOO_STRONG","real":{k:summ([x[k] for x in real_seam]) for k in real_seam[0]},"cpt":{k:summ([x[k] for x in cpt_seam]) for k in cpt_seam[0]},"fraction_cpt_gradient_le_real_max":float(np.mean(np.asarray(cg)<=max(rg))),"real_records":real_seam,"cpt_records":cpt_seam})
 vok=len(records)==120 and max(reuse.values())<=2 and all(x["donor_pair_id"]!=x["recipient_pair_id"] and x["recipient_pair_id"] in allowed for x in records) and Counter(x["class_name"] for x in records)==Counter({c:20 for c in CLASSES});save(a.output/"counterfactual_transport_validity.json",{"status":"PASS" if vok else "FAIL","donor_identity_exact":set(x["donor_instance_id"] for x in records)==set(donor_ids),"class_counts":dict(Counter(x["class_name"] for x in records)),"recipient_reuse_max":max(reuse.values()),"official_validation_recipients":0})
 if not (iok and sok and vok):raise RuntimeError("STAGE13A_TRANSPORT_GATE_FAILED")
 manifest={"status":"FROZEN","count":120,"records":records};save(a.output/"cpt120_manifest.json",manifest);mh=sha(a.output/"cpt120_manifest.json");freeze_doc={"status":"FROZEN_BEFORE_DETECTOR_TRAINING","manifest_sha256":mh,"must_not_change":True,"donor_identity_exactly_stage11b_uniform":True};save(a.output/"manifest_freeze_audit.json",freeze_doc);save(a.output/"cpt120_manifest_freeze.json",freeze_doc)
 print(json.dumps({"status":"STAGE13A_TRANSPORT_GATES_PASS","manifest_sha256":mh},indent=2))
if __name__=="__main__":main()
