"""Complete Stage14B fidelity attribution and freeze confirmation subsets before training."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys
from collections import Counter,defaultdict
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import Inception_V3_Weights,inception_v3
from analyze_plastic_bomo_stage14a import load,save,resolve_layout,CLASSES

SUBSETS=("random","high_fidelity","low_fidelity_diagnostic")
class InceptionFeatures(torch.nn.Module):
 def __init__(self):
  super().__init__();model=inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1);model.fc=torch.nn.Identity();model.dropout=torch.nn.Identity();self.model=model.eval()
 def forward(self,x):return self.model(x)
def unbiased_mmd(first,second):
 d=first.shape[1];kxx=(first@first.T/d+1.)**3;kyy=(second@second.T/d+1.)**3;kxy=(first@second.T/d+1.)**3;n=len(first);m=len(second)
 if n<2 or m<2:raise RuntimeError("STAGE14B_KID_REQUIRES_AT_LEAST_TWO_SAMPLES")
 return (kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def crop(path,box):
 im=Image.open(path).convert("RGB");x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0;side=max(w,h)*1.2;cx=(x0+x1)/2;cy=(y0+y1)/2;left=max(0,min(im.width-side,cx-side/2));top=max(0,min(im.height-side,cy-side/2));return im.crop((left,top,min(im.width,left+side),min(im.height,top+side))).resize((224,224),Image.Resampling.BILINEAR)
def geom(im,box):
 x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0;g=np.asarray(crop(im,box).convert("L"),np.float32);gy,gx=np.gradient(g);e=np.hypot(gx,gy);hist=np.histogram(e,bins=16,range=(0,max(float(e.max()),1)),density=False)[0].astype(float);p=hist/max(hist.sum(),1);bc=float(-(p[p>0]*np.log(p[p>0])).sum())
 return np.array([w,h,w*h,w/max(h,1e-6),float((e>np.quantile(e,.75)).mean()),bc],float)
def contrast(im,box):
 a=np.asarray(Image.open(im).convert("L"),np.float32);H,W=a.shape;x0,y0,x1,y1=map(int,box);x0=max(0,x0);y0=max(0,y0);x1=min(W,max(x0+1,x1));y1=min(H,max(y0+1,y1));w=x1-x0;h=y1-y0;cx0=max(0,int(x0-.1*w));cy0=max(0,int(y0-.1*h));cx1=min(W,int(x1+.1*w));cy1=min(H,int(y1+.1*h));c=a[y0:y1,x0:x1];z=a[cy0:cy1,cx0:cx1];mask=np.ones(z.shape,bool);mask[y0-cy0:y1-cy0,x0-cx0:x1-cx0]=False;r=z[mask]
 return np.array([c.mean(),r.mean(),abs(c.mean()-r.mean())/(r.std()+1e-6),abs(c.mean()-r.mean()),c.std()/(r.std()+1e-6)],float)
def edge(im,box):
 g=np.asarray(crop(im,box).convert("L"),np.float32);gy,gx=np.gradient(g);e=np.hypot(gx,gy);hist=np.histogram(e,bins=16,range=(0,max(float(e.max()),1)),density=True)[0];f=np.abs(np.fft.rfft2(g-g.mean()))**2;yy,xx=np.indices(f.shape);rad=np.sqrt((yy/f.shape[0])**2+(xx/f.shape[1])**2);hf=float(f[rad>.25].mean()) if np.any(rad>.25) else 0.;bd=float(np.r_[e[:5].ravel(),e[-5:].ravel(),e[:,:5].ravel(),e[:,-5:].ravel()].mean());return np.r_[hist,hf,bd]
def real_rows(data_yaml):
 root,ims,labs=resolve_layout(data_yaml);names={0:"flash",1:"black"};out=[]
 for ip in sorted(x for x in ims.iterdir() if x.is_file()):
  with Image.open(ip) as im:W,H=im.size
  for j,z in enumerate((labs/(ip.stem+".txt")).read_text().splitlines()):
   q=z.split();c=int(float(q[0]));cx,cy,w,h=map(float,q[1:5]);box=[(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H];out.append({"id":f"{ip.stem}:{j}","class_name":names[c],"image_path":str(ip),"bbox_xyxy":box})
 return out
def sm(v):return {"mean":float(np.mean(v)),"median":float(np.median(v)),"q90":float(np.quantile(v,.9)),"n":len(v)}
def extract_kid(crops,model,dev,batch):
 tf=transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);out=[]
 with torch.no_grad():
  for i in range(0,len(crops),batch):out.append(model(torch.stack([tf(x) for x in crops[i:i+batch]]).to(dev)).cpu().numpy())
 return np.concatenate(out).astype(np.float64)
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--stage14a",type=Path,required=True);p.add_argument("--candidate_pool",type=Path,required=True);p.add_argument("--real_data_yaml",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0");p.add_argument("--batch",type=int,default=32);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol)
 if sha(a.candidate_pool)!=cfg["candidate_pool_sha256"]:raise RuntimeError("STAGE14B_CANDIDATE_POOL_SHA_MISMATCH")
 pool=load(a.candidate_pool);syn=pool.get("candidates",pool.get("selected",[]));pool_counts=Counter(CLASSES[int(x["class_id"])] for x in syn)
 if len(syn)!=cfg["candidate_count"] or dict(pool_counts)!=cfg["class_counts"]:raise RuntimeError(f"STAGE14B_CANDIDATE_POOL_POPULATION_MISMATCH: {len(syn)} {dict(pool_counts)}")
 byid={x["candidate_id"]:x for x in syn};freeze=load(a.stage14a/"subset_definition.json");frozen={k:list(freeze["subsets"][k]) for k in SUBSETS};save(a.output/"stage14a_subset_freeze_audit.json",{"status":"PASS","candidate_pool_sha256":sha(a.candidate_pool),"candidate_count":len(syn),"candidate_class_counts":dict(pool_counts),"subsets":frozen,"counts":{k:dict(Counter(CLASSES[int(byid[i]["class_id"])] for i in v)) for k,v in frozen.items()},"lists_unchanged":True})
 real=real_rows(a.real_data_yaml);allrows=real+[{"id":x["candidate_id"],"class_name":CLASSES[int(x["class_id"])],"image_path":x["image_path"],"bbox_xyxy":x["bbox_xyxy"]} for x in syn]
 for x in allrows:x["morph"]=geom(x["image_path"],x["bbox_xyxy"]);x["contrast"]=contrast(x["image_path"],x["bbox_xyxy"]);x["edge"]=edge(x["image_path"],x["bbox_xyxy"])
 rb=defaultdict(list)
 for x in real:rb[x["class_name"]].append(x)
 morph=[];contr=[];edges=[]
 for x in allrows[len(real):]:
  rr=rb[x["class_name"]];M=np.stack([z["morph"] for z in rr]);med=np.median(M,0);iqr=np.quantile(M,.75,0)-np.quantile(M,.25,0);d=float(np.linalg.norm(((x["morph"]-med)/(iqr+1e-6))-((M-med)/(iqr+1e-6)),axis=1).min());C=np.stack([z["contrast"] for z in rr]);pct=[float(np.mean(C[:,j]<=x["contrast"][j])) for j in range(C.shape[1])];dc=float(np.median(np.abs(np.asarray(pct)-.5))*2);E=np.stack([z["edge"] for z in rr]);de=float(np.linalg.norm((x["edge"]-E)/(np.std(E,0)+1e-6),axis=1).min());morph.append({"candidate_id":x["id"],"class_name":x["class_name"],"morphology_distance":d,"features":x["morph"].tolist()});contr.append({"candidate_id":x["id"],"class_name":x["class_name"],"contrast_fidelity_distance":dc,"percentiles":pct,"features":x["contrast"].tolist()});edges.append({"candidate_id":x["id"],"class_name":x["class_name"],"edge_distance":de})
 save(a.output/"per_sample_morphology_distance.json",{"status":"COMPLETE","feature_order":["bbox_width","bbox_height","bbox_area","aspect_ratio","edge_density","boundary_complexity"],"epsilon":1e-6,"real_reference_records":[{"real_id":x["id"],"class_name":x["class_name"],"features":x["morph"].tolist()} for x in real],"records":morph});save(a.output/"contrast_fidelity.json",{"status":"COMPLETE","no_downstream_weights":True,"records":contr});save(a.output/"edge_fidelity.json",{"status":"COMPLETE","records":edges})
 dev=torch.device("cuda:"+a.device);import lpips as lp;net=lp.LPIPS(net="alex",verbose=False).to(dev).eval();tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)]);real_t={c:torch.stack([tf(crop(x["image_path"],x["bbox_xyxy"])) for x in rb[c]]) for c in CLASSES.values()};lpout=[]
 with torch.no_grad():
  for x in allrows[len(real):]:
   s=tf(crop(x["image_path"],x["bbox_xyxy"]));R=real_t[x["class_name"]];vals=[]
   for i in range(0,len(R),a.batch):vals.append(net(s[None].expand(min(a.batch,len(R)-i),-1,-1,-1).to(dev),R[i:i+a.batch].to(dev)).reshape(-1).cpu())
   v=torch.cat(vals);j=int(v.argmin());lpout.append({"candidate_id":x["id"],"class_name":x["class_name"],"nearest_real_id":rb[x["class_name"]][j]["id"],"nearest_real_lpips":float(v[j])})
 save(a.output/"per_sample_lpips.json",{"status":"COMPLETE","backbone":"alex","crop_protocol":cfg["crop_protocol"],"records":lpout})
 imap={x["candidate_id"]:x for x in lpout};mmap={x["candidate_id"]:x for x in morph};cmap={x["candidate_id"]:x for x in contr};emap={x["candidate_id"]:x for x in edges};model=InceptionFeatures().to(dev).eval();kid={"status":"COMPLETE","set_level_only":True,"not_used_for_ranking":True,"resamples":1000,"seed":2026,"sets":{}};rng=np.random.default_rng(2026)
 real_kid={c:extract_kid([crop(x["image_path"],x["bbox_xyxy"]) for x in rb[c]],model,dev,a.batch) for c in CLASSES.values()}
 for name,ids in frozen.items():
  kid["sets"][name]={};syn_kid={}
  for c in CLASSES.values():
   ss=[crop(byid[i]["image_path"],byid[i]["bbox_xyxy"]) for i in ids if CLASSES[int(byid[i]["class_id"])]==c];rf=real_kid[c];sf=extract_kid(ss,model,dev,a.batch);syn_kid[c]=sf;n=min(len(rf),len(sf));vals=[]
   for _ in range(1000):vals.append(float(unbiased_mmd(rf[rng.choice(len(rf),n,replace=True)],sf[rng.choice(len(sf),n,replace=True)]))*1000)
   kid["sets"][name][c]={"mean_x1000":float(np.mean(vals)),"ci95_x1000":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
  pooled_r=np.concatenate([real_kid[c] for c in CLASSES.values()]);pooled_s=np.concatenate([syn_kid[c] for c in CLASSES.values()]);n=min(len(pooled_r),len(pooled_s));vals=[float(unbiased_mmd(pooled_r[rng.choice(len(pooled_r),n,replace=True)],pooled_s[rng.choice(len(pooled_s),n,replace=True)]))*1000 for _ in range(1000)];kid["sets"][name]["overall"]={"mean_x1000":float(np.mean(vals)),"ci95_x1000":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]};kid["sets"][name]["macro_class_mean_x1000"]=float(np.mean([kid["sets"][name][c]["mean_x1000"] for c in CLASSES.values()]))
 save(a.output/"crop_kid_bootstrap.json",kid)
 comp={}
 for name,ids in frozen.items():comp[name]={"nearest_real_lpips":sm([imap[i]["nearest_real_lpips"] for i in ids]),"morphology_distance":sm([mmap[i]["morphology_distance"] for i in ids]),"contrast_fidelity":sm([cmap[i]["contrast_fidelity_distance"] for i in ids]),"edge_fidelity":sm([emap[i]["edge_distance"] for i in ids]),"crop_kid_x1000":kid["sets"][name]["overall"]["mean_x1000"]}
 broad=comp["high_fidelity"]["nearest_real_lpips"]["mean"]<comp["random"]["nearest_real_lpips"]["mean"] and comp["high_fidelity"]["crop_kid_x1000"]<comp["random"]["crop_kid_x1000"] and comp["high_fidelity"]["morphology_distance"]["mean"]<comp["random"]["morphology_distance"]["mean"];morph_only=comp["high_fidelity"]["morphology_distance"]["mean"]<comp["random"]["morphology_distance"]["mean"]
 label="BROAD_FIDELITY_DIFFERENCE_PRESENT" if broad else "MORPHOLOGY_ALIGNMENT_DIFFERENCE_PRESENT" if morph_only else "STAGE14A_FIDELITY_LABEL_NOT_VALIDATED";save(a.output/"completed_fidelity_audit.json",{"status":label,"metrics":comp,"kid_set_level_only":True,"kid_used_for_selection":False})
 conf={}
 for name,ids in frozen.items():
  z=[byid[i] for i in ids];seeds=Counter(str(x["seed"]) for x in z);parents=Counter(f"{x['class_id']}:{Path(x['source_image']).stem}" for x in z);generation_seeds=sorted({int(x["seed"]) for x in syn});cls={c:{str(s):sum(CLASSES[int(x["class_id"])]==c and int(x["seed"])==s for x in z) for s in generation_seeds} for c in CLASSES.values()};dom=any(max(Counter(str(x["seed"]) for x in z if CLASSES[int(x["class_id"])]==c).values())>14 for c in CLASSES.values());reuse=max(parents.values());configs=Counter(str(Path(x["source_image"]).parent.parent.parent) for x in z);conf[name]={"generation_seed_histogram":dict(seeds),"source_parent_histogram":dict(parents),"max_parent_reuse":reuse,"class_seed_histogram":cls,"bbox_area":{"mean":float(np.mean([x["area_fraction"] for x in z])),"median":float(np.median([x["area_fraction"] for x in z]))},"aspect_ratio":{"mean":float(np.mean([x["aspect_ratio"] for x in z])),"median":float(np.median([x["aspect_ratio"] for x in z]))},"generation_config_path_histogram":dict(configs),"prompt_identity":"NOT_AVAILABLE_IN_CANDIDATE_POOL","single_seed_over_35_percent_of_class":dom}
 high=conf["high_fidelity"];rnd=conf["random"];material=(high["single_seed_over_35_percent_of_class"] and not rnd["single_seed_over_35_percent_of_class"]) or (high["max_parent_reuse"]>3 and rnd["max_parent_reuse"]<=3)
 save(a.output/"selection_confound_audit.json",{"status":"FIDELITY_SELECTION_CONFOUNDED" if material else "NO_MATERIAL_CONFOUND","subsets":conf,"trigger":{"high_single_seed_over_35_percent_and_random_not_equivalent":high["single_seed_over_35_percent_of_class"] and not rnd["single_seed_over_35_percent_of_class"],"high_parent_over_3_and_random_not_equivalent":high["max_parent_reuse"]>3 and rnd["max_parent_reuse"]<=3},"official_validation_used":False})
 matched={"status":"NOT_REQUIRED_NO_MATERIAL_CONFOUND","subsets":frozen}
 if material:
  md={x["candidate_id"]:x["morphology_distance"] for x in morph};seeds=sorted({int(x["seed"]) for x in syn})
  def parent(x):return f"{x['class_id']}:{Path(x['source_image']).stem}"
  def build(cap):
   out={k:[] for k in ("random_matched","high_morphology_alignment","low_morphology_alignment")};used={k:Counter() for k in out}
   for c in CLASSES.values():
    for seed in seeds:
     z=[x for x in syn if CLASSES[int(x["class_id"])]==c and int(x["seed"])==seed];orders={"high_morphology_alignment":sorted(z,key=lambda x:(md[x["candidate_id"]],x["candidate_id"])),"low_morphology_alignment":sorted(z,key=lambda x:(-md[x["candidate_id"]],x["candidate_id"]))};q=list(z);random.Random(2026+seed+(0 if c=="flash" else 100000)).shuffle(q);orders["random_matched"]=q
     for arm,order in orders.items():
      take=[x for x in order if used[arm][parent(x)]<cap][:5]
      if len(take)<5:return None
      for x in take:out[arm].append(x["candidate_id"]);used[arm][parent(x)]+=1
   return out
  cap=1;out=build(cap)
  if out is None:cap=2;out=build(cap)
  if out is None:raise RuntimeError("STAGE14B_MATCHED_PARENT_CAP_INFEASIBLE_AT_2")
  matched={"status":"FROZEN_MATCHED_SUBSETS","reason":"FIDELITY_SELECTION_CONFOUNDED","selection_seed":2026,"quota_per_class_generation_seed":5,"parent_reuse_cap":cap,"subsets":out}
 save(a.output/"matched_subset_definition.json",matched);save(a.output/"subset_freeze_audit.json",{"status":"PASS","matched":material,"subset_definition_sha256":sha(a.output/"matched_subset_definition.json"),"counts":{k:dict(Counter(CLASSES[int(byid[i]["class_id"])] for i in v)) for k,v in matched["subsets"].items()},"frozen_before_detector_training":True});save(a.output/"stage14b_protocol.json",cfg);print(json.dumps({"status":"STAGE14B_ATTRIBUTION_COMPLETE","fidelity_label":label,"confounded":material,"confirmation_subsets":matched["status"]},indent=2))
if __name__=="__main__":main()
