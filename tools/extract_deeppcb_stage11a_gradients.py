"""Stage11A-1: real-only target-isolated OOF detector-gradient extraction."""
from __future__ import annotations
import argparse,hashlib,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
from PIL import Image

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def tensor_hash(items):
 h=hashlib.sha256()
 for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def mask(path):return np.asarray(Image.open(path).convert("L"))>127
def tight(m):
 y,x=np.where(m)
 if not len(x):raise RuntimeError("STAGE11A_EMPTY_MASK")
 return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
def crop_protocol(box,w,h,scale=4,minimum=64):
 x0,y0,x1,y1=box;side=min(w,h,max(minimum,int(math.ceil(max(x1-x0,y1-y0)*scale))));cx=(x0+x1)/2;cy=(y0+y1)/2;left=int(round(cx-side/2));top=int(round(cy-side/2));left=max(0,min(w-side,left));top=max(0,min(h-side,top));s=512/side;b=[(x0-left)*s,(y0-top)*s,(x1-left)*s,(y1-top)*s];return [left,top,left+side,top+side],b
def isolated(row,by_pair):
 defect=np.asarray(Image.open(row["defect_image"]).convert("RGB"));normal=np.asarray(Image.open(row["paired_normal_image"]).convert("RGB"));tm=mask(row["instance_mask_path"]);assert defect.shape==normal.shape and tm.shape==defect.shape[:2];other=np.zeros_like(tm)
 missing=[]
 for q in by_pair[row["pair_id"]]:
  if q["instance_id"]==row["instance_id"]:continue
  if not Path(q["instance_mask_path"]).is_file():missing.append(q["instance_id"]);continue
  om=mask(q["instance_mask_path"])
  if om.shape!=tm.shape:missing.append(q["instance_id"]);continue
  other|=om
 replace=other&~tm;out=defect.copy();out[replace]=normal[replace];changed=np.any(out!=defect,axis=2);audit={"instance_id":row["instance_id"],"pair_id":row["pair_id"],"class_name":row["class_name"],"target_mask_pixel_unchanged_fraction":float(np.mean(np.all(out[tm]==defect[tm],axis=1))) if tm.any() else 0.,"other_mask_pixels":int(other.sum()),"replacement_pixels":int(replace.sum()),"other_non_target_matches_normal_fraction":float(np.mean(np.all(out[replace]==normal[replace],axis=1))) if replace.any() else 1.,"changed_pixels_outside_other_non_target":int((changed&~replace).sum()),"paired_normal_size_match":defect.shape==normal.shape,"target_bbox_unchanged":tight(tm)==row["bbox_xyxy"],"missing_other_masks":missing,"image_valid":bool(np.isfinite(out).all())};return out,tm,audit
def selected_head(model):
 detect=next((m for m in reversed(list(model.modules())) if m.__class__.__name__.lower()=="detect"),None)
 if detect is None:raise RuntimeError("STAGE11A_DETECT_HEAD_NOT_FOUND")
 modules=[]
 for branch_name in ("cv2","cv3"):
  branch=getattr(detect,branch_name,None)
  if branch is None:raise RuntimeError("STAGE11A_HEAD_BRANCH_MISSING_"+branch_name)
  for seq in branch:
   convs=[m for m in seq.modules() if isinstance(m,torch.nn.Conv2d)]
   if not convs:raise RuntimeError("STAGE11A_FINAL_CONV_MISSING")
   modules.append((branch_name,convs[-1]))
 ids={id(p) for _,m in modules for p in m.parameters(recurse=False)};selected=[(n,p) for n,p in model.named_parameters() if id(p) in ids];return detect,selected
def prepare_loss_config(model):
 """Restore the attribute-style training hyperparameters required by YOLO loss.

 Checkpoints loaded through ``YOLO(...).model`` may expose ``model.args`` as a
 plain dict.  Prediction tolerates that representation, but the detection loss
 reads values such as ``hyp.box`` and therefore requires Ultralytics' namespace.
 """
 from ultralytics.cfg import DEFAULT_CFG_DICT
 from ultralytics.utils import IterableSimpleNamespace
 current=model.args if isinstance(model.args,dict) else vars(model.args)
 merged={**DEFAULT_CFG_DICT,**current}
 model.args=IterableSimpleNamespace(**merged)
 model.criterion=None
 required=("box","cls","dfl")
 if not all(hasattr(model.args,k) for k in required):raise RuntimeError("STAGE11A_LOSS_HYPERPARAMETERS_INCOMPLETE")
 return {k:float(getattr(model.args,k)) for k in required}
def percentile(v):
 from scipy.stats import rankdata
 a=np.asarray(v,float);return (rankdata(a,method="average")-1)/max(1,len(a)-1)
def main():
 p=argparse.ArgumentParser()
 for n in ("registry","folds","stage3r","stage3r_runs","output"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--device",default="0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=load(a.registry)["instances"];folds=load(a.folds);assert len(rows)==775 and folds["split_unit"]=="physical_pair_id" and folds["official_val_use_count"]==0
 foldof={pid:int(f["fold"]) for f in folds["folds"] for pid in f["holdout_pair_ids"]};trainof={int(f["fold"]):set(f["train_pair_ids"]) for f in folds["folds"]};by_pair=defaultdict(list)
 for x in rows:by_pair[x["pair_id"]].append(x)
 checkpoints={f:a.stage3r_runs/f"fold_{f}/weights/best.pt" for f in range(3)};teacher_audit={"total_instances":len(rows),"physical_pairs":len(by_pair),"folds":3,"class_counts":dict(Counter(x["class_name"] for x in rows)),"pair_excluded_from_teacher_train":all(x["pair_id"] not in trainof[foldof[x["pair_id"]]] for x in rows),"teacher_checkpoint_sha256":{str(f):sha_file(checkpoints[f]) for f in range(3)},"official_validation_use_count":0,"records":[{"instance_id":x["instance_id"],"pair_id":x["pair_id"],"fold":foldof[x["pair_id"]],"excluded":x["pair_id"] not in trainof[foldof[x["pair_id"]]]} for x in rows]};save(a.output/"oof_gradient_teacher_audit.json",teacher_audit)
 if not teacher_audit["pair_excluded_from_teacher_train"]:raise RuntimeError("STAGE11A_OOF_RELATION_FAILED")
 # Frozen, class-stratified 80-instance isolation audit.
 rng=random.Random(2026);audit_ids=set()
 for c in CLASSES:
  z=[x for x in rows if x["class_name"]==c];audit_ids.update(x["instance_id"] for x in rng.sample(z,20 if c in ("short","pinhole") else 10))
 isolation=[]
 for x in rows:
  if x["instance_id"] in audit_ids:isolation.append(isolated(x,by_pair)[2])
 iso_ok=len(isolation)==80 and all(x["target_mask_pixel_unchanged_fraction"]==1 and x["other_non_target_matches_normal_fraction"]==1 and x["changed_pixels_outside_other_non_target"]==0 and x["paired_normal_size_match"] and x["target_bbox_unchanged"] and not x["missing_other_masks"] and x["image_valid"] for x in isolation);save(a.output/"target_isolation_audit.json",{"status":"TARGET_ISOLATION_PASS" if iso_ok else "TARGET_ISOLATION_INCOMPLETE","sample_count":len(isolation),"sampling":{"short":20,"pinhole":20,"other_classes_each":10,"seed":2026},"records":isolation})
 save(a.output/"gradient_input_protocol.json",{"status":"FROZEN","context_scale":4,"minimum_crop":64,"crop":"square centered on target bbox","resize":512,"training_target":"one target bbox and one target class","dynamic_context":False,"synthetic_generation":False})
 if not iso_ok:raise RuntimeError("TARGET_ISOLATION_INCOMPLETE")
 from ultralytics import YOLO
 structures={};models={};selected={};before={};bn_before={};loss_hyp={}
 for f in range(3):
  model=YOLO(str(checkpoints[f])).model.to("cuda:"+a.device);loss_hyp[f]=prepare_loss_config(model);model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
  for q in model.parameters():q.requires_grad_(False)
  _,sel=selected_head(model)
  for _,q in sel:q.requires_grad_(True)
  structures[f]=[{"name":n,"shape":list(q.shape),"numel":q.numel(),"branch":"box" if ".cv2." in n else "classification"} for n,q in sel];models[f]=model;selected[f]=sel;before[f]=tensor_hash([(n,q) for n,q in sel]);bn_before[f]=tensor_hash([(n,b) for n,b in model.state_dict().items() if "running_mean" in n or "running_var" in n or "num_batches_tracked" in n])
 match=all([(x["name"],x["shape"]) for x in structures[0]]==[(x["name"],x["shape"]) for x in structures[f]] for f in (1,2));total=sum(x["numel"] for x in structures[0]);save(a.output/"gradient_parameter_audit.json",{"selected_scope":"Detect cv2/cv3 final Conv2d weights and biases only","folds":structures,"loss_hyperparameters":loss_hyp,"loss_config_normalization":"DEFAULT_CFG_DICT merged with checkpoint args and converted to IterableSimpleNamespace","total_selected_dimensions":total,"parameter_name_and_shape_match_across_folds":match})
 if not match:raise RuntimeError("STAGE11A_PARAMETER_STRUCTURE_MISMATCH")
 # Deterministic CountSketch fixed before any gradient result.
 prng=np.random.default_rng(2026);buckets=prng.integers(0,1024,size=total,dtype=np.int32);signs=prng.choice(np.asarray([-1.,1.],np.float32),size=total);bucket_t=torch.from_numpy(buckets.astype(np.int64)).to("cuda:"+a.device);sign_t=torch.from_numpy(signs).to("cuda:"+a.device)
 sample_ids=[]
 for f in range(3):
  for c in CLASSES:
   z=[x for x in rows if foldof[x["pair_id"]]==f and x["class_name"]==c];sample_ids.extend(x["instance_id"] for x in random.Random(2026+f*10+CLASSES.index(c)).sample(z,min(4,len(z))))
 sample_ids=set(sample_ids);raw_sample={};diag=[];emb=[];ids=[];pairs=[];fs=[];cs=[];losses=[];norms=[]
 for idx,x in enumerate(rows):
  f=foldof[x["pair_id"]];model=models[f];model.zero_grad(set_to_none=True);im,tm,_=isolated(x,by_pair);box=x["bbox_xyxy"];crop,b2=crop_protocol(box,im.shape[1],im.shape[0]);patch=np.asarray(Image.fromarray(im).crop(crop).resize((512,512),Image.Resampling.BILINEAR));img=torch.from_numpy(patch.copy()).permute(2,0,1).unsqueeze(0).to("cuda:"+a.device).float()/255.;x0,y0,x1,y1=b2;target={"img":img,"batch_idx":torch.zeros(1,device=img.device,dtype=torch.long),"cls":torch.tensor([[x["class_id"]]],device=img.device,dtype=torch.float32),"bboxes":torch.tensor([[(x0+x1)/1024,(y0+y1)/1024,(x1-x0)/512,(y1-y0)/512]],device=img.device,dtype=torch.float32)};loss,items=model(target);objective=loss.sum();objective.backward();flat=torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).detach().reshape(-1).float() for _,q in selected[f]]);gn=float(flat.norm());finite=bool(torch.isfinite(flat).all());zero=gn==0;proj=torch.zeros(1024,device=flat.device);proj.scatter_add_(0,bucket_t,flat*sign_t);pn=proj.norm();direction=(proj/pn.clamp_min(1e-30)).cpu().numpy()
  vals=loss.detach().float().reshape(-1).cpu().tolist()
  if len(vals)!=3:raise RuntimeError(f"STAGE11A_UNEXPECTED_LOSS_VECTOR_SHAPE: {list(loss.shape)}")
  boxloss,clsloss,dflloss=vals;aux_keys=sorted(str(k) for k in items) if isinstance(items,dict) else None;rec={"instance_id":x["instance_id"],"pair_id":x["pair_id"],"class_name":x["class_name"],"class_id":x["class_id"],"fold":f,"total_loss":float(objective.detach()),"loss_tensor_shape":list(loss.shape),"loss_reduction":"sum","loss_component_source":"primary loss tensor [box, cls, dfl]","auxiliary_return_type":type(items).__name__,"auxiliary_return_keys":aux_keys,"box_loss":boxloss,"cls_loss":clsloss,"dfl_loss":dflloss,"raw_gradient_l2_norm":gn,"raw_gradient_finite":finite,"zero_gradient":zero,"real_path":x["defect_image"],"mask_path":x["instance_mask_path"]};diag.append(rec);emb.append(direction);ids.append(x["instance_id"]);pairs.append(x["pair_id"]);fs.append(f);cs.append(x["class_id"]);losses.append([rec["total_loss"],boxloss,clsloss,dflloss]);norms.append(gn)
  if x["instance_id"] in sample_ids:raw_sample[x["instance_id"]]=flat.cpu().numpy()
  model.zero_grad(set_to_none=True)
 valid=all(x["raw_gradient_finite"] for x in diag) and np.mean([not x["zero_gradient"] for x in diag])>=.99;save(a.output/"raw_gradient_diagnostics.json",{"status":"PASS" if valid else "GRADIENT_EXTRACTION_INVALID","all_finite":all(x["raw_gradient_finite"] for x in diag),"nonzero_gradient_fraction":float(np.mean([not x["zero_gradient"] for x in diag])),"records":diag})
 after={f:tensor_hash([(n,q) for n,q in selected[f]]) for f in range(3)};bn_after={f:tensor_hash([(n,b) for n,b in models[f].state_dict().items() if "running_mean" in n or "running_var" in n or "num_batches_tracked" in n]) for f in range(3)};immut={"folds":{str(f):{"selected_before_sha256":before[f],"selected_after_sha256":after[f],"selected_unchanged":before[f]==after[f],"bn_before_sha256":bn_before[f],"bn_after_sha256":bn_after[f],"bn_running_state_unchanged":bn_before[f]==bn_after[f]} for f in range(3)},"optimizer_count":0,"parameter_update_count":0,"detector_training_count":0};save(a.output/"teacher_immutability_audit.json",immut)
 np.savez_compressed(a.output/"real_oof_gradient_embeddings.npz",embeddings=np.asarray(emb,np.float32),instance_ids=np.asarray(ids),pair_ids=np.asarray(pairs),folds=np.asarray(fs,np.int8),class_ids=np.asarray(cs,np.int8),loss=np.asarray(losses,np.float32),gradient_norm=np.asarray(norms,np.float32))
 # Projection fidelity only compares same-fold, same-class, different-pair samples.
 er={x["instance_id"]:np.asarray(emb[i]) for i,x in enumerate(rows)};raw_d=[];proj_d=[];sid=sorted(raw_sample)
 for i in range(len(sid)):
  for j in range(i+1,len(sid)):
   a0,b0=sid[i],sid[j];ra=next(x for x in rows if x["instance_id"]==a0);rb=next(x for x in rows if x["instance_id"]==b0)
   if foldof[ra["pair_id"]]!=foldof[rb["pair_id"]] or ra["class_id"]!=rb["class_id"] or ra["pair_id"]==rb["pair_id"]:continue
   u,v=raw_sample[a0],raw_sample[b0];raw_d.append(float(1-np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))));proj_d.append(float(1-np.dot(er[a0],er[b0])))
 from scipy.stats import spearmanr,pearsonr
 err=np.abs(np.asarray(raw_d)-np.asarray(proj_d));rho=float(spearmanr(raw_d,proj_d).statistic);pear=float(pearsonr(raw_d,proj_d).statistic);projection={"status":"GRADIENT_PROJECTION_PASS" if rho>=.95 and float(err.mean())<=.05 else "GRADIENT_PROJECTION_FAIL","output_dim":1024,"seed":2026,"raw_instance_count":len(raw_sample),"feasible_pair_count":len(raw_d),"spearman_rho":rho,"pearson":pear,"mae":float(err.mean()),"q95_absolute_error":float(np.quantile(err,.95)),"max_error":float(err.max()),"gate":{"spearman_min":.95,"mae_max":.05}};save(a.output/"gradient_projection_audit.json",projection)
 if not valid or not all(v["selected_unchanged"] and v["bn_running_state_unchanged"] for v in immut["folds"].values()):raise RuntimeError("GRADIENT_EXTRACTION_INVALID")
 if projection["status"]!="GRADIENT_PROJECTION_PASS":raise RuntimeError("GRADIENT_PROJECTION_FAIL")
 print(json.dumps({"status":"STAGE11A_EXTRACTION_COMPLETE","instances":len(rows),"projection":projection},indent=2))
if __name__=="__main__":main()
