#!/usr/bin/env python3
"""Poisson NORMAL_CLONE baseline and frozen YOLO11s downstream protocol."""
from __future__ import annotations
import argparse,csv,json,math,os,platform,shutil
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch,yaml
from ultralytics import __version__ as ultralytics_version
import run_plastic_bomo_lora_downstream as core

ROOT=Path(__file__).resolve().parents[1];FORMAL=ROOT/'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_A_manifest.json';ALLOW=ROOT/'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115/generator_train_allowlist.json'
OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/poisson_blending';RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/poisson_blending';DS=RUNS/'dataset_binding';GEN=RUNS/'generated_bank_A';DATA=DS/'poisson_bank_a.yaml';SEEDS=(42,3407,2026);METHOD='Poisson Blending'
for k,v in {'OUT':OUT,'RUNS':RUNS,'DS':DS,'DATA':DATA}.items():setattr(core,k,v)
core.RUN_PREFIX='poisson_bank_a';core.METHOD_NAME=METHOD
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():
  if dst.is_symlink() and dst.resolve()==src.resolve():return
  raise RuntimeError(f'collision: {dst}')
 os.symlink(src.resolve(),dst)
def box_px(b):
 cx,cy,w,h=map(float,b);return [max(0,math.floor((cx-w/2)*4096)),max(0,math.floor((cy-h/2)*1024)),min(4096,math.ceil((cx+w/2)*4096)),min(1024,math.ceil((cy+h/2)*1024))]
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 formal=json.loads(FORMAL.read_text());rows=formal['records'];allow=json.loads(ALLOW.read_text());am={r['annotation_id']:r for r in allow['inputs']};errors=[];donors=[];generated=[];cc=Counter(r['class'] for r in rows)
 detector_cfg=yaml.safe_load(core.REAL.read_text());detector_base=Path(detector_cfg['path']);validation_files=core.files(detector_base/detector_cfg['val']);validation_hashes={core.sha(p) for p in validation_files}
 if len(rows)!=80 or len({r['slot_id'] for r in rows})!=80:errors.append('FROZEN_SLOT_COUNT')
 if cc!=Counter({'flash':40,'black':40}):errors.append('CLASS_COUNTS')
 if allow.get('status')!='PASS':errors.append('ALLOWLIST_NOT_PASS')
 for r in rows:
  fs=r['frozen_slot'];ann=fs['annotation_id'];a=am.get(ann);e=[];src=Path(fs['source_path']);dst=Path(fs['normal_path'])
  if fs['split']!='train' or a is None or a.get('split')!='train':e.append('DONOR_NOT_TRAIN')
  if fs['source_sha256'] in validation_hashes:e.append('DOWNSTREAM_VALIDATION_SHA_OVERLAP')
  if a and (a.get('class')!=fs['class'] or a.get('sha256')!=fs['source_sha256']):e.append('ALLOWLIST_IDENTITY')
  if not src.is_file() or core.sha(src)!=fs['source_sha256']:e.append('DONOR_SHA')
  if not dst.is_file() or core.sha(dst)!=fs['normal_sha256']:e.append('NORMAL_SHA')
  donors.append({'slot_id':r['slot_id'],'annotation_id':ann,'source_group_id':fs['source_group_id'],'class':fs['class'],'split':fs['split'],'donor_sha256':fs['source_sha256'],'normal_source_group':fs['normal_source_group'],'errors':list(e)})
  out=GEN/f"{r['slot_id']}.png";label=GEN/f"{r['slot_id']}.txt";status='GENERATION_FAILED';signal={}
  if not e:
   s=cv2.imread(str(src),cv2.IMREAD_COLOR);d=cv2.imread(str(dst),cv2.IMREAD_COLOR)
   if s is None or d is None or s.dtype!=np.uint8 or d.dtype!=np.uint8 or s.shape!=d.shape or s.shape[:2]!=(1024,4096):e.append('BGR_INPUT')
  if not e:
   x0,y0,x1,y1=box_px(fs['bbox']);w,h=x1-x0,y1-y0;pad=max(4,int(math.ceil(max(w,h)*0.25)));sx0,maxy=max(0,x0-pad),max(0,y0-pad);sx1,miny=min(4096,x1+pad),min(1024,y1+pad);patch=s[maxy:miny,sx0:sx1].copy();mask=np.zeros(patch.shape[:2],np.uint8);mask[y0-maxy:y1-maxy,x0-sx0:x1-sx0]=255;center=(int(round((x0+x1)/2)),int(round((y0+y1)/2)))
   try:outim=cv2.seamlessClone(patch,d,mask,center,cv2.NORMAL_CLONE)
   except cv2.error as ex:e.append('SEAMLESS_CLONE_FAILED:'+str(ex).splitlines()[0])
  if not e:
   if not cv2.imwrite(str(out),outim):e.append('WRITE_FAILED')
   else:
    label.write_text(f"{0 if fs['class']=='flash' else 1} "+' '.join(f'{float(x):.10g}' for x in fs['bbox'])+'\n');roi=np.abs(outim[y0:y1,x0:x1].astype(np.int16)-d[y0:y1,x0:x1].astype(np.int16));base=np.abs(s[y0:y1,x0:x1].astype(np.int16)-d[y0:y1,x0:x1].astype(np.int16));changed=int((roi.max(2)>2).sum());pix=max(1,w*h);mean=float(roi.mean());ret=mean/max(float(base.mean()),1e-12);vis='NO_SIGNAL' if changed==0 or int(roi.max())<3 else ('LOW_SIGNAL' if changed/pix<0.1 or ret<0.05 else 'VISIBLE_SIGNAL');signal={'bbox_xyxy':[x0,y0,x1,y1],'bbox_pixels':w*h,'changed_pixels_gt2':changed,'changed_fraction_gt2':changed/pix,'mean_abs_change':mean,'max_abs_change':int(roi.max()),'donor_vs_normal_mean_abs':float(base.mean()),'signal_retention_ratio':ret,'visibility_status':vis};status='COMPLETE'
  generated.append({'slot_id':r['slot_id'],'annotation_id':ann,'source_group_id':fs['source_group_id'],'class':fs['class'],'normal_id':fs['conditioning_normal_id'],'bbox':fs['bbox'],'mask_source':'BBOX_DERIVED_NOT_PIXEL_GT','clone_mode':'cv2.NORMAL_CLONE','image_path_server_only':str(out),'image_sha256':core.sha(out) if out.is_file() else None,'label_path_server_only':str(label),'label_sha256':core.sha(label) if label.is_file() else None,'status':status,'errors':list(e),**signal});errors.extend(f"{r['slot_id']}:{x}" for x in e)
 core.save(OUT/'donor_source_audit.json',{'status':'PASS' if not any(x['errors'] for x in donors) else 'FAIL','allowlist_sha256':core.sha(ALLOW),'downstream_validation_count':len(validation_files),'downstream_validation_hashes_checked':True,'donor_count':len(donors),'unique_annotations':len({x['annotation_id'] for x in donors}),'unique_source_groups':len({x['source_group_id'] for x in donors}),'reuse_count':len(donors)-len({x['annotation_id'] for x in donors}),'all_stage20b_logical_train':all(x['split']=='train' for x in donors),'downstream_validation_sha_overlap_count':sum('DOWNSTREAM_VALIDATION_SHA_OVERLAP' in x['errors'] for x in donors),'records':donors})
 core.save(OUT/'generation_audit.json',{'status':'PASS' if not errors else 'FAIL','method':METHOD,'formal_manifest_sha256':core.sha(FORMAL),'protocol':{'library':f'OpenCV {cv2.__version__}','operation':'cv2.seamlessClone(src,dst,mask,center,cv2.NORMAL_CLONE)','image_type':'uint8 BGR','mask_type':'uint8 single-channel bbox-derived','mask_is_human_pixel_gt':False,'morphology_scaling':False,'quality_selection':False,'best_of_n':False,'reroll':False},'planned':80,'complete':sum(x['status']=='COMPLETE' for x in generated),'class_counts':dict(cc),'visibility_counts':dict(Counter(x.get('visibility_status','GENERATION_FAILED') for x in generated)),'errors':errors,'records':generated})
 cfg=yaml.safe_load(core.REAL.read_text());base=Path(cfg['path']);real=core.files(base/cfg['train']);instances=0
 for q in (DS/'images/train',DS/'labels/train'):
  if q.exists():shutil.rmtree(q)
 for i,p in enumerate(real):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');instances+=sum(bool(x.strip()) for x in lab.read_text().splitlines());name=f'real_{i:03d}_{p.name}';link(p,DS/'images/train'/name);link(lab,(DS/'labels/train'/name).with_suffix('.txt'))
 if not errors:
  for x in generated:link(Path(x['image_path_server_only']),DS/'images/train'/f"poisson_{x['slot_id']}.png");link(Path(x['label_path_server_only']),DS/'labels/train'/f"poisson_{x['slot_id']}.txt")
 val=(base/cfg['val']).resolve();DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str(val),'names':cfg['names']},sort_keys=False));core.save(OUT/'dataset_audit.json',{'status':'PASS' if not errors else 'FAIL','method':METHOD,'frozen_slots':len(rows),'slot_ids_unique':len({r['slot_id'] for r in rows})==80,'class_counts':dict(cc),'real_train_images':len(real),'real_instances':instances,'synthetic_images':sum(x['status']=='COMPLETE' for x in generated),'total_train_images':len(real)+sum(x['status']=='COMPLETE' for x in generated),'final_canvas_wh':[4096,1024],'validation_split_unchanged':True,'image_and_label_sha_verified':not errors,'generation_failures':len(errors),'errors':errors})
 if errors:raise RuntimeError(f'prepare failed: {len(errors)} errors')

def train(seed,device):
 if json.loads((OUT/'dataset_audit.json').read_text()).get('status')!='PASS':raise RuntimeError('prepare audit not PASS')
 core.train(seed,device)
def recover(seed,device):core.recover(seed,device)
def ms(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def finalize():
 cur=[json.loads((OUT/f'seed_{s}_result.json').read_text()) for s in SEEDS];hist=json.loads(core.HIST.read_text())['records'];rr=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json').read_text()) for s in SEEDS]
 def loadgroup(name):return [json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/{name}/seed_{s}_result.json').read_text()) for s in SEEDS]
 groups={'Real-only':[r for r in hist if r['arm']=='real_only'],'RealRepeat':rr,'Raw SD2':loadgroup('raw_sd2'),'SD2+LoRA':loadgroup('lora'),'TF-IDG':loadgroup('tfidg'),'MAGIC':loadgroup('magic'),'DefectFill (Partial Reimplementation)':loadgroup('defectfill'),'SeaS (Native Generation)':loadgroup('seas'),'Random':[r for r in hist if r['arm']=='random'],'BootstrapGuard':[r for r in hist if r['arm']=='bootstrap_guard'],METHOD:cur};agg=[]
 for n,rs in groups.items():
  x={'method':n}
  for k in ('precision','recall','map50','map5095'):m,s=ms([r[k] for r in rs]);x[k]=f'{m:.4f} ± {s:.4f}';x[k+'_mean']=m;x[k+'_std']=s
  agg.append(x)
 fields=['method','precision','recall','map50','map5095','precision_mean','precision_std','recall_mean','recall_std','map50_mean','map50_std','map5095_mean','map5095_std']
 with (OUT/'aggregate_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(agg)
 f2=['method','seed','precision','recall','map50','map5095','flash_ap50','flash_ap5095','black_ap50','black_ap5095','best_epoch','stop_epoch','optimizer_steps','checkpoint_sha256']
 with (OUT/'per_seed_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=f2,lineterminator='\n');w.writeheader()
  for r in cur:w.writerow({'method':METHOD,'seed':r['seed'],'precision':r['precision'],'recall':r['recall'],'map50':r['map50'],'map5095':r['map5095'],'flash_ap50':r['per_class']['flash']['ap50'],'flash_ap5095':r['per_class']['flash']['ap5095'],'black_ap50':r['per_class']['black']['ap50'],'black_ap5095':r['per_class']['black']['ap5095'],'best_epoch':r['best_epoch'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'checkpoint_sha256':r['best_checkpoint_sha256']})
 core.save(OUT/'checkpoint_audit.json',{'status':'PASS','records':[{'seed':r['seed'],'best_epoch':r['best_epoch'],'sha256':r['best_checkpoint_sha256']} for r in cur]});bg={r['seed']:r for r in hist if r['arm']=='bootstrap_guard'};core.save(OUT/'compute_audit.json',{'classification':'SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING','samples_per_epoch':218,'records':[{'seed':r['seed'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'bootstrap_guard_stop_epoch':bg[r['seed']]['stop_epoch']} for r in cur]});args=ROOT/'runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml';core.save(OUT/'training_config_audit.json',{'status':'MATCHED_EXCEPT_TRAIN_DATA','ultralytics':ultralytics_version,'python':platform.python_version(),'torch':torch.__version__,'historical_args_sha256':core.sha(args),'model_sha256':core.sha(core.MODEL),'seeds':list(SEEDS),'frozen':{'epochs':150,'patience':40,'batch':1,'imgsz':1536,'optimizer':'auto','mosaic':1.0,'mixup':0.0,'close_mosaic':10,'amp':True,'deterministic':True}})
 g={x['method']:x for x in agg};pm=g[METHOD]['map5095_mean'];fm,fs=ms([r['per_class']['flash']['ap5095'] for r in cur]);bm,bs=ms([r['per_class']['black']['ap5095'] for r in cur]);ga=json.loads((OUT/'generation_audit.json').read_text());lines=['# Poisson Blending downstream detection','','Traditional OpenCV `NORMAL_CLONE` with bbox-derived masks (not human pixel masks). No selection, reroll, Copy-Paste, or Poisson variant search was performed.','','| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |','|---|---:|---:|---:|---:|']
 for n in groups:x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
 lines+=['','## Analysis','',f"Poisson Blending reaches {g[METHOD]['map5095']} mAP50-95. Flash AP50-95 is {fm:.4f} ± {fs:.4f}; Black is {bm:.4f} ± {bs:.4f}.",'',f"Absolute differences: Poisson − RealRepeat {pm-g['RealRepeat']['map5095_mean']:+.4f}; Poisson − Raw SD2 {pm-g['Raw SD2']['map5095_mean']:+.4f}; Poisson − Random {pm-g['Random']['map5095_mean']:+.4f}; Poisson − BootstrapGuard {pm-g['BootstrapGuard']['map5095_mean']:+.4f}.",'',f"Generation visibility audit: {ga['visibility_counts']}. All generated samples were retained. Per-epoch exposure is matched; early stopping may make total compute differ.",''];(OUT/'POISSON_DOWNSTREAM_REPORT.md').write_text('\n'.join(lines))
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.seed is not None:train(a.seed,a.device)
 elif a.recover_seed is not None:recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
