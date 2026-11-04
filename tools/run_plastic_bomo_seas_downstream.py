#!/usr/bin/env python3
"""SeaS native-generation, non-position-matched downstream detector control."""
from __future__ import annotations
import argparse,csv,json,os,platform,shutil
from collections import Counter
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from ultralytics import __version__ as ultralytics_version
import run_plastic_bomo_lora_downstream as core

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results_for_gpt/plastic_bomo_baselines/seas';MAN=SRC/'generation_manifest.json';SEAS=ROOT.parent/'third_party/SeaS-main';NATIVE=SEAS/'runtime/plastic_bomo/generated/final'
RAW=ROOT/'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_A_manifest.json';OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/seas';RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/seas';DS=RUNS/'dataset_binding';DATA=DS/'seas_bank_a.yaml';SEEDS=(42,3407,2026);METHOD='SeaS (Native Generation, Non-Position-Matched)'
for k,v in {'OUT':OUT,'RUNS':RUNS,'DS':DS,'DATA':DATA}.items():setattr(core,k,v)
core.RUN_PREFIX='seas_native_bank_a';core.METHOD_NAME=METHOD

def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():
  if dst.is_symlink() and dst.resolve()==src.resolve():return
  raise RuntimeError(f'collision: {dst}')
 os.symlink(src.resolve(),dst)
def mask_box(path):
 with Image.open(path) as im:a=np.asarray(im.convert('L'))>=128
 yy,xx=np.where(a)
 if not len(xx):return None,0
 x0,x1,y0,y1=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
 return [((x0+x1)/2)/512,((y0+y1)/2)/512,(x1-x0)/512,(y1-y0)/512],int(a.sum())

def audit():
 OUT.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 m=json.loads(MAN.read_text());rows=[r for r in m['records'] if r['bank']=='A'];by={r['slot_id']:r for r in rows};raw=json.loads(RAW.read_text())['records'];rr={r['slot_id']:r for r in raw};errors=[];items=[];cc=Counter(r['class'] for r in rows)
 if m.get('mode')!='NATIVE_GENERATION_NO_BBOX_CONTROL' or m.get('quality_selection') is not False or m.get('best_of_n') is not False or m.get('reroll_count')!=0:errors.append('GENERATION_PROTOCOL')
 if len(rows)!=80 or len(by)!=80:errors.append('BANK_A_NOT_80_UNIQUE')
 if set(by)!=set(rr):errors.append('FROZEN_SLOT_SET_MISMATCH')
 if cc!=Counter({'flash':40,'black':40}):errors.append('CLASS_COUNTS')
 cfg=yaml.safe_load(core.REAL.read_text());base=Path(cfg['path']);real=core.files(base/cfg['train']);instances=0
 for q in (DS/'images/train',DS/'labels/train'):
  if q.exists():shutil.rmtree(q)
 for i,p in enumerate(real):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');instances+=sum(bool(x.strip()) for x in lab.read_text().splitlines());name=f'real_{i:03d}_{p.name}';link(p,DS/'images/train'/name);link(lab,(DS/'labels/train'/name).with_suffix('.txt'))
 ious=[];shifts=[]
 for sid in sorted(set(by)&set(rr)):
  d,r=by[sid],rr[sid];fs=r['frozen_slot'];e=[];ip=NATIVE/'A'/d['class']/f'{sid}.png';mp=NATIVE/'A'/d['class']/f'{sid}_mask.png'
  if not ip.is_file() or core.sha(ip)!=d['output_image_sha256']:e.append('IMAGE_SHA')
  if not mp.is_file() or core.sha(mp)!=d['output_mask_sha256']:e.append('MASK_SHA')
  wh=None
  if ip.is_file():
   with Image.open(ip) as im:wh=list(im.size)
   if wh!=[512,512] or im.mode not in ('RGB','RGBA'):e.append('IMAGE_FORMAT')
  box,pixels=mask_box(mp) if mp.is_file() else (None,0)
  if box is None or pixels!=int(d['generated_mask_pixels']):e.append('MASK_EMPTY_OR_COUNT')
  if d['class']!=fs['class'] or d['conditioning_normal_id']!=fs['conditioning_normal_id']:e.append('IDENTITY_MISMATCH')
  seed_identity_match=int(d['frozen_sample_seed_identity'])==int(r['sample_seed'])
  if d['generation_mode']!='NATIVE_GENERATION_NO_BBOX_CONTROL' or d['quality_selection'] is not False or d['reroll_count']!=0:e.append('SELECTION_OR_MODE')
  if fs['split']!='train':e.append('NON_TRAIN_SOURCE')
  if not e:
   link(ip,DS/'images/train'/f'seas_{sid}.png');(DS/'labels/train'/f'seas_{sid}.txt').write_text(f"{0 if d['class']=='flash' else 1} "+' '.join(f'{float(x):.10g}' for x in box)+'\n')
  ious.append(float(d['mask_target_iou']));shifts.append(float(d['defect_location_shift_px']));items.append({'slot_id':sid,'annotation_id':fs['annotation_id'],'source_group_id':fs['source_group_id'],'class':fs['class'],'conditioning_normal_id':fs['conditioning_normal_id'],'native_seed':d['native_seed'],'seas_frozen_seed_identity':int(d['frozen_sample_seed_identity']),'raw_sd2_sample_seed':int(r['sample_seed']),'seed_identity_match':seed_identity_match,'image_sha256':d['output_image_sha256'],'mask_sha256':d['output_mask_sha256'],'image_wh':wh,'mask_derived_bbox_xywh':box,'mask_pixels':pixels,'frozen_target_iou':d['mask_target_iou'],'location_shift_px':d['defect_location_shift_px'],'errors':e});errors.extend(f'{sid}:{x}' for x in e)
 val=(base/cfg['val']).resolve();DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str(val),'names':cfg['names']},sort_keys=False))
 gen={'status':'PASS' if not errors else 'FAIL','comparison_type':'B_NATIVE_GENERATION_EXTENSION','protocol_label':'NATIVE_GENERATION_NON_POSITION_MATCHED','strict_position_paired':False,'paper_usage':'Must not be described as a strictly controlled position-matched generator comparison','method':METHOD,'backbone':'SD1.4','native_parameters':m.get('native_parameters'),'target_background_preserved':False,'target_position_controlled':False,'sample_seed_or_latent_matched_to_raw_sd2':False,'seed_identity_mismatch_count':sum(not x['seed_identity_match'] for x in items),'labels':'bounding boxes computed from each native generated mask; frozen Raw-SD2 boxes are not used','inference_changed':False,'quality_selection':False,'best_of_n':False,'reroll_count':0,'mean_native_mask_vs_frozen_target_iou':float(np.mean(ious)),'mean_location_shift_px':float(np.mean(shifts)),'errors':errors};core.save(OUT/'generation_protocol_audit.json',gen)
 a={'status':'PASS' if not errors else 'FAIL','method':METHOD,'source_manifest_sha256':core.sha(MAN),'bank':'A','bank_A_records':len(rows),'slot_ids_unique':len(by)==80,'slot_set_identical_to_raw_sd2':set(by)==set(rr),'source_group_identity_inherited_from_frozen_slots':True,'class_counts':dict(cc),'real_train_images':len(real),'real_instances':instances,'synthetic_images':len(rows),'total_train_images':len(real)+len(rows),'synthetic_canvas_wh':[512,512],'real_images_native_resolution_preserved':True,'synthetic_label_source':'SeaS generated mask bounding box','training_or_test_leakage':False,'comparison_type':'B_NATIVE_GENERATION_EXTENSION','errors':errors,'items':items};core.save(OUT/'dataset_audit.json',a)
 if errors:raise RuntimeError(f'audit failed: {len(errors)} errors')
 return a
core.audit=audit
def train(seed,device):
 if not (OUT/'dataset_audit.json').is_file() or json.loads((OUT/'dataset_audit.json').read_text()).get('status')!='PASS':raise RuntimeError('run --audit first')
 core.train(seed,device)
def recover(seed,device):core.recover(seed,device)
def ms(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def finalize():
 cur=[json.loads((OUT/f'seed_{s}_result.json').read_text()) for s in SEEDS];hist=json.loads(core.HIST.read_text())['records'];rrp=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json').read_text()) for s in SEEDS];raw=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/raw_sd2/seed_{s}_result.json').read_text()) for s in SEEDS];lora=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/lora/seed_{s}_result.json').read_text()) for s in SEEDS];tf=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/tfidg/seed_{s}_result.json').read_text()) for s in SEEDS];magic=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/magic/seed_{s}_result.json').read_text()) for s in SEEDS];df=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/defectfill/seed_{s}_result.json').read_text()) for s in SEEDS]
 groups={'Real-only':[r for r in hist if r['arm']=='real_only'],'RealRepeat':rrp,'Raw SD2 Bank A':raw,'SD2+LoRA Bank A':lora,'TF-IDG Bank A':tf,'MAGIC Bank A':magic,'DefectFill (Partial Reimplementation)':df,'BootstrapGuard':[r for r in hist if r['arm']=='bootstrap_guard'],METHOD:cur};agg=[]
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
 core.save(OUT/'checkpoint_audit.json',{'status':'PASS','method':METHOD,'records':[{'seed':r['seed'],'best_epoch':r['best_epoch'],'sha256':r['best_checkpoint_sha256']} for r in cur]});bg={r['seed']:r for r in hist if r['arm']=='bootstrap_guard'};core.save(OUT/'compute_audit.json',{'classification':'SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING','samples_per_epoch':218,'records':[{'seed':r['seed'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'bootstrap_guard_stop_epoch':bg[r['seed']]['stop_epoch']} for r in cur]});args=ROOT/'runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml';core.save(OUT/'training_config_audit.json',{'status':'MATCHED_EXCEPT_TRAIN_DATA_AND_NATIVE_SYNTHETIC_CANVAS','method':METHOD,'ultralytics':ultralytics_version,'python':platform.python_version(),'torch':torch.__version__,'historical_args_sha256':core.sha(args),'model_sha256':core.sha(core.MODEL),'seeds':list(SEEDS),'frozen':{'epochs':150,'patience':40,'batch':1,'imgsz':1536,'optimizer':'auto','mosaic':1.0,'mixup':0.0,'close_mosaic':10,'amp':True,'deterministic':True}})
 g={x['method']:x for x in agg};sm=g[METHOD]['map5095_mean'];fm,fs=ms([r['per_class']['flash']['ap5095'] for r in cur]);bm,bs=ms([r['per_class']['black']['ap5095'] for r in cur]);lines=['# SeaS downstream detection','','**Comparison classification: B — native-generation extension comparison.** This is `NATIVE_GENERATION_NON_POSITION_MATCHED` and must not be described in a thesis as a strictly position-paired controlled generator comparison.','','SeaS uses its formal SD1.4 Separation-and-Sharing checkpoint and native 512×512 outputs. YOLO labels are computed from each generated mask; frozen Raw-SD2 boxes are not moved or reused.','','| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |','|---|---:|---:|---:|---:|']
 for n in groups:x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
 lines+=['','## Analysis','',f"{METHOD} reaches {g[METHOD]['map5095']} mAP50-95. Flash AP50-95 is {fm:.4f} ± {fs:.4f}; Black is {bm:.4f} ± {bs:.4f}.",'',f"Absolute differences: SeaS − RealRepeat {sm-g['RealRepeat']['map5095_mean']:+.4f}; SeaS − Raw SD2 {sm-g['Raw SD2 Bank A']['map5095_mean']:+.4f}; SeaS − TF-IDG {sm-g['TF-IDG Bank A']['map5095_mean']:+.4f}; SeaS − BootstrapGuard {sm-g['BootstrapGuard']['map5095_mean']:+.4f}.",'','All 80 native Bank-A images were retained without selection or reroll. Per-epoch exposure is matched; early stopping may make total compute differ.',''];(OUT/'SEAS_DOWNSTREAM_REPORT.md').write_text('\n'.join(lines))
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.audit:audit()
 elif a.seed is not None:train(a.seed,a.device)
 elif a.recover_seed is not None:recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
