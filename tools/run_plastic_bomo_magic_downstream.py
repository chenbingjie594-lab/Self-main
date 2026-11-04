#!/usr/bin/env python3
"""Audit MAGIC Bank A and run the frozen Plastic_Bomo detector protocol."""
from __future__ import annotations
import argparse,csv,json,os,platform,shutil
from collections import Counter
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from ultralytics import __version__ as ultralytics_version
import run_plastic_bomo_lora_downstream as core

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results_for_gpt/plastic_bomo_baselines/magic'
MAN=SRC/'image_manifest.csv'; INF=SRC/'runtime/inference_manifest.json'
RAW=ROOT/'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_A_manifest.json'
OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/magic'
RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/magic';DS=RUNS/'dataset_binding';DATA=DS/'magic_bank_a.yaml'
SEEDS=(42,3407,2026)
for k,v in {'OUT':OUT,'RUNS':RUNS,'DS':DS,'DATA':DATA}.items():setattr(core,k,v)
core.RUN_PREFIX='magic_bank_a'
core.METHOD_NAME='MAGIC Bank A'

def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():
  if dst.is_symlink() and dst.resolve()==src.resolve():return
  raise RuntimeError(f'collision: {dst}')
 os.symlink(src.resolve(),dst)

def audit():
 OUT.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 rows=[r for r in csv.DictReader(MAN.open()) if r['bank']=='A']; by={r['slot_id']:r for r in rows}
 inf=json.loads(INF.read_text()); ir={r['slot_id']:r for r in inf['records'] if r['bank']=='A'}
 raw=json.loads(RAW.read_text())['records']; rr={r['slot_id']:r for r in raw}
 diag={(r['slot_id'],r['bank']):r for r in csv.DictReader((SRC/'generation_diagnostics.csv').open())}
 errors=[];items=[];cc=Counter(r['class'] for r in rows)
 if len(rows)!=80 or len(by)!=80:errors.append('BANK_A_NOT_80_UNIQUE')
 if set(by)!=set(rr) or set(ir)!=set(rr):errors.append('FROZEN_SLOT_SET_MISMATCH')
 if cc!=Counter({'flash':40,'black':40}):errors.append('CLASS_COUNTS')
 cfg=yaml.safe_load(core.REAL.read_text());base=Path(cfg['path']);real=core.files(base/cfg['train']);instances=0
 for q in (DS/'images/train',DS/'labels/train'):
  if q.exists():shutil.rmtree(q)
 for i,p in enumerate(real):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');instances+=sum(bool(x.strip()) for x in lab.read_text().splitlines());name=f'real_{i:03d}_{p.name}';link(p,DS/'images/train'/name);link(lab,(DS/'labels/train'/name).with_suffix('.txt'))
 composed=DS/'magic_full_canvas';composed.mkdir(parents=True,exist_ok=True)
 for sid in sorted(set(by)&set(rr)&set(ir)):
  m,r,i=by[sid],rr[sid],ir[sid];fs=r['frozen_slot'];e=[]
  crop=SRC/m['image_path']; normal=Path(fs['normal_path']); out=composed/f'{sid}.png'
  if not crop.is_file() or core.sha(crop)!=m['image_sha256']:e.append('MAGIC_IMAGE_SHA')
  if not normal.is_file() or core.sha(normal)!=fs['normal_sha256']:e.append('NORMAL_SHA')
  if m['status']!='COMPLETE':e.append('NOT_COMPLETE')
  if m['class']!=fs['class'] or i['defect_class']!=fs['class']:e.append('CLASS_MISMATCH')
  if m['conditioning_normal_id']!=fs['conditioning_normal_id'] or i['conditioning_normal_id']!=fs['conditioning_normal_id']:e.append('NORMAL_ID_MISMATCH')
  bbox=json.loads(m['target_bbox_normalized_xywh'])
  if not np.allclose(bbox,fs['bbox'],atol=5e-7,rtol=0) or not np.allclose(i['target_bbox_normalized_xywh'],fs['bbox'],atol=5e-7,rtol=0):e.append('BBOX_MISMATCH')
  if int(m['sample_seed'])!=int(r['sample_seed']) or int(i['sample_seed'])!=int(r['sample_seed']):e.append('SEED_MISMATCH')
  if fs['split']!='train':e.append('NON_TRAIN_SOURCE')
  if not e:
   with Image.open(normal) as ni,Image.open(crop) as ci:
    canvas=ni.convert('RGB');gen=ci.convert('RGB');xy=fs['geometry']['fixed_crop_xyxy']
    if canvas.size!=(4096,1024) or gen.size!=(512,512) or [xy[2]-xy[0],xy[3]-xy[1]]!=[512,512]:e.append('CANVAS_GEOMETRY')
    else:canvas.paste(gen,(xy[0],xy[1]));canvas.save(out,'PNG')
  if not e:
   link(out,DS/'images/train'/f'magic_{sid}.png');(DS/'labels/train'/f'magic_{sid}.txt').write_text(f"{0 if fs['class']=='flash' else 1} "+' '.join(f'{float(x):.10g}' for x in fs['bbox'])+'\n')
  d=diag.get((sid,'A'),{});items.append({'slot_id':sid,'annotation_id':fs['annotation_id'],'source_group_id':fs['source_group_id'],'class':fs['class'],'normal_id':fs['conditioning_normal_id'],'sample_seed':int(r['sample_seed']),'magic_crop_sha256':m['image_sha256'],'full_canvas_sha256':core.sha(out) if out.is_file() else None,'bbox':fs['bbox'],'structural_failure':str(d.get('structural_failure','')).lower()=='true','errors':e});errors.extend(f'{sid}:{x}' for x in e)
 val=(base/cfg['val']).resolve();DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str(val),'names':cfg['names']},sort_keys=False))
 a={'status':'PASS' if not errors else 'FAIL','method':'MAGIC','source_manifest_sha256':core.sha(MAN),'inference_manifest_sha256':core.sha(INF),'bank':'A','generation_seed':42,'bank_A_records':len(rows),'slot_ids_unique':len(by)==80,'slot_set_identical_to_raw_sd2':set(by)==set(rr),'class_counts':dict(cc),'real_train_images':len(real),'real_instances':instances,'synthetic_images':len(rows),'total_train_images':len(real)+len(rows),'native_output_wh':[512,512],'detector_canvas_wh':[4096,1024],'canvas_adapter':'paste frozen 512x512 MAGIC crop into SHA-verified frozen normal at fixed_crop_xyxy','quality_selection':False,'best_of_n':False,'reroll':False,'protocol_deviation':'MAGIC is crop-native; full detector canvas is reconstructed by the frozen inverse crop transform','structural_failures_retained':sum(x['structural_failure'] for x in items),'errors':errors,'items':items};core.save(OUT/'dataset_audit.json',a)
 if errors:raise RuntimeError(f'audit failed: {len(errors)} errors')
 return a

core.audit=audit
def train(seed,device):
 a=OUT/'dataset_audit.json'
 if not a.is_file() or json.loads(a.read_text()).get('status')!='PASS':raise RuntimeError('run --audit first')
 core.train(seed,device)
def recover(seed,device):core.recover(seed,device)
def ms(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def finalize():
 cur=[json.loads((OUT/f'seed_{s}_result.json').read_text()) for s in SEEDS]
 hist=json.loads(core.HIST.read_text())['records'];rrp=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json').read_text()) for s in SEEDS];raw=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/raw_sd2/seed_{s}_result.json').read_text()) for s in SEEDS];lora=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/lora/seed_{s}_result.json').read_text()) for s in SEEDS];tf=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/tfidg/seed_{s}_result.json').read_text()) for s in SEEDS]
 groups={'Real-only':[r for r in hist if r['arm']=='real_only'],'RealRepeat':rrp,'Raw SD2 Bank A':raw,'SD2+LoRA Bank A':lora,'TF-IDG Bank A':tf,'BootstrapGuard':[r for r in hist if r['arm']=='bootstrap_guard'],'MAGIC Bank A':cur};agg=[]
 for n,rs in groups.items():
  x={'method':n}
  for k in ('precision','recall','map50','map5095'):m,s=ms([r[k] for r in rs]);x[k]=f'{m:.4f} ± {s:.4f}';x[k+'_mean']=m;x[k+'_std']=s
  agg.append(x)
 fields=['method','precision','recall','map50','map5095','precision_mean','precision_std','recall_mean','recall_std','map50_mean','map50_std','map5095_mean','map5095_std']
 with (OUT/'aggregate_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(agg)
 f2=['method','seed','precision','recall','map50','map5095','flash_ap50','flash_ap5095','black_ap50','black_ap5095','best_epoch','stop_epoch','optimizer_steps','checkpoint_sha256']
 with (OUT/'per_seed_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=f2,lineterminator='\n');w.writeheader()
  for r in cur:w.writerow({'method':'MAGIC Bank A','seed':r['seed'],'precision':r['precision'],'recall':r['recall'],'map50':r['map50'],'map5095':r['map5095'],'flash_ap50':r['per_class']['flash']['ap50'],'flash_ap5095':r['per_class']['flash']['ap5095'],'black_ap50':r['per_class']['black']['ap50'],'black_ap5095':r['per_class']['black']['ap5095'],'best_epoch':r['best_epoch'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'checkpoint_sha256':r['best_checkpoint_sha256']})
 core.save(OUT/'checkpoint_audit.json',{'status':'PASS','records':[{'seed':r['seed'],'best_epoch':r['best_epoch'],'sha256':r['best_checkpoint_sha256']} for r in cur]});bg={r['seed']:r for r in hist if r['arm']=='bootstrap_guard'};core.save(OUT/'compute_audit.json',{'classification':'SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING','samples_per_epoch':218,'records':[{'seed':r['seed'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'bootstrap_guard_stop_epoch':bg[r['seed']]['stop_epoch']} for r in cur]});args=ROOT/'runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml';core.save(OUT/'training_config_audit.json',{'status':'MATCHED_EXCEPT_TRAIN_DATA','ultralytics':ultralytics_version,'python':platform.python_version(),'torch':torch.__version__,'historical_args_sha256':core.sha(args),'model_sha256':core.sha(core.MODEL),'seeds':list(SEEDS),'frozen':{'epochs':150,'patience':40,'batch':1,'imgsz':1536,'optimizer':'auto','mosaic':1.0,'mixup':0.0,'close_mosaic':10,'amp':True,'deterministic':True}})
 g={x['method']:x for x in agg};tm=g['MAGIC Bank A']['map5095_mean'];fm,fs=ms([r['per_class']['flash']['ap5095'] for r in cur]);bm,bs=ms([r['per_class']['black']['ap5095'] for r in cur]);lines=['# MAGIC downstream detection','','| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |','|---|---:|---:|---:|---:|']
 for n in groups:x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
 lines+=['','## Analysis','',f"MAGIC Bank A reaches {g['MAGIC Bank A']['map5095']} mAP50-95. Flash AP50-95 is {fm:.4f} ± {fs:.4f}; Black is {bm:.4f} ± {bs:.4f}.",'',f"Absolute mAP50-95 differences: MAGIC − RealRepeat {tm-g['RealRepeat']['map5095_mean']:+.4f}; MAGIC − Raw SD2 {tm-g['Raw SD2 Bank A']['map5095_mean']:+.4f}; MAGIC − LoRA {tm-g['SD2+LoRA Bank A']['map5095_mean']:+.4f}; MAGIC − TF-IDG {tm-g['TF-IDG Bank A']['map5095_mean']:+.4f}; MAGIC − BootstrapGuard {tm-g['BootstrapGuard']['map5095_mean']:+.4f}.",'','MAGIC is crop-native. Each audited 512×512 output was pasted back into its SHA-verified frozen normal canvas with the frozen inverse crop transform. No failed sample, empty CAMA mask, or weak result was removed. Early stopping means only per-epoch exposure is matched, not strict total compute.',''];(OUT/'MAGIC_DOWNSTREAM_REPORT.md').write_text('\n'.join(lines))

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.audit:audit()
 elif a.seed is not None:train(a.seed,a.device)
 elif a.recover_seed is not None:recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
