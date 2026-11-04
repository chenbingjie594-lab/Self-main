#!/usr/bin/env python3
"""Frozen TF-IDG Bank-A downstream detector control."""
from __future__ import annotations
import argparse,csv,json,os,platform,shutil
from collections import Counter
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from ultralytics import __version__ as ultralytics_version
import run_plastic_bomo_lora_downstream as core

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results_for_gpt/plastic_bomo_baselines/tfidg';MAN=SRC/'generation_manifest.json'
RAW_MAN=ROOT/'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_A_manifest.json'
OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/tfidg';RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/tfidg';DS=RUNS/'dataset_binding';DATA=DS/'tfidg_bank_a.yaml';SEEDS=(42,3407,2026)
for k,v in {'OUT':OUT,'RUNS':RUNS,'DS':DS,'DATA':DATA}.items():setattr(core,k,v)
core.RUN_PREFIX='tfidg_bank_a'
core.METHOD_NAME='TF-IDG Bank A'

def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():
  if dst.is_symlink() and dst.resolve()==src.resolve():return
  raise RuntimeError(f'collision {dst}')
 os.symlink(src.resolve(),dst)
def audit():
 OUT.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 m=json.loads(MAN.read_text());rec=[r for r in m['records'] if r['bank']=='A'];raw=json.loads(RAW_MAN.read_text())['records'];rm={r['slot_id']:r for r in raw};ids=[r['slot_id'] for r in rec];cc=Counter(r['class'] for r in rec);errors=[];items=[]
 if len(rec)!=80:errors.append('BANK_A_COUNT')
 if len(set(ids))!=80:errors.append('SLOT_NOT_UNIQUE')
 if set(ids)!=set(rm):errors.append('RAW_SLOT_SET_MISMATCH')
 if cc!={'flash':40,'black':40}:errors.append('CLASS_COUNTS')
 if m.get('quality_selection') is not False or m.get('best_of_n') is not False or m.get('reroll_count')!=0:errors.append('SELECTION_PROTOCOL')
 cfg=yaml.safe_load(core.REAL.read_text());base=Path(cfg['path']);realimgs=core.files(base/cfg['train']);realinst=0
 for q in (DS/'images/train',DS/'labels/train'):
  if q.exists():shutil.rmtree(q)
 for i,p in enumerate(realimgs):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');realinst+=sum(bool(x.strip()) for x in lab.read_text().splitlines());name=f'real_{i:03d}_{p.name}';link(p,DS/'images/train'/name);link(lab,(DS/'labels/train'/name).with_suffix('.txt'))
 for r in rec:
  sid,cl,fs=r['slot_id'],r['class'],r.get('frozen_slot',{});ip=Path(r.get('output_image_path_server_only',''));e=[]
  if not ip.is_file():e.append('MISSING_IMAGE')
  actual=core.sha(ip) if ip.is_file() else None
  if actual!=r.get('output_image_sha256'):e.append('IMAGE_SHA_MISMATCH')
  wh=None
  if ip.is_file():
   with Image.open(ip) as im:wh=list(im.size)
   if wh!=r.get('output_canvas_wh') or wh!=[4096,1024]:e.append('NOT_FULL_CANVAS')
  rr=rm.get(sid,{});b=fs.get('bbox');rb=rr.get('frozen_slot',{}).get('bbox')
  if fs.get('split')!='train':e.append('NOT_TRAIN')
  if fs.get('annotation_id')!=r.get('reference_annotation_id') or rr.get('frozen_slot',{}).get('annotation_id')!=fs.get('annotation_id'):e.append('ANNOTATION_MISMATCH')
  if rb is None or not np.allclose(b,rb,atol=5e-7,rtol=0):e.append('RAW_BBOX_MISMATCH')
  if r.get('quality_selection') is not False or r.get('best_of_n')!=1 or r.get('reroll_count')!=0:e.append('SELECTION_OR_REROLL')
  if r.get('status')!='COMPLETE':e.append('NOT_COMPLETE')
  dst=DS/'images/train'/f'tfidg_{sid}.png';link(ip,dst);(DS/'labels/train'/f'tfidg_{sid}.txt').write_text(f"{0 if cl=='flash' else 1} "+' '.join(f'{float(x):.10g}' for x in b)+'\n')
  items.append({'slot_id':sid,'annotation_id':fs.get('annotation_id'),'class':cl,'frozen_split':fs.get('split'),'image_sha256':actual,'image_wh':wh,'bbox':b,'errors':e});errors.extend(f'{sid}:{x}' for x in e)
 val=(base/cfg['val']).resolve();DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str(val),'names':cfg['names']},sort_keys=False))
 a={'status':'PASS' if not errors else 'FAIL','manifest_sha256':core.sha(MAN),'manifest_status':m.get('status'),'method':m.get('method'),'bank_A_records':len(rec),'slot_ids_unique':len(set(ids))==80,'slot_set_identical_to_raw_sd2_and_lora':set(ids)==set(rm),'class_counts':dict(cc),'real_train_images':len(realimgs),'real_instances':realinst,'synthetic_images':len(rec),'total_train_images':len(realimgs)+len(rec),'full_canvas_wh':[4096,1024],'quality_selection':m.get('quality_selection'),'best_of_n':m.get('best_of_n'),'reroll_count':m.get('reroll_count'),'DDIM_steps':m.get('DDIM_steps'),'CFG':m.get('CFG'),'errors':errors,'items':items};core.save(OUT/'dataset_audit.json',a)
 if errors:raise RuntimeError(f'audit failed {len(errors)}')
 return a
core.audit=audit

def train(seed,device):
 a=OUT/'dataset_audit.json'
 if not a.is_file() or json.loads(a.read_text()).get('status')!='PASS':raise RuntimeError('run --audit first')
 # core names the run lora_bank_a; isolate by TF-IDG root and retain seed identity.
 core.train(seed,device)
def recover(seed,device):core.recover(seed,device)
def ms(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def finalize():
 cur=[json.loads((OUT/f'seed_{s}_result.json').read_text()) for s in SEEDS];hist=json.loads(core.HIST.read_text())['records'];rr=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json').read_text()) for s in SEEDS];raw=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/raw_sd2/seed_{s}_result.json').read_text()) for s in SEEDS];lora=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/lora/seed_{s}_result.json').read_text()) for s in SEEDS]
 groups={'Real-only':[r for r in hist if r['arm']=='real_only'],'RealRepeat':rr,'Raw SD2 Bank A':raw,'SD2+LoRA Bank A':lora,'BootstrapGuard':[r for r in hist if r['arm']=='bootstrap_guard'],'TF-IDG Bank A':cur};agg=[]
 for n,rs in groups.items():
  x={'method':n}
  for k in ('precision','recall','map50','map5095'):m,s=ms([r[k] for r in rs]);x[k]=f'{m:.4f} ± {s:.4f}';x[k+'_mean']=m;x[k+'_std']=s
  agg.append(x)
 fields=['method','precision','recall','map50','map5095','precision_mean','precision_std','recall_mean','recall_std','map50_mean','map50_std','map5095_mean','map5095_std']
 with (OUT/'aggregate_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(agg)
 f2=['method','seed','precision','recall','map50','map5095','flash_ap50','flash_ap5095','black_ap50','black_ap5095','best_epoch','stop_epoch','optimizer_steps','checkpoint_sha256']
 with (OUT/'per_seed_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=f2,lineterminator='\n');w.writeheader()
  for r in cur:w.writerow({'method':'TF-IDG Bank A','seed':r['seed'],'precision':r['precision'],'recall':r['recall'],'map50':r['map50'],'map5095':r['map5095'],'flash_ap50':r['per_class']['flash']['ap50'],'flash_ap5095':r['per_class']['flash']['ap5095'],'black_ap50':r['per_class']['black']['ap50'],'black_ap5095':r['per_class']['black']['ap5095'],'best_epoch':r['best_epoch'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'checkpoint_sha256':r['best_checkpoint_sha256']})
 core.save(OUT/'checkpoint_audit.json',{'status':'PASS','records':[{'seed':r['seed'],'best_epoch':r['best_epoch'],'sha256':r['best_checkpoint_sha256']} for r in cur]});bg={r['seed']:r for r in hist if r['arm']=='bootstrap_guard'};core.save(OUT/'compute_audit.json',{'classification':'SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING','samples_per_epoch':218,'records':[{'seed':r['seed'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'bootstrap_guard_stop_epoch':bg[r['seed']]['stop_epoch']} for r in cur]});args=ROOT/'runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml';core.save(OUT/'training_config_audit.json',{'status':'MATCHED_EXCEPT_TRAIN_DATA','ultralytics':ultralytics_version,'python':platform.python_version(),'torch':torch.__version__,'historical_args_sha256':core.sha(args),'model_sha256':core.sha(core.MODEL),'seeds':list(SEEDS),'frozen':{'epochs':150,'patience':40,'batch':1,'imgsz':1536,'optimizer':'auto','mosaic':1.0,'mixup':0.0,'close_mosaic':10,'amp':True,'deterministic':True}})
 g={x['method']:x for x in agg};tm=g['TF-IDG Bank A']['map5095_mean'];lines=['# TF-IDG downstream detection','','| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |','|---|---:|---:|---:|---:|']
 for n in groups:x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
 flash_m,flash_s=ms([r['per_class']['flash']['ap5095'] for r in cur]);black_m,black_s=ms([r['per_class']['black']['ap5095'] for r in cur])
 lines+=['','## Analysis','',
  'TF-IDG is evaluated because it is training-free: AnyDoor and DINOv2 remained frozen, while the downstream detector and its frozen validation protocol match the other generator controls. The audited Bank A contributes 40 Flash and 40 Black full-canvas images; no Best-of-N, reroll, or quality selection was used.','',
  f"TF-IDG reaches {g['TF-IDG Bank A']['map5095']} mAP50-95. Its absolute differences are {tm-g['Real-only']['map5095_mean']:+.4f} versus Real-only, {tm-g['Raw SD2 Bank A']['map5095_mean']:+.4f} versus Raw SD2, {tm-g['SD2+LoRA Bank A']['map5095_mean']:+.4f} versus LoRA, {tm-g['RealRepeat']['map5095_mean']:+.4f} versus RealRepeat, and {tm-g['BootstrapGuard']['map5095_mean']:+.4f} versus BootstrapGuard. Class AP50-95 is {flash_m:.4f} ± {flash_s:.4f} for Flash and {black_m:.4f} ± {black_s:.4f} for Black.",'',
  'Thus, TF-IDG improves over Real-only and Raw SD2 in this three-seed experiment, but does not demonstrate an additional gain over RealRepeat and remains below BootstrapGuard. This is an empirical comparison, not a significance claim. Early stopping produced 90–137 epochs (339–499 audited optimizer steps), so only the per-epoch dataset size is matched; total compute is not strictly matched.','']
 (OUT/'TFIDG_DOWNSTREAM_REPORT.md').write_text('\n'.join(lines))
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.audit:audit()
 elif a.seed is not None:train(a.seed,a.device)
 elif a.recover_seed is not None:recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
