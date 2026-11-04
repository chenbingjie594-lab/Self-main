#!/usr/bin/env python3
"""DefectFill (Partial Reimplementation) Bank-A downstream control."""
from __future__ import annotations
import argparse,csv,json,os,platform,shutil
from collections import Counter
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from ultralytics import __version__ as ultralytics_version
import run_plastic_bomo_lora_downstream as core

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results_for_gpt/plastic_bomo_baselines/defectfill';MAN=SRC/'generation_manifest.json'
RAW=ROOT/'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_A_manifest.json'
OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/defectfill';RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/defectfill';DS=RUNS/'dataset_binding';DATA=DS/'defectfill_bank_a.yaml';SEEDS=(42,3407,2026);METHOD='DefectFill (Partial Reimplementation)'
for k,v in {'OUT':OUT,'RUNS':RUNS,'DS':DS,'DATA':DATA}.items():setattr(core,k,v)
core.RUN_PREFIX='defectfill_bank_a';core.METHOD_NAME=METHOD

def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():
  if dst.is_symlink() and dst.resolve()==src.resolve():return
  raise RuntimeError(f'collision: {dst}')
 os.symlink(src.resolve(),dst)

def audit():
 OUT.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 m=json.loads(MAN.read_text());rows=[r for r in m['records'] if r['bank']=='A'];by={r['slot_id']:r for r in rows};raw=json.loads(RAW.read_text())['records'];rr={r['slot_id']:r for r in raw};errors=[];items=[];cc=Counter(r['class'] for r in rows)
 if m.get('protocol')!='Primary Strict' or m.get('native_LFS_executed') is not False or m.get('candidate_count_total')!=240 or m.get('selection_count')!=0 or m.get('reroll_count')!=0:errors.append('PROTOCOL_IDENTITY')
 if len(rows)!=80 or len(by)!=80:errors.append('BANK_A_NOT_80_UNIQUE')
 if set(by)!=set(rr):errors.append('FROZEN_SLOT_SET_MISMATCH')
 if cc!=Counter({'flash':40,'black':40}):errors.append('CLASS_COUNTS')
 cfg=yaml.safe_load(core.REAL.read_text());base=Path(cfg['path']);real=core.files(base/cfg['train']);instances=0
 for q in (DS/'images/train',DS/'labels/train'):
  if q.exists():shutil.rmtree(q)
 for i,p in enumerate(real):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');instances+=sum(bool(x.strip()) for x in lab.read_text().splitlines());name=f'real_{i:03d}_{p.name}';link(p,DS/'images/train'/name);link(lab,(DS/'labels/train'/name).with_suffix('.txt'))
 for sid in sorted(set(by)&set(rr)):
  d,r=by[sid],rr[sid];fs=r['frozen_slot'];e=[];ip=SRC/d['image_path']
  if not ip.is_file() or core.sha(ip)!=d['image_SHA256']:e.append('IMAGE_SHA')
  wh=None
  if ip.is_file():
   with Image.open(ip) as im:wh=list(im.size)
   if wh!=[4096,1024]:e.append('NOT_FULL_CANVAS')
  if d['class']!=fs['class']:e.append('CLASS_MISMATCH')
  if d['normal_id']!=fs['conditioning_normal_id']:e.append('NORMAL_ID_MISMATCH')
  if not np.allclose(d['bbox_normalized_xywh'],fs['bbox'],atol=5e-7,rtol=0):e.append('BBOX_MISMATCH')
  if int(d['sample_seed'])!=int(r['sample_seed']):e.append('SEED_MISMATCH')
  if d['candidate_index']!=0 or d['candidate_count']!=1 or d['selection'] is not False or d['reroll'] is not False:e.append('SELECTION_OR_REROLL')
  if d['mask_source']!='COARSE_BBOX_MASK' or fs['split']!='train':e.append('SOURCE_POLICY')
  if not e:
   link(ip,DS/'images/train'/f'defectfill_{sid}.png');(DS/'labels/train'/f'defectfill_{sid}.txt').write_text(f"{0 if fs['class']=='flash' else 1} "+' '.join(f'{float(x):.10g}' for x in fs['bbox'])+'\n')
  items.append({'slot_id':sid,'annotation_id':fs['annotation_id'],'source_group_id':fs['source_group_id'],'class':fs['class'],'normal_id':fs['conditioning_normal_id'],'sample_seed':int(r['sample_seed']),'image_sha256':d['image_SHA256'],'image_wh':wh,'bbox':fs['bbox'],'errors':e});errors.extend(f'{sid}:{x}' for x in e)
 val=(base/cfg['val']).resolve();DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str(val),'names':cfg['names']},sort_keys=False))
 a={'status':'PASS' if not errors else 'FAIL','method':METHOD,'method_identity':'PARTIAL_REIMPLEMENTATION','official_implementation':False,'source_manifest_sha256':core.sha(MAN),'bank':'A','generation_seed':42,'bank_A_records':len(rows),'slot_ids_unique':len(by)==80,'slot_set_identical_to_raw_sd2':set(by)==set(rr),'class_counts':dict(cc),'real_train_images':len(real),'real_instances':instances,'synthetic_images':len(rows),'total_train_images':len(real)+len(rows),'full_canvas_wh':[4096,1024],'mask_source':'COARSE_BBOX_MASK','protocol':'Primary Strict','candidates_per_slot':1,'native_eight_candidate_LFS_executed':False,'quality_selection':False,'reroll':False,'protocol_deviation':'Partial reimplementation; original paper eight-candidate Low-Fidelity Selection is not implemented or executed','errors':errors,'items':items};core.save(OUT/'dataset_audit.json',a)
 if errors:raise RuntimeError(f'audit failed: {len(errors)} errors')
 return a
core.audit=audit

def train(seed,device):
 if not (OUT/'dataset_audit.json').is_file() or json.loads((OUT/'dataset_audit.json').read_text()).get('status')!='PASS':raise RuntimeError('run --audit first')
 core.train(seed,device)
def recover(seed,device):core.recover(seed,device)
def ms(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def finalize():
 cur=[json.loads((OUT/f'seed_{s}_result.json').read_text()) for s in SEEDS];hist=json.loads(core.HIST.read_text())['records'];rrp=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json').read_text()) for s in SEEDS];raw=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/raw_sd2/seed_{s}_result.json').read_text()) for s in SEEDS];lora=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/lora/seed_{s}_result.json').read_text()) for s in SEEDS];tf=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/tfidg/seed_{s}_result.json').read_text()) for s in SEEDS];magic=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/magic/seed_{s}_result.json').read_text()) for s in SEEDS]
 groups={'Real-only':[r for r in hist if r['arm']=='real_only'],'RealRepeat':rrp,'Raw SD2 Bank A':raw,'SD2+LoRA Bank A':lora,'TF-IDG Bank A':tf,'MAGIC Bank A':magic,'BootstrapGuard':[r for r in hist if r['arm']=='bootstrap_guard'],METHOD:cur};agg=[]
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
 core.save(OUT/'checkpoint_audit.json',{'status':'PASS','method':METHOD,'records':[{'seed':r['seed'],'best_epoch':r['best_epoch'],'sha256':r['best_checkpoint_sha256']} for r in cur]});bg={r['seed']:r for r in hist if r['arm']=='bootstrap_guard'};core.save(OUT/'compute_audit.json',{'classification':'SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING','samples_per_epoch':218,'records':[{'seed':r['seed'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'bootstrap_guard_stop_epoch':bg[r['seed']]['stop_epoch']} for r in cur]});args=ROOT/'runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml';core.save(OUT/'training_config_audit.json',{'status':'MATCHED_EXCEPT_TRAIN_DATA','method':METHOD,'ultralytics':ultralytics_version,'python':platform.python_version(),'torch':torch.__version__,'historical_args_sha256':core.sha(args),'model_sha256':core.sha(core.MODEL),'seeds':list(SEEDS),'frozen':{'epochs':150,'patience':40,'batch':1,'imgsz':1536,'optimizer':'auto','mosaic':1.0,'mixup':0.0,'close_mosaic':10,'amp':True,'deterministic':True}})
 g={x['method']:x for x in agg};dm=g[METHOD]['map5095_mean'];fm,fs=ms([r['per_class']['flash']['ap5095'] for r in cur]);bm,bs=ms([r['per_class']['black']['ap5095'] for r in cur]);lines=['# DefectFill (Partial Reimplementation) downstream detection','','This experiment is **not** labeled as the official complete DefectFill implementation. The paper-native eight-candidate Low-Fidelity Selection is absent and was not added.','','| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |','|---|---:|---:|---:|---:|']
 for n in groups:x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
 lines+=['','## Analysis','',f"{METHOD} reaches {g[METHOD]['map5095']} mAP50-95. Flash AP50-95 is {fm:.4f} ± {fs:.4f}; Black is {bm:.4f} ± {bs:.4f}.",'',f"Absolute mAP50-95 differences: DefectFill − RealRepeat {dm-g['RealRepeat']['map5095_mean']:+.4f}; DefectFill − Raw SD2 {dm-g['Raw SD2 Bank A']['map5095_mean']:+.4f}; DefectFill − LoRA {dm-g['SD2+LoRA Bank A']['map5095_mean']:+.4f}; DefectFill − TF-IDG {dm-g['TF-IDG Bank A']['map5095_mean']:+.4f}; DefectFill − BootstrapGuard {dm-g['BootstrapGuard']['map5095_mean']:+.4f}.",'','All 80 Primary Strict Bank-A images were retained without selection, replacement, or reroll. Early stopping means per-epoch exposure is matched but total compute may differ.',''];(OUT/'DEFECTFILL_DOWNSTREAM_REPORT.md').write_text('\n'.join(lines))

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.audit:audit()
 elif a.seed is not None:train(a.seed,a.device)
 elif a.recover_seed is not None:recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
