#!/usr/bin/env python3
"""Frozen SD2-Inpainting Attention-LoRA Bank-A downstream detector control."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,shutil
from collections import Counter
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image
from ultralytics import YOLO, __version__ as ultralytics_version

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results_for_gpt/plastic_bomo_baselines/lora'; MAN=SRC/'generation_manifest.json'
RAW_MAN=ROOT/'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543/formal_bank_A_manifest.json'
REAL=Path('/mnt/sda/cbj/dataset/Plastic_Bomo/bootstrap_guard_downstream/real_only.yaml')
MODEL=ROOT/'pretrained/yolo11s.pt'; OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/lora'
RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/lora'; DS=RUNS/'dataset_binding'; DATA=DS/'lora_bank_a.yaml'
HIST=ROOT/'results_for_gpt/bootstrap_guard_downstream/validated_per_seed.json'; SEEDS=(42,3407,2026)
RUN_PREFIX='lora_bank_a'
METHOD_NAME='SD2 Inpainting + LoRA Bank A'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def files(p):return sorted(x.resolve() for x in Path(p).iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():
  if dst.is_symlink() and dst.resolve()==src.resolve():return
  raise RuntimeError(f'Binding collision: {dst}')
 os.symlink(src.resolve(),dst)

def audit():
 OUT.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 m=json.loads(MAN.read_text()); rec=[r for r in m['records'] if r['bank']=='A']; raw=json.loads(RAW_MAN.read_text())['records']
 errors=[]; ids=[r['slot_id'] for r in rec]; rawmap={r['slot_id']:r for r in raw}; cc=Counter(r['class'] for r in rec); items=[]
 if len(rec)!=80:errors.append('BANK_A_COUNT')
 if len(set(ids))!=80:errors.append('SLOT_NOT_UNIQUE')
 if set(ids)!=set(rawmap):errors.append('RAW_SLOT_SET_MISMATCH')
 if cc!={'flash':40,'black':40}:errors.append('CLASS_COUNTS')
 for k,v in {'all_outputs_retained':True,'quality_selection_count':0,'best_of_n':False,'reroll_count':0}.items():
  if m.get(k)!=v:errors.append(f'PROTOCOL_{k}')
 # Build isolated symlink view; originals are never modified.
 if DS.exists():
  for q in (DS/'images/train',DS/'labels/train'):
   if q.exists():shutil.rmtree(q)
 cfg=yaml.safe_load(REAL.read_text()); base=Path(cfg['path']); realimgs=files(base/cfg['train']); realinst=0
 for i,p in enumerate(realimgs):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt'); realinst+=sum(bool(x.strip()) for x in lab.read_text().splitlines())
  name=f'real_{i:03d}_{p.name}';link(p,DS/'images/train'/name);link(lab,(DS/'labels/train'/name).with_suffix('.txt'))
 for r in rec:
  sid,cl=r['slot_id'],r['class']; ip=SRC/r['image_path']; e=[]
  if not ip.is_file():e.append('MISSING_IMAGE')
  actual=sha(ip) if ip.is_file() else None
  if actual!=r.get('image_SHA256'):e.append('IMAGE_SHA_MISMATCH')
  wh=None
  if ip.is_file():
   with Image.open(ip) as im:wh=list(im.size)
   if wh!=[4096,1024]:e.append('NOT_FULL_CANVAS')
  rr=rawmap.get(sid,{}); rb=rr.get('frozen_slot',{}).get('bbox'); b=r.get('bbox_normalized_xywh')
  if rr and (rr.get('class')!=cl or rr.get('frozen_slot',{}).get('annotation_id')!=r.get('annotation_id')):e.append('RAW_IDENTITY_MISMATCH')
  if rb is None or not np.allclose(b,rb,atol=5e-7,rtol=0):e.append('RAW_BBOX_MISMATCH')
  if r.get('quality_selection') is not False or r.get('reroll') is not False:e.append('SELECTION_OR_REROLL')
  if not r.get('finite_pipeline_pixels'):e.append('NONFINITE')
  dst=DS/'images/train'/f'lora_{sid}.png';link(ip,dst)
  (DS/'labels/train'/f'lora_{sid}.txt').write_text(f"{0 if cl=='flash' else 1} "+' '.join(f'{float(x):.10g}' for x in b)+'\n')
  items.append({'slot_id':sid,'annotation_id':r.get('annotation_id'),'class':cl,'image_sha256':actual,'image_wh':wh,'bbox':b,'raw_bbox':rb,'errors':e});errors.extend(f'{sid}:{x}' for x in e)
 val=(base/cfg['val']).resolve(); DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str(val),'names':cfg['names']},sort_keys=False))
 result={'status':'PASS' if not errors else 'FAIL','manifest_sha256':sha(MAN),'manifest_status':m.get('status'),'bank_A_records':len(rec),'slot_ids_unique':len(set(ids))==80,'slot_set_identical_to_raw_sd2':set(ids)==set(rawmap),'class_counts':dict(cc),'real_train_images':len(realimgs),'real_instances':realinst,'synthetic_images':len(rec),'total_train_images':len(realimgs)+len(rec),'full_canvas_wh':[4096,1024],'all_outputs_retained':m.get('all_outputs_retained'),'quality_selection_count':m.get('quality_selection_count'),'best_of_n':m.get('best_of_n'),'reroll_count':m.get('reroll_count'),'data_yaml':str(DATA),'errors':errors,'items':items}
 save(OUT/'dataset_audit.json',result)
 if errors:raise RuntimeError(f'audit failed: {len(errors)} errors')
 return result

def payload(m):
 b=m.box;pc={}
 for i,n in m.names.items():pc['flash' if i==0 else 'black']={'ap50':float(b.ap50[i]),'ap5095':float(b.maps[i]),'precision':float(b.p[i]),'recall':float(b.r[i]),'dataset_name':n}
 return {'precision':float(b.mp),'recall':float(b.mr),'map50':float(b.map50),'map5095':float(b.map),'per_class':pc}
def train(seed,device):
 # Dataset binding is built and audited once before parallel seed launches.
 # Training workers consume it read-only to avoid cross-seed rebuild races.
 a=OUT/'dataset_audit.json'
 if not a.is_file() or json.loads(a.read_text()).get('status')!='PASS':
  raise RuntimeError('Run --audit successfully before launching training')
 run=RUNS/f'{RUN_PREFIX}_s{seed}'
 if run.exists():raise FileExistsError(run)
 YOLO(str(MODEL)).train(data=str(DATA),model=str(MODEL),epochs=150,patience=40,batch=1,imgsz=1536,device=device,workers=4,project=str(RUNS),name=run.name,exist_ok=False,pretrained=True,optimizer='auto',seed=seed,deterministic=True,rect=False,cos_lr=False,close_mosaic=10,amp=True,val=True,mosaic=1.0,mixup=0.0,lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,warmup_epochs=3.,warmup_momentum=.8,warmup_bias_lr=.1,box=7.5,cls=.5,dfl=1.5,nbs=64,hsv_h=.015,hsv_s=.7,hsv_v=.4,degrees=0.,translate=.1,scale=.5,shear=0.,perspective=0.,flipud=0.,fliplr=.5)
 recover(seed,device)
def recover(seed,device):
 run=RUNS/f'{RUN_PREFIX}_s{seed}';rawrows=list(csv.DictReader((run/'results.csv').open()));by_epoch={}
 for row in rawrows:
  try:e=int(row['epoch'])
  except (ValueError,TypeError):continue
  by_epoch[e]=row
 rows=[by_epoch[e] for e in sorted(by_epoch)];best,last=run/'weights/best.pt',run/'weights/last.pt';stop=len(rows);be=max(range(stop),key=lambda i:float(rows[i]['metrics/mAP50-95(B)']))+1;ck=torch.load(last,map_location='cpu',weights_only=False);up=ck.get('updates');src='last.pt:updates'
 if up is None:
  laststep=-1;up=0
  for ep in range(stop):
   acc=64
   for i in range(218):
    ni=i+218*ep
    if ni<654:acc=max(1,int(np.interp(ni,[0,654],[1,64]).round()))
    if ni-laststep>=acc:up+=1;laststep=ni
  src='exact_reconstruction_from_frozen_ultralytics_accumulation_schedule'
 v=YOLO(str(best)).val(data=str(DATA),imgsz=1536,batch=1,device=device,workers=4,plots=False);r={'method':METHOD_NAME,'seed':seed,'run_dir':str(run.relative_to(ROOT)),'best_epoch':be,'stop_epoch':stop,'optimizer_steps':int(up),'optimizer_steps_source':src,'best_checkpoint_sha256':sha(best),**payload(v)};save(OUT/f'seed_{seed}_result.json',r);print(json.dumps(r,indent=2))
def ms(xs):return float(np.mean(xs)),float(np.std(xs,ddof=1))
def finalize():
 cur=[json.loads((OUT/f'seed_{s}_result.json').read_text()) for s in SEEDS];hist=json.loads(HIST.read_text())['records'];rr=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json').read_text()) for s in SEEDS];raw=[json.loads((ROOT/f'results_for_gpt/bootstrap_guard_generator_downstream/raw_sd2/seed_{s}_result.json').read_text()) for s in SEEDS]
 groups={'Real-only':[r for r in hist if r['arm']=='real_only'],'RealRepeat':rr,'Raw SD2 Bank A':raw,'BootstrapGuard':[r for r in hist if r['arm']=='bootstrap_guard'],'SD2+LoRA Bank A':cur};agg=[]
 for n,rs in groups.items():
  x={'method':n}
  for k in ('precision','recall','map50','map5095'):m,s=ms([r[k] for r in rs]);x[k]=f'{m:.4f} ± {s:.4f}';x[k+'_mean']=m;x[k+'_std']=s
  agg.append(x)
 fields=['method','precision','recall','map50','map5095','precision_mean','precision_std','recall_mean','recall_std','map50_mean','map50_std','map5095_mean','map5095_std']
 with (OUT/'aggregate_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(agg)
 fields2=['method','seed','precision','recall','map50','map5095','flash_ap50','flash_ap5095','black_ap50','black_ap5095','best_epoch','stop_epoch','optimizer_steps','checkpoint_sha256']
 with (OUT/'per_seed_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields2,lineterminator='\n');w.writeheader()
  for r in cur:w.writerow({'method':r['method'],'seed':r['seed'],'precision':r['precision'],'recall':r['recall'],'map50':r['map50'],'map5095':r['map5095'],'flash_ap50':r['per_class']['flash']['ap50'],'flash_ap5095':r['per_class']['flash']['ap5095'],'black_ap50':r['per_class']['black']['ap50'],'black_ap5095':r['per_class']['black']['ap5095'],'best_epoch':r['best_epoch'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'checkpoint_sha256':r['best_checkpoint_sha256']})
 save(OUT/'checkpoint_audit.json',{'status':'PASS','records':[{'seed':r['seed'],'best_epoch':r['best_epoch'],'sha256':r['best_checkpoint_sha256']} for r in cur]});bg={r['seed']:r for r in hist if r['arm']=='bootstrap_guard'};save(OUT/'compute_audit.json',{'classification':'SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING','samples_per_epoch':218,'records':[{'seed':r['seed'],'stop_epoch':r['stop_epoch'],'optimizer_steps':r['optimizer_steps'],'bootstrap_guard_stop_epoch':bg[r['seed']]['stop_epoch']} for r in cur]})
 args=ROOT/'runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml';save(OUT/'training_config_audit.json',{'status':'MATCHED_EXCEPT_TRAIN_DATA','ultralytics':ultralytics_version,'python':platform.python_version(),'torch':torch.__version__,'historical_args_sha256':sha(args),'model_sha256':sha(MODEL),'seeds':list(SEEDS),'frozen':{'epochs':150,'patience':40,'batch':1,'imgsz':1536,'optimizer':'auto','mosaic':1.0,'mixup':0.0,'close_mosaic':10,'amp':True,'deterministic':True}})
 g={x['method']:x for x in agg};lines=['# SD2 Inpainting + Attention-LoRA downstream detection','','| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |','|---|---:|---:|---:|---:|']
 for n in groups:x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
 lm=g['SD2+LoRA Bank A']['map5095_mean'];lines+=['','## Conclusion','',f"LoRA − Real-only: {lm-g['Real-only']['map5095_mean']:+.4f}; LoRA − Raw SD2: {lm-g['Raw SD2 Bank A']['map5095_mean']:+.4f}; LoRA − RealRepeat: {lm-g['RealRepeat']['map5095_mean']:+.4f}; LoRA − BootstrapGuard: {lm-g['BootstrapGuard']['map5095_mean']:+.4f}. No sample or hyperparameter selection was performed.",''];(OUT/'LORA_DOWNSTREAM_REPORT.md').write_text('\n'.join(lines))
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.audit:audit()
 elif a.seed is not None:train(a.seed,a.device)
 elif a.recover_seed is not None:recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
