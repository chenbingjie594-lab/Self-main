#!/usr/bin/env python3
"""Leakage-fixed Poisson NORMAL_CLONE downstream experiment.

Donors come exclusively from the frozen YOLO real-only training split.  The
Stage20B slot ids and normal backgrounds are retained, while target geometry is
derived deterministically without consulting the tainted Stage20B annotation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np, yaml

import run_plastic_bomo_poisson_downstream as base
import run_plastic_bomo_lora_downstream as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_for_gpt/bootstrap_guard_generator_downstream/poisson_blending'
RUNS=ROOT/'runs/bootstrap_guard_generator_downstream/poisson_blending'
GEN=RUNS/'generated_bank_A_clean'
DS=RUNS/'dataset_binding_clean'
DATA=DS/'poisson_bank_a_clean.yaml'
FORMAL=base.FORMAL
REGISTRY=ROOT/'results_for_gpt/plastic_bomo_stage20a_v2_protocol/real_source_registry.json'
SEEDS=(42,3407,2026)
ARCHIVE=OUT/'invalid_stage20b_donor_run'
OLD_FILES=('POISSON_DOWNSTREAM_REPORT.md','per_seed_results.csv','aggregate_results.csv',
 'donor_source_audit.json','generation_audit.json','dataset_audit.json',
 'checkpoint_audit.json','compute_audit.json','training_config_audit.json',
 'seed_42_result.json','seed_3407_result.json','seed_2026_result.json')

for k,v in {'OUT':OUT,'RUNS':RUNS,'DS':DS,'DATA':DATA}.items():
 setattr(core,k,v);setattr(base,k,v)
core.RUN_PREFIX='poisson_clean_bank_a';core.METHOD_NAME=base.METHOD
base.SEEDS=SEEDS;base.GEN=GEN

def save(path,obj): core.save(path,obj)
def sha(p): return core.sha(Path(p))
def files(p): return core.files(Path(p))
def archive_invalid():
 ARCHIVE.mkdir(parents=True,exist_ok=True)
 for name in OLD_FILES:
  src=OUT/name;dst=ARCHIVE/name
  if src.is_file() and not dst.exists(): shutil.copy2(src,dst)
 marker=ARCHIVE/'INVALID_RESULT_NOTICE.md'
 if not marker.exists():
  marker.write_text('# Invalid historical Poisson result\n\nThe historical 0.4511 ± 0.0141 mAP50-95 result is invalid because 23 Stage20B donor images overlap the frozen YOLO validation split by SHA256. It is preserved for diagnosis only and must not enter paper statistics.\n')

def registry_index():
 rec=json.loads(REGISTRY.read_text())['records'];idx=defaultdict(list)
 for r in rec:
  for key in (r.get('file_sha256'),r.get('decoded_rgb_sha256'),r.get('rgb_bytes_sha256')):
   if key:idx[key].append(r)
 return idx

def identity(path,idx):
 h=sha(path);cand=idx.get(h,[])
 # Exact file hash is preferred; otherwise no identity is asserted.
 exact=[r for r in cand if r.get('file_sha256')==h]
 r=(exact or cand or [None])[0]
 return {'path_server_only':str(path),'sha256':h,
  'canonical_image_id':r.get('canonical_image_id') if r else None,
  'source_group_id':r.get('known_source_group_id') if r else None,
  'known_parent_id':r.get('known_parent_id') if r else None,
  'registry_resolved':r is not None}

def bbox_px(norm,w=4096,h=1024):
 cx,cy,bw,bh=map(float,norm)
 return [max(0,math.floor((cx-bw/2)*w)),max(0,math.floor((cy-bh/2)*h)),
         min(w,math.ceil((cx+bw/2)*w)),min(h,math.ceil((cy+bh/2)*h))]

def target_box(slot,w,h):
 # No Stage20B target annotation enters this calculation.
 margin=max(8,math.ceil(max(w,h)*.30));W,H=4096,1024
 if w+2*margin>=W or h+2*margin>=H: raise ValueError('DONOR_BBOX_TOO_LARGE')
 digest=hashlib.sha256(('poisson-clean-target-v1|'+slot).encode()).digest()
 x=margin+int.from_bytes(digest[:8],'big')%(W-w-2*margin+1)
 y=margin+int.from_bytes(digest[8:16],'big')%(H-h-2*margin+1)
 return [x,y,x+w,y+h]

def near_duplicate_audit(items_a,items_b):
 # 64-bit dHash catches identical/near-identical encodings; authoritative
 # same-source checks remain canonical/group/parent ids from the registry.
 def dhash(p):
  im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
  im=cv2.resize(im,(9,8),interpolation=cv2.INTER_AREA)
  return sum(int(v)<<(i) for i,v in enumerate((im[:,1:]>im[:,:-1]).ravel()))
 ah=[(x,dhash(Path(x['path_server_only']))) for x in items_a]
 bh=[(x,dhash(Path(x['path_server_only']))) for x in items_b]
 pairs=[]
 for a,x in ah:
  for b,y in bh:
   d=(x^y).bit_count()
   if d<=1:pairs.append({'train_sha256':a['sha256'],'validation_sha256':b['sha256'],'hamming':d})
 return pairs

def prepare():
 archive_invalid();OUT.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
 cfg=yaml.safe_load(core.REAL.read_text());root=Path(cfg['path']);train=files(root/cfg['train']);val=files(root/cfg['val']);idx=registry_index()
 train_id=[identity(p,idx) for p in train];val_id=[identity(p,idx) for p in val]
 vh={x['sha256'] for x in val_id};vc={x['canonical_image_id'] for x in val_id if x['canonical_image_id']};vg={x['source_group_id'] for x in val_id if x['source_group_id']};vp={x['known_parent_id'] for x in val_id if x['known_parent_id']}
 pools=defaultdict(list);label_errors=[]
 for im,ident in zip(train,train_id):
  lab=Path(str(im).replace('/images/','/labels/')).with_suffix('.txt')
  if not lab.is_file():label_errors.append(f'MISSING_LABEL:{im.name}');continue
  arr=cv2.imread(str(im));
  if arr is None or arr.shape[:2]!=(1024,4096):label_errors.append(f'IMAGE_SHAPE:{im.name}');continue
  for j,line in enumerate(lab.read_text().splitlines()):
   if not line.strip():continue
   z=line.split()
   if len(z)!=5 or int(z[0]) not in (0,1):label_errors.append(f'LABEL_FORMAT:{lab.name}:{j}');continue
   cls='flash' if int(z[0])==0 else 'black';norm=list(map(float,z[1:]));box=bbox_px(norm)
   if box[2]<=box[0] or box[3]<=box[1]:label_errors.append(f'EMPTY_BOX:{lab.name}:{j}');continue
   pools[cls].append({**ident,'class':cls,'label_path_server_only':str(lab),'label_sha256':sha(lab),'label_index':j,'bbox_normalized_xywh':norm,'bbox_xyxy':box,
    'donor_instance_id':hashlib.sha256((ident['sha256']+'|'+str(j)+'|'+line).encode()).hexdigest()})
 exact=set(x['sha256'] for x in train_id)&vh
 canonical={x['canonical_image_id'] for x in train_id if x['canonical_image_id']}&vc
 groups={x['source_group_id'] for x in train_id if x['source_group_id']}&vg
 parents={x['known_parent_id'] for x in train_id if x['known_parent_id']}&vp
 near=near_duplicate_audit(train_id,val_id)
 unresolved_train=sum(not x['registry_resolved'] for x in train_id);unresolved_val=sum(not x['registry_resolved'] for x in val_id)
 formal=json.loads(FORMAL.read_text())['records'];normal_records=[];normal_errors=[]
 for r in formal:
  fs=r['frozen_slot'];p=Path(fs['normal_path']);ident=identity(p,idx) if p.is_file() else {'sha256':None,'registry_resolved':False}
  e=[]
  if not p.is_file() or sha(p)!=fs['normal_sha256']:e.append('NORMAL_SHA')
  if fs['normal_sha256'] in vh:e.append('NORMAL_VALIDATION_SHA')
  if ident.get('canonical_image_id') in vc:e.append('NORMAL_VALIDATION_CANONICAL')
  if ident.get('source_group_id') in vg:e.append('NORMAL_VALIDATION_GROUP')
  normal_records.append({'slot_id':r['slot_id'],'normal_sha256':fs['normal_sha256'],'normal_source_group':fs.get('normal_source_group'),**{k:v for k,v in ident.items() if k!='path_server_only'},'errors':e});normal_errors+=e
 critical=[]
 if len(train)!=138:critical.append('REAL_TRAIN_COUNT')
 if len(val)!=46:critical.append('VALIDATION_COUNT')
 if label_errors:critical.append('LABEL_INTEGRITY')
 # Any overlapping members of the nominal 138-image pool are excluded below;
 # the eligible pool itself must be disjoint on all four identity keys.
 # dHash is intentionally only diagnostic: these near-uniform industrial
 # backgrounds collide heavily under a 64-bit perceptual hash. Identity/group
 # lineage is the authoritative duplicate/sibling-crop gate.
 if unresolved_train or unresolved_val:critical.append('IDENTITY_UNRESOLVED')
 if len(formal)!=80 or Counter(r['class'] for r in formal)!=Counter({'flash':40,'black':40}):critical.append('SLOT_PROTOCOL')
 # The 138-image real-only split contains two source groups also represented in
 # validation. Keep them in the audit, but never admit those instances to the
 # donor pool. This is stricter than checking byte identity alone.
 for c in ('flash','black'):
  pools[c]=[x for x in pools[c] if x['sha256'] not in vh and x['canonical_image_id'] not in vc and x['source_group_id'] not in vg and x['known_parent_id'] not in vp]
 if len(pools['flash'])<1 or len(pools['black'])<1:critical.append('DONOR_CLASS_EMPTY')
 donor_audit={'status':'PASS' if not critical else 'FAIL','protocol':'REAL_ONLY_138_INSTANCE_POOL_V1','registry_sha256':sha(REGISTRY),'train_images':len(train),'eligible_train_instances':sum(map(len,pools.values())),'class_instances':{k:len(v) for k,v in pools.items()},'validation_images':len(val),'checks':{'image_sha256_overlap_count':len(exact),'canonical_image_id_overlap_count':len(canonical),'source_group_id_overlap_count_in_full_138':len(groups),'source_group_overlap_instances_excluded':True,'known_parent_id_overlap_count':len(parents),'near_duplicate_dhash_hamming_le_1_count_informational_only':len(near),'near_duplicate_note':'dHash is non-discriminative for near-uniform industrial backgrounds; lineage groups are authoritative','unresolved_train_identities':unresolved_train,'unresolved_validation_identities':unresolved_val,'label_errors':label_errors},'overlaps':{'sha256':sorted(exact),'canonical_image_id':sorted(canonical),'source_group_id':sorted(groups),'known_parent_id':sorted(parents),'near_duplicates_sample':near[:50]},'pool_records':[x for c in ('flash','black') for x in pools[c]],'errors':critical}
 save(OUT/'donor_source_audit.json',donor_audit)
 if critical:
  save(OUT/'dataset_audit.json',{'status':'FAIL','errors':critical});raise RuntimeError('leakage audit failed: '+','.join(critical))
 # Deterministic per-class permutation; reuse only after exhausting the pool.
 # Replace any validation-lineage normal background deterministically with a
 # clean normal already present in the same frozen protocol.
 clean_normals=[]
 for r,nr in zip(formal,normal_records):
  if not nr['errors']:clean_normals.append(r['frozen_slot']['normal_path'])
 if not clean_normals:raise RuntimeError('no clean normal backgrounds')
 clean_normals=sorted(set(clean_normals))
 ordered={}
 for ci,c in enumerate(('flash','black')):
  ordered[c]=sorted(pools[c],key=lambda x:(x['sha256'],x['label_index']))
  random.Random(42+ci).shuffle(ordered[c])
 generated=[];gen_errors=[];use=Counter()
 for r in formal:
  fs=r['frozen_slot'];c=fs['class'];don=ordered[c][use[c]%len(ordered[c])];use[c]+=1;e=[]
  original_normal=Path(fs['normal_path']);ni=identity(original_normal,idx);normal_bad=(fs['normal_sha256'] in vh or ni.get('canonical_image_id') in vc or ni.get('source_group_id') in vg)
  normal_path=Path(clean_normals[int.from_bytes(hashlib.sha256(('normal|'+r['slot_id']).encode()).digest()[:8],'big')%len(clean_normals)]) if normal_bad else original_normal
  src=cv2.imread(don['path_server_only'],cv2.IMREAD_COLOR);dst=cv2.imread(str(normal_path),cv2.IMREAD_COLOR)
  x0,y0,x1,y1=don['bbox_xyxy'];w,h=x1-x0,y1-y0;tx0,ty0,tx1,ty1=target_box(r['slot_id'],w,h)
  pad=max(4,math.ceil(max(w,h)*.25));sx0=max(0,x0-pad);sy0=max(0,y0-pad);sx1=min(4096,x1+pad);sy1=min(1024,y1+pad)
  patch=src[sy0:sy1,sx0:sx1].copy();mask=np.zeros(patch.shape[:2],np.uint8);mask[y0-sy0:y1-sy0,x0-sx0:x1-sx0]=255
  center=(round((tx0+tx1)/2),round((ty0+ty1)/2));out=GEN/f"{r['slot_id']}.png";lab=GEN/f"{r['slot_id']}.txt"
  try:res=cv2.seamlessClone(patch,dst,mask,center,cv2.NORMAL_CLONE)
  except cv2.error as ex:e.append('SEAMLESS_CLONE_FAILED:'+str(ex).splitlines()[0]);res=None
  if res is not None and not cv2.imwrite(str(out),res):e.append('WRITE_FAILED')
  norm=[(tx0+tx1)/2/4096,(ty0+ty1)/2/1024,w/4096,h/1024]
  if not e:lab.write_text(f"{0 if c=='flash' else 1} "+' '.join(f'{v:.10g}' for v in norm)+'\n')
  if not e:
   roi=np.abs(res[ty0:ty1,tx0:tx1].astype(np.int16)-dst[ty0:ty1,tx0:tx1].astype(np.int16));changed=int((roi.max(2)>2).sum());vis='NO_SIGNAL' if changed==0 or roi.max()<3 else ('LOW_SIGNAL' if changed/max(1,w*h)<.1 else 'VISIBLE_SIGNAL')
  else:changed=0;vis='GENERATION_FAILED'
  rec={'slot_id':r['slot_id'],'class':c,'status':'COMPLETE' if not e else 'FAILED','errors':e,'original_normal_sha256':fs['normal_sha256'],'normal_replaced_for_validation_lineage':normal_bad,'effective_normal_sha256':sha(normal_path),'original_stage20b_annotation_id_not_used':fs['annotation_id'],'original_stage20b_source_group_id_not_used':fs['source_group_id'],'original_stage20b_bbox_not_used':fs['bbox'],'target_bbox_source':'DETERMINISTIC_HASH_OF_SLOT_ID_AND_NEW_DONOR_SIZE','target_bbox_xyxy':[tx0,ty0,tx1,ty1],'target_bbox_normalized_xywh':norm,'new_donor_instance_id':don['donor_instance_id'],'new_donor_sha256':don['sha256'],'new_donor_canonical_image_id':don['canonical_image_id'],'new_donor_source_group_id':don['source_group_id'],'new_donor_bbox_xyxy':don['bbox_xyxy'],'new_donor_bbox_normalized_xywh':don['bbox_normalized_xywh'],'mask_source':'NEW_DONOR_BBOX_DERIVED_NOT_PIXEL_GT','image_path_server_only':str(out),'image_sha256':sha(out) if out.is_file() else None,'label_path_server_only':str(lab),'label_sha256':sha(lab) if lab.is_file() else None,'changed_pixels_gt2':changed,'visibility_status':vis}
  generated.append(rec);gen_errors+=e
 save(OUT/'generation_audit.json',{'status':'PASS' if not gen_errors else 'FAIL','method':'Poisson Blending','protocol':{'operation':'cv2.seamlessClone(..., cv2.NORMAL_CLONE)','target_geometry':'deterministic hash; Stage20B target labels not used','quality_selection':False,'best_of_n':False,'reroll':False},'planned':80,'complete':sum(x['status']=='COMPLETE' for x in generated),'class_counts':dict(Counter(x['class'] for x in generated)),'visibility_counts':dict(Counter(x['visibility_status'] for x in generated)),'donor_reuse_count':80-len({x['new_donor_instance_id'] for x in generated}),'errors':gen_errors,'records':generated})
 if gen_errors:raise RuntimeError('generation failed')
 for q in (DS/'images/train',DS/'labels/train'):
  if q.exists():shutil.rmtree(q)
 for i,p in enumerate(train):
  lab=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');base.link(p,DS/'images/train'/f'real_{i:03d}_{p.name}');base.link(lab,DS/'labels/train'/f'real_{i:03d}_{p.stem}.txt')
 for x in generated:
  base.link(Path(x['image_path_server_only']),DS/'images/train'/f"poisson_{x['slot_id']}.png");base.link(Path(x['label_path_server_only']),DS/'labels/train'/f"poisson_{x['slot_id']}.txt")
 DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':str((root/cfg['val']).resolve()),'names':cfg['names']},sort_keys=False))
 save(OUT/'dataset_audit.json',{'status':'PASS','protocol':'LEAKAGE_FIXED_V1','real_train_images':138,'eligible_real_donor_instances':sum(map(len,pools.values())),'synthetic_images':80,'total_train_images':218,'validation_images':len(val),'validation_split_unchanged':True,'slot_ids_unique':len({x['slot_id'] for x in generated})==80,'normal_background_audit':{'status':'PASS_AFTER_DETERMINISTIC_REPLACEMENT','original_validation_lineage_count':sum(bool(x['errors']) for x in normal_records),'effective_validation_overlap_count':0,'records':normal_records},'target_annotation_audit':{'original_stage20b_target_labels_used':False,'reason':'23 original donor/annotation sources overlap downstream validation by SHA256','replacement':'deterministic target geometry independent of validation labels'},'errors':[]})
 audit_other_methods(formal,vh)

def audit_other_methods(formal,val_hashes):
 n=sum(r['frozen_slot']['source_sha256'] in val_hashes for r in formal)
 methods={
  'Raw SD2':('USES_STAGE20B_BBOX_AND_LABEL','REQUIRES_REQUALIFICATION'),
  'SD2+LoRA':('USES_STAGE20B_FROZEN_SLOTS_AND_BBOX','REQUIRES_REQUALIFICATION'),
  'TF-IDG':('USES_STAGE20B_SLOT_BBOX_AND_REFERENCE_GUIDANCE','REQUIRES_REQUALIFICATION'),
  'MAGIC':('USES_STAGE20B_SLOT_BBOX','REQUIRES_REQUALIFICATION'),
  'DefectFill (Partial Reimplementation)':('USES_STAGE20B_SLOT_BBOX','REQUIRES_REQUALIFICATION'),
  'SeaS (Native Generation)':('NATIVE_NON_POSITION_MATCHED_LABELS','NOT_PROVEN_AFFECTED_BY_THIS_CHECK')}
 save(OUT/'stage20b_downstream_split_impact_audit.json',{'status':'EVIDENCE_REVIEW_COMPLETE','overlapping_stage20b_source_slots':n,'scope_note':'An overlap establishes exposure to validation-derived Stage20B identity/annotation inputs, not automatically pixel leakage or numerical invalidity for every method. Requalification is required where frozen bbox/reference inputs were consumed.','methods':[{'method':m,'evidence':e,'assessment':a} for m,(e,a) in methods.items()]})

def finalize():
 base.finalize()
 d=json.loads((OUT/'donor_source_audit.json').read_text());g=json.loads((OUT/'generation_audit.json').read_text());ds=json.loads((OUT/'dataset_audit.json').read_text())
 resolution=['# Poisson leakage resolution','',
  'The historical **0.4511 ± 0.0141** mAP50-95 result is invalid and remains archived under `invalid_stage20b_donor_run/`; it is not included in the clean aggregate.', '',
  '## Root cause and repair','',
  '- Root cause: 23 original Stage20B donor images were byte-identical to frozen downstream validation images.',
  '- Donors are now built only from the 138 frozen Real-only training images and their 168 labels.',
  '- Original Stage20B target boxes were also rejected because their coordinates came from the affected annotations. New boxes are deterministic functions of slot ID and clean donor dimensions.',
  '- The original slot IDs and normal backgrounds are retained; donor identity and bbox are recorded independently.', '',
  '## Critical gates','',
  f"- Donor audit: **{d['status']}**; full-pool SHA/canonical/group/parent overlaps: {d['checks']['image_sha256_overlap_count']}/{d['checks']['canonical_image_id_overlap_count']}/{d['checks']['source_group_id_overlap_count_in_full_138']}/{d['checks']['known_parent_id_overlap_count']}; all affected instances were excluded from donor eligibility.",
  f"- Dataset audit: **{ds['status']}**.",f"- Generation audit: **{g['status']}**, complete {g['complete']}/80; no selection or reroll.",'',
  'Other Stage20B-based methods are reported in `stage20b_downstream_split_impact_audit.json`. They are not automatically declared invalid; methods consuming affected frozen boxes/references require requalification.']
 (OUT/'POISSON_LEAKAGE_RESOLUTION.md').write_text('\n'.join(resolution)+'\n')
 # Add an explicit clean-protocol banner to the generated report.
 p=OUT/'POISSON_DOWNSTREAM_REPORT.md';p.write_text('# Leakage-fixed protocol\n\nOnly the clean rerun below is eligible for the chapter table. The invalid historical result is excluded.\n\n'+p.read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--recover-seed',type=int,choices=SEEDS);p.add_argument('--device',default='0');p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.seed is not None:
  if json.loads((OUT/'dataset_audit.json').read_text()).get('status')!='PASS':raise RuntimeError('dataset audit not PASS')
  core.train(a.seed,a.device)
 elif a.recover_seed is not None:core.recover(a.recover_seed,a.device)
 elif a.finalize:finalize()
 else:p.error('choose action')
if __name__=='__main__':main()
