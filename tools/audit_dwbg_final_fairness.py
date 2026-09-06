"""Audit final frozen DWBG datasets and manifests without training."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
def digest(root,section, exclude_synthetic=False):
 root=Path(root)/section; h=hashlib.sha256()
 for p in sorted(root.rglob('*')):
  if p.is_file() and not (exclude_synthetic and p.stem.startswith('syn_')): h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset_root',type=Path,required=True); p.add_argument('--dwbg',type=Path,required=True); p.add_argument('--random',type=Path,required=True); p.add_argument('--freeze',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 d=json.loads(a.dwbg.read_text()); r=json.loads(a.random.read_text()); f=json.loads(a.freeze.read_text()); groups=['real_only','real_random_matched','real_dwbg_v2']
 real=[digest(a.dataset_root/g,'images/train',True)+digest(a.dataset_root/g,'labels/train',True) for g in groups]; vals=[digest(a.dataset_root/g,'images/val')+digest(a.dataset_root/g,'labels/val') for g in groups]
 def source_ok(xs):
  return len(xs)==40 and len({x['parent_source_id'] for x in xs})==40 and max(Counter(x['parent_source_id'] for x in xs).values())<=1 and max(Counter(x['seed'] for x in xs).values())<=8
 out={'real_train_identical':len(set(real))==1,'validation_identical':len(set(vals))==1,'random_source_diversity':all(source_ok([x for x in r['selected'] if x['class_id']==c]) for c in (0,1)),'dwbg_source_diversity':all(source_ok([x for x in d['selected'] if x['class_id']==c]) for c in (0,1)),'leakage':'NOT_VERIFIABLE','leakage_manual_paths':['OOF held-out fold manifests','feature-bank source manifests','real train/val provenance'],'candidate_pool_frozen':f['candidate_pool_frozen']}
 for c in (0,1):
  s=[x for x in d['selected'] if x['class_id']==c]; h=[x for x in s if x['selection_group']=='hard']; an=[x for x in s if x['selection_group']=='anchor']; out[f'class_{c}']={'total':len(s),'hard':len(h),'anchor':len(an),'hard_gate':all(x['manifold_valid'] and x['median_iou']>=.5 and (c or x['median_confidence']>=.05) for x in h),'anchors_iou_ge_05':all(x['median_iou']>=.5 for x in an)}
 out['overall']='READY_FOR_DOWNSTREAM_TRAINING' if all([out['real_train_identical'],out['validation_identical'],out['random_source_diversity'],out['dwbg_source_diversity']]+[out[f'class_{c}']['hard']==28 and out[f'class_{c}']['anchor']==12 and out[f'class_{c}']['hard_gate'] and out[f'class_{c}']['anchors_iou_ge_05'] for c in (0,1)]) else 'BLOCKED'
 a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
