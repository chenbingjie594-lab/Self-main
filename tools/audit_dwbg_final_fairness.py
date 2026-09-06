"""Strict audit for the frozen DWBG-v2 downstream comparison."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

def digest(root, section, exclude_synthetic=False):
    h=hashlib.sha256(); root=Path(root)/section
    for p in sorted(root.rglob('*')):
        if p.is_file() and not (exclude_synthetic and p.stem.startswith('syn_')):
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()
def by_class(items,cid): return [x for x in items if int(x['class_id'])==cid]
def diversity(items):
    return len(items)==40 and len({x['parent_source_id'] for x in items})==40 and max(Counter(x['parent_source_id'] for x in items).values(),default=0)<=1 and max(Counter(x['seed'] for x in items).values(),default=0)<=8
def cross_platform_basename(value):
    return str(value).replace('\\', '/').rstrip('/').split('/')[-1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset_root',type=Path,required=True); p.add_argument('--dwbg',type=Path,required=True); p.add_argument('--random',type=Path,required=True); p.add_argument('--freeze',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 d=json.loads(a.dwbg.read_text()); r=json.loads(a.random.read_text()); f=json.loads(a.freeze.read_text()); groups=['real_only','real_random_matched','real_dwbg_v2']
 real=[digest(a.dataset_root/g,'images/train',True)+digest(a.dataset_root/g,'labels/train',True) for g in groups]; vals=[digest(a.dataset_root/g,'images/val')+digest(a.dataset_root/g,'labels/val') for g in groups]
 syn={g:{'flash':sum(1 for x in (a.dataset_root/g/'labels/train').glob('syn_*.txt') if x.read_text().split()[0]=='0'),'black':sum(1 for x in (a.dataset_root/g/'labels/train').glob('syn_*.txt') if x.read_text().split()[0]=='1')} for g in groups}
 report={'candidate_pool_frozen':f.get('candidate_pool_frozen') is True,'real_train_identical':len(set(real))==1,'validation_identical':len(set(vals))==1,'validation_has_no_synthetic':all(not list((a.dataset_root/g/'images/val').glob('syn_*')) and not list((a.dataset_root/g/'labels/val').glob('syn_*')) for g in groups),'synthetic_counts':syn,'random_and_dwbg_same_frozen_pool':r.get('candidate_pool_frozen') is True and d.get('candidate_pool_frozen') is True and cross_platform_basename(r.get('pool',''))=='scored_candidates_v2_fixed_v3_expanded.json' and cross_platform_basename(d.get('scored_candidates',''))=='scored_candidates_v2_fixed_v3_expanded.json','leakage':'NOT_VERIFIABLE','leakage_manual_paths':['OOF held-out fold manifests','feature-bank source manifests','real train/val provenance']}
 checks=[]
 for cid,name,need in ((0,'flash',{'scale_bin':('large',13),'contrast_bin':('medium',9),'morphology_bin':('compact',9)}),(1,'black',{'scale_bin':('tiny',13),'contrast_bin':('low',13),'morphology_bin':('elongated',10)})):
  selected=by_class(d['selected'],cid); hard=[x for x in selected if x['selection_group']=='hard']; anchors=[x for x in selected if x['selection_group']=='anchor']; randoms=by_class(r['selected'],cid)
  quotas={f'{k}_{v[0]}':sum(x[k]==v[0] for x in hard) for k,v in need.items()}
  hard_gate=all(x['manifold_valid'] and x['median_iou']>=.5 and (cid or x['median_confidence']>=.05) for x in hard)
  anchor_gate=len(anchors)==12 and all(x['median_iou']>=.5 for x in anchors) and sum(x['median_iou']==0 for x in anchors)==0
  random_gate=len(randoms)==40 and all(x['manifold_valid'] for x in randoms) and diversity(randoms)
  report[name]={'total':len(selected),'hard':len(hard),'anchor':len(anchors),'hard_quotas':quotas,'hard_gate':hard_gate,'anchor_gate':anchor_gate,'random_gate':random_gate,'dwbg_diversity':diversity(selected),'random_unique_parent_sources':len({x['parent_source_id'] for x in randoms}),'dwbg_unique_parent_sources':len({x['parent_source_id'] for x in selected})}
  checks += [len(selected)==40,len(hard)==28,hard_gate,anchor_gate,random_gate,all(quotas[f'{k}_{v[0]}']>=v[1] for k,v in need.items())]
 checks += [report['candidate_pool_frozen'],report['real_train_identical'],report['validation_identical'],report['validation_has_no_synthetic'],syn['real_only']=={'flash':0,'black':0},syn['real_random_matched']=={'flash':40,'black':40},syn['real_dwbg_v2']=={'flash':40,'black':40},report['random_and_dwbg_same_frozen_pool']]
 report['overall']='READY_FOR_DOWNSTREAM_TRAINING' if all(checks) else 'BLOCKED'
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
