"""Rebuild the frozen-pool Random-matched 40/40 baseline without scoring."""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path
try: from .dwbg_final_selection import parent_source_id, diversity_report
except ImportError: from dwbg_final_selection import parent_source_id, diversity_report

def main():
 p=argparse.ArgumentParser(); p.add_argument('--pool',type=Path,required=True); p.add_argument('--dwbg_manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--diversity_output',type=Path,required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 pool=json.loads(a.pool.read_text(encoding='utf-8')); dwbg=json.loads(a.dwbg_manifest.read_text(encoding='utf-8'))
 rng=random.Random(a.seed); selected=[]
 for cid in (0,1):
  rows=[dict(x) for x in pool['candidates'] if int(x['class_id'])==cid and x.get('manifold_valid',False)]
  rng.shuffle(rows); chosen=[]; sources=set(); seeds=Counter()
  for x in rows:
   parent=parent_source_id(x); seed=int(x['seed'])
   if parent in sources or seeds[seed]>=8: continue
   x.update(parent_source_id=parent,selection_group='random_matched',selection_reason=['uniform_random_manifold_valid'],fallback_reason=None); chosen.append(x); sources.add(parent); seeds[seed]+=1
   if len(chosen)==40: break
  if len(chosen)!=40: raise RuntimeError(f'class {cid}: only {len(chosen)} random valid candidates under diversity caps')
  selected += chosen
 assert len({x['candidate_id'] for x in selected})==80
 manifest={'candidate_pool_frozen':True,'pool':str(a.pool),'dwbg_manifest_sha256':hashlib.sha256(a.dwbg_manifest.read_bytes()).hexdigest(),'seed':a.seed,'selection':'uniform random from manifold_valid only; no utility/quota/difficulty ranking','max_per_parent_source':1,'max_per_seed':8,'selected':selected,'counts':{'flash':40,'black':40}}
 a.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 report={'candidate_pool_frozen':True,'dwbg_flash':diversity_report([x for x in dwbg['selected'] if x['class_id']==0]),'dwbg_black':diversity_report([x for x in dwbg['selected'] if x['class_id']==1]),'random_flash':diversity_report([x for x in selected if x['class_id']==0]),'random_black':diversity_report([x for x in selected if x['class_id']==1])}
 a.diversity_output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
