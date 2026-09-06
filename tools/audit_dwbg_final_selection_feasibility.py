"""Audit whether the frozen scored pool can satisfy final DWBG-v2 constraints."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    from .dwbg_final_selection import constrained_hard_selection, hard_valid, parent_source_id, quota_counts
except ImportError:
    from dwbg_final_selection import constrained_hard_selection, hard_valid, parent_source_id, quota_counts

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scored_candidates',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    raw=json.loads(args.scored_candidates.read_text(encoding='utf-8')); cfg=json.loads(args.config.read_text(encoding='utf-8')); report={'candidate_pool_frozen':True,'scored_candidates':str(args.scored_candidates)}
    for cid,name in ((0,'flash'),(1,'black')):
        rows=[x for x in raw['candidates'] if int(x['class_id'])==cid]; selected, req, feasible=constrained_hard_selection(rows,28,cid,cfg,seed=args.seed)
        valid=[x for x in rows if hard_valid(x,cid,cfg)]; counts=quota_counts(valid,req)
        data={'hard_valid_total':len(valid),'unique_parent_sources':len({parent_source_id(x) for x in valid}),'unique_seed_count':len({x['seed'] for x in valid}),'constraints':{'hard_count':28,'max_per_source':1,'max_per_seed':8, **{f'{a}_{b}_min':v for (a,b),v in req.items()}},'constrained_selection_feasible':feasible,'constructive_selection_ids':[x['candidate_id'] for x in selected] if feasible else [],'quota_available':{f'{a}_{b}':counts[(a,b)] for a,b in req}}
        if cid==0:
            large=[x for x in valid if x['scale_bin']=='large']; data.update(flash_large_hard_valid_total=len(large),flash_large_unique_parent_sources=len({parent_source_id(x) for x in large}),flash_large_unique_seed_count=len({x['seed'] for x in large}))
        report[name]=data
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
