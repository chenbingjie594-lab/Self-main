"""Perform the frozen-pool DWBG-v2 final 40/40 selection (no model training)."""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path
import numpy as np
try:
    from .select_dwbg_candidates_v2 import real_confidence_intervals, score
    from .dwbg_final_selection import constrained_hard_selection, choose_anchors, diversity_report, hard_valid, parent_source_id, quota_counts, quota_requirements
except ImportError:
    from select_dwbg_candidates_v2 import real_confidence_intervals, score
    from dwbg_final_selection import constrained_hard_selection, choose_anchors, diversity_report, hard_valid, parent_source_id, quota_counts, quota_requirements

def args():
    p=argparse.ArgumentParser(); p.add_argument('--scored_candidates',type=Path,required=True); p.add_argument('--profile',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--output_dir',type=Path,required=True); p.add_argument('--seed',type=int,default=42); return p.parse_args()

def stats(items):
    if not items: return {'count':0}
    return {'count':len(items),'median_iou':float(np.median([x['median_iou'] for x in items])),'median_confidence':float(np.median([x['median_confidence'] for x in items])),'median_manifold_score':float(np.median([x['manifold_score'] for x in items])),'iou_zero_count':sum(float(x['median_iou'])==0 for x in items),'iou_lt_05_count':sum(float(x['median_iou'])<.5 for x in items),'fallback_count':sum(x.get('fallback_reason') is not None for x in items),'scale':dict(Counter(x['scale_bin'] for x in items)),'contrast':dict(Counter(x['contrast_bin'] for x in items)),'morphology':dict(Counter(x['morphology_bin'] for x in items))}

def serialise(item, group, reason):
    keys=('candidate_id','class_id','class_name','source_image','reference_image','image_path','label_path','background','scale_bin','contrast_bin','morphology_bin','median_iou','median_confidence','manifold_score','manifold_valid','consensus_score','weakness_score','boundary_score','final_score','fallback_reason','seed')
    out={k:item.get(k) for k in keys}; out['class']=out.pop('class_name'); out['selection_group']=group; out['parent_source_id']=parent_source_id(item); out['selection_reason']=reason; return out

def main():
    a=args(); raw=json.loads(a.scored_candidates.read_text(encoding='utf-8')); profile=json.loads(a.profile.read_text(encoding='utf-8')); cfg=json.loads(a.config.read_text(encoding='utf-8')); cfg['_match_confidence']=float(profile['settings']['match_confidence'])
    intervals=real_confidence_intervals(profile)
    rows=[]
    for item in raw['candidates']:
        row=score(item,profile,cfg,intervals)
        row['target_expansion']='_c' in row['candidate_id']; row['parent_source_id']=parent_source_id(row); rows.append(row)
    all_selected=[]; class_summary={}
    for cid,name in ((0,'flash'),(1,'black')):
        class_rows=[x for x in rows if int(x['class_id'])==cid]
        hard, requirements, feasible=constrained_hard_selection(class_rows,28,cid,cfg,seed=a.seed)
        if not feasible: raise RuntimeError(f'{name} constrained hard selection infeasible; no manifest created')
        anchors=choose_anchors(class_rows,hard,12,cid,cfg)
        if len(anchors)!=12: raise RuntimeError(f'{name} has only {len(anchors)}/12 valid anchors; no manifest created')
        hard_out=[serialise(x,'hard',['manifold_valid','hard_valid','constrained_marginal_quota','utility_rank']) for x in hard]
        anchor_out=[serialise(x,'anchor',['manifold_valid','detection_valid','representative_anchor']) for x in anchors]
        all_selected += hard_out+anchor_out
        qcounts=quota_counts(hard,requirements)
        class_summary[name]={'hard':stats(hard_out),'anchor':stats(anchor_out),'hard_quota':{f'{d}_{b}':{'required':r,'selected':qcounts[(d,b)]} for (d,b),r in requirements.items()},'hard_unique_parent_sources':len({parent_source_id(x) for x in hard}),'hard_unique_seeds':len({x['seed'] for x in hard})}
    if len({x['candidate_id'] for x in all_selected}) != 80: raise RuntimeError('duplicate candidate selected')
    # Random-matched: uniform from the same frozen, manifold-valid pool only.
    rng=random.Random(a.seed); random_out=[]
    for cid,name in ((0,'flash'),(1,'black')):
        pool=[x for x in rows if int(x['class_id'])==cid and x.get('manifold_valid',False)]
        if len(pool)<40: raise RuntimeError(f'{name}: insufficient manifold-valid candidates for random baseline')
        random_out += [serialise(x,'random_matched',['uniform_random_manifold_valid']) for x in rng.sample(pool,40)]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    frozen={'candidate_pool_frozen':True,'scored_candidates':str(a.scored_candidates),'config':cfg,'selected':all_selected,'counts':{'flash':40,'black':40,'hard_per_class':28,'anchor_per_class':12}}
    (a.output_dir/'dwbg_v2_manifest_40_40.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    random_manifest={'candidate_pool_frozen':True,'scored_candidates':str(a.scored_candidates),'seed':a.seed,'selection':'uniform random from manifold_valid candidates; no weakness/boundary/quota ranking','selected':random_out,'counts':{'flash':40,'black':40}}
    (a.output_dir/'random_matched_manifest_40_40_s42.json').write_text(json.dumps(random_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'candidate_pool_frozen':True,'DWBG_v2':class_summary,'hard_anchor_total':{'hard':sum(x['selection_group']=='hard' for x in all_selected),'anchor':sum(x['selection_group']=='anchor' for x in all_selected)}}
    (a.output_dir/'dwbg_v2_selection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    diversity={'candidate_pool_frozen':True,'dwbg_flash':diversity_report([x for x in all_selected if x['class_id']==0]),'dwbg_black':diversity_report([x for x in all_selected if x['class_id']==1]),'random_flash':diversity_report([x for x in random_out if x['class_id']==0]),'random_black':diversity_report([x for x in random_out if x['class_id']==1])}
    (a.output_dir/'source_diversity_report.json').write_text(json.dumps(diversity,ensure_ascii=False,indent=2),encoding='utf-8')
    # Required final assertions.
    for cid in (0,1):
        selected=[x for x in all_selected if x['class_id']==cid]; hard=[x for x in selected if x['selection_group']=='hard']; anchors=[x for x in selected if x['selection_group']=='anchor']; assert len(selected)==40 and len(hard)==28 and len(anchors)==12
        req=quota_requirements(cid,28,cfg); assert all(quota_counts(hard,req)[k]>=v for k,v in req.items()); assert max(Counter(x['parent_source_id'] for x in hard).values())<=1; assert max(Counter(x['seed'] for x in hard).values())<=8; assert sum(x['median_iou']==0 for x in anchors)==0
    print(json.dumps({'summary':summary,'diversity':diversity},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
