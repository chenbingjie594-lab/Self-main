"""DWBG-v2: real-manifold gated hard/anchor balanced candidate selection."""
from __future__ import annotations

import argparse, csv, json, math
from collections import Counter
from pathlib import Path
import numpy as np

try:
    from .dwbg_utils import native
    from .dwbg_v2_utils import (consensus_score, flash_boundary_score, geometric_score,
                                greedy_unique, interval_typicality)
    from .select_dwbg_candidates import (add_synthetic, copy_real, validation_hash, write_yaml,
                                         weakness_lookup, moderate_difficulty_score)
except ImportError:
    from dwbg_utils import native
    from dwbg_v2_utils import (consensus_score, flash_boundary_score, geometric_score,
                               greedy_unique, interval_typicality)
    from select_dwbg_candidates import (add_synthetic, copy_real, validation_hash, write_yaml,
                                        weakness_lookup, moderate_difficulty_score)

NAMES = {0: 'Flash point', 1: 'Big black spots'}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real_root', type=Path, required=True); p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--scored_candidates', type=Path, required=True); p.add_argument('--config', type=Path, default=Path('configs/dwbg_v2.json'))
    p.add_argument('--output_root', type=Path, required=True); p.add_argument('--output_dir', type=Path, default=Path('results/dwbg/v2'))
    p.add_argument('--flash_count', type=int, default=40); p.add_argument('--black_count', type=int, default=40)
    p.add_argument('--v1_manifest', type=Path, default=None,
                   help='Optional v1 manifest for a side-by-side selection summary only.')
    p.add_argument('--disable_manifold', action='store_true'); p.add_argument('--disable_consensus', action='store_true'); p.add_argument('--hard_ratio', type=float, default=None)
    p.add_argument('--seed', type=int, default=42); return p.parse_args()

def real_confidence_intervals(profile):
    output={}
    for cid in (0,1):
        values=[float(x['matched_confidence']) for x in profile['instances'] if int(x['class_id'])==cid and float(x['best_iou'])>=.5]
        if not values: values=[.25]
        output[cid]=(float(np.quantile(values,.25)), float(np.quantile(values,.75)))
    return output

def score(candidate, profile, cfg, intervals, disable_manifold=False, disable_consensus=False):
    cid=int(candidate['class_id']); policy=cfg['flash' if cid==0 else 'black']
    weakness=sum([weakness_lookup(profile,cid,'scale',candidate['scale_bin']), weakness_lookup(profile,cid,'contrast',candidate['contrast_bin']), weakness_lookup(profile,cid,'morphology',candidate['morphology_bin'])])/3
    iou, conf=float(candidate['median_iou']),float(candidate['median_confidence'])
    if cid==0:
        tau=policy.get('decision_threshold') or profile['settings']['match_confidence']
        sigma=policy.get('sigma_boundary') or .5
        hard=flash_boundary_score(conf,iou,tau,sigma,policy['min_iou'])
        boundary_valid=bool(iou>=policy['min_iou'] and conf>=tau)
    else:
        hard=moderate_difficulty_score(conf,iou,policy['difficulty_target'],policy['difficulty_sigma'],policy['min_iou'])
        boundary_valid=bool(iou>=policy['min_iou'])
    manifold=1.0 if disable_manifold else float(candidate.get('manifold_score',0.0))
    consensus=1.0 if disable_consensus else float(candidate.get('consensus_score',0.0))
    typical=interval_typicality(conf,*intervals[cid])
    anchor=manifold*consensus*typical
    final=geometric_score(weakness,hard,manifold,consensus,policy)
    return dict(candidate, scale_weakness=weakness_lookup(profile,cid,'scale',candidate['scale_bin']),
        contrast_weakness=weakness_lookup(profile,cid,'contrast',candidate['contrast_bin']), morphology_weakness=weakness_lookup(profile,cid,'morphology',candidate['morphology_bin']),
        weakness_score=weakness,boundary_score=hard, boundary_valid=boundary_valid, anchor_score=anchor, final_score=final,
        selection_reason=[], fallback_reason=None)

def select_class(rows, count, cid, cfg):
    hard_count=int(round(count*float(cfg['selection']['hard_ratio']))); anchor_count=count-hard_count
    caps=cfg['selection']; valid=[x for x in rows if x['manifold_valid']]
    hard_pool=[x for x in valid if x['boundary_valid'] and (x['boundary_score']>0 if cid==0 else True)]
    hard=greedy_unique(hard_pool,hard_count,'final_score',[],caps['max_per_source'],caps['max_per_seed'])
    fallback=None
    if len(hard)<hard_count and cid==0:
        relaxed=[x for x in valid if x['boundary_valid'] and x['candidate_id'] not in {y['candidate_id'] for y in hard}]
        hard += greedy_unique(relaxed,hard_count-len(hard),'final_score',hard,caps['max_per_source'],caps['max_per_seed'])
        fallback='relaxed_flash_boundary'
    if len(hard)<hard_count: fallback='fallback_from_anchor'
    anchor_pool=[x for x in valid if x['candidate_id'] not in {y['candidate_id'] for y in hard}]
    anchors=greedy_unique(anchor_pool,anchor_count + (hard_count-len(hard)),'anchor_score',hard,caps['max_per_source'],caps['max_per_seed'])
    if len(hard)+len(anchors)!=count: raise ValueError(f'class {cid}: insufficient unique manifold-valid candidates')
    for item in hard: item.update(selection_group='hard',fallback_reason=fallback,selection_reason=['manifold_valid','hard_valid'])
    for item in anchors: item.update(selection_group='anchor',fallback_reason=fallback,selection_reason=['manifold_valid','representative_anchor'])
    return hard+anchors, {'candidate_pool':len(rows),'manifold_rejected':len(rows)-len(valid),'boundary_rejected':len(valid)-len(hard_pool),'hard_selected':len(hard),'anchor_selected':len(anchors),'fallback_reason':fallback}

def summary(items, extra):
    result={}
    for cid in (0,1):
        subset=[x for x in items if int(x['class_id'])==cid]
        result[NAMES[cid]]={**extra[cid], 'median_confidence':float(np.median([x['median_confidence'] for x in subset])), 'median_iou':float(np.median([x['median_iou'] for x in subset])),
            'median_manifold_distance':float(np.median([x['median_normalized_manifold_distance'] for x in subset])), 'std_confidence':float(np.mean([x['std_confidence'] for x in subset])),
            'scale':dict(Counter(x['scale_bin'] for x in subset)), 'contrast':dict(Counter(x['contrast_bin'] for x in subset)), 'morphology':dict(Counter(x['morphology_bin'] for x in subset))}
    return result

def main():
    args=parse_args(); cfg=json.loads(args.config.read_text()); profile=json.loads(args.profile.read_text()); raw=json.loads(args.scored_candidates.read_text())
    if args.output_root.exists() and any(args.output_root.iterdir()): raise FileExistsError(args.output_root)
    if args.output_dir.exists() and (args.output_dir/'dwbg_v2_manifest_40_40.json').exists(): raise FileExistsError(args.output_dir/'dwbg_v2_manifest_40_40.json')
    if args.hard_ratio is not None: cfg['selection']['hard_ratio']=args.hard_ratio
    intervals=real_confidence_intervals(profile); scored=[score(x,profile,cfg,intervals,args.disable_manifold,args.disable_consensus) for x in raw['candidates']]
    if args.disable_manifold:
        for item in scored: item['manifold_valid'] = True
    chosen=[]; details={}
    for cid,count in ((0,args.flash_count),(1,args.black_count)):
        part,details[cid]=select_class([x for x in scored if int(x['class_id'])==cid],count,cid,cfg); chosen+=part
    args.output_root.mkdir(parents=True,exist_ok=True)
    for group in ('real_only','real_dwbg_v2'):
        copy_real(args.real_root,args.output_root/group); write_yaml(args.output_root,group)
    add_synthetic(chosen,args.output_root/'real_dwbg_v2')
    if validation_hash(args.output_root/'real_only')!=validation_hash(args.output_root/'real_dwbg_v2'): raise RuntimeError('Validation split changed')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selected_by_id={x['candidate_id']: x for x in chosen}
    for item in scored:
        if item['candidate_id'] in selected_by_id:
            item['selected']=True; item['selection_group']=selected_by_id[item['candidate_id']]['selection_group']
        else:
            item['selected']=False
    manifest={'version':2,'profile':str(args.profile.resolve()),'scored_candidates':str(args.scored_candidates.resolve()),'config':cfg,'selected':chosen,'counts':{'flash':args.flash_count,'black':args.black_count}}
    (args.output_dir/'dwbg_v2_manifest_40_40.json').write_text(json.dumps(native(manifest),ensure_ascii=False,indent=2),encoding='utf-8')
    report={'DWBG-v2':summary(chosen,details), 'hard_anchor_ratio':{'hard':sum(x['selection_group']=='hard' for x in chosen),'anchor':sum(x['selection_group']=='anchor' for x in chosen)}}
    if args.v1_manifest:
        v1=json.loads(args.v1_manifest.read_text(encoding='utf-8')).get('selected',[])
        report['DWBG-v1']={'Flash': {'count':sum(int(x['class_id'])==0 for x in v1)}, 'Black': {'count':sum(int(x['class_id'])==1 for x in v1)}}
    (args.output_dir/'dwbg_v2_selection_summary.json').write_text(json.dumps(native(report),ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['candidate_id','class_id','scale_bin','contrast_bin','morphology_bin','median_confidence','median_iou','std_confidence','std_iou','weakness_score','boundary_score','manifold_score','consensus_score','anchor_score','final_score','manifold_valid','selected','selection_group']
    with (args.output_dir/'all_candidates_scored.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows([{k:x.get(k) for k in fields}|{'selection_group':x.get('selection_group','')} for x in scored])
    print('DWBG-v2 Selection Summary'); print(json.dumps(native(report),ensure_ascii=False,indent=2)); print(f'Dataset: {args.output_root}')

if __name__=='__main__': main()
