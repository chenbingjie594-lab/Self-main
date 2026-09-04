import json
import numpy as np

from tools.dwbg_v2_utils import (clip_roi_xyxy, cosine_knn_distance, consensus_score,
    flash_boundary_score, geometric_score, leave_one_out_distances, manifold_reference)
from tools.select_dwbg_candidates_v2 import select_class, real_confidence_intervals, score

def row(i, cid=0, conf=.25, iou=.7, distance=.8, source=None, seed=1):
    return {'candidate_id':str(i),'class_id':cid,'reference_image':source or f's{i}','seed':seed,
      'scale_bin':'tiny','contrast_bin':'low','morphology_bin':'elongated','median_confidence':conf,'median_iou':iou,
      'std_confidence':.01,'std_iou':.01,'manifold_valid':distance<=1,'manifold_score':np.exp(-distance),
      'consensus_score':.9,'boundary_valid':iou>=.5,'boundary_score':.8,'final_score':.8,'anchor_score':.7}

def cfg():
    return {'selection':{'hard_ratio':.7,'max_per_source':1,'max_per_seed':8,'quota_bonus':.4},
            'flash': {'marginal_quotas': {'scale':{'tiny':.5},'contrast':{'low':.5},'morphology':{'elongated':.5}}},
            'black': {'marginal_quotas': {'scale':{'tiny':.5},'contrast':{'low':.5},'morphology':{'elongated':.5}}}}

def test_roi_is_clipped_and_tiny_nonempty():
    x0,y0,x1,y1=clip_roi_xyxy([-2,-1,.01,.01],100,100,10,10)
    assert 0<=x0<x1<=10 and 0<=y0<y1<=10

def test_cosine_knn_and_leave_one_out_exclude_self():
    bank=np.eye(3,dtype=np.float32)
    assert cosine_knn_distance(bank[0],bank,k=1,exclude_index=0)>0.9
    assert np.isfinite(leave_one_out_distances(bank,k=1)).all()

def test_q95_reference_and_class_separation():
    reference=manifold_reference(np.eye(4,dtype=np.float32),k=2,quantile=.95)
    assert reference['q95_real_distance']>=reference['median_real_distance']
    assert reference['q95_real_distance']>0

def test_consensus_monotonicity():
    assert consensus_score(.01,.01,.1,.1)>consensus_score(.2,.2,.1,.1)

def test_flash_boundary_peaks_at_threshold_and_requires_iou():
    middle=flash_boundary_score(.25,.7,.25,.5,.5)
    assert middle>flash_boundary_score(.99,.7,.25,.5,.5)>0
    assert middle>flash_boundary_score(.01,.7,.25,.5,.5)==0
    assert flash_boundary_score(.25,.49,.25,.5,.5)==0

def test_flash_conf_024_is_valid_two_sided_boundary_not_rejected_by_tau():
    at_tau=flash_boundary_score(.25,.7,.25,.5,.5,.05)
    below_tau=flash_boundary_score(.24,.7,.25,.5,.5,.05)
    assert 0 < below_tau < at_tau
    assert flash_boundary_score(.04,.7,.25,.5,.5,.05)==0

def test_geometric_score_uses_class_specific_exponents():
    a=geometric_score(.5,.5,.5,.5,{'weakness_exp':1,'boundary_exp':1,'manifold_exp':1,'consensus_exp':1})
    b=geometric_score(.5,.5,.5,.5,{'weakness_exp':2,'boundary_exp':1,'manifold_exp':1,'consensus_exp':1})
    assert b<a

def test_hard_anchor_ratio_unique_and_fallback():
    rows=[row(i,0,source=f's{i}',seed=i) for i in range(40)]
    picked, report=select_class(rows,40,0,cfg())
    assert len(picked)==40 and len({x['candidate_id'] for x in picked})==40
    assert sum(x['selection_group']=='hard' for x in picked)==28
    assert sum(x['selection_group']=='anchor' for x in picked)==12
    assert len(json.loads(json.dumps(picked))) == 40

def test_real_anchor_fallback_when_hard_pool_is_truly_insufficient():
    hard=[row(f'h{i}',0,conf=.25,source=f'h{i}',seed=i) for i in range(10)]
    anchors=[row(f'a{i}',0,conf=.04,source=f'a{i}',seed=100+i) for i in range(30)]
    for item in anchors: item.update(boundary_valid=False, boundary_score=0.0)
    picked, report=select_class(hard+anchors,40,0,cfg())
    assert report['fallback_reason']=='fallback_from_anchor'
    assert sum(x['selection_group']=='hard' for x in picked)==10
    assert sum(x['selection_group']=='anchor' for x in picked)==30

def test_hard_selection_uses_class_specific_marginal_quotas():
    local=cfg(); local['selection']['hard_ratio']=1.0
    local['flash']['marginal_quotas']={'scale':{'tiny':.5},'contrast':{'low':.5},'morphology':{'elongated':.5}}
    rows=[row('weak',0,source='w',seed=1), row('a',0,source='a',seed=2),
          row('b',0,source='b',seed=3), row('c',0,source='c',seed=4)]
    rows[1].update(scale_bin='large',contrast_bin='high',morphology_bin='compact')
    rows[2].update(scale_bin='medium',contrast_bin='medium',morphology_bin='medium')
    rows[3].update(scale_bin='large',contrast_bin='high',morphology_bin='compact')
    picked,_=select_class(rows,4,0,local)
    assert any('quota_scale_tiny' in x['selection_reason'] for x in picked)
    assert any('quota_contrast_low' in x['selection_reason'] for x in picked)

def test_anchor_interval_uses_oof_true_positives_only():
    profile={'settings':{'match_confidence':.25},'instances':[
      {'class_id':0,'best_iou':.8,'matched_confidence':.3},
      {'class_id':0,'best_iou':.8,'matched_confidence':.7},
      {'class_id':0,'best_iou':.8,'matched_confidence':.1},  # confidence-only failure
      {'class_id':0,'best_iou':.2,'matched_confidence':.9},  # localization failure
      {'class_id':1,'best_iou':.8,'matched_confidence':.5}, {'class_id':1,'best_iou':.8,'matched_confidence':.6}]}
    low,high=real_confidence_intervals(profile)[0]
    assert .3<=low<=high<=.7

def test_invalid_manifold_is_never_selected():
    rows=[row(i,1,distance=.5,source=f's{i}',seed=i) for i in range(40)]+[row('bad',1,distance=2)]
    picked,_=select_class(rows,40,1,cfg())
    assert all(x['manifold_valid'] for x in picked)
