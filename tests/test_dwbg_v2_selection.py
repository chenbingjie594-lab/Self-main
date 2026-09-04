import json
import numpy as np

from tools.dwbg_v2_utils import (clip_roi_xyxy, cosine_knn_distance, consensus_score,
    flash_boundary_score, geometric_score, leave_one_out_distances, manifold_reference)
from tools.select_dwbg_candidates_v2 import select_class

def row(i, cid=0, conf=.25, iou=.7, distance=.8, source=None, seed=1):
    return {'candidate_id':str(i),'class_id':cid,'reference_image':source or f's{i}','seed':seed,
      'scale_bin':'tiny','contrast_bin':'low','morphology_bin':'elongated','median_confidence':conf,'median_iou':iou,
      'std_confidence':.01,'std_iou':.01,'manifold_valid':distance<=1,'manifold_score':np.exp(-distance),
      'consensus_score':.9,'boundary_valid':iou>=.5,'boundary_score':.8,'final_score':.8,'anchor_score':.7}

def cfg(): return {'selection':{'hard_ratio':.7,'max_per_source':1,'max_per_seed':8}}

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

def test_invalid_manifold_is_never_selected():
    rows=[row(i,1,distance=.5,source=f's{i}',seed=i) for i in range(40)]+[row('bad',1,distance=2)]
    picked,_=select_class(rows,40,1,cfg())
    assert all(x['manifold_valid'] for x in picked)
