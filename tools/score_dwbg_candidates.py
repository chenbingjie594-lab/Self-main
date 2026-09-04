"""Compatibility helpers for the DWBG-v1 candidate scorer.

The server-side v1 CLI remains the experiment source of record.  This compact
module restores the shared ensemble aggregation API required by the checked-in
v1 unit tests; DWBG-v2 consumes that v1 scored JSON without altering it.
"""
from __future__ import annotations
import numpy as np

def ensemble_aggregate(rows):
    """Median detector response plus disagreement statistics, JSON-safe floats."""
    if not rows:
        return {'median_iou': 0.0, 'mean_iou': 0.0, 'std_iou': 0.0,
                'median_confidence': 0.0, 'mean_confidence': 0.0, 'std_confidence': 0.0}
    ious=np.asarray([x.get('best_iou',x.get('iou',0.0)) for x in rows],np.float32)
    confs=np.asarray([x.get('matched_confidence',x.get('confidence',0.0)) for x in rows],np.float32)
    return {'median_iou':float(np.median(ious)),'mean_iou':float(np.mean(ious)),'std_iou':float(np.std(ious)),
            'median_confidence':float(np.median(confs)),'mean_confidence':float(np.mean(confs)),'std_confidence':float(np.std(confs))}
