"""Add detector-feature manifold validity and consensus scores to DWBG candidates."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from .dwbg_feature_extraction import DetectInputExtractor
    from .dwbg_utils import native
    from .dwbg_v2_utils import cosine_knn_distance, consensus_score
except ImportError:
    from dwbg_feature_extraction import DetectInputExtractor
    from dwbg_utils import native
    from dwbg_v2_utils import cosine_knn_distance, consensus_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scored_candidates', type=Path, required=True,
                   help='DWBG-v1 scored_candidates JSON; candidate generation is unchanged.')
    p.add_argument('--bank', nargs=3, action='append', required=True, metavar=('NPZ','METADATA','WEIGHTS'))
    p.add_argument('--config', type=Path, default=Path('configs/dwbg_v2.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0'); p.add_argument('--imgsz', type=int, default=1536)
    return p.parse_args()


def detector_stats(candidate):
    rows = candidate.get('scores_by_detector', [])
    if rows:
        ious = np.asarray([x.get('best_iou', x.get('iou', 0.0)) for x in rows], np.float32)
        confs = np.asarray([x.get('matched_confidence', x.get('confidence', 0.0)) for x in rows], np.float32)
        return float(np.median(ious)), float(np.median(confs)), float(np.std(ious)), float(np.std(confs))
    return (float(candidate.get('median_iou', 0.0)), float(candidate.get('median_confidence', 0.0)),
            float(candidate.get('std_iou', 0.0)), float(candidate.get('std_confidence', 0.0)))


def main():
    args = parse_args(); config = json.loads(args.config.read_text(encoding='utf-8'))
    raw = json.loads(args.scored_candidates.read_text(encoding='utf-8')); candidates = raw.get('candidates', [])
    if args.output.exists(): raise FileExistsError(args.output)
    banks = []
    for fold, (npz_path, meta_path, weights) in enumerate(args.bank):
        data = np.load(npz_path); meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        banks.append((data['features'].astype(np.float32), data['class_ids'].astype(np.int64), meta, Path(weights)))
    if len(banks) != 3: raise ValueError('DWBG-v2 requires exactly three detector-specific banks')
    extractors = [DetectInputExtractor(weights, args.device, args.imgsz) for *_, weights in banks]
    try:
        for candidate in candidates:
            cid = int(candidate['class_id']); image_path = Path(candidate['image_path'])
            with Image.open(image_path) as im: width, height = im.size
            distances, normalized = [], []
            for extractor, (features, class_ids, meta, _) in zip(extractors, banks):
                feature = extractor.encode(image_path, candidate['bbox_xyxy'], width, height)
                real = features[class_ids == cid]
                distance = cosine_knn_distance(feature, real, config['manifold']['k'])
                # v2 banks expose a configurable quantile threshold; q95 is
                # retained separately for diagnostics and backward readability.
                reference = float(meta['references'][str(cid)].get('threshold_real_distance',
                                                                    meta['references'][str(cid)]['q95_real_distance']))
                distances.append(distance); normalized.append(distance / max(reference, 1e-8))
            med_iou, med_conf, std_iou, std_conf = detector_stats(candidate)
            median_norm = float(np.median(normalized)); manifold_score = float(np.exp(-median_norm))
            candidate.update({
                'median_iou': med_iou, 'median_confidence': med_conf,
                'std_iou': std_iou, 'std_confidence': std_conf,
                'manifold_distance_by_detector': distances,
                'normalized_manifold_distance_by_detector': normalized,
                'median_manifold_distance': float(np.median(distances)),
                'mean_manifold_distance': float(np.mean(distances)),
                'std_manifold_distance': float(np.std(distances)),
                'median_normalized_manifold_distance': median_norm,
                'manifold_score': manifold_score,
                'manifold_valid': bool((not config['manifold']['enabled']) or median_norm <= float(config['manifold']['max_normalized_distance'])),
                'consensus_score': consensus_score(std_conf, std_iou, config['consensus']['sigma_conf'], config['consensus']['sigma_iou']) if config['consensus']['enabled'] else 1.0,
            })
    finally:
        for extractor in extractors: extractor.close()
    output = native({'version': 2, 'base_scored_candidates': str(args.scored_candidates.resolve()),
                     'config': config, 'candidates': candidates})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'candidates': len(candidates), 'valid': sum(x['manifold_valid'] for x in candidates),
                      'output': str(args.output.resolve())}, ensure_ascii=False))

if __name__ == '__main__': main()
