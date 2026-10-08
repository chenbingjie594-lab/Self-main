"""Reproduce union masks and label transforms without generation, scoring or label edits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_plastic_bomo_stage17a_trainonly import REPO, load, save, sha, recovered_functions
from freeze_plastic_bomo_stage17a_slots import project_box


def check_coverage(slots, sources, annotations, source_root):
    base = REPO / 'results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability'
    functions = recovered_functions(base / 'historical_source_recovery/recovered_historical_crop_source.txt',
                                    base / 'historical_isolation_recovery/recovered_historical_mask_source.txt')
    by_source = {r['source_id']: r for r in sources}
    records = []
    for slot in slots:
        src = by_source[slot['source_id']]
        target = next(r for r in annotations if r['donor_id'] == src['annotation_id'])
        relatives = [r for r in annotations if r['image_id'] == src['real_parent'] and r['class'] == src['class']]
        w, h = target['image_size']
        info = functions['centered_square_crop_box'](w, h, *target['bbox_xywh'], 512, 4.)
        if list(info[3:]) != src['crop_center_xy'] or info[2] != src['crop_size']:
            raise RuntimeError('FROZEN_CROP_TRANSFORM_CHANGED')
        union = np.zeros((h, w), dtype=bool)
        contributing, unlabelled = [], []
        expected_labels = [project_box(r, src, [w, h]) for r in relatives]
        expected_labels = [r for r in expected_labels if r['intersects_output']]
        labelled_ids = {r['annotation_id'] for r in expected_labels}
        for r in relatives:
            local, (x0, y0, x1, y1) = functions['regular_ellipse_mask'](w, h, *r['bbox_xywh'], 2.8 if src['class'] == 'flash' else 1.45, 5.)
            individual = np.zeros((h, w), dtype=bool); individual[y0:y1, x0:x1] = local
            union |= individual
            crop = functions['crop_with_padding'](Image.fromarray(individual.astype('uint8') * 255), info, 0)
            if np.asarray(crop.resize((512, 512), Image.Resampling.NEAREST)).any():
                contributing.append(r['donor_id'])
                if r['donor_id'] not in labelled_ids: unlabelled.append(r['donor_id'])
        crop = functions['crop_with_padding'](Image.fromarray(union.astype('uint8') * 255), info, 0)
        reproduced = (np.asarray(crop.resize((512, 512), Image.Resampling.NEAREST)) > 127).astype('uint8') * 255
        mask_path = source_root / src['mask_relative']; image_path = source_root / src['image_relative']
        if sha(mask_path) != slot['mask_sha256'] or sha(image_path) != slot['image_sha256']:
            raise RuntimeError('FROZEN_SOURCE_BYTES_CHANGED')
        with Image.open(mask_path) as raw:
            actual = (np.asarray(raw.convert('L').resize((512, 512), Image.Resampling.NEAREST)) > 127).astype('uint8') * 255
        digest = hashlib.sha256(actual.tobytes()).hexdigest()
        target_label = next((r for r in expected_labels if r['annotation_id'] == src['annotation_id']), None)
        checks = {'mask_exact_historical_reproduction': bool(np.array_equal(actual, reproduced)),
                  'preprocessed_mask_hash_matches': digest == slot['preprocessed_mask_sha256'],
                  'all_frozen_labels_match_projection': expected_labels == slot['synthetic_labels'],
                  'target_fully_retained': target_label is not None and not target_label['clipping_applied'],
                  'no_unlabelled_mask_contributors': not unlabelled}
        records.append({'slot_id': slot['slot_id'], 'class': slot['class'], 'checks': checks,
                        'mask_contributing_annotation_ids': contributing,
                        'unlabelled_mask_contributors': unlabelled,
                        'clipped_neighbor_annotation_ids': [r['annotation_id'] for r in expected_labels if r['clipping_applied']],
                        'label_count': len(expected_labels), 'pass': all(checks.values())})
    return {'status': 'SLOT_MASK_LABEL_COVERAGE_PASS' if all(r['pass'] for r in records) else 'SLOT_MASK_LABEL_COVERAGE_FAILED',
            'slot_count': len(records), 'failed_slot_ids': [r['slot_id'] for r in records if not r['pass']],
            'projected_label_counts': dict(Counter({c: sum(r['label_count'] for r in records if r['class'] == c) for c in ('flash', 'black')})),
            'annotation_accuracy_or_generated_defect_identity_proven': False,
            'meaning': 'Frozen geometry and union-mask annotation correspondence only; not generated-image semantic fidelity',
            'labels_modified': False, 'validation_read': False, 'sampling_count': 0, 'records': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('slots', 'prepared', 'frozen', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scope', choices=('SERVER_COVERAGE', 'LOCAL_TEST'), default='SERVER_COVERAGE')
    p.add_argument('--local_source_root', type=Path)
    a = p.parse_args()
    if a.local_source_root and a.scope != 'LOCAL_TEST': raise RuntimeError('NO_LOCAL_OVERRIDE_ON_SERVER')
    status = load(a.slots / 'slot_freeze_status.json')
    path = a.slots / 'generation_slot_manifest.json'
    if sha(path) != status['slot_manifest_sha256']: raise RuntimeError('FROZEN_SLOTS_CHANGED')
    manifest = load(path); prep = load(a.prepared / 'trainonly_preparation_status.json')
    source_path = a.prepared / 'trainonly_source_manifest.json'
    if sha(source_path) != manifest['source_manifest_sha256'] or sha(a.frozen) != prep['frozen_manifest_sha256']:
        raise RuntimeError('SOURCE_OR_FROZEN_ANNOTATION_CHANGED')
    source = load(source_path)
    audit = check_coverage(manifest['slots'], source['records'], load(a.frozen)['records'], a.local_source_root or Path(source['dataset_root']))
    audit.update(scope=a.scope, slot_manifest_sha256=sha(path), formal_generation_authorized=False, detector_training_authorized=False)
    a.output.mkdir(parents=True, exist_ok=False)
    save(a.output / 'slot_mask_label_coverage_audit.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'records'}, indent=2))
    if audit['failed_slot_ids']: raise RuntimeError('SLOT_MASK_LABEL_COVERAGE_FAILED_NO_REPLACEMENT')


if __name__ == '__main__': main()
