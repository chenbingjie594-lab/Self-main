"""Freeze metadata-only 40+40 paired slots; no generation or detector calls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image

from prepare_plastic_bomo_stage17a_trainonly import REPO, load, save, sha
from audit_plastic_bomo_stage17a import sample_seed, baseline_errors


def canonical(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def assign_and_choose(sources, normals):
    """One normal per source first, then the prescribed joint metadata hash order."""
    normals = sorted(normals, key=lambda r: r['normal_filename'])
    if len(normals) != 174 or len({r['normal_filename'] for r in normals}) != 174:
        raise RuntimeError('FROZEN_174_NORMALS_REQUIRED')
    rng = random.Random(42)
    candidates = []
    for source in sorted(sources, key=lambda r: (r['class'], r['source_id'])):
        normal = rng.choice(normals)
        key = [source['class'], source['source_id'], normal['normal_filename'], source['annotation_line_index']]
        candidates.append({'source': source, 'normal': normal, 'slot_id': digest(key), 'canonical_key': key})
    slots = []
    for cls in ('flash', 'black'):
        rows = sorted((r for r in candidates if r['source']['class'] == cls), key=lambda r: (r['slot_id'], canonical(r['canonical_key'])))
        if len(rows) < 40:
            raise RuntimeError('INSUFFICIENT_FROZEN_SOURCE_COUNT_NO_REPLACEMENT')
        slots.extend(rows[:40])
    return candidates, slots


def project_box(row, source, size):
    w, h = size
    cx, cy, bw, bh = row['bbox_xywh']
    crop_size = source['crop_size']
    left = source['crop_center_xy'][0] - crop_size // 2
    top = source['crop_center_xy'][1] - crop_size // 2
    scale = 512. / crop_size
    box = [((cx - bw / 2) * w - left) * scale, ((cy - bh / 2) * h - top) * scale,
           ((cx + bw / 2) * w - left) * scale, ((cy + bh / 2) * h - top) * scale]
    if not all(math.isfinite(x) for x in box):
        raise RuntimeError('NONFINITE_FROZEN_BBOX')
    clipped = [max(0., min(512., x)) for x in box]
    intersects = clipped[2] > clipped[0] and clipped[3] > clipped[1]
    yolo = [(clipped[0]+clipped[2])/1024, (clipped[1]+clipped[3])/1024,
            (clipped[2]-clipped[0])/512, (clipped[3]-clipped[1])/512] if intersects else None
    return {'annotation_id': row['donor_id'], 'class_id': row['class_id'], 'projected_xyxy': box,
            'clipped_xyxy': clipped, 'yolo_xywh': yolo, 'intersects_output': intersects,
            'clipping_applied': box != clipped}


def independence(slots):
    result = {}
    for cls in ('flash', 'black'):
        rows = [r for r in slots if r['class'] == cls]
        reuse = Counter(r['real_parent'] for r in rows)
        normal_reuse = Counter(r['conditioning_id'] for r in rows)
        result[cls] = {'slot_count': len(rows), 'unique_real_source_images': len(reuse),
                       'unique_annotation_instances': len({r['annotation_id'] for r in rows}),
                       'unique_known_source_groups': len({r['source_group_id'] for r in rows if r['source_group_id']}),
                       'unknown_source_group_slots': sum(not r['source_group_id'] for r in rows),
                       'max_source_reuse': max(reuse.values()), 'unique_conditioning_normals': len(normal_reuse),
                       'max_conditioning_reuse': max(normal_reuse.values()),
                       'projected_label_count': sum(len(r['synthetic_labels']) for r in rows)}
    return {'per_class': result, 'slots_are_not_claimed_independent_parents': True,
            'known_source_groups_are_not_proven_physical_acquisition_independence': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'isolated_normals', 'runtime_probe', 'frozen', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scope', choices=('SERVER_FREEZE', 'LOCAL_TEST'), default='SERVER_FREEZE')
    p.add_argument('--local_source_root', type=Path)
    p.add_argument('--local_normal_root', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('NEW_FROZEN_OUTPUT_DIRECTORY_REQUIRED')
    if (a.local_source_root or a.local_normal_root) and a.scope != 'LOCAL_TEST':
        raise RuntimeError('LOCAL_OVERRIDE_FORBIDDEN_ON_SERVER')
    prep = load(a.prepared / 'trainonly_preparation_status.json')
    source_path = a.prepared / 'trainonly_source_manifest.json'
    if sha(source_path) != prep['source_manifest_sha256'] or sha(a.frozen) != prep['frozen_manifest_sha256']:
        raise RuntimeError('FROZEN_SOURCE_OR_ANNOTATIONS_CHANGED')
    source_manifest = load(source_path)
    if source_manifest['per_class_counts'] != {'flash': 88, 'black': 80}:
        raise RuntimeError('SOURCE_CLASS_COUNTS_CHANGED')
    ns = load(a.isolated_normals / 'isolated_normal_pool_status.json')
    normal_path = a.isolated_normals / 'isolated_normal_pool_manifest.json'
    if ns['status'] != 'ISOLATED_NORMAL_POOL_PREPARED' or ns['scope'] != 'SERVER_AUDIT' or sha(normal_path) != ns['pool_manifest_sha256']:
        raise RuntimeError('SERVER_ISOLATED_NORMAL_POOL_REQUIRED')
    normal_manifest = load(normal_path)
    if any(r['related_validation_parents'] for r in normal_manifest['records']):
        raise RuntimeError('NORMAL_SOURCE_GROUP_OVERLAP')
    probe = load(a.runtime_probe / 'runtime_probe_status.json')
    if probe['status'] != 'TRAINONLY_RUNTIME_PROBE_PASS' or not probe['reload_rgb_exact']:
        raise RuntimeError('TRAINONLY_RUNTIME_PASS_REQUIRED')
    baseline_path = REPO / 'configs/baseline_frozen_split70_s42.json'
    baseline = load(baseline_path)
    if baseline_errors(baseline) or sha(baseline_path) != normal_manifest['baseline_inference_config_sha256']:
        raise RuntimeError('BASELINE_INFERENCE_CONFIG_CHANGED')
    source_root = a.local_source_root or Path(source_manifest['dataset_root'])
    normal_root = a.local_normal_root or Path(normal_manifest['pool_root'])
    # Actual source/mask/normal bytes are checked, but never scored.
    for r in source_manifest['records']:
        for field in ('image', 'mask'):
            if sha(source_root / r[field + '_relative']) != r[field + '_sha256']:
                raise RuntimeError('ACTUAL_SOURCE_ASSET_CHANGED')
    for r in normal_manifest['records']:
        if sha(normal_root / r['normal_filename']) != r['normal_sha256']:
            raise RuntimeError('ACTUAL_NORMAL_ASSET_CHANGED')
    candidates, selected = assign_and_choose(source_manifest['records'], normal_manifest['records'])
    annotations = load(a.frozen)['records']
    output, invalid = [], []
    for selected_row in selected:
        src = selected_row['source']; normal = selected_row['normal']
        target = next(r for r in annotations if r['donor_id'] == src['annotation_id'])
        relatives = [r for r in annotations if r['image_id'] == src['real_parent'] and r['class'] == src['class']]
        projected = [project_box(r, src, target['image_size']) for r in relatives]
        target_box = next(r for r in projected if r['annotation_id'] == src['annotation_id'])
        labels = [r for r in projected if r['intersects_output']]
        with Image.open(source_root / src['mask_relative']) as raw:
            if raw.size != (src['crop_size'], src['crop_size']):
                raise RuntimeError('SOURCE_MASK_GEOMETRY_CHANGED')
            mask = np.asarray(raw.convert('L').resize((512, 512), Image.Resampling.NEAREST)) > 127
        valid = target_box['intersects_output'] and not target_box['clipping_applied'] and mask.any() and not mask.all()
        row = {'slot_id': selected_row['slot_id'], 'class': src['class'], 'source_id': src['source_id'],
               'source_group_id': src['source_group_id'], 'real_parent': src['real_parent'], 'annotation_id': src['annotation_id'],
               'conditioning_id': normal['normal_filename'], 'normal_sha256': normal['normal_sha256'],
               'normal_path': str(Path(normal_manifest['pool_root']) / normal['normal_filename']),
               'image_path': str(Path(source_manifest['dataset_root']) / src['image_relative']),
               'mask_path': str(Path(source_manifest['dataset_root']) / src['mask_relative']),
               'image_sha256': src['image_sha256'], 'mask_sha256': src['mask_sha256'],
               'preprocessed_mask_sha256': hashlib.sha256((mask.astype('uint8')*255).tobytes()).hexdigest(),
               'bbox_transform': {'source_image_size': target['image_size'], 'crop_size': src['crop_size'],
                                  'crop_center_xy': src['crop_center_xy'], 'output_size': [512, 512]},
               'synthetic_labels': labels, 'same_class_union_mask_annotation_policy': 'project all existing same-class boxes intersecting source crop; no new real annotations',
               'prompt': baseline['prompt'], 'generation_protocol': baseline,
               'bank_sample_seeds': {b: sample_seed(seed, selected_row['slot_id']) for b, seed in {'A':42,'B':3407,'C':2026}.items()},
               'structural_valid': bool(valid)}
        output.append(row)
        if not valid: invalid.append(row['slot_id'])
    a.output.mkdir(parents=True)
    save(a.output / 'generation_slot_manifest.json', {'scope': a.scope, 'slots': output,
         'selection_rule': 'Assign normal by independent Random42 over all sorted source metadata first, then SHA256([class,source_id,conditioning_id,annotation_line_index]); take first40/class without replacement',
         'source_manifest_sha256': sha(source_path), 'normal_pool_manifest_sha256': sha(normal_path),
         'baseline_config_sha256': sha(baseline_path), 'runtime_probe_status_sha256': sha(a.runtime_probe / 'runtime_probe_status.json'),
         'checkpoint_assets': load(a.runtime_probe / 'actual_checkpoint_asset_audit.json'),
         'normal_protocol_change': 'explicit user-authorized six same-source partition exclusions, unchanged filter',
         'candidate_assignments': [{'source_id': r['source']['source_id'], 'conditioning_id': r['normal']['normal_filename'], 'selection_hash': r['slot_id']} for r in candidates]})
    save(a.output / 'generation_slot_independence_audit.json', independence(output))
    status = {'status': 'SLOTS_FROZEN_STRUCTURAL_REVIEW_REQUIRED' if not invalid else 'SLOTS_FROZEN_WITH_STRUCTURAL_FAILURES',
              'scope': a.scope, 'slot_count': len(output), 'per_class_counts': dict(Counter(r['class'] for r in output)),
              'invalid_slot_ids': invalid, 'no_replacement_slots': True,
              'slot_manifest_sha256': sha(a.output / 'generation_slot_manifest.json'),
              'formal_generation_authorized': False, 'detector_training_authorized': False,
              'sampling_count': 0, 'training_count': 0,
              'next_gate': 'Mask-union label coverage review and new-normal-pool pre-generation paired RGB probes'}
    save(a.output / 'slot_freeze_status.json', status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__': main()
