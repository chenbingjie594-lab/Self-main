"""Reproduce train masks; audit validation filename overlap without reading val content."""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
from recover_plastic_bomo_stage17a_historical_crop import historical_source, load, digest


def recovered_mask_source(session):
    candidates = []
    with session.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if 'generate_regular_yolo_masks.py' not in line:
                continue
            q = json.loads(line).get('payload', {})
            patch = q.get('input', '')
            if q.get('name') == 'apply_patch' and '*** Add File:' in patch:
                source = '\n'.join(s[1:] for s in patch.splitlines() if s.startswith('+'))
                if 'def regular_ellipse_mask' in source:
                    candidates.append((number, source))
    if len(candidates) != 1:
        raise RuntimeError('MASK_HISTORY_NOT_UNIQUE')
    number, source = candidates[0]
    funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'regular_ellipse_mask']
    scope = {'math': math, 'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<historical_mask_only>', 'exec'), scope)
    return scope['regular_ellipse_mask'], source, number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'recovery', 'frozen', 'real_train_root', 'old_bbox_crop_root',
                 'validation_image_dir', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    report = load(a.recovery / 'historical_crop_reproduction_audit.json')
    frozen = load(a.frozen)['records']
    crops, _, _ = historical_source(a.session)
    ellipse, mask_source, mask_line = recovered_mask_source(a.session)
    # Only directory entries are inspected for validation. No images decoded,
    # no labels opened, no validation-based source filtering or substitutions.
    if not a.validation_image_dir.is_dir():
        raise RuntimeError('VALIDATION_FILENAME_DIRECTORY_MISSING')
    val_names = {p.stem for p in a.validation_image_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')}
    train_names = {Path(r['image_id']).stem for r in frozen}
    cache, mask_rows = {}, []
    for row in report['records']:
        if row['status'] != 'UNIQUE_EXACT_IMAGE_CROP_BOUND':
            continue
        hit = row['matches'][0]
        parent = hit['real_parent']
        records = [r for r in frozen if r['image_id'] == parent and r['class'] == row['class']]
        image = a.real_train_root / 'images/train' / parent
        label = a.real_train_root / 'labels/train' / (Path(parent).stem + '.txt')
        if digest(image.read_bytes()) != records[0]['image_sha256'] or digest(label.read_bytes()) != records[0]['annotation_sha256']:
            raise RuntimeError('FROZEN_TRAIN_CHANGED')
        w, h = records[0]['image_size']
        key = (parent, row['class'])
        if key not in cache:
            full = np.zeros((h, w), dtype=bool)
            for record in records:
                local, (x0, y0, x1, y1) = ellipse(w, h, *record['bbox_xywh'],
                                                 2.8 if row['class'] == 'flash' else 1.45, 5.0)
                full[y0:y1, x0:x1] |= local
            cache[key] = Image.fromarray(full.astype('uint8') * 255)
        record = next(r for r in records if r['donor_id'] == hit['annotation_id'])
        info = crops['centered_square_crop_box'](w, h, *record['bbox_xywh'], 512, 4.0)
        mask = crops['crop_with_padding'](cache[key], info, 0)
        stream = io.BytesIO(); mask.save(stream, format='PNG')
        reproduced = digest(stream.getvalue())
        mask_rows.append({'class': row['class'], 'source_id': row['source_id'],
                          'real_parent': parent, 'annotation_id': record['donor_id'],
                          'mask_sha256': row['mask_sha256'], 'reproduced_sha256': reproduced,
                          'exact_png_bytes_match': reproduced == row['mask_sha256']})
    catalog, all_rows, conflicts = {}, [], []
    for cls, folder in {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}.items():
        entries = []
        for old_split in ('train', 'val'):
            directory = a.old_bbox_crop_root / old_split / folder
            if not directory.is_dir():
                raise RuntimeError('OLD_CROP_FILENAME_DIRECTORY_MISSING')
            for path in sorted(directory.glob('*.jpg')):
                match = re.fullmatch(r'(.+)_(\d+)_x\d+_y\d+_w\d+_h\d+', path.stem)
                if not match:
                    raise RuntimeError('UNEXPECTED_OLD_CROP_FILENAME')
                entries.append({'parent_stem': match[1], 'old_split': old_split,
                                'old_annotation_index': int(match[2]), 'crop_filename': path.name})
        if len(entries) != (108 if cls == 'flash' else 51):
            raise RuntimeError('OLD_CATALOG_COUNT_DIFFERS_FROM_CONVERSION')
        catalog[cls] = entries
        for source in [r for r in report['records'] if r['class'] == cls]:
            position = int(Path(source['source_id']).stem)
            entry = entries[position]
            hit = source['matches'][0] if source['status'] == 'UNIQUE_EXACT_IMAGE_CROP_BOUND' else None
            if hit and Path(hit['real_parent']).stem != entry['parent_stem']:
                conflicts.append({'class': cls, 'source_id': source['source_id']})
            all_rows.append({'class': cls, 'source_id': source['source_id'], **entry,
                             'current_frozen_train_filename': entry['parent_stem'] in train_names,
                             'current_validation_filename': entry['parent_stem'] in val_names,
                             'parent_pixel_identity_verified': hit is not None,
                             'evidence': 'EXACT_IMAGE_CROP_REPRODUCTION' if hit else 'RECOVERED_NUMBERING_METADATA_ONLY'})
    suspicious = [r for r in all_rows if r['current_validation_filename']]
    summary = {'status': 'POSSIBLE_GENERATOR_TRAIN_VALIDATION_OVERLAP_METADATA_EVIDENCE',
               'source_count': len(all_rows), 'numbering_rule_verified_on_exact_sources': len(mask_rows),
               'numbering_conflicts': conflicts,
               'mask_reproduction_checked': len(mask_rows),
               'mask_exact_png_bytes_matches': sum(r['exact_png_bytes_match'] for r in mask_rows),
               'per_class_mask_matches': dict(Counter(r['class'] for r in mask_rows if r['exact_png_bytes_match'])),
               'candidate_validation_overlap_sources': len(suspicious),
               'candidate_validation_overlap_unique_parent_names': len({r['parent_stem'] for r in suspicious}),
               'per_class_candidate_validation_overlap': dict(Counter(r['class'] for r in suspicious)),
               'remaining_unassigned_to_train_or_val_names': [r for r in all_rows if not r['current_frozen_train_filename'] and not r['current_validation_filename']],
               'validation_filename_directory': str(a.validation_image_dir.resolve()),
               'validation_image_pixels_read': 0, 'validation_labels_read': 0,
               'filename_overlap_is_not_claimed_exact_pixel_leakage_confirmation': True,
               'mask_source_history_line': mask_line, 'mask_source_sha256': digest(mask_source.encode()),
               'generation_count': 0, 'training_count': 0, 'labels_modified': False,
               'FORMAL_GENERATION_AUTHORIZED': False, 'DETECTOR_TRAINING_AUTHORIZED': False,
               'protocol_changed': False, 'status_scope': 'LOCAL_RECOVERY_NOT_SERVER_ISOLATION_PASS'}
    a.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('recovered_isolation_status.json', summary),
                        ('mask_transform_reproduction.json', {'records': mask_rows}),
                        ('numbering_and_validation_filename_audit.json', {'records': all_rows, 'conflicts': conflicts})]:
        (a.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (a.output / 'recovered_historical_mask_source.txt').write_text(mask_source, encoding='utf-8')
    (a.output / 'ISOLATION_RECOVERY_REPORT.md').write_text(
        '# Stage17A historical mask and isolation recovery\n\n'
        f'Exact train-mask PNG reproductions: {summary["mask_exact_png_bytes_matches"]}/{len(mask_rows)}. '
        f'Numbering conflicts on exact image-bound sources: {len(conflicts)}.\n\n'
        f'Candidate generator-training sources whose proposed parent name appears in the current validation '
        f'directory: {len(suspicious)} ({summary["per_class_candidate_validation_overlap"]}); '
        f'unique proposed parent names: {summary["candidate_validation_overlap_unique_parent_names"]}.\n\n'
        'The source names were proposed from the recovered conversion ordering and old bbox-crop filename '
        'catalog. No validation pixels or annotations were accessed. These candidate overlaps are metadata '
        'evidence, not exact pixel identity confirmation. The old train/val split was mixed before the later '
        'split70 generation training split; the old splits are not assumed to be the current official split.\n\n'
        'Formal sampling and detector training remain blocked. Removing the suspicious slots alone would '
        'not address the fact that the current generator was trained on the full source pool. Do not filter '
        'slots, reuse its downstream scores as uncontaminated results, or retrain a new pool without a '
        'separately frozen authorized protocol. Confirming server split identity and exact lineage is the '
        'next read-only audit. No historical conclusions are retroactively rewritten.\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
