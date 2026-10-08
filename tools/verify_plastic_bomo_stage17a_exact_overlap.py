"""Exact JPEG crop identity audit only. No validation labels, scores or selection."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def crop_geometry(filename):
    match = re.search(r'_x(\d+)_y(\d+)_w(\d+)_h(\d+)\.(?:jpg|jpeg|png)$', filename, re.I)
    if not match:
        raise RuntimeError('OLD_BBOX_CROP_GEOMETRY_UNAVAILABLE')
    x, y, w, h = map(int, match.groups())
    # Original bbox-crop naming rounded bbox edges; historical center used
    # round(normalized_center * width). Test only the fixed +/-1 ambiguity.
    return round(x + w / 2), round(y + h / 2)


def exact_crops(image, center, expected_hash):
    hits = []
    cx, cy = center
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            box = (cx + dx - 256, cy + dy - 256, cx + dx + 256, cy + dy + 256)
            # Pillow pads out-of-bounds RGB with zero, matching recovered
            # historical black canvas + paste. No validation bbox is read.
            patch = image.crop(box)
            stream = io.BytesIO()
            patch.save(stream, format='JPEG', quality=95)
            if digest(stream.getvalue()) == expected_hash:
                hits.append({'crop_xyxy_including_padding': list(box),
                             'center_xy': [cx + dx, cy + dy],
                             'jpeg_quality': 95, 'reproduced_sha256': expected_hash})
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('numbering_audit', 'rebuild_audit', 'server_isolation',
                 'generation_train_root', 'validation_image_dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scope', choices=('LOCAL_EXACT_AUDIT', 'SERVER_EXACT_AUDIT'), default='LOCAL_EXACT_AUDIT')
    a = parser.parse_args()
    metadata = load(a.server_isolation)
    if metadata['status'] != 'GENERATOR_TRAIN_VALIDATION_METADATA_OVERLAP_REPRODUCED_ON_SERVER':
        raise RuntimeError('SERVER_METADATA_AUDIT_REQUIRED')
    if a.scope == 'SERVER_EXACT_AUDIT' and a.validation_image_dir.resolve() != Path(metadata['official_validation_directory']).resolve():
        raise RuntimeError('OFFICIAL_VALIDATION_PATH_DIFFERS')
    numbering = load(a.numbering_audit)
    if numbering['conflicts']:
        raise RuntimeError('HISTORICAL_NUMBERING_CONFLICT')
    candidates = metadata['overlap_records']
    numbering_keys = {(r['class'], r['source_id'], r['parent_stem']) for r in numbering['records']}
    if any((r['class'], r['source_id'], r['parent_stem']) not in numbering_keys for r in candidates):
        raise RuntimeError('CANDIDATE_NUMBERING_MISMATCH')
    rebuild = load(a.rebuild_audit / 'baseline_rebuild_preparation_audit.json')
    folders = {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}
    original = {(cls, Path(row['image']).name): row
                for cls, rows in rebuild['source_pairs'].items() for row in rows}
    paths = {}
    for p in a.validation_image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            if p.stem in paths:
                raise RuntimeError('AMBIGUOUS_VALIDATION_FILENAME')
            paths[p.stem] = p
    cache, records = {}, []
    for candidate in candidates:
        cls, name = candidate['class'], candidate['source_id']
        pair = original[cls, name]
        image = a.generation_train_root / 'Plastic_Bomo/test' / folders[cls] / name
        mask = a.generation_train_root / 'Plastic_Bomo/ground_truth' / folders[cls] / Path(pair['mask']).name
        if digest(image.read_bytes()) != pair['image_sha256'] or digest(mask.read_bytes()) != pair['mask_sha256']:
            raise RuntimeError('ACTUAL_GENERATOR_TRAINING_SOURCE_CHANGED')
        parent = candidate['parent_stem']
        row = {'class': cls, 'source_id': name, 'source_sha256': pair['image_sha256'],
               'mask_sha256': pair['mask_sha256'], 'proposed_parent': parent,
               'old_bbox_crop_filename': candidate['crop_filename'],
               'validation_annotation_read': False, 'scope': a.scope}
        if parent not in paths:
            row.update(status='PARENT_FILE_MISSING', matches=[])
        else:
            if parent not in cache:
                data = paths[parent].read_bytes()
                with Image.open(io.BytesIO(data)) as decoded:
                    cache[parent] = (decoded.convert('RGB'), digest(data))
            raw, parent_hash = cache[parent]
            hits = exact_crops(raw, crop_geometry(candidate['crop_filename']), pair['image_sha256'])
            row.update(status='EXACT_GENERATOR_TRAIN_VALIDATION_CROP_MATCH' if hits else 'NO_EXACT_CROP_MATCH',
                       validation_parent_path=str(paths[parent].resolve()), validation_parent_sha256=parent_hash,
                       original_image_size=list(raw.size), matches=hits)
        records.append(row)
    confirmed = [r for r in records if r['status'] == 'EXACT_GENERATOR_TRAIN_VALIDATION_CROP_MATCH']
    summary = {'status': 'GENERATOR_TRAIN_VALIDATION_CROP_OVERLAP_CONFIRMED' if confirmed else 'EXACT_OVERLAP_NOT_CONFIRMED_LINEAGE_UNRESOLVED',
               'scope': a.scope, 'candidate_count': len(records), 'confirmed_source_count': len(confirmed),
               'confirmed_unique_validation_parents': len({r['proposed_parent'] for r in confirmed}),
               'per_class_confirmed': dict(Counter(r['class'] for r in confirmed)),
               'all_candidates_exactly_confirmed': bool(records) and len(confirmed) == len(records),
               'generator_training_membership_evidence': 'source/mask SHA256 matches original rebuild preparation audit',
               'identity_evidence': 'historical 512 crop black-padding JPEG95 reconstructed bytes equal actual training image SHA256',
               'rounding_ambiguity_offsets': [-1, 0, 1], 'validation_parent_images_decoded': len(cache),
               'validation_labels_read': 0, 'validation_detector_forwards': 0,
               'validation_for_candidate_selection_used': False,
               'formal_generator_identity': 'baseline_stage17a_rebuild_split70_s42_noise_0.0',
               'current_checkpoint_clean_isolation_supported': False,
               'FORMAL_GENERATION_AUTHORIZED': False, 'DETECTOR_TRAINING_AUTHORIZED': False,
               'retraining_automatically_started': False, 'data_modified': False,
               'historical_results_modified': False, 'sampling_count': 0, 'optimizer_step_count': 0,
               'numbering_audit_sha256': digest(a.numbering_audit.read_bytes()),
               'server_metadata_audit_sha256': digest(a.server_isolation.read_bytes())}
    a.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('exact_overlap_status.json', summary), ('exact_overlap_records.json', {'records': records})]:
        (a.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (a.output / 'EXACT_OVERLAP_REPORT.md').write_text(
        '# Stage17A generator training / validation exact crop audit\n\n'
        f'Scope: {a.scope}. Exact sources: {len(confirmed)}/{len(records)}; unique validation parents: '
        f'{summary["confirmed_unique_validation_parents"]}; per class: {summary["per_class_confirmed"]}.\n\n'
        'Reconstruction used only previously recovered crop geometry and a fixed +/-1 center-rounding '
        'ambiguity. SHA256 equality of reconstructed JPEG bytes establishes identity, not perceptual similarity. '
        'Actual generator-training membership is bound to the rebuild source/mask hashes.\n\n'
        'Validation pixels were read solely for isolation diagnosis; no validation labels, detector losses, '
        'metrics or scores were accessed. No slots were selected or removed using these results.\n\n'
        'Do not run the current rebuilt checkpoint as a leakage-free Stage17A baseline. Dropping suspect '
        'generation slots cannot undo generator training exposure. A separately authorized, explicitly '
        'versioned train-only reconstruction/retraining protocol is required before clean downstream claims. '
        'Historical experiment statuses are not rewritten by this diagnostic.\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
