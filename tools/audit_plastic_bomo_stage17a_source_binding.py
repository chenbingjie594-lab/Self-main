"""Read-only exact-content source binding. No similarity or ordered-name matching."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_sha(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return (im.size, hashlib.sha256(im.tobytes()).hexdigest())


def crop_parent(name):
    match = re.fullmatch(r'(.+)_(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)\.(?:jpg|jpeg|png)', name, re.I)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('frozen', 'real_train_root', 'split_manifest', 'generation_train_root',
                'bbox_crop_train_root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--rebuild_audit', type=Path, required=True)
    args = parser.parse_args()
    frozen = load(args.frozen)['records']
    images = {}
    for row in frozen:
        images.setdefault(row['image_id'], []).append(row)
    if len(images) != 138 or len(frozen) != 168:
        raise RuntimeError('WRONG_FROZEN_REAL_SET')
    file_index, rgb_index = {}, {}
    def index(path, record):
        file_index.setdefault(sha(path), []).append(record)
        size, digest = rgb_sha(path)
        rgb_index.setdefault((record['class'], size, digest), []).append(record)
    for name, rows in images.items():
        image = args.real_train_root / 'images/train' / name
        label = args.real_train_root / 'labels/train' / (Path(name).stem + '.txt')
        if sha(image) != rows[0]['image_sha256'] or sha(label) != rows[0]['annotation_sha256']:
            raise RuntimeError('FROZEN_REAL_BYTES_CHANGED:' + name)
        for cls in sorted({r['class'] for r in rows}):
            index(image, {'kind': 'FROZEN_FULL_IMAGE', 'class': cls, 'real_parent': name,
                          'candidate_path': str(image.resolve()),
                          'annotation_indices': [r['annotation_line_index'] for r in rows if r['class'] == cls]})
    classes = {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}
    legal_crop_counts = {}
    by_stem = {Path(name).stem: rows for name, rows in images.items()}
    for cls, folder in classes.items():
        crop_dir = args.bbox_crop_train_root / folder
        if not crop_dir.is_dir():
            raise RuntimeError('CROP_TRAIN_DIRECTORY_MISSING:' + str(crop_dir))
        count = 0
        for path in sorted(crop_dir.iterdir()):
            parent = crop_parent(path.name)
            rows = by_stem.get(parent, [])
            rows = [r for r in rows if r['class'] == cls]
            # Filenames identify candidate lineage only, never establish provenance.
            if not rows:
                continue
            count += 1
            index(path, {'kind': 'NAMED_BBOX_CROP_CANDIDATE_ONLY', 'class': cls,
                         'real_parent': rows[0]['image_id'], 'candidate_path': str(path.resolve()),
                         'annotation_indices': [r['annotation_line_index'] for r in rows],
                         'crop_to_real_transform_verified': False})
        legal_crop_counts[cls] = count
    split = load(args.split_manifest)
    if split.get('seed') != 42 or split.get('train_ratio') != .7:
        raise RuntimeError('WRONG_GENERATION_SPLIT')
    rebuild = load(args.rebuild_audit / 'baseline_rebuild_preparation_audit.json')
    if sha(args.split_manifest) != rebuild['split_manifest_sha256']:
        raise RuntimeError('GENERATOR_TRAIN_SPLIT_CHANGED')
    records = []
    for cls, folder in classes.items():
        original_pairs = {Path(r['image']).name: r for r in rebuild['source_pairs'][cls]}
        for row in split['categories']['Plastic_Bomo'][folder]['train']:
            for key in ('image', 'mask'):
                if Path(row[key]).name != row[key] or '/' in row[key] or '\\' in row[key]:
                    raise RuntimeError('INVALID_MANIFEST_FILENAME')
            path = args.generation_train_root / 'Plastic_Bomo/test' / folder / row['image']
            mask = args.generation_train_root / 'Plastic_Bomo/ground_truth' / folder / row['mask']
            original = original_pairs[row['image']]
            if sha(path) != original['image_sha256'] or sha(mask) != original['mask_sha256']:
                raise RuntimeError('GENERATOR_TRAINING_SOURCE_BYTES_CHANGED:' + row['image'])
            size, digest = rgb_sha(path)
            exact = [r for r in file_index.get(sha(path), []) if r['class'] == cls]
            decoded = rgb_index.get((cls, size, digest), [])
            candidates = exact or decoded
            full = [r for r in candidates if r['kind'] == 'FROZEN_FULL_IMAGE']
            records.append({'class': cls, 'source_id': row['image'],
                            'source_path': str(path.resolve()), 'source_sha256': sha(path),
                            'decoded_rgb_sha256': digest, 'source_size': list(size),
                            'mask_path': str(mask.resolve()), 'mask_sha256': sha(mask),
                            'exact_file_candidates': exact, 'exact_decoded_candidates': decoded,
                            'verified_frozen_parent': full[0]['real_parent'] if len({r['real_parent'] for r in full}) == 1 else None,
                            'status': 'EXACT_FROZEN_FULL_IMAGE_BOUND' if full else
                                      'CROP_TRANSFORM_STILL_REQUIRED' if candidates else 'NO_EXACT_CONTENT_BINDING'})
    counts = {cls: dict(Counter(r['status'] for r in records if r['class'] == cls)) for cls in classes}
    # Even a full-image match does not resolve instance mask / spatial transforms.
    report = {'status': 'FORMAL_SLOT_LINEAGE_NOT_ESTABLISHED', 'scope': 'TRAIN_ONLY_SOURCE_CONTENT_AUDIT',
              'per_class_source_counts': dict(Counter(r['class'] for r in records)),
              'per_class_binding_counts': counts, 'frozen_real_images_verified': 138,
              'frozen_annotation_instances_verified': 168, 'named_frozen_parent_crop_candidates': legal_crop_counts,
              'rebuild_split_sha256_matches': True,
              'all_source_and_mask_bytes_match_generator_training_audit': True,
              'unknown_parent_is_not_proven_validation_leakage': True,
              'similarity_matching_used': False, 'numbered_source_order_used_as_mapping': False,
              'validation_content_read': 0, 'generation_count': 0, 'detector_training_count': 0,
              'FORMAL_80_SLOT_GENERATION_AUTHORIZED': False, 'records': records}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'source_parent_binding_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (args.output / 'STAGE17A_SOURCE_BINDING_REPORT.md').write_text(
        '# Stage17A source binding audit\n\n'
        'The rebuilt baseline passed its runtime smoke test. Formal 80-slot generation remains unauthorized.\n\n'
        f'Train sources: {report["per_class_source_counts"]}. Binding counts: {counts}.\n\n'
        'Generation images are numbered 512x512 sources, whereas the frozen detector set contains full images. '
        'Only file/RGB equality was tested here, not visual nearest-neighbour assignment. Candidate bbox-crop '
        'filenames alone do not prove their transform or parent identity. An unbound source does not prove validation leakage.\n\n'
        'Before formal slots, recover a verifiable source-to-real-parent and annotation/mask transform mapping. '
        'Do not silently create new crops, retrain a different pool, reorder numbered sources, select by quality, '
        'or replace RealRepeat with guessed parents. No historical statuses or labels were changed.\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
