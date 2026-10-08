"""Reproduce a recovered historical crop function for frozen train only."""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def historical_source(session):
    matches = []
    with session.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if 'make_class_center_crops_with_masks.py' not in line:
                continue
            value = json.loads(line).get('payload', {})
            patch = value.get('input', '')
            if value.get('name') == 'apply_patch' and '*** Add File:' in patch:
                source = '\n'.join(s[1:] for s in patch.splitlines() if s.startswith('+'))
                if 'def centered_square_crop_box' in source:
                    matches.append((number, source))
    if len(matches) != 1:
        raise RuntimeError('HISTORICAL_CROP_SOURCE_NOT_UNIQUE')
    number, source = matches[0]
    tree = ast.parse(source)
    names = {'centered_square_crop_box', 'crop_with_padding'}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in funcs} != names:
        raise RuntimeError('HISTORICAL_CROP_FUNCTIONS_MISSING')
    scope = {'Image': Image}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<historical_crop_functions_only>', 'exec'), scope)
    return scope, source, number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'conversion_script', 'frozen', 'real_train_root', 'split_manifest',
                 'generation_train_root', 'rebuild_audit', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    scope, source, line = historical_source(a.session)
    split = load(a.split_manifest)
    rebuild = load(a.rebuild_audit / 'baseline_rebuild_preparation_audit.json')
    if digest(a.split_manifest.read_bytes()) != rebuild['split_manifest_sha256']:
        raise RuntimeError('REBUILD_SPLIT_CHANGED')
    folders = {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}
    targets, by_hash = [], {}
    for cls, folder in folders.items():
        original = {Path(r['image']).name: r for r in rebuild['source_pairs'][cls]}
        for r in split['categories']['Plastic_Bomo'][folder]['train']:
            image = a.generation_train_root / 'Plastic_Bomo/test' / folder / r['image']
            mask = a.generation_train_root / 'Plastic_Bomo/ground_truth' / folder / r['mask']
            image_hash, mask_hash = digest(image.read_bytes()), digest(mask.read_bytes())
            if image_hash != original[r['image']]['image_sha256'] or mask_hash != original[r['image']]['mask_sha256']:
                raise RuntimeError('REBUILD_SOURCE_CHANGED')
            row = {'class': cls, 'source_id': r['image'], 'source_sha256': image_hash,
                   'mask_sha256': mask_hash, 'matches': []}
            targets.append(row)
            by_hash.setdefault((cls, image_hash), []).append(row)
    frozen = load(a.frozen)['records']
    groups = {}
    for r in frozen:
        groups.setdefault(r['image_id'], []).append(r)
    if len(groups) != 138 or len(frozen) != 168:
        raise RuntimeError('WRONG_FROZEN_REAL_SET')
    for name, rows in groups.items():
        image = a.real_train_root / 'images/train' / name
        label = a.real_train_root / 'labels/train' / (Path(name).stem + '.txt')
        if digest(image.read_bytes()) != rows[0]['image_sha256'] or digest(label.read_bytes()) != rows[0]['annotation_sha256']:
            raise RuntimeError('FROZEN_REAL_CHANGED:' + name)
        with Image.open(image) as raw:
            im = raw.convert('RGB')
            for r in rows:
                cx, cy, bw, bh = r['bbox_xywh']
                info = scope['centered_square_crop_box'](im.width, im.height, cx, cy, bw, bh, 512, 4.0)
                patch = scope['crop_with_padding'](im, info, (0, 0, 0))
                buffer = io.BytesIO()
                patch.save(buffer, format='JPEG', quality=95)
                rebuilt_hash = digest(buffer.getvalue())
                for row in by_hash.get((r['class'], rebuilt_hash), []):
                    row['matches'].append({'real_parent': name, 'real_image_sha256': r['image_sha256'],
                                           'annotation_id': r['donor_id'], 'annotation_line_index': r['annotation_line_index'],
                                           'source_group_id': r.get('source_group_id'),
                                           'crop_source_xyxy': list(info[0]), 'padding_offset_xy': list(info[1]),
                                           'crop_size': info[2], 'crop_center_xy': list(info[3:]),
                                           'reproduced_jpeg_sha256': rebuilt_hash,
                                           'exact_file_bytes_match': True,
                                           'mask_transform_reproduced': False})
    for r in targets:
        r['status'] = 'UNIQUE_EXACT_IMAGE_CROP_BOUND' if len(r['matches']) == 1 else 'AMBIGUOUS_EXACT_IMAGE_CROP' if r['matches'] else 'NO_FROZEN_PARENT_REPRODUCTION'
    counts = {c: dict(Counter(r['status'] for r in targets if r['class'] == c)) for c in folders}
    report = {'status': 'HISTORICAL_CONVERSION_RECOVERED_PARTIAL_FROZEN_PARENT_BINDING',
              'per_class_counts': counts, 'source_count': len(targets),
              'exact_unique_sources': sum(r['status'] == 'UNIQUE_EXACT_IMAGE_CROP_BOUND' for r in targets),
              'session_path': str(a.session.resolve()), 'historical_patch_line': line,
              'recovered_crop_source_sha256': digest(source.encode()),
              'conversion_script_path': str(a.conversion_script.resolve()),
              'conversion_script_sha256': digest(a.conversion_script.read_bytes()),
              'historical_conversion_merges_old_yolo_train_and_val_before_renumbering': True,
              'old_yolo_val_is_not_assumed_current_official_validation': True,
              'unmatched_sources_not_claimed_validation_leaks': True,
              'mask_reproduction_complete': False, 'all_generator_training_sources_bound': False,
              'FORMAL_GENERATION_AUTHORIZED': False, 'DETECTOR_TRAINING_AUTHORIZED': False,
              'validation_content_read': 0, 'similarity_matching_used': False,
              'data_or_models_modified': False, 'generation_count': 0, 'training_count': 0,
              'records': targets}
    a.output.mkdir(parents=True, exist_ok=False)
    # Evidence copy only, .txt: never execute the original script's file-writing main.
    (a.output / 'recovered_historical_crop_source.txt').write_text(source, encoding='utf-8')
    (a.output / 'historical_crop_reproduction_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (a.output / 'RECOVERY_REPORT.md').write_text(
        '# Historical Plastic_Bomo conversion recovery\n\n'
        'The conversion was found in local Codex history and the old anomalydiffusion-master project. '
        'The crop source was recovered from an actual historical apply_patch call, not reconstructed by guessing.\n\n'
        f'Sources: {len(targets)}; per-class binding counts: {counts}.\n\n'
        'The historical transform uses bbox-center rounding, crop size max(512,4*bbox-long-side), '
        'black padding at borders and JPEG quality95. It was reproduced in memory only, from '
        'hash-verified frozen real training images and annotations. Exact JPEG bytes bind the matched images.\n\n'
        'The conversion appends sorted old YOLO val crops after train crops, then renumbers. '
        'That old split is not assumed to equal current official validation. Unmatched sources are unresolved, '
        'not automatically validation leaks. Masks still require transform evidence. No formal slots, '
        'generation, retraining, detector training, label modifications or historical status changes were performed.\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
