"""Build a separately versioned generator pool from hash-verified frozen train only."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FOLDERS = {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}
CROP_SHA = '6c662c0bfe049c32031c8b9cf4433c1ac31dfd2c2da0b921cf5ae7291f6152df'
MASK_SHA = '0644a5e461f81c885ac120e425544dc94f306f603a4ef2e243df93542413b4e5'
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def validate_protocol(protocol):
    if (protocol['identity'] != 'baseline_stage17a_trainonly_s42_noise_0.0'
            or protocol['source_counts'] != {'flash': 88, 'black': 80}
            or protocol['frozen_real_images'] != 138
            or protocol['crop'] != {'minimum_size': 512, 'bbox_scale': 4.0, 'padding': 'black', 'jpeg_quality': 95}
            or protocol['mask'] != {'flash_scale': 2.8, 'black_scale': 1.45, 'minimum_radius': 5.0, 'same_class_full_image_union': True}):
        raise RuntimeError('TRAINONLY_PROTOCOL_CHANGED')
    old = load(REPO / 'experiments/baseline_stage17a_rebuild_split70_s42.json')['training']
    if any(old[k] != v for k, v in protocol['training'].items()):
        raise RuntimeError('TRAINING_HYPERPARAMETERS_CHANGED')


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def recovered_functions(crop_path, mask_path):
    scope = {'Image': Image, 'math': math, 'np': np}
    for path, expected, names in (
        (crop_path, CROP_SHA, {'centered_square_crop_box', 'crop_with_padding'}),
        (mask_path, MASK_SHA, {'regular_ellipse_mask'}),
    ):
        source = Path(path).read_text(encoding='utf-8')
        # Evidence copies use Windows or Unix line endings; original logical source hash is fixed.
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError('HISTORICAL_TRANSFORM_SOURCE_CHANGED:' + str(path))
        funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in funcs} != names:
            raise RuntimeError('HISTORICAL_FUNCTION_MISSING')
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<pinned_historical_functions>', 'exec'), scope)
    return scope


def yaml_directories(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    root = Path(cfg.get('path', Path(path).resolve().parent))
    if not root.is_absolute():
        root = (Path(path).resolve().parent / root).resolve()
    result = {}
    for key in ('train', 'val'):
        part = cfg[key]
        if not isinstance(part, str):
            raise RuntimeError('EXPLICIT_IMAGE_DIRECTORY_REQUIRED:' + key)
        result[key] = (Path(part) if Path(part).is_absolute() else root / part).resolve()
        if not result[key].is_dir():
            raise RuntimeError('DATA_DIRECTORY_MISSING:' + str(result[key]))
    return result


def label_directory(image_dir):
    if image_dir.parent.name != 'images':
        raise RuntimeError('EXPLICIT_YOLO_IMAGES_LAYOUT_REQUIRED')
    return image_dir.parent.parent / 'labels' / image_dir.name


def rgb_hash(path):
    with Image.open(path) as raw:
        im = raw.convert('RGB')
        h = hashlib.sha256(json.dumps(list(im.size)).encode())
        h.update(im.tobytes())
        return list(im.size), h.hexdigest()


def verify_inputs(frozen_path, train_dir, val_dir, expected_images=138, expected_counts=None):
    rows = load(frozen_path)['records']
    counts = dict(Counter(r['class'] for r in rows))
    if counts != (expected_counts or {'flash': 88, 'black': 80}):
        raise RuntimeError('FROZEN_CLASS_COUNTS_CHANGED')
    grouped = {}
    for r in rows:
        name = r['image_id']
        if Path(name).name != name or '\\' in name or '/' in name:
            raise RuntimeError('UNSAFE_IMAGE_ID')
        grouped.setdefault(name, []).append(r)
    current = {p.name for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS}
    if len(grouped) != expected_images or current != set(grouped):
        raise RuntimeError('FROZEN_REAL_IMAGE_SET_CHANGED')
    train_assets = {}
    for name, rs in grouped.items():
        image = train_dir / name
        label = label_directory(train_dir) / (Path(name).stem + '.txt')
        image_hash, label_hash = sha(image), sha(label)
        if any(r['image_sha256'] != image_hash or r['annotation_sha256'] != label_hash for r in rs):
            raise RuntimeError('FROZEN_PARENT_OR_LABEL_CHANGED:' + name)
        size, rgb = rgb_hash(image)
        if any(r['image_size'] != size or r['decoded_rgb_sha256'] != rgb for r in rs):
            raise RuntimeError('FROZEN_DECODED_IMAGE_CHANGED:' + name)
        lines = [line.split() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(lines) != len(rs) or len({r['annotation_line_index'] for r in rs}) != len(rs):
            raise RuntimeError('FROZEN_LABEL_COVERAGE_CHANGED')
        for r in rs:
            parts = lines[r['annotation_line_index'] - 1]
            box = r['bbox_xywh']
            if len(parts) != 5 or int(parts[0]) != r['class_id'] or [float(x) for x in parts[1:]] != box:
                raise RuntimeError('ANNOTATION_INSTANCE_CHANGED')
            if not all(math.isfinite(x) for x in box) or not all(0 <= x <= 1 for x in box) or min(box[2:]) <= 0:
                raise RuntimeError('INVALID_FROZEN_BBOX')
            if r['class_id'] != {'flash': 0, 'black': 1}[r['class']]:
                raise RuntimeError('CLASS_MAPPING_CHANGED')
        train_assets[name] = {'image_sha256': image_hash, 'label_sha256': label_hash, 'decoded_rgb_sha256': rgb, 'image_size': size}
    # Val images only for identity/isolation; never val labels, metrics or quality scores.
    val_assets = {}
    for path in sorted(val_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTS:
            size, rgb = rgb_hash(path)
            val_assets[path.name] = {'image_sha256': sha(path), 'decoded_rgb_sha256': rgb, 'image_size': size}
    if not val_assets:
        raise RuntimeError('EMPTY_VALIDATION_DIRECTORY')
    train_names = {Path(x).stem for x in train_assets}
    val_names = {Path(x).stem for x in val_assets}
    for field in ('image_sha256', 'decoded_rgb_sha256'):
        if {x[field] for x in train_assets.values()} & {x[field] for x in val_assets.values()}:
            raise RuntimeError('REAL_TRAIN_VALIDATION_EXACT_IDENTITY_OVERLAP:' + field)
    if train_names & val_names:
        raise RuntimeError('REAL_TRAIN_VALIDATION_FILENAME_OVERLAP')
    return rows, train_assets, val_assets


def build_pool(rows, train_dir, dataset_root, functions):
    records = []
    for name in sorted({r['image_id'] for r in rows}):
        same_parent = [r for r in rows if r['image_id'] == name]
        with Image.open(train_dir / name) as raw:
            im = raw.convert('RGB')
            for cls in sorted({r['class'] for r in same_parent}):
                class_rows = [r for r in same_parent if r['class'] == cls]
                full = np.zeros((im.height, im.width), dtype=bool)
                for r in class_rows:
                    local, (x0, y0, x1, y1) = functions['regular_ellipse_mask'](im.width, im.height, *r['bbox_xywh'], 2.8 if cls == 'flash' else 1.45, 5.0)
                    full[y0:y1, x0:x1] |= local
                full_mask = Image.fromarray(full.astype('uint8') * 255)
                for r in sorted(class_rows, key=lambda x: x['annotation_line_index']):
                    sid = hashlib.sha256(json.dumps([cls, name, r['annotation_line_index']], separators=(',', ':')).encode()).hexdigest()
                    info = functions['centered_square_crop_box'](im.width, im.height, *r['bbox_xywh'], 512, 4.0)
                    patch = functions['crop_with_padding'](im, info, (0, 0, 0))
                    mask = functions['crop_with_padding'](full_mask, info, 0)
                    if not mask.getbbox():
                        raise RuntimeError('EMPTY_SOURCE_MASK:' + sid)
                    image_rel = Path(cls) / 'images' / (sid + '.jpg')
                    mask_rel = Path(cls) / 'masks' / (sid + '.png')
                    for rel in (image_rel, mask_rel):
                        (dataset_root / rel).parent.mkdir(parents=True, exist_ok=True)
                    patch.save(dataset_root / image_rel, format='JPEG', quality=95)
                    mask.save(dataset_root / mask_rel, format='PNG')
                    records.append({'source_id': sid, 'class': cls, 'class_id': r['class_id'], 'real_parent': name,
                                    'parent_image_sha256': r['image_sha256'], 'parent_label_sha256': r['annotation_sha256'],
                                    'annotation_id': r['donor_id'], 'annotation_line_index': r['annotation_line_index'],
                                    'source_group_id': r.get('source_group_id'), 'bbox_xywh': r['bbox_xywh'],
                                    'image_relative': image_rel.as_posix(), 'mask_relative': mask_rel.as_posix(),
                                    'image_sha256': sha(dataset_root / image_rel), 'mask_sha256': sha(dataset_root / mask_rel),
                                    'crop_source_xyxy': list(info[0]), 'padding_offset_xy': list(info[1]),
                                    'crop_size': info[2], 'crop_center_xy': list(info[3:]),
                                    'annotation_accuracy_claimed': False})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('frozen', 'real_data_yaml', 'crop_source', 'mask_source', 'dataset_root', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage17a_trainonly.json')
    p.add_argument('--scope', choices=('SERVER_PREFLIGHT', 'LOCAL_TEST'), default='SERVER_PREFLIGHT')
    p.add_argument('--local_real_root', type=Path)
    a = p.parse_args()
    a.dataset_root = a.dataset_root.resolve(); a.output = a.output.resolve()
    for path in (a.dataset_root, a.output):
        if path.exists():
            raise RuntimeError('NEW_DIRECTORY_REQUIRED:' + str(path))
    if a.output == a.dataset_root or a.output in a.dataset_root.parents or a.dataset_root in a.output.parents:
        raise RuntimeError('SEPARATE_OUTPUT_AND_DATASET_REQUIRED')
    if a.local_real_root:
        if a.scope != 'LOCAL_TEST':
            raise RuntimeError('LOCAL_OVERRIDE_FORBIDDEN_ON_SERVER')
        dirs = {'train': a.local_real_root / 'images/train', 'val': a.local_real_root / 'images/val'}
    else:
        dirs = yaml_directories(a.real_data_yaml)
    functions = recovered_functions(a.crop_source, a.mask_source)
    protocol = load(a.protocol)
    validate_protocol(protocol)
    rows, train, val = verify_inputs(a.frozen, dirs['train'], dirs['val'])
    # Freeze exact parents and labels BEFORE creating any generator training artifacts.
    a.output.mkdir(parents=True); a.dataset_root.mkdir(parents=True)
    isolation = {'status': 'FROZEN_PARENT_EXACT_ISOLATION_PASS', 'train_images': train, 'validation_images': val,
                 'train_directory': str(dirs['train'].resolve()), 'validation_directory': str(dirs['val'].resolve()),
                 'official_data_yaml_sha256': sha(a.real_data_yaml), 'frozen_manifest_sha256': sha(a.frozen),
                 'validation_images_read_for_identity_only': len(val), 'validation_labels_read': 0,
                 'validation_for_selection_or_tuning_used': False, 'physical_acquisition_group_independence': 'NOT_ESTABLISHED'}
    save(a.output / 'source_isolation_audit.json', isolation)
    records = build_pool(rows, dirs['train'], a.dataset_root, functions)
    manifest = {'identity': 'baseline_stage17a_trainonly_s42_noise_0.0', 'scope': a.scope,
                'dataset_root': str(a.dataset_root), 'source_count': len(records),
                'per_class_counts': dict(Counter(r['class'] for r in records)),
                'source_policy': 'All frozen real-train annotations; no old split70 subset or selectors',
                'crop_source_sha256': CROP_SHA, 'mask_source_sha256': MASK_SHA, 'records': records}
    save(a.output / 'trainonly_source_manifest.json', manifest)
    status = {'status': 'TRAINONLY_POOL_PREPARED_ISOLATION_PASS', 'scope': a.scope,
              'generator_training_authorized': a.scope == 'SERVER_PREFLIGHT',
              'formal_generation_authorized': False, 'detector_training_authorized': False,
              'source_counts': manifest['per_class_counts'], 'parent_images': len(train),
              'frozen_manifest_sha256': sha(a.frozen), 'source_manifest_sha256': sha(a.output / 'trainonly_source_manifest.json'),
              'isolation_audit_sha256': sha(a.output / 'source_isolation_audit.json'),
              'protocol_sha256': sha(a.protocol), 'preparation_script_sha256': sha(Path(__file__)),
              'labels_modified': False, 'historical_results_modified': False,
              'sampling_count': 0, 'optimizer_step_count': 0}
    save(a.output / 'trainonly_preparation_status.json', status)
    (a.output / 'TRAINONLY_PROTOCOL_REPORT.md').write_text(
        '# Stage17A independent train-only baseline\n\n'
        'New 88 flash / 80 black instance crops from the unchanged frozen 138-image train set. '
        'This is a changed source protocol, NOT historical split70 baseline reproduction. '
        'Historical crop/mask functions are hash-pinned; all same-class ellipses on a parent are unioned before cropping.\n\n'
        'Validation image bytes/pixels were accessed solely for exact parent isolation, never val annotations, '
        'selection, task loss or metrics. No proof of unknown acquisition-group independence or annotation accuracy is claimed. '
        'Supplemental Black annotation-review limitations remain.\n\n'
        'No old checkpoint or labels changed. Generation remains blocked pending new-checkpoint/runtime, '
        'conditioning-normal lineage, slot/bbox transform and paired-RNG audits.\n', encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
