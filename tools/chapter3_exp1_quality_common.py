"""Frozen-image visual evaluation metadata/statistics. No ML import at module load."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs/plastic_bomo_chapter3_exp1.json'
OUTPUT = REPO / 'results_for_gpt/plastic_bomo_chapter3_exp1_msdf_generation_quality'
BANKS = ('A', 'B', 'C')
CLASSES = ('flash', 'black')
METHODS = ('Raw SD2', 'MSDF')
ZERO = dict(generator_training_count=0, generation_count=0, detector_training_count=0,
            BootstrapGuard=0, DWBG=0, selector=0, automatic_next_experiment=False)


class Stop(RuntimeError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(str(detail))


def require(ok, status, detail):
    if not ok:
        raise Stop(status, detail)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def portable(path):
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def inside(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    require(not Path(name).is_absolute() and p.is_relative_to(root), 'INPUT_PATH_INVALID', str(name))
    return p


def checked_file(path, digest):
    path = Path(path)
    require(path.is_file(), 'INPUT_ASSET_MISSING', str(path))
    require(sha(path) == digest, 'FROZEN_INPUT_CHANGED', str(path))
    return path


def crop_box(bbox, width, height, margin=.25):
    require(len(bbox) == 4 and all(math.isfinite(v) for v in bbox), 'BBOX_INVALID', bbox)
    cx, cy, bw, bh = bbox
    require(width > 0 and height > 0 and 0 <= cx <= 1 and 0 <= cy <= 1
            and 0 < bw <= 1 and 0 < bh <= 1 and margin == .25, 'BBOX_INVALID', bbox)
    cx_px, cy_px, bw_px, bh_px = cx * width, cy * height, bw * width, bh * height
    x0, y0 = cx_px - (.5 + margin) * bw_px, cy_px - (.5 + margin) * bh_px
    x1, y1 = cx_px + (.5 + margin) * bw_px, cy_px + (.5 + margin) * bh_px
    out = [max(0, math.floor(x0)), max(0, math.floor(y0)), min(width, math.ceil(x1)), min(height, math.ceil(y1))]
    require(out[2] > out[0] and out[3] > out[1], 'BBOX_EMPTY_AFTER_CLIP', out)
    return out


def verify_metadata(cfg):
    for name, expected in cfg['input_metadata_sha256'].items():
        checked_file(inside(REPO, name), expected)


def matched_records(cfg):
    """Pair exact frozen slots; no discovery, ranking, pooling or image selection."""
    p0 = resolve(cfg['p0_root'])
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    slot_map = {s['slot_id']: s for s in slots}
    require(len(slots) == len(slot_map) == 80, 'SLOT_PAIRING_INVALID', 'exact80 slots')
    rng = {r['slot_id']: r['banks'] for r in load(p0 / 'stage20b_rng_manifest.json')['records']}
    roots = {'Raw SD2': resolve(cfg['raw_root']), 'MSDF': resolve(cfg['msdf_root'])}
    entries = {}
    for method, root in roots.items():
        frozen = load(root / 'formal_synthetic_manifest.json')
        validity = load(root / 'formal_synthetic_validity.json')
        require(frozen['status'] == 'FROZEN' and frozen['count'] == len(frozen['records']) == 240
                and not frozen.get('quality_selection', frozen.get('selector_count', 0))
                and validity['status'] == 'PASS', 'FROZEN_GENERATION_INCOMPLETE', method)
        if 'manifest_sha256' in validity:
            require(sha(root / 'formal_synthetic_manifest.json') == validity['manifest_sha256'], 'FROZEN_INPUT_CHANGED', method)
        checkpoints = load(root / 'generator_checkpoint_binding.json')
        require(checkpoints['status'] == 'FROZEN' and set(checkpoints['classes']) == set(CLASSES), 'GENERATOR_BINDING_INVALID', method)
        combined = []
        for bank in BANKS:
            mf = load(root / f'formal_bank_{bank}_manifest.json')
            records = mf['records']
            require(mf['status'] == 'COMPLETE' and len(records) == 80 and mf['reroll_count'] == 0
                    and mf['quality_selection'] is False and {r['slot_id'] for r in records} == set(slot_map)
                    and Counter(r['class'] for r in records) == {'flash': 40, 'black': 40},
                    'SLOT_PAIRING_INVALID', method + ':' + bank)
            for row in records:
                slot = slot_map[row['slot_id']]
                expected = rng[row['slot_id']][bank]
                require(row['bank'] == bank and row['class'] == slot['class'] and row['frozen_slot'] == slot
                        and row['sample_seed'] == expected['sample_seed'] and row['bank_seed'] == expected['bank_seed']
                        and hashlib.sha256(canonical(slot).encode()).hexdigest() == expected['non_rng_metadata_sha256']
                        and row['checkpoint_tree_sha256'] == checkpoints['classes'][slot['class']]['checkpoint']['tree_sha256']
                        and row['quality_selection'] is False and row['finite_pipeline_pixels'],
                        'SLOT_PAIRING_INVALID', method + ':' + bank + ':' + row['slot_id'])
                image = checked_file(inside(root, row['image_path']), row['image_sha256'])
                checked_file(inside(root, row['label_path']), row['label_sha256'])
                entries[(method, bank, row['slot_id'])] = row, image
            combined.extend(records)
        require(combined == frozen['records'], 'SLOT_PAIRING_INVALID', method + ' combined manifest')
    result = []
    for bank in BANKS:
        for sid in sorted(slot_map):
            raw, rp = entries[('Raw SD2', bank, sid)]
            msdf, mp = entries[('MSDF', bank, sid)]
            require(raw['frozen_slot'] == msdf['frozen_slot'], 'SLOT_PAIRING_INVALID', sid)
            s = slot_map[sid]
            width, height = s['geometry']['final_canvas_wh']
            bbox = s['geometry']['final_bbox_normalized_xywh']
            require(bbox == s['bbox'], 'BBOX_INVALID', sid)
            result.append({'bank': bank, 'slot_id': sid, 'class': s['class'],
                'raw_image_path': portable(rp), 'raw_image_sha256': raw['image_sha256'],
                'msdf_image_path': portable(mp), 'msdf_image_sha256': msdf['image_sha256'],
                'bbox': bbox, 'bbox_format': 'normalized_xywh', 'canvas_wh': [width, height],
                'roi_xyxy': crop_box(bbox, width, height),
                'conditioning_normal_id': s['conditioning_normal_id'], 'conditioning_normal_sha256': s['normal_sha256'],
                'sample_seed': raw['sample_seed'], 'bank_seed': raw['bank_seed'],
                'slot_metadata_sha256': hashlib.sha256(canonical(s).encode()).hexdigest()})
    return result


def reference_records(cfg):
    stage = resolve(cfg['stage20a_root'])
    anns = load(stage / 'annotation_provenance_v2.json')['records']
    by_node = defaultdict(list)
    for a in anns:
        if a['split_v2'] == 'final_eval':
            by_node[a['node_id']].append(a)
    train_groups = {r['source_group_id'] for r in load(resolve(cfg['p0_root']) / 'generator_train_allowlist.json')['inputs']}
    reference_root = resolve(cfg['msdf_root']) / cfg['reference_copy_subdirectory']
    images, instances = [], []
    for row in load(stage / 'real_source_registry.json')['records']:
        if row['asset_role'] != 'detector' or row['split_v2'] != 'final_eval':
            continue
        require(row['known_source_group_id'] not in train_groups, 'REFERENCE_TRAIN_OVERLAP', row['node_id'])
        node = row['node_id']
        name = str(row['current_path']).replace('\\', '/').rsplit('/', 1)[-1]
        image = reference_root / 'images/final_eval' / (node + Path(name).suffix)
        label = reference_root / 'labels/final_eval' / (node + '.txt')
        checked_file(image, row['file_sha256'])
        rows = sorted(by_node[node], key=lambda r: r['line'])
        require(rows and len({a['label_sha256'] for a in rows}) == 1, 'REAL_ANNOTATION_INVALID', node)
        checked_file(label, rows[0]['label_sha256'])
        lines = label.read_text(encoding='utf-8').splitlines()
        require(len([s for s in lines if s.strip()]) == len(rows), 'REAL_ANNOTATION_INVALID', node + ' annotation coverage')
        image_row = {'node_id': node, 'image_path': portable(image), 'image_sha256': row['file_sha256'],
            'label_path': portable(label), 'label_sha256': rows[0]['label_sha256'],
            'canvas_wh': [row['width'], row['height']], 'classes': sorted({a['class'] for a in rows}),
            'source_group_id': row['known_source_group_id']}
        images.append(image_row)
        for a in rows:
            fields = lines[a['line'] - 1].split()
            require(len(fields) == 5 and int(fields[0]) == CLASSES.index(a['class'])
                    and all(abs(float(v) - w) < 1e-10 for v, w in zip(fields[1:], a['bbox'])),
                    'REAL_ANNOTATION_INVALID', a['annotation_id'])
            instances.append({**image_row, 'annotation_id': a['annotation_id'], 'class': a['class'],
                'bbox': a['bbox'], 'bbox_format': 'normalized_xywh', 'annotation_line_1based': a['line'],
                'roi_xyxy': crop_box(a['bbox'], row['width'], row['height'])})
    require(len(images) == cfg['reference_image_count'] == 48 and len({x['node_id'] for x in images}) == 48
            and Counter(x['class'] for x in instances) == cfg['reference_instance_counts'],
            'REFERENCE_POPULATION_INVALID', 'expected48 images, flash33/black27 boxes')
    require(len({a['annotation_id'] for a in instances}) == 60, 'REFERENCE_POPULATION_INVALID', 'unique60 annotations')
    return {'name': cfg['reference_name'], 'historically_observed': True, 'never_seen_untouched_test': False,
        'images': sorted(images, key=lambda r: r['node_id']), 'instances': sorted(instances, key=lambda r: r['annotation_id']),
        'image_count': 48, 'instance_count': 60, 'instance_class_counts': dict(Counter(r['class'] for r in instances)),
        'full_image_class_counts': {c: sum(c in r['classes'] for r in images) for c in CLASSES},
        'annotations_modified': False, 'training_source_group_overlap': 0}


def base_status(status='PREPARED'):
    return {'status': status, 'full_experiment_complete': False, 'scope': 'GENERATION_VISUAL_METRICS_ONLY',
        'real_reference_count': 48, 'real_reference_roi_count': 60, 'raw_image_count': 240,
        'msdf_image_count': 240, 'matched_slot_count': 80, 'matched_slot_bank_pair_count': 240,
        'full_FID_raw': None, 'full_FID_msdf': None, 'ROI_FID_raw': None, 'ROI_FID_msdf': None,
        'ROI_KID_raw_mean': None, 'ROI_KID_raw_sd': None, 'ROI_KID_msdf_mean': None, 'ROI_KID_msdf_sd': None,
        'LPIPS_raw': None, 'LPIPS_msdf': None, 'Flash_metrics': None, 'Black_metrics': None, 'bankwise_metrics': None, **ZERO}


def implementation_binding(config=CONFIG):
    paths = [Path(config), Path(__file__), Path(__file__).with_name('evaluate_plastic_bomo_chapter3_exp1.py')]
    return {portable(p): sha(p) for p in paths}


def prepare(output=OUTPUT, config=CONFIG):
    output, config = Path(output).resolve(), Path(config).resolve()
    cfg = load(config)
    if (output / 'preparation_binding.json').exists():
        verify_prepared(output, config)
        return load(output / 'visual_comparison_manifest.json'), load(output / 'real_visual_reference_manifest.json')
    require(not output.exists() or not any(output.iterdir()), 'OUTPUT_NOT_EMPTY', str(output))
    print('[配对] 核对冻结元数据和 Raw/MSDF 480 张图的 SHA256。', flush=True)
    verify_metadata(cfg)
    records = matched_records(cfg)
    reference = reference_records(cfg)
    output.mkdir(parents=True, exist_ok=True)
    mf = {'status': 'FROZEN', 'source_commit': cfg['source_commit'], 'path_base': 'repository root',
        'count': 240, 'matched_slot_count': 80, 'raw_image_count': 240, 'msdf_image_count': 240,
        'no_filtering': True, 'no_reroll': True, 'records': records}
    save(output / 'visual_comparison_manifest.json', mf)
    save(output / 'real_visual_reference_manifest.json', reference)
    save(output / 'metric_protocol.json', cfg)
    bindings = {n: sha(output / n) for n in ('visual_comparison_manifest.json', 'real_visual_reference_manifest.json', 'metric_protocol.json')}
    save(output / 'preparation_binding.json', {'status': 'FROZEN_BEFORE_METRICS', 'sha256': bindings,
        'implementation_sha256': implementation_binding(config), 'path_relocation': 'repo-relative paths only',
        'old_files_modified': False, 'detector_metric_files_read': 0, **ZERO})
    save(output / 'experiment1_status.json', base_status())
    return mf, reference


def verify_prepared(output, config=CONFIG):
    output = Path(output)
    b = load(output / 'preparation_binding.json')
    require(b['implementation_sha256'] == implementation_binding(config), 'EVALUATION_CODE_OR_PROTOCOL_CHANGED',
            '代码/协议已与准备时不同；不要覆盖旧结果或事后调整。')
    for n, digest in b['sha256'].items():
        checked_file(output / n, digest)
    cfg = load(output / 'metric_protocol.json')
    verify_metadata(cfg)
    for row in load(output / 'visual_comparison_manifest.json')['records']:
        for method in ('raw', 'msdf'):
            checked_file(resolve(row[method + '_image_path']), row[method + '_image_sha256'])
    for row in load(output / 'real_visual_reference_manifest.json')['images']:
        checked_file(resolve(row['image_path']), row['image_sha256'])
        checked_file(resolve(row['label_path']), row['label_sha256'])
    return cfg


def dependencies(cfg, inception_weights=None, alexnet_weights=None):
    versions = {}
    for name in ('torch', 'torchvision', 'torch-fidelity', 'lpips', 'numpy', 'scipy', 'Pillow', 'matplotlib', 'torchmetrics', 'diffusers', 'ultralytics'):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    required = ('torch', 'torchvision', 'torch-fidelity', 'lpips', 'numpy', 'Pillow', 'matplotlib')
    errors = [{'package': n, 'reason': 'MISSING'} for n in required if versions[n] is None]
    if versions['torch-fidelity'] not in (None, *cfg['distribution_metrics']['supported_versions']):
        errors.append({'package': 'torch-fidelity', 'reason': 'UNSUPPORTED_API_VERSION', 'actual': versions['torch-fidelity']})
    if versions['torch-fidelity'] == '0.3.0' and not versions['scipy']:
        errors.append({'package': 'scipy', 'reason': 'REQUIRED_BY_TORCH_FIDELITY_0_3'})
    hub = Path(os.environ.get('TORCH_HOME', str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'torch'))) / 'hub/checkpoints'
    inc = Path(inception_weights) if inception_weights else hub / cfg['distribution_metrics']['weights_filename']
    alex = Path(alexnet_weights) if alexnet_weights else hub / cfg['LPIPS']['backbone_weights_filename']
    if alex.name != cfg['LPIPS']['backbone_weights_filename'] or alex.parent.name != 'checkpoints':
        errors.append({'weight': 'alexnet', 'reason': 'ALEXNET_CACHE_LAYOUT_REQUIRED',
                       'detail': 'Use .../hub/checkpoints/alexnet-owt-7be5be79.pth; existing cache is never overwritten.'})
    weights = {}
    for name, path, prefix in (('inception', inc, cfg['distribution_metrics']['weights_sha256_prefix']),
                               ('alexnet', alex, cfg['LPIPS']['backbone_weights_sha256_prefix'])):
        digest = sha(path) if path.is_file() else None
        weights[name] = {'path': str(path.resolve()), 'sha256': digest}
        if not digest or not digest.startswith(prefix):
            errors.append({'weight': name, 'reason': 'MISSING_OR_WRONG_STANDARD_WEIGHT', 'expected_sha256_prefix': prefix, 'path': str(path)})
    linear = None
    if versions['lpips']:
        linear = Path(md.distribution('lpips').locate_file('lpips/weights/v0.1/alex.pth'))
        digest = sha(linear) if linear.is_file() else None
        weights['lpips_linear'] = {'path': str(linear), 'sha256': digest}
        if digest != cfg['LPIPS']['linear_weights_sha256']:
            errors.append({'weight': 'lpips_linear', 'reason': 'MISSING_OR_WRONG_OFFICIAL_V0_1_WEIGHT'})
    return {'status': 'PASS' if not errors else 'DEPENDENCY_REQUIRED', 'package_versions': versions,
        'weights': weights, 'errors': errors, 'package_install_or_upgrade_count': 0,
        'network_download_count': 0, 'model_load_count': 0, 'generator_or_detector_forward_count': 0,
        'protected_packages': {n: versions[n] for n in ('torch', 'torchvision', 'diffusers', 'ultralytics')}}


def mean_sd(values):
    require(len(values) >= 2 and all(math.isfinite(v) for v in values), 'METRIC_VALUES_INVALID', values)
    return {'mean': statistics.mean(values), 'sd': statistics.stdev(values), 'n': len(values), 'ddof': 1}


def summarize(distributions, lpips_rows):
    """Only descriptive aggregation; FID/KID formulas are delegated to the library."""
    summary = {}
    for method in METHODS:
        regions = {}
        for region in ('full_image', 'roi'):
            regions[region] = {}
            for cohort in ('overall', *CLASSES):
                rows = [r for r in distributions if r['method'] == method and r['region'] == region and r['cohort'] == cohort]
                require(len(rows) == 3 and {r['bank'] for r in rows} == set(BANKS), 'BANKWISE_RESULTS_INCOMPLETE', (method, region, cohort))
                regions[region][cohort] = {'FID': mean_sd([r['FID'] for r in rows]),
                    'KID': mean_sd([r['KID'] for r in rows]), 'bankwise': {r['bank']: r for r in rows}}
        regions['LPIPS'] = {}
        for cohort in ('overall', *CLASSES):
            rows = [r for r in lpips_rows if r['method'] == method and (cohort == 'overall' or r['class'] == cohort)]
            require(len(rows) == (80 if cohort == 'overall' else 40) and len({r['slot_id'] for r in rows}) == len(rows),
                    'LPIPS_SLOT_COVERAGE_INVALID', (method, cohort))
            regions['LPIPS'][cohort] = mean_sd([r['LPIPS_slot'] for r in rows])
        summary[method] = regions
    comparisons = {}
    for cohort in ('overall', *CLASSES):
        r, m = summary['Raw SD2']['roi'][cohort], summary['MSDF']['roi'][cohort]
        fid_lower, kid_lower = m['FID']['mean'] < r['FID']['mean'], m['KID']['mean'] < r['KID']['mean']
        status = ('MSDF_VISUAL_METRICS_IMPROVED' if fid_lower and kid_lower else
                  'MSDF_VISUAL_METRICS_NOT_IMPROVED' if not fid_lower and not kid_lower else 'MIXED_VISUAL_METRICS')
        comparisons[cohort] = {'status': status, 'ROI_FID_msdf_minus_raw': m['FID']['mean'] - r['FID']['mean'],
            'ROI_KID_msdf_minus_raw': m['KID']['mean'] - r['KID']['mean'],
            'LPIPS_msdf_minus_raw': summary['MSDF']['LPIPS'][cohort]['mean'] - summary['Raw SD2']['LPIPS'][cohort]['mean']}
    return {'status': comparisons['overall']['status'], 'methods': summary, 'comparisons': comparisons,
        'primary': 'defect-region KID', 'FID_auxiliary_small_sample_warning': True,
        'bank_SD_is_not_independent_training_seed_uncertainty': True,
        'LPIPS_is_diversity_not_realism_or_higher_quality': True, **ZERO}


def csv_write(path, header, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.writer(stream, lineterminator='\n')
        w.writerow(header); w.writerows(rows)


def paper_tables(output, summary):
    def display(x):
        return f"{x['mean']:.6g} ± {x['sd']:.6g}"
    overall, classes = [], []
    for method in METHODS:
        s = summary['methods'][method]
        overall.append([method, display(s['roi']['overall']['FID']), display(s['roi']['overall']['KID']), display(s['LPIPS']['overall'])])
        classes.append([method, *[display(s['roi'][c][k]) if k != 'LPIPS' else display(s['LPIPS'][c])
                                  for c in CLASSES for k in ('FID', 'KID', 'LPIPS')]])
    csv_write(Path(output) / 'paper_table_generation_quality.csv', ['Method', 'FID↓', 'KID↓', 'LPIPS↑'], overall)
    csv_write(Path(output) / 'paper_table_generation_quality_classwise.csv',
              ['Method', 'Flash FID', 'Flash KID', 'Flash LPIPS', 'Black FID', 'Black KID', 'Black LPIPS'], classes)
