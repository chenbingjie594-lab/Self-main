"""Stage25-P1 immutable contracts and path-only server binding. No ML imports."""
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stage20b_runtime_contract import Stop, require, load, save, sha, canonical, htext, verify_ledger
from stage20b_formal_common import tree_binding, verify_tree, verified_copy, offline

BANKS = ('A', 'B', 'C')


def now():
    return datetime.now(timezone.utc).isoformat()


def source_check(protocol, manifest):
    require(REPO.is_dir(), 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', '项目目录不存在：' + str(REPO))
    entries = load(manifest)['sha256']
    missing = []
    for name in entries:
        path = (REPO / name).resolve()
        require(path.is_relative_to(REPO), 'STAGE25_FROZEN_BINDING_CHANGED', {'file': name, 'reason': '运行清单路径越界'})
        if not path.is_file():
            missing.append(name)
    require(not missing, 'STAGE25_FROZEN_BINDING_CHANGED', {'reason': '服务器缺少运行必需文件，请按相对路径上传原文件。',
        'missing_files': missing, 'project_root': str(REPO), 'training_started': False})
    for name, expected in entries.items():
        path = (REPO / name).resolve()
        actual = sha(path)
        require(actual == expected, 'STAGE25_FROZEN_BINDING_CHANGED', {'file': name,
                'reason': '文件内容与冻结版本不一致，请上传对应原文件，不要改写 SHA256。',
                'expected_sha256': expected, 'actual_sha256': actual})
    cfg = load(protocol)
    parent_path = REPO / 'results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/msdf_v2_protocol.json'
    require(sha(parent_path) == cfg['parent_protocol_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', 'parent P0 recipe')
    parent = load(parent_path)
    for key in ('common_generator_training_target', 'MSDF_preserved_defaults', 'generation_target',
                'bank_seeds', 'detector_target', 'detector_exposure', 'primary_gates', 'utility_gates', 'final_eval_status'):
        require(cfg[key] == parent[key], 'STAGE25_FROZEN_BINDING_CHANGED', 'changed parent target: ' + key)
    for name, expected in cfg['server_contract']['historical_manifests'].items():
        require(len(expected) == 64, 'STAGE25_FROZEN_BINDING_CHANGED', name + ' authority hash')
    require(sha(protocol) == load(manifest)['sha256'][str(Path(protocol).resolve().relative_to(REPO)).replace('\\', '/')],
            'STAGE25_FROZEN_BINDING_CHANGED', 'protocol bytes')
    return cfg


def git_identity():
    try:
        return {'commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], stderr=subprocess.PIPE).decode().strip(),
                'available': True, 'uploaded_code_identity_also_hash_verified': True}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'available': False, 'identity_basis': 'exact frozen code manifest; manual file upload, NOT a verified Git checkout'}


def inspect_environment(cfg):
    versions = {}
    for name in cfg['server_contract']['package_versions']:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    try:
        versions['tqdm'] = md.version('tqdm')
    except md.PackageNotFoundError:
        versions['tqdm'] = None
    errors = [{'package': n, 'expected': v, 'actual': versions[n]} for n, v in cfg['server_contract']['package_versions'].items() if versions[n] != v]
    if not versions['tqdm']:
        errors.append({'package': 'tqdm', 'expected': 'installed; version recorded/frozen per run', 'actual': None})
    if sys.platform != 'linux':
        errors.append({'platform': sys.platform, 'expected': 'linux; Windows training forbidden'})
    if sys.version_info[:2] != (3, 10):
        errors.append({'python': sys.version, 'expected': '3.10 frozen Stage20B family'})
    result = {'python': sys.version, 'executable': sys.executable, 'platform': sys.platform,
              'project_root': str(REPO), 'git': git_identity(), 'package_versions': versions,
              'differences': errors, 'package_modification_count': 0, 'GPU_forward_test': 'NOT_RUN'}
    if errors:
        return result
    import torch
    result.update(torch_version=torch.__version__, CUDA_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4' or not torch.cuda.is_available():
        errors.append({'torch_build': torch.__version__, 'CUDA': torch.version.cuda,
                       'available': torch.cuda.is_available(), 'expected': '2.5.1+cu124 / CUDA12.4 / available GPU'})
    else:
        result['GPUs'] = [{'index': i, 'name': torch.cuda.get_device_name(i),
                          'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                         for i in range(torch.cuda.device_count())]
    return result


def installed_source_audit(p0):
    frozen = load(Path(p0) / 'stage20b_runtime_binding.json')['environment']['source_files']
    dist = md.distribution('ultralytics')
    rows = []
    for name, expected in frozen.items():
        path = Path(dist.locate_file(name)) if name.startswith('ultralytics/') else REPO / name
        actual = sha(path) if path.is_file() else None
        require(actual == expected['sha256'], 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', {'source': name, 'actual': actual})
        rows.append({'file': name, 'path': str(path), 'sha256': actual})
    return rows


def portable_name(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def real_rows(stage20a, real_root, split):
    ann = load(Path(stage20a) / 'annotation_provenance_v2.json')['records']
    hashes = {}
    for row in ann:
        if row['split_v2'] == split:
            hashes.setdefault(row['node_id'], set()).add(row['label_sha256'])
    rows = []
    for x in load(Path(stage20a) / 'real_source_registry.json')['records']:
        if x['asset_role'] != 'detector' or x['split_v2'] != split:
            continue
        require(len(hashes.get(x['node_id'], ())) == 1, 'STAGE25_FROZEN_BINDING_CHANGED', 'label identity')
        name = portable_name(x['current_path'])
        rows.append({'role_id': 'base_' + x['node_id'], 'node_id': x['node_id'],
            'image_path': str(Path(real_root) / 'images' / x['split_membership_old'] / name),
            'label_path': str(Path(real_root) / 'labels' / x['split_membership_old'] / (Path(name).stem + '.txt')),
            'image_sha256': x['file_sha256'], 'label_sha256': next(iter(hashes[x['node_id']])),
            'source_group': x['known_source_group_id'], 'split': split, 'width': x['width'], 'height': x['height']})
    return sorted(rows, key=lambda r: r['role_id'])


def bind_assets(a, cfg, out):
    pins = cfg['server_contract']['historical_manifests']
    verify_ledger(a.stage20a, 'frozen_artifact_manifest.json', pins['stage20a'])
    verify_ledger(a.p0, 'p0_frozen_artifact_manifest.json', pins['p0'])
    verify_ledger(a.r0, 'r0_artifact_manifest.json', pins['r0'])
    original = load(a.p0 / 'generator_train_allowlist.json')['inputs']
    normals = load(a.p0 / 'stage20b_normal_pool_freeze.json')['records']
    split = load(a.stage20a / 'v2_split_manifest.json')['group_splits']
    real_root = Path(a.real_root) if a.real_root else Path(original[0]['path']).parents[2]
    normal_root = Path(a.normal_root) if a.normal_root else Path(normals[0]['runtime_path']).parent
    require(real_root.is_dir() and normal_root.is_dir(), 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'supply actual --real_root/--normal_root; no guessed search')
    real = real_rows(a.stage20a, real_root, 'train')
    by_name = {portable_name(r['image_path']): r for r in real}
    require(len(by_name) == len(real) and len({Path(k).stem for k in by_name}) == len(real),
            'STAGE25_FROZEN_BINDING_CHANGED', 'ambiguous TRAIN parent names')
    def locate_real(path, label=False):
        name = portable_name(path)
        key = next((k for k in by_name if Path(k).stem == Path(name).stem), None) if label else name
        require(key in by_name, 'STAGE25_FROZEN_BINDING_CHANGED', 'unknown TRAIN parent ' + name)
        return by_name[key]['label_path' if label else 'image_path']
    view = out / 'p0_view'
    view.mkdir()
    for path in a.p0.iterdir():
        if path.is_file():
            shutil.copyfile(path, view / path.name)
    requests = {}
    train = deepcopy(original)
    for row in train:
        require(row['split'] == 'train' and split.get(row['source_group_id']) == 'train', 'STAGE25_FINAL_EVAL_ACCESS_FORBIDDEN', 'training group')
        row['path'] = locate_real(row['path'])
        row['full_label_path'] = locate_real(row['full_label_path'], True)
        requests[row['path']] = row['sha256']
        requests[row['full_label_path']] = row['full_label_sha256']
    normal_ids = {}
    for row in normals:
        require(row['split'] == 'train' and row['confirmatory_allowed'] and split.get(row['known_source_group']) == 'train',
                'STAGE25_FINAL_EVAL_ACCESS_FORBIDDEN', 'normal group')
        row['runtime_path'] = str(normal_root / portable_name(row['path']))
        requests[row['runtime_path']] = row['sha256']
        normal_ids[row['normal_id']] = row
    slots = load(a.p0 / 'stage20b_generation_slots.json')['slots']
    original_slots = deepcopy(slots)
    for row in slots:
        require(row['split'] == 'train' and split.get(row['source_group_id']) == 'train', 'STAGE25_FINAL_EVAL_ACCESS_FORBIDDEN', 'slot group')
        n = normal_ids[row['conditioning_normal_id']]
        require(row['normal_sha256'] == n['sha256'] and row['normal_source_group'] == n['known_source_group'],
                'STAGE25_FROZEN_BINDING_CHANGED', 'normal role')
        row['source_path'] = locate_real(row['source_path'])
        row['normal_path'] = n['runtime_path']
    rr = load(a.p0 / 'stage20b_realrepeat_manifest.json')
    for row in rr['records']:
        row['full_image_path'] = locate_real(row['full_image_path'])
        row['full_labels_path'] = locate_real(row['full_labels_path'], True)
    checks = []
    for path, expected in requests.items():
        actual = sha(path) if Path(path).is_file() else None
        checks.append({'path': path, 'expected_sha256': expected, 'actual_sha256': actual})
        if actual != expected:
            save(out / 'server_asset_hash_audit.json', {'status': 'FAIL', 'TRAIN_checks': checks,
                'first_mismatch': checks[-1], 'final_eval_content_read': 0, 'normal_filter_count': 0})
        require(actual == expected, 'STAGE25_FROZEN_BINDING_CHANGED', checks[-1])
    require(len(real) == 136 and len(requests) == 666 and len(normals) == 394
            and Counter(r['class'] for r in train) == {'flash': 75, 'black': 97}
            and Counter(s['class'] for s in slots) == {'flash': 40, 'black': 40}, 'STAGE25_FROZEN_BINDING_CHANGED', 'population')
    save(view / 'generator_train_allowlist.json', {**load(a.p0 / 'generator_train_allowlist.json'), 'inputs': train})
    save(view / 'stage20b_normal_pool_freeze.json', {**load(a.p0 / 'stage20b_normal_pool_freeze.json'), 'records': normals})
    save(view / 'stage20b_generation_slots.json', {**load(a.p0 / 'stage20b_generation_slots.json'), 'slots': slots})
    save(view / 'stage20b_realrepeat_manifest.json', rr)
    rng = {r['slot_id']: r['banks'] for r in load(a.p0 / 'stage20b_rng_manifest.json')['records']}
    for slot in original_slots:
        for bank, seed in cfg['bank_seeds'].items():
            expected = int(htext('Plastic_Bomo_Generation_V2' + slot['slot_id'] + str(seed))[:16], 16)
            require(rng[slot['slot_id']][bank]['sample_seed'] == expected, 'STAGE25_FROZEN_BINDING_CHANGED', 'slot-derived seed')
    weights = load(a.p0 / 'sd2_base_weight_binding.json')['files']
    for name, row in weights.items():
        require((a.base_model / name).is_file() and sha(a.base_model / name) == row['sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', 'public base ' + name)
    for component in ('unet', 'vae', 'text_encoder'):
        actual = {p.relative_to(a.base_model).as_posix() for p in (a.base_model / component).iterdir() if p.suffix in ('.bin', '.safetensors')}
        expected = {n for n in weights if n.startswith(component + '/') and n.endswith('.safetensors')}
        require(actual == expected, 'STAGE25_FROZEN_BINDING_CHANGED', 'ambiguous public weights ' + component)
    require(a.detector_model.is_file() and sha(a.detector_model) == cfg['detector_target']['initialization_sha256'],
            'STAGE25_FROZEN_BINDING_CHANGED', 'public detector initialization')
    controller_cfg = load(REPO / 'configs/plastic_bomo_stage20b_p0.json')
    require(controller_cfg['detector'] == cfg['detector_target']['training'], 'STAGE25_FROZEN_BINDING_CHANGED', 'detector protocol differs')
    save(out / 'controller_cfg.json', controller_cfg)
    result = {'status': 'PASS', 'p0_original': str(a.p0), 'p0_view': str(view), 'real_root': str(real_root.resolve()),
        'normal_root': str(normal_root.resolve()), 'TRAIN_checks': checks, 'TRAIN_normals': 394, 'TRAIN_real': 136,
        'TRAIN_class_instances': {'flash': 75, 'black': 97}, 'slot_count': 80,
        'normal_filter_count': 0, 'path_relocation_only': True, 'final_eval_content_read': 0,
        'physical_source_complete': False, 'view_binding': tree_binding(view, out)}
    save(out / 'server_asset_hash_audit.json', result)
    return result


def baseline_audit(a, cfg):
    """No final_eval content. Invalid/missing historical comparator triggers rebuild."""
    result = {'status': 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'reuse': False, 'errors': []}
    try:
        verify_ledger(a.baseline_run, 'formal_result_artifact_manifest.json', cfg['server_contract']['historical_manifests']['baseline'])
        formal = load(a.baseline_run / 'formal_protocol_binding.json')
        require(formal['status'] == 'PASS' and formal['P0_binding']['p0_manifest_sha256'] == cfg['server_contract']['historical_manifests']['p0']
                and formal['P0_binding']['protocol_sha256'] == sha(REPO / 'configs/plastic_bomo_stage20b_p0.json')
                and formal['P0_binding']['package_versions'] == cfg['server_contract']['package_versions']
                and formal['runtime']['torch_version'] == cfg['server_contract']['torch_build']
                and formal['runtime']['CUDA_version'] == cfg['server_contract']['CUDA'],
                'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'historical runtime/data identity')
        for name in ('tools/stage20b_formal_common.py', 'tools/train_plastic_bomo_stage20b_generator.py',
                     'tools/generate_plastic_bomo_stage20b_banks.py', 'tools/train_plastic_bomo_stage20b_detector.py',
                     'tools/evaluate_plastic_bomo_stage20b.py'):
            require(sha(REPO / name) == formal['formal_code_sha256'][name], 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', name)
        generators = load(a.baseline_run / 'generator_checkpoint_binding.json')
        require(generators['status'] == 'FROZEN' and set(generators['classes']) == {'flash', 'black'}, 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'two Raw classes')
        for row in generators['classes'].values():
            verify_tree(a.baseline_run, row['checkpoint'])
            require(row['successful_updates'] == 2000, 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'generator budget')
        detectors = load(a.baseline_run / 'detector_checkpoint_binding.json')['arms']
        require(set(detectors) == {'RR', *BANKS}, 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'four detector arms')
        original_slots = {r['slot_id']: r for r in load(a.p0 / 'stage20b_generation_slots.json')['slots']}
        for bank in BANKS:
            mf = load(a.baseline_run / ('formal_bank_' + bank + '_manifest.json'))
            require(mf['status'] == 'COMPLETE' and len(mf['records']) == 80 and not mf['quality_selection'] and mf['reroll_count'] == 0
                    and mf['generation_protocol_sha256'] == sha(a.p0 / 'stage20b_generation_protocol.json')
                    and {r['slot_id'] for r in mf['records']} == set(original_slots),
                    'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'raw population')
            for r in mf['records']:
                require(r['frozen_slot'] == original_slots[r['slot_id']]
                        and r['sample_seed'] == int(htext('Plastic_Bomo_Generation_V2' + r['slot_id'] + str(cfg['bank_seeds'][bank]))[:16], 16),
                        'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'slot identity')
                for field in ('image', 'label'):
                    require(sha(a.baseline_run / r[field + '_path']) == r[field + '_sha256'], 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'raw asset')
        metrics = {}
        for arm, row in detectors.items():
            training = load(a.baseline_run / ('detector_arm_' + arm + '_training.json'))
            require(training['status'] == 'COMPLETE' and training['actual_draws'] == 32400
                    and training['attempted_optimizer_updates'] == 600 and training['epochs'] == 150
                    and training['training_protocol_sha256'] == sha(a.p0 / 'stage20b_detector_protocol.json')
                    and training['order_spec_sha256'] == sha(a.p0 / 'stage20b_training_order_spec.json'),
                    'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'detector exposure/protocol')
            require(row['epochs'] == 150 and sha(a.baseline_run / row['path']) == row['sha256'], 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'last.pt')
            name = 'RR' if arm == 'RR' else 'Raw' + arm
            metrics[arm] = load(a.baseline_run / (name + '_metrics.json'))
            require(metrics[arm]['checkpoint_sha256'] == row['sha256']
                    and metrics[arm]['evaluation_protocol_sha256'] == sha(a.p0 / 'stage20b_detector_protocol.json'),
                    'RAW_COMPARATOR_REUSE_NOT_QUALIFIED', 'evaluation contract')
        result.update(status='RAW_COMPARATOR_REUSE_QUALIFIED', reuse=True, metrics=metrics,
                      authority=str(a.baseline_run), controller_sha256=sha(REPO / 'tools/stage20b_detector_controller.py'),
                      untouched_final_test_claim=False)
    except (OSError, KeyError, ValueError, Stop) as error:
        result['errors'].append(str(error))
    return result


def binding(run):
    run = Path(run).resolve()
    b = load(run / 'stage25_run_binding.json')
    cfg = source_check(Path(b['stage25_protocol']), Path(b['code_manifest']))
    require(sha(b['stage25_protocol']) == b['protocol_sha256'] and sha(b['code_manifest']) == b['code_manifest_sha256'],
            'STAGE25_FROZEN_BINDING_CHANGED', 'run recipe changed')
    runtime = Path(b['runtime_dir'])
    verify_ledger(runtime, 'stage25_runtime_artifact_manifest.json', b['runtime_artifact_manifest_sha256'])
    require(sha(runtime / 'stage25_runtime_qualification.json') == b['qualification_sha256']
            and load(runtime / 'stage25_runtime_qualification.json')['status'] == 'PASS', 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'qualification barrier')
    verify_tree(runtime, load(runtime / 'server_asset_hash_audit.json')['view_binding'])
    expected_env = load(runtime / 'stage25_server_environment.json')['package_versions']
    for package, version in expected_env.items():
        require(md.version(package) == version, 'STAGE25_FROZEN_BINDING_CHANGED', package)
    # No final_eval pixels/labels: only TRAIN assets and public initializations.
    for row in load(runtime / 'server_asset_hash_audit.json')['TRAIN_checks']:
        require(sha(row['path']) == row['expected_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', row['path'])
    for name, row in load(Path(b['p0_view']) / 'sd2_base_weight_binding.json')['files'].items():
        require(sha(Path(b['base_model']) / name) == row['sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', 'public base ' + name)
    require(sha(b['detector_model']) == cfg['detector_target']['initialization_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', 'public detector')
    return b, cfg


def safety_binding(runtime):
    b = load(Path(runtime) / 'stage25_runtime_binding.json')
    cfg = source_check(Path(b['stage25_protocol']), Path(b['code_manifest']))
    require(not load(Path(runtime) / 'stage25_server_environment.json')['differences'],
            'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'environment barrier')
    require(sha(Path(runtime) / 'stage25_server_environment.json') == b['environment_sha256'],
            'STAGE25_FROZEN_BINDING_CHANGED', 'environment audit bytes')
    for package, version in load(Path(runtime) / 'stage25_server_environment.json')['package_versions'].items():
        require(md.version(package) == version, 'STAGE25_FROZEN_BINDING_CHANGED', package)
    verify_tree(Path(runtime), load(Path(runtime) / 'server_asset_hash_audit.json')['view_binding'])
    installed_source_audit(b['p0_view'])
    return b, cfg


@contextmanager
def guard(run, evaluation=False, safety=False):
    b, cfg = safety_binding(run) if safety else binding(run)
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden, final_eval_content_forbidden, set_gpu_runtime, optimizer_steps_forbidden
    counts = {'forbidden_final_eval_open_attempts': 0, 'forbidden_optimizer_step_calls': 0}
    offline()
    with ExitStack() as stack:
        stack.enter_context(network_forbidden())
        if not evaluation:
            stack.enter_context(final_eval_content_forbidden(Path(b['stage20a']), Path(b['p0_view']), counts))
        import torch
        set_gpu_runtime(torch, b['device'])
        require('RTX 3090' in torch.cuda.get_device_name(), 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'frozen RTX3090 runtime required')
        env = load((Path(run) if safety else Path(b['runtime_dir'])) / 'stage25_server_environment.json')
        require(torch.__version__ == cfg['server_contract']['torch_build'] and torch.version.cuda == cfg['server_contract']['CUDA']
                and torch.backends.cudnn.version() == env['cudnn_version'], 'STAGE25_FROZEN_BINDING_CHANGED', 'Torch/CUDA/cuDNN drift')
        installed_source_audit(b['p0_view'])
        if safety:
            stack.enter_context(optimizer_steps_forbidden(torch, counts))
        yield torch
    require(counts['forbidden_final_eval_open_attempts'] == counts['forbidden_optimizer_step_calls'] == 0,
            'STAGE25_FINAL_EVAL_ACCESS_FORBIDDEN', counts)


def claim(run, name):
    path = Path(run) / 'logs' / (name + '_claim.json')
    require(not path.exists(), 'STAGE25_PHASE_ALREADY_ATTEMPTED', name + '; no automatic retry/resume')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical({'started_at': now(), 'phase': name}))


def update(run, **values):
    path = Path(run) / 'stage25_status.json'
    row = load(path)
    row.update(values)
    if 'role' not in row:  # main MSDF run; fallback run has separate Raw/RR counters
        row['generator_training_count'] = sum(bool(row.get('msdf_' + c + '_training_started')) for c in ('flash', 'black'))
        row['detector_training_count'] = sum(bool(row.get('msdf_detector_' + k + '_started')) for k in BANKS)
    else:
        row['detector_training_count'] = sum(bool(row.get('comparator_detector_' + k + '_started')) for k in ('RR', *BANKS))
    save(path, row)


def metrics_summary(msdf, raw, rr, cfg):
    import math
    import statistics as stats
    all_metrics = [rr, *raw.values(), *msdf.values()]
    require(set(msdf) == set(raw) == set(BANKS), 'STAGE25_METRICS_INVALID', 'exact ABC banks required')
    values = [m['map5095'] for m in all_metrics] + [m['per_class'][c]['ap5095'] for m in all_metrics for c in ('flash', 'black')]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'STAGE25_METRICS_INVALID', 'finite mAP/class AP required')
    delta = {k: (msdf[k]['map5095'] - raw[k]['map5095']) * 100 for k in BANKS}
    utility = {k: (msdf[k]['map5095'] - rr['map5095']) * 100 for k in BANKS}
    classes = {c: {'msdf_minus_raw_pp': {k: (msdf[k]['per_class'][c]['ap5095'] - raw[k]['per_class'][c]['ap5095']) * 100 for k in BANKS},
                   'msdf_minus_rr_pp': {k: (msdf[k]['per_class'][c]['ap5095'] - rr['per_class'][c]['ap5095']) * 100 for k in BANKS}}
               for c in ('flash', 'black')}
    for c in classes:
        classes[c]['mean_msdf_minus_raw_pp'] = stats.mean(classes[c]['msdf_minus_raw_pp'].values())
        classes[c]['mean_msdf_minus_rr_pp'] = stats.mean(classes[c]['msdf_minus_rr_pp'].values())
    wins, mean = sum(v > 0 for v in delta.values()), stats.mean(delta.values())
    gates = {'A': mean >= cfg['primary_gates']['A_mean_msdf_minus_raw_pp']['threshold'],
             'B': wins >= cfg['primary_gates']['B_strict_matched_bank_wins']['threshold'],
             'C': all(classes[c]['mean_msdf_minus_raw_pp'] >= cfg['primary_gates']['C_' + c + '_mean_delta_pp']['threshold'] for c in classes)}
    supported = all(gates.values())
    utility_wins = sum(v > 0 for v in utility.values())
    utility_ok = (stats.mean(utility.values()) >= cfg['utility_gates']['mean_msdf_minus_RR_pp']['threshold']
                  and utility_wins >= cfg['utility_gates']['strict_MSDF_bank_wins_vs_RR']['threshold'])
    status = ('MSDF_GENERATOR_IMPROVEMENT_NOT_CONFIRMED' if not supported else
              'MSDF_V2_REQUALIFICATION_PASS' if utility_ok else 'MSDF_GENERATOR_IMPROVEMENT_SUPPORTED_BUT_UTILITY_NOT_CONFIRMED')
    return {'status': status, 'gates': gates, 'MSDF_GENERATOR_IMPROVEMENT_SUPPORTED': supported,
        'MSDF_AUGMENTATION_UTILITY_CONFIRMED': utility_ok, 'RR_map': rr['map5095'],
        **{'raw_' + k + '_map': raw[k]['map5095'] for k in BANKS}, **{'msdf_' + k + '_map': msdf[k]['map5095'] for k in BANKS},
        **{'delta_' + k + '_msdf_vs_raw_pp': delta[k] for k in BANKS},
        'mean_msdf_vs_raw_pp': mean, 'msdf_vs_raw_wins': wins, 'mean_msdf_vs_rr_pp': stats.mean(utility.values()),
        'msdf_vs_rr_wins': utility_wins, 'Flash_mean_msdf_vs_raw_pp': classes['flash']['mean_msdf_minus_raw_pp'],
        'Black_mean_msdf_vs_raw_pp': classes['black']['mean_msdf_minus_raw_pp'], 'classwise': classes,
        'MSDF_bank_sample_SD': stats.stdev(msdf[k]['map5095'] * 100 for k in BANKS),
        'Raw_bank_sample_SD': stats.stdev(raw[k]['map5095'] * 100 for k in BANKS),
        'bank_SD_unit': 'pp; sample ddof1; NOT detector-seed uncertainty', 'MSDF_V2_REQUALIFICATION_FAILED': not (supported and utility_ok),
        'BootstrapGuard': 0, 'DWBG': 0, 'selector': 0, 'DeepPCB': 0, 'full_experiment_complete': True}
