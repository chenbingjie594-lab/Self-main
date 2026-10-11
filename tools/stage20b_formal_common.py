"""Formal Stage20B only. Immutable P0/R0 contracts are imported, never edited."""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stage20b_runtime_contract import (Stop, binding_recheck, canonical, htext, load,
                                      require, save, sha, verify_ledger)

R0_MANIFEST_SHA = '282a381e2dc402f372204c45df3cf7d137d9e437b0b2aac0835ec32643b59f85'
R0_CODE = {
    'tools/qualify_plastic_bomo_stage20b_r0.py': '255c979b44d68feabe454508f546ae93fbea70fb15a8da5ef12b4c750f4671a9',
    'tools/stage20b_runtime_contract.py': '8b12745e13446da11fc2158c2babc657125e98381925d70b06ca52d959bbbfb1',
    'tools/stage20b_detector_controller.py': 'df482a55305301d2b33dc711a7cef7811a71c9e7b0d51d5dc26624365e24697f',
}
STATES = ('INIT', 'FLASH_GENERATOR_TRAINED', 'BLACK_GENERATOR_TRAINED',
          'GENERATOR_CHECKPOINTS_FROZEN', 'BANK_A_GENERATED', 'BANK_B_GENERATED',
          'BANK_C_GENERATED', 'SYNTHETIC_BANKS_VALIDATED', 'DETECTOR_RR_TRAINED',
          'DETECTOR_A_TRAINED', 'DETECTOR_B_TRAINED', 'DETECTOR_C_TRAINED',
          'ALL_DETECTORS_FROZEN', 'FINAL_EVAL_COMPLETE')
FORMAL_FILES = (
    'stage20b_formal_common.py', 'train_plastic_bomo_stage20b_generator.py',
    'generate_plastic_bomo_stage20b_banks.py', 'train_plastic_bomo_stage20b_detector.py',
    'evaluate_plastic_bomo_stage20b.py', 'audit_plastic_bomo_stage20b.py',
    'run_plastic_bomo_stage20b_formal.py')
ARTIFACTS = (
    'generator_training_flash', 'generator_training_black', 'generator_checkpoint_binding',
    'trained_checkpoint_determinism_audit', 'formal_bank_A_manifest', 'formal_bank_B_manifest',
    'formal_bank_C_manifest', 'formal_synthetic_manifest', 'formal_synthetic_validity',
    'detector_arm_RR_training', 'detector_arm_A_training', 'detector_arm_B_training',
    'detector_arm_C_training', 'detector_checkpoint_binding', 'detector_compute_audit',
    'final_eval_access_audit', 'RR_metrics', 'RawA_metrics', 'RawB_metrics', 'RawC_metrics',
    'stage20b_headroom', 'stage20b_classwise', 'stage20b_generation_nuisance')


def now():
    return datetime.now(timezone.utc).isoformat()


def offline():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', YOLO_AUTOINSTALL='false',
                      YOLO_OFFLINE='true', NO_ALBUMENTATIONS_UPDATE='1', TOKENIZERS_PARALLELISM='false')


def initial_status():
    fields = ('RR_map5095', 'A_map5095', 'B_map5095', 'C_map5095', 'delta_A_pp',
              'delta_B_pp', 'delta_C_pp', 'mean_delta_pp', 'std_delta_pp', 'wins_vs_RR',
              'Flash_mean_delta_pp', 'Black_mean_delta_pp', 'generation_bank_std_pp',
              'generation_bank_range_pp')
    return {'status': 'FORMAL_RUNNING', 'state': 'INIT', 'Gate_A': 'NOT_RUN',
            'Gate_B': 'NOT_RUN', 'Gate_C': 'NOT_RUN', **dict.fromkeys(fields),
            'TDCRG_DEVELOPMENT_AUTHORIZED': False, 'TDCRG_implementation': 0,
            'generator_flash_successful_updates': 0, 'generator_black_successful_updates': 0,
            **{f'detector_{a}_successful_updates': 0 for a in ('RR', 'A', 'B', 'C')},
            'official_final_eval_rounds': 0, 'DeepPCB': 0, 'BootstrapGuard': 0,
            'generator_training_count': 0, 'synthetic_generation_count': 0,
            'checkpoint_sanity_sampling_count': 0, 'detector_training_count': 0,
            'full_experiment_complete': False, 'automatic_rescue': False,
            'physical_source_complete': False, 'stage20c_auto_start': False,
            'started_at': now(), 'state_history': [{'state': 'INIT', 'at': now()}]}


def transition(run, expected, following):
    s = load(run / 'stage20b_status.json')
    require(s['status'] == 'FORMAL_RUNNING' and s['state'] == expected
            and STATES.index(following) == STATES.index(expected) + 1,
            'FORMAL_STATE_SEQUENCE_INVALID', {'actual': s['state'], 'expected': expected, 'next': following})
    s['state'] = following
    s['state_history'].append({'state': following, 'at': now()})
    save(run / 'stage20b_status.json', s)


def update_status(run, **values):
    s = load(run / 'stage20b_status.json')
    s.update(values)
    save(run / 'stage20b_status.json', s)


def tree_binding(path, run):
    path = Path(path).resolve()
    rows = {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
    require(bool(rows), 'FORMAL_CHECKPOINT_INVALID', str(path))
    return {'path': path.relative_to(run).as_posix(), 'sha256': rows,
            'tree_sha256': htext(canonical(rows))}


def verify_tree(run, row):
    path = (run / row['path']).resolve()
    require(path.is_relative_to(run) and path.is_dir(), 'FORMAL_FROZEN_BINDING_CHANGED', 'checkpoint path: ' + str(path))
    actual = tree_binding(path, run)
    require(actual == row, 'FORMAL_FROZEN_BINDING_CHANGED', row['path'])
    return path


def verified_copy(source, dest, digest):
    source, dest = Path(source), Path(dest)
    require(source.is_file() and sha(source) == digest, 'FORMAL_FROZEN_BINDING_CHANGED', str(source))
    require(not dest.exists(), 'FORMAL_OUTPUT_EXISTS', str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    require(sha(dest) == digest, 'FORMAL_FROZEN_BINDING_CHANGED', 'copy: ' + str(dest))


def formal_binding(a):
    try:
        old = binding_recheck(REPO, a.stage20a, a.p0, a.protocol, a.base_model, a.detector_model)
        r0 = verify_ledger(a.r0, 'r0_artifact_manifest.json', R0_MANIFEST_SHA)
        for name, expected in R0_CODE.items():
            require(sha(REPO / name) == expected, 'FORMAL_FROZEN_BINDING_CHANGED', name)
        status = load(a.r0 / 'stage20b_r0_status.json')
        require(status['status'] == 'STAGE20B_RUNTIME_QUALIFICATION_PASS'
                and status['STAGE20B_FORMAL_EXECUTION_READY'], 'FORMAL_FROZEN_BINDING_CHANGED', 'R0 authority')
        from qualify_plastic_bomo_stage20b_r0 import set_gpu_runtime
        import torch
        set_gpu_runtime(torch, 'cuda:' + a.device)
        actual = {'python': sys.version, 'torch_version': torch.__version__, 'CUDA_version': torch.version.cuda,
                  'cudnn_version': torch.backends.cudnn.version(), 'device_name': torch.cuda.get_device_name(),
                  'capability': list(torch.cuda.get_device_capability())}
        runtime = load(a.r0 / 'runtime_environment.json')
        require(all(actual[k] == runtime[k] for k in actual), 'FORMAL_FROZEN_BINDING_CHANGED', actual)
    except Stop as e:
        raise Stop('FORMAL_FROZEN_BINDING_CHANGED', e.detail) from e
    try:
        commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], stderr=subprocess.PIPE).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'UNAVAILABLE_ON_SERVER_MANUAL_FILE_UPLOAD'
    code = {f'tools/{n}': sha(REPO / 'tools' / n) for n in FORMAL_FILES}
    return {'status': 'PASS', 'frozen': True, 'stage20a': str(a.stage20a), 'p0': str(a.p0),
            'r0': str(a.r0), 'protocol': str(a.protocol), 'base_model': str(a.base_model),
            'detector_model': str(a.detector_model), 'device': 'cuda:' + a.device,
            'P0_binding': old, 'R0_artifacts': r0, 'R0_code_sha256': R0_CODE,
            'formal_code_sha256': code, 'runtime': actual, 'git_commit': commit,
            'pid': int(os.environ.get('STAGE20B_PARENT_PID', os.getpid())),
            'runtime_binding_pid': os.getpid(), 'started_at': now(), 'result_directory': str(a.output),
            'launcher_log': os.environ.get('STAGE20B_LAUNCHER_LOG'),
            'hyperparameters': 'read immutable P0 JSON; no equivalent substitutions',
            'final_eval_content_read_at_binding': 0}


def context(run, states):
    run = Path(run).resolve()
    b = load(run / 'formal_protocol_binding.json')
    s = load(run / 'stage20b_status.json')
    require(b['status'] == 'PASS' and s['status'] == 'FORMAL_RUNNING' and s['state'] in states,
            'FORMAL_STATE_SEQUENCE_INVALID', {'state': s['state'], 'expected': states})
    require(os.environ.get('STAGE20B_FORMAL_BINDING_SHA256') == sha(run / 'formal_protocol_binding.json'),
            'FORMAL_FROZEN_BINDING_CHANGED', 'parent-bound one-shot formal protocol envelope')
    # Every child binds the new executable bytes; authoritative inputs have their own ledgers.
    for name, expected in {**b['R0_code_sha256'], **b['formal_code_sha256']}.items():
        path = REPO / name
        require(path.is_file() and sha(path) == expected, 'FORMAL_FROZEN_BINDING_CHANGED', name)
    from stage20b_runtime_contract import STAGE20A_MANIFEST_SHA, P0_MANIFEST_SHA
    stage20a, p0, r0 = map(Path, (b['stage20a'], b['p0'], b['r0']))
    verify_ledger(stage20a, 'frozen_artifact_manifest.json', STAGE20A_MANIFEST_SHA)
    verify_ledger(p0, 'p0_frozen_artifact_manifest.json', P0_MANIFEST_SHA)
    verify_ledger(r0, 'r0_artifact_manifest.json', R0_MANIFEST_SHA)
    return b, p0, stage20a


@contextmanager
def runtime_guard(run, b, p0, stage20a, evaluation=False):
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden, final_eval_content_forbidden, set_gpu_runtime
    offline()
    counts = {'forbidden_final_eval_open_attempts': 0}
    with ExitStack() as stack:
        stack.enter_context(network_forbidden())
        if not evaluation:
            stack.enter_context(final_eval_content_forbidden(stage20a, p0, counts))
        import torch
        set_gpu_runtime(torch, b['device'])
        yield torch
    require(counts['forbidden_final_eval_open_attempts'] == 0, 'FORMAL_FINAL_EVAL_ACCESS_FORBIDDEN', counts)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--run', type=Path, required=True)
    return p


def child_main(function, description, extra=None):
    p = parser(description)
    if extra:
        extra(p)
    a = p.parse_args()
    a.run = a.run.resolve()
    try:
        function(a)
    except Exception as error:
        reason = error.status if isinstance(error, Stop) else getattr(a, 'failure_reason', 'FORMAL_EXECUTION_FAILURE')
        if reason == 'R0_FINAL_EVAL_CONTENT_ACCESS_FORBIDDEN':
            reason = 'FORMAL_FINAL_EVAL_ACCESS_FORBIDDEN'
        elif reason.startswith('R0_') or reason == 'FROZEN_RUNTIME_BINDING_CHANGED':
            reason = 'FORMAL_FROZEN_BINDING_CHANGED'
        if (a.run / 'stage20b_status.json').is_file():
            update_status(a.run, status='STOPPED_' + reason, reason=reason, detail=str(error), stopped_at=now())
        print(traceback.format_exc(), flush=True)
        raise


def real_assets(stage20a, p0, split):
    """Metadata lookup only. No pixel/label opens (including final_eval) here."""
    roots = {Path(x['path']).parents[2] for x in load(p0 / 'generator_train_allowlist.json')['inputs']}
    require(len(roots) == 1, 'FORMAL_FROZEN_BINDING_CHANGED', 'real root')
    root = next(iter(roots))
    ann = load(stage20a / 'annotation_provenance_v2.json')['records']
    label_hashes = {}
    for row in ann:
        if row['split_v2'] == split:
            label_hashes.setdefault(row['node_id'], set()).add(row['label_sha256'])
    rows = []
    for x in load(stage20a / 'real_source_registry.json')['records']:
        if x['asset_role'] != 'detector' or x['split_v2'] != split:
            continue
        labels = label_hashes.get(x['node_id'], set())
        require(len(labels) == 1, 'FORMAL_FROZEN_BINDING_CHANGED', 'label hash: ' + x['node_id'])
        name = Path(x['portable_locator']['relative_path']).name
        rows.append({'role_id': 'base_' + x['node_id'], 'node_id': x['node_id'],
                     'image_path': str(root / x['portable_locator']['relative_path']),
                     'label_path': str(root / 'labels' / x['split_membership_old'] / (Path(name).stem + '.txt')),
                     'image_sha256': x['file_sha256'], 'label_sha256': next(iter(labels)),
                     'source_group': x['known_source_group_id'], 'split': split,
                     'width': x['width'], 'height': x['height']})
    return sorted(rows, key=lambda x: x['role_id'])


def validate_bbox(slot):
    g = slot['geometry']
    box = g['final_bbox_normalized_xywh']
    require(g['valid'] and box == slot['bbox'] and len(box) == 4 and all(math.isfinite(v) for v in box),
            'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'bbox metadata: ' + slot['slot_id'])
    x, y, w, h = box
    require(w > 0 and h > 0 and x - w / 2 >= 0 and y - h / 2 >= 0
            and x + w / 2 <= 1 and y + h / 2 <= 1,
            'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'bbox bounds: ' + slot['slot_id'])
    return box


def synthetic_label(slot):
    box = validate_bbox(slot)
    return str({'flash': 0, 'black': 1}[slot['class']]) + ' ' + ' '.join(format(v, '.17g') for v in box) + '\n'


def summarize_metrics(metrics, gate):
    import statistics
    rr = metrics['RR']['map5095']
    banks = [metrics['Raw' + k]['map5095'] for k in ('A', 'B', 'C')]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in [rr, *banks]), 'FORMAL_METRICS_INVALID', 'mAP')
    delta = [(v - rr) * 100 for v in banks]
    classwise = {}
    for cls in ('flash', 'black'):
        base = metrics['RR']['per_class'][cls]['ap5095']
        values = [(metrics['Raw' + k]['per_class'][cls]['ap5095'] - base) * 100 for k in ('A', 'B', 'C')]
        require(all(math.isfinite(v) for v in values), 'FORMAL_METRICS_INVALID', cls)
        classwise[cls] = {'delta_pp': dict(zip(('A', 'B', 'C'), values)), 'mean_delta_pp': statistics.mean(values)}
    mean, wins = statistics.mean(delta), sum(v > rr for v in banks)
    gates = {'Gate_A': mean >= gate['A_mean_raw_bank_minus_RR_ge_pp'],
             'Gate_B': wins >= gate['B_strict_bank_wins_ge'],
             'Gate_C': all(x['mean_delta_pp'] >= gate['C_every_class_mean_delta_ge_pp'] for x in classwise.values())}
    passed = all(gates.values())
    return {'status': 'GENERATOR_BASELINE_HEADROOM_CONFIRMED' if passed else gate['failure_status'],
            **gates, 'RR_map5095': rr, **{k + '_map5095': v for k, v in zip(('A', 'B', 'C'), banks)},
            **{'delta_' + k + '_pp': v for k, v in zip(('A', 'B', 'C'), delta)},
            'mean_delta_pp': mean, 'std_delta_pp': statistics.stdev(delta),
            'min_delta_pp': min(delta), 'max_delta_pp': max(delta), 'wins_vs_RR': wins,
            'Flash_mean_delta_pp': classwise['flash']['mean_delta_pp'],
            'Black_mean_delta_pp': classwise['black']['mean_delta_pp'],
            'generation_bank_std_pp': statistics.stdev(banks) * 100,
            'generation_bank_range_pp': (max(banks) - min(banks)) * 100,
            'TDCRG_DEVELOPMENT_AUTHORIZED': passed, 'classwise': classwise,
            'banks_are_detector_seed_replicates': False}
