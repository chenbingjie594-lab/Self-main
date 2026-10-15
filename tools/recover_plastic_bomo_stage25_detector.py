"""Explicit manual recovery of an interrupted, uncheckpointed Stage25 detector A.

The original frozen executables/recipe are not edited. Reuse both step2000
generators and all 240 images; restart A/B/C from the original initialization.
No epoch resume, automatic retry, sampling, selection or budget compensation.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import sys
import traceback

from stage25_msdf_contract import (BANKS, REPO, Stop, binding, canonical, htext,
    load, now, require, save, sha, verified_copy)
from stage20b_formal_common import verify_tree
import run_plastic_bomo_stage25_msdf_v2 as runner

SCRIPT = Path(__file__).resolve()
ROOT = runner.ROOT


def inside(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    path = (root / relative).resolve()
    require(not relative.is_absolute() and path.is_relative_to(root),
            'STAGE25_RECOVERY_PATH_INVALID', str(name))
    return path


def check_processes(proc_root=Path('/proc')):
    """Read-only Linux check; never signal another task or trust an old PID."""
    require(sys.platform == 'linux', 'STAGE25_RECOVERY_SERVER_REQUIRED',
            '仅在原 Linux 服务器执行；Windows 只做静态测试。')
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise Stop('STAGE25_RECOVERY_PROCESS_CHECK_UNAVAILABLE', str(error)) from error
        scripts = {'run_plastic_bomo_stage25_msdf_v2.py', SCRIPT.name}
        if any(Path(arg.decode('utf-8', 'replace')).name in scripts for arg in argv if arg):
            raise Stop('STAGE25_RECOVERY_TASK_STILL_RUNNING',
                       {'reason': '已有 Stage25 任务运行，不能并行启动恢复任务。', 'pid': int(proc.name)})


def interrupted_a(source):
    """Require pre-evaluation A interruption, not a failed scientific/numeric gate."""
    status = load(source / 'stage25_status.json')
    require(status['status'] in ('FORMAL_RUNNING', 'STAGE25_INTERRUPTED')
            and not status['full_experiment_complete'] and status['official_final_eval_rounds'] == 0,
            'STAGE25_RECOVERY_NOT_INTERRUPTED_A', status['status'])
    forbidden = ('detector_checkpoint_binding.json', 'final_eval_access_audit.json',
                 'unified_evaluation_checkpoint_binding.json', 'final_eval_inputs',
                 'logs/unified_final_eval_claim.json', 'logs/evaluate.log',
                 'stage25_tradeoff_summary.json', 'STAGE25_REPORT.md')
    require(not any((source / name).exists() for name in forbidden),
            'STAGE25_RECOVERY_ALREADY_FROZEN_OR_EVALUATED', str(source))
    for bank in BANKS:
        require(not (source / f'detector_{bank}/last.pt').exists()
                and not (source / f'MSDF_{bank}_metrics.json').exists(),
                'STAGE25_RECOVERY_CHECKPOINT_EXISTS', bank)
    for bank in ('B', 'C'):
        require(not (source / f'detector_arm_{bank}_training.json').exists()
                and not (source / f'logs/msdf_detector_{bank}_claim.json').exists()
                and not status.get(f'msdf_detector_{bank}_started'),
                'STAGE25_RECOVERY_NOT_INTERRUPTED_A', 'B/C 已尝试，不适用这个恢复入口。')
    audit = load(source / 'detector_arm_A_training.json')
    require(audit['arm'] == 'A' and audit['status'] in ('RUNNING', 'FAILED')
            and 0 <= audit['epochs'] < 150 and audit['actual_draws'] > 0
            and audit['nonfinite_losses'] == 0 and audit['official_final_eval_content_read'] == 0,
            'STAGE25_RECOVERY_NOT_INTERRUPTED_A', 'A 无完整 checkpoint；不允许借中断恢复绕过数值失败。')
    if (source / 'worker_failure.json').exists():
        failure = load(source / 'worker_failure.json')
        require(failure['status'] == 'STAGE25_INTERRUPTED',
                'STAGE25_RECOVERY_UNEXPECTED_FAILURE', failure)
    return status, audit


def precheck(source):
    """Read/hash only. No model import, CUDA initialization or final_eval content."""
    status, interrupted = interrupted_a(source)
    b, cfg = binding(source)  # Same original source envelope, runtime, TRAIN and public weights.
    require(load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')['reuse'],
            'STAGE25_RECOVERY_COMPARATOR_REBUILD_OUT_OF_SCOPE', '此入口仅用于已合格 Raw/RR 复用的中断任务。')
    files = {}

    def add(name, expected=None):
        path = inside(source, name)
        require(path.is_file(), 'STAGE25_RECOVERY_ASSET_MISSING', name)
        actual = sha(path)
        require(expected is None or actual == expected, 'STAGE25_FROZEN_BINDING_CHANGED', name)
        require(name not in files or files[name] == actual, 'STAGE25_FROZEN_BINDING_CHANGED', name)
        files[name] = actual

    for name in ('stage25_run_binding.json', 'msdf_v2_protocol.json', 'server_code_manifest.json',
                 'msdf_historical_component_audit.json', 'msdf_selector_separation_audit.json',
                 'generator_checkpoint_binding.json', 'msdf_checkpoint_binding.json',
                 'formal_synthetic_manifest.json', 'formal_synthetic_validity.json',
                 'msdf_generation_validity.json'):
        add(name)
    require(files['msdf_v2_protocol.json'] == b['protocol_sha256']
            and files['server_code_manifest.json'] == b['code_manifest_sha256'],
            'STAGE25_FROZEN_BINDING_CHANGED', 'original frozen protocol/code copy')
    generators = load(source / 'generator_checkpoint_binding.json')
    require(generators['status'] == 'FROZEN' and generators['both_frozen_before_sampling']
            and set(generators['classes']) == {'flash', 'black'}
            and load(source / 'msdf_checkpoint_binding.json') == generators,
            'STAGE25_GENERATOR_BARRIER_FAILED', 'both step2000 checkpoints')
    for cls, row in generators['classes'].items():
        name = f'msdf_{cls}_training_audit.json'
        trained = load(source / name)
        require(trained['status'] == 'COMPLETE' and trained['checkpoint_saved']
                and trained['successful_updates'] == row['successful_updates'] == 2000
                and trained['VAE_text_frozen_unchanged'] and trained['checkpoint'] == row['checkpoint'],
                'STAGE25_GENERATOR_BARRIER_FAILED', cls)
        verify_tree(source, row['checkpoint'])
        add(name)
        add(f'logs/msdf_{cls}_updates.jsonl', trained['update_log_sha256'])
        for name, digest in row['checkpoint']['sha256'].items():
            add((Path(row['checkpoint']['path']) / name).as_posix(), digest)
    validity = load(source / 'formal_synthetic_validity.json')
    synthetic = load(source / 'formal_synthetic_manifest.json')
    require(validity['status'] == 'PASS' and validity['count'] == 240 and validity['reroll'] == 0
            and validity['all_planned_outputs_retained'] and validity['structural_checks_only']
            and validity['manifest_sha256'] == files['formal_synthetic_manifest.json']
            and load(source / 'msdf_generation_validity.json') == validity
            and synthetic['status'] == 'FROZEN' and synthetic['count'] == len(synthetic['records']) == 240
            and synthetic['quality_selection'] is False,
            'STAGE25_BANK_BARRIER_FAILED', '240/240 frozen unchanged inputs')
    slots = load(Path(b['p0']) / 'stage20b_generation_slots.json')['slots']
    rng = {r['slot_id']: r['banks'] for r in load(Path(b['p0']) / 'stage20b_rng_manifest.json')['records']}
    require(len(slots) == 80 and len({r['slot_id'] for r in slots}) == 80,
            'STAGE25_BANK_BARRIER_FAILED', 'exact80 original slots')
    combined, asset_paths = [], set()
    for bank in BANKS:
        name = f'formal_bank_{bank}_manifest.json'
        mf = load(source / name)
        require(mf['status'] == 'COMPLETE' and mf['count'] == len(mf['records']) == 80
                and mf['bank'] == bank and mf['quality_selection'] is False and mf['reroll_count'] == 0
                and mf['generation_protocol_sha256'] == sha(Path(b['p0_view']) / 'stage20b_generation_protocol.json')
                and Counter(r['class'] for r in mf['records']) == {'flash': 40, 'black': 40}
                and load(source / f'msdf_bank_{bank}_manifest.json') == mf,
                'STAGE25_BANK_BARRIER_FAILED', bank)
        add(name); add(f'msdf_bank_{bank}_manifest.json')
        for slot, row in zip(slots, mf['records']):
            expected = rng[slot['slot_id']][bank]
            require(row['bank'] == bank and row['frozen_slot'] == slot and row['slot_id'] == slot['slot_id']
                    and row['class'] == slot['class'] and row['role_id'] == 'extra_' + slot['slot_id']
                    and row['MSDF_enabled'] is True and row['formal_synthetic'] is True
                    and row['quality_selection'] is False
                    and row['sample_seed'] == expected['sample_seed'] == int(htext(
                        'Plastic_Bomo_Generation_V2' + slot['slot_id'] + str(cfg['bank_seeds'][bank]))[:16], 16)
                    and row['bank_seed'] == expected['bank_seed']
                    and htext(canonical(row['frozen_slot'])) == expected['non_rng_metadata_sha256']
                    and row['checkpoint_tree_sha256'] == generators['classes'][row['class']]['checkpoint']['tree_sha256'],
                    'STAGE25_BANK_BARRIER_FAILED', bank + ':' + slot['slot_id'])
            for field in ('image', 'label'):
                name = row[field + '_path']
                require(inside(source, name).is_relative_to(source / ('synthetic_' + bank)) and name not in asset_paths,
                        'STAGE25_RECOVERY_PATH_INVALID', name)
                asset_paths.add(name)
                add(name, row[field + '_sha256'])
        combined.extend(mf['records'])
    require(combined == synthetic['records'], 'STAGE25_BANK_BARRIER_FAILED', 'combined ABC manifest')
    snapshots = {name: sha(source / name) for name in (
        'stage25_status.json', 'detector_arm_A_training.json', 'logs/detector_A.log',
        'logs/detector_A_draws.jsonl', 'logs/msdf_detector_A_claim.json',
        'detector_initialization_comparison.json', 'detector_A/train_role_binding.json',
        'worker_failure.json', 'logs/last_failure_traceback.txt') if (source / name).is_file()}
    return b, status, interrupted, files, snapshots


def verify_files(root, files):
    for name, digest in files.items():
        require(sha(inside(root, name)) == digest, 'STAGE25_RECOVERY_PRESERVED_ASSET_CHANGED', name)


def import_assets(source, output, files, snapshots):
    """No moves/deletes. Hardlink immutable weights/images; copy mutable metadata."""
    modes = {}
    for name, digest in files.items():
        src, dest = inside(source, name), inside(output, name)
        require(not dest.exists(), 'STAGE25_OUTPUT_EXISTS', str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith(('generator_checkpoints/', 'synthetic_')):
            try:
                os.link(src, dest)
                modes[name] = 'hardlink_frozen_asset_never_written_by_recovery'
            except OSError:
                verified_copy(src, dest, digest)
                modes[name] = 'verified_copy'
        else:
            verified_copy(src, dest, digest)
            modes[name] = 'verified_copy'
        require(sha(dest) == digest, 'STAGE25_FROZEN_BINDING_CHANGED', name)
    for name, digest in snapshots.items():
        verified_copy(inside(source, name), inside(output, 'recovery/source_snapshot/' + name), digest)
    return modes


def reserve(source, output):
    """Exactly one explicitly requested recovery, not a retry loop or overwrite."""
    path = source.parent / ('manual_detector_recovery_claim_' + htext(str(source))[:16] + '.json')
    try:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(canonical({'source_run': str(source), 'output': str(output),
                'started_at': now(), 'pid': os.getpid(), 'recovery_entry_sha256': sha(SCRIPT)}) + '\n')
    except FileExistsError as error:
        raise Stop('STAGE25_RECOVERY_ALREADY_CLAIMED',
                   {'reason': '该源任务已有手动恢复记录；不会重复启动。', 'claim': str(path), 'record': load(path)}) from error
    return path


def run_recovery(output, source, journal, b):
    """Only the original frozen detector and unified evaluation phases execute."""
    a = runner.parser().parse_args([])
    for key in ('code_manifest', 'stage20a', 'p0', 'r0', 'baseline_run', 'base_model', 'detector_model', 'real_root', 'normal_root'):
        setattr(a, key, Path(b[key]))
    a.protocol = Path(b['stage25_protocol'])
    a.device = b['device'].removeprefix('cuda:')
    a.run = output
    path = output / 'manual_detector_recovery.json'
    require(sha(SCRIPT) == journal['recovery_entry_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', SCRIPT.name)
    runner.formal_detector(a, output)  # A/B/C; frozen controller checks32400 draws/600 attempts.
    verify_files(source, journal['source_sha256'])
    verify_files(output, journal['reused_sha256'])
    require(sha(SCRIPT) == journal['recovery_entry_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', SCRIPT.name)
    journal.update(status='DETECTORS_FROZEN', source_run_unchanged_SHA_verified=True,
                   interrupted_A_excluded_from_primary=True)
    save(path, journal)
    runner.child(a, output, 'evaluate')  # Existing all3 checkpoint barrier; no earlier evaluation.
    verify_files(source, journal['source_sha256'])
    verify_files(output, journal['reused_sha256'])
    require(load(output / 'stage25_status.json')['full_experiment_complete'],
            'STAGE25_RECOVERY_INCOMPLETE', 'unified evaluation not complete')
    journal.update(status='COMPLETE', completed_at=now(), source_run_unchanged_SHA_verified=True)
    save(path, journal)
    report = output / 'STAGE25_REPORT.md'
    with report.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write('\n## Manual interruption recovery\n\n'
            + 'Source: `' + source.name + '`. Original run/logs/partial A remain unchanged.\n'
            + 'Both step2000 generators and all240 images were SHA-verified and reused; no new generation.\n'
            + 'A/B/C each restarted from the frozen public initialization, seed42, epoch1 through150. '
            + 'No optimizer/epoch resume, extra primary budget, or success-step compensation.\n'
            + 'Interrupted A physical compute is recorded separately in manual_detector_recovery.json '
            + 'and recovery/source_snapshot/, not hidden or pooled into matched primary exposure.\n'
            + 'SSH disconnection is user-reported, not independently established as the cause. '
            + 'Immutable assets may share hardlinks with the source; do not edit either copy.\n')
    ledger_path = output / 'stage25_result_artifact_manifest.json'
    ledger = load(ledger_path)
    for name in ('STAGE25_REPORT.md', 'manual_detector_recovery.json', 'recovery/entry.py'):
        ledger['sha256'][name] = sha(output / name)
    ledger.update(manual_interruption_recovery=True, interrupted_source_run=str(source),
                  separate_recovery_entry_sha256=journal['recovery_entry_sha256'])
    save(ledger_path, ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new sibling recovery directory')
    p.add_argument('--confirm-detector-restart', action='store_true', help='explicit restart A/B/C from epoch1; not resume')
    p.add_argument('--check-only', action='store_true', help='read/hash only; no output/claim/model/forward/training')
    a = p.parse_args()
    source, output = a.source_run.resolve(), a.output.resolve()
    active = None
    print(f'[启动] Stage25 检测阶段手动恢复；PID={os.getpid()}；不是第10轮续训。', flush=True)
    try:
        require(a.check_only or a.confirm_detector_restart, 'STAGE25_RECOVERY_CONFIRMATION_REQUIRED',
                '需要 --confirm-detector-restart；A/B/C 各从第1轮训练到150轮。')
        require(source.is_dir() and source.parent == ROOT.resolve() and output.parent == source.parent
                and output != source and not output.exists(),
                'STAGE25_RECOVERY_OUTPUT_INVALID', '必须使用新的 Stage25 同级结果目录，不覆盖原任务。')
        check_processes()
        print('[检查] 核对原冻结代码/环境、两个2000步模型和240张图的 SHA256；校验时可能暂时没有显卡进程。', flush=True)
        b, status, interrupted, files, snapshots = precheck(source)
        if a.check_only:
            print('[已退出] 只读复核通过，退出码=0；没有创建目录或训练。', flush=True)
            return
        check_processes()  # Recheck after potentially long asset hashing, before claiming.
        verify_files(source, snapshots)
        reserve(source, output)
        output.mkdir()
        (output / 'logs').mkdir()
        active = output
        current = {'status': 'FORMAL_RUNNING', 'full_experiment_complete': False, 'started_at': now(),
            'DeepPCB': 0, 'BootstrapGuard': 0, 'DWBG': 0, 'selector': 0, 'official_final_eval_rounds': 0,
            'generator_training_count': 2, 'synthetic_generation_count': 240, 'detector_training_count': 0,
            'msdf_flash_training_started': True, 'msdf_black_training_started': True,
            'msdf_flash_successful_updates': 2000, 'msdf_black_successful_updates': 2000,
            'new_generator_training_count': 0, 'new_synthetic_generation_count': 0,
            'reused_generator_training_count': 2, 'reused_synthetic_generation_count': 240,
            'comparator_generator_training_count': 0, 'comparator_synthetic_generation_count': 0,
            'manual_interruption_recovery': True, 'source_run': str(source),
            'automatic_rescue': False, 'stage26_auto_start': False}
        save(output / 'stage25_status.json', current)
        journal = {'status': 'PREPARING', 'source_run': str(source), 'output': str(output), 'started_at': now(),
            'authorization': 'explicit manual --confirm-detector-restart', 'recovery_entry_sha256': sha(SCRIPT),
            'source_sha256': {**files, **snapshots}, 'reused_sha256': files, 'snapshot_sha256': snapshots,
            'source_status_at_recovery': status['status'], 'source_run_modified': False,
            'restarted_detector_arms': list(BANKS), 'restart_epoch': 1, 'primary_epochs_per_arm': 150,
            'primary_draws_per_arm': 32400, 'primary_scheduled_attempts_per_arm': 600,
            'seed_and_hyperparameters_changed': False, 'new_generator_training_count': 0,
            'new_synthetic_generation_count': 0, 'interrupted_A_excluded_from_primary': True,
            'extra_interrupted_detector_attempts': 1, 'interrupted_counters_may_be_lower_bounds': True,
            'interrupted_A_observed_counters': {k: interrupted[k] for k in ('epochs', 'actual_draws',
                'attempted_optimizer_updates', 'successful_optimizer_updates', 'AMP_skipped_updates')},
            'external_interruption_cause_independently_verified': False,
            'reason': 'user-reported foreground SSH disconnect; incomplete A and no last.pt observed',
            'automatic_retry': False, 'automatic_next_stage': False}
        save(output / 'manual_detector_recovery.json', journal)
        print('[复用] 保留原目录；导入冻结资产，权重优先硬链接，不重复训练或生成。', flush=True)
        journal['import_modes'] = import_assets(source, output, files, snapshots)
        verified_copy(SCRIPT, output / 'recovery/entry.py', journal['recovery_entry_sha256'])
        journal['status'] = 'RUNNING'
        save(output / 'manual_detector_recovery.json', journal)
        print('[恢复目录] ' + str(output), flush=True)
        print('[开始] 从固定初始化训练检测 A/B/C；三组冻结后自动执行原统一评估。', flush=True)
        run_recovery(output, source, journal, b)
        print('[已退出] Stage25 检测恢复和统一评估完成，退出码=0；结果目录=' + str(output), flush=True)
    except BaseException as error:
        if not isinstance(error, (Exception, KeyboardInterrupt)):
            raise
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        if isinstance(error, KeyboardInterrupt):
            error = Stop('STAGE25_INTERRUPTED', '用户中断；本次恢复停止，没有自动重试。')
        if active is not None:
            runner.failure(active, error, exit_code=code)
            failure_status = getattr(error, 'status', 'STAGE25_RECOVERY_FAILED')
            journal_path = active / 'manual_detector_recovery.json'
            if journal_path.is_file():
                failed_journal = load(journal_path)
                failed_journal.update(status='STOPPED_' + failure_status, stopped_at=now(), detail=str(error))
                save(journal_path, failed_journal)
            save(active / 'recovery_failure.json', {'status': failure_status,
                'detail': str(error), 'automatic_retry': False, 'source_run_modified': False, 'at': now()})
            (active / 'logs/recovery_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        else:
            runner.describe_error(getattr(error, 'status', 'STAGE25_RECOVERY_PRECHECK_FAILED'),
                                  getattr(error, 'detail', str(error)), exit_code=code)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
