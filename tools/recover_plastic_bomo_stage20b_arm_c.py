"""Explicit manual SIGKILL recovery: reuse frozen assets/RR/A/B; restart only C."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace

from stage20b_formal_common import (ARTIFACTS, FORMAL_FILES, REPO, STATES, Stop,
    canonical, context, formal_binding, htext, load, now, offline, require, save,
    sha, transition, update_status, verify_tree)

REUSED = ('generator_training_flash', 'generator_training_black',
          'generator_checkpoint_binding', 'trained_checkpoint_determinism_audit',
          'formal_bank_A_manifest', 'formal_bank_B_manifest', 'formal_bank_C_manifest',
          'formal_synthetic_manifest', 'formal_synthetic_validity',
          'detector_arm_RR_training', 'detector_arm_A_training', 'detector_arm_B_training',
          'detector_initialization_comparison')
RECHECK_KEYS = ('stage20a', 'p0', 'r0', 'protocol', 'base_model', 'detector_model',
                'device', 'P0_binding', 'R0_artifacts', 'R0_code_sha256',
                'formal_code_sha256', 'runtime')
SCRIPT = Path(__file__).resolve()


def inside(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    require(not relative.is_absolute() and path.is_relative_to(root),
            'MANUAL_RECOVERY_PATH_INVALID', name)
    return path


def check_processes(source, binding):
    """Read-only process checks. Never send a signal, including to reused PIDs."""
    require(sys.platform.startswith('linux'), 'MANUAL_RECOVERY_SERVER_REQUIRED', sys.platform)
    pids = [binding.get('pid'), load(source / 'logs/train_plastic_bomo_stage20b_detector_arm_C_pid.json')['pid']]
    for pid in pids:
        if not pid:
            continue
        proc = Path('/proc') / str(int(pid))
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace')
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise Stop('MANUAL_RECOVERY_PROCESS_CHECK_UNAVAILABLE', str(error)) from error
        require(str(source) not in command and source.name not in command,
                'MANUAL_RECOVERY_SOURCE_STILL_RUNNING', {'pid': pid, 'command': command})


def check_completed_arm(source, arm, order, p0):
    row = load(source / f'detector_arm_{arm}_training.json')
    require(row['status'] == 'COMPLETE' and row['epochs'] == 150
            and row['actual_draws'] == order['scheduled_draws']
            and row['attempted_optimizer_updates'] == order['scheduled_optimizer_attempts']
            and row['successful_optimizer_updates'] + row['AMP_skipped_updates'] == row['attempted_optimizer_updates']
            and row['official_final_eval_content_read'] == 0 and len(row['epoch_records']) == 150
            and row['order_spec_sha256'] == sha(p0 / 'stage20b_training_order_spec.json')
            and row['training_protocol_sha256'] == sha(p0 / 'stage20b_detector_protocol.json')
            and row['last_checkpoint'] == f'detector_{arm}/last.pt',
            'MANUAL_RECOVERY_INCOMPLETE_REUSED_ARM', arm)
    for expected, actual in zip(order['epochs'], row['epoch_records']):
        require(actual['actual_role_sequence_sha256'] == expected['sequence_sha256']
                and actual['draws'] == 216
                and [x['normalizer'] for x in actual['attempts']] == expected['accumulation_group_lengths']
                and [x['position'] for x in actual['attempts']] == expected['optimizer_attempt_after_positions_0based'],
                'MANUAL_RECOVERY_REUSED_SCHEDULE_CHANGED', {'arm': arm, 'epoch': actual['epoch_0based']})
    return row


def precheck(source):
    """No mutation, CUDA context, evaluation pixels, or metric selection."""
    status = load(source / 'stage20b_status.json')
    require(status['status'].startswith('STOPPED_') and status['state'] == 'DETECTOR_B_TRAINED'
            and [x['state'] for x in status['state_history']] == list(STATES[:11])
            and not status['full_experiment_complete']
            and status['official_final_eval_rounds'] == 0,
            'MANUAL_RECOVERY_NOT_INTERRUPTED_C', status['state'])
    detail = str(status.get('detail', ''))
    require(re.search(r"['\"]exit_code['\"]\s*:\s*-9\b", detail)
            and 'train_plastic_bomo_stage20b_detector_arm_C.log' in detail,
            'MANUAL_RECOVERY_NOT_SIGKILL_C', detail)
    require(not (source / 'detector_C/last.pt').exists()
            and not (source / 'logs/final_eval_one_round_claim.json').exists()
            and not (source / 'final_eval_inputs').exists()
            and not (source.parent / 'authoritative_formal_run.json').exists(),
            'MANUAL_RECOVERY_ALREADY_COMPLETED_OR_EVALUATED', str(source))
    for name in ('detector_checkpoint_binding', 'detector_compute_audit', 'final_eval_access_audit',
                 'RR_metrics', 'RawA_metrics', 'RawB_metrics', 'RawC_metrics', 'stage20b_headroom'):
        require(load(source / (name + '.json'))['status'] == 'NOT_RUN',
                'MANUAL_RECOVERY_ALREADY_COMPLETED_OR_EVALUATED', name)
    interrupted = load(source / 'detector_arm_C_training.json')
    require(interrupted['status'] != 'COMPLETE' and interrupted['arm'] == 'C'
            and interrupted['official_final_eval_content_read'] == 0,
            'MANUAL_RECOVERY_NOT_INTERRUPTED_C', 'C audit')
    binding = load(source / 'formal_protocol_binding.json')
    require(binding['status'] == 'PASS' and binding['result_directory'] == str(source)
            and set(binding['formal_code_sha256']) == {f'tools/{n}' for n in FORMAL_FILES},
            'FORMAL_FROZEN_BINDING_CHANGED', 'source binding')
    for name, digest in {**binding['formal_code_sha256'], **binding['R0_code_sha256']}.items():
        require(sha(REPO / name) == digest, 'FORMAL_FROZEN_BINDING_CHANGED', name)
    p0 = Path(binding['p0'])
    order = load(p0 / 'stage20b_training_order_spec.json')
    checkpoint = load(source / 'generator_checkpoint_binding.json')
    require(checkpoint['status'] == 'FROZEN' and set(checkpoint['classes']) == {'flash', 'black'},
            'FORMAL_FROZEN_BINDING_CHANGED', 'generator freeze')
    require(load(source / 'trained_checkpoint_determinism_audit.json')['status'] == 'PASS'
            and load(source / 'formal_synthetic_validity.json')['status'] == 'PASS',
            'FORMAL_FROZEN_BINDING_CHANGED', 'completed sanity/structural audits')
    files = {}

    def add(name, digest=None):
        path = inside(source, name)
        require(path.is_file(), 'MANUAL_RECOVERY_ASSET_MISSING', name)
        actual = sha(path)
        require(digest is None or digest == actual, 'FORMAL_FROZEN_BINDING_CHANGED', name)
        require(name not in files or files[name] == actual, 'FORMAL_FROZEN_BINDING_CHANGED', name)
        files[name] = actual

    for name in REUSED:
        add(name + '.json')
    for cls, row in checkpoint['classes'].items():
        trained = load(source / ('generator_training_' + cls + '.json'))
        target = load(p0 / 'stage20b_generator_training_protocol.json')['max_train_steps']
        require(trained['status'] == 'COMPLETE' and trained['checkpoint'] == row['checkpoint']
                and trained['successful_updates'] == row['successful_updates'] == target
                and status[f'generator_{cls}_successful_updates'] == target
                and trained['frozen_components_unchanged'] and trained['last_step_legal'],
                'MANUAL_RECOVERY_GENERATOR_NOT_FROZEN', cls)
        verify_tree(source, row['checkpoint'])
        for name, digest in row['checkpoint']['sha256'].items():
            files[(Path(row['checkpoint']['path']) / name).as_posix()] = digest
    synthetic = load(source / 'formal_synthetic_manifest.json')
    require(synthetic['status'] == 'FROZEN' and synthetic['count'] == len(synthetic['records']) == 240
            and synthetic['bank_counts'] == {'A': 80, 'B': 80, 'C': 80}
            and synthetic['selector_count'] == synthetic['reroll_count'] == 0,
            'FORMAL_FROZEN_BINDING_CHANGED', 'all raw240 required')
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    rng = {x['slot_id']: x['banks'] for x in load(p0 / 'stage20b_rng_manifest.json')['records']}
    combined = []
    for bank in ('A', 'B', 'C'):
        manifest = load(source / ('formal_bank_' + bank + '_manifest.json'))
        require(manifest['status'] == 'COMPLETE' and len(manifest['records']) == 80
                and manifest['generation_protocol_sha256'] == sha(p0 / 'stage20b_generation_protocol.json'),
                'FORMAL_FROZEN_BINDING_CHANGED', bank + ' original bank manifest')
        for slot, row in zip(slots, manifest['records']):
            expected = rng[slot['slot_id']][bank]
            require(row['bank'] == bank and row['frozen_slot'] == slot and row['slot_id'] == slot['slot_id']
                    and row['non_rng_metadata_sha256'] == expected['non_rng_metadata_sha256']
                    and row['sample_seed'] == expected['sample_seed'] and row['bank_seed'] == expected['bank_seed']
                    and row['checkpoint_tree_sha256'] == checkpoint['classes'][slot['class']]['checkpoint']['tree_sha256']
                    and all(inside(source, row[k]).is_relative_to((source / ('synthetic_' + bank)).resolve())
                            for k in ('image_path', 'label_path')),
                    'FORMAL_FROZEN_BINDING_CHANGED', bank + ':' + slot['slot_id'])
        combined.extend(manifest['records'])
    require(combined == synthetic['records'], 'FORMAL_FROZEN_BINDING_CHANGED', 'consolidated bank records')
    for row in synthetic['records']:
        for field in ('image', 'label'):
            add(row[field + '_path'], row[field + '_sha256'])
    sanity = load(source / 'trained_checkpoint_determinism_audit.json')
    require(sanity['sampling_count'] == 4 and len(sanity['records']) == 2
            and {x['class'] for x in sanity['records']} == {'flash', 'black'}
            and all(len(x['runs']) == 2 and x['same_seed_RGB_identical'] for x in sanity['records']),
            'FORMAL_FROZEN_BINDING_CHANGED', 'original four-image determinism sanity')
    for row in sanity['records']:
        for sample in row['runs']:
            add(sample['image_path'], sample['image_sha256'])
    initialization = htext(canonical(load(source / 'detector_initialization_comparison.json')))
    for arm in ('RR', 'A', 'B'):
        row = check_completed_arm(source, arm, order, p0)
        require(row['initial_parameter_binding_sha256'] == initialization
                and status[f'detector_{arm}_successful_updates'] == row['successful_optimizer_updates'],
                'DETECTOR_INITIALIZATION_MISMATCH', arm)
        add(row['last_checkpoint'], row['last_sha256'])
        add(f'logs/detector_{arm}_draws.jsonl', row['per_draw_log_sha256'])
        add(f'detector_{arm}/train_role_binding.json')
        for suffix in ('.log', '_launch.json', '_pid.json'):
            add(f'logs/train_plastic_bomo_stage20b_detector_arm_{arm}' + suffix)
    snapshots = ['stage20b_status.json', 'formal_protocol_binding.json', 'STAGE20B_REPORT.md',
                 'detector_arm_C_training.json', 'logs/orchestrator_failure.txt',
                 'logs/train_plastic_bomo_stage20b_detector_arm_C.log',
                 'logs/train_plastic_bomo_stage20b_detector_arm_C_launch.json',
                 'logs/train_plastic_bomo_stage20b_detector_arm_C_pid.json', 'logs/detector_C_draws.jsonl']
    snapshot_hashes = {name: sha(inside(source, name)) for name in snapshots}
    return binding, status, interrupted, files, snapshot_hashes


def verify_files(root, records):
    for name, digest in records.items():
        require(sha(inside(root, name)) == digest, 'MANUAL_RECOVERY_PRESERVED_ASSET_CHANGED', name)


def import_assets(source, output, files, snapshots):
    """Never move/delete or write a source file. Hardlink immutable heavy assets only."""
    modes = {}
    for name, digest in files.items():
        src, dest = inside(source, name), inside(output, name)
        require(not dest.exists(), 'FORMAL_OUTPUT_EXISTS', str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        heavy = name.startswith(('generator_checkpoints/', 'synthetic_', 'checkpoint_sanity/')) or name.endswith('/last.pt')
        if heavy:
            try:
                os.link(src, dest)
                modes[name] = 'hardlink_frozen_read_only_by_workflow'
            except OSError:
                shutil.copyfile(src, dest)
                modes[name] = 'verified_copy'
        else:
            shutil.copyfile(src, dest)
            modes[name] = 'verified_copy'
        require(sha(dest) == digest, 'FORMAL_FROZEN_BINDING_CHANGED', name)
    for name, digest in snapshots.items():
        dest = inside(output, 'manual_recovery/source_snapshot/' + name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(source, name), dest)
        require(sha(dest) == digest, 'FORMAL_FROZEN_BINDING_CHANGED', name)
    return modes


def binding_only(a):
    require(os.environ.get('STAGE20B_MANUAL_PARENT_PID') == str(os.getppid()),
            'MANUAL_RECOVERY_INVALID_INTERNAL_CALL', 'binding subprocess only')
    old = load(a.source_run / 'formal_protocol_binding.json')
    args = SimpleNamespace(**{k: Path(old[k]) for k in ('stage20a', 'p0', 'r0', 'protocol', 'base_model', 'detector_model')},
                           output=a.output, device=old['device'].removeprefix('cuda:'))
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden
    with network_forbidden():
        fresh = formal_binding(args)
    require(all(fresh[k] == old[k] for k in RECHECK_KEYS), 'FORMAL_FROZEN_BINDING_CHANGED', 'manual recheck vs original')
    fresh['manual_recovery_source_binding_sha256'] = sha(a.source_run / 'formal_protocol_binding.json')
    fresh['manual_recovery_source_run'] = str(a.source_run)
    fresh['manual_recovery_code_sha256'] = sha(SCRIPT)
    save(a.output / 'formal_protocol_binding.json', fresh)


def step(run, script, args, expected, following):
    context(run, (expected,))
    journal = load(run / 'manual_arm_C_recovery.json')
    require(sha(SCRIPT) == journal['recovery_script_sha256'], 'FORMAL_FROZEN_BINDING_CHANGED', SCRIPT.name)
    stem = 'manual_' + script.removesuffix('.py') + ('_' + '_'.join(args).replace('--', '') if args else '')
    command = [sys.executable, str(REPO / 'tools' / script), '--run', str(run), *args]
    logfile = run / 'logs' / (stem + '.log')
    save(run / 'logs' / (stem + '_launch.json'), {'command': command, 'started_at': now()})
    print('Starting ' + stem, flush=True)
    with logfile.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        save(run / 'logs' / (stem + '_pid.json'), {'pid': process.pid, 'started_at': now()})
        code = process.wait()
    require(code == 0, 'MANUAL_RECOVERY_CHILD_FAILED', {'exit_code': code, 'log': str(logfile)})
    transition(run, expected, following)


def finish_recovery(run, source, files, snapshots):
    """Unmodified formal final checks/gates; publish only after provenance is sealed."""
    _, p0, stage20a = context(run, ('FINAL_EVAL_COMPLETE',))
    verify_files(source, {**files, **snapshots})
    verify_files(run, files)
    journal_path = run / 'manual_arm_C_recovery.json'
    journal = load(journal_path)
    require(sha(SCRIPT) == journal['recovery_script_sha256'], 'FORMAL_FROZEN_BINDING_CHANGED', SCRIPT.name)
    import audit_plastic_bomo_stage20b as audit
    pointer = run.parent / 'authoritative_formal_run.json'
    pending = []
    original_save = audit.save

    def defer_pointer(path, value):
        if Path(path) == pointer:
            pending.append(value)
        else:
            original_save(path, value)

    # Delay ONLY authority publication; no changed metric, threshold or final check.
    audit.save = defer_pointer
    try:
        audit.finish(run, p0, stage20a)
    finally:
        audit.save = original_save
    require(len(pending) == 1, 'MANUAL_RECOVERY_FINALIZATION_FAILED', 'authority publication')
    journal.update(status='COMPLETE', completed_at=now(), source_run_unchanged_SHA_verified=True,
                   restarted_C_final_checkpoint_sha256=load(run / 'detector_arm_C_training.json')['last_sha256'])
    save(journal_path, journal)
    report = run / 'STAGE20B_REPORT.md'
    report.write_text(report.read_text(encoding='utf-8') + '\n## Explicit manual interruption recovery\n\n'
        + 'Source stopped run: `' + source.name + '`. Original files/status were not overwritten.\n'
        + 'User reported a possible accidental process kill; its external cause is not independently verified.\n'
        + 'Generator checkpoints, all raw240 images and completed RR/A/B were reused after SHA verification.\n'
        + 'Only C restarted from the original public initialization and seed42, epoch1 through150, with the unchanged frozen schedule. '
        + 'No epoch77 resume or added budget for the primary C checkpoint.\n'
        + 'The interrupted C attempt is preserved separately, including its observed update/draw counters (potentially lower bounds). '
        + 'That extra physical work is not hidden or counted as matched primary exposure.\n'
        + 'See manual_arm_C_recovery.json and manual_recovery/source_snapshot/. '
        + 'Inherited state-history entries refer to the source execution, not fresh generator/RR/A/B execution.\n'
        + 'Immutable heavy files may share hardlinks with the source; do not edit either copy. No automatic next stage.\n',
        encoding='utf-8', newline='\n')
    ledger = load(run / 'formal_result_artifact_manifest.json')
    ledger['sha256']['STAGE20B_REPORT.md'] = sha(report)
    ledger['sha256']['manual_arm_C_recovery.json'] = sha(journal_path)
    for path in sorted((run / 'manual_recovery/source_snapshot').rglob('*')):
        if path.is_file():
            ledger['sha256'][path.relative_to(run).as_posix()] = sha(path)
    ledger['manual_interruption_recovery'] = True
    save(run / 'formal_result_artifact_manifest.json', ledger)
    value = {**pending[0], 'artifact_manifest_sha256': sha(run / 'formal_result_artifact_manifest.json'),
             'manual_interruption_recovery': True, 'source_run': source.name}
    candidate = run / 'logs/manual_authority_pointer_candidate.json'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    save(candidate, value)
    # Atomic exclusive publication on the sibling run's filesystem; no half-written pointer.
    os.link(candidate, pointer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirm-manual-c-restart', action='store_true')
    parser.add_argument('--_bind_only', action='store_true', help=argparse.SUPPRESS)
    a = parser.parse_args()
    a.source_run, a.output = a.source_run.resolve(), a.output.resolve()
    offline()
    if a._bind_only:
        binding_only(a)
        return
    require(a.confirm_manual_c_restart, 'MANUAL_RECOVERY_CONFIRMATION_REQUIRED', 'explicit C epoch1 restart only')
    require(a.source_run.is_dir() and a.output.parent == a.source_run.parent and a.output != a.source_run
            and not a.output.exists(), 'MANUAL_RECOVERY_OUTPUT_INVALID', 'new sibling output required')
    print('Read-only source/asset precheck. No sampling or final_eval.', flush=True)
    binding, status, interrupted, files, snapshots = precheck(a.source_run)
    check_processes(a.source_run, binding)
    claim = a.source_run.parent / ('manual_C_recovery_claim_' + htext(str(a.source_run))[:16] + '.json')
    # One manual attempt for this source, not a retry loop. Original run stays untouched.
    try:
        with claim.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(canonical({'source_run': str(a.source_run), 'output': str(a.output), 'started_at': now(), 'pid': os.getpid()}) + '\n')
    except FileExistsError as error:
        raise Stop('MANUAL_RECOVERY_ALREADY_CLAIMED', 'No automatic retry; see ' + str(claim)) from error
    a.output.mkdir()
    (a.output / 'logs').mkdir()
    active = deepcopy(status)
    for key in ('reason', 'detail', 'stopped_at'):
        active.pop(key, None)
    active.update(status='FORMAL_RUNNING', detector_C_successful_updates=0, detector_training_count=3,
                  manual_interruption_recovery=True, manual_recovery_source_run=str(a.source_run),
                  inherited_completed_states_source=str(a.source_run), recovery_started_at=now(),
                  interrupted_C_attempt_excluded_from_primary=True)
    save(a.output / 'stage20b_status.json', active)
    journal = {'status': 'RUNNING', 'source_run': str(a.source_run), 'output': str(a.output),
        'authorization': 'explicit user requested manual recovery; --confirm-manual-c-restart',
        'reason': 'user-reported possible accidental kill; observed child exit_code -9',
        'external_kill_cause_independently_verified': False, 'recovery_script_sha256': sha(SCRIPT),
        'started_at': now(), 'source_sha256': {**files, **snapshots}, 'snapshot_sha256': snapshots,
        'reused_sha256': files, 'reused_detector_arms': ['RR', 'A', 'B'],
        'restarted_detector_arms': ['C'], 'restart_epoch': 1, 'seed_and_hyperparameters_changed': False,
        'new_generator_training_count': 0, 'new_synthetic_generation_count': 0,
        'extra_interrupted_detector_attempts': 1, 'primary_detector_arms': 4,
        'interrupted_C_observed_counters': {k: interrupted.get(k) for k in ('epochs', 'actual_draws',
            'attempted_optimizer_updates', 'successful_optimizer_updates', 'AMP_skipped_updates')},
        'interrupted_counters_may_be_lower_bounds': True, 'automatic_rescue': False,
        'source_run_modified': False, 'automatic_next_stage': False}
    save(a.output / 'manual_arm_C_recovery.json', journal)
    try:
        print('Rechecking original frozen code/models/runtime in a short subprocess.', flush=True)
        command = [sys.executable, str(SCRIPT), '--source_run', str(a.source_run), '--output', str(a.output), '--_bind_only']
        environment = dict(os.environ, STAGE20B_MANUAL_PARENT_PID=str(os.getpid()), STAGE20B_PARENT_PID=str(os.getpid()))
        with (a.output / 'logs/manual_binding_recheck.log').open('x', encoding='utf-8') as log:
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=environment)
        require(code == 0, 'FORMAL_FROZEN_BINDING_CHANGED', {'exit_code': code, 'log': 'logs/manual_binding_recheck.log'})
        for name in ARTIFACTS:
            if name not in REUSED:
                save(a.output / (name + '.json'), {'status': 'NOT_RUN'})
        print('Importing verified frozen assets and completed RR/A/B. Source is not moved or deleted.', flush=True)
        journal['import_modes'] = import_assets(a.source_run, a.output, files, snapshots)
        save(a.output / 'manual_arm_C_recovery.json', journal)
        os.environ['STAGE20B_FORMAL_BINDING_SHA256'] = sha(a.output / 'formal_protocol_binding.json')
        step(a.output, 'train_plastic_bomo_stage20b_detector.py', ['--arm', 'C'], 'DETECTOR_B_TRAINED', 'DETECTOR_C_TRAINED')
        verify_files(a.source_run, {**files, **snapshots})
        verify_files(a.output, files)
        step(a.output, 'audit_plastic_bomo_stage20b.py', ['--action', 'freeze_detectors'], 'DETECTOR_C_TRAINED', 'ALL_DETECTORS_FROZEN')
        step(a.output, 'evaluate_plastic_bomo_stage20b.py', [], 'ALL_DETECTORS_FROZEN', 'FINAL_EVAL_COMPLETE')
        finish_recovery(a.output, a.source_run, files, snapshots)
        print('Manual C recovery and original unified evaluation complete. RESULT=' + str(a.output), flush=True)
    except Exception as error:
        reason = error.status if isinstance(error, Stop) else 'MANUAL_RECOVERY_FAILED'
        update_status(a.output, status='STOPPED_' + reason, reason=reason, detail=str(error), stopped_at=now(),
                      full_experiment_complete=False, TDCRG_DEVELOPMENT_AUTHORIZED=False)
        journal = load(a.output / 'manual_arm_C_recovery.json')
        journal.update(status='STOPPED_' + reason, stopped_at=now(), detail=str(error))
        save(a.output / 'manual_arm_C_recovery.json', journal)
        (a.output / 'STAGE20B_REPORT.md').write_text('# Manual Stage20B C recovery stopped\n\n'
            + 'Reason: `' + reason + '`. This is incomplete, not a headroom conclusion.\n'
            + 'Original stopped source run remains unchanged. No automatic retry or next stage.\n'
            + 'See manual_arm_C_recovery.json and logs/manual_recovery_failure.txt.\n', encoding='utf-8', newline='\n')
        (a.output / 'logs/manual_recovery_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
