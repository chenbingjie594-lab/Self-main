"""Authorized one-shot formal Stage20B: strict sequence, no restart/rescue/auto-next-stage."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from stage20b_formal_common import (ARTIFACTS, REPO, Stop, formal_binding, initial_status,
    load, now, offline, require, save, sha, transition, update_status)


def create_binding(a):
    """Recheck the required startup binding in a short process; no resident parent CUDA context."""
    command = [sys.executable, str(Path(__file__).resolve()), '--_bind_only', '--output', str(a.output), '--device', a.device]
    for name in ('stage20a', 'p0', 'r0', 'protocol', 'base_model', 'detector_model'):
        command += ['--' + name, str(getattr(a, name))]
    logfile = a.output / 'logs' / 'formal_startup_binding.log'
    environment = dict(os.environ, STAGE20B_PARENT_PID=str(os.getpid()))
    with logfile.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=environment)
        save(a.output / 'logs' / 'formal_startup_binding_launch.json',
             {'command': command, 'pid': process.pid, 'parent_pid': os.getpid(), 'started_at': now(),
              'log': str(logfile), 'not_a_new_preflight_stage': True})
        code = process.wait()
    require(code == 0, 'FORMAL_FROZEN_BINDING_CHANGED', {'exit_code': code, 'log': str(logfile)})
    return load(a.output / 'formal_protocol_binding.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage20a', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol')
    p.add_argument('--p0', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115')
    p.add_argument('--r0', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254')
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage20b_p0.json')
    p.add_argument('--base_model', type=Path, default=REPO / 'sd2-community/stable-diffusion-2-inpainting')
    p.add_argument('--detector_model', type=Path, default=REPO / 'pretrained/yolo11s.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--_bind_only', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    for key in ('stage20a', 'p0', 'r0', 'protocol', 'base_model', 'detector_model', 'output'):
        setattr(a, key, getattr(a, key).resolve())
    if a._bind_only:
        # Internal startup byte/environment recheck ONLY, at INIT, with no model
        # training/forward or new experiment gate. CUDA context dies at process exit.
        require(os.environ.get('STAGE20B_PARENT_PID') and a.output.is_dir()
                and load(a.output / 'stage20b_status.json')['state'] == 'INIT'
                and load(a.output / 'formal_protocol_binding.json')['status'] == 'NOT_RUN',
                'FORMAL_FROZEN_BINDING_CHANGED', 'invalid internal binding invocation')
        from qualify_plastic_bomo_stage20b_r0 import network_forbidden
        with network_forbidden():
            save(a.output / 'formal_protocol_binding.json', formal_binding(a))
        return
    require(not a.output.exists(), 'FORMAL_OUTPUT_EXISTS', 'new run required; no overwrite/resume')
    require(not (a.output.parent / 'authoritative_formal_run.json').exists(), 'FORMAL_AUTHORITY_ALREADY_EXISTS', 'one complete run only')
    a.output.mkdir(parents=True)
    (a.output / 'logs').mkdir()
    save(a.output / 'stage20b_status.json', initial_status())
    save(a.output / 'formal_protocol_binding.json', {'status': 'NOT_RUN'})
    for name in ARTIFACTS:
        save(a.output / (name + '.json'), {'status': 'NOT_RUN'})
    offline()
    phase = 'FORMAL_FROZEN_BINDING_CHANGED'
    try:
        binding = create_binding(a)
        save(a.output / 'formal_protocol_binding.json', binding)
        os.environ['STAGE20B_FORMAL_BINDING_SHA256'] = sha(a.output / 'formal_protocol_binding.json')
        print('Frozen binding PASS. Formal Stage20B begins: ' + str(a.output), flush=True)

        def step(script, args, expected, following=None):
            nonlocal phase
            phase = script.upper().replace('.PY', '') + '_FAILED'
            status = load(a.output / 'stage20b_status.json')
            require(status['state'] == expected and status['status'] == 'FORMAL_RUNNING',
                    'FORMAL_STATE_SEQUENCE_INVALID', expected)
            command = [sys.executable, str(REPO / 'tools' / script), '--run', str(a.output), *args]
            stem = script.replace('.py', '') + ('_' + '_'.join(args).replace('--', '') if args else '')
            logfile = a.output / 'logs' / (stem + '.log')
            save(a.output / 'logs' / (stem + '_launch.json'), {'command': command, 'started_at': now(), 'log': str(logfile)})
            print('Starting ' + stem, flush=True)
            with logfile.open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                save(a.output / 'logs' / (stem + '_pid.json'), {'pid': process.pid, 'started_at': now()})
                code = process.wait()
            if code != 0:
                s = load(a.output / 'stage20b_status.json')
                raise Stop(s.get('reason', phase), {'exit_code': code, 'log': str(logfile), 'detail': s.get('detail')})
            if following:
                transition(a.output, expected, following)

        step('train_plastic_bomo_stage20b_generator.py', ['--class_name', 'flash'], 'INIT', 'FLASH_GENERATOR_TRAINED')
        step('train_plastic_bomo_stage20b_generator.py', ['--class_name', 'black'], 'FLASH_GENERATOR_TRAINED', 'BLACK_GENERATOR_TRAINED')
        step('audit_plastic_bomo_stage20b.py', ['--action', 'freeze_generators'], 'BLACK_GENERATOR_TRAINED', 'GENERATOR_CHECKPOINTS_FROZEN')
        step('generate_plastic_bomo_stage20b_banks.py', ['--bank', 'sanity'], 'GENERATOR_CHECKPOINTS_FROZEN')
        for bank, before, after in (('A', 'GENERATOR_CHECKPOINTS_FROZEN', 'BANK_A_GENERATED'),
                                    ('B', 'BANK_A_GENERATED', 'BANK_B_GENERATED'),
                                    ('C', 'BANK_B_GENERATED', 'BANK_C_GENERATED')):
            step('generate_plastic_bomo_stage20b_banks.py', ['--bank', bank], before, after)
        step('audit_plastic_bomo_stage20b.py', ['--action', 'validate_banks'], 'BANK_C_GENERATED', 'SYNTHETIC_BANKS_VALIDATED')
        for arm, before, after in (('RR', 'SYNTHETIC_BANKS_VALIDATED', 'DETECTOR_RR_TRAINED'),
                                   ('A', 'DETECTOR_RR_TRAINED', 'DETECTOR_A_TRAINED'),
                                   ('B', 'DETECTOR_A_TRAINED', 'DETECTOR_B_TRAINED'),
                                   ('C', 'DETECTOR_B_TRAINED', 'DETECTOR_C_TRAINED')):
            step('train_plastic_bomo_stage20b_detector.py', ['--arm', arm], before, after)
        step('audit_plastic_bomo_stage20b.py', ['--action', 'freeze_detectors'], 'DETECTOR_C_TRAINED', 'ALL_DETECTORS_FROZEN')
        step('evaluate_plastic_bomo_stage20b.py', [], 'ALL_DETECTORS_FROZEN', 'FINAL_EVAL_COMPLETE')
        step('audit_plastic_bomo_stage20b.py', ['--action', 'finish'], 'FINAL_EVAL_COMPLETE')
        print('Stage20B complete. No TDCRG/Stage20C automatically started.', flush=True)
    except Exception as e:
        reason = e.status if isinstance(e, Stop) else phase
        if reason == 'FROZEN_RUNTIME_BINDING_CHANGED' or reason.startswith('R0_'):
            reason = 'FORMAL_FROZEN_BINDING_CHANGED'
        update_status(a.output, status='STOPPED_' + reason, reason=reason, detail=str(e), stopped_at=now(),
                      TDCRG_DEVELOPMENT_AUTHORIZED=False, full_experiment_complete=False)
        if phase == 'FORMAL_FROZEN_BINDING_CHANGED':
            save(a.output / 'formal_protocol_binding.json', {'status': 'FAILED', 'reason': reason, 'detail': str(e)})
        (a.output / 'logs' / 'orchestrator_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        (a.output / 'STAGE20B_REPORT.md').write_text('# Stage20B stopped\n\nReason: `' + reason + '`\n\n'
            'This is a partial run, not a headroom conclusion. No automatic retry, rescue or next stage. '
            'See stage20b_status.json and logs/.\n', encoding='utf-8')
        print(traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
