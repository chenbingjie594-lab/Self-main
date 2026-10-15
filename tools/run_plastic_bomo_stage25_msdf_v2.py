"""Stage25-P1 Linux/3090 runner. CLI/help/static imports never load ML packages.

Every GPU phase runs in a fresh child; failures stop without retries or rescue.
No SSH, installation, local training or historical-file mutation is performed.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

from stage25_msdf_contract import (BANKS, REPO, Stop, source_check, inspect_environment,
    installed_source_audit, bind_assets, baseline_audit, binding, safety_binding,
    claim, load, save, sha, require, now, update, verify_ledger, tree_binding, verified_copy)

ROOT = REPO / 'results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification'

PHASES = {'environment': '环境、数据和对照资产检查', 'safety': 'MSDF 单批次安全检查（无 optimizer.step）',
          'train': 'MSDF 生成器训练', 'raw_train': 'Raw 对照生成器重建', 'generate': '原始图像生成',
          'validate': '生成文件结构校验', 'detector': '检测器训练', 'evaluate': '统一评估'}


def describe_error(status, detail, run=None, exit_code=1):
    """Readable terminal diagnosis; JSON artifacts retain machine-readable details."""
    print(f'\n[已退出] Stage25 已停止，退出码={exit_code}，状态={status}', flush=True)
    if isinstance(detail, dict):
        if detail.get('reason'):
            print('原因：' + str(detail['reason']), flush=True)
        for name in detail.get('missing_files', []):
            print('缺少文件：' + name, flush=True)
        if detail.get('file'):
            print('问题文件：' + str(detail['file']), flush=True)
        if detail.get('log'):
            print('阶段日志：' + str(detail['log']), flush=True)
        print(json.dumps(detail, ensure_ascii=False, indent=2), flush=True)
    else:
        print('原因：' + str(detail), flush=True)
    if run is not None:
        print('本次结果目录：' + str(run), flush=True)
    print('当前执行已结束；没有自动重试。修正报错后再手动启动。', flush=True)


def stream_child(command, logfile, phase, cwd):
    """Tee one owned child to terminal + phase log; heartbeat while output is quiet."""
    started = time.monotonic()
    with Path(logfile).open('x', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'})
        print(f'[运行中] {phase}；子进程 PID={proc.pid}；日志={logfile}', flush=True)
        messages = queue.Queue()
        def pump():
            try:
                for line in proc.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        def emit(line):
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        try:
            while True:
                try:
                    line = messages.get(timeout=15)
                except queue.Empty:
                    if proc.poll() is None:
                        emit(f'[运行中] {phase}，已运行 {int(time.monotonic() - started)} 秒；等待阶段输出。\n')
                    continue
                if line is None:
                    break
                emit(line)
            code = proc.wait()
            reader.join(timeout=1)
        except BaseException:
            # Only this runner's child is stopped. No unrelated GPU job is touched.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            raise
        finally:
            proc.stdout.close()
        emit(f'[阶段结束] {phase}，退出码={code}，耗时={int(time.monotonic() - started)}秒。\n')
    return code


def create_output(root, prefix):
    root = Path(root).resolve()
    require(root == ROOT or root.is_relative_to(ROOT), 'STAGE25_OUTPUT_PATH_INVALID', 'outputs must be new children of Stage25 root')
    path = root / (prefix + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    path.mkdir(parents=True, exist_ok=False)
    (path / 'logs').mkdir()
    return path


def environment(a):
    cfg = source_check(a.protocol, a.code_manifest)
    result = inspect_environment(cfg)
    save(a.run / 'stage25_server_environment.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    require(not result['differences'], 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', result['differences'])
    import torch
    from qualify_plastic_bomo_stage20b_r0 import set_gpu_runtime
    set_gpu_runtime(torch, 'cuda:' + a.device)
    require('RTX 3090' in torch.cuda.get_device_name(), 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'RTX3090 only; no memory/algorithm rescue')
    result['source_hashes'] = installed_source_audit(a.p0)
    previous = load(a.r0 / 'runtime_environment.json')
    require(result['cudnn_version'] == previous['cudnn_version'] and list(torch.cuda.get_device_capability()) == previous['capability'],
            'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'native CUDA/cuDNN/capability drift')
    save(a.run / 'stage25_server_environment.json', result)
    assets = bind_assets(a, cfg, a.run)
    print('[检查] TRAIN 资产校验通过；正在检查历史 Raw/RR 复用资格。', flush=True)
    reuse = baseline_audit(a, cfg)
    save(a.run / 'baseline_reuse_audit_server.json', reuse)
    save(a.run / 'stage25_runtime_binding.json', {'stage25_protocol': str(a.protocol), 'code_manifest': str(a.code_manifest),
        'protocol_sha256': sha(a.protocol), 'code_manifest_sha256': sha(a.code_manifest),
        'stage20a': str(a.stage20a), 'p0': str(a.p0), 'p0_view': assets['p0_view'], 'r0': str(a.r0),
        'baseline_run': str(a.baseline_run), 'base_model': str(a.base_model), 'detector_model': str(a.detector_model),
        'real_root': assets['real_root'], 'normal_root': assets['normal_root'], 'device': 'cuda:' + a.device,
        'controller_cfg': str(a.run / 'controller_cfg.json'), 'environment_sha256': sha(a.run / 'stage25_server_environment.json')})


def qualify(runtime):
    b, cfg = safety_binding(runtime)
    for cls in ('flash', 'black'):
        row = load(runtime / ('msdf_' + cls + '_safety.json'))
        require(row['status'] == 'PASS' and row['optimizer_steps'] == row['sampling_count'] == row['final_eval_content_read'] == 0,
                'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', cls)
    result = {'status': 'PASS', 'scope': 'LINUX_RTX3090_SERVER_RUNTIME', 'classes': ['flash', 'black'],
        'MSDF_trainable_connectivity': 'PASS', 'UNet_trainable_connectivity': 'PASS',
        'optimizer_step_count': 0, 'sampling_count': 0, 'official_final_eval_content_read': 0,
        'environment_sha256': sha(runtime / 'stage25_server_environment.json'), 'protocol_sha256': b['protocol_sha256'],
        'code_manifest_sha256': b['code_manifest_sha256'], 'qualified_at': now(),
        'FORMAL_EXECUTION_AUTHORIZED': True, 'MSDF_performance_claim': 'NOT_TESTED'}
    save(runtime / 'stage25_runtime_qualification.json', result)
    files = ('stage25_server_environment.json', 'stage25_runtime_binding.json', 'stage25_runtime_qualification.json',
        'msdf_flash_safety.json', 'msdf_black_safety.json', 'server_asset_hash_audit.json',
        'baseline_reuse_audit_server.json', 'controller_cfg.json')
    save(runtime / 'stage25_runtime_artifact_manifest.json', {'sha256': {n: sha(runtime / n) for n in files}})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def common_command(a, run):
    args = [sys.executable, '-u', str(Path(__file__).resolve()), '--run', str(run), '--protocol', str(a.protocol),
        '--code_manifest', str(a.code_manifest), '--stage20a', str(a.stage20a), '--p0', str(a.p0),
        '--r0', str(a.r0), '--baseline_run', str(a.baseline_run), '--base_model', str(a.base_model),
        '--detector_model', str(a.detector_model), '--device', a.device]
    for k in ('real_root', 'normal_root'):
        if getattr(a, k):
            args += ['--' + k, str(getattr(a, k))]
    return args


def child(a, run, job, cls=None, bank=None, comparator=False):
    name = job + ('_' + cls if cls else '') + ('_' + bank if bank else '') + ('_raw' if comparator else '')
    command = common_command(a, run) + ['--_job', job]
    if cls:
        command += ['--class_name', cls]
    if bank:
        command += ['--bank', bank]
    if comparator:
        command += ['--comparator']
    log = Path(run) / 'logs' / (name + '.log')
    require(not log.exists(), 'STAGE25_PHASE_ALREADY_ATTEMPTED', name)
    phase = PHASES[job] + (' / ' + cls if cls else '') + (' / ' + bank if bank else '')
    print('[开始] ' + phase + '；结果目录：' + str(run), flush=True)
    code = stream_child(command, log, phase, REPO)
    if code:
        error = Path(run) / 'worker_failure.json'
        detail = load(error) if error.is_file() else {'status': 'STAGE25_CHILD_PROCESS_FAILED', 'exit_code': code}
        raise Stop(detail['status'], {**detail, 'log': str(log), 'exit_code': code})


def runtime_check(a):
    runtime = create_output(a.output_root, 'server_runtime')
    save(runtime / 'stage25_runtime_qualification.json', {'status': 'NOT_RUN', 'FORMAL_EXECUTION_AUTHORIZED': False})
    for name in ('stage25_server_environment.json', 'msdf_flash_safety.json', 'msdf_black_safety.json',
                 'server_asset_hash_audit.json', 'baseline_reuse_audit_server.json'):
        save(runtime / name, {'status': 'NOT_RUN', 'scope': 'SERVER_RUNTIME', 'optimizer_step_count': 0,
                             'sampling_count': 0, 'final_eval_content_read': 0})
    try:
        child(a, runtime, 'environment')
        for cls in ('flash', 'black'):
            child(a, runtime, 'safety', cls=cls)
        qualify(runtime)
    except KeyboardInterrupt:
        failure(runtime, Stop('STAGE25_INTERRUPTED', '用户中断，运行已停止，未自动重试。'), runtime=True, exit_code=130)
        raise
    except Exception as error:
        failure(runtime, error, runtime=True)
        raise
    print('Qualified runtime: ' + str(runtime), flush=True)
    return runtime


def new_formal(runtime, a):
    runtime = Path(runtime).resolve()
    rb = load(runtime / 'stage25_runtime_binding.json')
    require(load(runtime / 'stage25_runtime_qualification.json')['status'] == 'PASS', 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'runtime PASS prerequisite')
    require(rb['protocol_sha256'] == sha(a.protocol) and rb['code_manifest_sha256'] == sha(a.code_manifest), 'STAGE25_FROZEN_BINDING_CHANGED', 'same shared recipe')
    run = create_output(a.output_root, 'formal')
    b = {**rb, 'runtime_dir': str(runtime), 'qualification_sha256': sha(runtime / 'stage25_runtime_qualification.json'),
         'runtime_artifact_manifest_sha256': sha(runtime / 'stage25_runtime_artifact_manifest.json')}
    save(run / 'stage25_run_binding.json', b)
    save(run / 'stage25_status.json', {'status': 'FORMAL_RUNNING', 'full_experiment_complete': False,
        'DeepPCB': 0, 'BootstrapGuard': 0, 'DWBG': 0, 'selector': 0, 'official_final_eval_rounds': 0,
        'generator_training_count': 0, 'synthetic_generation_count': 0, 'detector_training_count': 0,
        'automatic_rescue': False, 'started_at': now()})
    binding(run)
    verified_copy(a.protocol, run / 'msdf_v2_protocol.json', sha(a.protocol))
    verified_copy(a.code_manifest, run / 'server_code_manifest.json', sha(a.code_manifest))
    for name in ('msdf_historical_component_audit.json', 'msdf_selector_separation_audit.json'):
        src = ROOT / name
        verified_copy(src, run / name, sha(src))
    print('Formal Stage25: ' + str(run), flush=True)
    return run


def comparator_run(run):
    b, _ = binding(run)
    require(not load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')['reuse'], 'STAGE25_COMPARATOR_REBUILD_NOT_NEEDED', 'prefer qualified historical reuse')
    root = Path(run) / 'comparator_rebuild'
    root.mkdir()
    (root / 'logs').mkdir()
    save(root / 'stage25_run_binding.json', b)
    save(root / 'stage25_status.json', {'status': 'FORMAL_RUNNING', 'role': 'Raw/RR same-runtime rebuild only', 'full_experiment_complete': False})
    return root


def formal_train(a, run):
    from stage25_msdf_generator import freeze_generators
    b, _ = binding(run)
    reuse = load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')
    if not reuse['reuse']:
        raw = comparator_run(run)
        for cls in ('flash', 'black'):
            child(a, raw, 'raw_train', cls=cls, comparator=True)
        freeze_generators(raw, raw=True)
    for cls in ('flash', 'black'):
        child(a, run, 'train', cls=cls)
    freeze_generators(run)
    update(run, generator_training_count=2, msdf_flash_successful_updates=2000, msdf_black_successful_updates=2000,
           comparator_generator_training_count=0 if reuse['reuse'] else 2)


def formal_generate(a, run):
    b, _ = binding(run)
    from stage25_msdf_generator import validate_banks
    reuse = load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')['reuse']
    if not reuse:
        raw = Path(run) / 'comparator_rebuild'
        for bank in BANKS:
            child(a, raw, 'generate', bank=bank, comparator=True)
        child(a, raw, 'validate', comparator=True)
    for bank in BANKS:
        child(a, run, 'generate', bank=bank)
    child(a, run, 'validate')
    update(run, synthetic_generation_count=240, comparator_synthetic_generation_count=0 if reuse else 240)


def formal_detector(a, run):
    b, _ = binding(run)
    from stage25_msdf_downstream import freeze_detectors
    if not load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')['reuse']:
        raw = Path(run) / 'comparator_rebuild'
        for bank in ('RR', *BANKS):
            child(a, raw, 'detector', bank=bank, comparator=True)
        freeze_detectors(raw, comparator=True)
    for bank in BANKS:
        child(a, run, 'detector', bank=bank)
    freeze_detectors(run)
    update(run, detector_training_count=3,
           comparator_detector_training_count=0 if load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')['reuse'] else 4)


def failure(run, error, runtime=False, exit_code=1):
    status = getattr(error, 'status', 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED' if runtime else 'STAGE25_EXECUTION_FAILED')
    reason = status
    if runtime:
        status = 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED'
    if 'out of memory' in str(error).lower() or type(error).__name__ == 'OutOfMemoryError':
        status = 'STAGE25_SERVER_RUNTIME_OOM' if runtime else 'STAGE25_FORMAL_OOM'
    detail = getattr(error, 'detail', str(error))
    try:
        json.dumps(detail, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        detail = str(detail)
    row = {'status': status, 'failure_reason': reason, 'detail': detail, 'at': now(),
        'full_experiment_complete': False, 'automatic_rescue': False, 'historical_P0_modified': False}
    if runtime:
        row['FORMAL_EXECUTION_AUTHORIZED'] = False
        save(Path(run) / 'stage25_runtime_qualification.json', row)
    elif (Path(run) / 'stage25_status.json').exists():
        update(run, **row)
    save(Path(run) / 'worker_failure.json', row)
    describe_error(status, detail, run, exit_code)


def worker(a):
    require(a.run is not None, 'STAGE25_ARGUMENT_INVALID', '--run required')
    # Internal runtime jobs may run before qualification; all others require PASS.
    if a._job == 'environment':
        environment(a)
    elif a._job == 'safety':
        from stage25_msdf_generator import safety
        safety(a.run, a.class_name)
    else:
        binding(a.run)
        require(load(a.run / 'stage25_status.json')['status'] == 'FORMAL_RUNNING', 'STAGE25_PHASE_SEQUENCE_INVALID', 'stopped/completed run')
        from stage25_msdf_generator import train, generate, validate_banks
        from stage25_msdf_downstream import raw_train, detector, evaluate
        if a._job == 'train':
            train(a.run, a.class_name)
        elif a._job == 'raw_train':
            raw_train(a.run, a.class_name)
        elif a._job == 'generate':
            generate(a.run, a.bank, raw=a.comparator)
        elif a._job == 'validate':
            claim(a.run, 'validate_banks')
            validate_banks(a.run)
        elif a._job == 'detector':
            detector(a.run, a.bank, comparator=a.comparator)
        elif a._job == 'evaluate':
            evaluate(a.run)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('runtime_check', 'train', 'generate', 'detector', 'evaluate', 'all'), default='runtime_check')
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage25_server.json')
    p.add_argument('--code_manifest', type=Path, default=REPO / 'configs/plastic_bomo_stage25_server_code_manifest.json')
    p.add_argument('--stage20a', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol')
    p.add_argument('--p0', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115')
    p.add_argument('--r0', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254')
    p.add_argument('--baseline_run', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543')
    p.add_argument('--base_model', type=Path, default=REPO / 'sd2-community/stable-diffusion-2-inpainting')
    p.add_argument('--detector_model', type=Path, default=REPO / 'pretrained/yolo11s.pt')
    p.add_argument('--real_root', type=Path, help='explicit actual path, identity-preserving relocation only')
    p.add_argument('--normal_root', type=Path, help='explicit frozen394 TRAIN normal directory')
    p.add_argument('--device', default='0', help='visible CUDA index, not physical GPU index')
    p.add_argument('--output_root', type=Path, default=ROOT)
    p.add_argument('--qualified_runtime', type=Path, help='runtime_check output to use with --mode train')
    p.add_argument('--run', type=Path, help='existing new formal dir for next phase; failed phases cannot be retried')
    p.add_argument('--_job', choices=('environment', 'safety', 'train', 'raw_train', 'generate', 'validate', 'detector', 'evaluate'), help=argparse.SUPPRESS)
    p.add_argument('--class_name', choices=('flash', 'black'), help=argparse.SUPPRESS)
    p.add_argument('--bank', choices=('RR', *BANKS), help=argparse.SUPPRESS)
    p.add_argument('--comparator', action='store_true', help=argparse.SUPPRESS)
    return p


def main():
    a = parser().parse_args()
    for k in ('protocol', 'code_manifest', 'stage20a', 'p0', 'r0', 'baseline_run', 'base_model', 'detector_model', 'output_root', 'run', 'qualified_runtime', 'real_root', 'normal_root'):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    target = a.run
    try:
        print(f'[启动] Stage25；模式={a._job or a.mode}；PID={os.getpid()}；项目={REPO}', flush=True)
        print('[检查] 冻结运行代码和协议。测试文件、命令说明不属于服务器运行必需文件。', flush=True)
        source_check(a.protocol, a.code_manifest)
        if a._job:
            worker(a)
            print('[阶段完成] ' + a._job + '，退出码=0。', flush=True)
            return
        if a.mode == 'runtime_check':
            runtime_check(a)
            print('[已退出] Runtime 检查完成，退出码=0；未启动正式训练。', flush=True)
            return
        if a.mode == 'all':
            runtime = runtime_check(a)
            target = new_formal(runtime, a)
            formal_train(a, target)
            formal_generate(a, target)
            formal_detector(a, target)
            child(a, target, 'evaluate')
        elif a.mode == 'train':
            require(a.qualified_runtime is not None and a.run is None, 'STAGE25_ARGUMENT_INVALID', 'train requires --qualified_runtime, new formal dir; no resume')
            target = new_formal(a.qualified_runtime, a)
            formal_train(a, target)
        else:
            require(a.run is not None, 'STAGE25_ARGUMENT_INVALID', '--run formal dir required')
            if a.mode == 'generate':
                formal_generate(a, a.run)
            elif a.mode == 'detector':
                formal_detector(a, a.run)
            elif a.mode == 'evaluate':
                child(a, a.run, 'evaluate')
        print('[已退出] 本次请求执行完成，退出码=0。请查看结果目录中的 status/report 判定科学结果。', flush=True)
    except KeyboardInterrupt:
        error = Stop('STAGE25_INTERRUPTED', '用户中断；仅停止本次任务，无自动重试。')
        if target is not None and target.is_dir():
            failure(target, error, exit_code=130)
        else:
            describe_error(error.status, error.detail, exit_code=130)
        raise SystemExit(130)
    except Exception as error:
        if target is not None and target.is_dir():
            failure(target, error, runtime=a._job in ('environment', 'safety'))
            (target / 'logs' / 'last_failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        else:
            describe_error(getattr(error, 'status', 'STAGE25_PREPARATION_FAILED'), getattr(error, 'detail', str(error)))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
