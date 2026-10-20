"""Check all uploads first, then optionally run native GPU probes serially."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
AD_BERT_TOKENIZER = Path('/mnt/sda/cbj/GLIP-main/bert-base-uncased')
METHODS = ('anomalydiffusion', 'defectfill')
REQUIRED_FILES = (
    'tools/chapter3_exp1_quality_common.py',
    'tools/chapter3_exp2_baseline_common.py',
    'tools/exp2_v2_data.py',
    'tools/check_plastic_bomo_exp2_runtime.py',
    'configs/plastic_bomo_chapter3_exp1.json',
    'configs/plastic_bomo_chapter3_exp2.json',
    'configs/plastic_bomo_exp2_source_repairs.json',
)


def sha(path, normalize=False):
    if normalize:
        return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_inputs(args, repo=REPO):
    """No ML imports, model forwards, asset conversion or environment changes."""
    errors = []
    for name in REQUIRED_FILES:
        if not (repo / name).is_file():
            errors.append('[文件缺失] ' + str(repo / name))
    repair = repo / 'configs/plastic_bomo_exp2_source_repairs.json'
    try:
        hashes = json.loads(repair.read_text(encoding='utf-8'))['patched_source_sha256_LF']
    except (OSError, ValueError, KeyError) as exc:
        hashes = {}
        errors.append('[清单无法读取] ' + str(exc))
    selected = METHODS if args.method == 'all' else (args.method,)
    for method in selected:
        source = getattr(args, method + '_source')
        python = getattr(args, method + '_python')
        model = getattr(args, method + '_model')
        for name, expected in hashes.get(method, {}).items():
            path = source / name
            if not path.is_file():
                errors.append('[源码缺失] ' + str(path))
            elif sha(path, normalize=True) != expected:
                errors.append('[源码版本不符] ' + str(path))
        if method not in hashes:
            errors.append('[清单缺项] ' + method)
        if not python.is_file() or not os.access(python, os.X_OK):
            errors.append('[解释器不存在或不可执行] ' + method + ': ' + str(python))
        if method == 'anomalydiffusion':
            if not model.is_file() or model.stat().st_size == 0:
                errors.append('[原生LDM权重缺失] ' + str(model))
            for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.txt'):
                if not (AD_BERT_TOKENIZER / name).is_file():
                    errors.append('[BERT tokenizer 文件缺失] ' + str(AD_BERT_TOKENIZER / name))
        elif not (model / 'model_index.json').is_file():
            errors.append('[本地SD2模型目录不完整] ' + str(model / 'model_index.json'))
    try:
        binding = json.loads((args.prepared / 'prepared_binding.json').read_text(encoding='utf-8'))
        if binding.get('status') != 'FROZEN':
            raise ValueError('prepared_binding.status must be FROZEN')
        for name, key in (
            ('training_manifest.json', 'training_manifest_sha256'),
            ('generation_slots.json', 'generation_slots_sha256'),
            ('name-anomaly.txt', 'name_anomaly_sha256'),
        ):
            if sha(args.prepared / name) != binding[key]:
                raise ValueError('prepared file hash mismatch: ' + name)
        value = json.loads((args.prepared / 'training_manifest.json').read_text(encoding='utf-8'))
        rows = value['records']
        if value.get('status') != 'TRAIN_ONLY_ASSETS_PREPARED' or len(rows) != 172 or Counter(r['class'] for r in rows) != {'flash': 75, 'black': 97}:
            raise ValueError('TRAIN population/status changed')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append('[准备数据无效] ' + str(exc))
    return selected, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True,
                        help='reuse the existing prepared_v2 directory; do not prepare again')
    parser.add_argument('--method', choices=('all', *METHODS), default='all')
    parser.add_argument('--run', action='store_true', help='run GPU probes ONLY after all selected checks pass')
    parser.add_argument('--gpu', default='1', help='physical CUDA_VISIBLE_DEVICES value')
    for method, folder, python, model in (
        ('anomalydiffusion', 'anomalydiffusion', '/mnt/sda/cbj/miniconda3/envs/Anomalydiffusion/bin/python',
         REPO.parent / 'third_party/anomalydiffusion/models/ldm/text2img-large/model.ckpt'),
        ('defectfill', 'defectfill', '/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python',
         REPO / 'sd2-community/stable-diffusion-2-inpainting'),
    ):
        parser.add_argument('--' + method + '_source', type=Path, default=REPO.parent / 'third_party' / folder)
        parser.add_argument('--' + method + '_python', type=Path, default=Path(python))
        parser.add_argument('--' + method + '_model', type=Path, default=Path(model))
    args = parser.parse_args()
    args.prepared = args.prepared.resolve()
    print('[启动] Exp2 上传/环境/数据绑定检查；不安装依赖，不训练或生成。', flush=True)
    try:
        selected, errors = inspect_inputs(args)
    except Exception as exc:
        print('[已退出] 前置检查异常：' + str(exc) + '；退出码=1', flush=True)
        return 1
    if errors:
        for item in errors:
            print(item, flush=True)
        print('[已退出] 前置检查失败，共%d项；未启动GPU检查。退出码=1' % len(errors), flush=True)
        return 1
    print('[检查通过] 所选源码、解释器路径、基础模型路径和准备数据绑定均通过；原生依赖/模型梯度尚未验证。', flush=True)
    if not args.run:
        print('[已退出] 仅静态前置检查完成；GPU检查未启动。退出码=0', flush=True)
        return 0
    output = args.prepared / ('runtime_checks_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    # AD's pinned PyTorch 1.12.1 does not recognize the newer
    # ``expandable_segments`` allocator option. Also discard any inherited
    # setting so a user's shell cannot inject an unsupported allocator option.
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    for method in selected:
        method_env = dict(env)
        if method == 'anomalydiffusion':
            # Torch 1.12 supports max_split_size_mb but not the newer
            # expandable_segments option.  Keep large transient 512x512
            # attention allocations from fragmenting the 24 GiB device.
            method_env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            # These pinned dependency checkouts are siblings of Self-main,
            # so they are not discoverable from the AD source path.
            dependency_sources = (
                REPO.parent / 'third_party' / 'taming-transformers',
                REPO.parent / 'third_party' / 'openai-CLIP',
            )
            inherited_pythonpath = method_env.get('PYTHONPATH', '')
            method_env['PYTHONPATH'] = os.pathsep.join(
                item for item in (*map(str, dependency_sources), inherited_pythonpath) if item
            )
            method_env['AD_BERT_TOKENIZER_PATH'] = str(AD_BERT_TOKENIZER.resolve())
            # The frozen AD environment has no Ninja package.  Its two
            # StyleGAN2 CUDA extensions are prebuilt with the matching Torch
            # ABI; this narrow shim validates and reuses only fresh artifacts.
            method_env['PATH'] = os.pathsep.join((
                str(REPO / 'tools' / 'exp2_ninja_shim'),
                method_env.get('PATH', ''),
            ))
        report = output / (method + '_runtime_probe.json')
        command = [str(getattr(args, method + '_python')), '-u', str(REPO / 'tools/check_plastic_bomo_exp2_runtime.py'),
                   '--method', method, '--source', str(getattr(args, method + '_source').resolve()),
                   '--manifest', str(args.prepared / 'training_manifest.json'),
                   '--base_model', str(getattr(args, method + '_model').resolve()),
                   '--device', '0', '--output', str(report)]
        print('[启动] 串行GPU检查 ' + method + '；报告：' + str(report), flush=True)
        try:
            code = subprocess.run(command, cwd=str(REPO), env=method_env, check=False).returncode
        except OSError as exc:
            print('[启动失败] ' + str(exc), flush=True)
            code = 1
        if code == 0:
            try:
                result = json.loads(report.read_text(encoding='utf-8'))
                if (result.get('status') != 'REPAIRED_GRADIENT_PROBE_PASS_NOT_FORMAL_AUTHORIZATION'
                        or result.get('method') != method
                        or result.get('formal_training_authorized') is not False
                        or result.get('formal_generation_authorized') is not False
                        or any(result.get(k) != 0 for k in ('optimizer_step_count', 'synthetic_generation_count',
                                                          'detector_training_count', 'final_eval_image_or_label_content_read'))
                        or sorted(r.get('class_name', '') for r in result.get('records', [])) != ['black', 'flash']):
                    raise ValueError('GPU probe report is missing required PASS/zero-action/class records')
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                print('[报告无效] ' + str(exc), flush=True)
                code = 1
        print('[已退出] ' + method + ' 检查退出码=' + str(code), flush=True)
        if code:
            print('[已退出] 检查失败，后续方法未启动；保留新旧报告。退出码=1', flush=True)
            return 1
    print('[已退出] 所选GPU梯度检查完成；仍未批准正式训练或生成。报告目录：' + str(output) + '；退出码=0', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
