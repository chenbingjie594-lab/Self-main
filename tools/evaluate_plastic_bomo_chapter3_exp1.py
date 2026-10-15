"""Experiment1 only: evaluate frozen Raw/MSDF images, never train or sample.

FID/KID and Inception are imported from torch-fidelity, LPIPS from official lpips.
No local metric formula, network, selector, detector or generator is implemented.
"""
import argparse
from contextlib import contextmanager
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

from chapter3_exp1_quality_common import (REPO, CONFIG, OUTPUT, BANKS, CLASSES, METHODS,
    ZERO, Stop, base_status, checked_file, csv_write, dependencies, implementation_binding,
    load, mean_sd, paper_tables, portable, prepare, require, resolve, save, sha, summarize, verify_prepared)


@contextmanager
def offline_metrics():
    """No implicit model-weight fetching. Existing environment is never installed/updated."""
    original_connect = socket.socket.connect
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise Stop('DEPENDENCY_REQUIRED', '指标权重必须预先存在；禁止自动联网下载。')
        return original_connect(sock, address)
    socket.socket.connect = connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect


@contextmanager
def heartbeat(label):
    done = threading.Event()
    started = time.monotonic()
    def wait():
        while not done.wait(20):
            print(f'[运行中] {label}，耗时 {int(time.monotonic()-started)} 秒。', flush=True)
    worker = threading.Thread(target=wait, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set(); worker.join(timeout=1)


def pending_outputs(output, status, detail, dependency=None):
    """Unavailable is null/blank, never zero or a fabricated visual-quality result."""
    if (output / 'experiment1_status.json').exists():
        require(not load(output / 'experiment1_status.json').get('full_experiment_complete'),
                'RESULT_ALREADY_COMPLETE', '完成的结果不会改回未计算状态。')
    row = base_status(status)
    row.update(detail=detail, metric_forward_count=0, metrics_computed=False, plot_status='NOT_RUN')
    save(output / 'experiment1_status.json', row)
    if dependency is not None:
        save(output / 'dependency_report.json', dependency)
    missing = {'status': 'NOT_COMPUTED', 'reason': status, 'values': None, **ZERO}
    for n in ('generation_quality_full_image.json', 'generation_quality_roi.json', 'fid_bankwise.json',
              'kid_bankwise.json', 'lpips_summary.json', 'generation_quality_classwise.json'):
        if not (output / n).exists():
            save(output / n, missing)
    save(output / 'generation_quality_summary.json', {**missing, 'detail': detail})
    if not (output / 'lpips_per_slot.csv').exists():
        csv_write(output / 'lpips_per_slot.csv', ['Method', 'slot_id', 'class', 'LPIPS_A_B', 'LPIPS_A_C', 'LPIPS_B_C', 'LPIPS_slot'], [])
    if not (output / 'paper_table_generation_quality.csv').exists():
        csv_write(output / 'paper_table_generation_quality.csv', ['Method', 'FID↓', 'KID↓', 'LPIPS↑'], [[m, None, None, None] for m in METHODS])
        csv_write(output / 'paper_table_generation_quality_classwise.csv',
                  ['Method', 'Flash FID', 'Flash KID', 'Flash LPIPS', 'Black FID', 'Black KID', 'Black LPIPS'],
                  [[m, *([None] * 6)] for m in METHODS])
    (output / 'EXPERIMENT1_REPORT.md').write_text(
        '# Experiment1 — frozen Raw SD2 vs MSDF visual metrics\n\n'
        + f'Status: `{status}`. **FID/KID/LPIPS have not been computed.**\n\n'
        + 'The paired480 generated images and48 real images /60 original boxes have been frozen in the manifests. '
        + 'No image selection, regeneration, training or detector evaluation is authorized.\n\n'
        + 'Missing values in CSV/JSON are unavailable, not zero. No metric chart is fabricated.\n\n'
        + 'See dependency_report.json for missing metric packages/weights. No package was installed or upgraded.\n',
        encoding='utf-8', newline='\n')


def runtime_modules(cfg, dep, device):
    import torch
    import numpy as np
    import lpips
    from PIL import Image
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric
    require(device == 'cpu' or (device.startswith('cuda:') and torch.cuda.is_available()),
            'DEPENDENCY_REQUIRED', '指定 CUDA 不可用；不会自动切换设备。')
    require(all(callable(x) for x in (fid_features_to_statistics, fid_statistics_to_metric, kid_features_to_metric)),
            'DEPENDENCY_REQUIRED', 'torch-fidelity API unavailable')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device.startswith('cuda:'):
        torch.cuda.set_device(torch.device(device))
    modules = ('torch_fidelity.feature_extractor_inceptionv3', 'torch_fidelity.interpolate_compat_tensorflow',
               'torch_fidelity.metric_fid', 'torch_fidelity.metric_kid', 'lpips.lpips', 'lpips.pretrained_networks')
    sources = {name: {'path': str(Path(importlib.import_module(name).__file__).resolve()),
                      'sha256': sha(importlib.import_module(name).__file__)} for name in modules}
    runtime = {**dep, 'status': 'FROZEN_BEFORE_METRIC_FORWARD', 'device': device,
        'python': sys.version, 'torch_build': torch.__version__, 'CUDA': torch.version.cuda,
        'metric_source_files': sources, 'backbone': 'inception-v3-compat 2048 + LPIPS AlexNet v0.1',
        'feature_extraction': 'float32 eval/inference_mode; uint8 RGB -> native torch-fidelity preprocessing',
        'metric_accumulation': 'float64 CPU', 'TF32': False, 'autocast': False,
        'model_training': False, 'weight_and_package_binding_same_for_both_methods': True}
    return {'torch': torch, 'np': np, 'Image': Image, 'lpips': lpips,
            'Inception': FeatureExtractorInceptionV3, 'fid_stats': fid_features_to_statistics,
            'fid': fid_statistics_to_metric, 'kid': kid_features_to_metric}, runtime


def freeze_json(path, value):
    if Path(path).exists():
        require(load(path) == value, 'METRIC_RUNTIME_CHANGED', '本实验已绑定其他权重/版本/设备；不覆盖旧记录：' + str(path))
    else:
        save(path, value)


def image_items(pairs, reference):
    result = []
    for r in reference['images']:
        result.append({'key': 'real:full:' + r['node_id'], 'region': 'full_image', 'role': 'real',
            'classes': r['classes'], 'path': r['image_path'], 'canvas_wh': r['canvas_wh'], 'crop': None})
    for r in reference['instances']:
        result.append({'key': 'real:roi:' + r['annotation_id'], 'region': 'roi', 'role': 'real',
            'classes': [r['class']], 'path': r['image_path'], 'canvas_wh': r['canvas_wh'], 'crop': r['roi_xyxy']})
    for method, prefix in zip(METHODS, ('raw', 'msdf')):
        for r in pairs['records']:
            for region in ('full_image', 'roi'):
                result.append({'key': f'{prefix}:{region}:{r["bank"]}:{r["slot_id"]}', 'region': region,
                    'role': method, 'bank': r['bank'], 'slot_id': r['slot_id'], 'classes': [r['class']],
                    'path': r[prefix + '_image_path'], 'canvas_wh': r['canvas_wh'],
                    'crop': r['roi_xyxy'] if region == 'roi' else None})
    require(len(result) == 1068 and len({x['key'] for x in result}) == 1068, 'FEATURE_POPULATION_INVALID', '1068 full/ROI features')
    return result


def open_rgb(item, Image):
    with Image.open(resolve(item['path'])) as source:
        require(list(source.size) == item['canvas_wh'], 'INPUT_CANVAS_CHANGED', item['path'])
        image = source.convert('RGB')
    return image.crop(item['crop']) if item['crop'] else image


def extract_features(output, items, lib, runtime):
    torch, np, Image = lib['torch'], lib['np'], lib['Image']
    cache = output / 'inception_features.npz'
    ledger = output / 'inception_feature_cache.json'
    keys = [r['key'] for r in items]
    if ledger.exists():
        b = load(ledger)
        checked_file(cache, b['sha256'])
        require(b['metric_runtime_sha256'] == sha(output / 'metric_runtime.json') and b['keys'] == keys,
                'FEATURE_CACHE_BINDING_CHANGED', str(cache))
        with np.load(cache, allow_pickle=False) as saved:
            array = saved['features'].copy()
    else:
        model = lib['Inception'](name='inception-v3-compat', features_list=['2048'],
                feature_extractor_weights_path=runtime['weights']['inception']['path']).float().eval().requires_grad_(False).to(runtime['device'])
        values = []
        with torch.inference_mode(), heartbeat('标准 Inception 特征提取；不训练生成器'):
            for i, item in enumerate(items):
                image = open_rgb(item, Image)
                data = np.asarray(image, dtype=np.uint8).copy()
                x = torch.from_numpy(data).permute(2, 0, 1)[None].to(runtime['device'])
                features = model(x)[0]
                require(tuple(features.shape) == (1, 2048) and bool(torch.isfinite(features).all()),
                        'NONFINITE_INCEPTION_FEATURE', item['key'])
                values.append(features[0].double().cpu().numpy())
                if (i + 1) % 40 == 0 or i == len(items) - 1:
                    print(f'[Inception] {i+1}/{len(items)} 个 full/ROI 输入，全部保留。', flush=True)
        array = np.stack(values)
        np.savez_compressed(cache, features=array)
        save(ledger, {'status': 'COMPLETE', 'sha256': sha(cache), 'keys': keys, 'feature_count': len(items),
            'dimension': 2048, 'same_standard_extractor_for_all': True,
            'metric_runtime_sha256': sha(output / 'metric_runtime.json')})
        del model
        if runtime['device'].startswith('cuda:'):
            torch.cuda.empty_cache()
    require(array.shape == (1068, 2048) and bool(np.isfinite(array).all()), 'FEATURE_CACHE_BINDING_CHANGED', 'feature matrix')
    return {key: torch.from_numpy(array[i]) for i, key in enumerate(keys)}


def kid_options(cfg, version, nreal, nfake):
    k = cfg['distribution_metrics']['KID']
    options = {'kid_subsets': k['subsets'], 'kid_subset_size': min(50, nreal, nfake), 'rng_seed': k['rng_seed'], 'verbose': False}
    if version == '0.3.0':
        options.update(kid_degree=k['degree'], kid_gamma=k['gamma'], kid_coef0=k['coef0'])
    elif version == '0.4.0':
        options.update(kid_kernel='poly', kid_kernel_poly_degree=k['degree'], kid_kernel_poly_gamma=k['gamma'], kid_kernel_poly_coef0=k['coef0'])
    else:
        raise Stop('DEPENDENCY_REQUIRED', 'Unsupported torch-fidelity API: ' + version)
    return options


def subset_coverage(np, nreal, nfake, options):
    """Reproduce ONLY library index choices to audit coverage; not a KID formula."""
    rng = np.random.RandomState(options['rng_seed'])
    real, fake = [0] * nreal, [0] * nfake
    choices = []
    for _ in range(options['kid_subsets']):
        rr = rng.choice(nreal, options['kid_subset_size'], replace=False).tolist()
        ff = rng.choice(nfake, options['kid_subset_size'], replace=False).tolist()
        for i in rr:
            real[i] += 1
        for i in ff:
            fake[i] += 1
        choices.append({'real_indices': rr, 'generated_indices': ff})
    require(min(real) > 0 and min(fake) > 0, 'KID_SUBSET_COVERAGE_INCOMPLETE', '不改 seed、不删图；本次停止并报告。')
    return {'all_real_inputs_used': True, 'all_generated_inputs_used': True, 'real_usage_counts': real,
            'generated_usage_counts': fake, 'subsets': choices}


def distribution_metrics(output, items, features, cfg, lib, runtime):
    torch, np = lib['torch'], lib['np']
    records, sampling = [], {}
    for region in ('full_image', 'roi'):
        for cohort in ('overall', *CLASSES):
            real = [r for r in items if r['role'] == 'real' and r['region'] == region and (cohort == 'overall' or cohort in r['classes'])]
            real_features = torch.stack([features[r['key']] for r in real]).double().cpu()
            print(f'[FID/KID] {region}/{cohort}，固定真实输入 {len(real)}。', flush=True)
            real_stats = lib['fid_stats'](real_features)
            for bank in BANKS:
                for method in METHODS:
                    generated = [r for r in items if r['role'] == method and r['region'] == region and r['bank'] == bank
                                 and (cohort == 'overall' or cohort in r['classes'])]
                    require(len(generated) == (80 if cohort == 'overall' else 40), 'BANKWISE_RESULTS_INCOMPLETE', (region, cohort, bank, method))
                    fake_features = torch.stack([features[r['key']] for r in generated]).double().cpu()
                    options = kid_options(cfg, runtime['package_versions']['torch-fidelity'], len(real), len(generated))
                    key = f'{region}/{cohort}/{bank}'
                    coverage = subset_coverage(np, len(real), len(generated), options)
                    if key in sampling:
                        require(sampling[key]['coverage'] == coverage, 'KID_PAIRING_INVALID', key)
                    else:
                        sampling[key] = {'real_keys': [r['key'] for r in real],
                            'generated_slot_ids': [r['slot_id'] for r in generated],
                            'same_indices_for_Raw_and_MSDF': True, 'options': options, 'coverage': coverage}
                    print(f'[计算] {method} {bank} {region}/{cohort}: FID + KID，{len(generated)} 张生成图。', flush=True)
                    with heartbeat(f'{method} {bank} {region}/{cohort} FID/KID'):
                        # Delegate both formulas, including numeric handling, to the installed frozen library.
                        kid = lib['kid'](real_features, fake_features, **options)
                        fid = lib['fid'](real_stats, lib['fid_stats'](fake_features), verbose=False)
                    values = [fid['frechet_inception_distance'], kid['kernel_inception_distance_mean'], kid['kernel_inception_distance_std']]
                    require(all(math.isfinite(v) for v in values), 'NONFINITE_METRIC', (method, region, cohort, bank))
                    row = {'method': method, 'bank': bank, 'region': region, 'cohort': cohort,
                        'real_count': len(real), 'generated_count': len(generated), 'FID': float(values[0]),
                        'KID': float(values[1]), 'KID_subset_std': float(values[2]), 'KID_subset_std_ddof': 0,
                        'KID_subsets': options['kid_subsets'], 'KID_subset_size': options['kid_subset_size'],
                        'KID_scale': 1.0, 'all_inputs_used': True}
                    records.append(row)
                    save(output / 'distribution_metrics_raw.json', {'status': 'RUNNING', 'records': records})
                    print(f'[完成] FID={values[0]:.6g} KID={values[1]:.6g} (subset SD={values[2]:.6g})', flush=True)
    require(len(records) == 36, 'BANKWISE_RESULTS_INCOMPLETE', '36 method/bank/region/cohort results')
    save(output / 'distribution_metrics_raw.json', {'status': 'COMPLETE', 'records': records})
    save(output / 'kid_subset_sampling_audit.json', sampling)
    return records


def lpips_metrics(output, pairs, lib, runtime, cfg):
    torch, np, Image = lib['torch'], lib['np'], lib['Image']
    old_hub = torch.hub.get_dir()
    # Official LPIPS internally asks torchvision for pretrained AlexNet. Point it
    # to the already verified cache, never use random trunk weights or download.
    alex = Path(runtime['weights']['alexnet']['path'])
    torch.hub.set_dir(str(alex.parent.parent))
    try:
        model = lib['lpips'].LPIPS(net='alex', version='0.1', lpips=True, pretrained=True,
            pnet_rand=False, model_path=runtime['weights']['lpips_linear']['path'], eval_mode=True, verbose=False)
    finally:
        torch.hub.set_dir(old_hub)
    model = model.eval().requires_grad_(False).to(runtime['device'])
    by_slot = {}
    for r in pairs['records']:
        by_slot.setdefault(r['slot_id'], {})[r['bank']] = r
    rows = []
    with torch.inference_mode(), heartbeat('LPIPS AlexNet，固定 slot 的 A-B/A-C/B-C 多样性'):
        for method, prefix in zip(METHODS, ('raw', 'msdf')):
            for index, (sid, banks) in enumerate(sorted(by_slot.items())):
                require(set(banks) == set(BANKS), 'LPIPS_SLOT_COVERAGE_INVALID', sid)
                tensors = {}
                for bank, r in banks.items():
                    image = open_rgb({'path': r[prefix + '_image_path'], 'canvas_wh': r['canvas_wh'], 'crop': r['roi_xyxy']}, Image)
                    image = image.resize(tuple(cfg['LPIPS']['resize_wh']), Image.Resampling.BILINEAR)
                    pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
                    tensors[bank] = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(runtime['device'])
                values = {f'LPIPS_{a}_{b}': float(model(tensors[a], tensors[b], normalize=False).item()) for a, b in itertools.combinations(BANKS, 2)}
                require(all(math.isfinite(v) for v in values.values()), 'NONFINITE_LPIPS', sid)
                rows.append({'method': method, 'slot_id': sid, 'class': banks['A']['class'], **values,
                    'LPIPS_slot': sum(values.values()) / 3})
                if (index + 1) % 20 == 0:
                    print(f'[LPIPS] {method}: {index+1}/80 slots。', flush=True)
    del model
    if runtime['device'].startswith('cuda:'):
        torch.cuda.empty_cache()
    header = ['Method', 'slot_id', 'class', 'LPIPS_A_B', 'LPIPS_A_C', 'LPIPS_B_C', 'LPIPS_slot']
    csv_write(output / 'lpips_per_slot.csv', header,
        [[r['method'], r['slot_id'], r['class'], r['LPIPS_A_B'], r['LPIPS_A_C'], r['LPIPS_B_C'], r['LPIPS_slot']] for r in rows])
    return rows


def plot_metrics(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, ('FID', 'KID', 'LPIPS')):
        values = [summary['methods'][m]['LPIPS']['overall'] if metric == 'LPIPS'
                  else summary['methods'][m]['roi']['overall'][metric] for m in METHODS]
        ax.bar(METHODS, [x['mean'] for x in values], yerr=[x['sd'] for x in values],
            color=['#4477AA', '#EE8844'], capsize=4, width=.58)
        ax.set_title('LPIPS appearance diversity' if metric == 'LPIPS' else metric + ' (lower is better)', fontsize=11)
        ax.set_ylabel('distance (unscaled)')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('Defect-region metrics: frozen Raw SD2 vs MSDF', fontsize=13)
    fig.text(.5, -.02, 'Error bars: SD across 3 banks for FID/KID; across 80 slot means for LPIPS. LPIPS is not realism.', ha='center', fontsize=8)
    fig.savefig(output / 'generation_quality_metrics.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def publish_results(output, records, lpips_rows, summary, reference, runtime):
    for region, name in (('full_image', 'generation_quality_full_image.json'), ('roi', 'generation_quality_roi.json')):
        save(output / name, {'status': 'COMPLETE', 'region': region,
            'methods': {m: summary['methods'][m][region] for m in METHODS},
            'bankwise': [r for r in records if r['region'] == region],
            'reference_image_count': reference['image_count'], 'reference_roi_count': reference['instance_count']})
    save(output / 'fid_bankwise.json', {'status': 'COMPLETE', 'scale': 1.0,
        'records': [{k: r[k] for k in ('method', 'bank', 'region', 'cohort', 'real_count', 'generated_count', 'FID')} for r in records]})
    save(output / 'kid_bankwise.json', {'status': 'COMPLETE', 'scale': 1.0, 'negative_unbiased_estimates_not_clipped': True,
        'records': [{k: v for k, v in r.items() if k != 'FID'} for r in records]})
    save(output / 'lpips_summary.json', {'status': 'COMPLETE', 'interpretation': 'appearance diversity only; not realism',
        'slot_pair_count_per_method': 240, 'slot_count_per_method': 80,
        'methods': {m: summary['methods'][m]['LPIPS'] for m in METHODS}, 'SD_ddof': 1})
    classes = {c: {m: {'full_image': summary['methods'][m]['full_image'][c], 'roi': summary['methods'][m]['roi'][c],
        'LPIPS': summary['methods'][m]['LPIPS'][c]} for m in METHODS} for c in CLASSES}
    save(output / 'generation_quality_classwise.json', {'status': 'COMPLETE', 'classes': classes})
    save(output / 'generation_quality_summary.json', summary)
    paper_tables(output, summary)
    plot_metrics(output, summary)
    raw, msdf = summary['methods']['Raw SD2'], summary['methods']['MSDF']
    status = {**base_status(summary['status']), 'full_experiment_complete': True, 'metrics_computed': True,
        'full_FID_raw': raw['full_image']['overall']['FID']['mean'], 'full_FID_msdf': msdf['full_image']['overall']['FID']['mean'],
        'ROI_FID_raw': raw['roi']['overall']['FID']['mean'], 'ROI_FID_msdf': msdf['roi']['overall']['FID']['mean'],
        'ROI_KID_raw_mean': raw['roi']['overall']['KID']['mean'], 'ROI_KID_raw_sd': raw['roi']['overall']['KID']['sd'],
        'ROI_KID_msdf_mean': msdf['roi']['overall']['KID']['mean'], 'ROI_KID_msdf_sd': msdf['roi']['overall']['KID']['sd'],
        'LPIPS_raw': raw['LPIPS']['overall']['mean'], 'LPIPS_msdf': msdf['LPIPS']['overall']['mean'],
        'Flash_metrics': classes['flash'], 'Black_metrics': classes['black'], 'bankwise_metrics': records,
        'Inception_input_count': 1068, 'LPIPS_pair_count': 480, 'plot_status': 'COMPLETE',
        'reference_historically_observed': True, 'independent_untouched_test': False,
        'package_install_or_upgrade_count': 0, 'GitHub_submission': 'NOT_PERFORMED_BY_EVALUATOR'}
    lines = ['# Experiment1 — MSDF visual quality baseline comparison', '', 'Status: `' + summary['status'] + '`.', '',
        'Only frozen Raw SD2 and MSDF generation outputs were evaluated. No generator/detector training, sampling, selection, BootstrapGuard or DWBG ran.', '',
        '## Defect-region primary table', '', '| Method | FID (bank mean ± SD) | KID (bank mean ± SD, unscaled) | LPIPS (slot mean ± SD) |', '| --- | ---: | ---: | ---: |']
    def display(v):
        return f"{v['mean']:.6g} ± {v['sd']:.6g}"
    for m in METHODS:
        s = summary['methods'][m]
        lines.append(f"| {m} | {display(s['roi']['overall']['FID'])} | {display(s['roi']['overall']['KID'])} | {display(s['LPIPS']['overall'])} |")
    lines += ['', 'Primary analysis: defect-region KID. FID is auxiliary because reference samples are small and FID is sample-size biased. KID negative unbiased estimates are retained, not clipped.', '',
        'Crop: original normalized bbox expanded by25% of its width/height on each side, floor/ceil integer bounds and image-boundary clipping. The same rule is used for all methods and real reference boxes. No minimum-size crop enlargement or manual adjustment.', '',
        f"Reference: {reference['image_count']} unique V2 final_eval images, {reference['instance_count']} original boxes; class boxes {reference['instance_class_counts']}. This frozen held-out visual reference set was historically observed and is not an untouched test set.", '',
        'Full-image class-wise references include unique images containing the class; a multi-class image can enter both class subsets. ROI references include every original annotation. No class balancing or image deletion was introduced. Counts are saved in each bankwise record.', '',
        'FID/KID are computed separately for A/B/C (80 images per bank,40 per class). SD across three banks uses ddof1 and is not detector-seed uncertainty or proof of statistical significance. KID uses100 fixed-seed subsets of min(50,n_real,n_generated); full subset coverage is audited. Within-bank subset SD (ddof0) is not the paper bank SD.', '',
        'LPIPS AlexNet v0.1 compares A-B/A-C/B-C defect crops within the same slot. Each slot is the mean of three distances, then80 slot means (40 per class) are summarized with sample SD. Inputs are bilinear-resized256x256 RGB and normalized to[-1,1]. LPIPS measures appearance diversity, not realism or a guarantee of higher quality.', '',
        '## Class-wise ROI directions', '']
    for c in CLASSES:
        d = summary['comparisons'][c]
        lines.append(f"- {c}: {d['status']}; MSDF−Raw FID={d['ROI_FID_msdf_minus_raw']:.6g}, KID={d['ROI_KID_msdf_minus_raw']:.6g}, LPIPS={d['LPIPS_msdf_minus_raw']:.6g}.")
    diversity = summary['comparisons']['overall']['LPIPS_msdf_minus_raw']
    lines += ['', '## Interpretation', '',
        'The descriptive status uses only directions of Overall ROI bank-mean FID/KID, not a newly weighted quality score. Class-wise outcomes above must accompany any overall statement.',
        'Measured appearance diversity ' + ('decreases' if diversity < 0 else 'increases' if diversity > 0 else 'is unchanged') + f' (LPIPS MSDF−Raw={diversity:.6g}). No arbitrary threshold is used to claim preservation or significance.', '',
        'Metric implementations, actual package versions, source-file hashes and pretrained-weight hashes are frozen before metric forwards in metric_runtime.json. The standard library implementations perform all FID/KID calculations; no alternative distance or feature network was implemented.', '',
        'No downstream mAP record was read to explain these results. No image is ranked, filtered or rerolled. No subsequent experiment is automatically authorized.', '',
        '## Primary implementation sources', '', *['- ' + url for url in load(output / 'metric_protocol.json')['primary_sources']], '']
    (output / 'EXPERIMENT1_REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    save(output / 'experiment1_status.json', status)
    files = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()
             and p.name != 'experiment1_artifact_manifest.json' and p.suffix in ('.json', '.md', '.csv', '.png')}
    save(output / 'experiment1_artifact_manifest.json', {'status': 'COMPLETE', 'sha256': files,
        'feature_cache_bound_separately': True, 'no_old_files_modified': True})


def evaluate(output, cfg, pairs, reference, dep, device, config=CONFIG):
    status = load(output / 'experiment1_status.json')
    if status.get('full_experiment_complete'):
        for name, digest in load(output / 'experiment1_artifact_manifest.json')['sha256'].items():
            checked_file(output / name, digest)
        print('[已完成] 现有结果完整，未重新计算或改动。', flush=True)
        return
    require(dep['status'] == 'PASS', 'DEPENDENCY_REQUIRED', dep['errors'])
    with offline_metrics():
        try:
            lib, runtime = runtime_modules(cfg, dep, device)
        except (ImportError, ModuleNotFoundError, OSError) as error:
            failed_dependency = {**dep, 'status': 'DEPENDENCY_REQUIRED',
                'errors': [*dep['errors'], {'reason': 'METRIC_PACKAGE_IMPORT_FAILED', 'detail': str(error)}]}
            save(output / 'dependency_report.json', failed_dependency)
            raise Stop('DEPENDENCY_REQUIRED', '指标库导入失败，未升级环境：' + str(error)) from error
        freeze_json(output / 'metric_runtime.json', runtime)
        save(output / 'experiment1_status.json', {**base_status('RUNNING'), 'metric_models_only': True})
        (output / 'EXPERIMENT1_REPORT.md').write_text(
            '# Experiment1 — metric evaluation running\n\nStatus: `RUNNING`. No final visual-quality conclusion yet.\n\n'
            + 'All frozen inputs are retained. Partial numerical records, when available, are in distribution_metrics_raw.json. '
            + 'No training, sampling, selection or detector evaluation is performed.\n', encoding='utf-8', newline='\n')
        items = image_items(pairs, reference)
        features = extract_features(output, items, lib, runtime)
        distributions = distribution_metrics(output, items, features, cfg, lib, runtime)
        lpips_rows = lpips_metrics(output, pairs, lib, runtime, cfg)
        summary = summarize(distributions, lpips_rows)
        # Recheck all inputs and environment before sealing tables/report.
        verify_prepared(output, config)
        after = dependencies(cfg, runtime['weights']['inception']['path'], runtime['weights']['alexnet']['path'])
        require(after['package_versions'] == dep['package_versions'] and after['weights'] == dep['weights'],
                'METRIC_RUNTIME_CHANGED', '包/权重在本次评价期间发生变化。')
        publish_results(output, distributions, lpips_rows, summary, reference, runtime)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'evaluate', 'all'), default='all')
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--device', default='0', help='visible CUDA index, cuda:N, or explicit cpu; no automatic fallback')
    p.add_argument('--inception_weights', type=Path)
    p.add_argument('--alexnet_weights', type=Path, help='official .../checkpoints/alexnet-owt-7be5be79.pth')
    a = p.parse_args()
    output = a.output.resolve()
    device = 'cuda:' + a.device if a.device.isdecimal() else a.device
    print(f'[启动] Experiment1 视觉指标评价；模式={a.mode}；PID={os.getpid()}。不会训练或生成。', flush=True)
    try:
        require(output == OUTPUT or output.is_relative_to(OUTPUT), 'OUTPUT_PATH_INVALID', '只写入 Experiment1 专用新目录。')
        if (output / 'experiment1_status.json').is_file() and load(output / 'experiment1_status.json').get('full_experiment_complete'):
            verify_prepared(output, a.config)
            for name, digest in load(output / 'experiment1_artifact_manifest.json')['sha256'].items():
                checked_file(output / name, digest)
            print('[已退出] 已有完整结果且哈希一致；不覆盖、不重新评价。退出码=0。', flush=True)
            return
        cfg = load(a.config)
        dep = dependencies(cfg, a.inception_weights, a.alexnet_weights) if a.mode != 'prepare' else None
        if dep is not None and dep['errors']:
            print('[依赖] 发现缺少的指标包/权重；不会安装或升级。先完成冻结配对清单。', flush=True)
        if a.mode in ('prepare', 'all'):
            pairs, reference = prepare(output, a.config)
        else:
            cfg = verify_prepared(output, a.config)
            pairs = load(output / 'visual_comparison_manifest.json')
            reference = load(output / 'real_visual_reference_manifest.json')
        if a.mode == 'prepare':
            pending_outputs(output, 'PREPARED_METRICS_NOT_RUN', 'Only frozen input metadata prepared; neural metric evaluation not run.')
            print('[已退出] 配对清单与25% crop 协议已冻结；尚未计算 FID/KID/LPIPS。退出码=0。', flush=True)
            return
        save(output / 'dependency_report.json', dep)
        if dep['status'] != 'PASS':
            pending_outputs(output, 'DEPENDENCY_REQUIRED', dep['errors'], dep)
            print(json.dumps(dep, ensure_ascii=False, indent=2), flush=True)
            print('[已退出] DEPENDENCY_REQUIRED；指标未计算，退出码=2；没有训练、生成或升级环境。', flush=True)
            raise SystemExit(2)
        evaluate(output, cfg, pairs, reference, dep, device, a.config)
        print('[已退出] Experiment1 完成；结果=' + str(output) + '；退出码=0。未启动后续实验。', flush=True)
    except Exception as error:
        reason = getattr(error, 'status', 'EXPERIMENT1_EVALUATION_FAILED')
        detail = getattr(error, 'detail', str(error))
        print('[已退出] ' + reason + '：' + str(detail), flush=True)
        if (output / 'preparation_binding.json').exists():
            state = load(output / 'experiment1_status.json')
            if not state.get('full_experiment_complete'):
                state.update(status=reason, detail=str(detail), full_experiment_complete=False)
                save(output / 'experiment1_status.json', state)
                save(output / 'generation_quality_summary.json', {'status': reason, 'detail': str(detail),
                    'full_experiment_complete': False, 'visual_quality_conclusion': None, **ZERO})
                (output / 'EXPERIMENT1_REPORT.md').write_text(
                    '# Experiment1 — incomplete evaluation\n\nStatus: `' + reason + '`.\n\n'
                    + str(detail) + '\n\nNo final visual-quality conclusion is available. '
                    + 'Inputs were not filtered or regenerated. Any partial metric records are retained for audit. '
                    + 'Do not change the crop, feature extractor or bank selection to obtain a positive result.\n',
                    encoding='utf-8', newline='\n')
        raise SystemExit(2 if reason == 'DEPENDENCY_REQUIRED' else 1)


if __name__ == '__main__':
    main()
