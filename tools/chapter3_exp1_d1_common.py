"""D1 metadata, paired descriptive statistics and fixed image diagnostics. No ML imports."""
import csv
import hashlib
import importlib.metadata as md
import json
import math
from pathlib import Path
import random
import statistics

import chapter3_exp1_quality_common as c

CONFIG = c.REPO / 'configs/plastic_bomo_chapter3_exp1_d1.json'
OUTPUT = c.REPO / 'results_for_gpt/plastic_bomo_chapter3_exp1_d1_black_diagnosis'
ZERO = {**c.ZERO, 'MSDF_change_count': 0}


def digest(x):
    return hashlib.sha256(c.canonical(x).encode()).hexdigest()


def write_once(path, value):
    if Path(path).exists():
        c.require(c.load(path) == value, 'D1_FROZEN_BINDING_CHANGED', str(path))
    else:
        c.save(path, value)


def implementation():
    return {c.portable(p): c.sha(p) for p in (CONFIG, Path(__file__), Path(__file__).with_name('diagnose_plastic_bomo_chapter3_exp1_d1.py'))}


def source(cfg):
    root = c.resolve(cfg['source_root'])
    c.checked_file(root / 'experiment1_artifact_manifest.json', cfg['source_artifact_manifest_sha256'])
    for name, sha in c.load(root / 'experiment1_artifact_manifest.json')['sha256'].items():
        c.checked_file(c.inside(root, name), sha)
    c.verify_prepared(root)
    c.checked_file(root / 'inception_features.npz', cfg['source_feature_cache_sha256'])
    c.require(c.load(root / 'experiment1_status.json')['full_experiment_complete'], 'D1_SOURCE_INCOMPLETE', str(root))
    return root, c.load(root / 'visual_comparison_manifest.json'), c.load(root / 'real_visual_reference_manifest.json')


def slots_and_normals(root, pairs, normal_root):
    old = c.load(root / 'metric_protocol.json')
    slots = {s['slot_id']: s for s in c.load(c.resolve(old['p0_root']) / 'stage20b_generation_slots.json')['slots']}
    assets = {}
    for r in pairs['records']:
        s = slots[r['slot_id']]
        c.require(s['bbox'] == r['bbox'] and s['normal_sha256'] == r['conditioning_normal_sha256'], 'D1_SLOT_MISMATCH', r['slot_id'])
        name = str(s['normal_path']).replace('\\', '/').rsplit('/', 1)[-1]
        path = c.checked_file(Path(normal_root) / name, s['normal_sha256'])
        assets[r['slot_id']] = {'slot_id': r['slot_id'], 'class': r['class'], 'normal_filename': name,
            'normal_id': s['conditioning_normal_id'], 'normal_sha256': s['normal_sha256'], 'geometry': s['geometry']}
    c.require(len(assets) == 80, 'D1_NORMAL_POPULATION_INVALID', len(assets))
    return assets


def selected_slots(pairs, seed=2026):
    rng = random.Random(seed)
    by_id = {r['slot_id']: r for r in pairs['records'] if r['bank'] == 'A'}
    out = []
    for cls in c.CLASSES:
        choices = rng.sample(sorted(sid for sid, r in by_id.items() if r['class'] == cls), 12)
        for i, sid in enumerate(choices):
            bank = c.BANKS[i // 4]
            out.append(next(r for r in pairs['records'] if r['slot_id'] == sid and r['bank'] == bank))
    return {'seed': seed, 'selection_uses_image_pixels_or_scores': False, 'records': out,
            'direct_heldout_real_reference_role': False}


def resampling_plan(np, real_counts, rounds=1000, seed=2026):
    banks = np.resize(np.asarray(c.BANKS), rounds).copy()
    np.random.default_rng(np.random.SeedSequence([seed, 0])).shuffle(banks)
    gen = np.random.default_rng(np.random.SeedSequence([seed, 1])).integers(0, 40, (rounds, 27))
    real = {cls: np.random.default_rng(np.random.SeedSequence([seed, 2, k])).integers(0, real_counts[cls], (rounds, 27))
            for k, cls in enumerate(c.CLASSES)}
    return [{'round': i, 'bank': str(banks[i]), 'generated_indices': gen[i].tolist(),
             'real_indices': {cls: real[cls][i].tolist() for cls in c.CLASSES}} for i in range(rounds)]


def summary(np, values):
    values = list(values)
    if not values or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        return {'n': len(values), 'mean': None, 'median': None, 'sd': None, 'IQR': None,
                'unavailable_reason': 'empty/nonfinite values retained; no exclusion or artificial cap'}
    q = np.quantile(values, [.25, .5, .75], method='linear')
    return {'n': len(values), 'mean': statistics.mean(values), 'median': float(q[1]),
            'sd': statistics.stdev(values) if len(values) > 1 else None, 'IQR': float(q[2] - q[0]), 'sd_ddof': 1}


def interval(np, values):
    a = np.asarray(values, dtype=float)
    c.require(len(a) > 1 and bool(np.isfinite(a).all()), 'D1_NONFINITE_BOOTSTRAP', 'do not drop or reroll rounds')
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'CI95': np.quantile(a, [.025, .975], method='linear').tolist(), 'P_gt_0': float((a > 0).mean()), 'n_rounds': len(a)}


def paired(np, raw, msdf, cohort, cfg):
    # Unit is an entire slot, with A/B/C already averaged within each method.
    r, m = np.asarray(raw, dtype=float), np.asarray(msdf, dtype=float)
    if len(r) != len(m) or not len(r) or not (np.isfinite(r).all() and np.isfinite(m).all()):
        return {'Raw SD2': summary(np, list(raw)), 'MSDF': summary(np, list(msdf)), 'paired_delta': None,
                'unavailable_reason': 'nonfinite/unequal paired slot values; no samples dropped'}
    k = ('overall', *c.CLASSES).index(cohort)
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 10, k]))
    idx = rng.integers(0, len(r), (cfg['rounds'], len(r)))
    rr, mm = r[idx].mean(1), m[idx].mean(1)
    return {'Raw SD2': {**summary(np, r.tolist()), 'mean_CI95': np.quantile(rr, [.025, .975]).tolist()},
            'MSDF': {**summary(np, m.tolist()), 'mean_CI95': np.quantile(mm, [.025, .975]).tolist()},
            'paired_delta': {**summary(np, (m-r).tolist()), 'mean_CI95': np.quantile(mm-rr, [.025, .975]).tolist()},
            'unit': 'unique matched slot, mean across A/B/C before resampling', 'rounds': cfg['rounds']}


def by_slot_pair(np, rows, metric, cls, cfg):
    ids = sorted({x['slot_id'] for x in rows if cls == 'overall' or x['class'] == cls})
    vectors = []
    for method in c.METHODS:
        vals = []
        for sid in ids:
            rr = [r for r in rows if r['method'] == method and r['slot_id'] == sid]
            c.require(len(rr) == 3 and {r['bank'] for r in rr} == set(c.BANKS), 'D1_PAIRED_POPULATION_INVALID', sid)
            v = [r[metric] for r in rr]
            vals.append(statistics.mean(v) if all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) else float('inf'))
        vectors.append(vals)
    return paired(np, *vectors, cls, cfg), vectors


def appearance_values(np, rgb):
    a = np.asarray(rgb, dtype=np.float64)
    gray = a @ np.asarray([.299, .587, .114])
    padded = np.pad(gray, 1, mode='reflect')
    gx = (-padded[:-2, :-2] + padded[:-2, 2:] - 2*padded[1:-1, :-2] + 2*padded[1:-1, 2:]
          - padded[2:, :-2] + padded[2:, 2:]) / 8
    gy = (-padded[:-2, :-2] - 2*padded[:-2, 1:-1] - padded[:-2, 2:]
          + padded[2:, :-2] + 2*padded[2:, 1:-1] + padded[2:, 2:]) / 8
    out = {'gray_mean': float(gray.mean()), 'gray_std': float(gray.std()),
           'RMS_contrast': float(gray.std()/255), 'mean_sobel_magnitude': float(np.hypot(gx, gy).mean())}
    for i, ch in enumerate('RGB'):
        out[ch+'_mean'], out[ch+'_std'] = float(a[:,:,i].mean()), float(a[:,:,i].std())
    return out


def background_masks(np, geometry, radius=5):
    l, t, r, b = geometry['paste_valid_crop_region_xyxy']
    x, y, _, _ = geometry['fixed_crop_xyxy']
    a, d, z, w = geometry['rasterized_mask_xyxy_exclusive']
    # B is the ORIGINAL generator rasterization translated to final coordinates.
    a, d, z, w = max(l, a+x)-l, max(t, d+y)-t, min(r, z+x)-l, min(b, w+y)-t
    mask = np.ones((b-t, r-l), dtype=bool)
    if z > a and w > d:
        mask[d:w, a:z] = False
    supported = np.zeros_like(mask)
    if min(mask.shape) >= 2*radius+1:
        windows = np.lib.stride_tricks.sliding_window_view(mask, (2*radius+1, 2*radius+1))
        supported[radius:mask.shape[0]-radius, radius:mask.shape[1]-radius] = windows.all(axis=(-1,-2))
    c.require(bool(mask.any()) and bool(supported.any()), 'D1_BACKGROUND_SUPPORT_EMPTY', geometry)
    return mask, supported


def background_values(np, normal, generated, geometry, metrics):
    mask, supported = background_masks(np, geometry)
    n, g = np.asarray(normal), np.asarray(generated)
    c.require(n.shape == g.shape and n.shape[:2] == mask.shape, 'D1_BACKGROUND_SHAPE_INVALID', n.shape)
    mse = float(metrics.mean_squared_error(n[mask], g[mask]))
    psnr = float(metrics.peak_signal_noise_ratio(n[mask], g[mask], data_range=255)) if mse else 'Infinity'
    _, smap = metrics.structural_similarity(n, g, channel_axis=2, data_range=255, win_size=11,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    smap = smap.mean(axis=2) if smap.ndim == 3 else smap
    return {'MSE': mse, 'PSNR': psnr, 'SSIM': float(smap[supported].mean()),
            'P_minus_B_pixels': int(mask.sum()), 'SSIM_supported_centres': int(supported.sum())}


def conclusion(boot, appearance, nearest, background):
    def above(ci): return ci is not None and ci[0] > 0
    def secondary(obj, cls, metric, sign=1):
        d = obj[cls][metric].get('paired_delta')
        ci = d.get('mean_CI95') if d else None
        return bool(ci and (ci[0] > 0 if sign == 1 else ci[1] < 0))
    evidence = {}
    for cls in c.CLASSES:
        robust = all(above(boot[cls][k]['CI95']) for k in ('delta_FID', 'delta_KID'))
        nr = all(secondary(nearest, cls, k) for k in ('minimum_LPIPS', 'top3_mean_LPIPS'))
        app = all(above(appearance['paired_deviation_bootstrap'][cls][k]['CI95']) for k in ('gray_mean', 'RMS_contrast'))
        bg = all(secondary(background, cls, k, sign) for k, sign in (('PSNR', -1), ('SSIM', -1), ('MSE', 1)))
        evidence[cls] = {'FID_KID_gap_robust': robust, 'nearest_reference_farther': nr,
                         'intensity_and_contrast_deviation_larger': app, 'local_appearance_association': nr and app,
                         'background_association': bg}
    b = evidence['black']
    if not b['FID_KID_gap_robust']:
        status = 'BLACK_GAP_NOT_ROBUST_UNDER_RESAMPLING'
    elif b['local_appearance_association'] and b['background_association']:
        status = 'BLACK_GAP_ASSOCIATED_WITH_MULTIPLE_VISUAL_SHIFTS'
    elif b['local_appearance_association']:
        status = 'BLACK_GAP_ASSOCIATED_WITH_LOCAL_APPEARANCE_SHIFT'
    elif b['background_association']:
        status = 'BLACK_GAP_ASSOCIATED_WITH_BACKGROUND_SHIFT'
    else:
        status = 'BLACK_GAP_ROBUST_BUT_MECHANISM_UNRESOLVED'
    evidence['black_specific_local_pattern'] = b['local_appearance_association'] and not evidence['flash']['local_appearance_association']
    evidence['black_specific_background_pattern'] = b['background_association'] and not evidence['flash']['background_association']
    return status, evidence
