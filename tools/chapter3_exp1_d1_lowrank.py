"""Authorized D1 FID computation amendment. No feature extraction or protocol reroll."""
from contextlib import closing, contextmanager
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import time

import chapter3_exp1_quality_common as c
import chapter3_exp1_d1_common as d
import evaluate_plastic_bomo_chapter3_exp1 as e

LEGACY_IMPLEMENTATION = {
    'configs/plastic_bomo_chapter3_exp1_d1.json': '72d7fe17c5b92dccd942286aa1f2f42cdd8f4dd44673a41d07e7061324469108',
    'tools/chapter3_exp1_d1_common.py': '762ffd7c3dd0d193ccbae89846802bde699e34140a77f0b34fa1a0e1c4d60e4e',
    'tools/diagnose_plastic_bomo_chapter3_exp1_d1.py': 'f464011920698d5bcd2a26f7b986ffa409350fa90efc1d6d0e1bc337d007648e',
}
ALGORITHM = 'float64_unbiased_covariance_exact_cross_gram_nuclear_norm_v1'
ATOL = 1e-5
RTOL = 1e-6
STRICT_IMPLEMENTATION = {
    **LEGACY_IMPLEMENTATION,
    'tools/diagnose_plastic_bomo_chapter3_exp1_d1.py': 'fd7a82643ca97059b55ea50b71e91a9b57915b44973a020bc7eb66dd146d9875',
    'tools/chapter3_exp1_d1_lowrank.py': '07d5e20425144455fd3a34d14a2835943f40d2facea64bb26c2f790b59363a97',
}
TOLERANCE_RULE = 'abs(lowrank_FID-standard_FID) <= 1e-5 + 1e-6*abs(standard_FID)'
TOLERANCE_ARCHIVE = 'strict_absolute_tolerance_failure'
VALIDATION_ROUNDS = 10
ARCHIVE = 'legacy_slow_validation'


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    c.save(tmp, value)
    tmp.replace(path)


def implementation():
    return {**d.implementation(), c.portable(__file__): c.sha(__file__)}


def verify_protocol(out, cfg):
    """Keep the scientific protocol byte-for-byte; authorize only the known code amendment."""
    protocol = c.load(out / 'diagnostic_protocol.json')
    recorded = protocol.pop('implementation_sha256')
    c.require(protocol == cfg, 'D1_SCIENTIFIC_PROTOCOL_CHANGED', 'No seed, sample, metric definition or conclusion change is authorized.')
    old_accelerated = {k:v for k,v in STRICT_IMPLEMENTATION.items() if k != 'tools/chapter3_exp1_d1_lowrank.py'}
    c.require(recorded in (LEGACY_IMPLEMENTATION, old_accelerated, d.implementation()),
              'D1_UNRECOGNIZED_LEGACY_IMPLEMENTATION', recorded)
    for name in ('configs/plastic_bomo_chapter3_exp1_d1.json', 'tools/chapter3_exp1_d1_common.py'):
        c.checked_file(c.REPO / name, LEGACY_IMPLEMENTATION[name])


def amendment_record(out, cfg):
    return {
        'authorization': 'User requested exact computational acceleration on 2026-10-17.',
        'supersedes': 'Only the old prohibition on a low-rank FID computational form; no scientific definition or other protocol field changes.',
        'original_diagnostic_protocol_sha256': c.sha(out / 'diagnostic_protocol.json'),
        'original_prepared_binding_sha256': c.sha(out / 'prepared_binding.json'),
        'resampling_index_plan_sha256': c.sha(out / 'resampling_index_plan.json'),
        'feature_cache_sha256': cfg['source_feature_cache_sha256'],
        'implementation_sha256': implementation(),
        'FID_algorithm': ALGORITHM,
        'FID_definition': '||mu_x-mu_y||^2 + ||Xc||F^2/(nx-1) + ||Yc||F^2/(ny-1) - 2*sum(svd(Xc@Yc.T/sqrt((nx-1)*(ny-1))))',
        'numerical_policy': 'float64; all singular values; no rank truncation, projection, jitter, epsilon, clipping or ridge.',
        'validation_tolerance_authorization': 'User explicitly approved the combined absolute/relative tolerance after the strict-only failure on 2026-10-17.',
        'validation': {'completed_legacy_rounds': VALIDATION_ROUNDS, 'classes': list(c.CLASSES),
                       'methods': list(c.METHODS), 'absolute_tolerance': ATOL, 'relative_tolerance': RTOL,
                       'acceptance_rule': TOLERANCE_RULE,
                       'standard_reference_reproducibility_absolute_tolerance': ATOL,
                       'fresh_standard_2048_FID_recalculation_required': True,
                       'failure_policy': 'STOP on any failed item; no automatic tolerance adjustment.'},
        'formal_policy': 'Restart every class at round 0 in a separate lowrank-only database. Legacy rows are validation references only.',
        'KID_changed': False, 'Inception_feature_forward_count': 0, **d.ZERO,
    }


def previous_strict_amendment(current):
    """Exact known predecessor, not a general code-binding bypass."""
    old = {k:v for k,v in current.items() if k != 'validation_tolerance_authorization'}
    old['implementation_sha256'] = STRICT_IMPLEMENTATION
    old['validation'] = {'completed_legacy_rounds': VALIDATION_ROUNDS, 'classes': list(c.CLASSES),
        'methods': list(c.METHODS), 'max_absolute_error': ATOL,
        'fresh_standard_2048_FID_recalculation_required': True,
        'failure_policy': 'STOP; never loosen tolerance or continue formal rounds.'}
    return old


def migrate_strict_failure(out, expected):
    """Archive the rejected strict-only attempt. Never reuse or overwrite formal rows."""
    archive = out/ARCHIVE/TOLERANCE_ARCHIVE
    journal = archive/'tolerance_migration_plan.json'
    completed = archive/'tolerance_migration_complete.json'
    for name in ('bootstrap_lowrank_progress.sqlite3', 'stage_bootstrap_complete.json', 'd1_artifact_manifest.json'):
        c.require(not (out/name).exists(), 'D1_TOLERANCE_CHANGE_AFTER_FORMAL_RESULTS_FORBIDDEN', name)
    if not journal.exists():
        c.require(c.load(out/'fid_acceleration_amendment.json') == previous_strict_amendment(expected),
                  'D1_UNRECOGNIZED_TOLERANCE_PREDECESSOR', 'Only the known strict-only implementation may migrate.')
        report = c.load(out/'fid_lowrank_equivalence_check.json')
        c.require(report.get('status') == 'FAIL' and not report.get('formal_FID_replacement_authorized'),
                  'D1_TOLERANCE_MIGRATION_REQUIRES_FAILED_VALIDATION', report.get('status'))
        c.require(report.get('max_absolute_error_allowed') == ATOL,
                  'D1_UNRECOGNIZED_PREVIOUS_TOLERANCE', report.get('max_absolute_error_allowed'))
        expected_fingerprint = d.digest({'amendment':c.sha(out/'fid_acceleration_amendment.json'),
            'runtime':c.sha(out/'diagnostic_runtime.json'), 'indices':d.digest(c.load(out/'fid_validation_index_sets.json'))})
        c.require(report['fingerprint'] == expected_fingerprint, 'D1_PREVIOUS_EQUIVALENCE_BINDING_CHANGED', report['fingerprint'])
        moves = ('fid_lowrank_equivalence_check.json', 'fid_validation_index_sets.json',
                 'bootstrap_runtime_comparison.json', 'fid_acceleration_amendment.json')
        copies = ('d1_status.json', 'D1_REPORT.md')
        archive.mkdir(parents=True, exist_ok=True)
        atomic_save(journal, {'authorization':expected['validation_tolerance_authorization'],
            'new_amendment_digest':d.digest(expected),
            'move':{n:c.sha(out/n) for n in moves if (out/n).is_file()},
            'copy':{n:c.sha(out/n) for n in copies if (out/n).is_file()},
            'original_failed_audit_retained':True, 'formal_rows_reused':0})
    plan = c.load(journal)
    c.require(plan['new_amendment_digest'] == d.digest(expected), 'D1_TOLERANCE_MIGRATION_BINDING_CHANGED', str(journal))
    allowed = {'move':{'fid_lowrank_equivalence_check.json','fid_validation_index_sets.json',
                      'bootstrap_runtime_comparison.json','fid_acceleration_amendment.json'},
               'copy':{'d1_status.json','D1_REPORT.md'}}
    for operation in ('copy','move'):
        c.require(set(plan[operation]) <= allowed[operation], 'D1_TOLERANCE_ARCHIVE_TARGET_INVALID', operation)
        for name, expected_sha in plan[operation].items():
            src, dst = c.inside(out,name), c.inside(archive,name)
            if dst.exists(): c.checked_file(dst,expected_sha); continue
            c.checked_file(src,expected_sha)
            if operation == 'move': src.replace(dst)
            else: shutil.copy2(src,dst)
            c.checked_file(dst,expected_sha)
    d.write_once(completed, {'migration_plan_sha256':c.sha(journal),
                            'original_strict_failure_preserved':True, 'scientific_protocol_changed':False})
    print('[迁移] 原绝对容差失败审计已保留在 '+str(archive)+'；新验证使用已授权的绝对＋相对容差。', flush=True)


def bind(out, cfg):
    verify_protocol(out, cfg)
    amendment = amendment_record(out,cfg)
    path = out/'fid_acceleration_amendment.json'
    journal = out/ARCHIVE/TOLERANCE_ARCHIVE/'tolerance_migration_plan.json'
    if (path.exists() and c.load(path) != amendment) or (not path.exists() and journal.exists()):
        migrate_strict_failure(out,amendment)
    d.write_once(path,amendment)


def fid_lowrank_exact(np, x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    c.require(x.ndim == y.ndim == 2 and x.shape[1] == y.shape[1] and min(len(x), len(y)) > 1,
              'D1_FID_INPUT_INVALID', (x.shape, y.shape))
    c.require(bool(np.isfinite(x).all() and np.isfinite(y).all()), 'D1_NONFINITE_FID_FEATURES', 'No filtering.')
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    mean = float(np.sum((mx - my) ** 2, dtype=np.float64))
    tx = float(np.sum(xc * xc, dtype=np.float64) / (len(x) - 1))
    ty = float(np.sum(yc * yc, dtype=np.float64) / (len(y) - 1))
    cross = xc @ yc.T / math.sqrt((len(x) - 1) * (len(y) - 1))
    singular = np.linalg.svd(cross, compute_uv=False)
    value = mean + tx + ty - 2 * float(singular.sum(dtype=np.float64))
    c.require(math.isfinite(value), 'D1_NONFINITE_LOWRANK_FID', value)
    return value


def archive_legacy(out):
    """Recoverable, allowlisted moves under the already locked D1 output only."""
    archive = out / ARCHIVE
    archive.mkdir(exist_ok=True)
    journal = archive / 'archive_plan.json'
    if not journal.exists():
        moves = ('bootstrap_progress.sqlite3', 'bootstrap_progress.sqlite3-journal',
                 'bootstrap_progress.sqlite3-wal', 'bootstrap_progress.sqlite3-shm',
                 'bootstrap_fid_kid.csv', 'bootstrap_summary.json', 'stage_bootstrap_complete.json')
        snapshots = ('diagnostic_protocol.json', 'prepared_binding.json', 'diagnostic_runtime.json',
                     'resampling_index_plan.json', 'd1_status.json', 'D1_REPORT.md')
        c.save(journal, {'move': {n: c.sha(out/n) for n in moves if (out/n).is_file()},
                         'copy': {n: c.sha(out/n) for n in snapshots if (out/n).is_file()},
                         'legacy_implementation_sha256': LEGACY_IMPLEMENTATION,
                         'purpose': 'Implementation consistency audit only; never formal bootstrap rows.'})
    plan = c.load(journal)
    for operation in ('copy', 'move'):
        for name, expected in plan[operation].items():
            src, dst = c.inside(out, name), c.inside(archive, name)
            if dst.exists():
                c.checked_file(dst, expected)
                c.require(operation != 'move' or not src.exists(), 'D1_ARCHIVE_CONFLICT', str(src))
                continue
            c.checked_file(src, expected)
            if operation == 'copy': shutil.copy2(src, dst)
            else: src.replace(dst)
            c.checked_file(dst, expected)
    d.write_once(archive / 'archive_complete.json', {'archive_plan_sha256': c.sha(journal),
                 'legacy_results_used_for_formal_bootstrap': False})
    return archive


def legacy_rows(archive, out, plan):
    path = archive / 'bootstrap_progress.sqlite3'
    c.require(path.is_file(), 'D1_COMPLETED_LEGACY_BOOTSTRAP_REQUIRED',
              'Need at least ten completed slow rounds for both classes; do not fabricate validation data.')
    fingerprint = d.digest({'plan': c.sha(out/'resampling_index_plan.json'),
                            'runtime': c.sha(out/'diagnostic_runtime.json')})
    # Preserve original SQLite bytes, even if SIGTERM left a hot rollback journal.
    # Recovery is only performed on a disposable copy, then a committed snapshot is sealed.
    readable = archive/'legacy_committed_snapshot.sqlite3'
    binding = archive/'legacy_committed_snapshot_binding.json'
    if binding.exists():
        c.checked_file(readable, c.load(binding)['sha256'])
    else:
        work = archive/'sqlite_recovery_copy'
        work.mkdir(exist_ok=True)
        for suffix in ('', '-journal', '-wal', '-shm'):
            original = path.with_name(path.name+suffix)
            if original.exists(): shutil.copy2(original, work/original.name)
        with closing(sqlite3.connect(work/path.name)) as recovered, closing(sqlite3.connect(readable)) as snapshot:
            c.require(recovered.execute('PRAGMA integrity_check').fetchall() == [('ok',)], 'D1_LEGACY_DB_INVALID', str(path))
            recovered.backup(snapshot)
        c.save(binding, {'sha256':c.sha(readable), 'original_database_sha256':c.sha(path),
                        'original_archived_bytes_modified':False, 'uncommitted_transactions_are_not_results':True})
    result = {}
    with closing(sqlite3.connect(readable.as_uri() + '?mode=ro', uri=True)) as db:
        c.require(db.execute('PRAGMA integrity_check').fetchall() == [('ok',)], 'D1_LEGACY_DB_INVALID', str(path))
        c.require(db.execute('SELECT value FROM binding').fetchall() == [(fingerprint,)],
                  'D1_LEGACY_BINDING_CHANGED', str(path))
        for key, payload, digest in db.execute('SELECT id,payload,digest FROM rounds'):
            row = json.loads(payload)
            c.require(d.digest(row) == digest, 'D1_LEGACY_ROW_CHANGED', key)
            r, cls = row['round'], row['class']
            c.require(cls in c.CLASSES and 0 <= r < len(plan) and key == f'{r}:{cls}', 'D1_LEGACY_ROW_INVALID', key)
            c.require(row['plan_sha256'] == d.digest({'class': cls, 'plan': plan[r]})
                      and row['bank'] == plan[r]['bank'] and row['real_sample_size'] == row['generated_sample_size'] == 27,
                      'D1_LEGACY_SAMPLE_CHANGED', key)
            for k in ('FID_raw', 'FID_msdf', 'KID_raw', 'KID_msdf'):
                c.require(math.isfinite(row[k]), 'D1_LEGACY_NONFINITE', (key, k))
            result[key] = row
    return result


def validation_selection(plan, rows):
    complete = [p for p in plan if all(f'{p["round"]}:{cls}' in rows for cls in c.CLASSES)]
    c.require(len(complete) >= VALIDATION_ROUNDS, 'D1_TOO_FEW_LEGACY_VALIDATION_ROUNDS', len(complete))
    chosen = []
    for bank in c.BANKS:
        candidates = [p['round'] for p in complete if p['bank'] == bank]
        c.require(bool(candidates), 'D1_LEGACY_BANK_MISSING', bank)
        chosen.append(candidates[0])
    for p in complete:
        if p['round'] not in chosen: chosen.append(p['round'])
        if len(chosen) == VALIDATION_ROUNDS: break
    return [plan[r] for r in sorted(chosen)]


def cached_populations(root, cfg, np, ref, pairs):
    path = c.checked_file(root/'inception_features.npz', cfg['source_feature_cache_sha256'])
    with np.load(path, allow_pickle=False) as z: features = z['features'].astype(np.float64, copy=True)
    c.require(features.shape == (1068, 2048) and bool(np.isfinite(features).all()),
              'D1_FEATURE_CACHE_INVALID', 'Exact original 1068x2048 cache required.')
    ledger = c.load(root/'inception_feature_cache.json')
    mapping = {key: i for i, key in enumerate(ledger['keys'])}
    c.require(len(mapping) == len(features), 'D1_FEATURE_KEYS_INVALID', len(mapping))
    items, populations, counts = e.image_items(pairs, ref), {}, {}
    for cls in c.CLASSES:
        rr = [x for x in items if x['role'] == 'real' and x['region'] == 'roi' and cls in x['classes']]
        c.require(len(rr) == ref['instance_class_counts'][cls], 'D1_REFERENCE_COUNT_CHANGED', cls)
        populations[('real', cls)] = np.stack([features[mapping[x['key']]] for x in rr])
        counts['real:' + cls] = len(rr)
        for bank in c.BANKS:
            for method in c.METHODS:
                gg = [x for x in items if x['role'] == method and x['region'] == 'roi' and x['bank'] == bank and cls in x['classes']]
                c.require(len(gg) == 40, 'D1_BOOTSTRAP_POPULATION_CHANGED', (cls, bank, method))
                populations[(method, cls, bank)] = np.stack([features[mapping[x['key']]] for x in gg])
                counts[f'{method}:{cls}:{bank}'] = len(gg)
    return populations, counts


def sample(populations, p, cls):
    return (populations[('real', cls)][p['real_indices'][cls]],
            populations[('Raw SD2', cls, p['bank'])][p['generated_indices']],
            populations[('MSDF', cls, p['bank'])][p['generated_indices']])


@contextmanager
def forbid_inception_forward(lib):
    original, extractor = e.extract_features, lib.get('Inception')
    def forbidden(*args, **kwargs):
        raise c.Stop('D1_BOOTSTRAP_IMAGE_FORWARD_FORBIDDEN', 'Bootstrap may only read cached features.')
    e.extract_features = forbidden
    lib['Inception'] = forbidden
    try: yield
    finally:
        e.extract_features = original
        if extractor is None: lib.pop('Inception', None)
        else: lib['Inception'] = extractor


def error_record(slow, fast):
    allowed = ATOL + RTOL * abs(slow)
    difference = abs(slow - fast)
    return {'absolute_difference': difference,
            'relative_difference': difference / max(abs(slow), 1e-12),
            'allowed_absolute_difference': allowed,
            'tolerance_ratio': difference / allowed,
            'within_tolerance': math.isfinite(slow) and math.isfinite(fast) and difference <= allowed}


def validate(out, cfg, lib, populations, plan, legacy, options):
    np, torch = lib['np'], lib['torch']
    chosen = validation_selection(plan, legacy)
    selected = {'selection': 'First completed round of each bank, then earliest remaining completed rounds; no metric-based choice.',
                'records': chosen, 'classes': list(c.CLASSES), 'methods': list(c.METHODS)}
    d.write_once(out/'fid_validation_index_sets.json', selected)
    previous_indices = out/ARCHIVE/TOLERANCE_ARCHIVE/'fid_validation_index_sets.json'
    if previous_indices.exists():
        c.require(c.load(previous_indices) == selected, 'D1_VALIDATION_INDEX_SETS_CHANGED', str(previous_indices))
    fingerprint = d.digest({'amendment': c.sha(out/'fid_acceleration_amendment.json'),
                            'runtime': c.sha(out/'diagnostic_runtime.json'), 'indices': d.digest(selected)})
    path = out/'fid_lowrank_equivalence_check.json'
    if path.exists():
        report = c.load(path)
        c.require(report['fingerprint'] == fingerprint, 'D1_EQUIVALENCE_BINDING_CHANGED', str(path))
        c.require(report['status'] != 'FAIL', 'D1_FID_NUMERICAL_EQUIVALENCE_FAILED', report)
    else:
        report = {'status': 'RUNNING', 'fingerprint': fingerprint,
                  'absolute_tolerance': ATOL, 'relative_tolerance': RTOL, 'acceptance_rule': TOLERANCE_RULE,
                  'feature_forward_count': 0, 'formal_FID_replacement_authorized': False,
                  'legacy_standard_comparisons': [], 'records': []}
    # Cheap first check against already computed standard FID, before paying for 40 fresh slow calls.
    if not report['legacy_standard_comparisons']:
        for p in chosen:
            for cls in c.CLASSES:
                real, raw, msdf = sample(populations, p, cls)
                for name, generated in (('raw', raw), ('msdf', msdf)):
                    fast = fid_lowrank_exact(np, real, generated)
                    old = legacy[f'{p["round"]}:{cls}']['FID_' + name]
                    report['legacy_standard_comparisons'].append({'round': p['round'], 'class': cls, 'bank': p['bank'],
                        'method': name, 'legacy_standard_FID': old, 'lowrank_exact_FID': fast, **error_record(old, fast)})
        worst = max(x['absolute_difference'] for x in report['legacy_standard_comparisons'])
        report['max_legacy_absolute_difference'] = worst
        report['max_absolute_difference'] = worst
        report['max_relative_difference'] = max(x['relative_difference'] for x in report['legacy_standard_comparisons'])
        report['legacy_slow_FID_evaluations_used_for_precheck'] = len(report['legacy_standard_comparisons'])
        report['max_tolerance_ratio'] = max(x['tolerance_ratio'] for x in report['legacy_standard_comparisons'])
        report['failed_legacy_comparisons'] = sum(not x['within_tolerance'] for x in report['legacy_standard_comparisons'])
        if report['failed_legacy_comparisons']:
            report.update(status='FAIL', reason='A standard sqrtm comparison exceeds the fixed user-authorized absolute/relative tolerance. No automatic tolerance change or formal restart.')
        atomic_save(path, report)
        c.require(report['status'] != 'FAIL', 'D1_FID_NUMERICAL_EQUIVALENCE_FAILED',
                  f'{report["failed_legacy_comparisons"]} failed comparisons; max abs error={worst:.12g}, max error/allowed={report["max_tolerance_ratio"]:.6g}; see {path}')
    completed = {(r['round'], r['class'], r['method']): r for r in report['records']}
    with e.heartbeat('D1等价性验证：仅10轮×两类×两方法复算慢版FID，逐项保存；通过后才启动快速1000轮'):
        for p in chosen:
            for cls in c.CLASSES:
                real, raw, msdf = sample(populations, p, cls)
                for name, generated in (('raw', raw), ('msdf', msdf)):
                    key = (p['round'], cls, name)
                    if key in completed: continue
                    rt = torch.from_numpy(real).double().cpu()
                    gt = torch.from_numpy(generated).double().cpu()
                    start = time.perf_counter()
                    slow = float(lib['fid'](lib['fid_stats'](rt), lib['fid_stats'](gt), verbose=False)['frechet_inception_distance'])
                    slow_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    fast = fid_lowrank_exact(np, real, generated)
                    fast_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    kid = float(lib['kid'](rt, gt, **options)['kernel_inception_distance_mean'])
                    kid_seconds = time.perf_counter() - start
                    old = legacy[f'{p["round"]}:{cls}']
                    row = {'round': p['round'], 'class': cls, 'bank': p['bank'], 'method': name,
                        'standard_2048_FID': slow, 'lowrank_exact_FID': fast, **error_record(slow, fast),
                        'legacy_standard_FID': old['FID_' + name], 'fresh_minus_legacy_absolute': abs(slow - old['FID_' + name]),
                        'KID_absolute_difference': abs(kid - old['KID_' + name]),
                        'standard_FID_seconds': slow_seconds, 'lowrank_FID_seconds': fast_seconds, 'KID_seconds': kid_seconds}
                    row['pass'] = row['within_tolerance'] and row['fresh_minus_legacy_absolute'] <= ATOL and math.isfinite(row['KID_absolute_difference']) and row['KID_absolute_difference'] <= 1e-12
                    report['records'].append(row)
                    report['max_absolute_difference'] = max(r['absolute_difference'] for r in report['records'])
                    report['max_relative_difference'] = max(r['relative_difference'] for r in report['records'])
                    report['max_tolerance_ratio'] = max(r['tolerance_ratio'] for r in report['records'])
                    if not row['pass']: report.update(status='FAIL', reason='Numerical equivalence or unchanged KID check failed; formal bootstrap is not authorized.')
                    atomic_save(path, report)
                    print(f'[FID等价性] {len(report["records"])}/40 {cls} {name} round={p["round"]+1} abs={row["absolute_difference"]:.3g} rel={row["relative_difference"]:.3g} slow={slow_seconds:.2f}s fast={fast_seconds:.4f}s', flush=True)
                    c.require(row['pass'], 'D1_FID_NUMERICAL_EQUIVALENCE_FAILED', row)
    c.require(len(report['records']) == VALIDATION_ROUNDS * 4, 'D1_EQUIVALENCE_INCOMPLETE', len(report['records']))
    report.update(status='PASS', slow_FID_evaluations_used_for_validation=len(report['records']),
                  lowrank_FID_evaluations_for_validation=2 * len(report['records']),
                  all_formal_rounds_must_be_recomputed=True, no_image_forward=True,
                  formal_FID_replacement_authorized=True)
    atomic_save(path, report)
    return report


def runtime_report(out, validation, rows, complete=False):
    counts = len(rows) * 2
    # One full protocol round comprises both classes and both methods (four FIDs and four KIDs).
    old_seconds = sum(r['standard_FID_seconds'] + r['KID_seconds'] for r in validation['records']) / VALIDATION_ROUNDS
    new_seconds = sum(r['computation_seconds'] for r in rows) / (len(rows) / 2) if rows else None
    value = {'status': 'COMPLETE' if complete else 'RUNNING',
        'round_definition': 'One bank/index round, both Flash and Black, both Raw and MSDF; includes unchanged KID.',
        'old_seconds_per_round': old_seconds, 'old_timing_source': 'Measured replay of ten completed legacy rounds using the original 2048-dimensional torch-fidelity FID, plus unchanged KID; not inferred from approximate user wall time.',
        'new_seconds_per_round': new_seconds, 'new_timing_scope': 'Feature slicing, both lowrank FIDs and both KIDs per class; excludes process startup, validation, SQLite/CSV I/O and other D1 diagnostics.',
        'speedup_ratio': old_seconds / new_seconds if new_seconds else None,
        'feature_forward_count': 0, 'Inception_image_forward_count_formal_bootstrap': 0,
        'slow_FID_evaluations_used_for_validation': validation['slow_FID_evaluations_used_for_validation'],
        'slow_FID_evaluations_in_formal_bootstrap': 0,
        'lowrank_FID_evaluations': counts, 'lowrank_FID_evaluations_for_validation': validation['lowrank_FID_evaluations_for_validation'],
        'lowrank_FID_evaluations_scope': 'Committed formal rows only; a killed, uncommitted calculation is not a result and may be retried.',
        'formal_class_round_rows': len(rows), 'formal_lowrank_only': True, 'legacy_rows_used_as_formal_results': 0,
        'resampling_index_plan_sha256': c.sha(out/'resampling_index_plan.json'),
        'inception_feature_cache_sha256': c.load(out/'fid_acceleration_amendment.json')['feature_cache_sha256'],
        'all_singular_values_used': True, 'feature_dimension': 2048, 'covariance_ddof': 1,
        'validation_included_in_new_round_timing': False}
    atomic_save(out/'bootstrap_runtime_comparison.json', value)


def bootstrap(out, root, cfg, lib, ref, pairs, stage_done, seal, csv_rows):
    if (out/'bootstrap_runtime_comparison.json').is_file() and stage_done(out, 'bootstrap'):
        c.require((out/'bootstrap_runtime_comparison.json').is_file() and
                  c.load(out/'bootstrap_runtime_comparison.json')['formal_lowrank_only'],
                  'D1_LEGACY_BOOTSTRAP_CANNOT_BE_FORMAL', 'Archive the old stage before use.')
        return
    bind(out, cfg)
    archive = archive_legacy(out)
    np, torch = lib['np'], lib['torch']
    plan = c.load(out/'resampling_index_plan.json')['records']
    c.require(len(plan) == cfg['rounds'] == 1000 and cfg['seed'] == 2026,
              'D1_SCIENTIFIC_PROTOCOL_CHANGED', 'Exactly 1000 unchanged rounds, seed2026 required.')
    legacy = legacy_rows(archive, out, plan)
    populations, counts = cached_populations(root, cfg, np, ref, pairs)
    d.write_once(out/'bootstrap_feature_cache_audit.json', {
        'source_cache_sha256': cfg['source_feature_cache_sha256'], 'cached_rows': 1068,
        'feature_dimension': 2048, 'ROI_population_counts': counts,
        'feature_forward_count': 0, 'no_feature_extractor_instantiated': True,
        'bootstrap_reads_cached_features_only': True})
    options = e.kid_options(c.load(root/'metric_protocol.json'), c.load(root/'metric_runtime.json')['package_versions']['torch-fidelity'], 27, 27)
    with forbid_inception_forward(lib):
        try:
            validation = validate(out, cfg, lib, populations, plan, legacy, options)
        except c.Stop:
            report = c.load(out/'fid_lowrank_equivalence_check.json') if (out/'fid_lowrank_equivalence_check.json').exists() else {}
            atomic_save(out/'bootstrap_runtime_comparison.json', {
                'status': 'STOPPED_BEFORE_FORMAL_BOOTSTRAP', 'old_seconds_per_round': None,
                'new_seconds_per_round': None, 'timing_unavailable_reason': 'Numerical equivalence has not passed; formal bootstrap not started.',
                'feature_forward_count': 0, 'Inception_image_forward_count_formal_bootstrap': 0,
                'slow_FID_evaluations_used_for_validation': len(report.get('records', [])),
                'legacy_slow_FID_evaluations_used_for_precheck': len(report.get('legacy_standard_comparisons', [])),
                'lowrank_FID_evaluations': 0, 'formal_lowrank_only': True,
                'legacy_rows_used_as_formal_results': 0, 'formal_class_round_rows': 0})
            raise
        fingerprint = d.digest({'amendment': c.sha(out/'fid_acceleration_amendment.json'),
            'validation': c.sha(out/'fid_lowrank_equivalence_check.json'),
            'plan': c.sha(out/'resampling_index_plan.json'), 'runtime': c.sha(out/'diagnostic_runtime.json')})
        with closing(sqlite3.connect(out/'bootstrap_lowrank_progress.sqlite3')) as db:
            db.execute('CREATE TABLE IF NOT EXISTS binding (value TEXT NOT NULL)')
            previous = db.execute('SELECT value FROM binding').fetchall()
            if previous: c.require(previous == [(fingerprint,)], 'D1_LOWRANK_CACHE_CHANGED', fingerprint)
            else: db.execute('INSERT INTO binding VALUES (?)', (fingerprint,)); db.commit()
            db.execute('CREATE TABLE IF NOT EXISTS rounds (id TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)'); db.commit()
            records, timings = [], []
            try:
                with e.heartbeat('D1快速bootstrap：1000轮，两类全部保留；仅小矩阵SVD和原KID，Inception forward=0'):
                    for p in plan:
                        for cls in c.CLASSES:
                            key, policy = f'{p["round"]}:{cls}', d.digest({'class': cls, 'plan': p})
                            saved = db.execute('SELECT payload,digest FROM rounds WHERE id=?', (key,)).fetchone()
                            if saved:
                                payload = json.loads(saved[0])
                                c.require(d.digest(payload) == saved[1] and payload['row']['plan_sha256'] == policy and payload['algorithm'] == ALGORITHM,
                                          'D1_LOWRANK_CACHE_CHANGED', key)
                            else:
                                start = time.perf_counter()
                                real, raw, msdf = sample(populations, p, cls)
                                rt = torch.from_numpy(real).double().cpu()
                                row = {'class': cls, 'round': p['round'], 'bank': p['bank'],
                                       'real_sample_size': 27, 'generated_sample_size': 27, 'plan_sha256': policy}
                                for name, generated in (('raw', raw), ('msdf', msdf)):
                                    row['FID_' + name] = fid_lowrank_exact(np, real, generated)
                                    gt = torch.from_numpy(generated).double().cpu()
                                    row['KID_' + name] = float(lib['kid'](rt, gt, **options)['kernel_inception_distance_mean'])
                                row['delta_FID'], row['delta_KID'] = row['FID_msdf'] - row['FID_raw'], row['KID_msdf'] - row['KID_raw']
                                c.require(all(math.isfinite(row[k]) for k in ('FID_raw','FID_msdf','KID_raw','KID_msdf','delta_FID','delta_KID')), 'D1_NONFINITE_BOOTSTRAP', key)
                                payload = {'row': row, 'algorithm': ALGORITHM, 'computation_seconds': time.perf_counter() - start}
                                db.execute('INSERT INTO rounds VALUES (?,?,?)', (key, c.canonical(payload), d.digest(payload))); db.commit()
                                print(f'[快速bootstrap] {cls} round={p["round"]+1}/1000 bank={p["bank"]} FID_delta={row["delta_FID"]:.6g} KID_delta={row["delta_KID"]:.6g} {payload["computation_seconds"]:.3f}s', flush=True)
                            records.append(payload['row']); timings.append({'computation_seconds': payload['computation_seconds']})
                        if (p['round'] + 1) % 10 == 0:
                            csv_rows(out/'bootstrap_fid_kid.csv', records)
                            runtime_report(out, validation, timings)
                c.require(len(records) == 2 * cfg['rounds'], 'D1_BOOTSTRAP_INCOMPLETE', len(records))
                c.checked_file(root/'inception_features.npz', cfg['source_feature_cache_sha256'])
                csv_rows(out/'bootstrap_fid_kid.csv', records)
                c.save(out/'bootstrap_summary.json', {cls: {k: d.interval(np, [r[k] for r in records if r['class'] == cls])
                       for k in ('FID_raw','FID_msdf','KID_raw','KID_msdf','delta_FID','delta_KID')} for cls in c.CLASSES})
                runtime_report(out, validation, timings, complete=True)
                seal(out, 'bootstrap', ['bootstrap_fid_kid.csv','bootstrap_summary.json',
                    'fid_lowrank_equivalence_check.json','fid_validation_index_sets.json',
                    'bootstrap_runtime_comparison.json','bootstrap_feature_cache_audit.json','fid_acceleration_amendment.json'])
            finally:
                if records: csv_rows(out/'bootstrap_fid_kid.csv', records)
