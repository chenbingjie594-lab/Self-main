"""Copy exactly 174 allowed normals to an independent pool; preserve all originals."""
import argparse
from pathlib import Path
import shutil

from verify_plastic_bomo_stage17a_normal_server import verify, load, sha
from audit_plastic_bomo_stage17a_normal_lineage import historical_filter
from prepare_plastic_bomo_stage17a_trainonly import REPO, save

EXCLUDED = {'076.jpg', '092.jpg', '133.jpg', '362.jpg', '396.jpg', '411.jpg'}


def eligible_records(records, accepted, excluded):
    if set(excluded) != EXCLUDED:
        raise RuntimeError('EXACT_AUTHORIZED_SIX_EXCLUSIONS_REQUIRED')
    if len(accepted) != 180 or not EXCLUDED <= set(accepted):
        raise RuntimeError('ORIGINAL_ACCEPTED_POOL_CHANGED')
    by_name = {r['normal_filename']: r for r in records}
    if len(by_name) != 420 or len(records) != 420:
        raise RuntimeError('ORIGINAL_CATALOG_CHANGED')
    related = {name for name in accepted if by_name[name]['related_validation_parents']}
    if related != EXCLUDED:
        raise RuntimeError('SOURCE_GROUP_OVERLAP_CHANGED_NO_AUTOMATIC_NEW_EXCLUSIONS')
    retained = [by_name[name] for name in sorted(accepted) if name not in EXCLUDED]
    if len(retained) != 174:
        raise RuntimeError('EXPECTED_174_NORMALS_REQUIRED')
    return retained


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('normal_audit', 'normal_root', 'runtime_probe', 'prepared', 'pool_root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage17a_normal_isolated.json')
    p.add_argument('--scope', choices=('SERVER_AUDIT', 'LOCAL_TEST'), default='SERVER_AUDIT')
    a = p.parse_args()
    a.pool_root = a.pool_root.resolve(); a.output = a.output.resolve(); a.normal_root = a.normal_root.resolve()
    for dest in (a.pool_root, a.output):
        if dest.exists() or dest == a.normal_root or a.normal_root in dest.parents:
            raise RuntimeError('INDEPENDENT_NEW_DIRECTORY_REQUIRED:' + str(dest))
    if a.pool_root in a.output.parents or a.output in a.pool_root.parents or a.output == a.pool_root:
        raise RuntimeError('SEPARATE_POOL_AND_AUDIT_REQUIRED')
    cfg = load(a.protocol)
    baseline_path = REPO / 'configs/baseline_frozen_split70_s42.json'
    baseline = load(baseline_path)
    if (cfg['identity'] != 'stage17a_trainonly_normal_isolation_v1' or cfg['new_pool_count'] != 174
            or cfg['normal_filter'] != baseline['normal_filter'] or cfg['generator_retraining']
            or cfg['normal_filter'] != {'enabled': True, 'black_threshold': 20, 'min_mean_luminance': 30.0, 'min_nonblack_ratio': 0.60}
            or cfg['generator_parameters_changed'] or cfg['generation_seeds_changed']
            or not cfg['original_files_retained']):
        raise RuntimeError('AUTHORIZED_NORMAL_PROTOCOL_CHANGED')
    evidence = load(a.normal_audit / 'normal_lineage_status.json')
    if sha(REPO / 'inference.py') != evidence['inference_sha256']:
        raise RuntimeError('HISTORICAL_FILTER_IMPLEMENTATION_CHANGED')
    bound = verify(a)
    conditions = load(a.runtime_probe / 'frozen_probe_conditions.json')
    accepted = {Path(k).name: v for k, v in conditions['normal_pool_files_sha256'].items()}
    actual_filtered = {Path(x).name: sha(x) for x in historical_filter(a.normal_root, cfg['normal_filter'])}
    if actual_filtered != accepted:
        raise RuntimeError('UNCHANGED_FILTER_ACCEPTANCE_MISMATCH')
    records = load(a.normal_audit / 'normal_origin_manifest.json')['records']
    retained = eligible_records(records, accepted, cfg['excluded_normal_filenames'])
    a.output.mkdir(parents=True); a.pool_root.mkdir(parents=True)
    result_records = []
    for r in retained:
        name = r['normal_filename']; dest = a.pool_root / name
        shutil.copyfile(a.normal_root / name, dest)
        if sha(dest) != r['normal_sha256']:
            raise RuntimeError('COPIED_NORMAL_HASH_MISMATCH:' + name)
        result_records.append({**r, 'new_pool_path': str(dest), 'copy_bytes_exact': True})
    rechecked = verify(a)
    if rechecked != bound:
        raise RuntimeError('ORIGINAL_POOL_CHANGED_DURING_COPY')
    if {Path(x).name: sha(x) for x in historical_filter(a.pool_root, cfg['normal_filter'])} != {r['normal_filename']: r['normal_sha256'] for r in retained}:
        raise RuntimeError('NEW_POOL_UNCHANGED_FILTER_MISMATCH')
    manifest = {'identity': cfg['identity'], 'pool_root': str(a.pool_root), 'normal_count': len(result_records),
                'records': result_records, 'excluded_records': [r for r in records if r['normal_filename'] in EXCLUDED],
                'protocol_sha256': sha(a.protocol), 'baseline_inference_config_sha256': sha(baseline_path),
                'inference_sha256': sha(REPO / 'inference.py'),
                'original_origin_manifest_sha256': sha(a.normal_audit / 'normal_origin_manifest.json'),
                'original_runtime_conditions_sha256': sha(a.runtime_probe / 'frozen_probe_conditions.json')}
    save(a.output / 'isolated_normal_pool_manifest.json', manifest)
    save(a.output / 'original_normal_server_binding.json', bound)
    status = {'status': 'ISOLATED_NORMAL_POOL_PREPARED' if a.scope == 'SERVER_AUDIT' else 'LOCAL_ISOLATED_NORMAL_POOL_TEST_PASS',
              'scope': a.scope, 'identity': cfg['identity'], 'normal_count': 174,
              'excluded_normal_filenames': sorted(EXCLUDED), 'original_files_unchanged': True,
              'related_validation_source_groups_in_new_pool': 0,
              'claim_scope': 'Recovered explicit pre_part/XML relationships only; unknown acquisition relationships are not ruled out',
              'generator_retrained': False, 'generation_parameters_changed': False,
              'formal_generation_authorized': False, 'detector_training_authorized': False,
              'sampling_count': 0, 'training_count': 0,
              'pool_manifest_sha256': sha(a.output / 'isolated_normal_pool_manifest.json')}
    save(a.output / 'isolated_normal_pool_status.json', status)
    (a.output / 'ISOLATED_NORMAL_POOL_REPORT.md').write_text(
        '# Stage17A isolated normal pool v1\n\n'
        'User-authorized protocol change: original accepted 180 normals minus exactly six confirmed '
        'validation-source-group-related partitions. New 174-file pool contains byte-exact copies with '
        'unchanged numbered filenames. All 420 original files are preserved and re-hashed. '
        'No new normals, quality selection, seed changes, thresholds or model training.\n\n'
        'The unchanged historical filter is checked both before and after copying. Isolation evidence '
        'covers explicit recovered XML/pre_part relationships, not unknown acquisition groups. '
        'The original smoke result is not a reproducibility result for this changed pool. '
        'Formal slot freezing, bbox/conditioning transforms and new-pool paired reproducibility remain '
        'required before generation and detector training.\n', encoding='utf-8')
    print(__import__('json').dumps(status, indent=2))


if __name__ == '__main__':
    main()
