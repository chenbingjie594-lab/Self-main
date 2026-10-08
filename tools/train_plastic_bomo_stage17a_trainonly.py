"""Train the isolated pool from original pretrained SD2, never the leaked rebuild."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess

from prepare_plastic_bomo_stage17a_trainonly import REPO, FOLDERS, load, save, sha, verify_inputs, yaml_directories, validate_protocol
from rebuild_plastic_bomo_stage17a_baseline import command


def verify_pool(prepared, protocol_path, frozen, data_yaml):
    status = load(prepared / 'trainonly_preparation_status.json')
    if status['status'] != 'TRAINONLY_POOL_PREPARED_ISOLATION_PASS' or status['scope'] != 'SERVER_PREFLIGHT' or not status['generator_training_authorized']:
        raise RuntimeError('SERVER_TRAINONLY_PREFLIGHT_REQUIRED')
    for path, expected in ((prepared / 'trainonly_source_manifest.json', status['source_manifest_sha256']),
                           (prepared / 'source_isolation_audit.json', status['isolation_audit_sha256']),
                           (protocol_path, status['protocol_sha256']), (frozen, status['frozen_manifest_sha256'])):
        if sha(path) != expected:
            raise RuntimeError('FROZEN_PREPARATION_CHANGED:' + str(path))
    protocol = load(protocol_path)
    validate_protocol(protocol)
    isolation = load(prepared / 'source_isolation_audit.json')
    if sha(data_yaml) != isolation['official_data_yaml_sha256']:
        raise RuntimeError('OFFICIAL_DATA_YAML_CHANGED')
    dirs = yaml_directories(data_yaml)
    rows, train, val = verify_inputs(frozen, dirs['train'], dirs['val'])
    if train != isolation['train_images'] or val != isolation['validation_images']:
        raise RuntimeError('FROZEN_TRAIN_OR_VALIDATION_CHANGED')
    manifest = load(prepared / 'trainonly_source_manifest.json')
    records = manifest['records']
    if manifest['identity'] != protocol['identity'] or dict(Counter(r['class'] for r in records)) != protocol['source_counts']:
        raise RuntimeError('SOURCE_POOL_CHANGED')
    expected_ids = {r['donor_id'] for r in rows}
    if len(records) != 168 or {r['annotation_id'] for r in records} != expected_ids or len({r['source_id'] for r in records}) != 168:
        raise RuntimeError('SOURCE_INSTANCE_COVERAGE_CHANGED')
    root = Path(manifest['dataset_root']).resolve()
    for row in records:
        for field in ('image', 'mask'):
            path = (root / row[field + '_relative']).resolve()
            if root not in path.parents or sha(path) != row[field + '_sha256']:
                raise RuntimeError('STAGED_SOURCE_CHANGED')
    for cls in FOLDERS:
        for kind, field in (('images', 'image'), ('masks', 'mask')):
            expected = {Path(r[field + '_relative']).name for r in records if r['class'] == cls}
            if {p.name for p in (root / cls / kind).iterdir()} != expected:
                raise RuntimeError('UNLISTED_TRAINING_ASSET')
    return protocol, root, status


def verify_base(base_model, old_audit):
    expected = load(old_audit / 'baseline_rebuild_preparation_audit.json')['base_model_files_sha256']
    actual = {p.relative_to(base_model).as_posix(): sha(p) for p in sorted(base_model.rglob('*')) if p.is_file()}
    if actual != expected:
        raise RuntimeError('ORIGINAL_PRETRAINED_BASE_HASH_MISMATCH_NO_FINETUNED_SUBSTITUTION')
    if load(base_model / 'unet/config.json').get('in_channels') != 9:
        raise RuntimeError('SD2_INPAINTING_BASE_REQUIRED')
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'frozen', 'real_data_yaml', 'base_model', 'original_rebuild_audit', 'model_root', 'output', 'logs'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage17a_trainonly.json')
    a = p.parse_args()
    for key in ('prepared', 'frozen', 'real_data_yaml', 'base_model', 'original_rebuild_audit', 'model_root', 'output', 'logs', 'protocol'):
        setattr(a, key, getattr(a, key).resolve())
    if a.model_root.name != 'baseline_stage17a_trainonly_s42_noise_0.0':
        raise RuntimeError('NEW_TRAINONLY_MODEL_IDENTITY_REQUIRED')
    for path in (a.model_root, a.output, a.logs):
        if path.exists():
            raise RuntimeError('NEW_DIRECTORY_REQUIRED_NO_OVERWRITE:' + str(path))
    protocol, dataset, prepared_status = verify_pool(a.prepared, a.protocol, a.frozen, a.real_data_yaml)
    base = verify_base(a.base_model, a.original_rebuild_audit)
    a.output.mkdir(parents=True); a.logs.mkdir(parents=True)
    audit = {'identity': protocol['identity'], 'initialization': 'EXACT_ORIGINAL_PRETRAINED_SD2',
             'base_model_files_sha256': base, 'base_model': str(a.base_model),
             'source_manifest_sha256': prepared_status['source_manifest_sha256'],
             'protocol_sha256': sha(a.protocol), 'training_entrypoint_sha256': sha(REPO / 'train_dreambooth_noise.py'),
             'command_builder_sha256': sha(REPO / 'tools/rebuild_plastic_bomo_stage17a_baseline.py'),
             'source_counts': protocol['source_counts'], 'commands': {}}
    for cls, folder in FOLDERS.items():
        audit['commands'][cls] = command(a.base_model, dataset / cls / 'images', dataset / cls / 'masks',
                                         a.model_root / 'Plastic_Bomo' / folder, a.output / f'{cls}_training_runtime.json')
    save(a.output / 'trainonly_training_launch_audit.json', audit)
    save(a.output / 'trainonly_training_status.json', {'status': 'TRAINONLY_TRAINING_RUNNING', 'formal_generation_authorized': False, 'detector_training_authorized': False})
    for cls, folder in FOLDERS.items():
        verify_pool(a.prepared, a.protocol, a.frozen, a.real_data_yaml)
        if sha(REPO / 'train_dreambooth_noise.py') != audit['training_entrypoint_sha256']:
            raise RuntimeError('TRAINING_CODE_CHANGED')
        with (a.logs / f'train_{cls}.log').open('x', encoding='utf-8') as stream:
            subprocess.run(audit['commands'][cls], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
        runtime = load(a.output / f'{cls}_training_runtime.json')
        if runtime['status'] != 'FINAL_PIPELINE_SAVED' or runtime['global_step'] != 2000 or runtime['seed'] != 42 or runtime['train_text_encoder']:
            raise RuntimeError('TRAINONLY_RUNTIME_PROTOCOL_MISMATCH')
        checkpoint = a.model_root / 'Plastic_Bomo' / folder
        assets = {f.relative_to(checkpoint).as_posix(): sha(f) for f in sorted(checkpoint.rglob('*')) if f.is_file() and 'logs' not in f.relative_to(checkpoint).parts}
        for required in ('model_index.json', 'unet/config.json', 'vae/config.json', 'text_encoder/config.json'):
            if required not in assets:
                raise RuntimeError('FINAL_CHECKPOINT_INCOMPLETE')
        save(a.output / f'{cls}_checkpoint_asset_audit.json', {'checkpoint': str(checkpoint), 'files_sha256': assets, 'runtime': runtime})
        print(cls + ' train-only baseline saved', flush=True)
    verify_pool(a.prepared, a.protocol, a.frozen, a.real_data_yaml)
    verify_base(a.base_model, a.original_rebuild_audit)
    final = {'status': 'TRAINONLY_BASELINE_TRAINING_COMPLETE_RUNTIME_AUDIT_REQUIRED', 'identity': protocol['identity'],
             'source_counts': protocol['source_counts'], 'old_models_overwritten': False, 'labels_modified': False,
             'historical_checkpoint_equivalence_claimed': False, 'formal_generation_authorized': False,
             'detector_training_authorized': False, 'sampling_count': 0, 'deep_pcb_training': 0, 'bootstrap_guard_modification': 0}
    save(a.output / 'trainonly_training_status.json', final)
    print(final, flush=True)


if __name__ == '__main__':
    main()
