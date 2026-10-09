"""New R0 helpers; the authoritative Stage20A/P0 files are never edited."""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

PROTOCOL_SHA = '3e3537cdd241b79e6711525432586d674c7fe31352cea505914987ae15d98c80'
P0_SCRIPT_SHA = 'f5fa4acbfff3a0713a120a7462a1aa386ce6a38d03f8b9cd538466c3602340e4'
P0_MANIFEST_SHA = '6660bb09bda4fdbfa09dc91e0b3bc9e3a5bcf3feb46a04a6307235f46aa29f3d'
STAGE20A_MANIFEST_SHA = '2caa71e8cb3af1b6cae70b7d70974d7ccf6d70d3ea3eee37b4e22bbad1c2a079'


class Stop(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(str(detail))
        self.status, self.detail = status, detail


def require(value, status, detail):
    if not value:
        raise Stop(status, detail)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def htext(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verify_ledger(root, filename, expected):
    root = Path(root).resolve()
    manifest = root / filename
    require(manifest.is_file() and sha(manifest) == expected, 'FROZEN_RUNTIME_BINDING_CHANGED', str(manifest))
    rows = []
    for name, digest in load(manifest)['sha256'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'FROZEN_RUNTIME_BINDING_CHANGED', 'ledger traversal: ' + name)
        actual = sha(path) if path.is_file() else None
        rows.append({'file': name, 'expected_sha256': digest, 'actual_sha256': actual})
        require(actual == digest, 'FROZEN_RUNTIME_BINDING_CHANGED', rows[-1])
    return rows


def augmentation_seed(protocol, epoch, position):
    return int(htext(protocol + 'augmentation42' + str(epoch) + str(position))[:16], 16)


def validate_schedule(cfg, order):
    require(len(order['epochs']) == 150, 'DETECTOR_RUNTIME_SCHEDULE_INVALID', 'expected 150 epochs')
    roles = set(order['epochs'][0]['role_sequence'])
    require(len(roles) == 216, 'DETECTOR_RUNTIME_SCHEDULE_INVALID', 'expected 216 distinct roles')
    for epoch, row in enumerate(order['epochs']):
        expected = sorted(roles, key=lambda r: (htext(cfg['protocol'] + '42' + str(epoch) + r), r))
        require(row['epoch_0based'] == epoch and row['role_sequence'] == expected
                and row['sequence_sha256'] == htext(canonical(expected))
                and row['augmentation_seeds'] == [augmentation_seed(cfg['protocol'], epoch, i) for i in range(216)]
                and row['optimizer_attempt_after_positions_0based'] == [63, 127, 191, 215]
                and row['accumulation_group_lengths'] == [64, 64, 64, 24],
                'DETECTOR_RUNTIME_SCHEDULE_INVALID', {'epoch': epoch})
    require(order['scheduled_draws'] == order['scheduled_batches'] == 32400
            and order['scheduled_optimizer_attempts'] == 600,
            'DETECTOR_RUNTIME_SCHEDULE_INVALID', 'frozen schedule totals')


def generator_budget_dryrun(protocol, class_inputs, skipped_attempts=()):
    """No tensors/optimizer; synthetic skip scenarios exercise successful-update accounting."""
    target, accum = protocol['max_train_steps'], protocol['gradient_accumulation_steps']
    require(target == 2000 and accum == 4 and protocol['train_batch_size'] == 1,
            'GENERATOR_RUNTIME_BUDGET_INVALID', 'frozen budget')
    require(class_inputs == {'flash': 75, 'black': 97}, 'GENERATOR_RUNTIME_BUDGET_INVALID', 'frozen class input counts')
    skipped_attempts = set(skipped_attempts)
    rows = {}
    for cls in ('flash', 'black'):
        attempts = successful = skips = draws = epoch = offset = 0
        tails = []
        while successful < target:
            length = min(accum, class_inputs[cls] - offset)
            draws += length
            offset += length
            attempts += 1
            skipped = attempts in skipped_attempts
            successful += int(not skipped)
            skips += int(skipped)
            if length < accum:
                tails.append({'epoch': epoch, 'draws': length})
            if offset == class_inputs[cls]:
                epoch += 1
                offset = 0
        rows[cls] = {'successful_updates': successful, 'attempted_updates': attempts, 'simulated_amp_skips': skips,
                     'simulated_microbatch_draws': draws, 'epoch_end_partial_groups': tails,
                     'checkpoint_may_be_named_final_step2000': successful == target}
    return rows


def binding_recheck(repo, stage20a, p0, protocol, base_model, detector_model):
    require(sha(protocol) == PROTOCOL_SHA and sha(repo / 'tools/preflight_plastic_bomo_stage20b_p0.py') == P0_SCRIPT_SHA,
            'FROZEN_RUNTIME_BINDING_CHANGED', 'config or frozen P0 script')
    ledgers = {'stage20a': verify_ledger(stage20a, 'frozen_artifact_manifest.json', STAGE20A_MANIFEST_SHA),
               'server_p0': verify_ledger(p0, 'p0_frozen_artifact_manifest.json', P0_MANIFEST_SHA)}
    s = load(p0 / 'stage20b_p0_status.json')
    require(s['status'] == 'STAGE20B_STRUCTURAL_PREFLIGHT_PASS' and s['STAGE20B_EXECUTION_AUTHORIZED']
            and s['protocol_sha256'] == PROTOCOL_SHA and s['script_sha256'] == P0_SCRIPT_SHA,
            'FROZEN_RUNTIME_BINDING_CHANGED', 'P0 authorization')
    runtime = load(p0 / 'stage20b_runtime_binding.json')['environment']
    versions = {}
    for package, expected in runtime['versions'].items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        require(versions[package] == expected, 'FROZEN_RUNTIME_BINDING_CHANGED', {'package': package, 'expected': expected, 'actual': versions[package]})
    dist = importlib.metadata.distribution('ultralytics')
    source_rows = []
    for name, expected in runtime['source_files'].items():
        path = Path(dist.locate_file(name)) if name.startswith('ultralytics/') else repo / name
        actual = sha(path) if path.is_file() else None
        require(actual == expected['sha256'], 'FROZEN_RUNTIME_BINDING_CHANGED', {'source': name, 'actual': actual})
        source_rows.append({'file': name, 'path': str(path), 'sha256': actual})
    base = load(p0 / 'sd2_base_weight_binding.json')
    weight_rows = []
    for name, row in base['files'].items():
        path = base_model / name
        actual = sha(path) if path.is_file() else None
        require(actual == row['sha256'], 'FROZEN_RUNTIME_BINDING_CHANGED', {'base_file': name, 'actual': actual})
        weight_rows.append({'file': name, 'sha256': actual})
    for component in ('unet', 'vae', 'text_encoder'):
        actual = {x.relative_to(base_model).as_posix() for x in (base_model / component).iterdir() if x.suffix in ('.bin', '.safetensors')}
        expected = {x for x in base['public_reference']['weight_files'] if x.startswith(component + '/')}
        require(actual == expected, 'FROZEN_RUNTIME_BINDING_CHANGED', 'ambiguous weights: ' + component)
    expected_detector = load(p0 / 'detector_initialization_binding.json')['sha256']
    require(detector_model.is_file() and sha(detector_model) == expected_detector, 'FROZEN_RUNTIME_BINDING_CHANGED', 'YOLO initialization')
    # Only manifest-authorized V2 TRAIN assets are opened, even if their old physical folder was val.
    requests = {}
    allow = load(p0 / 'generator_train_allowlist.json')
    for row in allow['inputs']:
        require(row['split'] == 'train', 'FROZEN_RUNTIME_BINDING_CHANGED', 'non-TRAIN allowlist')
        requests[row['path']] = row['sha256']
        requests[row['full_label_path']] = row['full_label_sha256']
    normals = load(p0 / 'stage20b_normal_pool_freeze.json')['records']
    for row in normals:
        require(row['split'] == 'train' and row['confirmatory_allowed'], 'FROZEN_RUNTIME_BINDING_CHANGED', 'non-TRAIN normal')
        requests[row['runtime_path']] = row['sha256']
    for path, expected in requests.items():
        require(Path(path).is_file() and sha(path) == expected, 'FROZEN_RUNTIME_BINDING_CHANGED', {'TRAIN_asset': path})
    require(len(requests) == 666 and Counter(x['class'] for x in allow['inputs']) == {'flash': 75, 'black': 97},
            'FROZEN_RUNTIME_BINDING_CHANGED', 'TRAIN counts')
    return {'status': 'PASS', 'ledgers': ledgers, 'protocol_sha256': PROTOCOL_SHA,
            'p0_script_sha256': P0_SCRIPT_SHA, 'p0_manifest_sha256': P0_MANIFEST_SHA,
            'package_versions': versions, 'runtime_source_files': source_rows,
            'sd2_files': weight_rows, 'detector_sha256': expected_detector,
            'TRAIN_assets_verified': len(requests), 'final_eval_content_read': 0}
