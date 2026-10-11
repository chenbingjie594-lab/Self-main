"""Read-only downloaded Stage20B review. No GPU, forward, training or authority edits."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from stage20b_formal_common import R0_CODE, R0_MANIFEST_SHA, STATES, real_assets
from stage20b_runtime_contract import canonical, htext, load

REPO = Path(__file__).resolve().parents[1]


class Review:
    def __init__(self):
        self.checks = []
        self.hashes = {}
        self.missing = set()

    def check(self, name, ok):
        self.checks.append({'check': name, 'passed': bool(ok)})

    def digest(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            if not path.is_file():
                self.missing.add(path)
                return None
            with path.open('rb') as handle:
                self.hashes[path] = hashlib.file_digest(handle, 'sha256').hexdigest()
        return self.hashes[path]

    def verify(self, root, records, label, optional=False):
        root = Path(root).resolve()
        errors = []
        for name, expected in records.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                errors.append(name + ': path escape')
                continue
            actual = self.digest(path)
            if actual != expected and not (optional and actual is None):
                errors.append(name)
        self.check(label, not errors)
        if errors:
            print(label + ': mismatches ' + repr(errors[:5]), flush=True)


def recompute(metrics, gate):
    rr = metrics['RR']['map5095']
    banks = [metrics['Raw' + bank]['map5095'] for bank in 'ABC']
    delta = [100 * (value - rr) for value in banks]
    classwise = {}
    for cls in ('flash', 'black'):
        values = [100 * (metrics['Raw' + b]['per_class'][cls]['ap5095']
                         - metrics['RR']['per_class'][cls]['ap5095']) for b in 'ABC']
        classwise[cls] = {'delta_pp': values, 'mean_delta_pp': statistics.mean(values)}
    mean = statistics.mean(delta)
    wins = sum(value > 0 for value in delta)
    gates = {'Gate_A': mean >= gate['A_mean_raw_bank_minus_RR_ge_pp'],
             'Gate_B': wins >= gate['B_strict_bank_wins_ge'],
             'Gate_C': all(x['mean_delta_pp'] >= gate['C_every_class_mean_delta_ge_pp']
                           for x in classwise.values())}
    return {'RR_map5095': rr, **{b + '_map5095': v for b, v in zip('ABC', banks)},
            **{'delta_' + b + '_pp': v for b, v in zip('ABC', delta)},
            'mean_delta_pp': mean, 'std_delta_pp': statistics.stdev(delta),
            'min_delta_pp': min(delta), 'max_delta_pp': max(delta), 'wins_vs_RR': wins,
            'Flash_mean_delta_pp': classwise['flash']['mean_delta_pp'],
            'Black_mean_delta_pp': classwise['black']['mean_delta_pp'],
            'generation_bank_std_pp': 100 * statistics.stdev(banks),
            'generation_bank_range_pp': 100 * (max(banks) - min(banks)),
            **gates, 'TDCRG_DEVELOPMENT_AUTHORIZED': all(gates.values()),
            'status': ('GENERATOR_BASELINE_HEADROOM_CONFIRMED' if all(gates.values())
                       else 'GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED'), 'classwise': classwise}


def review(run):
    run = Path(run).resolve()
    r = Review()
    pointer = load(run.parent / 'authoritative_formal_run.json')
    ledger = load(run / 'formal_result_artifact_manifest.json')
    state = load(run / 'stage20b_status.json')
    binding = load(run / 'formal_protocol_binding.json')
    p0 = REPO / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight' / Path(binding['p0']).name
    stage20a = REPO / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol'
    r0 = REPO / 'results_for_gpt/plastic_bomo_stage20b_r0_runtime' / Path(binding['r0']).name
    r.check('authority selects completed run and ledger bytes', pointer['run'] == run.name
            and pointer['status'] == state['status'] and state['full_experiment_complete']
            and r.digest(run / 'formal_result_artifact_manifest.json') == pointer['artifact_manifest_sha256'])
    r.verify(run, ledger['sha256'], 'formal ledger and interruption snapshot bytes')
    r.check('complete frozen state sequence', [x['state'] for x in state['state_history']] == list(STATES))
    for name, root in (('stage20a', stage20a), ('server_p0', p0)):
        rows = binding['P0_binding']['ledgers'][name]
        r.check(name + ' expected/actual binding agreement', all(x['expected_sha256'] == x['actual_sha256'] for x in rows))
        r.verify(root, {x['file']: x['expected_sha256'] for x in rows}, name + ' local authority hashes')
    r.verify(r0, {x['file']: x['expected_sha256'] for x in binding['R0_artifacts']}, 'R0 local artifact hashes')
    r.check('R0 manifest fixed authority', r.digest(r0 / 'r0_artifact_manifest.json') == R0_MANIFEST_SHA)
    r.check('R0 source identities frozen', binding['R0_code_sha256'] == R0_CODE)
    r.verify(REPO, {**R0_CODE, **binding['formal_code_sha256'],
                   'tools/recover_plastic_bomo_stage20b_arm_c.py': binding['manual_recovery_code_sha256'],
                   'configs/plastic_bomo_stage20b_p0.json': binding['P0_binding']['protocol_sha256'],
                   'tools/preflight_plastic_bomo_stage20b_p0.py': binding['P0_binding']['p0_script_sha256']},
             'local frozen executable and config bytes')
    runtime = load(r0 / 'runtime_environment.json')
    r.check('declared runtime equals qualified R0', all(runtime[k] == v for k, v in binding['runtime'].items()))
    r.check('no evaluation at binding', binding['final_eval_content_read_at_binding'] == 0)
    journal = load(run / 'manual_arm_C_recovery.json')
    source = run.parent / Path(journal['source_run']).name
    print('Checking unchanged original and imported frozen files (including large assets).', flush=True)
    r.verify(source, journal['source_sha256'], 'original stopped source bytes', optional=True)
    r.verify(run, journal['reused_sha256'], 'reused asset bytes', optional=True)
    old = load(run / 'manual_recovery/source_snapshot/stage20b_status.json')
    r.check('source stayed stopped and unevaluated', old['status'].startswith('STOPPED_')
            and old['state'] == 'DETECTOR_B_TRAINED' and old['official_final_eval_rounds'] == 0
            and state['state_history'][:len(old['state_history'])] == old['state_history'])
    interrupted = load(run / 'manual_recovery/source_snapshot/detector_arm_C_training.json')
    r.check('manual C fresh restart is explicit, without new generator/slots', journal['status'] == 'COMPLETE'
            and journal['reused_detector_arms'] == ['RR', 'A', 'B'] and journal['restarted_detector_arms'] == ['C']
            and journal['restart_epoch'] == 1 and not journal['seed_and_hyperparameters_changed']
            and journal['new_generator_training_count'] == journal['new_synthetic_generation_count'] == 0
            and not journal['source_run_modified'] and not journal['external_kill_cause_independently_verified']
            and journal['extra_interrupted_detector_attempts'] == 1
            and all(interrupted.get(k) == v for k, v in journal['interrupted_C_observed_counters'].items()))
    gc = load(run / 'generator_checkpoint_binding.json')['classes']
    public = {x['file']: x['sha256'] for x in binding['P0_binding']['sd2_files']}
    gen_protocol = p0 / 'stage20b_generator_training_protocol.json'
    for cls, n in (('flash', 75), ('black', 97)):
        g = load(run / ('generator_training_' + cls + '.json'))
        r.check(cls + ' generator exact legal successful budget', g['status'] == 'COMPLETE'
                and g['successful_updates'] == gc[cls]['successful_updates'] == 2000
                and g['attempted_updates'] == g['successful_updates'] + g['AMP_skips']
                and g['last_step_legal'] and g['frozen_components_unchanged']
                and g['checkpoint_saved'] and g['TRAIN_instances'] == n
                and len(set(g['input_annotation_ids'])) == n and g['unet_master_dtype'] == 'float32'
                and g['protocol_sha256'] == r.digest(gen_protocol)
                and g['official_final_eval_content_read'] == 0 and g['checkpoint'] == gc[cls]['checkpoint'])
        tree = gc[cls]['checkpoint']
        r.check(cls + ' checkpoint tree digest', htext(canonical(tree['sha256'])) == tree['tree_sha256'])
        r.verify(run / tree['path'], tree['sha256'], cls + ' checkpoint bytes', optional=True)
        r.check(cls + ' only UNet weights changed from public pipeline', all(v == public[k] for k, v in tree['sha256'].items()
                if k != 'unet/diffusion_pytorch_model.safetensors') and tree['sha256']['unet/diffusion_pytorch_model.safetensors']
                != public['unet/diffusion_pytorch_model.safetensors'])
    sanity = load(run / 'trained_checkpoint_determinism_audit.json')
    r.check('both trained checkpoint same-seed sanity, excluded from training', sanity['status'] == 'PASS'
            and sanity['sampling_count'] == 4 and sanity['formal_synthetic_count'] == 0
            and all(x['same_seed_RGB_identical'] and not x['detector_input_allowed'] for x in sanity['records']))
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    rng = {x['slot_id']: x['banks'] for x in load(p0 / 'stage20b_rng_manifest.json')['records']}
    syn = load(run / 'formal_synthetic_manifest.json')
    collected = []
    from PIL import Image
    for bank in 'ABC':
        records = load(run / ('formal_bank_' + bank + '_manifest.json'))['records']
        r.check(bank + ' exact frozen80 slots and40/40 classes', len(records) == 80
                and Counter(x['class'] for x in records) == {'flash': 40, 'black': 40}
                and [x['slot_id'] for x in records] == [x['slot_id'] for x in slots])
        for slot, row in zip(slots, records):
            expected = rng[slot['slot_id']][bank]
            r.check(bank + ' frozen metadata/RNG ' + slot['slot_id'], row['frozen_slot'] == slot
                    and all(row[k] == expected[k] for k in ('bank_seed', 'sample_seed', 'non_rng_metadata_sha256'))
                    and row['checkpoint_tree_sha256'] == gc[slot['class']]['checkpoint']['tree_sha256']
                    and not row['quality_selection'] and row['formal_synthetic'])
            r.verify(run, {row['image_path']: row['image_sha256'], row['label_path']: row['label_sha256']},
                     bank + ' synthetic bytes ' + slot['slot_id'], optional=True)
            path = run / row['image_path']
            if path.is_file():
                with Image.open(path) as im:
                    im.load()
                    r.check(bank + ' decoded RGB ' + slot['slot_id'], im.mode == 'RGB' and im.format == 'PNG'
                            and list(im.size) == slot['geometry']['final_canvas_wh']
                            and hashlib.sha256(im.tobytes()).hexdigest() == row['decoded_RGB_sha256'])
        collected.extend(records)
    r.check('all240 raw outputs included without reroll', syn['records'] == collected and syn['count'] == 240
            and syn['all_successful_outputs_used'] and syn['reroll_count'] == syn['selector_count'] == 0)
    order_path = p0 / 'stage20b_training_order_spec.json'
    order = load(order_path)
    init_hash = htext(canonical(load(run / 'detector_initialization_comparison.json')))
    detectors = load(run / 'detector_checkpoint_binding.json')
    compute = {}
    print('Checking four complete per-draw logs against frozen role/augmentation order.', flush=True)
    for arm in ('RR', 'A', 'B', 'C'):
        d = load(run / ('detector_arm_' + arm + '_training.json'))
        r.check(arm + ' complete matched150epoch schedule', d['status'] == 'COMPLETE' and d['epochs'] == 150
                and d['actual_draws'] == d['scheduled_draws'] == 32400
                and d['attempted_optimizer_updates'] == d['scheduled_optimizer_attempts'] == 600
                and d['successful_optimizer_updates'] + d['AMP_skipped_updates'] == 600
                and d['official_final_eval_content_read'] == 0 and not d['early_stopping_used']
                and not d['best_checkpoint_selection'] and not d['native_full_train_used']
                and d['order_spec_sha256'] == r.digest(order_path)
                and d['training_protocol_sha256'] == r.digest(p0 / 'stage20b_detector_protocol.json')
                and d['initial_parameter_binding_sha256'] == init_hash)
        attempts = [x for epoch in d['epoch_records'] for x in epoch['attempts']]
        r.check(arm + ' actual post-step totals match attempts', len(attempts) == 600
                and sum(x['optimizer_step_executed'] for x in attempts) == d['successful_optimizer_updates']
                and sum(x['AMP_skip'] for x in attempts) == d['AMP_skipped_updates'])
        r.check(arm + ' role hashes and64/64/64/24 tail normalization', len(d['epoch_records']) == 150 and all(
            actual['actual_role_sequence_sha256'] == expected['sequence_sha256'] and actual['draws'] == 216
            and [x['normalizer'] for x in actual['attempts']] == [64, 64, 64, 24]
            and [x['position'] for x in actual['attempts']] == [63, 127, 191, 215]
            for actual, expected in zip(d['epoch_records'], order['epochs'])))
        log = run / 'logs' / ('detector_' + arm + '_draws.jsonl')
        r.check(arm + ' draw log SHA', r.digest(log) == d['per_draw_log_sha256'])
        sequence_ok = True
        count = 0
        with log.open(encoding='utf-8') as handle:
            for count, line in enumerate(handle, 1):
                row = json.loads(line)
                epoch, pos = divmod(count - 1, 216)
                expected = order['epochs'][epoch] if epoch < 150 else None
                sequence_ok &= expected is not None and row['epoch'] == epoch and row['position'] == pos
                if expected:
                    sequence_ok &= row['role'] == expected['role_sequence'][pos] and row['augmentation_seed'] == expected['augmentation_seeds'][pos]
        r.check(arm + ' consumed32400 roles/augmentation seeds match P0', sequence_ok and count == 32400)
        r.verify(run, {d['last_checkpoint']: d['last_sha256']}, arm + ' primary last.pt bytes', optional=True)
        r.check(arm + ' metric checkpoint freeze lineage', detectors['arms'][arm]['sha256'] == d['last_sha256']
                and state['detector_' + arm + '_successful_updates'] == d['successful_optimizer_updates'])
        roles = load(run / ('detector_' + arm) / 'train_role_binding.json')
        r.check(arm + ' TRAIN-only136real+80extra roles', roles['full_real_base'] == 136 and roles['extra_roles'] == 80
                and roles['final_eval_reference_count'] == 0 and roles['V2_TRAIN_only']
                and set(roles['roles']) == set(order['epochs'][0]['role_sequence']))
        compute[arm] = {k: d[k] for k in ('epochs', 'actual_draws', 'attempted_optimizer_updates',
                                        'successful_optimizer_updates', 'AMP_skipped_updates')}
    access = load(run / 'final_eval_access_audit.json')
    population = real_assets(stage20a, p0, 'final_eval')
    expected_nodes = {x['node_id'] for x in population}
    eval_files = {}
    target_counts = Counter()
    available_labels = 0
    for row in population:
        image = 'final_eval_inputs/images/final_eval/' + row['node_id'] + Path(row['image_path']).suffix
        label = 'final_eval_inputs/labels/final_eval/' + row['node_id'] + '.txt'
        eval_files.update({image: row['image_sha256'], label: row['label_sha256']})
        if (run / label).is_file():
            available_labels += 1
            target_counts.update(int(line.split()[0]) for line in (run / label).read_text().splitlines() if line.strip())
    r.verify(run, eval_files, 'downloaded final_eval image/label bytes against frozen V2 population', optional=True)
    if available_labels == 48:
        r.check('downloaded final_eval annotations equal33Flash27Black', target_counts == {0: 33, 1: 27})
    r.check('one unified48image four-arm round only after checkpoint barrier', access['status'] == 'COMPLETE'
            and access['unified_rounds_started'] == access['unified_rounds_completed'] == 1
            and access['official_final_eval_image_forwards'] == 192 and len(expected_nodes) == 48
            and access['checkpoint_binding_sha256'] == r.digest(run / 'detector_checkpoint_binding.json')
            and access['state_at_first_access'] == 'ALL_DETECTORS_FROZEN'
            and datetime.fromisoformat(detectors['frozen_at']) < datetime.fromisoformat(access['first_authorized_content_access_at'])
            and all(x['image_forwards'] == 48 and len(x['node_ids']) == len(set(x['node_ids'])) == 48
                    and set(x['node_ids']) == expected_nodes for x in access['arms'].values()))
    metrics = {name: load(run / (name + '_metrics.json')) for name in ('RR', 'RawA', 'RawB', 'RawC')}
    r.check('all metrics primary epoch150, same protocol/population', all(x['status'] == 'COMPLETE'
            and x['evaluation_image_count'] == 48 and x['target_counts'] == {'flash': 33, 'black': 27}
            and x['checkpoint_sha256'] == detectors['arms']['RR' if name == 'RR' else name[-1]]['sha256']
            and x['evaluation_protocol_sha256'] == r.digest(p0 / 'stage20b_detector_protocol.json')
            and math.isclose(x['map5095'], statistics.mean(c['ap5095'] for c in x['per_class'].values()), abs_tol=1e-12)
            for name, x in metrics.items()))
    result = recompute(metrics, load(stage20a / 'future_baseline_gate.json'))
    r.check('independent arithmetic and all frozen gates match published status', all(
        math.isclose(v, state[k], rel_tol=1e-12, abs_tol=1e-12) if isinstance(v, (float, int)) and not isinstance(v, bool)
        else v == state[k] for k, v in result.items() if k != 'classwise'))
    r.check('no auto next-stage or forbidden components', not state['automatic_next_stage_started'] and not state['stage20c_auto_start']
            and state['DeepPCB'] == state['BootstrapGuard'] == state['TDCRG_implementation'] == 0)
    failed = [x['check'] for x in r.checks if not x['passed']]
    return {'status': 'LOCAL_FORMAL_REVIEW_PASS' if not failed else 'LOCAL_FORMAL_REVIEW_FAILED',
            'scope': 'LOCAL_DOWNLOADED_ARTIFACT_REVIEW', 'run': run.name,
            'reviewer_sha256': r.digest(Path(__file__)), 'formal_ledger_sha256': r.digest(run / 'formal_result_artifact_manifest.json'),
            'checks': r.checks, 'failed_checks': failed, 'hashed_files': len(r.hashes),
            'missing_optional_downloads': sorted(str(x.relative_to(REPO)) if x.is_relative_to(REPO) else str(x) for x in r.missing),
            'recomputed_result': result, 'primary_detector_compute': compute,
            'downloaded_final_eval_files_hashed': sum((run / n).is_file() for n in eval_files),
            'downloaded_final_eval_label_counts': dict(target_counts),
            'interrupted_C_observed_counters_excluded_from_primary': journal['interrupted_C_observed_counters'],
            'manual_recovery_disclosed': True, 'external_kill_cause_independently_verified': False,
            'GPU_execution_performed_locally': False, 'AP_recomputed_from_detector_predictions': False,
            'formal_result_files_modified': False, 'next_stage_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.resolve().is_relative_to(a.run.resolve()):
        parser.error('review output must be outside immutable formal run')
    if a.output.exists():
        parser.error('review output already exists; preserve the previous review')
    result = review(a.run)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('status', 'hashed_files', 'failed_checks')}, indent=2), flush=True)
    if result['failed_checks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
