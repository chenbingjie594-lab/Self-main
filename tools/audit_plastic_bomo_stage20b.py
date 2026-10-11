"""Immutable checkpoint/structural/compute freezes and final headroom report; no forward."""
from collections import Counter
import hashlib
from pathlib import Path

from stage20b_formal_common import (ARTIFACTS, STATES, canonical, child_main, context, htext,
    load, now, require, save, sha, summarize_metrics, synthetic_label, tree_binding,
    update_status, validate_bbox, verify_tree)


def freeze_generators(run, p0):
    classes = {}
    for cls in ('flash', 'black'):
        row = load(run / ('generator_training_' + cls + '.json'))
        cfg = load(p0 / 'stage20b_generator_training_protocol.json')
        require(row['status'] == 'COMPLETE' and row['successful_updates'] == cfg['max_train_steps']
                and row['last_step_legal'] and row['frozen_components_unchanged'] and row['checkpoint_saved'],
                cls.upper() + '_GENERATOR_TRAINING_FAILED', 'complete legal successful budget')
        verify_tree(run, row['checkpoint'])
        classes[cls] = {'role': row['checkpoint_role'], 'successful_updates': row['successful_updates'], 'checkpoint': row['checkpoint']}
    require(len({x['role'] for x in classes.values()}) == 2, 'FORMAL_CHECKPOINT_INVALID', 'unique class roles')
    save(run / 'generator_checkpoint_binding.json', {'status': 'FROZEN', 'classes': classes,
                                                    'frozen_at': now(), 'before_any_formal_sampling': True})


def validate_banks(run, p0):
    from PIL import Image
    import numpy as np
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    rng = {x['slot_id']: x['banks'] for x in load(p0 / 'stage20b_rng_manifest.json')['records']}
    checkpoint = load(run / 'generator_checkpoint_binding.json')['classes']
    for c in checkpoint.values():
        verify_tree(run, c['checkpoint'])
    records = []
    paired = {}
    for bank in ('A', 'B', 'C'):
        manifest = load(run / ('formal_bank_' + bank + '_manifest.json'))
        require(manifest['status'] == 'COMPLETE' and len(manifest['records']) == 80,
                'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', bank + ' missing raw outputs')
        require(Counter(x['class'] for x in manifest['records']) == {'flash': 40, 'black': 40},
                'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'class counts')
        require([x['slot_id'] for x in manifest['records']] == [x['slot_id'] for x in slots],
                'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'frozen slot order')
        for slot, row in zip(slots, manifest['records']):
            validate_bbox(slot)
            expected_rng = rng[slot['slot_id']][bank]
            require(row['frozen_slot'] == slot and row['non_rng_metadata_sha256'] == expected_rng['non_rng_metadata_sha256']
                    and row['sample_seed'] == expected_rng['sample_seed'] and row['bank_seed'] == expected_rng['bank_seed']
                    and row['checkpoint_tree_sha256'] == checkpoint[slot['class']]['checkpoint']['tree_sha256'],
                    'FORMAL_FROZEN_BINDING_CHANGED', 'paired slot/checkpoint/RNG metadata')
            for key in ('image_path', 'label_path'):
                path = (run / row[key]).resolve()
                require(path.is_relative_to((run / ('synthetic_' + bank)).resolve()) and path.is_file()
                        and sha(path) == row[key.replace('_path', '_sha256')],
                        'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', row[key])
            require((run / row['label_path']).read_text(encoding='utf-8') == synthetic_label(slot),
                    'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'label geometry')
            with Image.open(run / row['image_path']) as im:
                im.load()
                require(im.format == 'PNG' and im.mode == 'RGB' and list(im.size) == slot['geometry']['final_canvas_wh']
                        and bool(np.isfinite(np.asarray(im)).all())
                        and hashlib.sha256(im.tobytes()).hexdigest() == row['decoded_RGB_sha256'],
                        'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'decode/finite/canvas/RGB')
            paired.setdefault(slot['slot_id'], []).append((row['non_rng_metadata_sha256'], row['checkpoint_tree_sha256'], row['label_sha256']))
            records.append(row)
    require(len(records) == 240 and all(len(v) == 3 and len(set(v)) == 1 for v in paired.values()),
            'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'ABC nonRNG pairing')
    save(run / 'formal_synthetic_manifest.json', {'status': 'FROZEN', 'records': records, 'count': 240,
        'bank_counts': dict(Counter(x['bank'] for x in records)), 'frozen_at': now(),
        'all_successful_outputs_used': True, 'selector_count': 0, 'reroll_count': 0})
    save(run / 'formal_synthetic_validity.json', {'status': 'PASS', 'count': 240,
        'allowed_checks_only': ['file exists', 'decode valid', 'finite pixels', 'annotation geometry', 'pipeline structural success'],
        'ABC_non_RNG_metadata_identical': True, 'quality_gate': False, 'inclusion_exclusion_count': 0})


def freeze_detectors(run, p0):
    order = load(p0 / 'stage20b_training_order_spec.json')
    arms, compute = {}, {}
    for arm in ('RR', 'A', 'B', 'C'):
        row = load(run / ('detector_arm_' + arm + '_training.json'))
        require(row['status'] == 'COMPLETE' and row['epochs'] == 150 and row['actual_draws'] == order['scheduled_draws']
                and row['attempted_optimizer_updates'] == order['scheduled_optimizer_attempts']
                and row['successful_optimizer_updates'] + row['AMP_skipped_updates'] == row['attempted_optimizer_updates']
                and len(row['epoch_records']) == 150 and row['official_final_eval_content_read'] == 0,
                'FORMAL_DETECTOR_COMPUTE_INVALID', arm)
        for expected, actual in zip(order['epochs'], row['epoch_records']):
            require(actual['actual_role_sequence_sha256'] == expected['sequence_sha256'] and actual['draws'] == 216
                    and [x['normalizer'] for x in actual['attempts']] == expected['accumulation_group_lengths']
                    and [x['position'] for x in actual['attempts']] == expected['optimizer_attempt_after_positions_0based'],
                    'FORMAL_DETECTOR_COMPUTE_INVALID', {'arm': arm, 'epoch': actual['epoch_0based']})
        require(sha(run / row['last_checkpoint']) == row['last_sha256'], 'FORMAL_FROZEN_BINDING_CHANGED', arm + ' last.pt')
        arms[arm] = {'path': row['last_checkpoint'], 'sha256': row['last_sha256'], 'epochs': row['epochs'],
                     'successful_updates': row['successful_optimizer_updates'], 'primary': True,
                     'model_representation': row['primary_model_representation']}
        compute[arm] = {k: row[k] for k in ('actual_draws', 'scheduled_draws', 'attempted_optimizer_updates',
            'scheduled_optimizer_attempts', 'successful_optimizer_updates', 'AMP_skipped_updates', 'nonfinite_losses',
            'gradient_nonfinite_count', 'epochs', 'order_spec_sha256', 'initial_parameter_binding_sha256')}
    require(len({x['initial_parameter_binding_sha256'] for x in compute.values()}) == 1,
            'DETECTOR_INITIALIZATION_MISMATCH', 'four initial parameter bindings')
    save(run / 'detector_compute_audit.json', {'status': 'PASS', 'arms': compute,
        'same_scheduled_exposure': True, 'successful_updates_equal_required': False,
        'AMP_success_count_range': max(x['successful_optimizer_updates'] for x in compute.values()) - min(x['successful_optimizer_updates'] for x in compute.values()),
        'extra_training_to_equalize_successes': 0, 'train_final_eval_reads': 0})
    save(run / 'detector_checkpoint_binding.json', {'status': 'FROZEN', 'arms': arms,
        'frozen_at': now(), 'final_eval_may_begin_only_after_this_four_arm_barrier': True})


def finish(run, p0, stage20a):
    state = load(run / 'stage20b_status.json')
    require([x['state'] for x in state['state_history']] == list(STATES), 'FORMAL_STATE_SEQUENCE_INVALID', 'complete chain')
    for name, expected in (('generator_checkpoint_binding', 'FROZEN'), ('trained_checkpoint_determinism_audit', 'PASS'),
                           ('formal_synthetic_manifest', 'FROZEN'), ('formal_synthetic_validity', 'PASS'),
                           ('detector_checkpoint_binding', 'FROZEN'), ('detector_compute_audit', 'PASS')):
        require(load(run / (name + '.json'))['status'] == expected, 'FORMAL_AUDIT_FAILED', name)
    target = load(p0 / 'stage20b_generator_training_protocol.json')['max_train_steps']
    for cls, row in load(run / 'generator_checkpoint_binding.json')['classes'].items():
        require(state['generator_' + cls + '_successful_updates'] == row['successful_updates'] == target,
                'FORMAL_AUDIT_FAILED', cls + ' successful update accounting')
        verify_tree(run, row['checkpoint'])
    frozen_detectors = load(run / 'detector_checkpoint_binding.json')['arms']
    for arm, row in frozen_detectors.items():
        require(sha(run / row['path']) == row['sha256']
                and state['detector_' + arm + '_successful_updates'] == row['successful_updates'],
                'FORMAL_FROZEN_BINDING_CHANGED', arm + ' detector binding/accounting')
    access = load(run / 'final_eval_access_audit.json')
    require(access['status'] == 'COMPLETE' and access['unified_rounds_completed'] == 1
            and access['official_final_eval_image_forwards'] == 192, 'FORMAL_FINAL_EVAL_FAILED', 'unified round incomplete')
    metrics = {key: load(run / (key + '_metrics.json')) for key in ('RR', 'RawA', 'RawB', 'RawC')}
    require(all(x['status'] == 'COMPLETE' for x in metrics.values()), 'FORMAL_METRICS_INVALID', 'missing completed metric')
    for name, metric in metrics.items():
        arm = 'RR' if name == 'RR' else name[-1]
        require(metric['checkpoint_sha256'] == frozen_detectors[arm]['sha256']
                and metric['evaluation_image_count'] == 48
                and metric['evaluation_protocol_sha256'] == sha(p0 / 'stage20b_detector_protocol.json'),
                'FORMAL_METRICS_INVALID', name + ' metric lineage')
    # All result thresholds come from the Stage20A frozen gate, NOT a new config.
    result = summarize_metrics(metrics, load(stage20a / 'future_baseline_gate.json'))
    save(run / 'stage20b_headroom.json', {k: v for k, v in result.items() if k != 'classwise'})
    save(run / 'stage20b_classwise.json', {'status': 'COMPLETE', 'classes': result['classwise']})
    save(run / 'stage20b_generation_nuisance.json', {'status': 'COMPLETE',
        'sample_std_pp': result['generation_bank_std_pp'], 'range_pp': result['generation_bank_range_pp'],
        'bank_seeds': [load(p0 / 'stage20b_rng_manifest.json')['records'][0]['banks'][k]['bank_seed'] for k in ('A', 'B', 'C')],
        'detector_seed': load(p0 / 'stage20b_detector_protocol.json')['training']['seed'], 'detector_seed_replicates': False,
        'historical_nuisance_threshold_reused': False})
    update_status(run, **{k: v for k, v in result.items() if k != 'classwise'},
        full_experiment_complete=True, completed_at=now(), official_final_eval_rounds=1,
        final_eval_forward_count=192, automatic_next_stage_started=False)
    lines = ['# Stage20B — Raw SD2 baseline utility calibration', '', 'Status: `' + result['status'] + '`.', '',
        '| Arm | mAP50-95 | Delta vs RR (pp) |', '|---|---:|---:|',
        f"| RR | {result['RR_map5095']:.6f} | — |"]
    for k in ('A', 'B', 'C'):
        lines.append(f"| Raw{k} | {result[k+'_map5095']:.6f} | {result['delta_'+k+'_pp']:+.6f} |")
    lines += ['', f"Mean delta: {result['mean_delta_pp']:+.6f} pp; sample SD: {result['std_delta_pp']:.6f} pp; strict wins: {result['wins_vs_RR']}/3.",
        f"Gates A/B/C: {result['Gate_A']}/{result['Gate_B']}/{result['Gate_C']}.",
        f"Flash mean AP delta: {result['Flash_mean_delta_pp']:+.6f} pp; Black: {result['Black_mean_delta_pp']:+.6f} pp.",
        f"Generation-bank nuisance sample SD/range: {result['generation_bank_std_pp']:.6f}/{result['generation_bank_range_pp']:.6f} pp.", '',
        'All three raw banks use the same frozen80 slots and two class final_step2000 pipelines. No quality scoring, selector, reroll or discarded raw output.',
        'Four detectors share seed42, initialization, role order, augmentation seeds, 32400 draws and600 scheduled attempts. Actual AMP skips/success counts are preserved in detector_compute_audit.json; no compensatory training.',
        'Primary checkpoints are the pinned native EMA representation of epoch150 last.pt. No best.pt, early stopping or validation-based checkpoint selection.',
        'Only after all four checkpoint SHA freezes did one unified evaluation round open the48 V2 final_eval images and60 dataset-defined boxes.', '',
        '## Limits', '', 'physical_source_complete=false: no KNOWN source-group leakage, not proof of absence of unknown physical relationships.',
        'Labels are dataset-defined supervision, not universally human-verified physical truth. Rectangular bbox masks are weak geometry, not segmentation ground truth.',
        'Bank variation is generation-RNG variation, not detector-seed confirmation. No comparisons to historical Stage14–18 absolute mAP.',
        'BootstrapGuard=0; DeepPCB=0; TDCRG implementation=0. No automatic Stage20C/TDCRG, regardless of outcome.', '',
        '## Review files', '', 'formal_protocol_binding.json binds immutable Stage20A/P0/R0 and runtime/source/model bytes.',
        'generator_checkpoint_binding.json and detector_checkpoint_binding.json identify every primary pipeline/checkpoint.',
        'formal_synthetic_manifest.json binds all240 PNG/labels; final_eval_access_audit.json records the evaluation barrier/population.',
        'Full GPU update/draw logs and large weights remain under this run directory on the server.']
    (run / 'STAGE20B_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    required = ['formal_protocol_binding.json', 'stage20b_status.json', 'STAGE20B_REPORT.md'] + [x + '.json' for x in ARTIFACTS]
    save(run / 'formal_result_artifact_manifest.json', {'sha256': {n: sha(run / n) for n in required},
        'complete_run': True, 'heavy_model_and_image_assets_are_bound_in_separate_manifests': True})
    # One authoritative complete result, never a pointer to a failed/partial execution.
    pointer = run.parent / 'authoritative_formal_run.json'
    require(not pointer.exists(), 'FORMAL_AUTHORITY_ALREADY_EXISTS', 'do not overwrite an existing complete run')
    save(pointer, {'run': run.name, 'status': result['status'], 'artifact_manifest_sha256': sha(run / 'formal_result_artifact_manifest.json')})
    print(canonical(result), flush=True)


def audit(a):
    states = {'freeze_generators': 'BLACK_GENERATOR_TRAINED', 'validate_banks': 'BANK_C_GENERATED',
              'freeze_detectors': 'DETECTOR_C_TRAINED', 'finish': 'FINAL_EVAL_COMPLETE'}
    _, p0, stage20a = context(a.run, (states[a.action],))
    a.failure_reason = 'FORMAL_AUDIT_FAILED'
    if a.action == 'freeze_generators':
        freeze_generators(a.run, p0)
    elif a.action == 'validate_banks':
        validate_banks(a.run, p0)
    elif a.action == 'freeze_detectors':
        freeze_detectors(a.run, p0)
    else:
        finish(a.run, p0, stage20a)


if __name__ == '__main__':
    child_main(audit, __doc__, lambda p: p.add_argument('--action', choices=('freeze_generators', 'validate_banks', 'freeze_detectors', 'finish'), required=True))
