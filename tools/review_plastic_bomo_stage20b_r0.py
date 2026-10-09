"""Independently review downloaded R0 evidence; no model or original dataset access."""
import argparse
import hashlib
import math
from pathlib import Path

from stage20b_runtime_contract import (P0_MANIFEST_SHA, P0_SCRIPT_SHA, PROTOCOL_SHA,
    STAGE20A_MANIFEST_SHA, augmentation_seed, canonical, generator_budget_dryrun,
    htext, load, save, sha, verify_ledger)
from stage20b_detector_controller import Stage20BDetectorController, accumulation_dryrun
from qualify_plastic_bomo_stage20b_r0 import OUTPUTS, formal_plan

ROOT = Path(__file__).resolve().parents[1]
SERVER_MANIFEST_SHA = '282a381e2dc402f372204c45df3cf7d137d9e437b0b2aac0835ec32643b59f85'


def downloaded_ledger(server):
    server = server.resolve()
    rows = []
    for name, expected in load(server / 'r0_artifact_manifest.json')['sha256'].items():
        path = (server / name).resolve()
        if not path.is_relative_to(server):
            raise ValueError('R0_LEDGER_PATH_OUTSIDE_RESULT_DIRECTORY: ' + name)
        actual = sha(path) if path.is_file() else None
        rows.append({'file': name, 'expected_sha256': expected, 'actual_sha256': actual, 'pass': actual == expected})
    return {'manifest_sha256': sha(server / 'r0_artifact_manifest.json'), 'files': rows,
            'count': len(rows), 'pass': bool(rows) and all(x['pass'] for x in rows)}


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def valid_gradients(g):
    return (g['all_gradients_finite'] and g['all_trainable_parameters_fp32']
            and 0 < g['nonzero_gradients'] <= g['non_none_gradients'] <= g['parameters'])


def review(repo, server, p0, stage20a):
    cfg = load(repo / 'configs/plastic_bomo_stage20b_p0.json')
    status = load(server / 'stage20b_r0_status.json')
    artifacts = downloaded_ledger(server)
    checks, details = {}, {}
    checks['downloaded_manifest_identity'] = artifacts['manifest_sha256'] == SERVER_MANIFEST_SHA
    checks['all27_downloaded_artifact_bytes'] = artifacts['pass'] and artifacts['count'] == 27
    names = {x['file'] for x in artifacts['files']}
    expected_audits = {name + '.json' for name in OUTPUTS} | {'stage20b_r0_status.json', 'STAGE20B_R0_REPORT.md'}
    checks['required_audits_present_and_runtime_pass'] = expected_audits <= names and all(
        load(server / (name + '.json'))['status'] == 'PASS'
        for name in OUTPUTS if name not in ('formal_execution_plan', 'formal_launcher_manifest'))
    checks['local_stage20a_and_authoritative_p0_frozen'] = bool(
        verify_ledger(stage20a, 'frozen_artifact_manifest.json', STAGE20A_MANIFEST_SHA)
        and verify_ledger(p0, 'p0_frozen_artifact_manifest.json', P0_MANIFEST_SHA))
    checks['configuration_and_p0_script_frozen'] = (
        sha(repo / 'configs/plastic_bomo_stage20b_p0.json') == PROTOCOL_SHA
        and sha(repo / 'tools/preflight_plastic_bomo_stage20b_p0.py') == P0_SCRIPT_SHA)
    checks['actual_server_implementation_matches_local_code'] = all(
        sha(repo / 'tools' / name) == digest for name, digest in status['implementation_sha256'].items())
    checks['exact_three_executed_implementation_files'] = set(status['implementation_sha256']) == {
        'qualify_plastic_bomo_stage20b_r0.py', 'stage20b_runtime_contract.py', 'stage20b_detector_controller.py'}

    binding = load(server / 'frozen_binding_recheck.json')
    runtime = load(server / 'runtime_environment.json')
    p0_runtime = load(p0 / 'stage20b_runtime_binding.json')['environment']
    expected_sources = {name: row['sha256'] for name, row in p0_runtime['source_files'].items()}
    actual_sources = {row['file']: row['sha256'] for row in binding['runtime_source_files']}
    checks['runtime_versions_and_sources_match_server_p0'] = (
        binding['package_versions'] == runtime['package_versions'] == p0_runtime['versions']
        and actual_sources == expected_sources
        and runtime['source_hashes'] == binding['runtime_source_files'])
    checks['locally_available_frozen_repo_source_bytes'] = all(
        sha(repo / name) == digest for name, digest in expected_sources.items() if not name.startswith('ultralytics/'))
    checks['recorded_base_and_detector_bytes_match_p0'] = (
        {x['file']: x['sha256'] for x in binding['sd2_files']} ==
        {k: v['sha256'] for k, v in load(p0 / 'sd2_base_weight_binding.json')['files'].items()}
        and binding['detector_sha256'] == cfg['detector_weight_sha256'])
    checks['pre_and_post_probe_frozen_binding_recheck'] = (
        binding['post_probe_binding_recheck_identical'] and binding['protocol_sha256'] == PROTOCOL_SHA
        and binding['p0_script_sha256'] == P0_SCRIPT_SHA and binding['p0_manifest_sha256'] == P0_MANIFEST_SHA
        and binding['TRAIN_assets_verified'] == 666 and binding['final_eval_content_read'] == 0
        and all(x['expected_sha256'] == x['actual_sha256'] for rows in binding['ledgers'].values() for x in rows))
    checks['gpu_deterministic_settings_and_no_package_changes'] = (
        runtime['device'].startswith('cuda:') and runtime['CUDA_version'] is not None
        and runtime['deterministic_algorithms'] and not runtime['TF32'] and not runtime['cudnn_benchmark']
        and runtime['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and runtime['package_change_count'] == 0
        and runtime['final_eval_content_guard']['exact_image_label_paths_denied'] == 96
        and runtime == load(server / 'stage20b_runtime_environment.json'))

    model = load(server / 'generator_model_load_audit.json')
    train = load(p0 / 'stage20b_generator_training_protocol.json')
    optimizer = model['optimizer_configuration']
    checks['generator_model_and_optimizer_master_state'] = (
        all(model[k] for k in ('unet_all_trainable', 'unet_all_fp32_master_parameters', 'vae_frozen',
                              'text_encoder_frozen', 'gradient_checkpointing', 'optimizer_master_parameters_fp32'))
        and model['unet_in_channels'] == 9 and model['optimizer_steps'] == 0
        and model['loaded_components'] == ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
        and (optimizer['lr'], optimizer['betas'], optimizer['eps'], optimizer['weight_decay']) ==
            (train['learning_rate'], [train['adam_beta1'], train['adam_beta2']], train['adam_epsilon'], train['adam_weight_decay']))
    allow = {x['annotation_id']: x for x in load(p0 / 'generator_train_allowlist.json')['inputs']}
    fb = load(server / 'generator_forward_backward_audit.json')
    class_rows = fb['classes']
    checks['both_generator_losses_gradients_and_parameter_invariance'] = (
        len(class_rows) == 2 and {x['class'] for x in class_rows} == {'flash', 'black'} and fb['optimizer_steps'] == 0
        and all(x['pass'] and finite(x['loss']) and valid_gradients(x['gradients'])
                and x['frozen_component_gradients_all_None'] and x['parameter_bytes_unchanged']
                and x['parameter_hashes_before'] == x['parameter_hashes_after']
                and set(x['parameter_hashes_before']) == {'unet', 'vae', 'text_encoder'}
                and x['backward_loss_normalizer'] == 4 for x in class_rows))
    checks['both_generator_probes_are_frozen_train_annotations'] = all(
        x['annotation_id'] in allow and allow[x['annotation_id']]['split'] == 'train'
        and allow[x['annotation_id']]['class'] == x['class']
        and allow[x['annotation_id']]['sha256'] == x['source_sha256']
        and x['prompt'] == train['prompts'][x['class']] for x in class_rows)
    budget = load(server / 'generator_budget_dryrun.json')
    checks['2000_successful_update_simulations_independently_recomputed'] = (
        budget['protocol'] == train and budget['optimizer_steps'] == 0
        and budget['no_skip_scenario'] == generator_budget_dryrun(train, {'flash': 75, 'black': 97})
        and budget['injected_metadata_only_skip_scenario'] == generator_budget_dryrun(train, {'flash': 75, 'black': 97}, (1, 7, 31, 1999)))

    det = load(server / 'generator_determinism_probe.json')
    rng = load(server / 'generator_rng_probe.json')
    slots = {x['slot_id']: x for x in load(p0 / 'stage20b_generation_slots.json')['slots']}
    banks = {x['slot_id']: x['banks'] for x in load(p0 / 'stage20b_rng_manifest.json')['records']}
    probe_checks, probe_details = [], []
    from PIL import Image
    for row in det['records']:
        runs = row['runs']
        hashes = []
        for run in runs:
            path = (server / run['probe_image']).resolve()
            if not path.is_relative_to((server / 'runtime_probe').resolve()):
                raise ValueError('R0_PROBE_PATH_OUTSIDE_RUNTIME_PROBE')
            with Image.open(path) as im:
                hashes.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
                probe_checks.append(im.size == (512, 512) and im.mode == 'RGB' and im.format == 'PNG'
                    and hashes[-1] == run['generated_patch_RGB_sha256'] and sha(path) == run['probe_png_sha256'])
        probe_checks.append(len(runs) == 2 and hashes[0] == hashes[1] and row['two_same_seed_runs_bit_identical'])
        for key in ('before_state_sha256', 'after_state_sha256', 'initial_noise_sha256'):
            probe_checks.append(runs[0][key] == runs[1][key])
        probe_details.append({'class': row['class'], 'slot_id': row['slot_id'], 'decoded_RGB_sha256': hashes[0]})
    checks['four_downloaded_probe_pngs_bit_identical_per_class'] = (
        len(det['records']) == 2 and {x['class'] for x in det['records']} == {'flash', 'black'}
        and all(probe_checks) and det['NOT_FORMAL_SYNTHETIC'] and not det['formal_class_checkpoint_tested'])
    independent = len(rng['records']) == 2 and {x['class'] for x in rng['records']} == {'flash', 'black'}
    for row in rng['records']:
        sid, runs = row['slot_id'], row['runs']
        a0, a1, b = runs
        independent &= (sid in slots and slots[sid]['class'] == row['class']
            and [x['bank'] for x in runs] == ['A', 'A', 'B'] and row['A_B_initial_noise_different']
            and b['noise_only'] and not a0['noise_only'] and not a1['noise_only']
            and all(x['sample_seed'] == banks[sid][x['bank']]['sample_seed']
                    and x['non_rng_metadata_sha256'] == banks[sid][x['bank']]['non_rng_metadata_sha256']
                    and x['CPU_generator'] and x['shape'] == [1, 4, 64, 64] for x in runs)
            and a0['initial_noise_sha256'] != b['initial_noise_sha256']
            and a0['before_state_sha256'] != b['before_state_sha256']
            and a0['after_noise_state_sha256'] != b['after_noise_state_sha256'])
    checks['A_B_noise_independence_and_frozen_rng_metadata'] = bool(independent) and not rng['final_A_B_image_difference_required']
    checks['determinism_and_rng_audits_agree'] = all(
        row['runs'] == next(x['runs'][:2] for x in rng['records'] if x['slot_id'] == row['slot_id']) for row in det['records'])

    detector = load(server / 'detector_model_load_audit.json')
    checks['detector_load_and_exact_initialization'] = (
        detector['pretrained_sha256'] == cfg['detector_weight_sha256'] and detector['ultralytics_version'] == '8.4.145'
        and detector['nc'] == 2 and detector['names'] == {'0': 'flash', '1': 'black'}
        and detector['head_initialization_seed'] == 42 and detector['FP32_parameters']
        and detector['optimizer'] == 'AdamW' and detector['optimizer_steps'] == 0 and not detector['automatic_download_allowed'])
    dfb = load(server / 'detector_forward_backward_audit.json')['records']
    checks['detector_native_losses_gradients_and_parameter_invariance'] = len(dfb) == 1 and all(
        finite(x['loss']) and set(x['native_loss_items']) == {'box_loss', 'cls_loss', 'dfl_loss'}
        and all(finite(v) for v in x['native_loss_items'].values()) and valid_gradients(x['gradients'])
        and x['parameter_bytes_unchanged'] and x['parameter_hashes_before'] == x['parameter_hashes_after']
        and x['parameter_hashes_before'] == detector['training_model_parameter_sha256']
        and x['backward_normalizer'] == 64 and x['optimizer_steps'] == 0 for x in dfb)
    order = load(p0 / 'stage20b_training_order_spec.json')
    accumulation = load(server / 'detector_accumulation_dryrun.json')
    expected_accumulation = accumulation_dryrun(cfg, order)
    checks['all600_group_attempts_and_32400_draws_independently_recomputed'] = all(
        accumulation[k] == v for k, v in expected_accumulation.items())
    checks['real_GPU_autograd64_and24_mean_normalization'] = accumulation['GPU_autograd_normalization_groups'] == [
        {'draws': n, 'normalizer': n, 'gradient': (n + 1) / 2, 'expected_mean_gradient': (n + 1) / 2} for n in (64, 24)]
    controller = Stage20BDetectorController(cfg, order)
    lr_rows = accumulation['native_optimizer_group_LR_probes']
    group_config = detector['optimizer_groups']
    checks['native_AdamW_groups_and_warmup_LR'] = (
        {g['param_group'] for g in group_config} == {'weight', 'bn', 'bias'}
        and all(g['betas'] == [.9, .999] and g['lr'] == g['initial_lr'] == .001
                and g['weight_decay'] == (.0005 if g['param_group'] == 'weight' else 0.) for g in group_config)
        and len(lr_rows) == 6 and all(x['learning_rates'] == controller.learning_rates(x['epoch'], x['position'], group_config)
            and x['group_names'] == [g['param_group'] for g in group_config]
            and x['AdamW_betas_unchanged'] == [[.9, .999]] * 3 for x in lr_rows))
    aug = load(server / 'detector_augmentation_rng_audit.json')
    positions = [(0, 0), (0, 63), (0, 64), (0, 215), (149, 215)]
    checks['five_native_augmentation_seed_probes_recomputed_and_repeatable'] = (
        [(x['epoch'], x['position']) for x in aug['records']] == positions
        and aug['mosaic_mixup_copy_paste_cutmix'] == [0.] * 4 and aug['per_draw_fresh_seed']
        and aug['official_final_eval_inputs'] == 0 and all(
            x['seed_uint64'] == x['python_torch_seed'] == augmentation_seed(cfg['protocol'], x['epoch'], x['position'])
            and x['numpy_seed'] == x['seed_uint64'] % 2**32 and x['cv2_seed'] == x['seed_uint64'] % (2**31 - 1)
            and x['repeat_identical'] and x['repeated_augmentation_tensor_hashes'][0] == x['repeated_augmentation_tensor_hashes'][1]
            for x in aug['records']))
    noeval = load(server / 'detector_no_final_eval_audit.json')
    checks['normal_and_exception_exit_without_automatic_validation'] = (
        noeval['native_final_eval_validate_YOLO_val_mock_calls'] == 0 and noeval['exception_exit_cleanup_calls'] == 2
        and noeval['normal_exit_counts'] == {'epochs': 150, 'draws': 32400, 'scheduled_optimizer_attempts': 600}
        and not noeval['native_training_loop_invoked'] and noeval['early_stopper_not_called']
        and noeval['no_val_dataset_or_validator_constructed'])
    checks['no_formal_steps_training_sampling_eval_or_module_development'] = all(status[k] == 0 for k in (
        'optimizer_step_count', 'forbidden_optimizer_step_calls', 'forbidden_final_eval_open_attempts',
        'generator_formal_training_count', 'formal_synthetic_generation_count', 'generator_training_count',
        'synthetic_generation_count', 'detector_training_count', 'final_eval_forward_count',
        'official_final_eval_forward', 'final_eval_image_or_label_content_read', 'official_final_eval_content_read',
        'DeepPCB', 'BootstrapGuard', 'TDCRG_implementation'))
    checks['formal_plan_only_no_launcher_or_autostart'] = (
        load(server / 'formal_execution_plan.json') == formal_plan()
        and load(server / 'formal_launcher_manifest.json')['status'] == 'NOT_CREATED_OR_STARTED'
        and not load(server / 'formal_launcher_manifest.json')['auto_start'] and not status['formal_auto_start']
        and not status['TDCRG_DEVELOPMENT_AUTHORIZED'])
    checks['reported_runtime_pass_not_utility_claim'] = (
        status['status'] == 'STAGE20B_RUNTIME_QUALIFICATION_PASS' and status['STAGE20B_FORMAL_EXECUTION_READY']
        and status['runtime_probe_generated_images'] == 4 and not status['Stage20A_modified'] and not status['Stage20B_P0_modified'])
    # Check copied TRAIN evidence without opening server/original dataset paths or loading unsafe train.cache.
    checks['copied_probe_train_inputs_match_allowlist'] = all(
        sha(server / 'runtime_probe/train_batch_inputs/images/train' / (x['annotation_id'] + Path(allow[x['annotation_id']]['path']).suffix)) == x['source_sha256']
        and sha(server / 'runtime_probe/train_batch_inputs/labels/train' / (x['annotation_id'] + '.txt')) == allow[x['annotation_id']]['full_label_sha256']
        for x in class_rows)
    details.update(generator_probes=[{k: x[k] for k in ('class', 'annotation_id', 'loss', 'gradients')} for x in class_rows],
        decoded_probe_hashes=probe_details, detector_loss=dfb[0]['loss'], detector_native_losses=dfb[0]['native_loss_items'],
        detector_gradients=dfb[0]['gradients'], gpu=runtime['device_name'], CUDA=runtime['CUDA_version'])
    ok = all(checks.values())
    return {'status': 'STAGE20B_R0_SERVER_REVIEW_PASS' if ok else 'STAGE20B_R0_SERVER_REVIEW_FAILED',
        'server_run': server.name, 'server_manifest_sha256': artifacts['manifest_sha256'],
        'checks': checks, 'checks_passed': sum(checks.values()), 'checks_total': len(checks),
        'downloaded_artifacts': artifacts, 'details': details,
        'formal_status': status['status'] if ok else 'REVIEW_NOT_ACCEPTED',
        'STAGE20B_FORMAL_EXECUTION_READY': bool(ok), 'TDCRG_DEVELOPMENT_AUTHORIZED': False,
        'formal_launcher_created': False, 'formal_auto_start': False,
        'original_server_results_modified': False, 'original_dataset_content_read': 0,
        'local_GPU_rerun': False, 'review_tool_sha256': sha(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--p0', type=Path, default=ROOT / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115')
    p.add_argument('--stage20a', type=Path, default=ROOT / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('R0_REVIEW_OUTPUT_EXISTS_NO_OVERWRITE: ' + str(a.output))
    result = review(ROOT, a.server.resolve(), a.p0.resolve(), a.stage20a.resolve())
    a.output.mkdir(parents=True)
    save(a.output / 'server_review_audit.json', result)
    d = result['details']
    lines = [f"# Stage20B-R0 independent server review\n\nStatus: `{result['status']}`. Accepted formal status: `{result['formal_status']}`.",
        f"\nServer run: `{result['server_run']}`. {result['checks_passed']}/{result['checks_total']} independent checks passed; {result['downloaded_artifacts']['count']} artifact hashes verified.",
        f"\nGPU: {d['gpu']}; CUDA {d['CUDA']}. The downloaded base-pipeline PNGs were decoded locally for RGB-hash verification only; no quality selection was performed.",
        '\n| Probe | Finite loss | Connected gradients | Nonzero gradients |', '|---|---:|---:|---:|']
    for x in d['generator_probes']:
        g = x['gradients']
        lines.append(f"| SD2 {x['class']} | {x['loss']:.9g} | {g['non_none_gradients']}/{g['parameters']} | {g['nonzero_gradients']} |")
    g = d['detector_gradients']
    lines.append(f"| YOLO11s | {d['detector_loss']:.9g} | {g['non_none_gradients']}/{g['parameters']} | {g['nonzero_gradients']} |")
    lines.extend(['\nAll parameter hash maps before/after the backward probes agree. Frozen VAE/text encoder gradients are None. Flash/Black same-seed decoded512 RGB hashes agree independently; A/B noise/state hashes differ with the same frozen metadata/seeds.',
        '\nThe detector controller has32400 scheduled draws and600 metadata-only optimizer attempts with64/64/64/24 groups. Actual GPU autograd checks mean normalization for64 and24. Five native augmentation probes repeat exactly. No automatic final_eval/validator calls occur on either normal or injected exceptional exit.',
        '\nOptimizer steps, formal generator training, formal synthetic generation, detector training and final_eval content/forwards remain0. Four base probe PNGs are NOT_FORMAL_SYNTHETIC. No executable formal launcher was created or run. Stage20A/P0 and historical results are unchanged.',
        '\nScope limits: this is runtime qualification, not synthetic headroom or downstream utility. Unscaled fp16-autocast backward demonstrates finite connected gradients; many generator gradients are zero in this probe, and neither formal GradScaler gradient distributions nor2000-step training stability are certified. Future fine-tuned checkpoint bytes and their determinism are untested. The full150-epoch detector optimizer trajectory is untested. Server package/weight/input identities are reviewed from hash-bound server records, not reloaded locally; these records are not digital signatures or independent execution attestations. The file-open guard is not a whole-system sandbox. Physical source completeness remains false; rectangular masks are weak bbox geometry.',
        '\nFormal execution ready means the R0 runtime gate passed, not that launchers or trained checkpoints exist. Further formal implementation/execution requires an explicit next instruction. TDCRG, BootstrapGuard and DeepPCB remain untouched. No P0/GPU probes were rerun in this review.'])
    (a.output / 'STAGE20B_R0_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    save(a.output / 'review_artifact_manifest.json', {'sha256': {x.name: sha(x) for x in a.output.iterdir() if x.is_file()}})
    print(result['status'], f"{result['checks_passed']}/{result['checks_total']}")
    if not all(result['checks'].values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
