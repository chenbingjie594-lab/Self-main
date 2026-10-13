"""Frozen Stage20B detector implementation with Stage25 binding/data shell only."""
from collections import Counter
import gc
from pathlib import Path
from types import SimpleNamespace

from stage25_msdf_contract import (BANKS, binding, guard, claim, load, save, sha, require,
    verified_copy, real_rows, metrics_summary, now, update, tree_binding)


def legacy_module(run, module_name):
    """Do NOT invoke old CLI/main. Only its fixed train/controller implementation."""
    import importlib.util
    from stage25_msdf_contract import REPO
    spec = importlib.util.spec_from_file_location('_stage25_' + module_name, REPO / 'tools' / (module_name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    b, cfg = binding(run)
    def context(path, old_states):
        current, _ = binding(path)
        require(load(Path(path) / 'stage25_status.json')['status'] == 'FORMAL_RUNNING', 'STAGE25_PHASE_SEQUENCE_INVALID', 'formal stopped/completed')
        current = {**current, 'protocol': current['controller_cfg'], 'P0_binding': {'detector_sha256': cfg['detector_target']['initialization_sha256']}}
        return current, Path(current['p0_view']), Path(current['stage20a'])
    module.context = context
    module.runtime_guard = lambda path, *args, **kwargs: guard(path, evaluation=kwargs.get('evaluation', False))
    module.real_assets = lambda stage20a, p0, split: real_rows(stage20a, b['real_root'], split)
    def record(path, **values):
        # The legacy RR/A/B/C ordinal is not a Stage25 scientific count.
        if 'detector_training_count' in values:
            values['legacy_detector_ordinal'] = values.pop('detector_training_count')
        update(path, **values)
    module.update_status = record
    return module


def raw_train(run, cls):
    claim(run, 'raw_train_' + cls)
    legacy_module(run, 'train_plastic_bomo_stage20b_generator').train(SimpleNamespace(run=Path(run), class_name=cls))


def detector(run, bank, comparator=False):
    run = Path(run)
    b, _ = binding(run)
    validity = load(run / 'formal_synthetic_validity.json')
    require(validity['status'] == 'PASS' and sha(run / 'formal_synthetic_manifest.json') == validity['manifest_sha256'],
            'STAGE25_BANK_BARRIER_FAILED', '240 frozen inputs')
    if bank in ('B', 'C'):
        previous = load(run / ('detector_arm_' + BANKS[BANKS.index(bank) - 1] + '_training.json'))
        require(previous['status'] == 'COMPLETE', 'STAGE25_DETECTOR_SEQUENCE_INVALID', bank)
    if comparator and bank == 'A':
        require(load(run / 'detector_arm_RR_training.json')['status'] == 'COMPLETE', 'STAGE25_DETECTOR_SEQUENCE_INVALID', 'RR first')
    if not comparator:
        require(bank in BANKS, 'STAGE25_DETECTOR_SEQUENCE_INVALID', 'MSDF has no RR arm')
        # Prove identical head/init rather than retraining RR just to produce this audit.
        init = run / 'detector_initialization_comparison.json'
        if not init.exists():
            reuse = load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')
            source = Path(b['baseline_run']) if reuse['reuse'] else run / 'comparator_rebuild'
            verified_copy(source / init.name, init, sha(source / init.name))
        else:
            require(bank != 'A', 'STAGE25_PHASE_ALREADY_ATTEMPTED', 'MSDF-A init exists')
    claim(run, ('comparator' if comparator else 'msdf') + '_detector_' + bank)
    update(run, **{('comparator' if comparator else 'msdf') + '_detector_' + bank + '_started': True})
    legacy_module(run, 'train_plastic_bomo_stage20b_detector').train(SimpleNamespace(run=run, arm=bank))
    audit = load(run / ('detector_arm_' + bank + '_training.json'))
    require(audit['status'] == 'COMPLETE' and audit['actual_draws'] == 32400 and audit['epochs'] == 150
            and audit['attempted_optimizer_updates'] == 600, 'STAGE25_DETECTOR_BUDGET_INVALID', bank)
    save(run / ('detector_' + bank + '_training_audit.json'), {**audit, 'Stage25_role': ('Raw' if comparator else 'MSDF') + bank,
        'controller_source_unchanged': True, 'successful_updates_not_equalized': True})


def freeze_detectors(run, comparator=False):
    run = Path(run)
    arms = {}
    for name in (('RR', *BANKS) if comparator else BANKS):
        x = load(run / ('detector_arm_' + name + '_training.json'))
        require(x['status'] == 'COMPLETE' and x['epochs'] == 150 and x['actual_draws'] == 32400
                and x['attempted_optimizer_updates'] == 600 and sha(run / x['last_checkpoint']) == x['last_sha256'],
                'STAGE25_EVALUATION_BARRIER_FAILED', name)
        arms[name] = {'path': x['last_checkpoint'], 'sha256': x['last_sha256'], 'epochs': 150,
                      'scheduled_draws': 32400, 'scheduled_optimizer_attempts': 600,
                      'successful_optimizer_updates': x['successful_optimizer_updates'], 'AMP_skipped_updates': x['AMP_skipped_updates']}
    require(not (run / 'detector_checkpoint_binding.json').exists(), 'STAGE25_OUTPUT_EXISTS', 'detector binding')
    save(run / 'detector_checkpoint_binding.json', {'status': 'FROZEN', 'arms': arms, 'all_frozen_before_new_evaluation': True})
    save(run / 'detector_compute_audit.json', {'status': 'PASS', 'arms': arms, 'draws_per_arm': 32400,
        'scheduled_attempts_per_arm': 600, 'successful_AMP_updates_not_compensated': True,
        'epoch150_native_EMA_last_pt': True, 'validation_checkpoint_selection': False})


def evaluation_checkpoints(run):
    b, _ = binding(run)
    frozen = load(Path(run) / 'detector_checkpoint_binding.json')
    require(frozen['status'] == 'FROZEN' and set(frozen['arms']) == set(BANKS), 'STAGE25_EVALUATION_BARRIER_FAILED', 'MSDF3/3 required')
    arms = {'MSDF_' + k: {**v, 'absolute_path': str(Path(run) / v['path'])} for k, v in frozen['arms'].items()}
    reuse = load(Path(b['runtime_dir']) / 'baseline_reuse_audit_server.json')
    if not reuse['reuse']:
        root = Path(run) / 'comparator_rebuild'
        raw = load(root / 'detector_checkpoint_binding.json')
        require(raw['status'] == 'FROZEN' and set(raw['arms']) == {'RR', *BANKS}, 'STAGE25_EVALUATION_BARRIER_FAILED', 'rebuilt4/4 required')
        arms.update({('RR' if k == 'RR' else 'Raw_' + k): {**v, 'absolute_path': str(root / v['path'])} for k, v in raw['arms'].items()})
    for k, v in arms.items():
        require(v['epochs'] == 150 and sha(v['absolute_path']) == v['sha256'], 'STAGE25_EVALUATION_BARRIER_FAILED', k)
    if reuse['reuse']:
        # Recheck historical immutable authority/weights too; do not open any eval content.
        from stage25_msdf_contract import baseline_audit
        check = baseline_audit(SimpleNamespace(baseline_run=Path(b['baseline_run']), p0=Path(b['p0'])), binding(run)[1])
        require(check['reuse'] and check['metrics'] == reuse['metrics'], 'STAGE25_COMPARATOR_BINDING_CHANGED', check.get('errors'))
    return arms, reuse


def evaluate(run):
    run = Path(run)
    b, cfg = binding(run)
    arms, reuse = evaluation_checkpoints(run)
    claim(run, 'unified_final_eval')
    update(run, official_final_eval_rounds=1)
    save(run / 'unified_evaluation_checkpoint_binding.json', {'status': 'FROZEN', 'arms': arms,
        'historical_comparator_reused': reuse['reuse'], 'new_evaluation_arms': len(arms), 'frozen_at': now()})
    audit = {'status': 'RUNNING', 'unified_rounds_started': 1, 'unified_rounds_completed': 0,
        'official_final_eval_image_forwards': 0, 'arms': {}, 'training_final_eval_content_reads': 0,
        'source_images': 0, 'source_labels': 0, 'warmup_uses_final_eval_images': False,
        'historically_observed_final_eval': True, 'independent_untouched_test': False,
        'checkpoint_binding_sha256': sha(run / 'unified_evaluation_checkpoint_binding.json')}
    path = run / 'final_eval_access_audit.json'
    save(path, audit)
    results = {}
    try:
        with guard(run, evaluation=True) as torch:
            import yaml
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionValidator
            from evaluate_plastic_bomo_stage20b import metrics_record
            rows = real_rows(b['stage20a'], b['real_root'], 'final_eval')
            require(len(rows) == 48, 'STAGE25_METRICS_INVALID', 'V2 final_eval48')
            root = run / 'final_eval_inputs'
            root.mkdir()
            for r in rows:
                verified_copy(r['image_path'], root / 'images/final_eval' / (r['node_id'] + Path(r['image_path']).suffix), r['image_sha256'])
                audit['source_images'] += 1
                verified_copy(r['label_path'], root / 'labels/final_eval' / (r['node_id'] + '.txt'), r['label_sha256'])
                audit['source_labels'] += 1
                save(path, audit)
            valdir = str((root / 'images/final_eval').resolve())
            data = {'path': str(root), 'train': None, 'val': valdir, 'final_eval': valdir,
                    'names': {0: 'flash', 1: 'black'}, 'nc': 2, 'channels': 3}
            yaml_path = root / 'data.yaml'
            yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
            expected = {r['node_id'] for r in rows}
            p0 = Path(b['p0_view'])
            native = load(p0 / 'stage20b_detector_resolved_runtime_arguments.json')['arguments']
            ev = load(p0 / 'stage20b_detector_protocol.json')['evaluation']
            val_args = {**native, **{k: v for k, v in ev.items() if k != 'metrics'}}
            val_args.update(data=str(yaml_path), device=b['device'], workers=native['workers'],
                            project=str(run / 'final_eval_outputs'), mode='val', task='detect', exist_ok=False)
            for name, row in arms.items():
                observed, gt = [], []
                class AuditedValidator(DetectionValidator):
                    def update_metrics(self, preds, batch):
                        names = [Path(f).stem for f in batch['im_file']]
                        require(all(n in expected for n in names), 'STAGE25_METRICS_INVALID', 'unexpected evaluation input')
                        observed.extend(names)
                        gt.extend(int(v) for v in batch['cls'].reshape(-1).cpu().tolist())
                        audit['official_final_eval_image_forwards'] += len(names)
                        audit['arms'][name] = {'image_forwards': len(observed), 'node_ids': list(observed)}
                        save(path, audit)
                        super().update_metrics(preds, batch)
                require(sha(row['absolute_path']) == row['sha256'], 'STAGE25_EVALUATION_BARRIER_FAILED', name)
                model = YOLO(row['absolute_path'])
                metric = model.val(validator=AuditedValidator, **{**val_args, 'name': name, 'model': row['absolute_path']})
                require(len(observed) == len(set(observed)) == 48 and set(observed) == expected
                        and Counter(gt) == {0: 33, 1: 27}, 'STAGE25_METRICS_INVALID', 'evaluation population')
                results[name] = {**metrics_record(metric), 'arm': name, 'checkpoint_sha256': row['sha256'],
                    'evaluation_protocol_sha256': sha(p0 / 'stage20b_detector_protocol.json'),
                    'evaluation_image_count': 48, 'target_counts': {'flash': 33, 'black': 27}, 'primary_checkpoint': 'epoch150 last.pt'}
                save(run / (name + '_metrics.json'), results[name])
                del model, metric
                gc.collect(); torch.cuda.empty_cache()
            require(audit['official_final_eval_image_forwards'] == 48 * len(arms), 'STAGE25_METRICS_INVALID', 'one unified round')
            audit.update(status='COMPLETE', unified_rounds_completed=1)
            if reuse['reuse']:
                for k, v in reuse['metrics'].items():
                    name = 'RR' if k == 'RR' else 'Raw_' + k
                    results[name] = v
                    save(run / (name + '_metrics.json'), {**v, 'historical_formal_reuse': True, 'new_evaluation_forward_count': 0})
            summary = metrics_summary({k: results['MSDF_' + k] for k in BANKS}, {k: results['Raw_' + k] for k in BANKS}, results['RR'], cfg)
            summary.update(server_runtime_qualified=True, raw_comparator_reused=reuse['reuse'],
                final_eval_status=cfg['final_eval_status'], new_final_eval_image_forwards=audit['official_final_eval_image_forwards'])
            save(run / 'stage25_tradeoff_summary.json', summary)
            save(run / 'stage25_classwise.json', summary['classwise'])
            save(run / 'msdf_classwise.json', summary['classwise'])
            save(run / 'msdf_vs_raw_metrics.json', {'metrics': {k: {'MSDF': results['MSDF_' + k], 'Raw': results['Raw_' + k],
                'delta_pp': summary['delta_' + k + '_msdf_vs_raw_pp']} for k in BANKS},
                'gates': summary['gates'], 'mean_delta_pp': summary['mean_msdf_vs_raw_pp'], 'wins': summary['msdf_vs_raw_wins']})
            save(run / 'msdf_vs_rr_metrics.json', {'RR': results['RR'], 'MSDF': {k: results['MSDF_' + k] for k in BANKS},
                'mean_delta_pp': summary['mean_msdf_vs_rr_pp'], 'wins': summary['msdf_vs_rr_wins'],
                'MSDF_AUGMENTATION_UTILITY_CONFIRMED': summary['MSDF_AUGMENTATION_UTILITY_CONFIRMED']})
            import statistics
            save(run / 'msdf_bank_variance.json', {'MSDF_sample_SD_pp': summary['MSDF_bank_sample_SD'],
                'Raw_sample_SD_pp': summary['Raw_bank_sample_SD'], 'matched_delta_sample_SD_pp': statistics.stdev(
                    summary['delta_' + k + '_msdf_vs_raw_pp'] for k in BANKS), 'ddof': 1,
                'interpretation': 'three generation banks, not detector-seed uncertainty or independent untouched test'})
            update(run, **summary)
            if not reuse['reuse']:
                update(run / 'comparator_rebuild', status='COMPARATOR_REBUILD_COMPLETE_CONSUMED_BY_PARENT',
                    unified_evaluation_parent=str(run), full_experiment_complete=True)
            report = ('# Stage25 MSDF-V2 Clean Requalification\n\n' + summary['status'] + '\n\n'
                + f'Mean MSDF−Raw: {summary["mean_msdf_vs_raw_pp"]:.6f} pp; strict wins {summary["msdf_vs_raw_wins"]}/3.\n\n'
                + f'Mean MSDF−RR: {summary["mean_msdf_vs_rr_pp"]:.6f} pp; strict wins {summary["msdf_vs_rr_wins"]}/3.\n\n'
                + 'Final_eval was historically observed in Stage20B, not an independent untouched test. No Stage25 result-dependent tuning.\n\n'
                + 'Core/literal historical losses frozen. Runtime/data shell differences D7/D8 are declared in the shared protocol.\n'
                + 'Raw outputs retained240/240; no selector/filter/reroll. Epoch150 native EMA last.pt, no validation selection.\n')
            (run / 'STAGE25_REPORT.md').write_text(report, encoding='utf-8', newline='\n')
            print(__import__('json').dumps(summary, indent=2), flush=True)
    finally:
        if audit['status'] != 'COMPLETE':
            audit['status'] = 'FAILED_PARTIAL_NO_RETRY'
        save(path, audit)
    ledger = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob('*') if p.is_file()
              and p.suffix in ('.json', '.md') and 'train_inputs' not in p.parts and 'final_eval_inputs' not in p.parts}
    save(run / 'stage25_result_artifact_manifest.json', {'sha256': ledger,
         'weights_and_images_bound_by_separate_checkpoint_and_bank_ledgers': True})
