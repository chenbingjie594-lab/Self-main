"""CPU/unit/static tests only. NOT a claim of full formal GPU execution."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import stage20b_formal_common as common
import audit_plastic_bomo_stage20b as audit
import evaluate_plastic_bomo_stage20b as evaluator
import run_plastic_bomo_stage20b_formal as orchestrator

P0 = ROOT / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115'
A = ROOT / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol'
R0 = ROOT / 'results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254'


def metrics(rr=.4, banks=(.41, .42, .43), flash=(.41, .42, .43), black=(.41, .42, .43)):
    rows = {}
    for key, val, f, b in zip(('RR', 'RawA', 'RawB', 'RawC'), (rr, *banks), (.4, *flash), (.4, *black)):
        rows[key] = {'status': 'COMPLETE', 'map5095': val, 'per_class': {'flash': {'ap5095': f}, 'black': {'ap5095': b}}}
    return rows


class FormalTests(unittest.TestCase):
    def test_R0_manifest_and_code_exactly_unchanged(self):
        self.assertTrue(common.verify_ledger(R0, 'r0_artifact_manifest.json', common.R0_MANIFEST_SHA))
        for name, digest in common.R0_CODE.items():
            self.assertEqual(common.sha(ROOT / name), digest)

    def test_all_new_entrypoints_parse_and_have_lazy_GPU_imports(self):
        for name in common.FORMAL_FILES:
            tree = ast.parse((ROOT / 'tools' / name).read_text(encoding='utf-8'))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.split('.')[0] in ('torch', 'ultralytics', 'diffusers') for n in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0], ('torch', 'ultralytics', 'diffusers'))

    def test_states_no_skipping_or_resume(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            common.save(run / 'stage20b_status.json', common.initial_status())
            with self.assertRaises(common.Stop):
                common.transition(run, 'INIT', 'BLACK_GENERATOR_TRAINED')
            common.transition(run, 'INIT', 'FLASH_GENERATOR_TRAINED')
            with self.assertRaises(common.Stop):
                common.transition(run, 'INIT', 'FLASH_GENERATOR_TRAINED')
            common.update_status(run, status='STOPPED_TEST')
            with self.assertRaises(common.Stop):
                common.transition(run, 'FLASH_GENERATOR_TRAINED', 'BLACK_GENERATOR_TRAINED')

    def test_tree_detects_changes_and_added_files(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            path = run / 'ckpt'
            path.mkdir()
            (path / 'weights').write_bytes(b'frozen')
            row = common.tree_binding(path, run)
            self.assertEqual(common.verify_tree(run, row), path)
            (path / 'extra').write_bytes(b'new')
            with self.assertRaises(common.Stop):
                common.verify_tree(run, row)

    def test_verified_copy_preserves_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source'
            source.write_bytes(b'image bytes')
            dest = root / 'copies/dest'
            common.verified_copy(source, dest, common.sha(source))
            self.assertEqual(dest.read_bytes(), source.read_bytes())
            with self.assertRaises(common.Stop):
                common.verified_copy(source, dest, common.sha(source))

    def test_frozen_slots_labels_and_pairing_geometry(self):
        slots = common.load(P0 / 'stage20b_generation_slots.json')['slots']
        rng = {x['slot_id']: x['banks'] for x in common.load(P0 / 'stage20b_rng_manifest.json')['records']}
        for s in slots:
            label = common.synthetic_label(s).split()
            self.assertEqual(int(label[0]), {'flash': 0, 'black': 1}[s['class']])
            self.assertEqual([float(v) for v in label[1:]], s['bbox'])
            for bank in ('A', 'B', 'C'):
                self.assertEqual(common.htext(common.canonical(s)), rng[s['slot_id']][bank]['non_rng_metadata_sha256'])
        changed = deepcopy(slots[0])
        changed['bbox'][2] = -1
        with self.assertRaises(common.Stop):
            common.validate_bbox(changed)

    def test_real_asset_population_resolves_metadata_not_historical_val(self):
        train = common.real_assets(A, P0, 'train')
        final = common.real_assets(A, P0, 'final_eval')
        self.assertEqual(len(train), 136)
        self.assertEqual(len(final), 48)
        self.assertTrue(any('/images/val/' in x['image_path'].replace('\\', '/') for x in train))
        self.assertFalse({x['image_path'] for x in train} & {x['image_path'] for x in final})
        self.assertFalse({x['source_group'] for x in train} & {x['source_group'] for x in final})
        roles = common.load(P0 / 'stage20b_training_order_spec.json')['epochs'][0]['role_sequence']
        extra = ['extra_' + r['slot_id'] for r in common.load(P0 / 'stage20b_realrepeat_manifest.json')['records']]
        self.assertEqual(set(roles), {x['role_id'] for x in train} | set(extra))

    def test_frozen_success_gates_and_sample_std(self):
        result = common.summarize_metrics(metrics(), common.load(A / 'future_baseline_gate.json'))
        self.assertEqual(result['status'], 'GENERATOR_BASELINE_HEADROOM_CONFIRMED')
        self.assertEqual(result['wins_vs_RR'], 3)
        self.assertAlmostEqual(result['mean_delta_pp'], 2)
        self.assertAlmostEqual(result['std_delta_pp'], 1)
        self.assertAlmostEqual(result['generation_bank_std_pp'], 1)
        self.assertAlmostEqual(result['generation_bank_range_pp'], 2)

    def test_ties_not_wins_and_class_gate_not_overall_only(self):
        gate = common.load(A / 'future_baseline_gate.json')
        tied = common.summarize_metrics(metrics(banks=(.4, .4, .5)), gate)
        self.assertEqual(tied['wins_vs_RR'], 1)
        self.assertFalse(tied['Gate_B'])
        collapsed = common.summarize_metrics(metrics(black=(.3, .3, .3)), gate)
        self.assertFalse(collapsed['Gate_C'])
        self.assertFalse(collapsed['TDCRG_DEVELOPMENT_AUTHORIZED'])

    def test_nonfinite_metrics_rejected_not_replaced(self):
        with self.assertRaises(common.Stop):
            common.summarize_metrics(metrics(banks=(math.nan, .5, .5)), common.load(A / 'future_baseline_gate.json'))

    def test_native_metric_class_index_mapping(self):
        box = SimpleNamespace(ap_class_index=[1, 0], p=[.6, .7], r=[.8, .9], ap50=[.5, .4],
                              ap=[.3, .2], all_ap=[[.3], [.2]], map=.25, map50=.45, mp=.65, mr=.85)
        result = evaluator.metrics_record(SimpleNamespace(box=box))
        self.assertEqual(result['per_class']['flash']['ap5095'], .2)
        self.assertEqual(result['per_class']['black']['recall'], .8)

    def test_missing_metric_class_is_not_silently_imputed(self):
        box = SimpleNamespace(ap_class_index=[0], p=[.7], r=[.9], ap50=[.4], ap=[.2], all_ap=[[.2]])
        with self.assertRaises(common.Stop):
            evaluator.metrics_record(SimpleNamespace(box=box))

    def test_no_native_full_training_or_validation_in_training_scripts(self):
        for name in ('train_plastic_bomo_stage20b_generator.py', 'train_plastic_bomo_stage20b_detector.py'):
            source = (ROOT / 'tools' / name).read_text(encoding='utf-8')
            self.assertNotIn('.val(', source)
            self.assertNotIn('.final_eval(', source)
            self.assertNotIn('.save_model(', source)
        source = (ROOT / 'tools/train_plastic_bomo_stage20b_detector.py').read_text(encoding='utf-8')
        self.assertNotIn('model.train(', source)
        self.assertIn('controller.run(', source)
        self.assertIn('controller.backward(objective, length, scaler=scaler)', source)
        self.assertNotIn('__class__=', source)

    def test_final_eval_barrier_precedes_copy_and_val(self):
        source = (ROOT / 'tools/evaluate_plastic_bomo_stage20b.py').read_text(encoding='utf-8')
        self.assertLess(source.index("context(a.run, ('ALL_DETECTORS_FROZEN',))"), source.index('verified_copy(row['))
        self.assertLess(source.index("binding['status'] == 'FROZEN'"), source.index('model.val('))
        self.assertLess(source.index('final_eval_one_round_claim.json'), source.index('model.val('))

    def test_orchestrator_stops_on_first_child_failure_no_next_launch(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'formal_test'
            calls = []
            def child(command, **kwargs):
                calls.append(command)
                common.update_status(run, reason='FLASH_GENERATOR_TRAINING_FAILED', detail='injected CPU-only failure')
                return SimpleNamespace(pid=12345, wait=lambda: 1)
            with patch.object(sys, 'argv', ['run', '--output', str(run)]), patch.object(orchestrator, 'create_binding', return_value={'status': 'PASS'}), \
                 patch.object(orchestrator.subprocess, 'Popen', side_effect=child), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    orchestrator.main()
            self.assertEqual(len(calls), 1)
            s = common.load(run / 'stage20b_status.json')
            self.assertEqual(s['status'], 'STOPPED_FLASH_GENERATOR_TRAINING_FAILED')
            self.assertEqual(s['state'], 'INIT')
            self.assertEqual(s['official_final_eval_rounds'], 0)
            self.assertIsNone(s['RR_map5095'])
            self.assertFalse((run.parent / 'authoritative_formal_run.json').exists())

    def test_output_overwrite_refused_before_binding(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(sys, 'argv', ['run', '--output', d]), patch.object(orchestrator, 'create_binding') as bind:
                with self.assertRaises(common.Stop):
                    orchestrator.main()
                bind.assert_not_called()

    def test_context_cannot_enter_final_eval_before_four_detector_barrier(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            common.save(run / 'formal_protocol_binding.json', {'status': 'PASS'})
            common.save(run / 'stage20b_status.json', common.initial_status())
            with self.assertRaises(common.Stop):
                common.context(run, ('ALL_DETECTORS_FROZEN',))

    def test_full_orchestrator_order_CPU_metadata_simulation_only(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'formal_simulation'
            calls = []
            expected = [
                ('INIT', 'class_name', 'flash'), ('FLASH_GENERATOR_TRAINED', 'class_name', 'black'),
                ('BLACK_GENERATOR_TRAINED', 'action', 'freeze_generators'),
                ('GENERATOR_CHECKPOINTS_FROZEN', 'bank', 'sanity'),
                ('GENERATOR_CHECKPOINTS_FROZEN', 'bank', 'A'), ('BANK_A_GENERATED', 'bank', 'B'),
                ('BANK_B_GENERATED', 'bank', 'C'), ('BANK_C_GENERATED', 'action', 'validate_banks'),
                ('SYNTHETIC_BANKS_VALIDATED', 'arm', 'RR'), ('DETECTOR_RR_TRAINED', 'arm', 'A'),
                ('DETECTOR_A_TRAINED', 'arm', 'B'), ('DETECTOR_B_TRAINED', 'arm', 'C'),
                ('DETECTOR_C_TRAINED', 'action', 'freeze_detectors'),
                ('ALL_DETECTORS_FROZEN', None, None), ('FINAL_EVAL_COMPLETE', 'action', 'finish')]
            def child(command, **kwargs):
                state, key, val = expected[len(calls)]
                self.assertEqual(common.load(run / 'stage20b_status.json')['state'], state)
                if key:
                    self.assertEqual(command[-2:], ['--' + key, val])
                else:
                    self.assertEqual(Path(command[1]).name, 'evaluate_plastic_bomo_stage20b.py')
                calls.append(command)
                return SimpleNamespace(pid=12345, wait=lambda: 0)
            with patch.object(sys, 'argv', ['run', '--output', str(run)]), patch.object(orchestrator, 'create_binding', return_value={'status': 'PASS'}), \
                 patch.object(orchestrator.subprocess, 'Popen', side_effect=child), redirect_stdout(io.StringIO()):
                orchestrator.main()
            self.assertEqual(len(calls), 15)
            s = common.load(run / 'stage20b_status.json')
            self.assertEqual([x['state'] for x in s['state_history']], list(common.STATES))
            self.assertFalse(s['full_experiment_complete'])  # simulated children did NOT execute formal audits
            self.assertFalse((run.parent / 'authoritative_formal_run.json').exists())

    def test_binding_failure_never_claims_generation_or_formal_complete(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'bad_binding'
            with patch.object(sys, 'argv', ['run', '--output', str(run)]), \
                 patch.object(orchestrator, 'create_binding', side_effect=common.Stop('FORMAL_FROZEN_BINDING_CHANGED', 'injected')), \
                 patch.object(orchestrator.subprocess, 'Popen') as launch, redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    orchestrator.main()
                launch.assert_not_called()
            s = common.load(run / 'stage20b_status.json')
            self.assertEqual(s['generator_training_count'], 0)
            self.assertEqual(s['synthetic_generation_count'], 0)
            self.assertEqual(s['detector_training_count'], 0)
            self.assertEqual(s['status'], 'STOPPED_FORMAL_FROZEN_BINDING_CHANGED')

    def test_compute_freeze_allows_AMP_variation_but_not_wrong_tail(self):
        order = common.load(P0 / 'stage20b_training_order_spec.json')
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            for arm, successes in zip(('RR', 'A', 'B', 'C'), (600, 599, 598, 599)):
                folder = run / ('detector_' + arm)
                folder.mkdir()
                checkpoint = folder / 'last.pt'
                checkpoint.write_bytes(b'CPU-only fake checkpoint for compute audit test')
                records = [{'epoch_0based': x['epoch_0based'], 'draws': 216,
                    'actual_role_sequence_sha256': x['sequence_sha256'],
                    'attempts': [{'position': pos, 'normalizer': length} for pos, length in
                                 zip(x['optimizer_attempt_after_positions_0based'], x['accumulation_group_lengths'])]} for x in order['epochs']]
                common.save(run / ('detector_arm_' + arm + '_training.json'), {
                    'status': 'COMPLETE', 'epochs': 150, 'actual_draws': 32400, 'scheduled_draws': 32400,
                    'attempted_optimizer_updates': 600, 'scheduled_optimizer_attempts': 600,
                    'successful_optimizer_updates': successes, 'AMP_skipped_updates': 600 - successes,
                    'official_final_eval_content_read': 0, 'nonfinite_losses': 0, 'gradient_nonfinite_count': 0,
                    'epoch_records': records, 'last_checkpoint': checkpoint.relative_to(run).as_posix(),
                    'last_sha256': common.sha(checkpoint), 'primary_model_representation': 'CPU test fixture',
                    'order_spec_sha256': common.sha(P0 / 'stage20b_training_order_spec.json'),
                    'initial_parameter_binding_sha256': 'same_initial_parameters_fixture'})
            audit.freeze_detectors(run, P0)
            comp = common.load(run / 'detector_compute_audit.json')
            self.assertEqual(comp['AMP_success_count_range'], 2)
            self.assertEqual(comp['extra_training_to_equalize_successes'], 0)
            row = common.load(run / 'detector_arm_A_training.json')
            row['epoch_records'][0]['attempts'][-1]['normalizer'] = 64
            common.save(run / 'detector_arm_A_training.json', row)
            with self.assertRaises(common.Stop):
                audit.freeze_detectors(run, P0)

    def test_finish_rejects_state_chain_without_actual_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            s = common.initial_status()
            s['state'] = 'FINAL_EVAL_COMPLETE'
            s['state_history'] = [{'state': x} for x in common.STATES]
            common.save(run / 'stage20b_status.json', s)
            common.save(run / 'generator_checkpoint_binding.json', {'status': 'NOT_RUN'})
            with self.assertRaises(common.Stop):
                audit.finish(run, P0, A)
            self.assertFalse((run.parent / 'authoritative_formal_run.json').exists())

    def test_immutable_parent_envelope_rejects_changed_run_binding(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            b = {'status': 'PASS', 'device': 'cuda:0'}
            common.save(run / 'formal_protocol_binding.json', b)
            digest = common.sha(run / 'formal_protocol_binding.json')
            common.save(run / 'stage20b_status.json', common.initial_status())
            b['device'] = 'cuda:7'
            common.save(run / 'formal_protocol_binding.json', b)
            with patch.dict(common.os.environ, {'STAGE20B_FORMAL_BINDING_SHA256': digest}):
                with self.assertRaises(common.Stop) as err:
                    common.context(run, ('INIT',))
            self.assertEqual(err.exception.status, 'FORMAL_FROZEN_BINDING_CHANGED')

    def test_structural_bank_check_uses_all_outputs_and_no_quality_filter(self):
        # Tiny RGB fixtures test the file/geometry/paired-ledger path; NOT generated
        # images, inference or a replacement for the formal4096x1024 GPU outputs.
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            fake_p0 = run / 'fixture_p0'
            fake_p0.mkdir()
            slots = deepcopy(common.load(P0 / 'stage20b_generation_slots.json')['slots'])
            rngs = deepcopy(common.load(P0 / 'stage20b_rng_manifest.json')['records'])
            by_rng = {r['slot_id']: r for r in rngs}
            ck = {}
            for cls in ('flash', 'black'):
                folder = run / ('fake_checkpoint_' + cls)
                folder.mkdir()
                (folder / 'test_bytes').write_bytes(b'CPU fake non-model fixture')
                ck[cls] = {'checkpoint': common.tree_binding(folder, run)}
            common.save(run / 'generator_checkpoint_binding.json', {'classes': ck})
            for s in slots:
                s['geometry']['final_canvas_wh'] = [32, 8]
                for r in by_rng[s['slot_id']]['banks'].values():
                    r['non_rng_metadata_sha256'] = common.htext(common.canonical(s))
            common.save(fake_p0 / 'stage20b_generation_slots.json', {'slots': slots})
            common.save(fake_p0 / 'stage20b_rng_manifest.json', {'records': rngs})
            for bank in ('A', 'B', 'C'):
                folder = run / ('synthetic_' + bank)
                folder.mkdir()
                rows = []
                for s in slots:
                    image = folder / (s['slot_id'] + '.png')
                    label = folder / (s['slot_id'] + '.txt')
                    im = Image.new('RGB', (32, 8))  # identical blank fixtures MUST NOT be quality-excluded
                    im.save(image)
                    label.write_text(common.synthetic_label(s), encoding='utf-8', newline='\n')
                    r = by_rng[s['slot_id']]['banks'][bank]
                    rows.append({'slot_id': s['slot_id'], 'class': s['class'], 'bank': bank,
                        'frozen_slot': s, 'non_rng_metadata_sha256': r['non_rng_metadata_sha256'],
                        'sample_seed': r['sample_seed'], 'bank_seed': r['bank_seed'],
                        'checkpoint_tree_sha256': ck[s['class']]['checkpoint']['tree_sha256'],
                        'image_path': image.relative_to(run).as_posix(), 'image_sha256': common.sha(image),
                        'label_path': label.relative_to(run).as_posix(), 'label_sha256': common.sha(label),
                        'decoded_RGB_sha256': common.hashlib.sha256(im.tobytes()).hexdigest()})
                common.save(run / ('formal_bank_' + bank + '_manifest.json'), {'status': 'COMPLETE', 'records': rows})
            audit.validate_banks(run, fake_p0)
            self.assertEqual(common.load(run / 'formal_synthetic_manifest.json')['count'], 240)
            self.assertFalse(common.load(run / 'formal_synthetic_validity.json')['quality_gate'])
            # One missing/changed label stops the WHOLE bank set, not a smaller subset.
            bad = run / 'synthetic_B' / (slots[0]['slot_id'] + '.txt')
            bad.write_bytes(b'changed')
            with self.assertRaises(common.Stop) as err:
                audit.validate_banks(run, fake_p0)
            self.assertEqual(err.exception.status, 'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE')


if __name__ == '__main__':
    unittest.main()
