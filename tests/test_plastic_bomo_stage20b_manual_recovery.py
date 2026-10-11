"""CPU fixtures only; no claim of server recovery, sampling or detector training."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import stage20b_formal_common as common
import audit_plastic_bomo_stage20b as audit
import recover_plastic_bomo_stage20b_arm_c as recovery

P0 = ROOT / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115'


def write(root, name, data=b'CPU non-model fixture'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return common.sha(path)


def fixture(root):
    source = root / 'source_stopped_C'
    source.mkdir()
    status = common.initial_status()
    status.update(status='STOPPED_TRAIN_PLASTIC_BOMO_STAGE20B_DETECTOR_FAILED', state='DETECTOR_B_TRAINED',
                  detail="{'exit_code': -9, 'log': 'train_plastic_bomo_stage20b_detector_arm_C.log'}",
                  state_history=[{'state': s, 'at': 'CPU fixture'} for s in common.STATES[:11]])
    binding = {'status': 'PASS', 'p0': str(P0), 'pid': 12345, 'result_directory': str(source),
               'formal_code_sha256': {f'tools/{n}': common.sha(ROOT / 'tools' / n) for n in common.FORMAL_FILES},
               'R0_code_sha256': common.R0_CODE}
    common.save(source / 'formal_protocol_binding.json', binding)
    for name in common.ARTIFACTS:
        common.save(source / (name + '.json'), {'status': 'NOT_RUN'})
    generators = {}
    for cls in ('flash', 'black'):
        folder = source / 'generator_checkpoints' / cls / 'final_step2000'
        write(folder, 'fake_weights')
        ck = common.tree_binding(folder, source)
        generators[cls] = {'checkpoint': ck, 'successful_updates': 2000}
        common.save(source / ('generator_training_' + cls + '.json'), {'status': 'COMPLETE',
            'checkpoint': ck, 'successful_updates': 2000, 'frozen_components_unchanged': True, 'last_step_legal': True})
        status['generator_' + cls + '_successful_updates'] = 2000
    common.save(source / 'generator_checkpoint_binding.json', {'status': 'FROZEN', 'classes': generators})
    sanity_records = []
    for cls in ('flash', 'black'):
        samples = []
        for repeat in range(2):
            path = f'checkpoint_sanity/{cls}_{repeat}.png'
            samples.append({'image_path': path, 'image_sha256': write(source, path)})
        sanity_records.append({'class': cls, 'runs': samples, 'same_seed_RGB_identical': True})
    common.save(source / 'trained_checkpoint_determinism_audit.json', {'status': 'PASS',
        'records': sanity_records, 'sampling_count': 4})
    slots = common.load(P0 / 'stage20b_generation_slots.json')['slots']
    rng = {x['slot_id']: x['banks'] for x in common.load(P0 / 'stage20b_rng_manifest.json')['records']}
    combined = []
    for bank in ('A', 'B', 'C'):
        rows = []
        for slot in slots:
            im = f'synthetic_{bank}/{slot["slot_id"]}.png'
            lab = f'synthetic_{bank}/{slot["slot_id"]}.txt'
            rows.append({'bank': bank, 'slot_id': slot['slot_id'], 'frozen_slot': slot,
                'image_path': im, 'image_sha256': write(source, im),
                'label_path': lab, 'label_sha256': write(source, lab),
                'checkpoint_tree_sha256': generators[slot['class']]['checkpoint']['tree_sha256'],
                **{k: rng[slot['slot_id']][bank][k] for k in ('sample_seed', 'bank_seed', 'non_rng_metadata_sha256')}})
        common.save(source / ('formal_bank_' + bank + '_manifest.json'), {'status': 'COMPLETE', 'records': rows,
            'generation_protocol_sha256': common.sha(P0 / 'stage20b_generation_protocol.json')})
        combined.extend(rows)
    common.save(source / 'formal_synthetic_manifest.json', {'status': 'FROZEN', 'count': 240, 'records': combined,
        'bank_counts': {'A': 80, 'B': 80, 'C': 80}, 'selector_count': 0, 'reroll_count': 0})
    common.save(source / 'formal_synthetic_validity.json', {'status': 'PASS'})
    init = {'CPU': 'identical fake initialization'}
    common.save(source / 'detector_initialization_comparison.json', init)
    order = common.load(P0 / 'stage20b_training_order_spec.json')
    for arm in ('RR', 'A', 'B'):
        last = f'detector_{arm}/last.pt'
        draws = f'logs/detector_{arm}_draws.jsonl'
        row = {'status': 'COMPLETE', 'arm': arm, 'epochs': 150, 'actual_draws': 32400,
            'attempted_optimizer_updates': 600, 'successful_optimizer_updates': 592, 'AMP_skipped_updates': 8,
            'official_final_eval_content_read': 0, 'last_checkpoint': last, 'last_sha256': write(source, last),
            'order_spec_sha256': common.sha(P0 / 'stage20b_training_order_spec.json'),
            'training_protocol_sha256': common.sha(P0 / 'stage20b_detector_protocol.json'),
            'initial_parameter_binding_sha256': common.htext(common.canonical(init)),
            'per_draw_log_sha256': write(source, draws), 'epoch_records': []}
        for expected in order['epochs']:
            row['epoch_records'].append({'epoch_0based': expected['epoch_0based'], 'draws': 216,
                'actual_role_sequence_sha256': expected['sequence_sha256'], 'attempts': [
                    {'normalizer': length, 'position': pos} for length, pos in
                    zip(expected['accumulation_group_lengths'], expected['optimizer_attempt_after_positions_0based'])]})
        common.save(source / f'detector_arm_{arm}_training.json', row)
        status[f'detector_{arm}_successful_updates'] = 592
        write(source, f'detector_{arm}/train_role_binding.json')
        for suffix in ('.log', '_launch.json', '_pid.json'):
            write(source, f'logs/train_plastic_bomo_stage20b_detector_arm_{arm}' + suffix)
    common.save(source / 'detector_arm_C_training.json', {'status': 'RUNNING', 'arm': 'C', 'epochs': 76,
        'actual_draws': 16416, 'successful_optimizer_updates': 296, 'official_final_eval_content_read': 0})
    common.save(source / 'stage20b_status.json', status)
    for name in ('STAGE20B_REPORT.md', 'logs/orchestrator_failure.txt', 'logs/detector_C_draws.jsonl',
                 'logs/train_plastic_bomo_stage20b_detector_arm_C.log',
                 'logs/train_plastic_bomo_stage20b_detector_arm_C_launch.json'):
        write(source, name)
    common.save(source / 'logs/train_plastic_bomo_stage20b_detector_arm_C_pid.json', {'pid': 54321})
    return source


class RecoveryTests(unittest.TestCase):
    def test_valid_precheck_and_import_never_write_source_or_import_partial_C(self):
        with tempfile.TemporaryDirectory() as d:
            source = fixture(Path(d))
            before = common.tree_binding(source, source)['sha256']
            _, _, _, files, snapshots = recovery.precheck(source)
            output = source.parent / 'new_manual_C'
            output.mkdir()
            recovery.import_assets(source, output, files, snapshots)
            recovery.verify_files(source, {**files, **snapshots})
            recovery.verify_files(output, files)
            self.assertEqual(before, common.tree_binding(source, source)['sha256'])
            self.assertFalse((output / 'detector_C').exists())
            self.assertFalse((output / 'detector_arm_C_training.json').exists())
            self.assertEqual(common.load(output / 'manual_recovery/source_snapshot/detector_arm_C_training.json')['epochs'], 76)

    def test_changed_RR_weight_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            source = fixture(Path(d))
            (source / 'detector_RR/last.pt').write_bytes(b'changed')
            with self.assertRaises(common.Stop):
                recovery.precheck(source)

    def test_changed_frozen_slot_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = fixture(Path(d))
            path = source / 'formal_bank_C_manifest.json'
            value = common.load(path)
            value['records'][0]['sample_seed'] += 1
            common.save(path, value)
            with self.assertRaises(common.Stop):
                recovery.precheck(source)

    def test_incomplete_B_or_wrong_tail_or_prior_eval_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = fixture(Path(d))
            path = source / 'detector_arm_B_training.json'
            original = common.load(path)
            changed = deepcopy(original)
            changed['epoch_records'][149]['attempts'][-1]['normalizer'] = 64
            common.save(path, changed)
            with self.assertRaises(common.Stop):
                recovery.precheck(source)
            common.save(path, original)
            common.save(source / 'RawA_metrics.json', {'status': 'COMPLETE'})
            with self.assertRaises(common.Stop):
                recovery.precheck(source)

    def test_NON_SIGKILL_and_finished_C_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = fixture(Path(d))
            path = source / 'stage20b_status.json'
            status = common.load(path)
            status['detail'] = status['detail'].replace('-9', '1')
            common.save(path, status)
            with self.assertRaises(common.Stop):
                recovery.precheck(source)
            status['detail'] = status['detail'].replace("'exit_code': 1", "'exit_code': -9")
            common.save(path, status)
            write(source, 'detector_C/last.pt')
            with self.assertRaises(common.Stop):
                recovery.precheck(source)

    def test_no_manual_confirmation_no_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source'
            source.mkdir()
            output = root / 'new'
            args = ['recover', '--source_run', str(source), '--output', str(output)]
            with patch.object(recovery.sys, 'argv', args):
                with self.assertRaises(common.Stop) as error:
                    recovery.main()
            self.assertEqual(error.exception.status, 'MANUAL_RECOVERY_CONFIRMATION_REQUIRED')
            self.assertFalse(output.exists())

    def test_duplicate_manual_attempt_rejected_without_new_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source'
            source.mkdir()
            output = root / 'new'
            write(root, 'manual_C_recovery_claim_' + common.htext(str(source.resolve()))[:16] + '.json')
            args = ['recover', '--source_run', str(source), '--output', str(output), '--confirm-manual-c-restart']
            with patch.object(recovery.sys, 'argv', args), patch.object(recovery, 'precheck', return_value=({}, {}, {}, {}, {})), patch.object(recovery, 'check_processes'):
                with redirect_stdout(io.StringIO()), self.assertRaises(common.Stop) as error:
                    recovery.main()
            self.assertEqual(error.exception.status, 'MANUAL_RECOVERY_ALREADY_CLAIMED')
            self.assertFalse(output.exists())

    def test_path_escape_and_changed_preserved_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(common.Stop):
                recovery.inside(root, '../outside')
            digest = write(root, 'file')
            (root / 'file').write_bytes(b'changed')
            with self.assertRaises(common.Stop):
                recovery.verify_files(root, {'file': digest})

    def test_copy_fallback_and_existing_destination_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, output = root / 'source', root / 'output'
            source.mkdir(); output.mkdir()
            name = 'detector_RR/last.pt'
            digest = write(source, name)
            with patch.object(recovery.os, 'link', side_effect=OSError('fixture: cross-device')):
                modes = recovery.import_assets(source, output, {name: digest}, {})
            self.assertEqual(modes[name], 'verified_copy')
            with self.assertRaises(common.Stop):
                recovery.import_assets(source, output, {name: digest}, {})

    def test_reused_PID_not_killed_and_live_source_process_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'formal_source'
            (source / 'logs').mkdir(parents=True)
            common.save(source / 'logs/train_plastic_bomo_stage20b_detector_arm_C_pid.json', {'pid': 123})
            with patch.object(recovery.sys, 'platform', 'linux'), patch.object(Path, 'read_bytes', return_value=b'unrelated task'):
                recovery.check_processes(source, {'pid': 123})
            # Parent may have a RELATIVE --output; basename still detects it.
            with patch.object(recovery.sys, 'platform', 'linux'), patch.object(Path, 'read_bytes', return_value=b'python --output results/formal_source'):
                with self.assertRaises(common.Stop):
                    recovery.check_processes(source, {'pid': 123})

    def test_failed_child_does_not_transition_or_eval(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'logs').mkdir()
            common.save(run / 'manual_arm_C_recovery.json', {'recovery_script_sha256': common.sha(recovery.SCRIPT)})
            with patch.object(recovery, 'context'), patch.object(recovery.subprocess, 'Popen') as popen, patch.object(recovery, 'transition') as transition:
                popen.return_value.pid = 999
                popen.return_value.wait.return_value = -9
                with redirect_stdout(io.StringIO()), self.assertRaises(common.Stop):
                    recovery.step(run, 'train_plastic_bomo_stage20b_detector.py', ['--arm', 'C'], 'DETECTOR_B_TRAINED', 'DETECTOR_C_TRAINED')
                transition.assert_not_called()
                self.assertEqual(popen.call_count, 1)

    def test_finish_defers_authority_until_manual_provenance_is_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            source, run = Path(d) / 'source', Path(d) / 'new'
            source.mkdir(); run.mkdir()
            write(run, 'manual_recovery/source_snapshot/interrupted.log')
            common.save(run / 'manual_arm_C_recovery.json', {'recovery_script_sha256': common.sha(recovery.SCRIPT)})
            common.save(run / 'detector_arm_C_training.json', {'last_sha256': 'CPU fake checkpoint hash'})
            pointer = run.parent / 'authoritative_formal_run.json'
            observed = []

            def fake_finish(current, p0, stage20a):
                report = current / 'STAGE20B_REPORT.md'
                report.write_text('CPU fake audit report\n', encoding='utf-8')
                audit.save(current / 'formal_result_artifact_manifest.json', {'sha256': {'STAGE20B_REPORT.md': common.sha(report)}})
                audit.save(pointer, {'run': current.name, 'status': 'CPU_FAKE_RESULT'})
                observed.append(pointer.exists())

            with patch.object(recovery, 'context', return_value=({}, P0, ROOT)), patch.object(audit, 'finish', side_effect=fake_finish):
                recovery.finish_recovery(run, source, {}, {})
            self.assertEqual(observed, [False])
            ledger = common.load(run / 'formal_result_artifact_manifest.json')
            for name, digest in ledger['sha256'].items():
                self.assertEqual(common.sha(run / name), digest)
            self.assertTrue(common.load(pointer)['manual_interruption_recovery'])
            self.assertEqual(common.load(pointer)['artifact_manifest_sha256'], common.sha(run / 'formal_result_artifact_manifest.json'))
            self.assertEqual(common.load(run / 'manual_arm_C_recovery.json')['status'], 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
