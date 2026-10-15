"""Stdlib fixtures only: no models, GPU, real image decoding or optimizer steps."""
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import recover_plastic_bomo_stage25_detector as recovery
from stage25_msdf_contract import canonical, htext, load, save, sha, tree_binding


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'formal_source'
        self.source.mkdir()
        self.output = self.root / 'detector_recovery'
        self.p0 = self.root / 'p0'
        self.runtime = self.root / 'runtime'
        self.p0.mkdir(); self.runtime.mkdir()
        self.b = {'p0': str(self.p0), 'p0_view': str(self.p0), 'runtime_dir': str(self.runtime),
                  **{k: str(self.root / k) for k in ('code_manifest', 'stage20a', 'r0', 'baseline_run',
                      'base_model', 'detector_model', 'real_root', 'normal_root', 'stage25_protocol')}, 'device': 'cuda:0'}
        self.cfg = {'bank_seeds': {'A': 42, 'B': 3407, 'C': 2026}}
        self.put('stage25_status.json', {'status': 'FORMAL_RUNNING', 'full_experiment_complete': False,
            'official_final_eval_rounds': 0, 'msdf_detector_A_started': True})
        self.put('detector_arm_A_training.json', {'arm': 'A', 'status': 'RUNNING', 'epochs': 9,
            'actual_draws': 2072, 'nonfinite_losses': 0, 'official_final_eval_content_read': 0,
            'attempted_optimizer_updates': 38, 'successful_optimizer_updates': 30, 'AMP_skipped_updates': 8})
        save(self.runtime / 'baseline_reuse_audit_server.json', {'reuse': True})
        self.put('msdf_v2_protocol.json', {'fixture': 'not scientific recipe'})
        self.put('server_code_manifest.json', {'fixture': 'not executable envelope'})
        self.b.update(protocol_sha256=sha(self.source / 'msdf_v2_protocol.json'),
                      code_manifest_sha256=sha(self.source / 'server_code_manifest.json'))
        self.put('stage25_run_binding.json', self.b)
        for name in ('msdf_historical_component_audit.json', 'msdf_selector_separation_audit.json'):
            self.put(name, {'fixture': True})
        generators = {'status': 'FROZEN', 'both_frozen_before_sampling': True, 'classes': {}}
        for cls in ('flash', 'black'):
            path = self.source / 'generator_checkpoints' / cls
            path.mkdir(parents=True)
            (path / 'FAKE_NOT_MODEL.bin').write_bytes(cls.encode())
            cp = tree_binding(path, self.source)
            generators['classes'][cls] = {'successful_updates': 2000, 'checkpoint': cp}
            log = self.source / 'logs' / f'msdf_{cls}_updates.jsonl'
            log.parent.mkdir(exist_ok=True)
            log.write_text('{"fixture":true}\n', encoding='utf-8')
            self.put(f'msdf_{cls}_training_audit.json', {'status': 'COMPLETE', 'checkpoint_saved': True,
                'successful_updates': 2000, 'VAE_text_frozen_unchanged': True,
                'checkpoint': cp, 'update_log_sha256': sha(log)})
        self.put('generator_checkpoint_binding.json', generators)
        self.put('msdf_checkpoint_binding.json', generators)
        slots = [{'slot_id': f's{i}', 'class': 'flash' if i < 40 else 'black'} for i in range(80)]
        save(self.p0 / 'stage20b_generation_slots.json', {'slots': slots})
        save(self.p0 / 'stage20b_generation_protocol.json', {'fixture': True})
        rng = []
        for slot in slots:
            rng.append({'slot_id': slot['slot_id'], 'banks': {bank: {'bank_seed': seed,
                'sample_seed': int(htext('Plastic_Bomo_Generation_V2' + slot['slot_id'] + str(seed))[:16], 16),
                'non_rng_metadata_sha256': htext(canonical(slot))} for bank, seed in self.cfg['bank_seeds'].items()}})
        save(self.p0 / 'stage20b_rng_manifest.json', {'records': rng})
        combined = []
        for bank in recovery.BANKS:
            records = []
            for slot, random in zip(slots, rng):
                row = {**slot, 'bank': bank, 'role_id': 'extra_' + slot['slot_id'],
                    'frozen_slot': slot, 'MSDF_enabled': True, 'formal_synthetic': True,
                    'quality_selection': False, **random['banks'][bank],
                    'checkpoint_tree_sha256': generators['classes'][slot['class']]['checkpoint']['tree_sha256']}
                # MSDF trace lacks Raw's redundant non_rng_metadata_sha256 field.
                row.pop('non_rng_metadata_sha256')
                for field in ('image', 'label'):
                    name = f'synthetic_{bank}/{slot["slot_id"]}.{field}'
                    path = self.source / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b'FAKE_NOT_IMAGE_OR_LABEL')
                    row.update({field + '_path': name, field + '_sha256': sha(path)})
                records.append(row)
            mf = {'status': 'COMPLETE', 'bank': bank, 'records': records, 'count': 80,
                'quality_selection': False, 'reroll_count': 0,
                'generation_protocol_sha256': sha(self.p0 / 'stage20b_generation_protocol.json')}
            self.put(f'formal_bank_{bank}_manifest.json', mf)
            self.put(f'msdf_bank_{bank}_manifest.json', mf)
            combined.extend(records)
        self.put('formal_synthetic_manifest.json', {'status': 'FROZEN', 'count': 240,
            'records': combined, 'quality_selection': False})
        validity = {'status': 'PASS', 'count': 240, 'reroll': 0, 'all_planned_outputs_retained': True,
            'structural_checks_only': True, 'manifest_sha256': sha(self.source / 'formal_synthetic_manifest.json')}
        self.put('formal_synthetic_validity.json', validity)
        self.put('msdf_generation_validity.json', validity)

    def put(self, name, row):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save(path, row)

    def check(self):
        with patch.object(recovery, 'binding', return_value=(self.b, self.cfg)):
            return recovery.precheck(self.source)

    def test_precheck_all240_both_models_without_ML(self):
        before = tree_binding(self.source, self.root)
        b, _, audit, files, snapshots = self.check()
        self.assertEqual(audit['epochs'], 9)
        self.assertEqual(sum(n.startswith('synthetic_') for n in files), 480)
        self.assertEqual(b, self.b)
        self.assertIn('stage25_status.json', snapshots)
        self.assertEqual(tree_binding(self.source, self.root), before)
        for module in ('torch', 'ultralytics', 'diffusers', 'PIL'):
            self.assertNotIn(module, sys.modules)

    def test_byte_identical_import_source_preserved(self):
        *_, files, snapshots = self.check()
        original = tree_binding(self.source, self.root)
        self.output.mkdir()
        modes = recovery.import_assets(self.source, self.output, files, snapshots)
        recovery.verify_files(self.output, files)
        self.assertEqual(tree_binding(self.source, self.root), original)
        self.assertEqual(load(self.output / 'formal_synthetic_manifest.json'), load(self.source / 'formal_synthetic_manifest.json'))
        self.assertNotIn('stage25_status.json', modes)
        self.assertFalse((self.output / 'detector_A').exists())
        self.assertFalse((self.output / 'detector_initialization_comparison.json').exists())
        for cp in load(self.output / 'generator_checkpoint_binding.json')['classes'].values():
            recovery.verify_tree(self.output, cp['checkpoint'])

    def test_copy_fallback_when_hardlinks_unavailable(self):
        name = 'synthetic_A/s0.image'
        self.output.mkdir()
        with patch.object(recovery.os, 'link', side_effect=OSError('fixture filesystem')):
            modes = recovery.import_assets(self.source, self.output, {name: sha(self.source / name)}, {})
        self.assertEqual(modes[name], 'verified_copy')
        self.assertEqual(sha(self.output / name), sha(self.source / name))

    def test_manifest_image_tamper_rejected(self):
        (self.source / 'synthetic_C/s79.image').write_bytes(b'CHANGED')
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_sample_seed_change_rejected_without_reroll(self):
        mf = load(self.source / 'formal_bank_B_manifest.json')
        mf['records'][0]['sample_seed'] += 1
        self.put('formal_bank_B_manifest.json', mf)
        self.put('msdf_bank_B_manifest.json', mf)
        with self.assertRaises(recovery.Stop) as caught:
            self.check()
        self.assertEqual(caught.exception.status, 'STAGE25_BANK_BARRIER_FAILED')

    def test_changed_checkpoint_rejected(self):
        (self.source / 'generator_checkpoints/black/FAKE_NOT_MODEL.bin').write_bytes(b'CHANGED')
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_1999_updates_rejected(self):
        row = load(self.source / 'msdf_flash_training_audit.json')
        row['successful_updates'] = 1999
        self.put('msdf_flash_training_audit.json', row)
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_nonfinite_loss_cannot_be_reclassified_interruption(self):
        row = load(self.source / 'detector_arm_A_training.json')
        row['nonfinite_losses'] = 1
        self.put('detector_arm_A_training.json', row)
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_completed_checkpoint_blocks_restart(self):
        path = self.source / 'detector_A/last.pt'
        path.parent.mkdir()
        path.write_bytes(b'NOT_MODEL')
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_any_B_or_C_attempt_blocks_recovery(self):
        for bank in ('B', 'C'):
            path = self.source / f'detector_arm_{bank}_training.json'
            with self.subTest(bank=bank):
                save(path, {'status': 'RUNNING'})
                with self.assertRaises(recovery.Stop):
                    self.check()
                path.unlink()  # Test fixture only.

    def test_prior_evaluation_blocks_restart(self):
        self.put('final_eval_access_audit.json', {'status': 'FAILED'})
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_numerical_worker_failure_blocks_restart(self):
        self.put('worker_failure.json', {'status': 'STAGE25_FORMAL_OOM'})
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_explicit_interruption_status_accepted(self):
        row = load(self.source / 'stage25_status.json')
        row['status'] = 'STAGE25_INTERRUPTED'
        self.put('stage25_status.json', row)
        self.put('worker_failure.json', {'status': 'STAGE25_INTERRUPTED'})
        self.check()

    def test_no_comparator_rebuild_in_this_recovery(self):
        save(self.runtime / 'baseline_reuse_audit_server.json', {'reuse': False})
        with self.assertRaises(recovery.Stop):
            self.check()

    def test_path_traversal_rejected(self):
        for name in ('../escape', str(self.root / 'outside')):
            with self.assertRaises(recovery.Stop):
                recovery.inside(self.source, name)

    def test_one_manual_recovery_claim_only(self):
        claim = recovery.reserve(self.source, self.output)
        self.assertEqual(load(claim)['output'], str(self.output))
        with self.assertRaises(recovery.Stop):
            recovery.reserve(self.source, self.root / 'second_output')

    def test_detector_failure_never_runs_evaluation(self):
        self.output.mkdir()
        journal = {'recovery_entry_sha256': sha(recovery.SCRIPT)}
        with patch.object(recovery.runner, 'formal_detector', side_effect=RuntimeError('interrupted')) as train, \
             patch.object(recovery.runner, 'child') as child:
            with self.assertRaises(RuntimeError):
                recovery.run_recovery(self.output, self.source, journal, self.b)
        train.assert_called_once()
        child.assert_not_called()

    def test_full_execution_order_and_final_ledger(self):
        _, _, _, files, snapshots = self.check()
        self.output.mkdir()
        recovery.import_assets(self.source, self.output, files, snapshots)
        (self.output / 'recovery').mkdir(exist_ok=True)
        recovery.verified_copy(recovery.SCRIPT, self.output / 'recovery/entry.py', sha(recovery.SCRIPT))
        journal = {'recovery_entry_sha256': sha(recovery.SCRIPT), 'source_sha256': {**files, **snapshots}, 'reused_sha256': files}
        events = []
        def detector(args, run):
            events.append('ABC_frozen')
            self.assertEqual(args.device, '0')
        def evaluation(args, run, name):
            self.assertEqual(name, 'evaluate')
            events.append('unified_eval')
            save(run / 'stage25_status.json', {'full_experiment_complete': True})
            save(run / 'stage25_result_artifact_manifest.json', {'sha256': {}})
            (run / 'STAGE25_REPORT.md').write_text('TEST NOT A RESULT\n', encoding='utf-8')
        with patch.object(recovery.runner, 'formal_detector', side_effect=detector), \
             patch.object(recovery.runner, 'child', side_effect=evaluation):
            recovery.run_recovery(self.output, self.source, journal, self.b)
        self.assertEqual(events, ['ABC_frozen', 'unified_eval'])
        ledger = load(self.output / 'stage25_result_artifact_manifest.json')
        for name, digest in ledger['sha256'].items():
            self.assertEqual(sha(self.output / name), digest)
        self.assertTrue(load(self.output / 'manual_detector_recovery.json')['source_run_unchanged_SHA_verified'])

    def test_check_only_creates_no_output_or_claim(self):
        before = tree_binding(self.source, self.root)
        argv = ['recover', '--source_run', str(self.source), '--output', str(self.output), '--check-only']
        with patch.object(recovery, 'ROOT', self.root), patch.object(recovery, 'check_processes'), \
             patch.object(recovery, 'binding', return_value=(self.b, self.cfg)), \
             patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            recovery.main()
        self.assertEqual(tree_binding(self.source, self.root), before)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('manual_detector_recovery_claim_*')))

    def test_main_keeps_old_A_and_zero_new_generation(self):
        before = tree_binding(self.source, self.root)
        argv = ['recover', '--source_run', str(self.source), '--output', str(self.output), '--confirm-detector-restart']
        with patch.object(recovery, 'ROOT', self.root), patch.object(recovery, 'check_processes'), \
             patch.object(recovery, 'binding', return_value=(self.b, self.cfg)), \
             patch.object(recovery, 'run_recovery') as execute, \
             patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            recovery.main()
        execute.assert_called_once()
        self.assertEqual(tree_binding(self.source, self.root), before)
        status = load(self.output / 'stage25_status.json')
        self.assertEqual(status['new_generator_training_count'], 0)
        self.assertEqual(status['new_synthetic_generation_count'], 0)
        self.assertEqual(status['detector_training_count'], 0)
        journal = load(self.output / 'manual_detector_recovery.json')
        self.assertEqual(journal['interrupted_A_observed_counters']['epochs'], 9)
        self.assertEqual(journal['primary_scheduled_attempts_per_arm'], 600)
        self.assertFalse((self.output / 'detector_A').exists())

    def test_main_failure_seals_stop_without_touching_source(self):
        before = tree_binding(self.source, self.root)
        argv = ['recover', '--source_run', str(self.source), '--output', str(self.output), '--confirm-detector-restart']
        with patch.object(recovery, 'ROOT', self.root), patch.object(recovery, 'check_processes'), \
             patch.object(recovery, 'binding', return_value=(self.b, self.cfg)), \
             patch.object(recovery, 'run_recovery', side_effect=RuntimeError('injected stop')), \
             patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                recovery.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(tree_binding(self.source, self.root), before)
        self.assertTrue(load(self.output / 'manual_detector_recovery.json')['status'].startswith('STOPPED_'))
        self.assertFalse(load(self.output / 'recovery_failure.json')['automatic_retry'])
        self.assertFalse(load(self.output / 'stage25_status.json')['full_experiment_complete'])
        self.assertFalse((self.output / 'final_eval_inputs').exists())

    def test_process_check_read_only_blocks_existing_task(self):
        procroot = self.root / 'proc_fixture'
        task = procroot / '1234567'
        task.mkdir(parents=True)
        command = b'python\0tools/run_plastic_bomo_stage25_msdf_v2.py\0--_job\0detector\0'
        (task / 'cmdline').write_bytes(command)
        with patch.object(recovery.sys, 'platform', 'linux'):
            with self.assertRaises(recovery.Stop) as caught:
                recovery.check_processes(procroot)
        self.assertEqual(caught.exception.status, 'STAGE25_RECOVERY_TASK_STILL_RUNNING')
        self.assertEqual((task / 'cmdline').read_bytes(), command)

    def test_own_process_and_unrelated_gpu_task_not_blocked(self):
        procroot = self.root / 'proc_fixture'
        for pid, command in ((os.getpid(), str(recovery.SCRIPT)), (99999999, 'unrelated_training.py')):
            task = procroot / str(pid)
            task.mkdir(parents=True)
            (task / 'cmdline').write_bytes(b'python\0' + command.encode() + b'\0')
        with patch.object(recovery.sys, 'platform', 'linux'):
            recovery.check_processes(procroot)

    def test_Windows_training_is_rejected(self):
        with patch.object(recovery.sys, 'platform', 'win32'):
            with self.assertRaises(recovery.Stop) as caught:
                recovery.check_processes()
        self.assertEqual(caught.exception.status, 'STAGE25_RECOVERY_SERVER_REQUIRED')


class CLI(unittest.TestCase):
    def test_help_is_stdlib_only(self):
        p = subprocess.run([sys.executable, str(recovery.SCRIPT), '--help'], cwd=REPO,
            capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--confirm-detector-restart', p.stdout)
        self.assertIn('--check-only', p.stdout)

    def test_confirmation_required_before_any_binding_or_model(self):
        with tempfile.TemporaryDirectory() as d, patch.object(recovery, 'binding') as bind, \
             patch.object(sys, 'argv', ['recover', '--source_run', d, '--output', d + '_new']), redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as caught:
                recovery.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('STAGE25_RECOVERY_CONFIRMATION_REQUIRED', out.getvalue())
        bind.assert_not_called()

    def test_no_changes_to_original_hash_envelope(self):
        mf = load(REPO / 'configs/plastic_bomo_stage25_server_code_manifest.json')
        for name, expected in mf['sha256'].items():
            self.assertEqual(sha(REPO / name), expected, name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
