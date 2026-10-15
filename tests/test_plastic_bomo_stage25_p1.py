"""Stdlib-only static/contract tests. Zero tensors, models, GPU forwards or steps."""
import ast
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import stage25_msdf_contract as contract
import stage25_msdf_generator as generator
import stage25_msdf_downstream as downstream
import run_plastic_bomo_stage25_msdf_v2 as runner
from stage20b_detector_controller import accumulation_dryrun, Stage20BDetectorController
from stage20b_runtime_contract import generator_budget_dryrun

CFG = contract.load(REPO / 'configs/plastic_bomo_stage25_server.json')
P0 = REPO / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115'
LOCAL25 = REPO / 'results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification'


def metric(value, flash=None, black=None):
    return {'map5095': value, 'per_class': {'flash': {'ap5095': value if flash is None else flash},
                                         'black': {'ap5095': value if black is None else black}}}


class Stage25Static(unittest.TestCase):
    def test_no_ML_imports(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('ultralytics', sys.modules)
        self.assertNotIn('diffusers', sys.modules)

    def test_frozen_source_recipe(self):
        self.assertEqual(contract.source_check(REPO / 'configs/plastic_bomo_stage25_server.json',
            REPO / 'configs/plastic_bomo_stage25_server_code_manifest.json'), CFG)

    def test_parent_protocol_preserved(self):
        parent = contract.load(LOCAL25 / 'msdf_v2_protocol.json')
        self.assertEqual(contract.sha(LOCAL25 / 'msdf_v2_protocol.json'), CFG['parent_protocol_sha256'])
        for key in ('common_generator_training_target', 'MSDF_preserved_defaults', 'generation_target',
                    'bank_seeds', 'detector_target', 'detector_exposure', 'primary_gates', 'utility_gates', 'final_eval_status'):
            self.assertEqual(CFG[key], parent[key], key)
        self.assertEqual(CFG['core_algorithm_changes'], [])

    def test_entire_original_P0_ledger_unchanged(self):
        for row in contract.load(LOCAL25 / 'artifact_manifest.json')['files']:
            self.assertEqual(contract.sha(LOCAL25 / row['path']), row['sha256'], row['path'])

    def test_old_algorithm_hashes(self):
        for row in CFG['algorithm_binding']:
            self.assertEqual(contract.sha(REPO / row['path']), row['sha256'])

    def test_literal_historical_code_only(self):
        source, helpers, branch = generator.historical_nodes()
        self.assertEqual({x.name for x in helpers}, set(generator.HELPERS))
        compile(ast.Module(body=helpers, type_ignores=[]), 'frozen_helpers', 'exec')
        compile(ast.Module(body=branch.body, type_ignores=[]), 'frozen_losses', 'exec')
        self.assertEqual(ast.unparse(branch.test), 'msdf is not None')
        names = {x.attr for s in branch.body for x in ast.walk(s) if isinstance(x, ast.Attribute)
                 and isinstance(x.value, ast.Name) and x.value.id == 'args'}
        self.assertEqual(names, {'msdf_' + n + '_weight' for n in generator.LOSSES})
        self.assertIn('reliability = alphas.clamp_min(0.05)', source)
        self.assertEqual(CFG['MSDF_preserved_defaults']['MSDF_learning_rate'], 1e-4)
        self.assertEqual(CFG['common_generator_training_target']['learning_rate'], 1e-6)

    def test_all_new_Python_compiles(self):
        files = list((REPO / 'tools').glob('stage25_msdf_*.py')) + [REPO / 'tools/run_plastic_bomo_stage25_msdf_v2.py']
        self.assertEqual(len(files), 4)
        for p in files:
            compile(p.read_text(encoding='utf-8'), str(p), 'exec')

    def test_new_shell_uses_actual_root_and_no_installs(self):
        text = (REPO / 'tools/launch_plastic_bomo_stage25_server.sh').read_text(encoding='utf-8')
        self.assertIn('BASH_SOURCE[0]', text)
        self.assertIn('--device 0', text)
        self.assertNotIn('pip install', text)
        self.assertNotIn('ssh ', text)
        self.assertNotIn('\r', text)

    def test_launcher_runs_in_foreground_and_announces_exit(self):
        text = (REPO / 'tools/launch_plastic_bomo_stage25_server.sh').read_text(encoding='utf-8')
        self.assertNotIn('nohup ', text)
        self.assertNotIn('2>&1 &', text)
        self.assertIn('tee "$TASK25_LOG"', text)
        self.assertIn('[已退出]', text)
        self.assertIn('trap ', text)

    def test_nonruntime_artifacts_are_optional_on_server(self):
        mf = contract.load(REPO / 'configs/plastic_bomo_stage25_server_code_manifest.json')
        optional = {'tests/test_plastic_bomo_stage25_p1.py', 'Stage25服务器命令.txt',
                    'tools/launch_plastic_bomo_stage25_server.sh', 'run_train_msdf.py'}
        self.assertTrue(optional <= set(mf['audit_only_sha256']))
        self.assertFalse(optional & set(mf['sha256']))
        isfile = Path.is_file
        excluded = {(REPO / n).resolve() for n in optional}
        with patch.object(Path, 'is_file', lambda p: False if p.resolve() in excluded else isfile(p)):
            self.assertEqual(contract.source_check(REPO / 'configs/plastic_bomo_stage25_server.json',
                REPO / 'configs/plastic_bomo_stage25_server_code_manifest.json'), CFG)

    def test_actual_execution_hashes_remain_required(self):
        real_sha = contract.sha
        path = REPO / 'diffusers/pipelines/stable_diffusion/msdf_guidance.py'
        with patch.object(contract, 'sha', side_effect=lambda p: '0' * 64 if Path(p).resolve() == path else real_sha(p)):
            with self.assertRaises(contract.Stop) as caught:
                contract.source_check(REPO / 'configs/plastic_bomo_stage25_server.json',
                    REPO / 'configs/plastic_bomo_stage25_server_code_manifest.json')
        self.assertEqual(caught.exception.detail['file'], path.relative_to(REPO).as_posix())

    def test_help_never_imports_ML(self):
        result = subprocess.run([sys.executable, str(REPO / 'tools/run_plastic_bomo_stage25_msdf_v2.py'), '--help'],
            capture_output=True, text=True, cwd=REPO)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('runtime_check,train,generate,detector,evaluate,all', result.stdout)

    def test_no_generic_main_launch(self):
        text = (REPO / 'tools/run_plastic_bomo_stage25_msdf_v2.py').read_text(encoding='utf-8')
        for name in ('run_train_msdf.py', 'inference.py', 'train_rda.py'):
            self.assertNotIn(name, text)
        g = (REPO / 'tools/stage25_msdf_generator.py').read_text(encoding='utf-8')
        self.assertIn('from generate_plastic_bomo_stage20b_banks import sample', g)
        self.assertIn('register_forward_pre_hook(prepare, with_kwargs=True)', g)
        self.assertNotIn('normal_filter(', g)
        self.assertNotIn('StableDiffusionInpaintPipeline_dynamic', g)


class Gates(unittest.TestCase):
    def summary(self, values, raw=.3, rr=.3):
        return contract.metrics_summary({b: metric(v) for b, v in zip(contract.BANKS, values)},
            {b: metric(raw) for b in contract.BANKS}, metric(rr), CFG)

    def test_primary_and_utility_pass(self):
        x = self.summary([.31, .32, .33])
        self.assertEqual(x['status'], 'MSDF_V2_REQUALIFICATION_PASS')
        self.assertEqual(x['msdf_vs_raw_wins'], 3)
        self.assertAlmostEqual(x['mean_msdf_vs_raw_pp'], 2)

    def test_primary_fail(self):
        x = self.summary([.29, .28, .27])
        self.assertEqual(x['status'], 'MSDF_GENERATOR_IMPROVEMENT_NOT_CONFIRMED')
        self.assertFalse(x['MSDF_V2_REQUALIFICATION_FAILED'] is False)

    def test_utility_fail(self):
        x = self.summary([.31, .32, .33], rr=.4)
        self.assertEqual(x['status'], 'MSDF_GENERATOR_IMPROVEMENT_SUPPORTED_BUT_UTILITY_NOT_CONFIRMED')

    def test_ties_not_strict_wins(self):
        x = self.summary([.31, .30, .30])
        self.assertEqual(x['msdf_vs_raw_wins'], 1)
        self.assertFalse(x['gates']['B'])

    def test_exact_half_pp_threshold(self):
        x = self.summary([.005, .005, .005], raw=0, rr=0)
        self.assertTrue(x['gates']['A'])
        self.assertEqual(x['mean_msdf_vs_raw_pp'], .5)

    def test_class_collapse_blocks_primary(self):
        x = contract.metrics_summary({b: metric(.4, black=.269) for b in contract.BANKS},
             {b: metric(.3) for b in contract.BANKS}, metric(.3), CFG)
        self.assertFalse(x['gates']['C'])

    def test_nonfinite_and_invalid_class_AP_rejected(self):
        for value in (float('nan'), float('inf'), -1, 2):
            with self.assertRaises(contract.Stop):
                contract.metrics_summary({b: metric(.4, black=value) for b in contract.BANKS},
                    {b: metric(.3) for b in contract.BANKS}, metric(.3), CFG)

    def test_missing_extra_banks_rejected(self):
        for names in (('A', 'B'), ('A', 'B', 'C', 'D')):
            with self.assertRaises(contract.Stop):
                contract.metrics_summary({b: metric(.4) for b in names}, {b: metric(.3) for b in contract.BANKS}, metric(.3), CFG)


class BudgetsAndBarriers(unittest.TestCase):
    def test_detector_controller_32400_600_tail24(self):
        cfg = contract.load(REPO / 'configs/plastic_bomo_stage20b_p0.json')
        order = contract.load(P0 / 'stage20b_training_order_spec.json')
        x = accumulation_dryrun(cfg, order)
        self.assertEqual((x['draws'], x['epochs'], x['scheduled_optimizer_attempts']), (32400, 150, 600))
        self.assertEqual([r['draws'] for r in x['attempt_groups'][:4]], [64, 64, 64, 24])
        self.assertEqual(x['actual_optimizer_steps'], 0)

    def test_detector_failure_has_no_eval_callback(self):
        cfg = contract.load(REPO / 'configs/plastic_bomo_stage20b_p0.json')
        order = contract.load(P0 / 'stage20b_training_order_spec.json')
        cleaned = []
        def fail(*args):
            raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):
            Stage20BDetectorController(cfg, order).run(fail, fail, cleanup=lambda: cleaned.append(True))
        self.assertEqual(cleaned, [True])

    def test_uint64_seeds_match_original80x3(self):
        rows = contract.load(P0 / 'stage20b_rng_manifest.json')['records']
        values = set()
        for r in rows:
            for bank, seed in CFG['bank_seeds'].items():
                expected = int(contract.htext('Plastic_Bomo_Generation_V2' + r['slot_id'] + str(seed))[:16], 16)
                self.assertEqual(r['banks'][bank]['sample_seed'], expected)
                self.assertLess(expected, 2**64)
                values.add(expected)
        self.assertEqual(len(rows), 80)
        self.assertEqual(len(values), 240)

    def test_claim_is_one_shot(self):
        with tempfile.TemporaryDirectory() as d:
            contract.claim(d, 'phase')
            with self.assertRaises(contract.Stop):
                contract.claim(d, 'phase')

    def test_no_parent_output_overwrite(self):
        with self.assertRaises(contract.Stop):
            runner.create_output(REPO, 'formal')

    def test_generator_1999_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            contract.save(root / 'msdf_flash_training_audit.json', {'status': 'COMPLETE', 'successful_updates': 1999})
            with self.assertRaises(contract.Stop):
                generator.freeze_generators(root)
            self.assertFalse((root / 'generator_checkpoint_binding.json').exists())

    def test_missing_black_checkpoint_barrier(self):
        with tempfile.TemporaryDirectory() as d, patch.object(generator, 'verify_tree'):
            root = Path(d)
            contract.save(root / 'msdf_flash_training_audit.json', {'status': 'COMPLETE', 'successful_updates': 2000, 'checkpoint': {}})
            with self.assertRaises(FileNotFoundError):
                generator.freeze_generators(root)
            self.assertFalse((root / 'generator_checkpoint_binding.json').exists())

    def test_detector_599_attempts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            contract.save(root / 'detector_arm_A_training.json', {'status': 'COMPLETE', 'epochs': 150,
                'actual_draws': 32400, 'attempted_optimizer_updates': 599})
            with self.assertRaises(contract.Stop):
                downstream.freeze_detectors(root)
            self.assertFalse((root / 'detector_checkpoint_binding.json').exists())

    def test_missing_third_detector_blocks_eval(self):
        with tempfile.TemporaryDirectory() as d, patch.object(downstream, 'binding', return_value=({}, CFG)):
            root = Path(d)
            contract.save(root / 'detector_checkpoint_binding.json', {'status': 'FROZEN', 'arms': {'A': {}, 'B': {}}})
            with self.assertRaises(contract.Stop):
                downstream.evaluation_checkpoints(root)
            self.assertFalse((root / 'final_eval_inputs').exists())

    def test_missing_comparator_marks_rebuild_not_MSDF_failure(self):
        with tempfile.TemporaryDirectory() as d:
            a = type('A', (), {'baseline_run': Path(d), 'p0': P0})()
            x = contract.baseline_audit(a, CFG)
            self.assertEqual(x['status'], 'RAW_COMPARATOR_REUSE_NOT_QUALIFIED')
            self.assertFalse(x['reuse'])

    def test_runtime_mismatch_does_not_import_torch_or_install(self):
        def missing(name):
            raise contract.md.PackageNotFoundError(name)
        with patch.object(contract.md, 'version', side_effect=missing):
            x = contract.inspect_environment(CFG)
        self.assertTrue(x['differences'])
        self.assertEqual(x['package_modification_count'], 0)
        self.assertEqual(x['GPU_forward_test'], 'NOT_RUN')
        self.assertNotIn('torch', sys.modules)

    def test_full_real_metadata_layout_and_allowlist(self):
        rows = contract.real_rows(REPO / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol', Path('/explicit/real_only'), 'train')
        self.assertEqual(len(rows), 136)
        self.assertTrue(all(r['split'] == 'train' and '/explicit/real_only/' in r['image_path'].replace('\\', '/') for r in rows))
        normals = contract.load(P0 / 'stage20b_normal_pool_freeze.json')['records']
        self.assertEqual(len(normals), 394)
        self.assertTrue(all(r['split'] == 'train' and r['confirmatory_allowed'] for r in normals))

    def test_actual_metadata_path_relocation_shell(self):
        """666 fake bytes/SHA fixture, NOT a scientific image or weight validation."""
        actual_sha = contract.sha
        stage20a = REPO / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            real = root / 'real'
            normals = root / 'normals'
            base = root / 'public'
            out = root / 'new_runtime'
            out.mkdir(); normals.mkdir()
            fixtures = {}
            def fixture(path, digest):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'STATIC_TEST_FIXTURE_NOT_MODEL_OR_IMAGE')
                fixtures[str(path.resolve())] = digest
            for row in contract.real_rows(stage20a, real, 'train'):
                fixture(Path(row['image_path']), row['image_sha256'])
                fixture(Path(row['label_path']), row['label_sha256'])
            for row in contract.load(P0 / 'stage20b_normal_pool_freeze.json')['records']:
                fixture(normals / contract.portable_name(row['path']), row['sha256'])
            for name, row in contract.load(P0 / 'sd2_base_weight_binding.json')['files'].items():
                fixture(base / name, row['sha256'])
            detector = root / 'yolo11s.pt'
            fixture(detector, CFG['detector_target']['initialization_sha256'])
            a = SimpleNamespace(stage20a=stage20a, p0=P0, r0=REPO / 'results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254',
                real_root=real, normal_root=normals, base_model=base, detector_model=detector)
            def fixture_sha(path):
                return fixtures.get(str(Path(path).resolve())) or actual_sha(path)
            with patch.object(contract, 'sha', side_effect=fixture_sha):
                x = contract.bind_assets(a, CFG, out)
            self.assertEqual(len(x['TRAIN_checks']), 666)
            self.assertEqual(x['normal_filter_count'], 0)
            slots = contract.load(out / 'p0_view/stage20b_generation_slots.json')['slots']
            original = contract.load(P0 / 'stage20b_generation_slots.json')['slots']
            for new, old in zip(slots, original):
                self.assertEqual({k: v for k, v in new.items() if k not in ('source_path', 'normal_path')},
                                 {k: v for k, v in old.items() if k not in ('source_path', 'normal_path')})
            self.assertEqual(contract.load(P0 / 'stage20b_generation_slots.json')['slots'], original)

    def test_historical_comparator_positive_metadata_contract(self):
        """Mock absent large assets only; exercise real historical metadata schema."""
        root = REPO / 'results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543'
        assets = {}
        for bank in contract.BANKS:
            for r in contract.load(root / ('formal_bank_' + bank + '_manifest.json'))['records']:
                for field in ('image', 'label'):
                    assets[str((root / r[field + '_path']).resolve())] = r[field + '_sha256']
        for r in contract.load(root / 'detector_checkpoint_binding.json')['arms'].values():
            assets[str((root / r['path']).resolve())] = r['sha256']
        actual = contract.sha
        with patch.object(contract, 'verify_tree'), patch.object(contract, 'verify_ledger'), \
             patch.object(contract, 'sha', side_effect=lambda p: assets.get(str(Path(p).resolve())) or actual(p)):
            x = contract.baseline_audit(SimpleNamespace(baseline_run=root, p0=P0), CFG)
        self.assertTrue(x['reuse'], x['errors'])
        self.assertEqual(set(x['metrics']), {'RR', 'A', 'B', 'C'})
        self.assertFalse(x['untouched_final_test_claim'])


class ConsoleDiagnostics(unittest.TestCase):
    def test_quiet_child_gets_running_heartbeat(self):
        with tempfile.TemporaryDirectory() as d:
            proc = Mock(pid=12345, stdout=io.StringIO())
            proc.poll.return_value = None
            proc.wait.return_value = 0
            messages = Mock()
            messages.get.side_effect = [runner.queue.Empty(), None]
            out = io.StringIO()
            with patch.object(runner.subprocess, 'Popen', return_value=proc), \
                 patch.object(runner.threading, 'Thread'), patch.object(runner.queue, 'Queue', return_value=messages), redirect_stdout(out):
                code = runner.stream_child(['FAKE_METADATA_PROCESS'], Path(d) / 'quiet.log', '文件哈希校验', REPO)
            self.assertEqual(code, 0)
            self.assertIn('等待阶段输出', out.getvalue())
            self.assertIn('文件哈希校验', out.getvalue())
            proc.terminate.assert_not_called()

    def test_interrupt_stops_only_owned_child(self):
        with tempfile.TemporaryDirectory() as d:
            proc = Mock(pid=12345, stdout=io.StringIO())
            proc.poll.return_value = None
            proc.wait.return_value = 0
            messages = Mock()
            messages.get.side_effect = KeyboardInterrupt()
            with patch.object(runner.subprocess, 'Popen', return_value=proc), \
                 patch.object(runner.threading, 'Thread'), patch.object(runner.queue, 'Queue', return_value=messages), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.stream_child(['FAKE_METADATA_PROCESS'], Path(d) / 'interrupt.log', '静态模拟', REPO)
            proc.terminate.assert_called_once()
            proc.kill.assert_not_called()
            self.assertTrue(proc.stdout.closed)

    def test_missing_runtime_files_reported_together(self):
        with tempfile.TemporaryDirectory() as d:
            mf = Path(d) / 'manifest.json'
            contract.save(mf, {'sha256': {'missing_stage25_one.py': '0' * 64, 'missing_stage25_two.py': '0' * 64}})
            with self.assertRaises(contract.Stop) as caught:
                contract.source_check(REPO / 'configs/plastic_bomo_stage25_server.json', mf)
            self.assertEqual(caught.exception.detail['missing_files'], ['missing_stage25_one.py', 'missing_stage25_two.py'])
            self.assertFalse(caught.exception.detail['training_started'])

    def test_chinese_diagnosis_and_exit_code_are_readable(self):
        out = io.StringIO()
        with redirect_stdout(out):
            runner.describe_error('STAGE25_FROZEN_BINDING_CHANGED', {'reason': '服务器缺少运行文件',
                'missing_files': ['缺少的文件.py']})
        self.assertIn('[已退出]', out.getvalue())
        self.assertIn('退出码=1', out.getvalue())
        self.assertIn('缺少文件：缺少的文件.py', out.getvalue())
        self.assertNotIn('\\u', out.getvalue())

    def test_CLI_missing_file_exits_before_runtime_or_ML(self):
        with tempfile.TemporaryDirectory() as d:
            mf = Path(d) / 'manifest.json'
            contract.save(mf, {'sha256': {'不存在的运行文件.py': '0' * 64}})
            p = subprocess.run([sys.executable, '-u', str(REPO / 'tools/run_plastic_bomo_stage25_msdf_v2.py'),
                '--mode', 'all', '--code_manifest', str(mf)], capture_output=True, text=True, encoding='utf-8',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, cwd=REPO)
            self.assertEqual(p.returncode, 1)
            self.assertIn('缺少文件：不存在的运行文件.py', p.stdout)
            self.assertIn('[已退出]', p.stdout)
            self.assertNotIn('\\u', p.stdout)
            self.assertNotIn('Traceback', p.stdout + p.stderr)

    def test_child_stdout_stderr_and_failure_are_teed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'child.log'
            out = io.StringIO()
            command = [sys.executable, '-u', '-c', "import sys; print('中文阶段输出'); print('错误详情',file=sys.stderr); sys.exit(7)"]
            with redirect_stdout(out):
                code = runner.stream_child(command, path, '静态模拟', REPO)
            self.assertEqual(code, 7)
            for message in ('中文阶段输出', '错误详情', '退出码=7'):
                self.assertIn(message, out.getvalue())
                self.assertIn(message, path.read_text(encoding='utf-8'))

    def test_child_output_arrives_before_process_completes(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / 'finished'
            observed = []
            class LiveOutput(io.StringIO):
                def write(self, text):
                    if text == 'CHILD_READY\n':
                        observed.append(not marker.exists())
                    return super().write(text)
            command = [sys.executable, '-u', '-c', "import pathlib,time; print('CHILD_READY',flush=True); time.sleep(.2); pathlib.Path(" + repr(str(marker)) + ").touch()"]
            with redirect_stdout(LiveOutput()):
                code = runner.stream_child(command, Path(d) / 'child.log', '静态模拟', REPO)
            self.assertEqual(code, 0)
            self.assertEqual(observed, [True])


if __name__ == '__main__':
    unittest.main(verbosity=2)
