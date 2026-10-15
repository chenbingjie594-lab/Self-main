"""Experiment1 stdlib/fixture tests; no model load, optimizer, sampling or metric forward."""
import ast
from collections import Counter
from copy import deepcopy
import csv
import importlib.metadata as md
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
from contextlib import redirect_stdout

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import chapter3_exp1_quality_common as c
import evaluate_plastic_bomo_chapter3_exp1 as e


def sample_results(fid_msdf=9., kid_msdf=.09, lpips_msdf=.2):
    results, slots = [], []
    for method in c.METHODS:
        for region in ('full_image', 'roi'):
            for cohort in ('overall', *c.CLASSES):
                for i, bank in enumerate(c.BANKS):
                    results.append({'method': method, 'region': region, 'cohort': cohort, 'bank': bank,
                        'FID': (10. if method == 'Raw SD2' else fid_msdf) + i,
                        'KID': (.1 if method == 'Raw SD2' else kid_msdf) + i * .001})
        for i in range(80):
            slots.append({'method': method, 'slot_id': f's{i}', 'class': 'flash' if i < 40 else 'black',
                          'LPIPS_slot': .3 if method == 'Raw SD2' else lpips_msdf})
    return results, slots


class Protocol(unittest.TestCase):
    def test_no_model_packages_imported(self):
        for name in ('torch', 'torch_fidelity', 'lpips', 'ultralytics', 'diffusers'):
            self.assertNotIn(name, sys.modules)

    def test_all_frozen_metadata_pins(self):
        c.verify_metadata(c.load(c.CONFIG))

    def test_crop_25_percent_on_each_side(self):
        self.assertEqual(c.crop_box([.5, .5, .2, .4], 100, 100), [35, 20, 65, 80])
        self.assertEqual(c.crop_box([.5, .5, .4, .4], 100, 100), [20, 20, 80, 80])

    def test_crop_clips_without_min_size_enlargement(self):
        self.assertEqual(c.crop_box([.01, .01, .1, .1], 100, 100), [0, 0, 9, 9])
        tiny = c.crop_box([.5, .5, .001, .001], 100, 100)
        self.assertLessEqual(tiny[2] - tiny[0], 2)

    def test_invalid_bbox_or_changed_margin_rejected(self):
        for box in ([.5, .5, 0, .1], [.5, .5, .1, -1], [float('nan'), .5, .1, .1]):
            with self.assertRaises(c.Stop):
                c.crop_box(box, 100, 100)
        with self.assertRaises(c.Stop):
            c.crop_box([.5, .5, .1, .1], 100, 100, .5)

    def test_metadata_imports_and_help_do_not_load_models(self):
        p = subprocess.run([sys.executable, '-B', str(REPO / 'tools/evaluate_plastic_bomo_chapter3_exp1.py'), '--help'],
            cwd=REPO, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('prepare,evaluate,all', p.stdout)

    def test_no_training_or_custom_metric_formula(self):
        for path in (REPO / 'tools/chapter3_exp1_quality_common.py', REPO / 'tools/evaluate_plastic_bomo_chapter3_exp1.py'):
            text = path.read_text(encoding='utf-8')
            tree = ast.parse(text)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            forbidden = {'backward', 'optimizer_step', 'fit', 'sqrtm', 'eigvals', 'inception_v3', 'unbiased_mmd', 'polynomial_kernel'}
            self.assertFalse([n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in forbidden])
            self.assertNotIn('from ultralytics', text)
            self.assertNotIn('from diffusers', text)
            self.assertNotIn('pip install', text)

    def test_official_metric_delegation_is_used(self):
        text = (REPO / 'tools/evaluate_plastic_bomo_chapter3_exp1.py').read_text(encoding='utf-8')
        for marker in ('torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric',
                       'torch_fidelity.metric_kid import kid_features_to_metric', "lib['fid'](", "lib['kid']("):
            self.assertIn(marker, text)

    def test_config_has_no_utility_or_selector(self):
        cfg = c.load(c.CONFIG)
        self.assertEqual(cfg['crop']['context_margin_each_side'], .25)
        self.assertEqual(cfg['distribution_metrics']['KID']['role'], 'PRIMARY, especially defect region')
        for key in ('generator_training_count', 'generation_count', 'detector_training_count', 'BootstrapGuard', 'DWBG', 'selector'):
            self.assertEqual(cfg[key], 0)
        self.assertFalse(cfg['reference_never_seen_untouched_test'])


class StatisticsAndTables(unittest.TestCase):
    def test_mean_sd_ddof1(self):
        row = c.mean_sd([1., 2., 3.])
        self.assertEqual((row['mean'], row['sd'], row['ddof']), (2., 1., 1))

    def test_both_distribution_metrics_lower(self):
        s = c.summarize(*sample_results())
        self.assertEqual(s['status'], 'MSDF_VISUAL_METRICS_IMPROVED')
        self.assertLess(s['comparisons']['overall']['LPIPS_msdf_minus_raw'], 0)
        # Improved distribution metrics do not make declining diversity disappear.
        self.assertEqual(s['methods']['MSDF']['LPIPS']['overall']['n'], 80)

    def test_mixed_and_nonimproved(self):
        self.assertEqual(c.summarize(*sample_results(fid_msdf=11))['status'], 'MIXED_VISUAL_METRICS')
        self.assertEqual(c.summarize(*sample_results(fid_msdf=11, kid_msdf=.11))['status'], 'MSDF_VISUAL_METRICS_NOT_IMPROVED')

    def test_negative_unbiased_kid_not_clipped(self):
        summary = c.summarize(*sample_results(kid_msdf=-.01))
        self.assertLess(summary['methods']['MSDF']['roi']['overall']['KID']['mean'], 0)

    def test_missing_bank_cannot_be_pooled_away(self):
        rows, slots = sample_results()
        rows = [r for r in rows if r['bank'] != 'C']
        with self.assertRaises(c.Stop):
            c.summarize(rows, slots)

    def test_duplicate_lpips_slot_rejected(self):
        rows, slots = sample_results()
        slots[1]['slot_id'] = slots[0]['slot_id']
        with self.assertRaises(c.Stop):
            c.summarize(rows, slots)

    def test_csv_paper_shape_and_summary_values(self):
        s = c.summarize(*sample_results())
        with tempfile.TemporaryDirectory() as d:
            c.paper_tables(d, s)
            with (Path(d) / 'paper_table_generation_quality.csv').open(encoding='utf-8-sig', newline='') as f:
                table = list(csv.reader(f))
            self.assertEqual(table[0], ['Method', 'FID↓', 'KID↓', 'LPIPS↑'])
            self.assertEqual(len(table), 3)
            self.assertEqual([r[0] for r in table[1:]], list(c.METHODS))
            self.assertIn('±', table[1][1])
            with (Path(d) / 'paper_table_generation_quality_classwise.csv').open(encoding='utf-8-sig', newline='') as f:
                classes = list(csv.reader(f))
            self.assertEqual(len(classes[0]), 7)
            self.assertEqual(len(classes), 3)

    def test_unavailable_is_not_zero_and_no_fake_chart(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            e.pending_outputs(out, 'DEPENDENCY_REQUIRED', ['test-only missing dependency'])
            state = c.load(out / 'experiment1_status.json')
            self.assertIsNone(state['ROI_KID_raw_mean'])
            self.assertFalse(state['full_experiment_complete'])
            self.assertFalse((out / 'generation_quality_metrics.png').exists())
            with (out / 'paper_table_generation_quality.csv').open(encoding='utf-8-sig', newline='') as f:
                self.assertEqual(list(csv.reader(f))[1][1:], ['', '', ''])

    def test_completed_results_not_reset_to_pending(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            c.save(out / 'experiment1_status.json', {'full_experiment_complete': True})
            with self.assertRaises(c.Stop):
                e.pending_outputs(out, 'DEPENDENCY_REQUIRED', [])


class DependencyAndLibraryOptions(unittest.TestCase):
    def test_missing_dependencies_never_import_or_install(self):
        def missing(name):
            raise md.PackageNotFoundError(name)
        with patch.object(c.md, 'version', side_effect=missing):
            dep = c.dependencies(c.load(c.CONFIG))
        self.assertEqual(dep['status'], 'DEPENDENCY_REQUIRED')
        self.assertEqual(dep['package_install_or_upgrade_count'], 0)
        self.assertEqual(dep['model_load_count'], 0)
        self.assertNotIn('torch', sys.modules)

    def test_kid_known_library_version_kwargs(self):
        cfg = c.load(c.CONFIG)
        old = e.kid_options(cfg, '0.3.0', 27, 40)
        new = e.kid_options(cfg, '0.4.0', 27, 40)
        self.assertEqual(old['kid_subset_size'], 27)
        self.assertEqual(old['kid_subsets'], 100)
        self.assertEqual(old['kid_degree'], 3)
        self.assertEqual(new['kid_kernel'], 'poly')
        self.assertEqual(new['kid_kernel_poly_degree'], 3)
        with self.assertRaises(c.Stop):
            e.kid_options(cfg, 'unexpected', 27, 40)

    def test_subset_indices_same_for_methods_and_all_images_included(self):
        import numpy as np
        for nreal, nfake in ((48, 80), (60, 80), (33, 40), (27, 40), (16, 40)):
            opts = e.kid_options(c.load(c.CONFIG), '0.3.0', nreal, nfake)
            first = e.subset_coverage(np, nreal, nfake, opts)
            second = e.subset_coverage(np, nreal, nfake, opts)
            self.assertEqual(first, second)
            self.assertGreater(min(first['real_usage_counts']), 0)
            self.assertGreater(min(first['generated_usage_counts']), 0)

    def test_freeze_refuses_version_weight_or_device_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'runtime.json'
            e.freeze_json(path, {'version': '0.3.0'})
            e.freeze_json(path, {'version': '0.3.0'})
            with self.assertRaises(c.Stop):
                e.freeze_json(path, {'version': '0.4.0'})

    def test_network_guard_restores_socket(self):
        original = e.socket.socket.connect
        sock = e.socket.socket(e.socket.AF_INET, e.socket.SOCK_STREAM)
        try:
            with e.offline_metrics():
                with self.assertRaises(c.Stop):
                    sock.connect(('127.0.0.1', 1))
            self.assertIs(e.socket.socket.connect, original)
        finally:
            sock.close()


class RealFrozenMetadata(unittest.TestCase):
    """Only JSON is decoded; heavy image bytes can be stubbed for schema tests."""
    def test_real_reference_metadata_and_original_boxes(self):
        cfg = c.load(c.CONFIG)
        ref = c.reference_records(cfg)
        self.assertEqual(ref['image_count'], 48)
        self.assertEqual(ref['instance_class_counts'], {'flash': 33, 'black': 27})
        self.assertFalse(ref['never_seen_untouched_test'])
        self.assertEqual(len(ref['instances']), 60)
        for r in ref['instances']:
            self.assertEqual(r['roi_xyxy'], c.crop_box(r['bbox'], *r['canvas_wh']))

    def test_raw_msdf_real_schema_pairing_preserves_all_slots(self):
        cfg = c.load(c.CONFIG)
        with patch.object(c, 'checked_file', side_effect=lambda p, h: Path(p)):
            rows = c.matched_records(cfg)
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({r['slot_id'] for r in rows}), 80)
        self.assertEqual(Counter((r['bank'], r['class']) for r in rows), Counter({(b, cls): 40 for b in c.BANKS for cls in c.CLASSES}))
        for r in rows:
            self.assertFalse(Path(r['raw_image_path']).is_absolute())
            self.assertFalse(Path(r['msdf_image_path']).is_absolute())
            self.assertIn('conditioning_normal_id', r)

    def test_wrong_slot_metadata_is_not_silently_matched(self):
        cfg = c.load(c.CONFIG)
        actual_load = c.load
        def tamper(path):
            obj = actual_load(path)
            if str(path).endswith('formal_bank_A_manifest.json'):
                obj['records'][0]['frozen_slot']['bbox'][0] += .0001
            return obj
        with patch.object(c, 'load', side_effect=tamper), patch.object(c, 'checked_file', side_effect=lambda p, h: Path(p)):
            with self.assertRaises(c.Stop):
                c.matched_records(cfg)

    def test_input_items_1068_and_no_bank_pooling(self):
        cfg = c.load(c.CONFIG)
        with patch.object(c, 'checked_file', side_effect=lambda p, h: Path(p)):
            pairs = {'records': c.matched_records(cfg)}
        ref = c.reference_records(cfg)
        items = e.image_items(pairs, ref)
        self.assertEqual(len(items), 1068)
        for method in c.METHODS:
            for bank in c.BANKS:
                for region in ('full_image', 'roi'):
                    self.assertEqual(sum(x['role'] == method and x.get('bank') == bank and x['region'] == region for x in items), 80)


class LibraryDelegation(unittest.TestCase):
    def test_publication_pipeline_with_temporary_fixture_only(self):
        records, slots = sample_results()
        for row in records:
            row.update(real_count=48, generated_count=80, KID_subset_std=.002)
        summary = c.summarize(records, slots)
        reference = {'image_count': 48, 'instance_count': 60,
                     'instance_class_counts': {'flash': 33, 'black': 27}}
        with tempfile.TemporaryDirectory(prefix='exp1_test_fixture_') as d:
            out = Path(d)
            c.save(out / 'metric_protocol.json', c.load(c.CONFIG))
            e.publish_results(out, records, slots, summary, reference, {})
            state = c.load(out / 'experiment1_status.json')
            self.assertTrue(state['full_experiment_complete'])
            self.assertEqual(state['ROI_KID_msdf_mean'], summary['methods']['MSDF']['roi']['overall']['KID']['mean'])
            self.assertEqual(state['generation_count'], 0)
            self.assertEqual(state['GitHub_submission'], 'NOT_PERFORMED_BY_EVALUATOR')
            self.assertEqual(len(c.load(out / 'fid_bankwise.json')['records']), 36)
            ledger = c.load(out / 'experiment1_artifact_manifest.json')['sha256']
            for name, digest in ledger.items():
                self.assertEqual(c.sha(out / name), digest)
            from PIL import Image
            with Image.open(out / 'generation_quality_metrics.png') as picture:
                picture.load()
                self.assertEqual(picture.format, 'PNG')
                self.assertGreater(picture.width, 1500)
                self.assertGreater(picture.height, 500)
            self.assertNotIn('torch', sys.modules)

    def test_36_distribution_calls_delegate_with_identical_paired_inputs(self):
        import numpy as np
        cfg = c.load(c.CONFIG)
        with patch.object(c, 'checked_file', side_effect=lambda p, h: Path(p)):
            pairs = {'records': c.matched_records(cfg)}
        ref = c.reference_records(cfg)
        items = e.image_items(pairs, ref)
        class FakeTensor:
            def __init__(self, rows): self.rows = list(rows)
            def double(self): return self
            def cpu(self): return self
            def __len__(self): return len(self.rows)
        class FakeTorch:
            stack = staticmethod(FakeTensor)
        calls = []
        def kid(real, fake, **kwargs):
            calls.append((len(real), len(fake), kwargs))
            return {'kernel_inception_distance_mean': -.001, 'kernel_inception_distance_std': .0002}
        fid = Mock(return_value={'frechet_inception_distance': 12.5})
        lib = {'torch': FakeTorch, 'np': np, 'fid_stats': Mock(return_value={'TEST_FIXTURE': True}), 'fid': fid, 'kid': kid}
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            records = e.distribution_metrics(Path(d), items, {r['key']: r['key'] for r in items}, cfg, lib,
                                              {'package_versions': {'torch-fidelity': '0.3.0'}})
            sampling = c.load(Path(d) / 'kid_subset_sampling_audit.json')
        self.assertEqual(len(records), 36)
        self.assertEqual(fid.call_count, 36)
        self.assertEqual(len(calls), 36)
        self.assertEqual(len(sampling), 18)
        self.assertTrue(all(r['KID'] < 0 for r in records))
        for left, right in zip(calls[::2], calls[1::2]):
            self.assertEqual(left, right)

    def test_import_failure_does_not_install_or_create_metrics(self):
        cfg = c.load(c.CONFIG)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            c.save(out / 'experiment1_status.json', c.base_status())
            dep = {'status': 'PASS', 'errors': [], 'package_versions': {}, 'weights': {}}
            with patch.object(e, 'runtime_modules', side_effect=ImportError('TEST ABI unavailable')):
                with self.assertRaises(c.Stop) as caught:
                    e.evaluate(out, cfg, {}, {}, dep, 'cpu')
            self.assertEqual(caught.exception.status, 'DEPENDENCY_REQUIRED')
            self.assertEqual(c.load(out / 'dependency_report.json')['status'], 'DEPENDENCY_REQUIRED')
            self.assertFalse((out / 'distribution_metrics_raw.json').exists())

    def test_plot_labels_diversity_not_higher_quality(self):
        text = (REPO / 'tools/evaluate_plastic_bomo_chapter3_exp1.py').read_text(encoding='utf-8')
        self.assertIn('LPIPS appearance diversity', text)
        self.assertNotIn('LPIPS higher quality', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
