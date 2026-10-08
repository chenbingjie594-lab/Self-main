import ast
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from probe_plastic_bomo_stage17a_rebuilt import choose_probes, historical_runtime


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.classes = {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}
        self.split = {'categories': {'Plastic_Bomo': {
            folder: {'train': [{'image': f'{i}.jpg', 'mask': f'{i}_mask.png'}
                               for i in range(10)]} for folder in self.classes.values()}}}

    def test_five_probes_and_both_classes(self):
        rows = choose_probes(self.split, self.classes)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(r['class'] == 'flash' for r in rows), 3)
        self.assertEqual(len({r['probe_id'] for r in rows}), 5)

    def test_order_independent_no_scores(self):
        reverse = copy.deepcopy(self.split)
        for row in reverse['categories']['Plastic_Bomo'].values():
            row['train'].reverse()
        self.assertEqual(choose_probes(self.split, self.classes), choose_probes(reverse, self.classes))

    def test_no_replacement_when_insufficient(self):
        self.split['categories']['Plastic_Bomo']['02_Big_black_spots']['train'] = []
        with self.assertRaises(RuntimeError):
            choose_probes(self.split, self.classes)

    def test_historical_import_restores_argv(self):
        from unittest.mock import patch
        previous = sys.argv
        with patch('probe_plastic_bomo_stage17a_rebuilt.importlib.import_module', side_effect=ValueError):
            with self.assertRaises(ValueError):
                historical_runtime(Path('model'), Path('config.json'))
        self.assertIs(sys.argv, previous)

    def test_inference_defaults_for_disabled_baseline(self):
        tree = ast.parse((ROOT / 'inference.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inpaint')
        defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):], fn.args.defaults))
        for key in ('anomaly_strength', 'eta', 'eta_mask', 'mdap_strength', 'blur_factor'):
            self.assertEqual(ast.literal_eval(defaults[key]), 0)
        text = (ROOT / 'tools/probe_plastic_bomo_stage17a_rebuilt.py').read_text(encoding='utf-8')
        self.assertIn('runtime.inpaint(', text)
        self.assertIn('torch.use_deterministic_algorithms(True)', text)


if __name__ == '__main__':
    unittest.main()
