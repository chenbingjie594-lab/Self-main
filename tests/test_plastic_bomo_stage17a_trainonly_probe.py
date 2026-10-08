import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from probe_plastic_bomo_stage17a_trainonly import choose_trainonly_probes, paired_rgb_pass, safetensor_layout


class TrainOnlyProbeTests(unittest.TestCase):
    def test_metadata_order_and_five_conditions(self):
        rows = [{'class': c, 'source_id': f'{c}_{i:03d}'} for c in ('flash', 'black') for i in range(6)]
        chosen = choose_trainonly_probes(rows)
        self.assertEqual(chosen, choose_trainonly_probes(list(reversed(rows))))
        self.assertEqual(len(chosen), 5)
        self.assertEqual(sum(r['class'] == 'flash' for r in chosen), 3)

    def test_missing_sources_not_replaced(self):
        with self.assertRaisesRegex(RuntimeError, 'INSUFFICIENT_PROBE_SOURCES_NO_REPLACEMENT'):
            choose_trainonly_probes([{'class': 'flash', 'source_id': 'x'}])

    def test_exact_repeat_groups(self):
        rows = [{'bank': bank, 'probe_id': str(i), 'repeat': repeat, 'rgb_sha256': str(i)}
                for bank in ('A', 'B', 'C') for i in range(5) for repeat in (0, 1)]
        self.assertTrue(paired_rgb_pass(rows))
        wrong = copy.deepcopy(rows); wrong[-1]['rgb_sha256'] = 'different'
        self.assertFalse(paired_rgb_pass(wrong))
        self.assertFalse(paired_rgb_pass(rows[:-1]))
        wrong = copy.deepcopy(rows); wrong[-1]['repeat'] = 0
        self.assertFalse(paired_rgb_pass(wrong))

    def test_safetensor_header_and_truncation(self):
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'tiny.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 8)
            result = safetensor_layout(path)
            self.assertEqual(result['tensor_count'], 1); self.assertEqual(result['payload_bytes'], 8)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(RuntimeError, 'PAYLOAD_EXTENT_MISMATCH'):
                safetensor_layout(path)

    def test_source_syntax_and_no_formal_auth(self):
        import ast
        source = (ROOT / 'tools/probe_plastic_bomo_stage17a_trainonly.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn("'formal_generation_authorized': False", source)
        self.assertIn("'normal_lineage_formal_gate': 'NOT_ESTABLISHED'", source)
        self.assertIn('torch.use_deterministic_algorithms(True)', source)


if __name__ == '__main__':
    unittest.main()
