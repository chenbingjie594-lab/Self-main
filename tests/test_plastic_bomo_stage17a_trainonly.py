import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_plastic_bomo_stage17a_trainonly import build_pool, recovered_functions, rgb_hash, sha, verify_inputs, validate_protocol
from train_plastic_bomo_stage17a_trainonly import verify_base, verify_pool

EVIDENCE = ROOT / 'results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability'


class TrainOnlyTests(unittest.TestCase):
    def functions(self):
        return recovered_functions(EVIDENCE / 'historical_source_recovery/recovered_historical_crop_source.txt',
                                   EVIDENCE / 'historical_isolation_recovery/recovered_historical_mask_source.txt')

    def fixture(self, root):
        train = root / 'images/train'; train.mkdir(parents=True)
        val = root / 'images/val'; val.mkdir()
        labels = root / 'labels/train'; labels.mkdir(parents=True)
        rows = []
        for cls, cid in [('flash', 0), ('black', 1)]:
            image = train / (cls + '.jpg'); Image.new('RGB', (64, 64), (80 + cid * 60, 10, 30)).save(image)
            label = labels / (cls + '.txt'); label.write_text(f'{cid} 0.5 0.5 0.1 0.2\n')
            size, rgb = rgb_hash(image)
            rows.append({'class': cls, 'class_id': cid, 'image_id': image.name, 'image_sha256': sha(image),
                         'annotation_sha256': sha(label), 'decoded_rgb_sha256': rgb, 'image_size': size,
                         'annotation_line_index': 1, 'bbox_xywh': [.5, .5, .1, .2], 'donor_id': cls + ':line1'})
        Image.new('RGB', (64, 64), (10, 120, 200)).save(val / 'heldout.jpg')
        frozen = root / 'frozen.json'; frozen.write_text(json.dumps({'records': rows}))
        return frozen, train, val, rows

    def test_valid_frozen_isolation_and_pool(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); frozen, train, val, rows = self.fixture(root)
            checked, parents, heldout = verify_inputs(frozen, train, val, 2, {'flash': 1, 'black': 1})
            self.assertEqual(len(checked), 2); self.assertEqual(len(parents), 2); self.assertEqual(len(heldout), 1)
            records = build_pool(rows, train, root / 'pool', self.functions())
            self.assertEqual(len(records), 2)
            for r in records:
                self.assertEqual(sha(root / 'pool' / r['image_relative']), r['image_sha256'])
                self.assertEqual(Path(r['mask_relative']).stem, Path(r['image_relative']).stem)
                with Image.open(root / 'pool' / r['mask_relative']) as mask:
                    self.assertIsNotNone(mask.getbbox()); self.assertEqual(mask.size, (512, 512))

    def test_validation_alias_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            frozen, train, val, _ = self.fixture(Path(d))
            (val / 'renamed.jpg').write_bytes((train / 'flash.jpg').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'EXACT_IDENTITY_OVERLAP'):
                verify_inputs(frozen, train, val, 2, {'flash': 1, 'black': 1})

    def test_changed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); frozen, train, val, _ = self.fixture(root)
            (root / 'labels/train/flash.txt').write_text('0 0.6 0.5 0.1 0.2\n')
            with self.assertRaisesRegex(RuntimeError, 'FROZEN_PARENT_OR_LABEL_CHANGED'):
                verify_inputs(frozen, train, val, 2, {'flash': 1, 'black': 1})

    def test_union_mask_matches_historical_function(self):
        f = self.functions()
        local, box = f['regular_ellipse_mask'](64, 64, .5, .5, .1, .2, 2.8, 5.)
        self.assertEqual(local.dtype, np.dtype('bool')); self.assertTrue(local.any())
        crop = f['centered_square_crop_box'](64, 64, .5, .5, .1, .2, 512, 4.)
        self.assertEqual(crop[2], 512); self.assertEqual(crop[1], (224, 224))

    def test_local_preparation_cannot_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'trainonly_preparation_status.json').write_text(json.dumps({'status': 'TRAINONLY_POOL_PREPARED_ISOLATION_PASS', 'scope': 'LOCAL_TEST'}))
            with self.assertRaisesRegex(RuntimeError, 'SERVER_TRAINONLY_PREFLIGHT_REQUIRED'):
                verify_pool(root, root / 'cfg', root / 'frozen', root / 'yaml')

    def test_finetuned_initialization_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root / 'base'; base.mkdir(); audit = root / 'audit'; audit.mkdir()
            (base / 'weights.bin').write_bytes(b'finetuned')
            (audit / 'baseline_rebuild_preparation_audit.json').write_text(json.dumps({'base_model_files_sha256': {'weights.bin': 'original pretrained hash'}}))
            with self.assertRaisesRegex(RuntimeError, 'ORIGINAL_PRETRAINED_BASE_HASH_MISMATCH'):
                verify_base(base, audit)

    def test_fixed_training_not_relabelled_old_identity(self):
        new = json.loads((ROOT / 'configs/plastic_bomo_stage17a_trainonly.json').read_text())
        old = json.loads((ROOT / 'experiments/baseline_stage17a_rebuild_split70_s42.json').read_text())
        for key, value in new['training'].items():
            self.assertEqual(value, old['training'][key])
        self.assertFalse(new['formal_generation_authorized'])
        self.assertFalse(new['historical_checkpoint_equivalence_claimed'])

    def test_unannounced_mask_protocol_change_rejected(self):
        cfg = json.loads((ROOT / 'configs/plastic_bomo_stage17a_trainonly.json').read_text())
        cfg['mask']['black_scale'] = 2.8
        with self.assertRaisesRegex(RuntimeError, 'TRAINONLY_PROTOCOL_CHANGED'):
            validate_protocol(cfg)


if __name__ == '__main__':
    unittest.main()
