"""CPU tests. No GPU/runtime differentiability is tested by this suite."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from audit_plastic_bomo_stage16a_r0 import digest, lineage, load, project_bbox, provenance, run, save, sha


class R0Tests(unittest.TestCase):
    def test_unknown_provenance_is_rejected(self):
        status, reasons = provenance({}, None, ROOT)
        self.assertEqual(status, 'REJECTED_PROVENANCE_UNKNOWN')
        self.assertTrue(reasons)

    def test_prelabel_needs_bound_human_review(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            record = base/'record.txt'; record.write_text('archived review record')
            row = {'image_sha256': 'image', 'annotation_sha256': 'label', 'annotation_line_index': 1, 'class_id': 0}
            evidence = {**row, 'annotation_source': 'PRELABEL', 'human_verified': False,
                        'supporting_file': 'record.txt', 'supporting_file_sha256': sha(record),
                        'supporting_record_locator': 'row1', 'provenance_review_completed': True,
                        'provenance_reviewer': 'reviewer'}
            self.assertEqual(provenance(row, evidence, base)[0], 'REJECTED_PRELABEL_UNVERIFIED')
            evidence['human_verified'] = True
            self.assertEqual(provenance(row, evidence, base)[0], 'ACCEPTED_HUMAN_VERIFIED')
            evidence['annotation_sha256'] = 'other'
            self.assertEqual(provenance(row, evidence, base)[0], 'REJECTED_PROVENANCE_UNKNOWN')
            evidence['annotation_sha256'] = 'label'; evidence['supporting_file_sha256'] = 'wrong'
            self.assertEqual(provenance(row, evidence, base)[0], 'REJECTED_PROVENANCE_UNKNOWN')

    def test_geometry_keeps_original_center_crop_no_truncation(self):
        row = project_bbox([.5, .5, .1, .2], [4096, 1024])
        self.assertEqual(row['resized_size'], [2048, 512])
        self.assertEqual(row['crop_offsets'], [768, 0])
        self.assertTrue(row['eligible'])
        self.assertFalse(project_bbox([.1, .5, .1, .2], [4096, 1024])['eligible'])
        self.assertFalse(project_bbox([.5, .5, .4, .2], [4096, 1024])['eligible'])
        odd = project_bbox([.5, .5, .01, .01], [777, 512])
        self.assertEqual(odd['crop_offsets'][0], 132)  # Python/torchvision rounding of 132.5
        self.assertNotEqual(project_bbox([.5, .5, .01, .01], [999, 700])['resize_scale'][0],
                            project_bbox([.5, .5, .01, .01], [999, 700])['resize_scale'][1])

    def test_connected_siblings_and_alias_no_physical_provenance_claim(self):
        images = {'img_12_pre_part_0.jpg': {'decoded_rgb_sha256': 'a'},
                  'img_12_pre_part_1.jpg': {'decoded_rgb_sha256': 'b'},
                  'alias.jpg': {'decoded_rgb_sha256': 'a'},
                  'other.jpg': {'decoded_rgb_sha256': 'z'}}
        graph, groups = lineage(images)
        self.assertEqual(groups['alias.jpg'], groups['img_12_pre_part_1.jpg'])
        self.assertNotEqual(groups['alias.jpg'], groups['other.jpg'])
        self.assertFalse(graph['physical_original_recovery_claimed'])
        self.assertTrue(all(not r['physical_original_recovered'] for r in graph['connected_components']))

    def test_complete_entry_gate_stop_emits_honest_not_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); images = base/'images'; labels = base/'labels'
            images.mkdir(); labels.mkdir()
            names = [f'real_{i:03d}.png' for i in range(138)]
            for i, name in enumerate(names):
                Image.new('RGB', (16, 16), (i, 30, 70)).save(images/name)
                cls = 0 if i < 88 else 1
                lines = [f'{cls} .5 .5 .1 .1']
                if 88 <= i < 118:
                    lines.append('1 .7 .7 .1 .1')
                (labels/(Path(name).stem+'.txt')).write_text('\n'.join(lines))
            save(base/'integrity.json', {'original': {'image_names': names}, 'original_counts': {'0': 88, '1': 80}})
            args = argparse.Namespace(protocol=ROOT/'configs/plastic_bomo_stage16a_r0.json', train_images=images,
                train_labels=labels, oof_integrity=base/'integrity.json', annotation_evidence=None,
                output=base/'audit', scope='LOCAL_TRAINING_ASSET_AUDIT')
            state = run(args)
            self.assertEqual(state['status'], 'ANNOTATION_PROVENANCE_INSUFFICIENT')
            self.assertEqual(state['sampling_count'], 0)
            self.assertFalse(state['STAGE16A_R_FORMAL_GENERATION_AUTHORIZED'])
            ann = load(args.output/'annotation_provenance_audit.json')
            self.assertEqual(ann['instance_count'], 168)
            self.assertEqual(ann['accepted_unique_source_groups_per_class'], {'flash': 0, 'black': 0})
            self.assertEqual(load(args.output/'differentiability_audit.json')['status'], 'NOT_RUN')
            self.assertEqual(digest(run(args)), digest(state))
            save(args.output/'stage16a_r0_status.json', {**state, 'sampling_count': 1})
            with self.assertRaises(RuntimeError):
                run(args)


if __name__ == '__main__':
    unittest.main()
