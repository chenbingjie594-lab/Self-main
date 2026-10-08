import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from freeze_plastic_bomo_stage17a_slots import assign_and_choose, project_box


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'class': c, 'source_id': f'{c}_{i:04d}', 'annotation_line_index': 1} for c in ('flash', 'black') for i in range(44)]
        self.normals = [{'normal_filename': f'{i:03d}.jpg'} for i in range(174)]

    def test_joint_hash_selection_deterministic_and_80(self):
        before = copy.deepcopy(self.sources)
        candidates, rows = assign_and_choose(self.sources, self.normals)
        self.assertEqual(len(candidates), 88); self.assertEqual(len(rows), 80)
        self.assertEqual(sum(r['source']['class'] == 'flash' for r in rows), 40)
        self.assertEqual((candidates, rows), assign_and_choose(list(reversed(self.sources)), list(reversed(self.normals))))
        self.assertEqual(self.sources, before)
        self.assertEqual(len({r['slot_id'] for r in rows}), 80)

    def test_not_enough_sources_no_replacement(self):
        with self.assertRaisesRegex(RuntimeError, 'INSUFFICIENT_FROZEN_SOURCE_COUNT_NO_REPLACEMENT'):
            assign_and_choose(self.sources[:44], self.normals)

    def test_cannot_change_normal_count(self):
        with self.assertRaisesRegex(RuntimeError, 'FROZEN_174_NORMALS_REQUIRED'):
            assign_and_choose(self.sources, self.normals[:-1])

    def test_padding_and_resize_bbox_geometry(self):
        r = {'bbox_xywh': [.5, .5, .1, .2], 'donor_id': 'x', 'class_id': 0}
        b = project_box(r, {'crop_size': 512, 'crop_center_xy': [32, 32]}, [64, 64])
        self.assertTrue(b['intersects_output']); self.assertFalse(b['clipping_applied'])
        self.assertAlmostEqual(b['yolo_xywh'][0], .5); self.assertAlmostEqual(b['yolo_xywh'][2], 6.4 / 512)

    def test_neighbor_outside_crop_not_labelled(self):
        r = {'bbox_xywh': [.9, .9, .02, .02], 'donor_id': 'other', 'class_id': 1}
        b = project_box(r, {'crop_size': 512, 'crop_center_xy': [200, 200]}, [4096, 1024])
        self.assertFalse(b['intersects_output']); self.assertIsNone(b['yolo_xywh'])


if __name__ == '__main__': unittest.main()
