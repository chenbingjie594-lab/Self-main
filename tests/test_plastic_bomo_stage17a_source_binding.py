from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_plastic_bomo_stage17a_source_binding import crop_parent


class SourceBindingTests(unittest.TestCase):
    def test_parent_from_explicit_crop_name(self):
        self.assertEqual(crop_parent('img_10012_pre_part_1_00_x581_y506_w34_h36.jpg'),
                         'img_10012_pre_part_1')

    def test_numbered_sources_are_not_parent_mappings(self):
        self.assertIsNone(crop_parent('000.jpg'))
        self.assertIsNone(crop_parent('blackinst_1.jpg'))

    def test_partial_name_not_accepted(self):
        self.assertIsNone(crop_parent('img_1_00_x1_y2_w3.jpg'))


if __name__ == '__main__':
    unittest.main()
