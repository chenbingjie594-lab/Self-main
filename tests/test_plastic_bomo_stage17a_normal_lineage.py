from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_plastic_bomo_stage17a_normal_lineage import source_key, xml_source, historical_filter


class NormalLineageTests(unittest.TestCase):
    def test_explicit_partition_group_not_arbitrary_prefix(self):
        self.assertEqual(source_key('img_15887_pre_part_0'), source_key('img_15887_pre_part_1'))
        self.assertIsNone(source_key('blackinst_1'))
        self.assertIsNone(source_key('img_15887'))
        self.assertNotEqual(source_key('img_15887_pre_part_0'), source_key('img_15888_pre_part_0'))

    def test_xml_source_metadata_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'normal.xml'
            p.write_text('<annotation><filename>img_pre.jpg</filename><path>/original/img_pre.jpg</path><object><name>unused</name></object></annotation>')
            r = xml_source(p)
            self.assertEqual(r['original_filename'], 'img_pre.jpg')
            self.assertEqual(r['original_path'], '/original/img_pre.jpg')
            self.assertNotIn('object', r)

    def test_unchanged_normal_filter(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new('RGB', (64, 64), (100, 100, 100)).save(root / '001.jpg')
            Image.new('RGB', (64, 64), (0, 0, 0)).save(root / '000.jpg')
            accepted = historical_filter(root, {'enabled': True, 'black_threshold': 20, 'min_mean_luminance': 30., 'min_nonblack_ratio': .6})
            self.assertEqual([Path(x).name for x in accepted], ['001.jpg'])


if __name__ == '__main__':
    unittest.main()
