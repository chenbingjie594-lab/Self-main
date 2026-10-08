import hashlib
import io
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from verify_plastic_bomo_stage17a_exact_overlap import crop_geometry, exact_crops


class ExactOverlapTests(unittest.TestCase):
    def test_geometry_metadata_only(self):
        self.assertEqual(crop_geometry('img_1_00_x100_y200_w20_h40.jpg'), (110, 220))

    def test_invalid_filename_rejected(self):
        with self.assertRaises(RuntimeError):
            crop_geometry('000.jpg')

    def test_exact_byte_match_with_padding(self):
        im = Image.new('RGB', (100, 100), (20, 70, 110))
        b = io.BytesIO()
        im.crop((-206, -206, 306, 306)).save(b, format='JPEG', quality=95)
        hits = exact_crops(im, (50, 50), hashlib.sha256(b.getvalue()).hexdigest())
        self.assertTrue(any(h['center_xy'] == [50, 50] for h in hits))

    def test_wrong_hash_not_accepted(self):
        self.assertEqual(exact_crops(Image.new('RGB', (512, 512)), (256, 256), '0' * 64), [])


if __name__ == '__main__':
    unittest.main()
