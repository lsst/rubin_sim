import math
import unittest

from rubin_sim.site_models import Field

class TestField(unittest.TestCase):

    def setUp(self):
        self.field = Field(1, 30.0, -30.0, -45.0, 45.0, 60.0, -60.0, 3.0)

    def test_basic_information_after_creation(self):
        self.assertEqual(self.field.fid, 1)
        self.assertEqual(self.field.ra, 30.0)
        self.assertEqual(self.field.dec, -30.0)
        self.assertEqual(self.field.gl, -45.0)
        self.assertEqual(self.field.gb, 45.0)
        self.assertEqual(self.field.el, 60.0)
        self.assertEqual(self.field.eb, -60.0)
        self.assertEqual(self.field.fov, 3.0)
        self.assertEqual(self.field.ra_rad, math.pi / 6)
        self.assertEqual(self.field.dec_rad, -math.pi / 6)
        self.assertEqual(self.field.gl_rad, -math.pi / 4)
        self.assertEqual(self.field.gb_rad, math.pi / 4)
        self.assertEqual(self.field.el_rad, math.pi / 3)
        self.assertEqual(self.field.eb_rad, -math.pi / 3)
        self.assertAlmostEqual(self.field.fov_rad, math.pi / 60, delta=1e-7)

    def test_create_from_db_row(self):
        row = [1, 3.0, 30.0, -30.0, -45.0, 45.0, 60.0, -60.0]
        field = Field.from_db_row(row)
        self.assertEqual(field.fid, 1)
        self.assertEqual(field.ra, 30.0)
        self.assertEqual(field.ra_rad, math.pi / 6)

if __name__ == '__main__':
    unittest.main()
