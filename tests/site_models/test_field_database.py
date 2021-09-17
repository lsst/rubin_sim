import unittest
import os

from rubin_sim.data import get_data_dir
from rubin_sim.site_models import FieldsDatabase


class TestFieldDatabase(unittest.TestCase):

    def setUp(self):
        db_name = os.path.join(get_data_dir(), 'site_models', 'Fields.db')
        self.fields_db = FieldsDatabase(db_name)
        self.query = "select * from Field limit 2;"

    def test_basic_information_after_creation(self):
        self.assertIsNotNone(self.fields_db.connect)

    def test_opsim3_userregions(self):
        result = self.fields_db.get_opsim3_userregions(self.query)
        truth_result = """userRegion = 0.00,-90.00,0.03
userRegion = 180.00,-87.57,0.03"""
        self.assertEqual(result, truth_result)

    def test_get_ra_dec_arrays(self):
        ra, dec = self.fields_db.get_ra_dec_arrays(self.query)
        self.assertEqual(ra.size, 2)
        self.assertEqual(dec.size, 2)
        self.assertEqual(ra[1], 180.0)
        self.assertAlmostEqual(dec[1], -87.57, delta=1e-2)

    def test_get_id_ra_dec_arrays(self):
        fieldid, ra, dec = self.fields_db.get_id_ra_dec_arrays(self.query)
        self.assertEqual(ra[1], 180.0)
        self.assertEqual(fieldid[1], 2)
        self.assertNotEqual(fieldid[0], 0)
        self.assertAlmostEqual(dec[1], -87.57, delta=1e-2)

    def test_get_rows(self):
        rows = self.fields_db.get_rows(self.query)
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 8)

    def test_get_field_set(self):
        field_set = self.fields_db.get_field_set(self.query)
        truth_set = set()
        truth_set.add((1, 3.5, 0.0, -90.0, -57.068082, -27.128251,
                       -89.93121, -66.561358))
        truth_set.add((2, 3.5, 180.0, -87.568555, -57.663825, -24.756541,
                       -96.024547, -66.442665))
        self.assertEqual(len(field_set), 2)
        self.assertSetEqual(field_set, truth_set)

if __name__ == '__main__':
    unittest.main()
