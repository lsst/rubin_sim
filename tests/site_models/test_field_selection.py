import unittest
from rubin_sim.site_models import FieldSelection


class TestFieldSelection(unittest.TestCase):

    def setUp(self):
        self.fs = FieldSelection()

        self.truth_base_query = "select * from Field"
        self.truth_galactic_exclusion = '(abs(fieldGB) > (10.0 - '\
                                        '(9.9 * abs(fieldGL)) / 90.0))'
        self.truth_galactic_region = '(abs(fieldGB) <= (10.0 - '\
                                     '(9.9 * abs(fieldGL)) / 90.0))'
        self.truth_normal_ra_region = 'fieldRA between 90.0 and 270.0'
        self.truth_cross_region = '(fieldRA between 270.0 and 360 or '\
                                  'fieldRA between 0 and 90.0)'
        self.truth_normal_dec_region = 'fieldDec between -90.0 and -61.0'
        self.truth_user_regions = 'fieldId=2 or fieldId=256 or fieldId=2935'

    def test_base_select(self):
        self.assertEqual(self.fs.base_select(), self.truth_base_query)

    def test_finish_query(self):
        query = "silly query"
        self.assertEqual(self.fs.finish_query(query), query + ";")

    def test_galactic_region(self):
        self.assertEqual(self.fs.galactic_region(10.0, 0.1, 90.0,
                                                 exclusion=True),
                         self.truth_galactic_exclusion)
        self.assertEqual(self.fs.galactic_region(10.0, 0.1, 90.0),
                         self.truth_galactic_region)

    def test_select_region(self):
        self.assertEqual(self.fs.select_region("RA", 90.0, 270.0),
                         self.truth_normal_ra_region)
        self.assertEqual(self.fs.select_region("RA", 270.0, 90.0),
                         self.truth_cross_region)

    def test_combine_queries(self):
        query1 = self.fs.select_region("RA", 90.0, 270.0)
        query2 = self.fs.select_region("Dec", -90.0, -61.0)
        combiners = ("and",)

        truth_query_parts = [self.truth_base_query]
        truth_query_parts.append("where")
        truth_query_parts.append(self.truth_normal_ra_region)
        truth_query_parts.append("and")
        truth_query_parts.append(self.truth_normal_dec_region)
        truth_query_parts.append("order by fieldId")

        truth_query = " ".join(truth_query_parts) + ";"
        self.assertEqual(self.fs.combine_queries(query1, query2,
                                                 combiners=combiners),
                         truth_query)

    def test_bad_combine_queries(self):
        query1 = self.fs.select_region("RA", 90.0, 270.0)
        query2 = self.fs.select_region("Dec", -90.0, -61.0)
        with self.assertRaises(RuntimeError):
            self.fs.combine_queries(query1, query2)

        combiners = ("and",)
        with self.assertRaises(RuntimeError):
            self.fs.combine_queries(query1, combiners=combiners)

    def test_get_all_fields(self):
        self.assertEqual(self.fs.get_all_fields(),
                         self.truth_base_query + ";")

    def test_get_user_regions(self):
        user_regions = (2, 256, 2935)
        query = self.fs.select_user_regions(user_regions)
        combiners = ()

        truth_query_parts = [self.truth_base_query]
        truth_query_parts.append("where")
        truth_query_parts.append(self.truth_user_regions)
        truth_query_parts.append("order by fieldId")

        truth_query = " ".join(truth_query_parts) + ";"
        self.assertEqual(self.fs.combine_queries(query, combiners=combiners),
                         truth_query)


if __name__ == '__main__':
    unittest.main()
