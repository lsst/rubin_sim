import unittest

import rubin_sim.maf.batches as batches


class TestCommon(unittest.TestCase):
    def test_col_map(self):
        colmap = batches.col_map_dict("opsimv4")
        self.assertEqual(colmap["raDecDeg"], True)
        self.assertEqual(colmap["ra"], "fieldRA")
        colmap = batches.col_map_dict("fbs")
        self.assertEqual(colmap["raDecDeg"], True)
        self.assertEqual(colmap["skyBrightness"], "skyBrightness")

    def test_filter_list(self):
        filterlist, colors, orders, sqls, info_label = batches.common.filter_list(all=False, extra_sql=None)
        self.assertEqual(len(filterlist), 6)
        self.assertEqual(len(colors), 6)
        self.assertEqual(sqls["u"], 'filter = "u"')
        filterlist, colors, orders, sqls, info_label = batches.common.filter_list(all=True, extra_sql=None)
        self.assertIn("all", filterlist)
        self.assertEqual(sqls["all"], "")
        filterlist, colors, orders, sqls, info_label = batches.common.filter_list(
            all=True, extra_sql="night=3"
        )
        self.assertEqual(sqls["all"], "night=3")
        self.assertEqual(sqls["u"], '(night=3) and (filter = "u")')
        self.assertEqual(info_label["u"], "night=3 u band")
        filterlist, colors, orders, sqls, info_label = batches.common.filter_list(
            all=True, extra_sql="night=3", extra_info_label="night 3"
        )
        self.assertEqual(info_label["u"], "night 3 u band")


if __name__ == "__main__":
    unittest.main()
