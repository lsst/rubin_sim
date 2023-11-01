import unittest

import rubin_sim.maf.metrics as metrics


class TestBaseMetric(unittest.TestCase):
    def test_reduce_dict(self):
        """Test that reduce dictionary is created."""
        testmetric = metrics.BaseMetric("testcol")
        self.assertEqual(list(testmetric.reduce_funcs.keys()), [])

    def test_metric_name(self):
        """Test that metric name is set appropriately automatically
        and when explicitly passed.
        """
        # Test automatic setting of metric name
        testmetric = metrics.BaseMetric("testcol")
        self.assertEqual(testmetric.name, "Base testcol")
        testmetric = metrics.BaseMetric(["testcol1", "testcol2"])
        self.assertEqual(testmetric.name, "Base testcol1, testcol2")
        # Test explicit setting of metric name
        testmetric = metrics.BaseMetric("testcol", metric_name="Test")
        self.assertEqual(testmetric.name, "Test")

    def test_col_registry(self):
        """Test column registry adds to colRegistry as expected"""
        # Clear the registry to make sure we start clear
        colreg = metrics.ColRegistry()
        colreg.clear_reg()

        cols = "onecolumn"
        colset = set()
        colset.add(cols)
        testmetric = metrics.BaseMetric(cols)
        # Class registry should have dictionary with values =
        # set of columns for metric class
        for item in colset:
            self.assertIn(item, testmetric.col_registry.col_set)
        cols = ["onecolumn", "twocolumn"]
        colset.add("twocolumn")
        testmetric = metrics.BaseMetric(cols)
        for item in colset:
            self.assertIn(item, testmetric.col_registry.col_set)
        # Test with additional (different) metric
        cols = "twocolumn"
        testmetric2 = metrics.MeanMetric(cols)
        for item in colset:
            self.assertIn(item, testmetric2.col_registry.col_set)

        # test that the registry can be cleared
        colreg.clear_reg()
        assert len(colreg.col_set) == 0
        assert len(colreg.db_set) == 0
        assert len(colreg.stacker_dict) == 0

    def test_metric_dtype(self):
        """Test that base metric data value type set appropriately"""
        cols = "onecolumn"
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.metric_dtype, "float")
        testmetric = metrics.BaseMetric(cols, metric_dtype="object")
        self.assertEqual(testmetric.metric_dtype, "object")

    def test_units(self):
        """Test unit setting (including units set by utils.getColInfo)"""
        cols = "onecolumn"
        # Test for column not in colInfo, units not set by hand.
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, "")
        # Test for column not in colInfo, units set by hand.
        testmetric = metrics.BaseMetric(cols, units="Test")
        self.assertEqual(testmetric.units, "Test")
        # Test for column in colInfo (looking up units in colInfo)
        cols = "finSeeing"
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, "arcsec")
        # Test for column in colInfo but units overriden
        testmetric = metrics.BaseMetric(cols, units="Test")
        self.assertEqual(testmetric.units, "Test")
        # Test for multiple columns not in colInfo
        cols = ["onecol", "twocol"]
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, "")
        # Test for multiple columns in colInfo
        cols = ["finSeeing", "filtSkyBrightness"]
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, "arcsec mag/sq arcsec")
        # Test for multiple columns, only one in colInfo
        cols = ["finSeeing", "twocol"]
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, "arcsec ")
        # Test for multiple columns both in colInfo but repeated
        cols = ["finSeeing", "finSeeing"]
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, "arcsec arcsec")


if __name__ == "__main__":
    unittest.main()
