import matplotlib
matplotlib.use("Agg")
import unittest
import rubin_sim.maf.metrics as metrics


class TestBaseMetric(unittest.TestCase):

    def testReduceDict(self):
        """Test that reduce dictionary is created."""
        testmetric = metrics.BaseMetric('testcol')
        self.assertEqual(list(testmetric.reduceFuncs.keys()), [])

    def testMetricName(self):
        """Test that metric name is set appropriately automatically and explicitly"""
        # Test automatic setting of metric name
        testmetric = metrics.BaseMetric('testcol')
        self.assertEqual(testmetric.name, 'Base testcol')
        testmetric = metrics.BaseMetric(['testcol1', 'testcol2'])
        self.assertEqual(testmetric.name, 'Base testcol1, testcol2')
        # Test explicit setting of metric name
        testmetric = metrics.BaseMetric('testcol', metricName='Test')
        self.assertEqual(testmetric.name, 'Test')

    def testColRegistry(self):
        """Test column registry adds to colRegistry as expected"""
        # Clear the registry to make sure we start clear
        colreg = metrics.ColRegistry()
        colreg.clearReg()

        cols = 'onecolumn'
        colset = set()
        colset.add(cols)
        testmetric = metrics.BaseMetric(cols)
        # Class registry should have dictionary with values = set of columns for metric class
        for item in colset:
            self.assertIn(item, testmetric.colRegistry.colSet)
        cols = ['onecolumn', 'twocolumn']
        colset.add('twocolumn')
        testmetric = metrics.BaseMetric(cols)
        for item in colset:
            self.assertIn(item, testmetric.colRegistry.colSet)
        # Test with additional (different) metric
        cols = 'twocolumn'
        testmetric2 = metrics.MeanMetric(cols)
        for item in colset:
            self.assertIn(item, testmetric2.colRegistry.colSet)

        # test that the registry can be cleared
        colreg.clearReg()
        assert(len(colreg.colSet) == 0)
        assert(len(colreg.dbSet) == 0)
        assert(len(colreg.stackerDict) == 0)

    def testMetricDtype(self):
        """Test that base metric data value type set appropriately"""
        cols = 'onecolumn'
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.metricDtype, 'float')
        testmetric = metrics.BaseMetric(cols, metricDtype='object')
        self.assertEqual(testmetric.metricDtype, 'object')

    def testUnits(self):
        """Test unit setting (including units set by utils.getColInfo)"""
        cols = 'onecolumn'
        # Test for column not in colInfo, units not set by hand.
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, '')
        # Test for column not in colInfo, units set by hand.
        testmetric = metrics.BaseMetric(cols, units='Test')
        self.assertEqual(testmetric.units, 'Test')
        # Test for column in colInfo (looking up units in colInfo)
        cols = 'finSeeing'
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec')
        # Test for column in colInfo but units overriden
        testmetric = metrics.BaseMetric(cols, units='Test')
        self.assertEqual(testmetric.units, 'Test')
        # Test for multiple columns not in colInfo
        cols = ['onecol', 'twocol']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, '')
        # Test for multiple columns in colInfo
        cols = ['finSeeing', 'filtSkyBrightness']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec mag/sq arcsec')
        # Test for multiple columns, only one in colInfo
        cols = ['finSeeing', 'twocol']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec ')
        # Test for multiple columns both in colInfo but repeated
        cols = ['finSeeing', 'finSeeing']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec arcsec')


if __name__ == "__main__":
    unittest.main()
