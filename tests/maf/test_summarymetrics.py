import matplotlib

matplotlib.use("Agg")
import numpy as np
import healpy as hp
import unittest
import rubin_sim.maf.metrics as metrics


class TestSummaryMetrics(unittest.TestCase):
    def testTableFractionMetric(self):
        """Test the table summary metric"""
        metricdata1 = np.arange(0, 1.5, 0.02)
        metricdata = np.array(list(zip(metricdata1)), dtype=[("testdata", "float")])
        for nbins in [10, 20, 5]:
            metric = metrics.TableFractionMetric("testdata", nbins=nbins)
            table = metric.run(metricdata)
            self.assertEqual(len(table), nbins + 3)
            self.assertEqual(table["value"][0], np.size(np.where(metricdata1 == 0)[0]))
            self.assertEqual(table["value"][-1], np.size(np.where(metricdata1 > 1)[0]))
            self.assertEqual(table["value"][-2], np.size(np.where(metricdata1 == 1)[0]))
            self.assertEqual(table["value"].sum(), metricdata1.size)

    def testIdentityMetric(self):
        """Test identity metric."""
        dv = np.arange(0, 10, 0.5)
        dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        testmetric = metrics.IdentityMetric("testdata")
        np.testing.assert_equal(testmetric.run(dv), dv["testdata"])

    def testfONv(self):
        """
        Test the fONv metric.
        """
        nside = 128
        npix = hp.nside2npix(nside)
        names = ["metricdata"]
        types = [int]
        data = np.zeros(npix, dtype=list(zip(names, types)))
        data["metricdata"] += 826
        metric = metrics.fONv(col="ack", nside=nside, Nvisit=825, Asky=18000.0)
        slicePoint = {"sid": 0}
        result = metric.run(data, slicePoint)
        # result is recarray with 'min' and 'median' number of visits
        # over the Asky area.
        # All pixels had 826 visits, so that is min and median here.
        min_nvis = result["value"][np.where(result["name"] == "MinNvis")]
        median_nvis = result["value"][np.where(result["name"] == "MedianNvis")]
        self.assertEqual(min_nvis, 826)
        self.assertEqual(median_nvis, 826)
        # Now update so that 13k of sky is 826, rest 0.
        deginsph = 41253
        npix_nk = int(npix * (13000.0 / deginsph))
        data["metricdata"] = 0
        data["metricdata"][:npix_nk] = 826
        result = metric.run(data, slicePoint)
        min_nvis = result["value"][np.where(result["name"] == "MinNvis")]
        median_nvis = result["value"][np.where(result["name"] == "MedianNvis")]
        self.assertEqual(min_nvis, 0)
        self.assertEqual(median_nvis, 826)

    def testfOArea(self):
        """Test fOArea metric."""
        nside = 128
        npix = hp.nside2npix(nside)
        names = ["metricdata"]
        types = [int]
        data = np.zeros(npix, dtype=list(zip(names, types)))
        data["metricdata"] += 826
        metric = metrics.fOArea(col="ack", nside=nside, Nvisit=825, Asky=18000.0)
        slicePoint = {"sid": 0}
        result = metric.run(data, slicePoint)
        # fOArea returns the area with at least Nvisits.
        deginsph = 129600.0 / np.pi
        np.testing.assert_almost_equal(result, deginsph)
        data["metricdata"][: data.size // 2] = 0
        result = metric.run(data, slicePoint)
        np.testing.assert_almost_equal(result, deginsph / 2.0)

    def testNormalizeMetric(self):
        """Test normalize metric."""
        data = np.ones(10, dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.NormalizeMetric(col="testcol", normVal=5.5)
        result = metric.run(data)
        np.testing.assert_equal(result, np.ones(10, float) / 5.5)

    def testZeropointMetric(self):
        """Test zeropoint metric."""
        data = np.ones(10, dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.ZeropointMetric(col="testcol", zp=5.5)
        result = metric.run(data)
        np.testing.assert_equal(result, np.ones(10, float) + 5.5)

    def testTotalPowerMetric(self):
        nside = 128
        data = np.ones(12 * nside**2, dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.TotalPowerMetric(col="testcol")
        result = metric.run(data)
        np.testing.assert_equal(result, 0.0)


if __name__ == "__main__":
    unittest.main()
