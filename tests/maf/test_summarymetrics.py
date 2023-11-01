import unittest

import healpy as hp
import numpy as np

import rubin_sim.maf.metrics as metrics


class TestSummaryMetrics(unittest.TestCase):
    def test_identity_metric(self):
        """Test identity metric."""
        dv = np.arange(0, 10, 0.5)
        dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        testmetric = metrics.IdentityMetric("testdata")
        np.testing.assert_equal(testmetric.run(dv), dv["testdata"])

    def testf_o_nv(self):
        """
        Test the fONv metric.
        """
        nside = 128
        npix = hp.nside2npix(nside)
        names = ["metricdata"]
        types = [int]
        data = np.zeros(npix, dtype=list(zip(names, types)))
        data["metricdata"] += 826
        metric = metrics.FONv(col="ack", nside=nside, n_visit=825, asky=18000.0)
        slice_point = {"sid": 0}
        result = metric.run(data, slice_point)
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
        result = metric.run(data, slice_point)
        min_nvis = result["value"][np.where(result["name"] == "MinNvis")]
        median_nvis = result["value"][np.where(result["name"] == "MedianNvis")]
        self.assertEqual(min_nvis, 0)
        self.assertEqual(median_nvis, 826)

    def testf_o_area(self):
        """Test fOArea metric."""
        nside = 128
        npix = hp.nside2npix(nside)
        names = ["metricdata"]
        types = [int]
        data = np.zeros(npix, dtype=list(zip(names, types)))
        data["metricdata"] += 826
        metric = metrics.FOArea(col="ack", nside=nside, n_visit=825, asky=18000.0)
        slice_point = {"sid": 0}
        result = metric.run(data, slice_point)
        # fOArea returns the area with at least Nvisits.
        deginsph = 129600.0 / np.pi
        np.testing.assert_almost_equal(result, deginsph)
        data["metricdata"][: data.size // 2] = 0
        result = metric.run(data, slice_point)
        np.testing.assert_almost_equal(result, deginsph / 2.0)

    def test_normalize_metric(self):
        """Test normalize metric."""
        data = np.ones(10, dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.NormalizeMetric(col="testcol", norm_val=5.5)
        result = metric.run(data)
        np.testing.assert_equal(result, np.ones(10, float) / 5.5)

    def test_zeropoint_metric(self):
        """Test zeropoint metric."""
        data = np.ones(10, dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.ZeropointMetric(col="testcol", zp=5.5)
        result = metric.run(data)
        np.testing.assert_equal(result, np.ones(10, float) + 5.5)

    def test_total_power_metric(self):
        nside = 128
        data = np.ones(12 * nside**2, dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.TotalPowerMetric(col="testcol")
        result = metric.run(data)
        np.testing.assert_equal(result, 0.0)


if __name__ == "__main__":
    unittest.main()
