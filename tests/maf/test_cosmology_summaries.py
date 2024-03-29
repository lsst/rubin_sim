import unittest

import healpy as hp
import numpy as np

import rubin_sim.maf.metrics as metrics


class TestCosmologySummaryMetrics(unittest.TestCase):

    def test_total_power_metric(self):
        nside = 64
        data = np.ones(hp.nside2npix(nside), dtype=list(zip(["testcol"], ["float"])))
        metric = metrics.TotalPowerMetric(col="testcol")
        result = metric.run(data)
        np.testing.assert_equal(result, 0.0)


if __name__ == "__main__":
    unittest.main()
