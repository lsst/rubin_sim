import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestSchedviewMetrics(unittest.TestCase):
    def test_age_metric(self):
        data = np.rec.fromrecords([(1, 60000)], names="id,observationStartMJD")

        assert metrics.AgeMetric(60002).run(data) == 2
        assert metrics.AgeMetric(60002.5).run(data) == 2.5
        assert np.isnan(metrics.AgeMetric(70000.0).run(data))
