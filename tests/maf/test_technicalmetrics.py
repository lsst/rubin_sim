import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestTechnicalMetrics(unittest.TestCase):
    def test_n_changes_metric(self):
        """
        Test the NChanges metric.
        """
        filters = np.array(["u", "u", "g", "g", "r"])
        visit_times = np.arange(0, filters.size, 1)
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.NChangesMetric()
        result = metric.run(data)
        self.assertEqual(result, 2)
        filters = np.array(["u", "g", "u", "g", "r"])
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.NChangesMetric()
        result = metric.run(data)
        self.assertEqual(result, 4)

    def test_min_time_between_states_metric(self):
        """
        Test the minTimeBetweenStates metric.
        """
        filters = np.array(["u", "g", "g", "r"])
        visit_times = np.array([0, 5, 6, 7])  # days
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.MinTimeBetweenStatesMetric()
        result = metric.run(data)  # minutes
        self.assertEqual(result, 2 * 24.0 * 60.0)
        data["filter"] = np.array(["u", "u", "u", "u"])
        result = metric.run(data)
        self.assertEqual(result, metric.badval)

    def test_n_state_changes_faster_than_metric(self):
        """
        Test the NStateChangesFasterThan metric.
        """
        filters = np.array(["u", "g", "g", "r"])
        visit_times = np.array([0, 5, 6, 7])  # days
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.NStateChangesFasterThanMetric(cutoff=3 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 1)

    def test_max_state_changes_within_metric(self):
        """
        Test the MaxStateChangesWithin metric.
        """
        filters = np.array(["u", "g", "r", "u", "g", "r"])
        visit_times = np.array([0, 1, 1, 4, 6, 7])  # days
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 2)
        filters = np.array(["u", "g", "g", "u", "g", "r", "g", "r"])
        visit_times = np.array([0, 1, 1, 4, 4, 7, 8, 8])  # days
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 3)

        filters = np.array(["u", "g"])
        visit_times = np.array([0, 1])  # days
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 1)

        filters = np.array(["u", "u"])
        visit_times = np.array([0, 1])  # days
        data = np.core.records.fromarrays([visit_times, filters], names=["observationStartMJD", "filter"])
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 0)

    def test_teff_metric(self):
        """
        Test the Teff (time_effective) metric.
        """
        filters = np.array(["g", "g", "g", "g", "g"])
        m5 = np.zeros(len(filters), float) + 25.0
        data = np.core.records.fromarrays([m5, filters], names=["fiveSigmaDepth", "filter"])
        metric = metrics.TeffMetric(fiducial_depth={"g": 25}, teff_base=30.0)
        result = metric.run(data)
        self.assertEqual(result, 30.0 * m5.size)
        filters = np.array(["g", "g", "g", "u", "u"])
        m5 = np.zeros(len(filters), float) + 25.0
        m5[3:5] = 20.0
        data = np.core.records.fromarrays([m5, filters], names=["fiveSigmaDepth", "filter"])
        metric = metrics.TeffMetric(fiducial_depth={"u": 20, "g": 25}, teff_base=30.0)
        result = metric.run(data)
        self.assertEqual(result, 30.0 * m5.size)

    def test_open_shutter_fraction_metric(self):
        """
        Test the open shutter fraction metric.
        """
        nvisit = 10
        exptime = 30.0
        slewtime = 30.0
        visit_exp_time = np.ones(nvisit, dtype="float") * exptime
        visit_time = np.ones(nvisit, dtype="float") * (exptime + 0.0)
        slew_time = np.ones(nvisit, dtype="float") * slewtime
        data = np.core.records.fromarrays(
            [visit_exp_time, visit_time, slew_time],
            names=["visitExposureTime", "visitTime", "slewTime"],
        )
        metric = metrics.OpenShutterFractionMetric()
        result = metric.run(data)
        self.assertEqual(result, 0.5)

    def test_brute_osf_metric(self):
        """
        Test the open shutter fraction metric.
        """
        nvisit = 10
        exptime = 30.0
        slewtime = 30.0
        visit_exp_time = np.ones(nvisit, dtype="float") * exptime
        visit_time = np.ones(nvisit, dtype="float") * (exptime + 0.0)
        slew_time = np.ones(nvisit, dtype="float") * slewtime
        mjd = np.zeros(nvisit) + np.add.accumulate(visit_exp_time) + np.add.accumulate(slew_time)
        mjd = mjd / 60.0 / 60.0 / 24.0
        data = np.core.records.fromarrays(
            [visit_exp_time, visit_time, slew_time, mjd],
            names=[
                "visitExposureTime",
                "visit_time",
                "slew_time",
                "observationStartMJD",
            ],
        )
        metric = metrics.BruteOSFMetric()
        result = metric.run(data)
        self.assertGreater(result, 0.5)
        self.assertLess(result, 0.6)


if __name__ == "__main__":
    unittest.main()
