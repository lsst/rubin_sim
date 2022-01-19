import matplotlib

matplotlib.use("Agg")
import numpy as np
import unittest
import rubin_sim.maf.metrics as metrics


class TestTechnicalMetrics(unittest.TestCase):
    def testNChangesMetric(self):
        """
        Test the NChanges metric.
        """
        filters = np.array(["u", "u", "g", "g", "r"])
        visitTimes = np.arange(0, filters.size, 1)
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.NChangesMetric()
        result = metric.run(data)
        self.assertEqual(result, 2)
        filters = np.array(["u", "g", "u", "g", "r"])
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.NChangesMetric()
        result = metric.run(data)
        self.assertEqual(result, 4)

    def testMinTimeBetweenStatesMetric(self):
        """
        Test the minTimeBetweenStates metric.
        """
        filters = np.array(["u", "g", "g", "r"])
        visitTimes = np.array([0, 5, 6, 7])  # days
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.MinTimeBetweenStatesMetric()
        result = metric.run(data)  # minutes
        self.assertEqual(result, 2 * 24.0 * 60.0)
        data["filter"] = np.array(["u", "u", "u", "u"])
        result = metric.run(data)
        self.assertEqual(result, metric.badval)

    def testNStateChangesFasterThanMetric(self):
        """
        Test the NStateChangesFasterThan metric.
        """
        filters = np.array(["u", "g", "g", "r"])
        visitTimes = np.array([0, 5, 6, 7])  # days
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.NStateChangesFasterThanMetric(cutoff=3 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 1)

    def testMaxStateChangesWithinMetric(self):
        """
        Test the MaxStateChangesWithin metric.
        """
        filters = np.array(["u", "g", "r", "u", "g", "r"])
        visitTimes = np.array([0, 1, 1, 4, 6, 7])  # days
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 2)
        filters = np.array(["u", "g", "g", "u", "g", "r", "g", "r"])
        visitTimes = np.array([0, 1, 1, 4, 4, 7, 8, 8])  # days
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 3)

        filters = np.array(["u", "g"])
        visitTimes = np.array([0, 1])  # days
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 1)

        filters = np.array(["u", "u"])
        visitTimes = np.array([0, 1])  # days
        data = np.core.records.fromarrays(
            [visitTimes, filters], names=["observationStartMJD", "filter"]
        )
        metric = metrics.MaxStateChangesWithinMetric(timespan=1 * 24 * 60)
        result = metric.run(data)  # minutes
        self.assertEqual(result, 0)

    def testTeffMetric(self):
        """
        Test the Teff (time_effective) metric.
        """
        filters = np.array(["g", "g", "g", "g", "g"])
        m5 = np.zeros(len(filters), float) + 25.0
        data = np.core.records.fromarrays(
            [m5, filters], names=["fiveSigmaDepth", "filter"]
        )
        metric = metrics.TeffMetric(fiducialDepth={"g": 25}, teffBase=30.0)
        result = metric.run(data)
        self.assertEqual(result, 30.0 * m5.size)
        filters = np.array(["g", "g", "g", "u", "u"])
        m5 = np.zeros(len(filters), float) + 25.0
        m5[3:5] = 20.0
        data = np.core.records.fromarrays(
            [m5, filters], names=["fiveSigmaDepth", "filter"]
        )
        metric = metrics.TeffMetric(fiducialDepth={"u": 20, "g": 25}, teffBase=30.0)
        result = metric.run(data)
        self.assertEqual(result, 30.0 * m5.size)

    def testOpenShutterFractionMetric(self):
        """
        Test the open shutter fraction metric.
        """
        nvisit = 10
        exptime = 30.0
        slewtime = 30.0
        visitExpTime = np.ones(nvisit, dtype="float") * exptime
        visitTime = np.ones(nvisit, dtype="float") * (exptime + 0.0)
        slewTime = np.ones(nvisit, dtype="float") * slewtime
        data = np.core.records.fromarrays(
            [visitExpTime, visitTime, slewTime],
            names=["visitExposureTime", "visitTime", "slewTime"],
        )
        metric = metrics.OpenShutterFractionMetric()
        result = metric.run(data)
        self.assertEqual(result, 0.5)

    def testBruteOSFMetric(self):
        """
        Test the open shutter fraction metric.
        """
        nvisit = 10
        exptime = 30.0
        slewtime = 30.0
        visitExpTime = np.ones(nvisit, dtype="float") * exptime
        visitTime = np.ones(nvisit, dtype="float") * (exptime + 0.0)
        slewTime = np.ones(nvisit, dtype="float") * slewtime
        mjd = (
            np.zeros(nvisit)
            + np.add.accumulate(visitExpTime)
            + np.add.accumulate(slewTime)
        )
        mjd = mjd / 60.0 / 60.0 / 24.0
        data = np.core.records.fromarrays(
            [visitExpTime, visitTime, slewTime, mjd],
            names=["visitExposureTime", "visitTime", "slewTime", "observationStartMJD"],
        )
        metric = metrics.BruteOSFMetric()
        result = metric.run(data)
        self.assertGreater(result, 0.5)
        self.assertLess(result, 0.6)


if __name__ == "__main__":
    unittest.main()
