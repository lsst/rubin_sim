import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.stackers as stackers
from rubin_sim.maf.metrics.base_metric import BaseMetric


class OldTeffMetric(BaseMetric):
    """
    Effective time equivalent for a given set of visits.
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        metric_name="tEff",
        fiducial_depth=None,
        teff_base=30.0,
        normed=False,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col
        if fiducial_depth is None:
            # From reference von Karman 500nm zenith seeing of 0.69"
            # median zenith dark seeing from sims_skybrightness_pre
            # airmass = 1
            # 2 "snaps" of 15 seconds each
            # m5_flat_sed sysEngVals from rubin_sim
            #   commit 6d03bd49550972e48648503ed60784a4e6775b82 (2021-05-18)
            # These include constants from:
            #   https://github.com/lsst-pst/syseng_throughputs/blob/master/notebooks/generate_sims_values.ipynb
            #   commit 7abb90951fcbc70d9c4d0c805c55a67224f9069f (2021-05-05)
            # See https://github.com/lsst-sims/smtn-002/blob/master/notebooks/teff_fiducial.ipynb
            self.depth = {
                "u": 23.71,
                "g": 24.67,
                "r": 24.24,
                "i": 23.82,
                "z": 23.21,
                "y": 22.40,
            }
        else:
            if isinstance(fiducial_depth, dict):
                self.depth = fiducial_depth
            else:
                raise ValueError("fiducial_depth should be None or dictionary")
        self.teff_base = teff_base
        self.normed = normed
        if self.normed:
            units = ""
        else:
            units = "seconds"
        super(OldTeffMetric, self).__init__(
            col=[m5_col, filter_col], metric_name=metric_name, units=units, **kwargs
        )
        if self.normed:
            self.comment = "Normalized effective time"
        else:
            self.comment = "Effect time"
        self.comment += " of a series of observations, evaluating the equivalent amount of time"
        self.comment += " each observation would require if taken at a fiducial limiting magnitude."
        self.comment += " Fiducial depths are : %s" % self.depth
        if self.normed:
            self.comment += " Normalized by the total amount of time actual on-sky."

    def run(self, data_slice, slice_point=None):
        filters = np.unique(data_slice[self.filter_col])
        teff = 0.0
        for f in filters:
            match = np.where(data_slice[self.filter_col] == f)[0]
            teff += (10.0 ** (0.8 * (data_slice[self.m5_col][match] - self.depth[f]))).sum()
        teff *= self.teff_base
        if self.normed:
            # Normalize by the t_eff equivalent if each observation
            # was at the fiducial depth.
            teff = teff / (self.teff_base * data_slice[self.m5_col].size)
        return teff


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

    def test_teff_regression(self):
        """Test this teff implementation matches the old one."""
        num_points = 50
        bands = tuple("ugrizy")
        rng = np.random.default_rng(seed=6563)

        m5 = 24 + rng.random(num_points)
        filters = rng.choice(bands, num_points)
        fiducial_depth = {b: 24 + rng.random() for b in bands}
        exposure_time = np.full(num_points, 30.0, dtype=float)
        data = np.core.records.fromarrays(
            [m5, filters, exposure_time], names=["fiveSigmaDepth", "filter", "visitExposureTime"]
        )
        teff_stacker = stackers.TeffStacker(fiducial_depth=fiducial_depth, teff_base=30.0)
        data = teff_stacker.run(data)

        metric = metrics.SumMetric(col="t_eff")
        result = metric.run(data)
        old_metric = OldTeffMetric(fiducial_depth=fiducial_depth, teff_base=30.0)
        old_result = old_metric.run(data)
        self.assertEqual(result, old_result)

        data = np.core.records.fromarrays(
            [m5, filters, exposure_time], names=["fiveSigmaDepth", "filter", "visitExposureTime"]
        )
        teff_stacker = stackers.TeffStacker(fiducial_depth=fiducial_depth, teff_base=30.0, normed=True)
        data = teff_stacker.run(data)
        metric = metrics.MeanMetric(col="t_eff")
        result = metric.run(data)
        old_metric = OldTeffMetric(fiducial_depth=fiducial_depth, teff_base=30.0, normed=True)
        old_result = old_metric.run(data)
        self.assertAlmostEqual(result, old_result)

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
