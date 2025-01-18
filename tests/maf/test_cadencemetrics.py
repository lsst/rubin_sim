import math
import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestCadenceMetrics(unittest.TestCase):
    def test_phase_gap_metric(self):
        """
        Test the phase gap metric
        """
        data = np.zeros(10, dtype=list(zip(["observationStartMJD"], [float])))
        data["observationStartMJD"] += np.arange(10) * 0.25

        pgm = metrics.PhaseGapMetric(n_periods=1, period_min=0.5, period_max=0.5)
        metric_val = pgm.run(data)

        mean_gap = pgm.reduce_mean_gap(metric_val)
        median_gap = pgm.reduce_median_gap(metric_val)
        worst_period = pgm.reduce_worst_period(metric_val)
        largest_gap = pgm.reduce_largest_gap(metric_val)

        self.assertEqual(mean_gap, 0.5)
        self.assertEqual(median_gap, 0.5)
        self.assertEqual(worst_period, 0.5)
        self.assertEqual(largest_gap, 0.5)

        pgm = metrics.PhaseGapMetric(n_periods=2, period_min=0.25, period_max=0.5)
        metric_val = pgm.run(data)

        mean_gap = pgm.reduce_mean_gap(metric_val)
        median_gap = pgm.reduce_median_gap(metric_val)
        worst_period = pgm.reduce_worst_period(metric_val)
        largest_gap = pgm.reduce_largest_gap(metric_val)

        self.assertEqual(mean_gap, 0.75)
        self.assertEqual(median_gap, 0.75)
        self.assertEqual(worst_period, 0.25)
        self.assertEqual(largest_gap, 1.0)

    def test_template_exists(self):
        """
        Test the TemplateExistsMetric.
        """
        names = ["finSeeing", "observationStartMJD"]
        types = [float, float]
        data = np.zeros(10, dtype=list(zip(names, types)))
        data["finSeeing"] = [2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.4, 1.0]
        data["observationStartMJD"] = np.arange(10)
        slice_point = {"sid": 0}
        # so here we have 4 images w/o good previous templates
        metric = metrics.TemplateExistsMetric(seeing_col="finSeeing")
        result = metric.run(data, slice_point)
        self.assertEqual(result, 6.0 / 10.0)

    def test_uniformity_metric(self):
        names = ["observationStartMJD"]
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        metric = metrics.UniformityMetric()
        result1 = metric.run(data)
        # If all the observations are on the 1st day, should be 1
        self.assertEqual(result1, 1)
        data["observationStartMJD"] = data["observationStartMJD"] + 365.25 * 10
        slice_point = {"sid": 0}
        result2 = metric.run(data, slice_point)
        # All on last day should also be 1
        self.assertEqual(result2, 1)
        # Make a perfectly uniform dist
        data["observationStartMJD"] = np.arange(0.0, 365.25 * 10, 365.25 * 10 / 100)
        result3 = metric.run(data, slice_point)
        # Result should be zero for uniform
        np.testing.assert_almost_equal(result3, 0.0)
        # A single obseravtion should give a result of 1
        data = np.zeros(1, dtype=list(zip(names, types)))
        result4 = metric.run(data, slice_point)
        self.assertEqual(result4, 1)

    def test_general_uniformity_metric(self):
        names = ["observationStartMJD"]
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        metric = metrics.GeneralUniformityMetric(col="observationStartMJD", min_value=0, max_value=1)
        result1 = metric.run(data)
        # If all the observations are at the minimum value, should be 1
        self.assertEqual(result1, 1)
        data["observationStartMJD"] = data["observationStartMJD"] + 1
        slice_point = {"sid": 0}
        result2 = metric.run(data, slice_point)
        # All on max value should also be 1
        self.assertEqual(result2, 1)
        # Make a perfectly uniform dist
        step = 1 / 99
        data["observationStartMJD"] = np.arange(0.0, 1 + step / 2, step)
        result3 = metric.run(data, slice_point)
        # Result should be zero for uniform
        np.testing.assert_almost_equal(result3, 0.0)
        # Test setup with max_value > 1
        metric = metrics.GeneralUniformityMetric(col="observationStartMJD", min_value=0, max_value=10)
        # Make a perfectly uniform dist covering limited range
        # (0-1 out of 0-10 range)
        data["observationStartMJD"] = np.arange(0.0, 1 + step / 2, step)
        result5 = metric.run(data, slice_point)
        self.assertEqual(result5, 0.9)
        # Test setup with max_value None
        metric = metrics.GeneralUniformityMetric(col="observationStartMJD", min_value=None, max_value=None)
        # Make a perfectly uniform dist
        data["observationStartMJD"] = np.arange(0.0, 1 + step / 2, step)
        result5 = metric.run(data, slice_point)
        self.assertAlmostEqual(result5, 0)
        # A single observation should give a result of 1
        data = np.zeros(1, dtype=list(zip(names, types)))
        result4 = metric.run(data, slice_point)
        self.assertEqual(result4, 1)

    def test_t_gap_metric(self):
        names = ["observationStartMJD"]
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        # All 1-day gaps
        data["observationStartMJD"] = np.arange(100)

        metric = metrics.TgapsMetric(bins=np.arange(1, 100, 1))
        result1 = metric.run(data)
        # By default, should all be in first bin
        self.assertEqual(result1[0], data.size - 1)
        self.assertEqual(np.sum(result1), data.size - 1)
        data["observationStartMJD"] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        self.assertEqual(result2[1], data.size - 1)
        self.assertEqual(np.sum(result2), data.size - 1)

        data = np.zeros(4, dtype=list(zip(names, types)))
        data["observationStartMJD"] = [10, 20, 30, 40]
        metric = metrics.TgapsMetric(all_gaps=True, bins=np.arange(1, 100, 10))
        result3 = metric.run(data)
        self.assertEqual(result3[1], 2)
        ngaps = math.factorial(data.size - 1)
        self.assertEqual(np.sum(result3), ngaps)

    def test_t_gaps_percent_metric(self):
        names = ["observationStartMJD"]
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))

        # All 1-day gaps
        data["observationStartMJD"] = np.arange(100)
        # All intervals are one day, so should be 100% within the gap specified
        metric = metrics.TgapsPercentMetric(min_time=0.5, max_time=1.5, all_gaps=False)
        result1 = metric.run(data)
        self.assertEqual(result1, 100)

        # All 2 day gaps
        data["observationStartMJD"] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        self.assertEqual(result2, 0)

        # Run with all_gaps = True
        data = np.zeros(4, dtype=list(zip(names, types)))
        data["observationStartMJD"] = [1, 2, 3, 4]
        metric = metrics.TgapsPercentMetric(min_time=0.5, max_time=1.5, all_gaps=True)
        result3 = metric.run(data)
        # This should be 50% -- 3 gaps of 1 day, 2 gaps of 2 days, and
        # 1 gap of 3 days
        self.assertEqual(result3, 50)

    def test_night_gap_metric(self):
        names = ["night"]
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        # All 1-day gaps
        data["night"] = np.arange(100)

        metric = metrics.NightgapsMetric(bins=np.arange(1, 100, 1))
        result1 = metric.run(data)
        # By default, should all be in first bin
        self.assertEqual(result1[0], data.size - 1)
        self.assertEqual(np.sum(result1), data.size - 1)
        data["night"] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        self.assertEqual(result2[1], data.size - 1)
        self.assertEqual(np.sum(result2), data.size - 1)

        data = np.zeros(4, dtype=list(zip(names, types)))
        data["night"] = [10, 20, 30, 40]
        metric = metrics.NightgapsMetric(all_gaps=True, bins=np.arange(1, 100, 10))
        result3 = metric.run(data)
        self.assertEqual(result3[1], 2)
        ngaps = math.factorial(data.size - 1)
        self.assertEqual(np.sum(result3), ngaps)

        data = np.zeros(6, dtype=list(zip(names, types)))
        data["night"] = [1, 1, 2, 3, 3, 5]
        metric = metrics.NightgapsMetric(bins=np.arange(0, 5, 1))
        result4 = metric.run(data)
        self.assertEqual(result4[0], 0)
        self.assertEqual(result4[1], 2)
        self.assertEqual(result4[2], 1)

    def test_n_visits_per_night_metric(self):
        names = ["night"]
        types = [float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        # One visit per night.
        data["night"] = np.arange(100)

        bins = np.arange(0, 5, 1)
        metric = metrics.NVisitsPerNightMetric(bins=bins)
        result = metric.run(data)
        # All nights have one visit.
        expected_result = np.zeros(len(bins) - 1, dtype=int)
        expected_result[1] = len(data)
        np.testing.assert_array_equal(result, expected_result)

        data["night"] = np.floor(np.arange(0, 100) / 2)
        result = metric.run(data)
        expected_result = np.zeros(len(bins) - 1, dtype=int)
        expected_result[2] = len(data) / 2
        np.testing.assert_array_equal(result, expected_result)

    def test_rapid_revisit_uniformity_metric(self):
        data = np.zeros(100, dtype=list(zip(["observationStartMJD"], [float])))
        # Uniformly distribute time _differences_ between 0 and 100
        dtimes = np.arange(100)
        data["observationStartMJD"] = dtimes.cumsum()
        # Set up "rapid revisit" metric to look for visits between 5 and 25
        metric = metrics.RapidRevisitUniformityMetric(d_tmin=5, d_tmax=55, min_nvisits=50)
        result = metric.run(data)
        # This should be uniform.
        self.assertLess(result, 0.1)
        self.assertGreaterEqual(result, 0)
        # Set up non-uniform distribution of time differences
        dtimes = np.zeros(100) + 5
        data["observationStartMJD"] = dtimes.cumsum()
        result = metric.run(data)
        self.assertGreaterEqual(result, 0.5)
        dtimes = np.zeros(100) + 15
        data["observationStartMJD"] = dtimes.cumsum()
        result = metric.run(data)
        self.assertGreaterEqual(result, 0.5)
        """
        # Let's see how much dmax/result can vary
        resmin = 1
        resmax = 0
        rng = np.random.RandomState(88123100)
        for i in range(10000):
            dtimes = rng.rand(100)
            data['observationStartMJD'] = dtimes.cumsum()
            metric = metrics.RapidRevisitUniformityMetric(d_tmin=0.1,
                                d_tmax=0.8, min_nvisits=50)
            result = metric.run(data)
            resmin = np.min([resmin, result])
            resmax = np.max([resmax, result])
        print("RapidRevisitUniformity .. range", resmin, resmax)
        """

    def test_rapid_revisit_metric(self):
        data = np.zeros(100, dtype=list(zip(["observationStartMJD"], [float])))
        dtimes = np.arange(100) / 24.0 / 60.0
        data["observationStartMJD"] = dtimes.cumsum()
        # Set metric parameters to the actual N1/N2 values for these dtimes.
        metric = metrics.RapidRevisitMetric(
            d_tmin=40.0 / 60.0 / 60.0 / 24.0,
            d_tpairs=20.0 / 60.0 / 24.0,
            d_tmax=30.0 / 60.0 / 24.0,
            min_n1=19,
            min_n2=29,
        )
        result = metric.run(data)
        self.assertEqual(result, 1)
        # Set metric parameters to > N1/N2 values, to see it return 0.
        metric = metrics.RapidRevisitMetric(
            d_tmin=40.0 / 60.0 / 60.0 / 24.0,
            d_tpairs=20.0 / 60.0 / 24.0,
            d_tmax=30.0 / 60.0 / 24.0,
            min_n1=30,
            min_n2=50,
        )
        result = metric.run(data)
        self.assertEqual(result, 0)
        # Test with single value data.
        data = np.zeros(1, dtype=list(zip(["observationStartMJD"], [float])))
        result = metric.run(data)
        self.assertEqual(result, 0)

    def test_n_revisits_metric(self):
        data = np.zeros(100, dtype=list(zip(["observationStartMJD"], [float])))
        dtimes = np.arange(100) / 24.0 / 60.0
        data["observationStartMJD"] = dtimes.cumsum()
        metric = metrics.NRevisitsMetric(d_t=50.0)
        result = metric.run(data)
        self.assertEqual(result, 50)
        metric = metrics.NRevisitsMetric(d_t=50.0, normed=True)
        result = metric.run(data)
        self.assertEqual(result, 0.5)

    def test_transient_metric(self):
        names = ["observationStartMJD", "fiveSigmaDepth", "filter"]
        types = [float, float, "<U1"]

        ndata = 100
        data_slice = np.zeros(ndata, dtype=list(zip(names, types)))
        data_slice["observationStartMJD"] = np.arange(ndata)
        data_slice["fiveSigmaDepth"] = 25
        data_slice["filter"] = "g"

        metric = metrics.TransientMetric(survey_duration=ndata / 365.25)

        # Should detect everything
        self.assertEqual(metric.run(data_slice), 1.0)

        # Double to survey duration, should now only detect half
        metric = metrics.TransientMetric(survey_duration=ndata / 365.25 * 2)
        self.assertEqual(metric.run(data_slice), 0.5)

        # Set half of the m5 of the observations very bright,
        # so kill another half.
        data_slice["fiveSigmaDepth"][0:50] = 20
        self.assertEqual(metric.run(data_slice), 0.25)

        data_slice["fiveSigmaDepth"] = 25
        # Demand lots of early observations
        metric = metrics.TransientMetric(peak_time=0.5, n_pre_peak=3, survey_duration=ndata / 365.25)
        self.assertEqual(metric.run(data_slice), 0.0)

        # Demand a reasonable number of early observations
        metric = metrics.TransientMetric(peak_time=2, n_pre_peak=2, survey_duration=ndata / 365.25)
        self.assertEqual(metric.run(data_slice), 1.0)

        # Demand multiple filters
        metric = metrics.TransientMetric(n_filters=2, survey_duration=ndata / 365.25)
        self.assertEqual(metric.run(data_slice), 0.0)

        data_slice["filter"] = ["r", "g"] * 50
        self.assertEqual(metric.run(data_slice), 1.0)

        # Demad too many observation per light curve
        metric = metrics.TransientMetric(n_per_lc=20, survey_duration=ndata / 365.25)
        self.assertEqual(metric.run(data_slice), 0.0)

        # Test both filter and number of LC samples
        metric = metrics.TransientMetric(n_filters=2, n_per_lc=3, survey_duration=ndata / 365.25)
        self.assertEqual(metric.run(data_slice), 1.0)

    def test_season_length_metric(self):
        times = np.arange(0, 3650, 10)
        data = np.zeros(
            len(times),
            dtype=list(zip(["observationStartMJD", "visitExposureTime"], [float, float])),
        )
        data["observationStartMJD"] = times
        data["visitExposureTime"] = 30.0
        metric = metrics.SeasonLengthMetric(reduce_func=np.median)
        slice_point = {"ra": 0}
        result = metric.run(data, slice_point)
        self.assertEqual(result, 350)
        times = np.arange(0, 3650, 365)
        data = np.zeros(
            len(times) * 2,
            dtype=list(zip(["observationStartMJD", "visitExposureTime"], [float, float])),
        )
        data["observationStartMJD"][0 : len(times)] = times
        data["observationStartMJD"][len(times) :] = times + 10
        data["observationStartMJD"] = np.sort(data["observationStartMJD"])
        data["visitExposureTime"] = 30.0
        metric = metrics.SeasonLengthMetric(reduce_func=np.median)
        slice_point = {"ra": 0}
        result = metric.run(data, slice_point)
        self.assertEqual(result, 10)
        times = np.arange(0, 3650 - 365, 365)
        data = np.zeros(
            len(times) * 2,
            dtype=list(zip(["observationStartMJD", "visitExposureTime"], [float, float])),
        )
        data["observationStartMJD"][0 : len(times)] = times
        data["observationStartMJD"][len(times) :] = times + 10
        data["observationStartMJD"] = np.sort(data["observationStartMJD"])
        data["visitExposureTime"] = 30.0
        metric = metrics.SeasonLengthMetric(reduce_func=np.size)
        slice_point = {"ra": 0}
        result = metric.run(data, slice_point)
        self.assertEqual(result, 9)


if __name__ == "__main__":
    unittest.main()
