import matplotlib

matplotlib.use("Agg")

import unittest

import numpy as np

import rubin_sim.maf.metric_bundles as metricBundle
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers


class Test2D(unittest.TestCase):
    def setUp(self):
        names = [
            "night",
            "fieldId",
            "fieldRA",
            "fieldDec",
            "fiveSigmaDepth",
            "observationStartMJD",
            "rotSkyPos",
        ]
        types = [int, int, float, float, float, float, float]

        self.m5_1 = 25.0
        self.m5_2 = 24.0

        self.n1 = 50
        self.n2 = 49

        # Picking RA and Dec values that will hit nside=16 healpixels
        self.sim_data = np.zeros(self.n1 + self.n2, dtype=list(zip(names, types)))
        self.sim_data["night"][0 : self.n1] = 1
        self.sim_data["fieldId"][0 : self.n1] = 1
        self.sim_data["fieldRA"][0 : self.n1] = 10.0
        self.sim_data["fieldDec"][0 : self.n1] = 0.0
        self.sim_data["fiveSigmaDepth"][0 : self.n1] = self.m5_1

        self.sim_data["night"][self.n1 :] = 2
        self.sim_data["fieldId"][self.n1 :] = 2
        self.sim_data["fieldRA"][self.n1 :] = 190.0
        self.sim_data["fieldDec"][self.n1 :] = -20.0
        self.sim_data["fiveSigmaDepth"][self.n1 :] = self.m5_2

        self.field_data = np.zeros(
            2, dtype=list(zip(["fieldId", "fieldRA", "fieldDec"], [int, float, float]))
        )
        self.field_data["fieldId"] = [1, 2]
        self.field_data["fieldRA"] = np.radians([10.0, 190.0])
        self.field_data["fieldDec"] = np.radians([0.0, -20.0])

        self.sim_data["observationStartMJD"] = self.sim_data["night"]

    def test_user_points2d_slicer(self):
        metric = metrics.AccumulateCountMetric(bins=[0.5, 1.5, 2.5])
        slicer = slicers.UserPointsSlicer(
            ra=np.degrees(self.field_data["fieldRA"]),
            dec=np.degrees(self.field_data["fieldDec"]),
            lat_lon_deg=True,
        )
        sql = ""
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.fieldData = self.field_data
        mbg.run_current("", sim_data=self.sim_data)
        expected = np.array([[self.n1, self.n1], [-666.0, self.n2]])
        assert np.array_equal(mb.metric_values.data, expected)

    def test_healpix2d_slicer(self):
        metric = metrics.AccumulateCountMetric(bins=[0.5, 1.5, 2.5])
        slicer = slicers.HealpixSlicer(nside=16)
        sql = ""
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)

        good = np.where(mb.metric_values.mask[:, -1] == False)[0]
        expected = np.array([[self.n1, self.n1], [-666.0, self.n2]])
        assert np.array_equal(mb.metric_values.data[good, :], expected)

    def test_histogram_metric(self):
        metric = metrics.HistogramMetric(bins=[0.5, 1.5, 2.5])
        slicer = slicers.HealpixSlicer(nside=16)
        sql = ""
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)

        good = np.where(mb.metric_values.mask[:, -1] == False)[0]
        expected = np.array([[self.n1, 0.0], [0.0, self.n2]])
        assert np.array_equal(mb.metric_values.data[good, :], expected)

        # Check that I can run a different statistic
        metric = metrics.HistogramMetric(col="fiveSigmaDepth", statistic="sum", bins=[0.5, 1.5, 2.5])
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)
        expected = np.array([[self.m5_1 * self.n1, 0.0], [0.0, self.m5_2 * self.n2]])
        assert np.array_equal(mb.metric_values.data[good, :], expected)

    def test_accumulate_metric(self):
        metric = metrics.AccumulateMetric(col="fiveSigmaDepth", bins=[0.5, 1.5, 2.5])
        slicer = slicers.HealpixSlicer(nside=16)
        sql = ""
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)
        good = np.where(mb.metric_values.mask[:, -1] == False)[0]
        expected = np.array([[self.n1 * self.m5_1, self.n1 * self.m5_1], [-666.0, self.n2 * self.m5_2]])
        assert np.array_equal(mb.metric_values.data[good, :], expected)

    def test_histogram_m5_metric(self):
        metric = metrics.HistogramM5Metric(bins=[0.5, 1.5, 2.5])
        slicer = slicers.HealpixSlicer(nside=16)
        sql = ""
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)
        good = np.where((mb.metric_values.mask[:, 0] == False) | (mb.metric_values.mask[:, 1] == False))[0]

        check_metric = metrics.Coaddm5Metric()
        temp_slice = np.zeros(self.n1, dtype=list(zip(["fiveSigmaDepth"], [float])))
        temp_slice["fiveSigmaDepth"] += self.m5_1
        val1 = check_metric.run(temp_slice)
        temp_slice = np.zeros(self.n2, dtype=list(zip(["fiveSigmaDepth"], [float])))
        temp_slice["fiveSigmaDepth"] += self.m5_2
        val2 = check_metric.run(temp_slice)

        expected = np.array([[val1, -666.0], [-666.0, val2]])
        assert np.array_equal(mb.metric_values.data[good, :], expected)

    def test_accumulate_m5_metric(self):
        metric = metrics.AccumulateM5Metric(bins=[0.5, 1.5, 2.5])
        slicer = slicers.HealpixSlicer(nside=16)
        sql = ""
        mb = metricBundle.MetricBundle(metric, slicer, sql)
        # Clobber the stacker that gets auto-added
        mb.stacker_list = []
        mbg = metricBundle.MetricBundleGroup({0: mb}, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)
        good = np.where(mb.metric_values.mask[:, -1] == False)[0]

        check_metric = metrics.Coaddm5Metric()
        temp_slice = np.zeros(self.n1, dtype=list(zip(["fiveSigmaDepth"], [float])))
        temp_slice["fiveSigmaDepth"] += self.m5_1
        val1 = check_metric.run(temp_slice)
        temp_slice = np.zeros(self.n2, dtype=list(zip(["fiveSigmaDepth"], [float])))
        temp_slice["fiveSigmaDepth"] += self.m5_2
        val2 = check_metric.run(temp_slice)

        expected = np.array([[val1, val1], [-666.0, val2]])
        assert np.array_equal(mb.metric_values.data[good, :], expected)

    def test_accumulate_uniformity_metric(self):
        names = ["night"]
        types = ["float"]
        data_slice = np.zeros(3652, dtype=list(zip(names, types)))

        # Test that a uniform distribution is very close to zero
        data_slice["night"] = np.arange(1, data_slice.size + 1)
        metric = metrics.AccumulateUniformityMetric()
        result = metric.run(data_slice)
        assert np.max(result) < 1.0 / 365.25
        assert np.min(result) >= 0

        # Test that if everythin on night 1 or last night, then result is ~1
        data_slice["night"] = 1
        result = metric.run(data_slice)
        assert np.max(result) >= 1.0 - 1.0 / 365.25
        data_slice["night"] = 3652
        result = metric.run(data_slice)
        assert np.max(result) >= 1.0 - 1.0 / 365.25

        # Test if all taken in the middle, result ~0.5
        data_slice["night"] = 3652 / 2
        result = metric.run(data_slice)
        assert np.max(result) >= 0.5 - 1.0 / 365.25

    def test_run_regular_too(self):
        """
        Test that a binned slicer and a regular slicer can run together
        """
        bundle_list = []
        metric = metrics.AccumulateM5Metric(bins=[0.5, 1.5, 2.5])
        slicer = slicers.HealpixSlicer(nside=16)
        sql = ""
        bundle_list.append(metricBundle.MetricBundle(metric, slicer, sql))
        metric = metrics.Coaddm5Metric()
        slicer = slicers.HealpixSlicer(nside=16)
        bundle_list.append(metricBundle.MetricBundle(metric, slicer, sql))
        for bundle in bundle_list:
            bundle.stacker_list = []
        bd = metricBundle.make_bundles_dict_from_list(bundle_list)
        mbg = metricBundle.MetricBundleGroup(bd, None, save_early=False)
        mbg.set_current("")
        mbg.run_current("", sim_data=self.sim_data)

        assert np.array_equal(
            bundle_list[0].metric_values[:, 1].compressed(),
            bundle_list[1].metric_values.compressed(),
        )


if __name__ == "__main__":
    unittest.main()
