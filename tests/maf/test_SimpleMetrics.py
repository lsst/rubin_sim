import matplotlib

matplotlib.use("Agg")
import numpy as np
import unittest
import rubin_sim.maf.metrics as metrics


class TestSimpleMetrics(unittest.TestCase):
    def setUp(self):
        dv = np.arange(0, 10, 0.5)
        dv2 = np.arange(-10, 10.25, 0.5)
        self.dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        self.dv2 = np.array(list(zip(dv2)), dtype=[("testdata", "float")])

    def testMaxMetric(self):
        """Test max metric."""
        testmetric = metrics.MaxMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), self.dv["testdata"].max())

    def testMinMetric(self):
        """Test min metric."""
        testmetric = metrics.MinMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), self.dv["testdata"].min())

    def testMeanMetric(self):
        """Test mean metric."""
        testmetric = metrics.MeanMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), self.dv["testdata"].mean())

    def testMedianMetric(self):
        """Test median metric."""
        testmetric = metrics.MedianMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), np.median(self.dv["testdata"]))

    def testAbsMedianMetric(self):
        testmetric = metrics.AbsMedianMetric("testdata")
        self.assertEqual(
            testmetric.run(self.dv), np.abs(np.median(self.dv["testdata"]))
        )

    def testFullRangeMetric(self):
        """Test full range metric."""
        testmetric = metrics.FullRangeMetric("testdata")
        self.assertEqual(
            testmetric.run(self.dv),
            self.dv["testdata"].max() - self.dv["testdata"].min(),
        )

    def testCoaddm5Metric(self):
        """Test coaddm5 metric."""
        testmetric = metrics.Coaddm5Metric(m5Col="testdata")
        self.assertEqual(
            testmetric.run(self.dv),
            1.25 * np.log10(np.sum(10.0 ** (0.8 * self.dv["testdata"]))),
        )

    def testRmsMetric(self):
        """Test rms metric."""
        testmetric = metrics.RmsMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), np.std(self.dv["testdata"]))

    def testSumMetric(self):
        """Test Sum metric."""
        testmetric = metrics.SumMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), self.dv["testdata"].sum())

    def testCountUniqueMetric(self):
        """Test CountUniqueMetric"""
        testmetric = metrics.CountUniqueMetric("testdata")
        self.assertEqual(
            testmetric.run(self.dv), np.size(np.unique(self.dv["testdata"]))
        )
        d2 = self.dv.copy()
        d2["testdata"][1] = d2["testdata"][0]
        self.assertEqual(testmetric.run(d2), np.size(np.unique(d2)))

    def testCountMetric(self):
        """Test count metric."""
        testmetric = metrics.CountMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), np.size(self.dv["testdata"]))

    def testCountRatioMetric(self):
        """Test countratio metric."""
        testmetric = metrics.CountRatioMetric("testdata", normVal=2.0)
        self.assertEqual(testmetric.run(self.dv), np.size(self.dv["testdata"]) / 2.0)

    def testCountSubsetMetric(self):
        """Test countsubset metric."""
        testmetric = metrics.CountSubsetMetric("testdata", subset=0)
        self.assertEqual(testmetric.run(self.dv), 1)

    def testMaxPercentMetric(self):
        testmetric = metrics.MaxPercentMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), 1.0 / len(self.dv) * 100.0)
        self.assertEqual(testmetric.run(self.dv2), 1.0 / len(self.dv2) * 100.0)

    def testAbsMaxPercentMetric(self):
        testmetric = metrics.AbsMaxPercentMetric("testdata")
        self.assertEqual(testmetric.run(self.dv), 1.0 / len(self.dv) * 100.0)
        self.assertEqual(testmetric.run(self.dv2), 2.0 / len(self.dv2) * 100.0)

    def testRobustRmsMetric(self):
        """Test Robust RMS metric."""
        testmetric = metrics.RobustRmsMetric("testdata")
        rms_approx = (
            np.percentile(self.dv["testdata"], 75)
            - np.percentile(self.dv["testdata"], 25)
        ) / 1.349
        self.assertEqual(testmetric.run(self.dv), rms_approx)

    def testFracAboveMetric(self):
        cutoff = 5.1
        testmetric = metrics.FracAboveMetric("testdata", cutoff=cutoff)
        self.assertEqual(
            testmetric.run(self.dv),
            np.size(np.where(self.dv["testdata"] >= cutoff)[0])
            / float(np.size(self.dv)),
        )
        testmetric = metrics.FracAboveMetric("testdata", cutoff=cutoff, scale=2)
        self.assertEqual(
            testmetric.run(self.dv),
            2.0
            * np.size(np.where(self.dv["testdata"] >= cutoff)[0])
            / float(np.size(self.dv)),
        )

    def testFracBelowMetric(self):
        cutoff = 5.1
        testmetric = metrics.FracBelowMetric("testdata", cutoff=cutoff)
        self.assertEqual(
            testmetric.run(self.dv),
            np.size(np.where(self.dv["testdata"] <= cutoff)[0])
            / float(np.size(self.dv)),
        )
        testmetric = metrics.FracBelowMetric("testdata", cutoff=cutoff, scale=2)
        self.assertEqual(
            testmetric.run(self.dv),
            2.0
            * np.size(np.where(self.dv["testdata"] <= cutoff)[0])
            / float(np.size(self.dv)),
        )

    def testNoutliersNsigma(self):
        data = self.dv
        testmetric = metrics.NoutliersNsigmaMetric("testdata", nSigma=1.0)
        med = np.mean(data["testdata"])
        shouldBe = np.size(np.where(data["testdata"] > med + data["testdata"].std())[0])
        self.assertEqual(shouldBe, testmetric.run(data))
        testmetric = metrics.NoutliersNsigmaMetric("testdata", nSigma=-1.0)
        shouldBe = np.size(np.where(data["testdata"] < med - data["testdata"].std())[0])
        self.assertEqual(shouldBe, testmetric.run(data))

    def testMeanAngleMetric(self):
        """Test mean angle metric."""
        rng = np.random.RandomState(6573)
        dv1 = np.arange(0, 32, 2.5)
        dv2 = (dv1 - 20.0) % 360.0
        dv1 = np.array(list(zip(dv1)), dtype=[("testdata", "float")])
        dv2 = np.array(list(zip(dv2)), dtype=[("testdata", "float")])
        testmetric = metrics.MeanAngleMetric("testdata")
        result1 = testmetric.run(dv1)
        result2 = testmetric.run(dv2)
        self.assertAlmostEqual(result1, (result2 + 20) % 360.0)
        dv = rng.rand(10000) * 360.0
        dv = dv
        dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        result = testmetric.run(dv)
        result = result
        self.assertAlmostEqual(result, 180)

    def testFullRangeAngleMetric(self):
        """Test full range angle metric."""
        rng = np.random.RandomState(5422)
        dv1 = np.arange(0, 32, 2.5)
        dv2 = (dv1 - 20.0) % 360.0
        dv1 = np.array(list(zip(dv1)), dtype=[("testdata", "float")])
        dv2 = np.array(list(zip(dv2)), dtype=[("testdata", "float")])
        testmetric = metrics.FullRangeAngleMetric("testdata")
        result1 = testmetric.run(dv1)
        result2 = testmetric.run(dv2)
        self.assertAlmostEqual(result1, result2)
        dv = np.arange(0, 358, 5)
        dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        result = testmetric.run(dv)
        self.assertAlmostEqual(result, 355)
        dv = rng.rand(10000) * 360.0
        dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        result = testmetric.run(dv)
        result = result
        self.assertGreater(result, 355)

    def testSurfbMetric(self):
        """Test the surface brightness metric"""
        testmetric = metrics.SurfaceBrightLimitMetric()
        names = [
            "airmass",
            "visitExposureTime",
            "skyBrightness",
            "numExposures",
            "filter",
        ]
        types = [float] * 4 + ["|U1"]
        data = np.zeros(10, dtype=list(zip(names, types)))
        data["airmass"] = 1.2
        data["visitExposureTime"] = 30.0
        data["skyBrightness"] = 25.0
        data["numExposures"] = 2.0
        data["filter"] = "r"
        result = testmetric.run(data, None)


if __name__ == "__main__":
    unittest.main()
