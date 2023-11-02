import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.stackers as stackers


class TestCalibrationMetrics(unittest.TestCase):
    def test_parallax_metric(self):
        """
        Test the parallax metric.
        """
        names = [
            "observationStartMJD",
            "finSeeing",
            "fiveSigmaDepth",
            "fieldRA",
            "fieldDec",
            "filter",
        ]
        types = [float, float, float, float, float, (np.str_, 1)]
        data = np.zeros(700, dtype=list(zip(names, types)))
        slice_point = {"sid": 0}
        data["observationStartMJD"] = np.arange(700) + 56762
        data["finSeeing"] = 0.7
        data["filter"][0:100] = str("r")
        data["filter"][100:200] = str("u")
        data["filter"][200:] = str("g")
        data["fiveSigmaDepth"] = 24.0
        stacker = stackers.ParallaxFactorStacker()
        data = stacker.run(data)
        norm_flags = [False, True]
        for flag in norm_flags:
            data["finSeeing"] = 0.7
            data["fiveSigmaDepth"] = 24.0
            baseline = metrics.ParallaxMetric(normalize=flag, seeing_col="finSeeing").run(data, slice_point)
            data["finSeeing"] = data["finSeeing"] + 0.3
            worse1 = metrics.ParallaxMetric(normalize=flag, seeing_col="finSeeing").run(data, slice_point)
            worse2 = metrics.ParallaxMetric(normalize=flag, rmag=22.0, seeing_col="finSeeing").run(
                data, slice_point
            )
            worse3 = metrics.ParallaxMetric(normalize=flag, rmag=22.0, seeing_col="finSeeing").run(
                data[0:300], slice_point
            )
            data["fiveSigmaDepth"] = data["fiveSigmaDepth"] - 1.0
            worse4 = metrics.ParallaxMetric(normalize=flag, rmag=22.0, seeing_col="finSeeing").run(
                data[0:300], slice_point
            )
            # Make sure the RMS increases as seeing increases,
            # the star gets fainter, the background gets brighter,
            # or the baseline decreases.
            if flag:
                pass
            else:
                assert worse1 > baseline
                assert worse2 > worse1
                assert worse3 > worse2
                assert worse4 > worse3

    def test_proper_motion_metric(self):
        """
        Test the ProperMotion metric.
        """
        names = [
            "observationStartMJD",
            "finSeeing",
            "fiveSigmaDepth",
            "fieldRA",
            "fieldDec",
            "filter",
        ]
        types = [float, float, float, float, float, (np.str_, 1)]
        data = np.zeros(700, dtype=list(zip(names, types)))
        slice_point = [0]
        stacker = stackers.ParallaxFactorStacker()
        norm_flags = [False, True]
        data["observationStartMJD"] = np.arange(700) + 56762
        data["finSeeing"] = 0.7
        data["filter"][0:100] = str("r")
        data["filter"][100:200] = str("u")
        data["filter"][200:] = str("g")
        data["fiveSigmaDepth"] = 24.0
        data = stacker.run(data)
        for flag in norm_flags:
            data["finSeeing"] = 0.7
            data["fiveSigmaDepth"] = 24
            baseline = metrics.ProperMotionMetric(normalize=flag, seeing_col="finSeeing").run(
                data, slice_point
            )
            data["finSeeing"] = data["finSeeing"] + 0.3
            worse1 = metrics.ProperMotionMetric(normalize=flag, seeing_col="finSeeing").run(data, slice_point)
            worse2 = metrics.ProperMotionMetric(normalize=flag, rmag=22.0, seeing_col="finSeeing").run(
                data, slice_point
            )
            worse3 = metrics.ProperMotionMetric(normalize=flag, rmag=22.0, seeing_col="finSeeing").run(
                data[0:300], slice_point
            )
            data["fiveSigmaDepth"] = data["fiveSigmaDepth"] - 1.0
            worse4 = metrics.ProperMotionMetric(normalize=flag, rmag=22.0, seeing_col="finSeeing").run(
                data[0:300], slice_point
            )
            # Make sure the RMS increases as seeing increases,
            # the star gets fainter, the background gets brighter,
            # or the baseline decreases.
            if flag:
                # When normalized, mag of star and m5 don't matter
                # (just scheduling).
                self.assertAlmostEqual(worse2, worse1)
                self.assertAlmostEqual(worse4, worse3)
                # But using fewer points should make proper motion worse.
                # survey assumed to have same seeing and limiting mags.
                assert worse3 < worse2
            else:
                assert worse1 > baseline
                assert worse2 > worse1
                assert worse3 > worse2
                assert worse4 > worse3

    def test_parallax_coverage_metric(self):
        """
        Test the parallax coverage
        """
        names = [
            "observationStartMJD",
            "finSeeing",
            "fiveSigmaDepth",
            "fieldRA",
            "fieldDec",
            "filter",
            "ra_pi_amp",
            "dec_pi_amp",
        ]
        types = [float, float, float, float, float, "<U1", float, float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        data["filter"] = "r"
        data["fiveSigmaDepth"] = 25.0
        data["ra_pi_amp"] = 1.0
        data["dec_pi_amp"] = 1.0

        # All the parallax amplitudes are the same, should return zero
        metric = metrics.ParallaxCoverageMetric(seeing_col="finSeeing")
        val = metric.run(data)
        assert val == 0

        # Half at (1,1), half at (0.5,0.5)
        data["ra_pi_amp"][0:50] = 1
        data["dec_pi_amp"][0:50] = 1
        data["ra_pi_amp"][50:] = -1
        data["dec_pi_amp"][50:] = -1
        val = metric.run(data)
        self.assertAlmostEqual(val, 2.0**0.5)

        data["ra_pi_amp"][0:50] = 0.5
        data["dec_pi_amp"][0:50] = 0.5
        data["ra_pi_amp"][50:] = -0.5
        data["dec_pi_amp"][50:] = -0.5
        val = metric.run(data)
        self.assertAlmostEqual(val, 0.5 * 2**0.5)

        data["ra_pi_amp"][0:50] = 1
        data["dec_pi_amp"][0:50] = 0
        data["ra_pi_amp"][50:] = -1
        data["dec_pi_amp"][50:] = 0
        val = metric.run(data)
        assert val == 1

    def test_parallax_dcr_degen_metric(self):
        """
        Test the parallax-DCR degeneracy metric
        """
        names = [
            "observationStartMJD",
            "finSeeing",
            "fiveSigmaDepth",
            "fieldRA",
            "fieldDec",
            "filter",
            "ra_pi_amp",
            "dec_pi_amp",
            "ra_dcr_amp",
            "dec_dcr_amp",
        ]
        types = [float, float, float, float, float, "<U1", float, float, float, float]
        data = np.zeros(100, dtype=list(zip(names, types)))
        data["filter"] = "r"
        data["fiveSigmaDepth"] = 25.0

        # Set so ra is perfecly correlated
        data["ra_pi_amp"] = 1.0
        data["dec_pi_amp"] = 0.01
        data["ra_dcr_amp"] = 0.2

        metric = metrics.ParallaxDcrDegenMetric(seeing_col="finSeeing")
        val = metric.run(data)
        np.testing.assert_almost_equal(np.abs(val), 1.0, decimal=2)

        # set so the offsets are always nearly perpendicular
        data["ra_pi_amp"] = 0.001
        data["dec_pi_amp"] = 1.0
        data["ra_dcr_amp"] = 0.2

        metric = metrics.ParallaxDcrDegenMetric(seeing_col="finSeeing")
        val = metric.run(data)
        np.testing.assert_almost_equal(val, 0.0, decimal=2)

        # Generate a random distribution that should have
        # little or no correlation
        rng = np.random.RandomState(42)

        data["ra_pi_amp"] = rng.rand(100) * 2 - 1.0
        data["dec_pi_amp"] = rng.rand(100) * 2 - 1.0
        data["ra_dcr_amp"] = rng.rand(100) * 2 - 1.0
        data["dec_dcr_amp"] = rng.rand(100) * 2 - 1.0

        val = metric.run(data)
        assert np.abs(val) < 0.2

    def test_radius_obs_metric(self):
        """
        Test the RadiusObsMetric
        """

        names = ["fieldRA", "fieldDec"]
        dt = ["float"] * 2
        data = np.zeros(3, dtype=list(zip(names, dt)))
        data["fieldDec"] = [-0.1, 0, 0.1]
        slice_point = {"ra": 0.0, "dec": 0.0}
        metric = metrics.RadiusObsMetric()
        result = metric.run(data, slice_point)
        for i, r in enumerate(result):
            np.testing.assert_almost_equal(r, abs(data["fieldDec"][i]))
        assert metric.reduce_mean(result) == np.mean(result)
        assert metric.reduce_rms(result) == np.std(result)
        np.testing.assert_almost_equal(
            metric.reduce_full_range(result),
            np.max(np.abs(data["fieldDec"])) - np.min(np.abs(data["fieldDec"])),
        )


if __name__ == "__main__":
    unittest.main()
