"""Tests for ExtinctionStacker and CloudExtinctionStacker."""

import unittest

import numpy as np

import rubin_sim.maf.stackers as stackers
from rubin_sim.maf.stackers.extinction_stacker import (
    _fit_extinction_one_group,
    _fit_zp_extinction_one_group,
    _limits_from_prior,
)
from rubin_sim.phot_utils import predicted_zeropoint


def _make_visits(
    band,
    day_obs,
    true_k,
    true_zp_1s,
    n_visits=50,
    airmass_range=(1.0, 2.0),
    noise_sigma=0.01,
    exptime=30.0,
    rng=None,
):
    """Return a numpy structured array of synthetic visits for one group.

    Parameters
    ----------
    true_zp_1s : `float`
        True zenith zeropoint at 1-second exposure time.
    exptime : `float`
        Exposure time in seconds.  The raw zeropoint stored in
        ``zero_point_median`` will include the ``2.5*log10(exptime)`` term.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    airmass = rng.uniform(*airmass_range, size=n_visits)
    noise = rng.normal(0, noise_sigma, size=n_visits)
    # Raw zeropoint = 1s zeropoint - k*airmass + 2.5*log10(exptime) + noise
    zp = true_zp_1s - true_k * airmass + 2.5 * np.log10(exptime) + noise
    dtype = [
        ("band", "U1"),
        ("dayObs", int),
        ("airmass", float),
        ("zero_point_median", float),
        ("visitExposureTime", float),
    ]
    data = np.empty(n_visits, dtype=dtype)
    data["band"] = band
    data["dayObs"] = day_obs
    data["airmass"] = airmass
    data["zero_point_median"] = zp
    data["visitExposureTime"] = exptime
    return data


def _stack_many(arrays):
    """Concatenate a list of structured arrays with the same dtype."""
    return np.concatenate(arrays)


class TestFitZpExtinctionOneGroup(unittest.TestCase):
    """Unit tests for the _fit_zp_extinction_one_group helper (free ZP + k)."""

    def test_clean_data_recovery(self):
        """Fit recovers known k/zp on clean synthetic data."""
        rng = np.random.default_rng(0)
        airmass = rng.uniform(1.0, 2.0, 100)
        true_k, true_zp = 0.12, 32.4
        zp = true_zp - true_k * airmass + rng.normal(0, 0.01, 100)
        k, fitted_zp = _fit_zp_extinction_one_group(
            airmass,
            zp,
            k_min=0.0,
            k_max=0.4,
            zp_min=32.0,
            zp_max=33.0,
            zp_prior=32.5,
            residual_threshold=0.05,
            min_inliers=10,
            min_inlier_fraction=0.1,
        )
        self.assertFalse(np.isnan(k), "Fit should succeed on clean data")
        self.assertAlmostEqual(k, true_k, delta=0.02)
        self.assertAlmostEqual(fitted_zp, true_zp, delta=0.05)

    def test_outlier_rejection(self):
        """RANSAC should reject cloud-affected outliers and still recover k."""
        rng = np.random.default_rng(1)
        n = 80
        airmass = rng.uniform(1.0, 2.0, n)
        true_k, true_zp = 0.10, 32.4
        zp = true_zp - true_k * airmass + rng.normal(0, 0.01, n)
        # Inject 15 % heavily clouded outliers (very low zeropoint)
        n_out = int(0.15 * n)
        outlier_idx = rng.choice(n, n_out, replace=False)
        zp[outlier_idx] -= 2.0
        k, _ = _fit_zp_extinction_one_group(
            airmass,
            zp,
            k_min=0.0,
            k_max=0.4,
            zp_min=32.0,
            zp_max=33.0,
            zp_prior=32.5,
            residual_threshold=0.05,
            min_inliers=10,
            min_inlier_fraction=0.1,
        )
        self.assertFalse(np.isnan(k), "Fit should succeed despite outliers")
        self.assertAlmostEqual(k, true_k, delta=0.03)

    def test_too_few_visits_returns_nan(self):
        """Fit should return NaN when fewer than min_inliers visits."""
        rng = np.random.default_rng(2)
        airmass = rng.uniform(1.0, 2.0, 5)
        zp = 32.4 - 0.1 * airmass
        k, fzp = _fit_zp_extinction_one_group(
            airmass,
            zp,
            k_min=0.0,
            k_max=0.4,
            zp_min=32.0,
            zp_max=33.0,
            zp_prior=32.5,
            residual_threshold=0.05,
            min_inliers=10,
            min_inlier_fraction=0.1,
        )
        self.assertTrue(np.isnan(k))
        self.assertTrue(np.isnan(fzp))

    def test_unphysical_k_returns_nan(self):
        """Fit should return NaN when the true k is outside allowed bounds."""
        rng = np.random.default_rng(3)
        airmass = rng.uniform(1.0, 2.0, 50)
        # True k = 0.9, far above k_max=0.4
        zp = 32.4 - 0.9 * airmass + rng.normal(0, 0.005, 50)
        k, fzp = _fit_zp_extinction_one_group(
            airmass,
            zp,
            k_min=0.0,
            k_max=0.4,
            zp_min=32.0,
            zp_max=33.0,
            zp_prior=32.5,
            residual_threshold=0.05,
            min_inliers=10,
            min_inlier_fraction=0.1,
        )
        self.assertTrue(np.isnan(k))
        self.assertTrue(np.isnan(fzp))

    def test_empty_input_returns_nan(self):
        """Empty arrays should produce NaN without raising."""
        k, fzp = _fit_zp_extinction_one_group(
            np.array([]),
            np.array([]),
            k_min=0.0,
            k_max=0.4,
            zp_min=32.0,
            zp_max=33.0,
            zp_prior=32.5,
            residual_threshold=0.05,
            min_inliers=10,
            min_inlier_fraction=0.1,
        )
        self.assertTrue(np.isnan(k))
        self.assertTrue(np.isnan(fzp))


class TestFitExtinctionOneGroup(unittest.TestCase):
    """Unit tests for _fit_extinction_one_group (fixed ZP, k-only fit)."""

    # Common parameters used across tests.
    ZP_FIXED = 32.5
    K_MIN = 0.0
    K_MAX = 0.4

    def _call(self, airmass, zero_point, min_inliers=10, min_inlier_fraction=0.1):
        return _fit_extinction_one_group(
            airmass,
            zero_point,
            zp_fixed=self.ZP_FIXED,
            k_min=self.K_MIN,
            k_max=self.K_MAX,
            min_inliers=min_inliers,
            min_inlier_fraction=min_inlier_fraction,
        )

    def test_clean_data_recovery(self):
        """Median k should recover the true extinction on clean data."""
        rng = np.random.default_rng(50)
        true_k = 0.12
        airmass = rng.uniform(1.0, 2.0, 100)
        zp = self.ZP_FIXED - true_k * airmass + rng.normal(0, 0.01, 100)
        k, fitted_zp = self._call(airmass, zp)
        self.assertFalse(np.isnan(k), "Fit should succeed on clean data")
        self.assertAlmostEqual(k, true_k, delta=0.02)
        self.assertEqual(fitted_zp, self.ZP_FIXED, "Returned zeropoint must equal zp_fixed")

    def test_cloud_clip(self):
        """Cloud-dimmed visits (k_implied > k_max) should be excluded."""
        rng = np.random.default_rng(51)
        true_k = 0.10
        n = 80
        airmass = rng.uniform(1.0, 2.0, n)
        zp = self.ZP_FIXED - true_k * airmass + rng.normal(0, 0.01, n)
        # Inject 20 % heavily clouded visits: subtract 2 mag (high implied k)
        n_cloud = int(0.20 * n)
        cloud_idx = rng.choice(n, n_cloud, replace=False)
        zp[cloud_idx] -= 2.0
        k, _ = self._call(airmass, zp)
        self.assertFalse(np.isnan(k), "Fit should succeed despite cloudy outliers")
        self.assertAlmostEqual(k, true_k, delta=0.03)

    def test_low_extinction_not_clipped(self):
        """Visits with k_implied just above k_min should be retained."""
        rng = np.random.default_rng(52)
        # Use k close to k_min but still above it; all visits should survive.
        true_k = 0.02  # above K_MIN=0.0 but low
        airmass = rng.uniform(1.0, 2.0, 60)
        zp = self.ZP_FIXED - true_k * airmass + rng.normal(0, 0.005, 60)
        k, _ = self._call(airmass, zp)
        self.assertFalse(np.isnan(k), "Low-extinction visits must not be clipped")
        self.assertAlmostEqual(k, true_k, delta=0.02)

    def test_too_few_visits_returns_nan(self):
        """Fewer than min_inliers surviving visits should yield NaN."""
        rng = np.random.default_rng(53)
        airmass = rng.uniform(1.0, 2.0, 5)
        zp = self.ZP_FIXED - 0.10 * airmass
        k, fzp = self._call(airmass, zp)
        self.assertTrue(np.isnan(k))
        self.assertTrue(np.isnan(fzp))

    def test_empty_input_returns_nan(self):
        """Empty arrays should return NaN without raising."""
        k, fzp = self._call(np.array([]), np.array([]))
        self.assertTrue(np.isnan(k))
        self.assertTrue(np.isnan(fzp))


class TestExtinctionStacker(unittest.TestCase):
    """Integration tests for the ExtinctionStacker class."""

    # True 1-second zeropoints at airmass=0 (intercept of the fit model,
    # within DEFAULT_BAND_LIMITS zp windows)
    TRUE_ZP_1S_R = 28.48
    TRUE_ZP_1S_G = 28.72

    def _make_multi_band_data(self, rng=None):
        """Build a structured array with visits across two bands and nights."""
        if rng is None:
            rng = np.random.default_rng(10)
        pieces = []
        # Two bands, two nights each
        for band, true_k, true_zp_1s in [
            ("r", 0.09, self.TRUE_ZP_1S_R),
            ("g", 0.15, self.TRUE_ZP_1S_G),
        ]:
            for day_obs in [20260101, 20260102]:
                v = _make_visits(band, day_obs, true_k, true_zp_1s, rng=rng)
                pieces.append(v)
        return _stack_many(pieces)

    def test_columns_added(self):
        """Stacker should add extinction_k and fitted_zeropoint columns."""
        data = self._make_multi_band_data()
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(data)
        self.assertIn("extinction_k", result.dtype.names)
        self.assertIn("fitted_zeropoint", result.dtype.names)

    def test_recovered_k_per_group(self):
        """Stacker should recover reasonable k values per band/night."""
        rng = np.random.default_rng(11)
        true_k_r = 0.09
        visits = _make_visits("r", 20260101, true_k_r, self.TRUE_ZP_1S_R, n_visits=80, rng=rng)
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(visits)
        # All visits in one group should have the same k
        k_vals = result["extinction_k"]
        self.assertFalse(np.all(np.isnan(k_vals)), "Fit should succeed")
        self.assertTrue(
            np.allclose(k_vals, k_vals[0], equal_nan=False), "All visits in a group should share the same k"
        )
        self.assertAlmostEqual(float(k_vals[0]), true_k_r, delta=0.02)

    def test_nan_for_failed_fit(self):
        """Groups with insufficient data should yield NaN."""
        rng = np.random.default_rng(12)
        # Only 3 visits — below min_inliers=10
        visits = _make_visits("r", 20260103, 0.09, self.TRUE_ZP_1S_R, n_visits=3, rng=rng)
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(visits)
        self.assertTrue(np.all(np.isnan(result["extinction_k"])))
        self.assertTrue(np.all(np.isnan(result["fitted_zeropoint"])))

    def test_multiple_bands_independent(self):
        """Each band/night should get its own independent k estimate."""
        rng = np.random.default_rng(13)
        r_visits = _make_visits("r", 20260101, 0.09, self.TRUE_ZP_1S_R, n_visits=60, rng=rng)
        g_visits = _make_visits("g", 20260101, 0.18, self.TRUE_ZP_1S_G, n_visits=60, rng=rng)
        data = _stack_many([r_visits, g_visits])
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(data)

        r_mask = result["band"] == "r"
        g_mask = result["band"] == "g"
        k_r = result["extinction_k"][r_mask][0]
        k_g = result["extinction_k"][g_mask][0]
        self.assertFalse(np.isnan(k_r))
        self.assertFalse(np.isnan(k_g))
        # k values should differ (r ~0.09, g ~0.18)
        self.assertGreater(k_g, k_r)

    def test_cols_present_skips_recalculation(self):
        """If extinction_k already present (cols_present=True), skip refit."""
        rng = np.random.default_rng(14)
        visits = _make_visits("r", 20260101, 0.09, self.TRUE_ZP_1S_R, n_visits=60, rng=rng)
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(visits)
        # Manually corrupt the result to detect if _run is called again
        result["extinction_k"][:] = -999.0
        # Running again without override should not overwrite (cols_present)
        result2 = stacker.run(result)
        self.assertTrue(
            np.all(result2["extinction_k"] == -999.0),
            "Stacker should not recalculate when cols already present",
        )

    def test_unknown_band_no_crash(self):
        """Visits with unknown band should produce NaN, not crash."""
        rng = np.random.default_rng(15)
        visits = _make_visits("q", 20260101, 0.10, 27.0, n_visits=60, rng=rng)
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(visits)
        self.assertTrue(np.all(np.isnan(result["extinction_k"])))

    def test_custom_band_priors(self):
        """Custom band_priors should override the defaults."""
        rng = np.random.default_rng(16)
        custom_priors = {
            "r": {
                "expected_k": 0.09,
                "k_fraction_tolerance": 0.5,
                "expected_zp_X0": self.TRUE_ZP_1S_R,
                "zp_window": 0.5,
            }
        }
        visits = _make_visits("r", 20260101, 0.09, self.TRUE_ZP_1S_R, n_visits=60, rng=rng)
        stacker = stackers.ExtinctionStacker(band_priors=custom_priors)
        result = stacker.run(visits)
        self.assertFalse(
            np.all(np.isnan(result["extinction_k"])), "Should fit with custom priors that allow k~0.09"
        )

    def test_stage3_fallback_high_scatter(self):
        """Stage 3 should recover k when scatter is high enough to push the
        free-fit intercept outside its allowed window."""
        # Construct data where the free ZP intercept will be pulled out of
        # range by high scatter, but the slope (k) is physically reasonable.
        # Use a very tight zp_window so stages 1 and 2 reliably fail on the
        # intercept check, then confirm stage 3 returns a valid result.
        rng = np.random.default_rng(99)
        true_k = 0.09
        true_zp = self.TRUE_ZP_1S_R
        n = 60
        airmass = rng.uniform(1.0, 1.3, n)  # short lever arm → poor intercept
        # Large scatter: sigma = 0.3 mag, enough to pull the free-fit ZP out
        # of a ±0.05 mag window around true_zp.
        noise = rng.normal(0, 0.3, n)
        zp_1s = true_zp - true_k * airmass + noise

        tight_prior = {
            "expected_k": true_k,
            "k_fraction_tolerance": 0.5,
            "expected_zp_X0": true_zp,
            "zp_window": 0.05,  # very tight: stages 1 & 2 likely fail on ZP
        }
        k_min, k_max, zp_min, zp_max = _limits_from_prior(tight_prior)
        k, fitted_zp = _fit_zp_extinction_one_group(
            airmass,
            zp_1s,
            k_min=k_min,
            k_max=k_max,
            zp_min=zp_min,
            zp_max=zp_max,
            zp_prior=tight_prior["expected_zp_X0"],
            residual_threshold=0.05,
            min_inliers=10,
            min_inlier_fraction=0.1,
        )
        self.assertFalse(np.isnan(k), "Stage 3 should recover k on high-scatter data")
        # k must be within the allowed bounds (guaranteed by the bounded solver)
        self.assertGreaterEqual(k, k_min)
        self.assertLessEqual(k, k_max)
        # zp must also be within bounds
        self.assertGreaterEqual(fitted_zp, zp_min)
        self.assertLessEqual(fitted_zp, zp_max)

    def test_zp_window_zero_uses_k_only(self):
        """When zp_window=0 the stacker should fix zp and fit only k."""
        rng = np.random.default_rng(20)
        true_k = 0.09
        # Use a slightly wrong expected_zp_X0 to confirm it is not adjusted.
        fixed_zp = self.TRUE_ZP_1S_R + 0.05
        custom_priors = {
            "r": {
                "expected_k": true_k,
                "k_fraction_tolerance": 0.5,
                "expected_zp_X0": fixed_zp,
                "zp_window": 0.0,  # fixed-ZP mode
            }
        }
        visits = _make_visits("r", 20260101, true_k, self.TRUE_ZP_1S_R, n_visits=80, rng=rng)
        stacker = stackers.ExtinctionStacker(band_priors=custom_priors)
        result = stacker.run(visits)

        k_vals = result["extinction_k"]
        zp_vals = result["fitted_zeropoint"]

        self.assertFalse(np.all(np.isnan(k_vals)), "Fixed-ZP fit should succeed")
        # All visits in the group share the same k
        self.assertTrue(np.allclose(k_vals, k_vals[0], equal_nan=False))
        # k should be reasonable (within 0.05 of true; slight bias from wrong ZP is OK)
        self.assertAlmostEqual(float(k_vals[0]), true_k, delta=0.05)
        # fitted_zeropoint must be exactly the fixed value, never adjusted
        self.assertTrue(
            np.all(zp_vals == fixed_zp),
            "fitted_zeropoint must equal expected_zp_X0 when zp_window=0",
        )

    def test_registered_in_registry(self):
        """ExtinctionStacker should appear in the stacker registry."""
        self.assertIn("ExtinctionStacker", stackers.BaseStacker.registry)

    def test_cols_added_dtypes(self):
        """Both added columns should be float dtype."""
        rng = np.random.default_rng(17)
        visits = _make_visits("r", 20260101, 0.09, self.TRUE_ZP_1S_R, n_visits=60, rng=rng)
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(visits)
        self.assertTrue(np.issubdtype(result["extinction_k"].dtype, np.floating))
        self.assertTrue(np.issubdtype(result["fitted_zeropoint"].dtype, np.floating))

    def test_mixed_exposure_times(self):
        """Stacker should handle mixed exposure times within a group."""
        rng = np.random.default_rng(18)
        true_k = 0.10
        true_zp_1s = self.TRUE_ZP_1S_R
        # Create visits with 15s and 30s exposures mixed together
        v1 = _make_visits("r", 20260101, true_k, true_zp_1s, n_visits=40, exptime=15.0, rng=rng)
        v2 = _make_visits("r", 20260101, true_k, true_zp_1s, n_visits=40, exptime=30.0, rng=rng)
        data = _stack_many([v1, v2])
        stacker = stackers.ExtinctionStacker()
        result = stacker.run(data)
        k_vals = result["extinction_k"]
        self.assertFalse(np.all(np.isnan(k_vals)), "Fit should succeed with mixed exptimes")
        self.assertAlmostEqual(float(k_vals[0]), true_k, delta=0.02)


def _make_visits_with_extinction_cols(
    band,
    day_obs,
    true_k,
    true_zp_1s,
    cloud_ext=0.0,
    n_visits=50,
    airmass_range=(1.0, 2.0),
    noise_sigma=0.01,
    exptime=30.0,
    rng=None,
):
    """Return visits with extinction_k and fitted_zeropoint already set.

    Simulates what the data looks like after ExtinctionStacker has run.
    The zero_point_median includes the effect of clouds.

    Parameters
    ----------
    true_zp_1s : `float`
        True zenith zeropoint at 1-second exposure time.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    airmass = rng.uniform(*airmass_range, size=n_visits)
    noise = rng.normal(0, noise_sigma, size=n_visits)
    # Raw observed zp includes exptime term, extinction, and clouds
    observed_zp = true_zp_1s - true_k * airmass + 2.5 * np.log10(exptime) - cloud_ext + noise
    dtype = [
        ("band", "U1"),
        ("dayObs", int),
        ("airmass", float),
        ("zero_point_median", float),
        ("visitExposureTime", float),
        ("extinction_k", float),
        ("fitted_zeropoint", float),
    ]
    data = np.empty(n_visits, dtype=dtype)
    data["band"] = band
    data["dayObs"] = day_obs
    data["airmass"] = airmass
    data["zero_point_median"] = observed_zp
    data["visitExposureTime"] = exptime
    data["extinction_k"] = true_k
    data["fitted_zeropoint"] = true_zp_1s  # 1-second scale
    return data


class TestCloudExtinctionStacker(unittest.TestCase):
    """Tests for the CloudExtinctionStacker class."""

    def test_column_added(self):
        """Stacker should add cloud_extinction column."""
        rng = np.random.default_rng(30)
        visits = _make_visits_with_extinction_cols("r", 20260101, 0.10, 32.35, rng=rng)
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)
        self.assertIn("cloud_extinction", result.dtype.names)

    def test_clear_sky_near_zero(self):
        """cloud_extinction should be near zero for clear-sky visits."""
        rng = np.random.default_rng(31)
        visits = _make_visits_with_extinction_cols(
            "r",
            20260101,
            0.10,
            32.35,
            cloud_ext=0.0,
            n_visits=100,
            noise_sigma=0.01,
            rng=rng,
        )
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)
        # Should be near zero (within noise)
        self.assertAlmostEqual(float(np.median(result["cloud_extinction"])), 0.0, delta=0.02)

    def test_cloudy_visits_positive(self):
        """cloud_extinction should be positive for cloudy visits."""
        rng = np.random.default_rng(32)
        cloud_mag = 0.5
        visits = _make_visits_with_extinction_cols(
            "r",
            20260101,
            0.10,
            32.35,
            cloud_ext=cloud_mag,
            n_visits=100,
            noise_sigma=0.01,
            rng=rng,
        )
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)
        median_cloud = float(np.median(result["cloud_extinction"]))
        self.assertAlmostEqual(median_cloud, cloud_mag, delta=0.03)

    def test_fallback_to_predicted_zeropoint_when_nan(self):
        """When extinction_k is NaN, predicted_zeropoint should be used."""
        rng = np.random.default_rng(33)
        exptime = 30.0
        cloud_mag = 0.3
        n_visits = 60

        # Create visits with airmass/band/exptime, compute what
        # predicted_zeropoint would give, then set observed_zp accordingly.
        airmass = rng.uniform(1.0, 2.0, n_visits)
        noise = rng.normal(0, 0.01, n_visits)
        # predicted_zeropoint gives the expected zp for clear sky
        expected_zp = predicted_zeropoint("r", airmass, exptime)
        observed_zp = expected_zp - cloud_mag + noise

        dtype = [
            ("band", "U1"),
            ("dayObs", int),
            ("airmass", float),
            ("zero_point_median", float),
            ("visitExposureTime", float),
            ("extinction_k", float),
            ("fitted_zeropoint", float),
        ]
        visits = np.empty(n_visits, dtype=dtype)
        visits["band"] = "r"
        visits["dayObs"] = 20260101
        visits["airmass"] = airmass
        visits["zero_point_median"] = observed_zp
        visits["visitExposureTime"] = exptime
        visits["extinction_k"] = np.nan
        visits["fitted_zeropoint"] = np.nan

        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)

        # Should recover ~cloud_mag via the predicted_zeropoint fallback
        median_cloud = float(np.median(result["cloud_extinction"]))
        self.assertAlmostEqual(median_cloud, cloud_mag, delta=0.03)

    def test_mixed_nan_and_fitted(self):
        """NaN fallback should apply only to visits with NaN extinction_k."""
        rng = np.random.default_rng(34)
        exptime = 30.0

        # Night 1: fit succeeded, no clouds
        v1 = _make_visits_with_extinction_cols(
            "r",
            20260101,
            0.10,
            32.35,
            cloud_ext=0.0,
            n_visits=40,
            noise_sigma=0.005,
            exptime=exptime,
            rng=rng,
        )
        # Night 2: fit failed (NaN), 0.4 mag clouds
        # Use predicted_zeropoint to generate the "true" expected values
        airmass_n2 = rng.uniform(1.0, 2.0, 40)
        noise_n2 = rng.normal(0, 0.005, 40)
        expected_zp_n2 = predicted_zeropoint("r", airmass_n2, exptime)
        observed_zp_n2 = expected_zp_n2 - 0.4 + noise_n2

        dtype = v1.dtype
        v2 = np.empty(40, dtype=dtype)
        v2["band"] = "r"
        v2["dayObs"] = 20260102
        v2["airmass"] = airmass_n2
        v2["zero_point_median"] = observed_zp_n2
        v2["visitExposureTime"] = exptime
        v2["extinction_k"] = np.nan
        v2["fitted_zeropoint"] = np.nan

        data = np.concatenate([v1, v2])
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(data)

        # Night 1 visits should be near zero
        night1_mask = result["dayObs"] == 20260101
        self.assertAlmostEqual(
            float(np.median(result["cloud_extinction"][night1_mask])),
            0.0,
            delta=0.02,
        )
        # Night 2 visits should be near 0.4
        night2_mask = result["dayObs"] == 20260102
        self.assertAlmostEqual(
            float(np.median(result["cloud_extinction"][night2_mask])),
            0.4,
            delta=0.03,
        )

    def test_cols_present_skips_recalculation(self):
        """If cloud_extinction already present, skip recalculation."""
        rng = np.random.default_rng(36)
        visits = _make_visits_with_extinction_cols("r", 20260101, 0.10, 32.35, rng=rng)
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)
        # Corrupt and re-run
        result["cloud_extinction"][:] = -999.0
        result2 = stacker.run(result)
        self.assertTrue(
            np.all(result2["cloud_extinction"] == -999.0),
            "Stacker should not recalculate when col already present",
        )

    def test_unknown_band_nan_fallback(self):
        """Visits with a band not supported by predicted_zeropoint get NaN."""
        rng = np.random.default_rng(37)
        visits = _make_visits_with_extinction_cols("q", 20260101, 0.10, 31.0, rng=rng)
        visits["extinction_k"] = np.nan
        visits["fitted_zeropoint"] = np.nan
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)
        self.assertTrue(np.all(np.isnan(result["cloud_extinction"])))

    def test_registered_in_registry(self):
        """CloudExtinctionStacker should appear in the stacker registry."""
        self.assertIn("CloudExtinctionStacker", stackers.BaseStacker.registry)

    def test_missing_required_column_no_crash(self):
        """Should produce NaN and not crash if a required column is missing."""
        rng = np.random.default_rng(38)
        # Make data WITHOUT extinction_k column
        visits = _make_visits("r", 20260101, 0.10, 32.35, n_visits=30, rng=rng)
        stacker = stackers.CloudExtinctionStacker()
        result = stacker.run(visits)
        self.assertIn("cloud_extinction", result.dtype.names)
        self.assertTrue(np.all(np.isnan(result["cloud_extinction"])))

    def test_exptime_affects_fallback(self):
        """Different exptimes should give different fallback zeropoints."""
        rng = np.random.default_rng(39)
        n_visits = 40
        airmass = rng.uniform(1.0, 1.5, n_visits)

        # Two sets with different exposure times, both with NaN extinction
        for exptime in [15.0, 30.0]:
            expected_zp = predicted_zeropoint("r", airmass, exptime)
            cloud_mag = 0.25
            observed_zp = expected_zp - cloud_mag

            dtype = [
                ("band", "U1"),
                ("dayObs", int),
                ("airmass", float),
                ("zero_point_median", float),
                ("visitExposureTime", float),
                ("extinction_k", float),
                ("fitted_zeropoint", float),
            ]
            visits = np.empty(n_visits, dtype=dtype)
            visits["band"] = "r"
            visits["dayObs"] = 20260101
            visits["airmass"] = airmass
            visits["zero_point_median"] = observed_zp
            visits["visitExposureTime"] = exptime
            visits["extinction_k"] = np.nan
            visits["fitted_zeropoint"] = np.nan

            stacker = stackers.CloudExtinctionStacker()
            result = stacker.run(visits)
            median_cloud = float(np.median(result["cloud_extinction"]))
            self.assertAlmostEqual(median_cloud, cloud_mag, delta=0.01, msg=f"Failed for exptime={exptime}")

    def test_zeropoint_offsets(self):
        """zeropoint_offsets should shift the fallback predicted zeropoint."""
        rng = np.random.default_rng(40)
        exptime = 30.0
        n_visits = 60
        offset = 0.15  # instrument is 0.15 mag brighter than model

        airmass = rng.uniform(1.0, 2.0, n_visits)
        # The "true" expected zp includes the offset
        true_expected_zp = predicted_zeropoint("r", airmass, exptime) + offset
        cloud_mag = 0.2
        observed_zp = true_expected_zp - cloud_mag

        dtype = [
            ("band", "U1"),
            ("dayObs", int),
            ("airmass", float),
            ("zero_point_median", float),
            ("visitExposureTime", float),
            ("extinction_k", float),
            ("fitted_zeropoint", float),
        ]
        visits = np.empty(n_visits, dtype=dtype)
        visits["band"] = "r"
        visits["dayObs"] = 20260101
        visits["airmass"] = airmass
        visits["zero_point_median"] = observed_zp
        visits["visitExposureTime"] = exptime
        visits["extinction_k"] = np.nan
        visits["fitted_zeropoint"] = np.nan

        # Without offset, cloud_extinction would be biased by 0.15
        stacker_no_offset = stackers.CloudExtinctionStacker()
        result_no_offset = stacker_no_offset.run(visits.copy())
        median_no_offset = float(np.median(result_no_offset["cloud_extinction"]))
        # Should be cloud_mag - offset = 0.05 (biased low)
        self.assertAlmostEqual(median_no_offset, cloud_mag - offset, delta=0.02)

        # With offset, should correctly recover cloud_mag
        stacker_with_offset = stackers.CloudExtinctionStacker(zeropoint_offsets={"r": offset})
        result_with_offset = stacker_with_offset.run(visits.copy())
        median_with_offset = float(np.median(result_with_offset["cloud_extinction"]))
        self.assertAlmostEqual(median_with_offset, cloud_mag, delta=0.02)


if __name__ == "__main__":
    unittest.main()
