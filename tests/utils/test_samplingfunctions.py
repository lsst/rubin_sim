"""Tests for sims.utils.samplingFunctions.py

1.  test_raiseWraparoundError: The `sample_patch_on_sphere` function  does not
 wrap around theta values near the pole. Check if the approporiate error is
 by such a call
2. test_checkWithinBounds : Check that the samples are indeed within the bounds
prescribed by ObsMetaData
3. test_sample_patch_on_sphere : Check functionality by showing that binning up in
    dec results in numbers in dec bins changing with area.
 """
import unittest

import numpy as np

from rubin_sim.utils import ObservationMetaData, sample_patch_on_sphere, spatially_sample_obsmetadata


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """"""
        cls.obs_meta_datafor_cat = ObservationMetaData(
            bound_type="circle",
            bound_length=np.degrees(0.25),
            pointing_ra=np.degrees(0.13),
            pointing_dec=np.degrees(-1.2),
            bandpass_name=["r"],
            mjd=49350.0,
        )
        obs_meta_data = cls.obs_meta_datafor_cat
        cls.samples = spatially_sample_obsmetadata(obs_meta_data, size=1000)

        cls.theta_c = -60.0
        cls.phi_c = 30.0
        cls.delta = 30.0
        cls.size = 1000000

        cls.dense_samples = sample_patch_on_sphere(
            phi=cls.phi_c, theta=cls.theta_c, delta=cls.delta, size=cls.size, seed=42
        )

    def test_raise_wraparound_error(self):
        """
        Test that appropriate errors are raised when at the poles
        """
        # thetamax to be exceeded
        deltamax = np.abs(self.theta_c - 0.05)
        # to be lower than thetamin
        deltamin = 2.67
        with self.assertRaises(ValueError):
            sample_patch_on_sphere(
                phi=self.phi_c,
                theta=self.theta_c,
                delta=deltamax,
                size=self.size,
                seed=42,
            )
            sample_patch_on_sphere(
                phi=self.phi_c,
                theta=self.theta_c,
                delta=deltamin,
                size=self.size,
                seed=42,
            )

    def test_check_within_bounds(self):
        delta = self.obs_meta_datafor_cat.bound_length
        # delta = np.radians(delta)
        min_phi = 0.13 - delta
        max_phi = 0.13 + delta
        min_theta = -1.2 - delta
        max_theta = -1.2 + delta

        self.assertTrue(
            all(np.radians(self.samples[0]) <= max_phi),
            msg="samples are not <= max_phi",
        )
        self.assertTrue(
            all(np.radians(self.samples[0]) >= min_phi),
            msg="samples are not >= min_phi",
        )
        self.assertTrue(
            all(np.radians(self.samples[1]) >= min_theta),
            msg="samples are not >= min_theta",
        )
        self.assertTrue(
            all(np.radians(self.samples[1]) <= max_theta),
            msg="samples are not <= max_theta",
        )

    def test_sample_patch_on_sphere(self):
        def A(theta_min, theta_max):
            return np.sin(theta_max) - np.sin(theta_min)

        theta_c = np.radians(self.theta_c)
        delta = np.radians(self.delta)

        theta_min = theta_c - delta
        theta_max = theta_c + delta
        tvals = np.arange(theta_min, theta_max, 0.001)
        tvals_shifted = np.zeros(len(tvals))
        tvals_shifted[:-1] = tvals[1:]

        area = A(tvals, tvals_shifted)

        bin_size = np.unique(np.diff(tvals))
        self.assertEqual(bin_size.size, 1)
        normval = np.sum(area) * bin_size[0]

        theta_samps = np.radians(self.dense_samples[1])
        binnedvals = np.histogram(theta_samps, bins=tvals[:-1], density=True)[0]
        resids = area[:-2] / normval - binnedvals

        five_sigma = np.sqrt(binnedvals) * 5.0
        np.testing.assert_array_less(resids, five_sigma)


if __name__ == "__main__":
    unittest.main()
