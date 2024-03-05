"""
Module to test that Approximate Bandpasses are in sync with the official LSST
Bandpasses from SYSENG

Note: While the LSST bandpasses list throughput values corresponding to
wavelengths in the range of 300.0-1150.0 nm, the `approximate_baseline`
directory of throughputs is created by a script manually. It is thus possible
for this directory to fall out of sync with the SYSENG values in `baseline`.
This module is intended to test whether this is happening.
"""

import os
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.phot_utils import Bandpass, PhotometricParameters


class ApproximateBandPassTest(unittest.TestCase):
    """
    Tests for the approximate Bandpasses in the throughputs directory
    """

    long_message = True

    def setUp(self):
        """Setup before tests."""
        throughputs_dir = os.path.join(get_data_dir(), "throughputs")
        self.approx_band_pass_dir = os.path.join(throughputs_dir, "approximate_baseline")
        self.approx_band_pass_dict = {}
        for kk in ["u", "g", "r", "i", "z", "y"]:
            self.approx_band_pass_dict[kk] = Bandpass()
            self.approx_band_pass_dict[kk].read_throughput(
                os.path.join(self.approx_band_pass_dir, f"total_{kk}.dat")
            )
        self.ref_band_pass_dir = os.path.join(throughputs_dir, "baseline")
        self.ref_band_pass_dict = {}
        for kk in ["u", "g", "r", "i", "z", "y"]:
            self.ref_band_pass_dict[kk] = Bandpass()
            self.ref_band_pass_dict[kk].read_throughput(
                os.path.join(self.ref_band_pass_dir, f"total_{kk}.dat")
            )
        self.error_msg = "The failure of this test indicates that the"
        " approximate bandpasses in the lsst throughputs directory do not"
        "sync up with the baseline bandpasses is throughputs. "
        " This may require running the script :"
        " throughputs.approximate_baseline/approximateBandpasses.py"

    def test_band_pass_integrals(self):
        """Test that the ratio of the quantity
        int dlambda T(lambda) = band flux for a SED proportional to $lambda$
        for the approximate bandpasses to the SYSENG band passes is 1.0 to an
        absolute tolerance hard coded to be 1.0e-4

        """
        phot_params = PhotometricParameters()

        for bn in "ugrizy":
            ref_band_pass = self.ref_band_pass_dict[bn]
            approx_band_pass = self.approx_band_pass_dict[bn]

            zp_ref = ref_band_pass.calc_zp_t(phot_params)
            zp_approx = approx_band_pass.calc_zp_t(phot_params)
            self.assertAlmostEqual(zp_ref, zp_approx, delta=1.0e-4, msg=self.error_msg)

    def test_bandpasses_indiv(self):
        """
        Test that individual transmission values at wavelengths kept in the
        approximate bandpasses match the individual transmission values in
        the reference bandpasses
        """
        for bn in "ugrizy":
            approx_band_pass = self.approx_band_pass_dict[bn]
            ref_band_pass = self.ref_band_pass_dict[bn]
            refwavelen = ref_band_pass.wavelen
            approxwavelen = approx_band_pass.wavelen
            mask = np.zeros(len(refwavelen), dtype=bool)
            for i, wave in enumerate(refwavelen):
                if wave in approxwavelen:
                    mask[i] = True
            # Assume that the wavelengths are in order
            np.testing.assert_array_almost_equal(ref_band_pass.sb[mask], approx_band_pass.sb, decimal=3)

        def tearDown(self):
            pass


if __name__ == "__main__":
    unittest.main()
