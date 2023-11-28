import os
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils.code_utilities import sims_clean_up

from rubin_sim.phot_utils.bandpass import Bandpass
from rubin_sim.phot_utils.sed import Sed


class PhotometryUnitTest(unittest.TestCase):
    @classmethod
    def tearDown_class(cls):
        sims_clean_up()

    def test_alternate_bandpasses_stars(self):
        """Test our ability to do photometry using non-LSST bandpasses.

        Calculate the photometry by built-in methods and 'by hand'.
        """
        bandpass_dir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData")

        test_band_passes = {}
        keys = ["u", "g", "r", "i", "z"]

        bplist = []

        for kk in keys:
            test_band_passes[kk] = Bandpass()
            test_band_passes[kk].read_throughput(os.path.join(bandpass_dir, "test_bandpass_%s.dat" % kk))
            bplist.append(test_band_passes[kk])

        sed_obj = Sed()
        phi_array, wave_len_step = sed_obj.setup_phi_array(bplist)

        sed_file_name = os.path.join(get_data_dir(), "tests", "cartoonSedTestData/starSed/")
        sed_file_name = os.path.join(sed_file_name, "kurucz", "km20_5750.fits_g40_5790.gz")
        ss = Sed()
        ss.read_sed_flambda(sed_file_name)

        control_bandpass = Bandpass()
        control_bandpass.imsim_bandpass()
        ff = ss.calc_flux_norm(22.0, control_bandpass)
        ss.multiply_flux_norm(ff)

        test_mags = []
        for kk in keys:
            test_mags.append(ss.calc_mag(test_band_passes[kk]))

        ss.resample_sed(wavelen_match=bplist[0].wavelen)
        ss.flambda_tofnu()
        mags = -2.5 * np.log10(np.sum(phi_array * ss.fnu, axis=1) * wave_len_step) - ss.zp
        self.assertEqual(len(mags), len(test_mags))
        self.assertGreater(len(mags), 0)
        for j in range(len(mags)):
            self.assertAlmostEqual(mags[j], test_mags[j], 3)


if __name__ == "__main__":
    unittest.main()
