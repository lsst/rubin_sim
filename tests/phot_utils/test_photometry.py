import numpy as np

import os
import unittest
from rubin_sim.utils import ObservationMetaData
from rubin_sim.utils.code_utilities import sims_clean_up
from rubin_sim.phot_utils.bandpass import Bandpass
from rubin_sim.phot_utils.sed import Sed
from rubin_sim.phot_utils import BandpassDict
from rubin_sim.data import get_data_dir
import rubin_sim


class PhotometryUnitTest(unittest.TestCase):
    @classmethod
    def tearDown_class(cls):
        sims_clean_up()

    def setUp(self):
        self.obs_metadata = ObservationMetaData(
            mjd=52000.7,
            bandpassName="i",
            boundType="circle",
            pointingRA=200.0,
            pointingDec=-30.0,
            boundLength=1.0,
            m5=25.0,
        )

    def tearDown(self):
        del self.obs_metadata

    def test_alternate_bandpasses_stars(self):
        """
        This will test our ability to do photometry using non-LSST bandpasses.

        It will first calculate the magnitudes using the getters in cartoonPhotometryStars.

        It will then load the alternate bandpass files 'by hand' and re-calculate the magnitudes
        and make sure that the magnitude values agree.  This is guarding against the possibility
        that some default value did not change and the code actually ended up loading the
        LSST bandpasses.
        """
        bandpass_dir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData")

        cartoon_dict = BandpassDict.load_total_bandpasses_from_files(
            ["u", "g", "r", "i", "z"],
            bandpass_dir=bandpass_dir,
            bandpass_root="test_bandpass_",
        )

        test_band_passes = {}
        keys = ["u", "g", "r", "i", "z"]

        bplist = []

        for kk in keys:
            test_band_passes[kk] = Bandpass()
            test_band_passes[kk].read_throughput(
                os.path.join(bandpass_dir, "test_bandpass_%s.dat" % kk)
            )
            bplist.append(test_band_passes[kk])

        sed_obj = Sed()
        phi_array, wave_len_step = sed_obj.setup_phi_array(bplist)

        sed_file_name = os.path.join(
            get_data_dir(), "tests", "cartoonSedTestData/starSed/"
        )
        sed_file_name = os.path.join(
            sed_file_name, "kurucz", "km20_5750.fits_g40_5790.gz"
        )
        ss = Sed()
        ss.read_sed_flambda(sed_file_name)

        control_bandpass = Bandpass()
        control_bandpass.imsim_bandpass()
        ff = ss.calc_flux_norm(22.0, control_bandpass)
        ss.multiply_flux_norm(ff)

        test_mags = cartoon_dict.mag_list_for_sed(ss)

        ss.resample_sed(wavelen_match=bplist[0].wavelen)
        ss.flambda_tofnu()
        mags = (
            -2.5 * np.log10(np.sum(phi_array * ss.fnu, axis=1) * wave_len_step) - ss.zp
        )
        self.assertEqual(len(mags), len(test_mags))
        self.assertGreater(len(mags), 0)
        for j in range(len(mags)):
            self.assertAlmostEqual(mags[j], test_mags[j], 10)


if __name__ == "__main__":
    unittest.main()
