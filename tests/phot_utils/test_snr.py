import os
import numpy as np
import unittest
from rubin_sim.utils import ObservationMetaData
import rubin_sim.phot_utils.signaltonoise as snr
from rubin_sim.phot_utils import Sed, Bandpass, PhotometricParameters, LSSTdefaults
from rubin_sim.phot_utils.utils import setM5
import rubin_sim
from rubin_sim.data import get_data_dir


class TestSNRmethods(unittest.TestCase):
    def setUp(self):

        star_name = os.path.join(
            get_data_dir(), "tests", "cartoonSedTestData", "starSed"
        )
        star_name = os.path.join(star_name, "kurucz", "km20_5750.fits_g40_5790.gz")
        self.star_sed = Sed()
        self.star_sed.readSED_flambda(star_name)
        imsimband = Bandpass()
        imsimband.imsimBandpass()
        f_norm = self.star_sed.calc_fluxNorm(22.0, imsimband)
        self.star_sed.multiplyFluxNorm(f_norm)

        hardware_dir = os.path.join(get_data_dir(), "throughputs", "baseline")
        component_list = [
            "detector.dat",
            "m1.dat",
            "m2.dat",
            "m3.dat",
            "lens1.dat",
            "lens2.dat",
            "lens3.dat",
        ]
        self.sky_sed = Sed()
        self.sky_sed.readSED_flambda(os.path.join(hardware_dir, "darksky.dat"))

        total_name_list = [
            "total_u.dat",
            "total_g.dat",
            "total_r.dat",
            "total_i.dat",
            "total_z.dat",
            "total_y.dat",
        ]

        self.bp_list = []
        self.hardware_list = []
        for name in total_name_list:
            dummy = Bandpass()
            dummy.read_throughput(os.path.join(hardware_dir, name))
            self.bp_list.append(dummy)

            dummy = Bandpass()
            hardware_name_list = [os.path.join(hardware_dir, name)]
            for component in component_list:
                hardware_name_list.append(os.path.join(hardware_dir, component))
            dummy.read_throughput_list(hardware_name_list)
            self.hardware_list.append(dummy)

        self.filter_name_list = ["u", "g", "r", "i", "z", "y"]

    def test_mag_error(self):
        """
        Make sure that calc_magError_sed and calc_magError_m5
        agree to within 0.001
        """
        defaults = LSSTdefaults()
        phot_params = PhotometricParameters()

        # create a cartoon spectrum to test on
        spectrum = Sed()
        spectrum.setFlatSED()
        spectrum.multiplyFluxNorm(1.0e-9)

        # find the magnitudes of that spectrum in our bandpasses
        mag_list = []
        for total in self.bp_list:
            mag_list.append(spectrum.calc_mag(total))
        mag_list = np.array(mag_list)

        # try for different normalizations of the skySED
        for f_norm in np.arange(1.0, 5.0, 1.0):
            self.sky_sed.multiplyFluxNorm(f_norm)

            for total, hardware, filterName, mm in zip(
                self.bp_list, self.hardware_list, self.filter_name_list, mag_list
            ):

                fwhm_eff = defaults.FWHMeff(filterName)

                m5 = snr.calcM5(
                    self.sky_sed, total, hardware, phot_params, fwhm_eff=fwhm_eff
                )

                sigma_sed = snr.calc_magError_sed(
                    spectrum,
                    total,
                    self.sky_sed,
                    hardware,
                    phot_params,
                    fwhm_eff=fwhm_eff,
                )

                sigma_m5, gamma = snr.calc_magError_m5(mm, total, m5, phot_params)

                self.assertAlmostEqual(sigma_m5, sigma_sed, 3)

    def test_verbose_snr(self):
        """
        Make sure that calcSNR_sed has everything it needs to run in verbose mode
        """
        phot_params = PhotometricParameters()

        # create a cartoon spectrum to test on
        spectrum = Sed()
        spectrum.setFlatSED()
        spectrum.multiplyFluxNorm(1.0e-9)

        snr.calcSNR_sed(
            spectrum,
            self.bp_list[0],
            self.sky_sed,
            self.hardware_list[0],
            phot_params,
            FWHMeff=0.7,
            verbose=True,
        )

    def test_signal_to_noise(self):
        """
        Test that calcSNR_m5 and calcSNR_sed give similar results
        """
        defaults = LSSTdefaults()
        phot_params = PhotometricParameters()

        m5 = []
        for i in range(len(self.hardware_list)):
            m5.append(
                snr.calcM5(
                    self.sky_sed,
                    self.bp_list[i],
                    self.hardware_list[i],
                    phot_params,
                    FWHMeff=defaults.FWHMeff(self.filter_name_list[i]),
                )
            )

        sed_dir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData/starSed/")
        sed_dir = os.path.join(sed_dir, "kurucz")
        file_name_list = os.listdir(sed_dir)

        rng = np.random.RandomState(42)
        offset = rng.random_sample(len(file_name_list)) * 2.0

        for ix, name in enumerate(file_name_list):
            if ix > 100:
                break
            spectrum = Sed()
            spectrum.readSED_flambda(os.path.join(sed_dir, name))
            ff = spectrum.calc_fluxNorm(m5[2] - offset[ix], self.bp_list[2])
            spectrum.multiplyFluxNorm(ff)
            for i in range(len(self.bp_list)):
                control_snr = snr.calcSNR_sed(
                    spectrum,
                    self.bp_list[i],
                    self.sky_sed,
                    self.hardware_list[i],
                    phot_params,
                    defaults.FWHMeff(self.filter_name_list[i]),
                )

                mag = spectrum.calc_mag(self.bp_list[i])

                test_snr, gamma = snr.calcSNR_m5(
                    mag, self.bp_list[i], m5[i], phot_params
                )
                self.assertLess((test_snr - control_snr) / control_snr, 0.001)

    def test_systematic_uncertainty(self):
        """
        Test that systematic uncertainty is added correctly.
        """
        sigma_sys = 0.002
        m5_list = [23.5, 24.3, 22.1, 20.0, 19.5, 21.7]
        phot_params = PhotometricParameters(sigmaSys=sigma_sys)

        obs_metadata = ObservationMetaData(
            pointing_ra=23.0,
            pointing_dec=45.0,
            m5=m5_list,
            bandpass_name=self.filter_name_list,
        )
        magnitude_list = []
        for bp in self.bp_list:
            mag = self.star_sed.calc_mag(bp)
            magnitude_list.append(mag)

        for bp, hardware, filterName, mm, m5 in zip(
            self.bp_list,
            self.hardware_list,
            self.filter_name_list,
            magnitude_list,
            m5_list,
        ):

            sky_dummy = Sed()
            sky_dummy.readSED_flambda(
                os.path.join(get_data_dir(), "throughputs", "baseline", "darksky.dat")
            )

            normalized_sky_dummy = setM5(
                obs_metadata.m5[filterName],
                sky_dummy,
                bp,
                hardware,
                FWHMeff=LSSTdefaults().FWHMeff(filterName),
                phot_params=phot_params,
            )

            sigma, gamma = snr.calc_magError_m5(mm, bp, m5, phot_params)

            snrat = snr.calcSNR_sed(
                self.star_sed,
                bp,
                normalized_sky_dummy,
                hardware,
                FWHMeff=LSSTdefaults().FWHMeff(filterName),
                phot_params=PhotometricParameters(),
            )

            test_snr, gamma = snr.calcSNR_m5(
                mm, bp, m5, phot_params=PhotometricParameters(sigmaSys=0.0)
            )

            self.assertAlmostEqual(
                snrat,
                test_snr,
                10,
                msg="failed on calcSNR_m5 test %e != %e " % (snrat, test_snr),
            )

            control = np.sqrt(
                np.power(snr.magErrorFromSNR(test_snr), 2) + np.power(sigma_sys, 2)
            )

            msg = "%e is not %e; failed" % (sigma, control)

            self.assertAlmostEqual(sigma, control, 10, msg=msg)

    def test_no_systematic_uncertainty(self):
        """
        Test that systematic uncertainty is handled correctly when set to None.
        """
        m5_list = [23.5, 24.3, 22.1, 20.0, 19.5, 21.7]
        phot_params = PhotometricParameters(sigmaSys=0.0)

        obs_metadata = ObservationMetaData(
            pointing_ra=23.0,
            pointing_dec=45.0,
            m5=m5_list,
            bandpass_name=self.filter_name_list,
        )

        magnitude_list = []
        for bp in self.bp_list:
            mag = self.star_sed.calc_mag(bp)
            magnitude_list.append(mag)

        for bp, hardware, filterName, mm, m5 in zip(
            self.bp_list,
            self.hardware_list,
            self.filter_name_list,
            magnitude_list,
            m5_list,
        ):

            sky_dummy = Sed()
            sky_dummy.readSED_flambda(
                os.path.join(get_data_dir(), "throughputs", "baseline", "darksky.dat")
            )

            normalized_sky_dummy = setM5(
                obs_metadata.m5[filterName],
                sky_dummy,
                bp,
                hardware,
                FWHMeff=LSSTdefaults().FWHMeff(filterName),
                phot_params=phot_params,
            )

            sigma, gamma = snr.calc_magError_m5(mm, bp, m5, phot_params)

            snrat = snr.calcSNR_sed(
                self.star_sed,
                bp,
                normalized_sky_dummy,
                hardware,
                FWHMeff=LSSTdefaults().FWHMeff(filterName),
                phot_params=PhotometricParameters(),
            )

            test_snr, gamma = snr.calcSNR_m5(
                mm, bp, m5, phot_params=PhotometricParameters(sigmaSys=0.0)
            )

            self.assertAlmostEqual(
                snrat,
                test_snr,
                10,
                msg="failed on calcSNR_m5 test %e != %e " % (snrat, test_snr),
            )

            control = snr.magErrorFromSNR(test_snr)

            msg = "%e is not %e; failed" % (sigma, control)

            self.assertAlmostEqual(sigma, control, 10, msg=msg)

    def test_fwh_mconversions(self):
        fwhm_eff = 0.8
        fwh_mgeom = snr.FWHMeff2FWHMgeom(fwhm_eff)
        self.assertEqual(fwh_mgeom, (0.822 * fwhm_eff + 0.052))
        fwh_mgeom = 0.8
        fwhm_eff = snr.FWHMgeom2FWHMeff(fwh_mgeom)
        self.assertEqual(fwhm_eff, (fwh_mgeom - 0.052) / 0.822)

    def test_astrometric_error(self):
        fwhm_geom = 0.7
        m5 = 24.5
        # For bright objects, error should be systematic floor
        mag = 10
        astrometric_err = snr.calcAstrometricError(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=1, systematicFloor=10
        )
        self.assertAlmostEqual(astrometric_err, 10, 3)
        # Even if you increase the number of visits, the systemic floor doesn't change
        astrometric_err = snr.calcAstrometricError(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=100
        )
        self.assertAlmostEqual(astrometric_err, 10, 3)
        # For a single visit, fainter source, larger error and nvisits matters
        mag = 24.5
        astrometric_err1 = snr.calcAstrometricError(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=1, systematicFloor=10
        )
        astrometric_err100 = snr.calcAstrometricError(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=100, systematicFloor=10
        )
        self.assertGreater(astrometric_err1, astrometric_err100)
        self.assertAlmostEqual(astrometric_err1, 140.357, 3)

    def test_snr_arr(self):
        """
        Test that calcSNR_m5 works on numpy arrays of magnitudes
        """
        rng = np.random.RandomState(17)
        mag_list = rng.random_sample(100) * 5.0 + 15.0

        phot_params = PhotometricParameters()
        bp = self.bp_list[0]
        m5 = 24.0
        control_list = []
        for mm in mag_list:
            ratio, gamma = snr.calcSNR_m5(mm, bp, m5, phot_params)
            control_list.append(ratio)
        control_list = np.array(control_list)

        test_list, gamma = snr.calcSNR_m5(mag_list, bp, m5, phot_params)

        np.testing.assert_array_equal(control_list, test_list)

    def test_error_arr(self):
        """
        Test that calc_magError_m5 works on numpy arrays of magnitudes
        """
        rng = np.random.RandomState(17)
        mag_list = rng.random_sample(100) * 5.0 + 15.0

        phot_params = PhotometricParameters()
        bp = self.bp_list[0]
        m5 = 24.0
        control_list = []
        for mm in mag_list:
            sig, gamma = snr.calc_magError_m5(mm, bp, m5, phot_params)
            control_list.append(sig)
        control_list = np.array(control_list)

        test_list, gamma = snr.calc_magError_m5(mag_list, bp, m5, phot_params)

        np.testing.assert_array_equal(control_list, test_list)


if __name__ == "__main__":
    unittest.main()
