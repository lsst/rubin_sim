import os
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir

import rubin_sim.phot_utils.signaltonoise as snr
from rubin_sim.phot_utils import Bandpass, PhotometricParameters, Sed, scale_sky_m5


class TestSNRmethods(unittest.TestCase):
    def setUp(self):
        star_name = os.path.join(get_data_dir(), "tests", "cartoonSedTestData", "starSed")
        star_name = os.path.join(star_name, "kurucz", "km20_5750.fits_g40_5790.gz")
        self.star_sed = Sed()
        self.star_sed.read_sed_flambda(star_name)
        imsimband = Bandpass()
        imsimband.imsim_bandpass()
        f_norm = self.star_sed.calc_flux_norm(22.0, imsimband)
        self.star_sed.multiply_flux_norm(f_norm)

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
        self.sky_sed.read_sed_flambda(os.path.join(hardware_dir, "darksky.dat"))

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

        self.seeing_defaults = {
            "u": 0.92,
            "g": 0.87,
            "r": 0.83,
            "i": 0.80,
            "z": 0.78,
            "y": 0.76,
        }

    def test_mag_error(self):
        """
        Make sure that calc_mag_error_sed and calc_mag_error_m5
        agree to within 0.001
        """
        phot_params = PhotometricParameters()

        # create a cartoon spectrum to test on
        spectrum = Sed()
        spectrum.set_flat_sed()
        spectrum.multiply_flux_norm(1.0e-9)

        # find the magnitudes of that spectrum in our bandpasses
        mag_list = []
        for total in self.bp_list:
            mag_list.append(spectrum.calc_mag(total))
        mag_list = np.array(mag_list)

        # try for different normalizations of the skySED
        for f_norm in np.arange(1.0, 5.0, 1.0):
            self.sky_sed.multiply_flux_norm(f_norm)

            for total, hardware, filterName, mm in zip(
                self.bp_list, self.hardware_list, self.filter_name_list, mag_list
            ):
                fwhm_eff = self.seeing_defaults[filterName]

                m5 = snr.calc_m5(self.sky_sed, total, hardware, phot_params, fwhm_eff=fwhm_eff)

                sigma_sed = snr.calc_mag_error_sed(
                    spectrum,
                    total,
                    self.sky_sed,
                    hardware,
                    phot_params,
                    fwhm_eff=fwhm_eff,
                )

                sigma_m5, gamma = snr.calc_mag_error_m5(mm, total, m5, phot_params)

                self.assertAlmostEqual(sigma_m5, sigma_sed, 3)

    def test_verbose_snr(self):
        """
        Make sure that calc_snr_sed has everything
        it needs to run in verbose mode
        """
        phot_params = PhotometricParameters()

        # create a cartoon spectrum to test on
        spectrum = Sed()
        spectrum.set_flat_sed()
        spectrum.multiply_flux_norm(1.0e-9)

        snr.calc_snr_sed(
            spectrum,
            self.bp_list[0],
            self.sky_sed,
            self.hardware_list[0],
            phot_params,
            fwhm_eff=0.7,
            verbose=True,
        )

    def test_signal_to_noise(self):
        """
        Test that calc_snr_m5 and calc_snr_sed give similar results
        """
        phot_params = PhotometricParameters()

        m5 = []
        for i in range(len(self.hardware_list)):
            m5.append(
                snr.calc_m5(
                    self.sky_sed,
                    self.bp_list[i],
                    self.hardware_list[i],
                    phot_params,
                    fwhm_eff=self.seeing_defaults[self.filter_name_list[i]],
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
            spectrum.read_sed_flambda(os.path.join(sed_dir, name))
            ff = spectrum.calc_flux_norm(m5[2] - offset[ix], self.bp_list[2])
            spectrum.multiply_flux_norm(ff)
            for i in range(len(self.bp_list)):
                control_snr = snr.calc_snr_sed(
                    spectrum,
                    self.bp_list[i],
                    self.sky_sed,
                    self.hardware_list[i],
                    phot_params,
                    self.seeing_defaults[self.filter_name_list[i]],
                )

                mag = spectrum.calc_mag(self.bp_list[i])

                test_snr, gamma = snr.calc_snr_m5(mag, self.bp_list[i], m5[i], phot_params)
                self.assertLess((test_snr - control_snr) / control_snr, 0.001)

    def test_systematic_uncertainty(self):
        """
        Test that systematic uncertainty is added correctly.
        """
        sigma_sys = 0.002
        m5_list = [23.5, 24.3, 22.1, 20.0, 19.5, 21.7]
        phot_params = PhotometricParameters(sigma_sys=sigma_sys)

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
            sky_dummy.read_sed_flambda(os.path.join(get_data_dir(), "throughputs", "baseline", "darksky.dat"))

            normalized_sky_dummy = scale_sky_m5(
                m5,
                sky_dummy,
                bp,
                hardware,
                fwhm_eff=self.seeing_defaults[filterName],
                phot_params=phot_params,
            )

            sigma, gamma = snr.calc_mag_error_m5(mm, bp, m5, phot_params)

            snrat = snr.calc_snr_sed(
                self.star_sed,
                bp,
                normalized_sky_dummy,
                hardware,
                fwhm_eff=self.seeing_defaults[filterName],
                phot_params=PhotometricParameters(),
            )

            test_snr, gamma = snr.calc_snr_m5(mm, bp, m5, phot_params=PhotometricParameters(sigma_sys=0.0))

            self.assertAlmostEqual(
                snrat,
                test_snr,
                10,
                msg="failed on calc_snr_m5 test %e != %e " % (snrat, test_snr),
            )

            control = np.sqrt(np.power(snr.mag_error_from_snr(test_snr), 2) + np.power(sigma_sys, 2))

            msg = "%e is not %e; failed" % (sigma, control)

            self.assertAlmostEqual(sigma, control, 10, msg=msg)

    def test_no_systematic_uncertainty(self):
        """
        Test that systematic uncertainty is handled correctly
        when set to None.
        """
        m5_list = [23.5, 24.3, 22.1, 20.0, 19.5, 21.7]
        phot_params = PhotometricParameters(sigma_sys=0.0)

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
            sky_dummy.read_sed_flambda(os.path.join(get_data_dir(), "throughputs", "baseline", "darksky.dat"))

            normalized_sky_dummy = scale_sky_m5(
                m5,
                sky_dummy,
                bp,
                hardware,
                fwhm_eff=self.seeing_defaults[filterName],
                phot_params=phot_params,
            )

            sigma, gamma = snr.calc_mag_error_m5(mm, bp, m5, phot_params)

            snrat = snr.calc_snr_sed(
                self.star_sed,
                bp,
                normalized_sky_dummy,
                hardware,
                fwhm_eff=self.seeing_defaults[filterName],
                phot_params=PhotometricParameters(),
            )

            test_snr, gamma = snr.calc_snr_m5(mm, bp, m5, phot_params=PhotometricParameters(sigma_sys=0.0))

            self.assertAlmostEqual(
                snrat,
                test_snr,
                10,
                msg="failed on calc_snr_m5 test %e != %e " % (snrat, test_snr),
            )

            control = snr.mag_error_from_snr(test_snr)

            msg = "%e is not %e; failed" % (sigma, control)

            self.assertAlmostEqual(sigma, control, 10, msg=msg)

    def test_fwh_mconversions(self):
        fwhm_eff = 0.8
        fwh_mgeom = snr.fwhm_eff2_fwhm_geom(fwhm_eff)
        self.assertEqual(fwh_mgeom, (0.822 * fwhm_eff + 0.052))
        fwh_mgeom = 0.8
        fwhm_eff = snr.fwhm_geom2_fwhm_eff(fwh_mgeom)
        self.assertEqual(fwhm_eff, (fwh_mgeom - 0.052) / 0.822)

    def test_astrometric_error(self):
        fwhm_geom = 0.7
        m5 = 24.5
        # For bright objects, error should be systematic floor
        mag = 10
        astrometric_err = snr.calc_astrometric_error(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=1, systematic_floor=10
        )
        self.assertAlmostEqual(astrometric_err, 10, 3)
        # Even if you increase the number of visits,
        # the systemic floor doesn't change
        astrometric_err = snr.calc_astrometric_error(mag, m5, fwhm_geom=fwhm_geom, nvisit=100)
        self.assertAlmostEqual(astrometric_err, 10, 3)
        # For a single visit, fainter source, larger error and nvisits matters
        mag = 24.5
        astrometric_err1 = snr.calc_astrometric_error(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=1, systematic_floor=10
        )
        astrometric_err100 = snr.calc_astrometric_error(
            mag, m5, fwhm_geom=fwhm_geom, nvisit=100, systematic_floor=10
        )
        self.assertGreater(astrometric_err1, astrometric_err100)
        self.assertAlmostEqual(astrometric_err1, 140.357, 3)

    def test_snr_arr(self):
        """
        Test that calc_snr_m5 works on numpy arrays of magnitudes
        """
        rng = np.random.RandomState(17)
        mag_list = rng.random_sample(100) * 5.0 + 15.0

        phot_params = PhotometricParameters()
        bp = self.bp_list[0]
        m5 = 24.0
        control_list = []
        for mm in mag_list:
            ratio, gamma = snr.calc_snr_m5(mm, bp, m5, phot_params)
            control_list.append(ratio)
        control_list = np.array(control_list)

        test_list, gamma = snr.calc_snr_m5(mag_list, bp, m5, phot_params)

        np.testing.assert_array_equal(control_list, test_list)

    def test_error_arr(self):
        """
        Test that calc_mag_error_m5 works on numpy arrays of magnitudes
        """
        rng = np.random.RandomState(17)
        mag_list = rng.random_sample(100) * 5.0 + 15.0

        phot_params = PhotometricParameters()
        bp = self.bp_list[0]
        m5 = 24.0
        control_list = []
        for mm in mag_list:
            sig, gamma = snr.calc_mag_error_m5(mm, bp, m5, phot_params)
            control_list.append(sig)
        control_list = np.array(control_list)

        test_list, gamma = snr.calc_mag_error_m5(mag_list, bp, m5, phot_params)

        np.testing.assert_array_equal(control_list, test_list)


if __name__ == "__main__":
    unittest.main()
