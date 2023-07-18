import copy
import os
import unittest

import numpy as np

import rubin_sim
from rubin_sim.data import get_data_dir
from rubin_sim.phot_utils import Bandpass, BandpassDict, Sed, SedList


class BandpassDictTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(32)
        self.bandpass_possibilities = ["u", "g", "r", "i", "z", "y"]
        self.bandpass_dir = os.path.join(get_data_dir(), "throughputs", "baseline")
        self.sed_dir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData/galaxySed")
        self.sed_possibilities = os.listdir(self.sed_dir)

    def get_list_of_sed_names(self, n_names):
        return [
            self.sed_possibilities[ii].replace(".gz", "")
            for ii in self.rng.randint(0, len(self.sed_possibilities) - 1, n_names)
        ]

    def get_list_of_bandpasses(self, n_bp):
        """
        Generate a list of n_bp bandpass names and bandpasses

        Intentionally do so a nonsense order so that we can test
        that order is preserved in the BandpassDict
        """
        dex_list = self.rng.randint(0, len(self.bandpass_possibilities) - 1, n_bp)
        bandpass_name_list = []
        bandpass_list = []
        for dex in dex_list:
            name = self.bandpass_possibilities[dex]
            bp = Bandpass()
            bp.read_throughput(os.path.join(self.bandpass_dir, "total_%s.dat" % name))
            while name in bandpass_name_list:
                name += "0"
            bandpass_name_list.append(name)
            bandpass_list.append(bp)

        return bandpass_name_list, bandpass_list

    def test_initialization(self):
        """
        Test that all of the member variables of BandpassDict are set
        to the correct value upon construction.
        """

        for n_bp in range(3, 10, 1):
            name_list, bp_list = self.get_list_of_bandpasses(n_bp)
            test_dict = BandpassDict(bp_list, name_list)

            self.assertEqual(len(test_dict), n_bp)

            for control_name, testName in zip(name_list, test_dict):
                self.assertEqual(control_name, testName)

            for control_name, testName in zip(name_list, test_dict.keys()):
                self.assertEqual(control_name, testName)

            for name, bp in zip(name_list, bp_list):
                np.testing.assert_array_almost_equal(bp.wavelen, test_dict[name].wavelen, 10)
                np.testing.assert_array_almost_equal(bp.sb, test_dict[name].sb, 10)

            for bp_control, bpTest in zip(bp_list, test_dict.values()):
                np.testing.assert_array_almost_equal(bp_control.wavelen, bpTest.wavelen, 10)
                np.testing.assert_array_almost_equal(bp_control.sb, bpTest.sb, 10)

    def test_wavelen_match(self):
        """
        Test that when you load bandpasses sampled over different
        wavelength grids, they all get sampled to the same wavelength
        grid.
        """
        dwav_list = np.arange(5.0, 25.0, 5.0)
        bp_list = []
        bp_name_list = []
        for ix, dwav in enumerate(dwav_list):
            name = "bp_%d" % ix
            wavelen = np.arange(10.0, 1500.0, dwav)
            sb = np.exp(-0.5 * (np.power((wavelen - 100.0 * ix) / 100.0, 2)))
            bp = Bandpass(wavelen=wavelen, sb=sb)
            bp_list.append(bp)
            bp_name_list.append(name)

        # First make sure that we have created distinct wavelength grids
        for ix in range(len(bp_list)):
            for iy in range(ix + 1, len(bp_list)):
                self.assertNotEqual(len(bp_list[ix].wavelen), len(bp_list[iy].wavelen))

        test_dict = BandpassDict(bp_list, bp_name_list)

        # Now make sure that the wavelength grids in the dict were resampled, but that
        # the original wavelength grids were not changed
        for ix in range(len(bp_list)):
            np.testing.assert_array_almost_equal(test_dict.values()[ix].wavelen, test_dict.wavelen_match, 19)
            if ix != 0:
                self.assertNotEqual(len(test_dict.wavelen_match), len(bp_list[ix].wavelen))

    def test_phi_array(self):
        """
        Test that the phi array is correctly calculated by BandpassDict
        upon construction.
        """

        for n_bp in range(3, 10, 1):
            name_list, bp_list = self.get_list_of_bandpasses(n_bp)
            test_dict = BandpassDict(bp_list, name_list)
            dummy_sed = Sed()
            control_phi, control_wavelen_step = dummy_sed.setup_phi_array(bp_list)
            np.testing.assert_array_almost_equal(control_phi, test_dict.phi_array, 19)
            self.assertAlmostEqual(control_wavelen_step, test_dict.wavelen_step, 10)

    def test_exceptions(self):
        """
        Test that the correct exceptions are thrown by BandpassDict
        """

        name_list, bp_list = self.get_list_of_bandpasses(4)
        dummy_name_list = copy.deepcopy(name_list)
        dummy_name_list[1] = dummy_name_list[0]

        with self.assertRaises(RuntimeError) as context:
            test_dict = BandpassDict(bp_list, dummy_name_list)

        self.assertIn("occurs twice", context.exception.args[0])

        test_dict = BandpassDict(bp_list, name_list)

        with self.assertRaises(AttributeError) as context:
            test_dict.phi_array = None

        with self.assertRaises(AttributeError) as context:
            test_dict.wavelen_step = 0.9

        with self.assertRaises(AttributeError) as context:
            test_dict.wavelen_match = np.arange(10.0, 100.0, 1.0)

    def test_mag_list_for_sed(self):
        """
        Test that mag_list_for_sed calculates the correct magnitude
        """

        wavelen = np.arange(10.0, 2000.0, 1.0)
        flux = (wavelen * 2.0 - 5.0) * 1.0e-6
        spectrum = Sed(wavelen=wavelen, flambda=flux)

        for n_bp in range(3, 10, 1):
            name_list, bp_list = self.get_list_of_bandpasses(n_bp)
            test_dict = BandpassDict(bp_list, name_list)
            self.assertNotEqual(len(test_dict.values()[0].wavelen), len(spectrum.wavelen))

            mag_list = test_dict.mag_list_for_sed(spectrum)
            for ix, (name, bp, magTest) in enumerate(zip(name_list, bp_list, mag_list)):
                mag_control = spectrum.calc_mag(bp)
                self.assertAlmostEqual(magTest, mag_control, 5)

    def test_mag_dict_for_sed(self):
        """
        Test that mag_dict_for_sed calculates the correct magnitude
        """

        wavelen = np.arange(10.0, 2000.0, 1.0)
        flux = (wavelen * 2.0 - 5.0) * 1.0e-6
        spectrum = Sed(wavelen=wavelen, flambda=flux)

        for n_bp in range(3, 10, 1):
            name_list, bp_list = self.get_list_of_bandpasses(n_bp)
            test_dict = BandpassDict(bp_list, name_list)
            self.assertNotEqual(len(test_dict.values()[0].wavelen), len(spectrum.wavelen))

            mag_dict = test_dict.mag_dict_for_sed(spectrum)
            for ix, (name, bp) in enumerate(zip(name_list, bp_list)):
                mag_control = spectrum.calc_mag(bp)
                self.assertAlmostEqual(mag_dict[name], mag_control, 5)

    def test_mag_list_for_sed_list(self):
        """
        Test that mag_list_for_sed_list calculates the correct magnitude
        """

        n_bandpasses = 7
        bp_name_list, bp_list = self.get_list_of_bandpasses(n_bandpasses)
        test_bp_dict = BandpassDict(bp_list, bp_name_list)

        n_sed = 20
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1

        # first, test on an SedList without a wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )

        mag_list = test_bp_dict.mag_list_for_sed_list(test_sed_list)
        self.assertEqual(mag_list.shape[0], n_sed)
        self.assertEqual(mag_list.shape[1], n_bandpasses)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(test_bp_dict):
                mag = dummy_sed.calc_mag(bp_list[iy])
                self.assertAlmostEqual(mag, mag_list[ix][iy], 2)

        # now use wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=test_bp_dict.wavelen_match,
        )

        mag_list = test_bp_dict.mag_list_for_sed_list(test_sed_list)
        self.assertEqual(mag_list.shape[0], n_sed)
        self.assertEqual(mag_list.shape[1], n_bandpasses)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(test_bp_dict):
                mag = dummy_sed.calc_mag(bp_list[iy])
                self.assertAlmostEqual(mag, mag_list[ix][iy], 2)

    def test_mag_array_for_sed_list(self):
        """
        Test that mag_array_for_sed_list calculates the correct magnitude
        """

        n_bandpasses = 7
        bp_name_list, bp_list = self.get_list_of_bandpasses(n_bandpasses)
        test_bp_dict = BandpassDict(bp_list, bp_name_list)

        n_sed = 20
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1

        # first, test on an SedList without a wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )

        mag_array = test_bp_dict.mag_array_for_sed_list(test_sed_list)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(bp_name_list):
                mag = dummy_sed.calc_mag(bp_list[iy])
                self.assertAlmostEqual(mag, mag_array[bp][ix], 2)

        # now use wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=test_bp_dict.wavelen_match,
        )

        mag_array = test_bp_dict.mag_array_for_sed_list(test_sed_list)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(bp_name_list):
                mag = dummy_sed.calc_mag(bp_list[iy])
                self.assertAlmostEqual(mag, mag_array[bp][ix], 2)

    def test_indices_on_magnitudes(self):
        """
        Test that, when you pass a list of indices into the calc_magList
        methods, you get the correct magnitudes out.
        """

        n_bandpasses = 7
        name_list, bp_list = self.get_list_of_bandpasses(n_bandpasses)
        test_bp_dict = BandpassDict(bp_list, name_list)

        # first try it with a single Sed
        wavelen = np.arange(10.0, 2000.0, 1.0)
        flux = (wavelen * 2.0 - 5.0) * 1.0e-6
        spectrum = Sed(wavelen=wavelen, flambda=flux)
        indices = [1, 2, 5]

        mag_list = test_bp_dict.mag_list_for_sed(spectrum, indices=indices)
        ct_na_n = 0
        for ix, (name, bp, magTest) in enumerate(zip(name_list, bp_list, mag_list)):
            if ix in indices:
                mag_control = spectrum.calc_mag(bp)
                self.assertAlmostEqual(magTest, mag_control, 5)
            else:
                ct_na_n += 1
                np.testing.assert_equal(magTest, np.NaN)

        self.assertEqual(ct_na_n, 4)

        n_sed = 20
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1

        # now try a SedList without a wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )

        mag_list = test_bp_dict.mag_list_for_sed_list(test_sed_list, indices=indices)
        mag_array = test_bp_dict.mag_array_for_sed_list(test_sed_list, indices=indices)
        self.assertEqual(mag_list.shape[0], n_sed)
        self.assertEqual(mag_list.shape[1], n_bandpasses)
        self.assertEqual(mag_array.shape[0], n_sed)
        for bpname in test_bp_dict:
            self.assertEqual(len(mag_array[bpname]), n_sed)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            ct_na_n = 0
            for iy, bp in enumerate(test_bp_dict):
                if iy in indices:
                    mag = dummy_sed.calc_mag(test_bp_dict[bp])
                    self.assertAlmostEqual(mag, mag_list[ix][iy], 2)
                    self.assertAlmostEqual(mag, mag_array[ix][iy], 2)
                    self.assertAlmostEqual(mag, mag_array[bp][ix], 2)
                else:
                    ct_na_n += 1
                    np.testing.assert_equal(mag_list[ix][iy], np.NaN)
                    np.testing.assert_equal(mag_array[ix][iy], np.NaN)
                    np.testing.assert_equal(mag_array[bp][ix], np.NaN)

            self.assertEqual(ct_na_n, 4)

        # now use wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=test_bp_dict.wavelen_match,
        )

        mag_list = test_bp_dict.mag_list_for_sed_list(test_sed_list, indices=indices)
        mag_array = test_bp_dict.mag_array_for_sed_list(test_sed_list, indices=indices)
        self.assertEqual(mag_list.shape[0], n_sed)
        self.assertEqual(mag_list.shape[1], n_bandpasses)
        self.assertEqual(mag_array.shape[0], n_sed)
        for bpname in test_bp_dict:
            self.assertEqual(len(mag_array[bpname]), n_sed)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            ct_na_n = 0
            for iy, bp in enumerate(test_bp_dict):
                if iy in indices:
                    mag = dummy_sed.calc_mag(test_bp_dict[bp])
                    self.assertAlmostEqual(mag, mag_list[ix][iy], 2)
                    self.assertAlmostEqual(mag, mag_array[ix][iy], 2)
                    self.assertAlmostEqual(mag, mag_array[bp][ix], 2)
                else:
                    ct_na_n += 1
                    np.testing.assert_equal(mag_list[ix][iy], np.NaN)
                    np.testing.assert_equal(mag_array[ix][iy], np.NaN)
                    np.testing.assert_equal(mag_array[bp][ix], np.NaN)

            self.assertEqual(ct_na_n, 4)

    def test_flux_list_for_sed(self):
        """
        Test that flux_list_for_sed calculates the correct fluxes
        """

        wavelen = np.arange(10.0, 2000.0, 1.0)
        flux = (wavelen * 2.0 - 5.0) * 1.0e-6
        spectrum = Sed(wavelen=wavelen, flambda=flux)

        for n_bp in range(3, 10, 1):
            name_list, bp_list = self.get_list_of_bandpasses(n_bp)
            test_dict = BandpassDict(bp_list, name_list)
            self.assertNotEqual(len(test_dict.values()[0].wavelen), len(spectrum.wavelen))

            flux_list = test_dict.flux_list_for_sed(spectrum)
            for ix, (name, bp, fluxTest) in enumerate(zip(name_list, bp_list, flux_list)):
                flux_control = spectrum.calc_flux(bp)
                self.assertAlmostEqual(fluxTest / flux_control, 1.0, 2)

    def test_flux_dict_for_sed(self):
        """
        Test that flux_dict_for_sed calculates the correct fluxes
        """

        wavelen = np.arange(10.0, 2000.0, 1.0)
        flux = (wavelen * 2.0 - 5.0) * 1.0e-6
        spectrum = Sed(wavelen=wavelen, flambda=flux)

        for n_bp in range(3, 10, 1):
            name_list, bp_list = self.get_list_of_bandpasses(n_bp)
            test_dict = BandpassDict(bp_list, name_list)
            self.assertNotEqual(len(test_dict.values()[0].wavelen), len(spectrum.wavelen))

            flux_dict = test_dict.flux_dict_for_sed(spectrum)
            for ix, (name, bp) in enumerate(zip(name_list, bp_list)):
                flux_control = spectrum.calc_flux(bp)
                self.assertAlmostEqual(flux_dict[name] / flux_control, 1.0, 2)

    def test_flux_list_for_sed_list(self):
        """
        Test that flux_list_for_sed_list calculates the correct fluxes
        """

        n_bandpasses = 7
        bp_name_list, bp_list = self.get_list_of_bandpasses(n_bandpasses)
        test_bp_dict = BandpassDict(bp_list, bp_name_list)

        n_sed = 20
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1

        # first, test on an SedList without a wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )

        flux_list = test_bp_dict.flux_list_for_sed_list(test_sed_list)
        self.assertEqual(flux_list.shape[0], n_sed)
        self.assertEqual(flux_list.shape[1], n_bandpasses)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(test_bp_dict):
                flux = dummy_sed.calc_flux(bp_list[iy])
                self.assertAlmostEqual(flux / flux_list[ix][iy], 1.0, 2)

        # now use wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=test_bp_dict.wavelen_match,
        )

        flux_list = test_bp_dict.flux_list_for_sed_list(test_sed_list)
        self.assertEqual(flux_list.shape[0], n_sed)
        self.assertEqual(flux_list.shape[1], n_bandpasses)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(test_bp_dict):
                flux = dummy_sed.calc_flux(bp_list[iy])
                self.assertAlmostEqual(flux / flux_list[ix][iy], 1.0, 2)

    def test_flux_array_for_sed_list(self):
        """
        Test that flux_array_for_sed_list calculates the correct fluxes
        """

        n_bandpasses = 7
        bp_name_list, bp_list = self.get_list_of_bandpasses(n_bandpasses)
        test_bp_dict = BandpassDict(bp_list, bp_name_list)

        n_sed = 20
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1

        # first, test on an SedList without a wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )

        flux_array = test_bp_dict.flux_array_for_sed_list(test_sed_list)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(bp_name_list):
                flux = dummy_sed.calc_flux(bp_list[iy])
                self.assertAlmostEqual(flux / flux_array[bp][ix], 1.0, 2)

        # now use wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=test_bp_dict.wavelen_match,
        )

        flux_array = test_bp_dict.flux_array_for_sed_list(test_sed_list)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            for iy, bp in enumerate(bp_name_list):
                flux = dummy_sed.calc_flux(bp_list[iy])
                self.assertAlmostEqual(flux / flux_array[bp][ix], 1.0, 2)

    def test_indices_on_flux(self):
        """
        Test that, when you pass a list of indices into the calc_flux_list
        methods, you get the correct fluxes out.
        """

        n_bandpasses = 7
        name_list, bp_list = self.get_list_of_bandpasses(n_bandpasses)
        test_bp_dict = BandpassDict(bp_list, name_list)

        # first try it with a single Sed
        wavelen = np.arange(10.0, 2000.0, 1.0)
        flux = (wavelen * 2.0 - 5.0) * 1.0e-6
        spectrum = Sed(wavelen=wavelen, flambda=flux)
        indices = [1, 2, 5]

        flux_list = test_bp_dict.flux_list_for_sed(spectrum, indices=indices)
        ct_na_n = 0
        for ix, (name, bp, fluxTest) in enumerate(zip(name_list, bp_list, flux_list)):
            if ix in indices:
                flux_control = spectrum.calc_flux(bp)
                self.assertAlmostEqual(fluxTest / flux_control, 1.0, 2)
            else:
                ct_na_n += 1
                np.testing.assert_equal(fluxTest, np.NaN)

        self.assertEqual(ct_na_n, 4)

        n_sed = 20
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1

        # now try a SedList without a wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )

        flux_list = test_bp_dict.flux_list_for_sed_list(test_sed_list, indices=indices)
        flux_array = test_bp_dict.flux_array_for_sed_list(test_sed_list, indices=indices)
        self.assertEqual(flux_list.shape[0], n_sed)
        self.assertEqual(flux_list.shape[1], n_bandpasses)
        self.assertEqual(flux_array.shape[0], n_sed)
        for bpname in test_bp_dict:
            self.assertEqual(len(flux_array[bpname]), n_sed)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            ct_na_n = 0
            for iy, bp in enumerate(test_bp_dict):
                if iy in indices:
                    flux = dummy_sed.calc_flux(test_bp_dict[bp])
                    self.assertAlmostEqual(flux / flux_list[ix][iy], 1.0, 2)
                    self.assertAlmostEqual(flux / flux_array[ix][iy], 1.0, 2)
                    self.assertAlmostEqual(flux / flux_array[bp][ix], 1.0, 2)
                else:
                    ct_na_n += 1
                    np.testing.assert_equal(flux_list[ix][iy], np.NaN)
                    np.testing.assert_equal(flux_array[ix][iy], np.NaN)
                    np.testing.assert_equal(flux_array[bp][ix], np.NaN)

            self.assertEqual(ct_na_n, 4)

        # now use wavelen_match
        test_sed_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=test_bp_dict.wavelen_match,
        )

        flux_list = test_bp_dict.flux_list_for_sed_list(test_sed_list, indices=indices)
        flux_array = test_bp_dict.flux_array_for_sed_list(test_sed_list, indices=indices)
        self.assertEqual(flux_list.shape[0], n_sed)
        self.assertEqual(flux_list.shape[1], n_bandpasses)
        self.assertEqual(flux_array.shape[0], n_sed)
        for bpname in test_bp_dict:
            self.assertEqual(len(flux_array[bpname]), n_sed)

        for ix, sedObj in enumerate(test_sed_list):
            dummy_sed = Sed(
                wavelen=copy.deepcopy(sedObj.wavelen),
                flambda=copy.deepcopy(sedObj.flambda),
            )

            ct_na_n = 0
            for iy, bp in enumerate(test_bp_dict):
                if iy in indices:
                    flux = dummy_sed.calc_flux(test_bp_dict[bp])
                    self.assertAlmostEqual(flux / flux_list[ix][iy], 1.0, 2)
                    self.assertAlmostEqual(flux / flux_array[ix][iy], 1.0, 2)
                    self.assertAlmostEqual(flux / flux_array[bp][ix], 1.0, 2)
                else:
                    ct_na_n += 1
                    np.testing.assert_equal(flux_list[ix][iy], np.NaN)
                    np.testing.assert_equal(flux_array[ix][iy], np.NaN)
                    np.testing.assert_equal(flux_array[bp][ix], np.NaN)

            self.assertEqual(ct_na_n, 4)

    def test_load_total_bandpasses_from_files(self):
        """
        Test that the class method load_total_bandpasses_from_files produces the
        expected result
        """

        bandpass_dir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData")
        bandpass_names = ["g", "r", "u"]
        bandpass_root = "test_bandpass_"

        bandpass_dict = BandpassDict.load_total_bandpasses_from_files(
            bandpass_names=bandpass_names,
            bandpass_dir=bandpass_dir,
            bandpass_root=bandpass_root,
        )

        control_bandpass_list = []
        for bpn in bandpass_names:
            dummy_bp = Bandpass()
            dummy_bp.read_throughput(os.path.join(bandpass_dir, bandpass_root + bpn + ".dat"))
            control_bandpass_list.append(dummy_bp)

        w_min = control_bandpass_list[0].wavelen[0]
        w_max = control_bandpass_list[0].wavelen[-1]
        w_step = control_bandpass_list[0].wavelen[1] - control_bandpass_list[0].wavelen[0]

        for bp in control_bandpass_list:
            bp.resample_bandpass(wavelen_min=w_min, wavelen_max=w_max, wavelen_step=w_step)

        for test, control in zip(bandpass_dict.values(), control_bandpass_list):
            np.testing.assert_array_almost_equal(test.wavelen, control.wavelen, 19)
            np.testing.assert_array_almost_equal(test.sb, control.sb, 19)

    def test_load_bandpasses_from_files(self):
        """
        Test that running the classmethod load_bandpasses_from_files produces
        expected result
        """

        file_dir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData")
        bandpass_names = ["g", "z", "i"]
        bandpass_root = "test_bandpass_"
        component_list = ["toy_mirror.dat"]
        atmo = os.path.join(file_dir, "toy_atmo.dat")

        bandpass_dict, hardware_dict = BandpassDict.load_bandpasses_from_files(
            bandpass_names=bandpass_names,
            filedir=file_dir,
            bandpass_root=bandpass_root,
            component_list=component_list,
            atmo_transmission=atmo,
        )

        control_bandpass_list = []
        control_hardware_list = []

        for bpn in bandpass_names:
            component_list = [
                os.path.join(file_dir, bandpass_root + bpn + ".dat"),
                os.path.join(file_dir, "toy_mirror.dat"),
            ]

            dummy_bp = Bandpass()
            dummy_bp.read_throughput_list(component_list)
            control_hardware_list.append(dummy_bp)

            component_list = [
                os.path.join(file_dir, bandpass_root + bpn + ".dat"),
                os.path.join(file_dir, "toy_mirror.dat"),
                os.path.join(file_dir, "toy_atmo.dat"),
            ]

            dummy_bp = Bandpass()
            dummy_bp.read_throughput_list(component_list)
            control_bandpass_list.append(dummy_bp)

        w_min = control_bandpass_list[0].wavelen[0]
        w_max = control_bandpass_list[0].wavelen[-1]
        w_step = control_bandpass_list[0].wavelen[1] - control_bandpass_list[0].wavelen[0]

        for bp, hh in zip(control_bandpass_list, control_hardware_list):
            bp.resample_bandpass(wavelen_min=w_min, wavelen_max=w_max, wavelen_step=w_step)
            hh.resample_bandpass(wavelen_min=w_min, wavelen_max=w_max, wavelen_step=w_step)

        for test, control in zip(bandpass_dict.values(), control_bandpass_list):
            np.testing.assert_array_almost_equal(test.wavelen, control.wavelen, 19)
            np.testing.assert_array_almost_equal(test.sb, control.sb, 19)

        for test, control in zip(hardware_dict.values(), control_hardware_list):
            np.testing.assert_array_almost_equal(test.wavelen, control.wavelen, 19)
            np.testing.assert_array_almost_equal(test.sb, control.sb, 19)


if __name__ == "__main__":
    unittest.main()
