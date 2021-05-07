import numpy as np
import warnings
import unittest
import gzip
import os
import tempfile
import shutil

import rubin_sim.photUtils.Sed as Sed
import rubin_sim.photUtils.Bandpass as Bandpass
from rubin_sim.photUtils import PhotometricParameters
import rubin_sim
from rubin_sim.utils import getPackageDir

ROOT = os.path.abspath(os.path.dirname(__file__))


class TestSedWavelenLimits(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('always')
        self.wmin = 500
        self.wmax = 1500
        self.bandpasswavelen = np.arange(self.wmin, self.wmax+.5, 1)
        self.bandpasssb = np.ones(len(self.bandpasswavelen))
        self.testbandpass = Bandpass(wavelen=self.bandpasswavelen, sb=self.bandpasssb)

    def tearDown(self):
        del self.bandpasswavelen
        del self.bandpasssb
        del self.testbandpass
        del self.wmin
        del self.wmax

    def testSedWavelenRange(self):
        """Test setting sed with wavelength range different from standard values works properly."""
        sedwavelen = self.bandpasswavelen * 1.0
        sedflambda = np.ones(len(sedwavelen))
        testsed = Sed(wavelen=sedwavelen, flambda=sedflambda, name='TestSed')
        np.testing.assert_equal(testsed.wavelen, sedwavelen)
        np.testing.assert_equal(testsed.flambda, sedflambda)
        self.assertEqual(testsed.name, 'TestSed')

    def testSedBandpassMatch(self):
        """Test errors when bandpass and sed do not completely overlap in wavelength range."""
        # Test case where they do match (no error message)
        sedwavelen = np.arange(self.wmin, self.wmax+.5, 1)
        sedflambda = np.ones(len(sedwavelen))
        testsed = Sed(wavelen=sedwavelen, flambda=sedflambda)
        print('')
        # Test that no warning is made.
        with warnings.catch_warnings(record=True) as wa:
            w, f = testsed.resampleSED(wavelen_match=self.testbandpass.wavelen,
                                       wavelen=testsed.wavelen, flux=testsed.flambda)
            self.assertEqual(len(wa), 0)
        np.testing.assert_equal(w, testsed.wavelen)
        np.testing.assert_equal(f, testsed.flambda)
        # Test that warning is given for non-overlap at either top or bottom end of wavelength range.
        sedwavelen = np.arange(self.wmin, self.wmax - 50, 1)
        sedflambda = np.ones(len(sedwavelen))
        testsed = Sed(wavelen=sedwavelen, flambda=sedflambda)
        with warnings.catch_warnings(record=True) as wa:
            testsed.resampleSED(wavelen_match=self.testbandpass.wavelen)
            self.assertEqual(len(wa), 1)
            self.assertIn('non-overlap', str(wa[-1].message))
        np.testing.assert_equal(testsed.flambda[-1:], np.NaN)
        sedwavelen = np.arange(self.wmin+50, self.wmax, 1)
        sedflambda = np.ones(len(sedwavelen))
        testsed = Sed(wavelen=sedwavelen, flambda=sedflambda)
        with warnings.catch_warnings(record=True) as wa:
            testsed.resampleSED(wavelen_match=self.testbandpass.wavelen)
            self.assertEqual(len(wa), 1)
            self.assertIn('non-overlap', str(wa[-1].message))
        np.testing.assert_equal(testsed.flambda[0], np.NaN)
        np.testing.assert_equal(testsed.flambda[49], np.NaN)

    def testSedMagErrors(self):
        """Test error handling at mag and adu calculation levels of sed."""
        sedwavelen = np.arange(self.wmin+50, self.wmax, 1)
        sedflambda = np.ones(len(sedwavelen))
        testsed = Sed(wavelen=sedwavelen, flambda=sedflambda)
        # Test handling in calcMag
        with warnings.catch_warnings(record=True) as w:
            mag = testsed.calcMag(self.testbandpass)
            self.assertEqual(len(w), 1)
            self.assertIn("non-overlap", str(w[-1].message))
        np.testing.assert_equal(mag, np.NaN)
        # Test handling in calcADU
        with warnings.catch_warnings(record=True) as w:
            adu = testsed.calcADU(self.testbandpass,
                                  photParams=PhotometricParameters())
            self.assertEqual(len(w), 1)
            self.assertIn("non-overlap", str(w[-1].message))
        np.testing.assert_equal(adu, np.NaN)
        # Test handling in calcFlux
        with warnings.catch_warnings(record=True) as w:
            flux = testsed.calcFlux(self.testbandpass)
            self.assertEqual(len(w), 1)
            self.assertIn("non-overlap", str(w[-1].message))
        np.testing.assert_equal(flux, np.NaN)


class TestSedName(unittest.TestCase):
    def setUp(self):
        self.wmin = 500
        self.wmax = 1500
        self.wavelen = np.arange(self.wmin, self.wmax+.5, 1)
        self.flambda = np.ones(len(self.wavelen))
        self.name = 'TestSed'
        self.testsed = Sed(self.wavelen, self.flambda, name=self.name)

    def tearDown(self):
        del self.wmin, self.wmax, self.wavelen, self.flambda
        del self.name
        del self.testsed

    def testSetName(self):
        self.assertEqual(self.testsed.name, self.name)

    def testRedshiftName(self):
        testsed = Sed(self.testsed.wavelen, self.testsed.flambda, name=self.testsed.name)
        redshift = .2
        testsed.redshiftSED(redshift=redshift)
        newname = testsed.name + '_Z' + '%.2f' % (redshift)
        testsed.name = newname
        self.assertEqual(testsed.name, newname)


class SedBasicFunctionsTestCase(unittest.TestCase):

    longMessage = True

    def test_read_sed_flambda(self):
        """
        Test how readSED_flambda handles the reading of SED filenames
        when we fail to correctly specify their gzipped state.
        """

        scratch_dir = tempfile.mkdtemp(prefix='test_read_sed_flambda',
                                       dir=ROOT)

        rng = np.random.RandomState(88)
        zipped_name = os.path.join(scratch_dir, "zipped_sed.txt.gz")
        unzipped_name = os.path.join(scratch_dir, "unzipped_sed.txt")
        if os.path.exists(zipped_name):
            os.unlink(zipped_name)
        if os.path.exists(unzipped_name):
            os.unlink(unzipped_name)
        wv = np.arange(100.0, 1000.0, 10.0)
        flux = rng.random_sample(len(wv))
        with gzip.open(zipped_name, "wt") as output_file:
            for ww, ff in zip(wv, flux):
                output_file.write("%e %e\n" % (ww, ff))
        with open(unzipped_name, "wt") as output_file:
            for ww, ff in zip(wv, flux):
                output_file.write("%e %e\n" % (ww, ff))

        ss = Sed()
        ss.readSED_flambda(zipped_name)
        ss.readSED_flambda(zipped_name[:-3])
        ss.readSED_flambda(unzipped_name)
        ss.readSED_flambda(unzipped_name+'.gz')

        # make sure an error is raised when you try to read
        # a file that does not exist
        with self.assertRaises(IOError) as context:
            ss.readSED_flambda(os.path.join(scratch_dir, "nonsense.txt"))
        self.assertIn("sed file", context.exception.args[0])

        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)

    def test_eq(self):
        """
        Test that __eq__ in Sed works correctly
        """
        sed_dir = os.path.join(getPackageDir(rubin_sim), 'tests/photUtils',
                               'cartoonSedTestData', 'starSed', 'kurucz')
        list_of_seds = os.listdir(sed_dir)
        sedname1 = os.path.join(sed_dir, list_of_seds[0])
        sedname2 = os.path.join(sed_dir, list_of_seds[1])
        ss1 = Sed()
        ss1.readSED_flambda(sedname1)
        ss2 = Sed()
        ss2.readSED_flambda(sedname2)
        ss3 = Sed()
        ss3.readSED_flambda(sedname1)

        self.assertFalse(ss1 == ss2)
        self.assertTrue(ss1 != ss2)
        self.assertTrue(ss1 == ss3)
        self.assertFalse(ss1 != ss3)

        ss3.flambdaTofnu()

        self.assertFalse(ss1 == ss3)
        self.assertTrue(ss1 != ss3)

    def test_cache(self):
        """
        Verify that loading an SED from the cache gives identical
        results to loading the same SED from ASCII (since we are
        not calling cache_LSST_seds(), as soon as we load an SED
        with readSED_flambda, it should get stored in the
        _global_misc_sed_cache)
        """
        sed_dir = os.path.join(getPackageDir(rubin_sim), 'tests/photUtils',
                               'cartoonSedTestData', 'starSed', 'kurucz')

        sed_name_list = os.listdir(sed_dir)
        msg = ('An SED loaded from the cache is not '
               'identical to the same SED loaded from disk')
        for ix in range(5):
            full_name = os.path.join(sed_dir, sed_name_list[ix])
            ss_uncache = Sed()
            ss_uncache.readSED_flambda(full_name)
            ss_cache = Sed()
            ss_cache.readSED_flambda(full_name)

            self.assertEqual(ss_cache, ss_uncache, msg=msg)

        # test that modifications to an SED don't get pushed
        # to the cache
        full_name = os.path.join(sed_dir, sed_name_list[0])
        ss1 = Sed()
        ss1.readSED_flambda(full_name)
        ss2 = Sed()
        ss2.readSED_flambda(full_name)
        ss2.flambda *= 2.0
        ss3 = Sed()
        ss3.readSED_flambda(full_name)
        msg = "Changes to SED made it into the cache"
        self.assertEqual(ss1, ss3, msg=msg)
        self.assertNotEqual(ss1, ss2, msg=msg)
        self.assertNotEqual(ss2, ss3, msg=msg)

    def test_calcErgs(self):
        """
        Test that calcErgs actually calculates the flux of a source in
        ergs/s/cm^2 by running it on black bodies with flat bandpasses
        and comparing to the Stefan-Boltzmann law.
        """

        boltzmann_k = 1.3807e-16  # in ergs/Kelvin
        planck_h = 6.6261e-27  # in cm^2*g/s
        speed_of_light = 2.9979e10  # in cm/s
        stefan_boltzmann_sigma = 5.6705e-5  # in ergs/cm^2/s/Kelvin

        wavelen_arr = np.arange(10.0, 200000.0, 10.0)  # in nm
        bp = Bandpass(wavelen=wavelen_arr, sb=np.ones(len(wavelen_arr)))

        log10_bb_factor = np.log10(2.0) + np.log10(planck_h)
        log10_bb_factor += 2.0*np.log10(speed_of_light)
        log10_bb_factor -= 5.0*(np.log10(wavelen_arr) - 7.0)  # convert wavelen to cm

        for temp in np.arange(5000.0, 7000.0, 250.0):
            log10_exp_arg = np.log10(planck_h) + np.log10(speed_of_light)
            log10_exp_arg -= (np.log10(wavelen_arr) - 7.0)
            log10_exp_arg -= (np.log10(boltzmann_k) + np.log10(temp))

            exp_arg = np.power(10.0, log10_exp_arg)
            log10_bose_factor = -1.0*np.log10(np.exp(exp_arg)-1.0)

            # the -7.0 below is because, otherwise, flambda will be in
            # ergs/s/cm^2/cm and we want ergs/s/cm^2/nm
            #
            # the np.pi comes from the integral in the 'Stefan-Boltzmann'
            # section of
            # https://en.wikipedia.org/wiki/Planck%27s_law#Stefan.E2.80.93Boltzmann_law
            #
            bb_flambda = np.pi*np.power(10.0, log10_bb_factor+log10_bose_factor-7.0)

            sed = Sed(wavelen=wavelen_arr, flambda=bb_flambda)
            ergs = sed.calcErgs(bp)

            log10_ergs = np.log10(stefan_boltzmann_sigma) + 4.0*np.log10(temp)
            ergs_truth = np.power(10.0, log10_ergs)

            msg = '\ntemp:%e\nergs: %e\nergs_truth: %e' % (temp, ergs, ergs_truth)
            self.assertAlmostEqual(ergs/ergs_truth, 1.0, 3, msg=msg)

        # Now test it on a bandpass with throughput=0.25 and an wavelength
        # array that is not the same as the SED

        wavelen_arr = np.arange(10.0, 100000.0, 146.0)  # in nm
        bp = Bandpass(wavelen=wavelen_arr, sb=0.25*np.ones(len(wavelen_arr)))

        wavelen_arr = np.arange(5.0, 200000.0, 17.0)

        log10_bb_factor = np.log10(2.0) + np.log10(planck_h)
        log10_bb_factor += 2.0*np.log10(speed_of_light)
        log10_bb_factor -= 5.0*(np.log10(wavelen_arr) - 7.0)  # convert wavelen to cm

        for temp in np.arange(5000.0, 7000.0, 250.0):
            log10_exp_arg = np.log10(planck_h) + np.log10(speed_of_light)
            log10_exp_arg -= (np.log10(wavelen_arr) - 7.0)
            log10_exp_arg -= (np.log10(boltzmann_k) + np.log10(temp))

            exp_arg = np.power(10.0, log10_exp_arg)
            log10_bose_factor = -1.0*np.log10(np.exp(exp_arg)-1.0)

            # the -7.0 below is because, otherwise, flambda will be in
            # ergs/s/cm^2/cm and we want ergs/s/cm^2/nm
            #
            # the np.pi comes from the integral in the 'Stefan-Boltzmann'
            # section of
            # https://en.wikipedia.org/wiki/Planck%27s_law#Stefan.E2.80.93Boltzmann_law
            #
            bb_flambda = np.pi*np.power(10.0, log10_bb_factor+log10_bose_factor-7.0)

            sed = Sed(wavelen=wavelen_arr, flambda=bb_flambda)
            ergs = sed.calcErgs(bp)

            log10_ergs = np.log10(stefan_boltzmann_sigma) + 4.0*np.log10(temp)
            ergs_truth = np.power(10.0, log10_ergs)

            msg = '\ntemp: %e\nergs: %e\nergs_truth: %e' % (temp,ergs, ergs_truth)
            self.assertAlmostEqual(ergs/ergs_truth, 0.25, 3, msg=msg)

    def test_mags_vs_flux(self):
        """
        Verify that the relationship between Sed.calcMag() and Sed.calcFlux()
        is as expected
        """
        wavelen = np.arange(100.0, 1500.0, 1.0)
        flambda = np.exp(-0.5*np.power((wavelen-500.0)/100.0,2))
        sb = (wavelen-100.0)/1400.0

        ss = Sed(wavelen=wavelen, flambda=flambda)
        bp = Bandpass(wavelen=wavelen, sb=sb)

        mag = ss.calcMag(bp)
        flux = ss.calcFlux(bp)

        self.assertAlmostEqual(ss.magFromFlux(flux)/mag, 1.0, 10)
        self.assertAlmostEqual(ss.fluxFromMag(mag)/flux, 1.0, 10)


if __name__ == "__main__":
    unittest.main()
