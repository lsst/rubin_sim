import unittest
import os
import numpy as np

from rubin_sim.phot_utils import Bandpass, Sed, SedList
import rubin_sim
from rubin_sim.data import get_data_dir


class SedListTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(18233)
        self.sed_dir = os.path.join(
            get_data_dir(), "tests", "cartoonSedTestData", "galaxySed"
        )
        self.sed_possibilities = os.listdir(self.sed_dir)

    def get_list_of_sed_names(self, n_names):
        return [
            self.sed_possibilities[ii].replace(".gz", "")
            for ii in self.rng.randint(0, len(self.sed_possibilities) - 1, n_names)
        ]

    def test_exceptions(self):
        """
        Test that exceptions are raised when they should be
        """
        n_sed = 10
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        test_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=wavelen_match,
        )

        with self.assertRaises(AttributeError) as context:
            test_list.wavelen_match = np.arange(10.0, 1000.0, 1000.0)

        with self.assertRaises(AttributeError) as context:
            test_list.cosmological_dimming = False

        with self.assertRaises(AttributeError) as context:
            test_list.redshift_list = [1.8]

        with self.assertRaises(AttributeError) as context:
            test_list.internal_av_list = [2.5]

        with self.assertRaises(AttributeError) as context:
            test_list.galactic_av_list = [1.9]

        test_list = SedList(sed_name_list, mag_norm_list, file_dir=self.sed_dir)

        with self.assertRaises(RuntimeError) as context:
            test_list.load_seds_from_list(
                sed_name_list, mag_norm_list, internal_av_list=internal_av_list
            )
        self.assertIn("does not contain internal_av_list", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            test_list.load_seds_from_list(
                sed_name_list, mag_norm_list, galactic_av_list=galactic_av_list
            )
        self.assertIn("does not contain galactic_av_list", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            test_list.load_seds_from_list(
                sed_name_list, mag_norm_list, redshift_list=redshift_list
            )
        self.assertIn("does not contain redshift_list", context.exception.args[0])

    def test_setUp(self):
        """
        Test the SedList can be successfully initialized
        """

        ############## Try just reading in an normalizing some SEDs
        n_sed = 10
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        test_list = SedList(sed_name_list, mag_norm_list, file_dir=self.sed_dir)
        self.assertEqual(len(test_list), n_sed)
        self.assertIsNone(test_list.internal_av_list)
        self.assertIsNone(test_list.galactic_av_list)
        self.assertIsNone(test_list.redshift_list)
        self.assertIsNone(test_list.wavelen_match)
        self.assertTrue(test_list.cosmological_dimming)

        imsim_band = Bandpass()
        imsim_band.imsim_bandpass()

        for name, norm, sedTest in zip(sed_name_list, mag_norm_list, test_list):
            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))
            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            np.testing.assert_array_equal(sed_control.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sedTest.fnu)

        ################# now add an internalAv
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        test_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
        )
        self.assertIsNone(test_list.galactic_av_list)
        self.assertIsNone(test_list.redshift_list)
        self.assertIsNone(test_list.wavelen_match)
        self.assertTrue(test_list.cosmological_dimming)
        for av_control, avTest in zip(internal_av_list, test_list.internal_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        for name, norm, av, sedTest in zip(
            sed_name_list, mag_norm_list, internal_av_list, test_list
        ):
            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))
            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=av)

            np.testing.assert_array_equal(sed_control.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sedTest.fnu)

        ################ now add redshift
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        test_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
        )
        self.assertIsNone(test_list.galactic_av_list)
        self.assertIsNone(test_list.wavelen_match)
        self.assertTrue(test_list.cosmological_dimming)
        for av_control, avTest in zip(internal_av_list, test_list.internal_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        for z_control, zTest in zip(redshift_list, test_list.redshift_list):
            self.assertAlmostEqual(z_control, zTest, 10)

        for name, norm, av, zz, sedTest in zip(
            sed_name_list, mag_norm_list, internal_av_list, redshift_list, test_list
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))
            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=av)

            sed_control.redshift_sed(zz, dimming=True)

            np.testing.assert_array_equal(sed_control.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sedTest.fnu)

        ################# without cosmological dimming
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        test_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            cosmological_dimming=False,
        )
        self.assertIsNone(test_list.galactic_av_list)
        self.assertIsNone(test_list.wavelen_match)
        self.assertFalse(test_list.cosmological_dimming)
        for av_control, avTest in zip(internal_av_list, test_list.internal_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        for z_control, zTest in zip(redshift_list, test_list.redshift_list):
            self.assertAlmostEqual(z_control, zTest, 10)

        for name, norm, av, zz, sedTest in zip(
            sed_name_list, mag_norm_list, internal_av_list, redshift_list, test_list
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))
            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=av)

            sed_control.redshift_sed(zz, dimming=False)

            np.testing.assert_array_equal(sed_control.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sedTest.fnu)

        ################ now add galacticAv
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        test_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
        )
        self.assertIsNone(test_list.wavelen_match)
        self.assertTrue(test_list.cosmological_dimming)
        for av_control, avTest in zip(internal_av_list, test_list.internal_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        for z_control, zTest in zip(redshift_list, test_list.redshift_list):
            self.assertAlmostEqual(z_control, zTest, 10)

        for av_control, avTest in zip(galactic_av_list, test_list.galactic_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        for name, norm, av, zz, gav, sedTest in zip(
            sed_name_list,
            mag_norm_list,
            internal_av_list,
            redshift_list,
            galactic_av_list,
            test_list,
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))
            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=av)

            sed_control.redshift_sed(zz, dimming=True)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            np.testing.assert_array_equal(sed_control.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sedTest.fnu)

        ################ now use a wavelen_match
        sed_name_list = self.get_list_of_sed_names(n_sed)
        mag_norm_list = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list = self.rng.random_sample(n_sed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        test_list = SedList(
            sed_name_list,
            mag_norm_list,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list,
            redshift_list=redshift_list,
            galactic_av_list=galactic_av_list,
            wavelen_match=wavelen_match,
        )

        self.assertTrue(test_list.cosmological_dimming)
        for av_control, avTest in zip(internal_av_list, test_list.internal_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        for z_control, zTest in zip(redshift_list, test_list.redshift_list):
            self.assertAlmostEqual(z_control, zTest, 10)

        for av_control, avTest in zip(galactic_av_list, test_list.galactic_av_list):
            self.assertAlmostEqual(av_control, avTest, 10)

        np.testing.assert_array_equal(wavelen_match, test_list.wavelen_match)

        for name, norm, av, zz, gav, sedTest in zip(
            sed_name_list,
            mag_norm_list,
            internal_av_list,
            redshift_list,
            galactic_av_list,
            test_list,
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=av)

            sed_control.redshift_sed(zz, dimming=True)
            sed_control.resample_sed(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            np.testing.assert_array_equal(sed_control.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sedTest.fnu)

    def test_adding_to_list(self):
        """
        Test that we can add Seds to an already instantiated SedList
        """
        imsim_band = Bandpass()
        imsim_band.imsim_bandpass()
        n_sed = 10
        sed_name_list_0 = self.get_list_of_sed_names(n_sed)
        mag_norm_list_0 = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list_0 = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        test_list = SedList(
            sed_name_list_0,
            mag_norm_list_0,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list_0,
            redshift_list=redshift_list_0,
            galactic_av_list=galactic_av_list_0,
            wavelen_match=wavelen_match,
        )

        # experiment with adding different combinations of physical parameter lists
        # as None and not None
        for add_iav in [True, False]:
            for add_redshift in [True, False]:
                for add_gav in [True, False]:

                    test_list = SedList(
                        sed_name_list_0,
                        mag_norm_list_0,
                        file_dir=self.sed_dir,
                        internal_av_list=internal_av_list_0,
                        redshift_list=redshift_list_0,
                        galactic_av_list=galactic_av_list_0,
                        wavelen_match=wavelen_match,
                    )

                    sed_name_list_1 = self.get_list_of_sed_names(n_sed)
                    mag_norm_list_1 = self.rng.random_sample(n_sed) * 5.0 + 15.0

                    if add_iav:
                        internal_av_list_1 = self.rng.random_sample(n_sed) * 0.3 + 0.1
                    else:
                        internal_av_list_1 = None

                    if add_redshift:
                        redshift_list_1 = self.rng.random_sample(n_sed) * 5.0
                    else:
                        redshift_list_1 = None

                    if add_gav:
                        galactic_av_list_1 = self.rng.random_sample(n_sed) * 0.3 + 0.1
                    else:
                        galactic_av_list_1 = None

                    test_list.load_seds_from_list(
                        sed_name_list_1,
                        mag_norm_list_1,
                        internal_av_list=internal_av_list_1,
                        galactic_av_list=galactic_av_list_1,
                        redshift_list=redshift_list_1,
                    )

                    self.assertEqual(len(test_list), 2 * n_sed)
                    np.testing.assert_array_equal(
                        wavelen_match, test_list.wavelen_match
                    )

                    for ix in range(len(sed_name_list_0)):
                        self.assertAlmostEqual(
                            internal_av_list_0[ix], test_list.internal_av_list[ix], 10
                        )
                        self.assertAlmostEqual(
                            galactic_av_list_0[ix], test_list.galactic_av_list[ix], 10
                        )
                        self.assertAlmostEqual(
                            redshift_list_0[ix], test_list.redshift_list[ix], 10
                        )

                    for ix in range(len(sed_name_list_1)):
                        if add_iav:
                            self.assertAlmostEqual(
                                internal_av_list_1[ix],
                                test_list.internal_av_list[ix + n_sed],
                                10,
                            )
                        else:
                            self.assertIsNone(test_list.internal_av_list[ix + n_sed])

                        if add_gav:
                            self.assertAlmostEqual(
                                galactic_av_list_1[ix],
                                test_list.galactic_av_list[ix + n_sed],
                                10,
                            )
                        else:
                            self.assertIsNone(test_list.galactic_av_list[ix + n_sed])

                        if add_redshift:
                            self.assertAlmostEqual(
                                redshift_list_1[ix],
                                test_list.redshift_list[ix + n_sed],
                                10,
                            )
                        else:
                            self.assertIsNone(test_list.redshift_list[ix + n_sed])

                    for ix, (name, norm, iav, gav, zz) in enumerate(
                        zip(
                            sed_name_list_0,
                            mag_norm_list_0,
                            internal_av_list_0,
                            galactic_av_list_0,
                            redshift_list_0,
                        )
                    ):

                        sed_control = Sed()
                        sed_control.read_sed_flambda(
                            os.path.join(self.sed_dir, name + ".gz")
                        )

                        fnorm = sed_control.calc_flux_norm(norm, imsim_band)
                        sed_control.multiply_flux_norm(fnorm)

                        a_coeff, b_coeff = sed_control.setup_ccm_ab()
                        sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

                        sed_control.redshift_sed(zz, dimming=True)
                        sed_control.resample_sed(wavelen_match=wavelen_match)

                        a_coeff, b_coeff = sed_control.setup_ccm_ab()
                        sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

                        sed_test = test_list[ix]

                        np.testing.assert_array_equal(
                            sed_control.wavelen, sed_test.wavelen
                        )
                        np.testing.assert_array_equal(
                            sed_control.flambda, sed_test.flambda
                        )
                        np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

                    if not add_iav:
                        internal_av_list_1 = [None] * n_sed

                    if not add_redshift:
                        redshift_list_1 = [None] * n_sed

                    if not add_gav:
                        galactic_av_list_1 = [None] * n_sed

                    for ix, (name, norm, iav, gav, zz) in enumerate(
                        zip(
                            sed_name_list_1,
                            mag_norm_list_1,
                            internal_av_list_1,
                            galactic_av_list_1,
                            redshift_list_1,
                        )
                    ):

                        sed_control = Sed()
                        sed_control.read_sed_flambda(
                            os.path.join(self.sed_dir, name + ".gz")
                        )

                        fnorm = sed_control.calc_flux_norm(norm, imsim_band)
                        sed_control.multiply_flux_norm(fnorm)

                        if add_iav:
                            a_coeff, b_coeff = sed_control.setup_ccm_ab()
                            sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

                        if add_redshift:
                            sed_control.redshift_sed(zz, dimming=True)

                        sed_control.resample_sed(wavelen_match=wavelen_match)

                        if add_gav:
                            a_coeff, b_coeff = sed_control.setup_ccm_ab()
                            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

                        sed_test = test_list[ix + n_sed]

                        np.testing.assert_array_equal(
                            sed_control.wavelen, sed_test.wavelen
                        )
                        np.testing.assert_array_equal(
                            sed_control.flambda, sed_test.flambda
                        )
                        np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

    def test_adding_nones_to_list(self):
        """
        Test what happens if you add SEDs to an SedList that have None for
        one or more of the physical parameters (i.e. galacticAv, internalAv, or redshift)
        """
        imsim_band = Bandpass()
        imsim_band.imsim_bandpass()
        n_sed = 10
        sed_name_list_0 = self.get_list_of_sed_names(n_sed)
        mag_norm_list_0 = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list_0 = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        test_list = SedList(
            sed_name_list_0,
            mag_norm_list_0,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list_0,
            redshift_list=redshift_list_0,
            galactic_av_list=galactic_av_list_0,
            wavelen_match=wavelen_match,
        )

        sed_name_list_1 = self.get_list_of_sed_names(n_sed)
        mag_norm_list_1 = list(self.rng.random_sample(n_sed) * 5.0 + 15.0)
        internal_av_list_1 = list(self.rng.random_sample(n_sed) * 0.3 + 0.1)
        redshift_list_1 = list(self.rng.random_sample(n_sed) * 5.0)
        galactic_av_list_1 = list(self.rng.random_sample(n_sed) * 0.3 + 0.1)

        internal_av_list_1[0] = None
        redshift_list_1[1] = None
        galactic_av_list_1[2] = None

        internal_av_list_1[3] = None
        redshift_list_1[3] = None

        internal_av_list_1[4] = None
        galactic_av_list_1[4] = None

        redshift_list_1[5] = None
        galactic_av_list_1[5] = None

        internal_av_list_1[6] = None
        redshift_list_1[6] = None
        galactic_av_list_1[6] = None

        test_list.load_seds_from_list(
            sed_name_list_1,
            mag_norm_list_1,
            internal_av_list=internal_av_list_1,
            galactic_av_list=galactic_av_list_1,
            redshift_list=redshift_list_1,
        )

        self.assertEqual(len(test_list), 2 * n_sed)
        np.testing.assert_array_equal(wavelen_match, test_list.wavelen_match)

        for ix in range(len(sed_name_list_0)):
            self.assertAlmostEqual(
                internal_av_list_0[ix], test_list.internal_av_list[ix], 10
            )
            self.assertAlmostEqual(
                galactic_av_list_0[ix], test_list.galactic_av_list[ix], 10
            )
            self.assertAlmostEqual(redshift_list_0[ix], test_list.redshift_list[ix], 10)

        for ix in range(len(sed_name_list_1)):
            self.assertAlmostEqual(
                internal_av_list_1[ix], test_list.internal_av_list[ix + n_sed], 10
            )
            self.assertAlmostEqual(
                galactic_av_list_1[ix], test_list.galactic_av_list[ix + n_sed], 10
            )
            self.assertAlmostEqual(
                redshift_list_1[ix], test_list.redshift_list[ix + n_sed], 10
            )

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sed_name_list_0,
                mag_norm_list_0,
                internal_av_list_0,
                galactic_av_list_0,
                redshift_list_0,
            )
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

            sed_control.redshift_sed(zz, dimming=True)
            sed_control.resample_sed(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            sed_test = test_list[ix]

            np.testing.assert_array_equal(sed_control.wavelen, sed_test.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sed_test.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sed_name_list_1,
                mag_norm_list_1,
                internal_av_list_1,
                galactic_av_list_1,
                redshift_list_1,
            )
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            if iav is not None:
                a_coeff, b_coeff = sed_control.setup_ccm_ab()
                sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

            if zz is not None:
                sed_control.redshift_sed(zz, dimming=True)

            sed_control.resample_sed(wavelen_match=wavelen_match)

            if gav is not None:
                a_coeff, b_coeff = sed_control.setup_ccm_ab()
                sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            sed_test = test_list[ix + n_sed]

            np.testing.assert_array_equal(sed_control.wavelen, sed_test.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sed_test.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

    def test_alternate_normalizing_bandpass(self):
        """
        A reiteration of testAddingToList, but testing with a non-imsim_bandpass
        normalizing bandpass
        """
        normalizing_band = Bandpass()
        normalizing_band.read_throughput(
            os.path.join(get_data_dir(), "throughputs", "baseline", "total_r.dat")
        )
        n_sed = 10
        sed_name_list_0 = self.get_list_of_sed_names(n_sed)
        mag_norm_list_0 = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list_0 = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        test_list = SedList(
            sed_name_list_0,
            mag_norm_list_0,
            file_dir=self.sed_dir,
            normalizing_bandpass=normalizing_band,
            internal_av_list=internal_av_list_0,
            redshift_list=redshift_list_0,
            galactic_av_list=galactic_av_list_0,
            wavelen_match=wavelen_match,
        )

        sed_name_list_1 = self.get_list_of_sed_names(n_sed)
        mag_norm_list_1 = self.rng.random_sample(n_sed) * 5.0 + 15.0

        internal_av_list_1 = self.rng.random_sample(n_sed) * 0.3 + 0.1

        redshift_list_1 = self.rng.random_sample(n_sed) * 5.0

        galactic_av_list_1 = self.rng.random_sample(n_sed) * 0.3 + 0.1

        test_list.load_seds_from_list(
            sed_name_list_1,
            mag_norm_list_1,
            internal_av_list=internal_av_list_1,
            galactic_av_list=galactic_av_list_1,
            redshift_list=redshift_list_1,
        )

        self.assertEqual(len(test_list), 2 * n_sed)
        np.testing.assert_array_equal(wavelen_match, test_list.wavelen_match)

        for ix in range(len(sed_name_list_0)):
            self.assertAlmostEqual(
                internal_av_list_0[ix], test_list.internal_av_list[ix], 10
            )
            self.assertAlmostEqual(
                galactic_av_list_0[ix], test_list.galactic_av_list[ix], 10
            )
            self.assertAlmostEqual(redshift_list_0[ix], test_list.redshift_list[ix], 10)

        for ix in range(len(sed_name_list_1)):
            self.assertAlmostEqual(
                internal_av_list_1[ix], test_list.internal_av_list[ix + n_sed], 10
            )
            self.assertAlmostEqual(
                galactic_av_list_1[ix], test_list.galactic_av_list[ix + n_sed], 10
            )
            self.assertAlmostEqual(
                redshift_list_1[ix], test_list.redshift_list[ix + n_sed], 10
            )

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sed_name_list_0,
                mag_norm_list_0,
                internal_av_list_0,
                galactic_av_list_0,
                redshift_list_0,
            )
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, normalizing_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

            sed_control.redshift_sed(zz, dimming=True)
            sed_control.resample_sed(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            sed_test = test_list[ix]

            np.testing.assert_array_equal(sed_control.wavelen, sed_test.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sed_test.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sed_name_list_1,
                mag_norm_list_1,
                internal_av_list_1,
                galactic_av_list_1,
                redshift_list_1,
            )
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, normalizing_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

            sed_control.redshift_sed(zz, dimming=True)

            sed_control.resample_sed(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            sed_test = test_list[ix + n_sed]

            np.testing.assert_array_equal(sed_control.wavelen, sed_test.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sed_test.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

    def test_flush(self):
        """
        Test that the flush method of SedList behaves properly
        """
        imsim_band = Bandpass()
        imsim_band.imsim_bandpass()
        n_sed = 10
        sed_name_list_0 = self.get_list_of_sed_names(n_sed)
        mag_norm_list_0 = self.rng.random_sample(n_sed) * 5.0 + 15.0
        internal_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        redshift_list_0 = self.rng.random_sample(n_sed) * 5.0
        galactic_av_list_0 = self.rng.random_sample(n_sed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        test_list = SedList(
            sed_name_list_0,
            mag_norm_list_0,
            file_dir=self.sed_dir,
            internal_av_list=internal_av_list_0,
            redshift_list=redshift_list_0,
            galactic_av_list=galactic_av_list_0,
            wavelen_match=wavelen_match,
        )

        self.assertEqual(len(test_list), n_sed)
        np.testing.assert_array_equal(wavelen_match, test_list.wavelen_match)

        for ix in range(len(sed_name_list_0)):
            self.assertAlmostEqual(
                internal_av_list_0[ix], test_list.internal_av_list[ix], 10
            )
            self.assertAlmostEqual(
                galactic_av_list_0[ix], test_list.galactic_av_list[ix], 10
            )
            self.assertAlmostEqual(redshift_list_0[ix], test_list.redshift_list[ix], 10)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sed_name_list_0,
                mag_norm_list_0,
                internal_av_list_0,
                galactic_av_list_0,
                redshift_list_0,
            )
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

            sed_control.redshift_sed(zz, dimming=True)
            sed_control.resample_sed(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            sed_test = test_list[ix]

            np.testing.assert_array_equal(sed_control.wavelen, sed_test.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sed_test.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)

        test_list.flush()

        sed_name_list_1 = self.get_list_of_sed_names(n_sed // 2)
        mag_norm_list_1 = self.rng.random_sample(n_sed // 2) * 5.0 + 15.0
        internal_av_list_1 = self.rng.random_sample(n_sed // 2) * 0.3 + 0.1
        redshift_list_1 = self.rng.random_sample(n_sed // 2) * 5.0
        galactic_av_list_1 = self.rng.random_sample(n_sed // 2) * 0.3 + 0.1

        test_list.load_seds_from_list(
            sed_name_list_1,
            mag_norm_list_1,
            internal_av_list=internal_av_list_1,
            galactic_av_list=galactic_av_list_1,
            redshift_list=redshift_list_1,
        )

        self.assertEqual(len(test_list), n_sed / 2)
        self.assertEqual(len(test_list.redshift_list), n_sed / 2)
        self.assertEqual(len(test_list.internal_av_list), n_sed / 2)
        self.assertEqual(len(test_list.galactic_av_list), n_sed / 2)
        np.testing.assert_array_equal(wavelen_match, test_list.wavelen_match)

        for ix in range(len(sed_name_list_1)):
            self.assertAlmostEqual(
                internal_av_list_1[ix], test_list.internal_av_list[ix], 10
            )
            self.assertAlmostEqual(
                galactic_av_list_1[ix], test_list.galactic_av_list[ix], 10
            )
            self.assertAlmostEqual(redshift_list_1[ix], test_list.redshift_list[ix], 10)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sed_name_list_1,
                mag_norm_list_1,
                internal_av_list_1,
                galactic_av_list_1,
                redshift_list_1,
            )
        ):

            sed_control = Sed()
            sed_control.read_sed_flambda(os.path.join(self.sed_dir, name + ".gz"))

            fnorm = sed_control.calc_flux_norm(norm, imsim_band)
            sed_control.multiply_flux_norm(fnorm)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=iav)

            sed_control.redshift_sed(zz, dimming=True)
            sed_control.resample_sed(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sed_control.setup_ccm_ab()
            sed_control.add_dust(a_coeff, b_coeff, a_v=gav)

            sed_test = test_list[ix]

            np.testing.assert_array_equal(sed_control.wavelen, sed_test.wavelen)
            np.testing.assert_array_equal(sed_control.flambda, sed_test.flambda)
            np.testing.assert_array_equal(sed_control.fnu, sed_test.fnu)


if __name__ == "__main__":
    unittest.main()
