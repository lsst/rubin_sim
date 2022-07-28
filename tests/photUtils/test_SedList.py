import unittest
import os
import numpy as np

from rubin_sim.photUtils import Bandpass, Sed, SedList
import rubin_sim
from rubin_sim.data import get_data_dir


class SedListTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(18233)
        self.sedDir = os.path.join(
            get_data_dir(), "tests", "cartoonSedTestData", "galaxySed"
        )
        self.sedPossibilities = os.listdir(self.sedDir)

    def getListOfSedNames(self, nNames):
        return [
            self.sedPossibilities[ii].replace(".gz", "")
            for ii in self.rng.randint(0, len(self.sedPossibilities) - 1, nNames)
        ]

    def testExceptions(self):
        """
        Test that exceptions are raised when they should be
        """
        nSed = 10
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList = self.rng.random_sample(nSed) * 5.0
        galacticAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        testList = SedList(
            sedNameList,
            magNormList,
            fileDir=self.sedDir,
            internalAvList=internalAvList,
            redshiftList=redshiftList,
            galacticAvList=galacticAvList,
            wavelenMatch=wavelen_match,
        )

        with self.assertRaises(AttributeError) as context:
            testList.wavelenMatch = np.arange(10.0, 1000.0, 1000.0)

        with self.assertRaises(AttributeError) as context:
            testList.cosmologicalDimming = False

        with self.assertRaises(AttributeError) as context:
            testList.redshiftList = [1.8]

        with self.assertRaises(AttributeError) as context:
            testList.internalAvList = [2.5]

        with self.assertRaises(AttributeError) as context:
            testList.galacticAvList = [1.9]

        testList = SedList(sedNameList, magNormList, fileDir=self.sedDir)

        with self.assertRaises(RuntimeError) as context:
            testList.loadSedsFromList(
                sedNameList, magNormList, internalAvList=internalAvList
            )
        self.assertIn("does not contain internalAvList", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            testList.loadSedsFromList(
                sedNameList, magNormList, galacticAvList=galacticAvList
            )
        self.assertIn("does not contain galacticAvList", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            testList.loadSedsFromList(
                sedNameList, magNormList, redshiftList=redshiftList
            )
        self.assertIn("does not contain redshiftList", context.exception.args[0])

    def testSetUp(self):
        """
        Test the SedList can be successfully initialized
        """

        ############## Try just reading in an normalizing some SEDs
        nSed = 10
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        testList = SedList(sedNameList, magNormList, fileDir=self.sedDir)
        self.assertEqual(len(testList), nSed)
        self.assertIsNone(testList.internalAvList)
        self.assertIsNone(testList.galacticAvList)
        self.assertIsNone(testList.redshiftList)
        self.assertIsNone(testList.wavelenMatch)
        self.assertTrue(testList.cosmologicalDimming)

        imsimBand = Bandpass()
        imsimBand.imsimBandpass()

        for name, norm, sedTest in zip(sedNameList, magNormList, testList):
            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))
            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        ################# now add an internalAv
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        testList = SedList(
            sedNameList, magNormList, fileDir=self.sedDir, internalAvList=internalAvList
        )
        self.assertIsNone(testList.galacticAvList)
        self.assertIsNone(testList.redshiftList)
        self.assertIsNone(testList.wavelenMatch)
        self.assertTrue(testList.cosmologicalDimming)
        for avControl, avTest in zip(internalAvList, testList.internalAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        for name, norm, av, sedTest in zip(
            sedNameList, magNormList, internalAvList, testList
        ):
            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))
            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=av)

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        ################ now add redshift
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList = self.rng.random_sample(nSed) * 5.0
        testList = SedList(
            sedNameList,
            magNormList,
            fileDir=self.sedDir,
            internalAvList=internalAvList,
            redshiftList=redshiftList,
        )
        self.assertIsNone(testList.galacticAvList)
        self.assertIsNone(testList.wavelenMatch)
        self.assertTrue(testList.cosmologicalDimming)
        for avControl, avTest in zip(internalAvList, testList.internalAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        for zControl, zTest in zip(redshiftList, testList.redshiftList):
            self.assertAlmostEqual(zControl, zTest, 10)

        for name, norm, av, zz, sedTest in zip(
            sedNameList, magNormList, internalAvList, redshiftList, testList
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))
            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=av)

            sedControl.redshiftSED(zz, dimming=True)

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        ################# without cosmological dimming
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList = self.rng.random_sample(nSed) * 5.0
        testList = SedList(
            sedNameList,
            magNormList,
            fileDir=self.sedDir,
            internalAvList=internalAvList,
            redshiftList=redshiftList,
            cosmologicalDimming=False,
        )
        self.assertIsNone(testList.galacticAvList)
        self.assertIsNone(testList.wavelenMatch)
        self.assertFalse(testList.cosmologicalDimming)
        for avControl, avTest in zip(internalAvList, testList.internalAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        for zControl, zTest in zip(redshiftList, testList.redshiftList):
            self.assertAlmostEqual(zControl, zTest, 10)

        for name, norm, av, zz, sedTest in zip(
            sedNameList, magNormList, internalAvList, redshiftList, testList
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))
            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=av)

            sedControl.redshiftSED(zz, dimming=False)

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        ################ now add galacticAv
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList = self.rng.random_sample(nSed) * 5.0
        galacticAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        testList = SedList(
            sedNameList,
            magNormList,
            fileDir=self.sedDir,
            internalAvList=internalAvList,
            redshiftList=redshiftList,
            galacticAvList=galacticAvList,
        )
        self.assertIsNone(testList.wavelenMatch)
        self.assertTrue(testList.cosmologicalDimming)
        for avControl, avTest in zip(internalAvList, testList.internalAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        for zControl, zTest in zip(redshiftList, testList.redshiftList):
            self.assertAlmostEqual(zControl, zTest, 10)

        for avControl, avTest in zip(galacticAvList, testList.galacticAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        for name, norm, av, zz, gav, sedTest in zip(
            sedNameList,
            magNormList,
            internalAvList,
            redshiftList,
            galacticAvList,
            testList,
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))
            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=av)

            sedControl.redshiftSED(zz, dimming=True)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        ################ now use a wavelen_match
        sedNameList = self.getListOfSedNames(nSed)
        magNormList = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList = self.rng.random_sample(nSed) * 5.0
        galacticAvList = self.rng.random_sample(nSed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        testList = SedList(
            sedNameList,
            magNormList,
            fileDir=self.sedDir,
            internalAvList=internalAvList,
            redshiftList=redshiftList,
            galacticAvList=galacticAvList,
            wavelenMatch=wavelen_match,
        )

        self.assertTrue(testList.cosmologicalDimming)
        for avControl, avTest in zip(internalAvList, testList.internalAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        for zControl, zTest in zip(redshiftList, testList.redshiftList):
            self.assertAlmostEqual(zControl, zTest, 10)

        for avControl, avTest in zip(galacticAvList, testList.galacticAvList):
            self.assertAlmostEqual(avControl, avTest, 10)

        np.testing.assert_array_equal(wavelen_match, testList.wavelenMatch)

        for name, norm, av, zz, gav, sedTest in zip(
            sedNameList,
            magNormList,
            internalAvList,
            redshiftList,
            galacticAvList,
            testList,
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=av)

            sedControl.redshiftSED(zz, dimming=True)
            sedControl.resampleSED(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

    def testAddingToList(self):
        """
        Test that we can add Seds to an already instantiated SedList
        """
        imsimBand = Bandpass()
        imsimBand.imsimBandpass()
        nSed = 10
        sedNameList_0 = self.getListOfSedNames(nSed)
        magNormList_0 = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList_0 = self.rng.random_sample(nSed) * 5.0
        galacticAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        testList = SedList(
            sedNameList_0,
            magNormList_0,
            fileDir=self.sedDir,
            internalAvList=internalAvList_0,
            redshiftList=redshiftList_0,
            galacticAvList=galacticAvList_0,
            wavelenMatch=wavelen_match,
        )

        # experiment with adding different combinations of physical parameter lists
        # as None and not None
        for addIav in [True, False]:
            for addRedshift in [True, False]:
                for addGav in [True, False]:

                    testList = SedList(
                        sedNameList_0,
                        magNormList_0,
                        fileDir=self.sedDir,
                        internalAvList=internalAvList_0,
                        redshiftList=redshiftList_0,
                        galacticAvList=galacticAvList_0,
                        wavelenMatch=wavelen_match,
                    )

                    sedNameList_1 = self.getListOfSedNames(nSed)
                    magNormList_1 = self.rng.random_sample(nSed) * 5.0 + 15.0

                    if addIav:
                        internalAvList_1 = self.rng.random_sample(nSed) * 0.3 + 0.1
                    else:
                        internalAvList_1 = None

                    if addRedshift:
                        redshiftList_1 = self.rng.random_sample(nSed) * 5.0
                    else:
                        redshiftList_1 = None

                    if addGav:
                        galacticAvList_1 = self.rng.random_sample(nSed) * 0.3 + 0.1
                    else:
                        galacticAvList_1 = None

                    testList.loadSedsFromList(
                        sedNameList_1,
                        magNormList_1,
                        internalAvList=internalAvList_1,
                        galacticAvList=galacticAvList_1,
                        redshiftList=redshiftList_1,
                    )

                    self.assertEqual(len(testList), 2 * nSed)
                    np.testing.assert_array_equal(wavelen_match, testList.wavelenMatch)

                    for ix in range(len(sedNameList_0)):
                        self.assertAlmostEqual(
                            internalAvList_0[ix], testList.internalAvList[ix], 10
                        )
                        self.assertAlmostEqual(
                            galacticAvList_0[ix], testList.galacticAvList[ix], 10
                        )
                        self.assertAlmostEqual(
                            redshiftList_0[ix], testList.redshiftList[ix], 10
                        )

                    for ix in range(len(sedNameList_1)):
                        if addIav:
                            self.assertAlmostEqual(
                                internalAvList_1[ix],
                                testList.internalAvList[ix + nSed],
                                10,
                            )
                        else:
                            self.assertIsNone(testList.internalAvList[ix + nSed])

                        if addGav:
                            self.assertAlmostEqual(
                                galacticAvList_1[ix],
                                testList.galacticAvList[ix + nSed],
                                10,
                            )
                        else:
                            self.assertIsNone(testList.galacticAvList[ix + nSed])

                        if addRedshift:
                            self.assertAlmostEqual(
                                redshiftList_1[ix], testList.redshiftList[ix + nSed], 10
                            )
                        else:
                            self.assertIsNone(testList.redshiftList[ix + nSed])

                    for ix, (name, norm, iav, gav, zz) in enumerate(
                        zip(
                            sedNameList_0,
                            magNormList_0,
                            internalAvList_0,
                            galacticAvList_0,
                            redshiftList_0,
                        )
                    ):

                        sedControl = Sed()
                        sedControl.readSED_flambda(
                            os.path.join(self.sedDir, name + ".gz")
                        )

                        fnorm = sedControl.calcFluxNorm(norm, imsimBand)
                        sedControl.multiplyFluxNorm(fnorm)

                        a_coeff, b_coeff = sedControl.setupCCM_ab()
                        sedControl.addDust(a_coeff, b_coeff, A_v=iav)

                        sedControl.redshiftSED(zz, dimming=True)
                        sedControl.resampleSED(wavelen_match=wavelen_match)

                        a_coeff, b_coeff = sedControl.setupCCM_ab()
                        sedControl.addDust(a_coeff, b_coeff, A_v=gav)

                        sedTest = testList[ix]

                        np.testing.assert_array_equal(
                            sedControl.wavelen, sedTest.wavelen
                        )
                        np.testing.assert_array_equal(
                            sedControl.flambda, sedTest.flambda
                        )
                        np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

                    if not addIav:
                        internalAvList_1 = [None] * nSed

                    if not addRedshift:
                        redshiftList_1 = [None] * nSed

                    if not addGav:
                        galacticAvList_1 = [None] * nSed

                    for ix, (name, norm, iav, gav, zz) in enumerate(
                        zip(
                            sedNameList_1,
                            magNormList_1,
                            internalAvList_1,
                            galacticAvList_1,
                            redshiftList_1,
                        )
                    ):

                        sedControl = Sed()
                        sedControl.readSED_flambda(
                            os.path.join(self.sedDir, name + ".gz")
                        )

                        fnorm = sedControl.calcFluxNorm(norm, imsimBand)
                        sedControl.multiplyFluxNorm(fnorm)

                        if addIav:
                            a_coeff, b_coeff = sedControl.setupCCM_ab()
                            sedControl.addDust(a_coeff, b_coeff, A_v=iav)

                        if addRedshift:
                            sedControl.redshiftSED(zz, dimming=True)

                        sedControl.resampleSED(wavelen_match=wavelen_match)

                        if addGav:
                            a_coeff, b_coeff = sedControl.setupCCM_ab()
                            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

                        sedTest = testList[ix + nSed]

                        np.testing.assert_array_equal(
                            sedControl.wavelen, sedTest.wavelen
                        )
                        np.testing.assert_array_equal(
                            sedControl.flambda, sedTest.flambda
                        )
                        np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

    def testAddingNonesToList(self):
        """
        Test what happens if you add SEDs to an SedList that have None for
        one or more of the physical parameters (i.e. galacticAv, internalAv, or redshift)
        """
        imsimBand = Bandpass()
        imsimBand.imsimBandpass()
        nSed = 10
        sedNameList_0 = self.getListOfSedNames(nSed)
        magNormList_0 = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList_0 = self.rng.random_sample(nSed) * 5.0
        galacticAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        testList = SedList(
            sedNameList_0,
            magNormList_0,
            fileDir=self.sedDir,
            internalAvList=internalAvList_0,
            redshiftList=redshiftList_0,
            galacticAvList=galacticAvList_0,
            wavelenMatch=wavelen_match,
        )

        sedNameList_1 = self.getListOfSedNames(nSed)
        magNormList_1 = list(self.rng.random_sample(nSed) * 5.0 + 15.0)
        internalAvList_1 = list(self.rng.random_sample(nSed) * 0.3 + 0.1)
        redshiftList_1 = list(self.rng.random_sample(nSed) * 5.0)
        galacticAvList_1 = list(self.rng.random_sample(nSed) * 0.3 + 0.1)

        internalAvList_1[0] = None
        redshiftList_1[1] = None
        galacticAvList_1[2] = None

        internalAvList_1[3] = None
        redshiftList_1[3] = None

        internalAvList_1[4] = None
        galacticAvList_1[4] = None

        redshiftList_1[5] = None
        galacticAvList_1[5] = None

        internalAvList_1[6] = None
        redshiftList_1[6] = None
        galacticAvList_1[6] = None

        testList.loadSedsFromList(
            sedNameList_1,
            magNormList_1,
            internalAvList=internalAvList_1,
            galacticAvList=galacticAvList_1,
            redshiftList=redshiftList_1,
        )

        self.assertEqual(len(testList), 2 * nSed)
        np.testing.assert_array_equal(wavelen_match, testList.wavelenMatch)

        for ix in range(len(sedNameList_0)):
            self.assertAlmostEqual(
                internalAvList_0[ix], testList.internalAvList[ix], 10
            )
            self.assertAlmostEqual(
                galacticAvList_0[ix], testList.galacticAvList[ix], 10
            )
            self.assertAlmostEqual(redshiftList_0[ix], testList.redshiftList[ix], 10)

        for ix in range(len(sedNameList_1)):
            self.assertAlmostEqual(
                internalAvList_1[ix], testList.internalAvList[ix + nSed], 10
            )
            self.assertAlmostEqual(
                galacticAvList_1[ix], testList.galacticAvList[ix + nSed], 10
            )
            self.assertAlmostEqual(
                redshiftList_1[ix], testList.redshiftList[ix + nSed], 10
            )

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sedNameList_0,
                magNormList_0,
                internalAvList_0,
                galacticAvList_0,
                redshiftList_0,
            )
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=iav)

            sedControl.redshiftSED(zz, dimming=True)
            sedControl.resampleSED(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            sedTest = testList[ix]

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sedNameList_1,
                magNormList_1,
                internalAvList_1,
                galacticAvList_1,
                redshiftList_1,
            )
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            if iav is not None:
                a_coeff, b_coeff = sedControl.setupCCM_ab()
                sedControl.addDust(a_coeff, b_coeff, A_v=iav)

            if zz is not None:
                sedControl.redshiftSED(zz, dimming=True)

            sedControl.resampleSED(wavelen_match=wavelen_match)

            if gav is not None:
                a_coeff, b_coeff = sedControl.setupCCM_ab()
                sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            sedTest = testList[ix + nSed]

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

    def testAlternateNormalizingBandpass(self):
        """
        A reiteration of testAddingToList, but testing with a non-imsimBandpass
        normalizing bandpass
        """
        normalizingBand = Bandpass()
        normalizingBand.read_throughput(
            os.path.join(get_data_dir(), "throughputs", "baseline", "total_r.dat")
        )
        nSed = 10
        sedNameList_0 = self.getListOfSedNames(nSed)
        magNormList_0 = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList_0 = self.rng.random_sample(nSed) * 5.0
        galacticAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        testList = SedList(
            sedNameList_0,
            magNormList_0,
            fileDir=self.sedDir,
            normalizingBandpass=normalizingBand,
            internalAvList=internalAvList_0,
            redshiftList=redshiftList_0,
            galacticAvList=galacticAvList_0,
            wavelenMatch=wavelen_match,
        )

        sedNameList_1 = self.getListOfSedNames(nSed)
        magNormList_1 = self.rng.random_sample(nSed) * 5.0 + 15.0

        internalAvList_1 = self.rng.random_sample(nSed) * 0.3 + 0.1

        redshiftList_1 = self.rng.random_sample(nSed) * 5.0

        galacticAvList_1 = self.rng.random_sample(nSed) * 0.3 + 0.1

        testList.loadSedsFromList(
            sedNameList_1,
            magNormList_1,
            internalAvList=internalAvList_1,
            galacticAvList=galacticAvList_1,
            redshiftList=redshiftList_1,
        )

        self.assertEqual(len(testList), 2 * nSed)
        np.testing.assert_array_equal(wavelen_match, testList.wavelenMatch)

        for ix in range(len(sedNameList_0)):
            self.assertAlmostEqual(
                internalAvList_0[ix], testList.internalAvList[ix], 10
            )
            self.assertAlmostEqual(
                galacticAvList_0[ix], testList.galacticAvList[ix], 10
            )
            self.assertAlmostEqual(redshiftList_0[ix], testList.redshiftList[ix], 10)

        for ix in range(len(sedNameList_1)):
            self.assertAlmostEqual(
                internalAvList_1[ix], testList.internalAvList[ix + nSed], 10
            )
            self.assertAlmostEqual(
                galacticAvList_1[ix], testList.galacticAvList[ix + nSed], 10
            )
            self.assertAlmostEqual(
                redshiftList_1[ix], testList.redshiftList[ix + nSed], 10
            )

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sedNameList_0,
                magNormList_0,
                internalAvList_0,
                galacticAvList_0,
                redshiftList_0,
            )
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, normalizingBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=iav)

            sedControl.redshiftSED(zz, dimming=True)
            sedControl.resampleSED(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            sedTest = testList[ix]

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sedNameList_1,
                magNormList_1,
                internalAvList_1,
                galacticAvList_1,
                redshiftList_1,
            )
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, normalizingBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=iav)

            sedControl.redshiftSED(zz, dimming=True)

            sedControl.resampleSED(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            sedTest = testList[ix + nSed]

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

    def testFlush(self):
        """
        Test that the flush method of SedList behaves properly
        """
        imsimBand = Bandpass()
        imsimBand.imsimBandpass()
        nSed = 10
        sedNameList_0 = self.getListOfSedNames(nSed)
        magNormList_0 = self.rng.random_sample(nSed) * 5.0 + 15.0
        internalAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        redshiftList_0 = self.rng.random_sample(nSed) * 5.0
        galacticAvList_0 = self.rng.random_sample(nSed) * 0.3 + 0.1
        wavelen_match = np.arange(300.0, 1500.0, 10.0)
        testList = SedList(
            sedNameList_0,
            magNormList_0,
            fileDir=self.sedDir,
            internalAvList=internalAvList_0,
            redshiftList=redshiftList_0,
            galacticAvList=galacticAvList_0,
            wavelenMatch=wavelen_match,
        )

        self.assertEqual(len(testList), nSed)
        np.testing.assert_array_equal(wavelen_match, testList.wavelenMatch)

        for ix in range(len(sedNameList_0)):
            self.assertAlmostEqual(
                internalAvList_0[ix], testList.internalAvList[ix], 10
            )
            self.assertAlmostEqual(
                galacticAvList_0[ix], testList.galacticAvList[ix], 10
            )
            self.assertAlmostEqual(redshiftList_0[ix], testList.redshiftList[ix], 10)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sedNameList_0,
                magNormList_0,
                internalAvList_0,
                galacticAvList_0,
                redshiftList_0,
            )
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=iav)

            sedControl.redshiftSED(zz, dimming=True)
            sedControl.resampleSED(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            sedTest = testList[ix]

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)

        testList.flush()

        sedNameList_1 = self.getListOfSedNames(nSed // 2)
        magNormList_1 = self.rng.random_sample(nSed // 2) * 5.0 + 15.0
        internalAvList_1 = self.rng.random_sample(nSed // 2) * 0.3 + 0.1
        redshiftList_1 = self.rng.random_sample(nSed // 2) * 5.0
        galacticAvList_1 = self.rng.random_sample(nSed // 2) * 0.3 + 0.1

        testList.loadSedsFromList(
            sedNameList_1,
            magNormList_1,
            internalAvList=internalAvList_1,
            galacticAvList=galacticAvList_1,
            redshiftList=redshiftList_1,
        )

        self.assertEqual(len(testList), nSed / 2)
        self.assertEqual(len(testList.redshiftList), nSed / 2)
        self.assertEqual(len(testList.internalAvList), nSed / 2)
        self.assertEqual(len(testList.galacticAvList), nSed / 2)
        np.testing.assert_array_equal(wavelen_match, testList.wavelenMatch)

        for ix in range(len(sedNameList_1)):
            self.assertAlmostEqual(
                internalAvList_1[ix], testList.internalAvList[ix], 10
            )
            self.assertAlmostEqual(
                galacticAvList_1[ix], testList.galacticAvList[ix], 10
            )
            self.assertAlmostEqual(redshiftList_1[ix], testList.redshiftList[ix], 10)

        for ix, (name, norm, iav, gav, zz) in enumerate(
            zip(
                sedNameList_1,
                magNormList_1,
                internalAvList_1,
                galacticAvList_1,
                redshiftList_1,
            )
        ):

            sedControl = Sed()
            sedControl.readSED_flambda(os.path.join(self.sedDir, name + ".gz"))

            fnorm = sedControl.calcFluxNorm(norm, imsimBand)
            sedControl.multiplyFluxNorm(fnorm)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=iav)

            sedControl.redshiftSED(zz, dimming=True)
            sedControl.resampleSED(wavelen_match=wavelen_match)

            a_coeff, b_coeff = sedControl.setupCCM_ab()
            sedControl.addDust(a_coeff, b_coeff, A_v=gav)

            sedTest = testList[ix]

            np.testing.assert_array_equal(sedControl.wavelen, sedTest.wavelen)
            np.testing.assert_array_equal(sedControl.flambda, sedTest.flambda)
            np.testing.assert_array_equal(sedControl.fnu, sedTest.fnu)


if __name__ == "__main__":
    unittest.main()
