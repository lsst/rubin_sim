import numpy as np

import os
import unittest
from rubin_sim.utils import ObservationMetaData
from rubin_sim.utils.CodeUtilities import sims_clean_up
from rubin_sim.photUtils.Bandpass import Bandpass
from rubin_sim.photUtils.Sed import Sed
from rubin_sim.photUtils import BandpassDict
from rubin_sim.data import get_data_dir
import rubin_sim


class photometryUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
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

    def testAlternateBandpassesStars(self):
        """
        This will test our ability to do photometry using non-LSST bandpasses.

        It will first calculate the magnitudes using the getters in cartoonPhotometryStars.

        It will then load the alternate bandpass files 'by hand' and re-calculate the magnitudes
        and make sure that the magnitude values agree.  This is guarding against the possibility
        that some default value did not change and the code actually ended up loading the
        LSST bandpasses.
        """
        bandpassDir = os.path.join(get_data_dir(), "tests", "cartoonSedTestData")

        cartoon_dict = BandpassDict.loadTotalBandpassesFromFiles(
            ["u", "g", "r", "i", "z"],
            bandpassDir=bandpassDir,
            bandpassRoot="test_bandpass_",
        )

        testBandPasses = {}
        keys = ["u", "g", "r", "i", "z"]

        bplist = []

        for kk in keys:
            testBandPasses[kk] = Bandpass()
            testBandPasses[kk].readThroughput(
                os.path.join(bandpassDir, "test_bandpass_%s.dat" % kk)
            )
            bplist.append(testBandPasses[kk])

        sedObj = Sed()
        phiArray, waveLenStep = sedObj.setupPhiArray(bplist)

        sedFileName = os.path.join(
            get_data_dir(), "tests", "cartoonSedTestData/starSed/"
        )
        sedFileName = os.path.join(sedFileName, "kurucz", "km20_5750.fits_g40_5790.gz")
        ss = Sed()
        ss.readSED_flambda(sedFileName)

        controlBandpass = Bandpass()
        controlBandpass.imsimBandpass()
        ff = ss.calcFluxNorm(22.0, controlBandpass)
        ss.multiplyFluxNorm(ff)

        testMags = cartoon_dict.magListForSed(ss)

        ss.resampleSED(wavelen_match=bplist[0].wavelen)
        ss.flambdaTofnu()
        mags = -2.5 * np.log10(np.sum(phiArray * ss.fnu, axis=1) * waveLenStep) - ss.zp
        self.assertEqual(len(mags), len(testMags))
        self.assertGreater(len(mags), 0)
        for j in range(len(mags)):
            self.assertAlmostEqual(mags[j], testMags[j], 10)


if __name__ == "__main__":
    unittest.main()
