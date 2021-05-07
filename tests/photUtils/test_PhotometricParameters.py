import os
import numpy as np
import unittest
from rubin_sim.data import get_data_dir
from rubin_sim.photUtils import Bandpass, Sed, PhotometricParameters, PhysicalParameters


class PhotometricParametersUnitTest(unittest.TestCase):

    def testInit(self):
        """
        Test that the init and getters of PhotometricParameters work
        properly
        """
        defaults = PhotometricParameters()
        params = ['exptime', 'nexp', 'effarea',
                  'gain', 'readnoise', 'darkcurrent',
                  'othernoise', 'platescale', 'sigmaSys']

        for attribute in params:
            kwargs = {}
            kwargs[attribute] = -100.0
            testCase = PhotometricParameters(**kwargs)

            for pp in params:
                if pp != attribute:
                    self.assertEqual(defaults.__getattribute__(pp),
                                     testCase.__getattribute__(pp))
                else:
                    self.assertNotEqual(defaults.__getattribute__(pp),
                                        testCase.__getattribute__(pp))

                    self.assertEqual(testCase.__getattribute__(pp), -100.0)

    def testExceptions(self):
        """
        Test that exceptions get raised when they ought to by the
        PhotometricParameters constructor

        We will instantiate PhotometricParametrs with different incomplete
        lists of parameters set.  We will verify that the returned
        error messages correctly point out which parameters were ignored.
        """

        expectedMessage = {'exptime': 'did not set exptime',
                           'nexp': 'did not set nexp',
                           'effarea': 'did not set effarea',
                           'gain': 'did not set gain',
                           'platescale': 'did not set platescale',
                           'sigmaSys': 'did not set sigmaSys',
                           'readnoise': 'did not set readnoise',
                           'darkcurrent': 'did not set darkcurrent',
                           'othernoise': 'did not set othernoise'}

        with self.assertRaises(RuntimeError) as context:
            PhotometricParameters(bandpass='x')

        for name in expectedMessage:
            self.assertIn(expectedMessage[name], context.exception.args[0])

        for name1 in expectedMessage:
            for name2 in expectedMessage:
                setParameters = {name1: 2.0, name2: 2.0}
                with self.assertRaises(RuntimeError) as context:
                    PhotometricParameters(bandpass='x', **setParameters)

                for name3 in expectedMessage:
                    if name3 not in setParameters:
                        self.assertIn(expectedMessage[name3], context.exception.args[0])
                    else:
                        self.assertNotIn(expectedMessage[name3], context.exception.args[0])

    def testDefaults(self):
        """
        Test that PhotometricParameters are correctly assigned to defaults
        """
        bandpassNames = ['u', 'g', 'r', 'i', 'z', 'y', None]
        for bp in bandpassNames:
            photParams = PhotometricParameters(bandpass=bp)
            self.assertEqual(photParams.bandpass, bp)
            self.assertAlmostEqual(photParams.exptime, 15.0, 7)
            self.assertAlmostEqual(photParams.nexp, 2, 7)
            self.assertAlmostEqual(photParams.effarea/(np.pi*(6.423*100/2.0)**2), 1.0, 7)
            self.assertAlmostEqual(photParams.gain, 2.3, 7)
            self.assertAlmostEqual(photParams.darkcurrent, 0.2, 7)
            self.assertAlmostEqual(photParams.readnoise, 8.8, 7)
            self.assertAlmostEqual(photParams.othernoise, 0, 7)
            self.assertAlmostEqual(photParams.platescale, 0.2, 7)
            if bp not in ['u', 'z', 'y']:
                self.assertAlmostEqual(photParams.sigmaSys, 0.005, 7)
            else:
                self.assertAlmostEqual(photParams.sigmaSys, 0.0075, 7)

    def testNoBandpass(self):
        """
        Test that if no bandpass is set, bandpass stays 'None' even after all other
        parameters are assigned.
        """
        photParams = PhotometricParameters()
        self.assertEqual(photParams.bandpass, None)
        self.assertAlmostEqual(photParams.exptime, 15.0, 7)
        self.assertAlmostEqual(photParams.nexp, 2, 7)
        self.assertAlmostEqual(photParams.effarea/(np.pi*(6.423*100/2.0)**2), 1.0, 7)
        self.assertAlmostEqual(photParams.gain, 2.3, 7)
        self.assertAlmostEqual(photParams.darkcurrent, 0.2, 7)
        self.assertAlmostEqual(photParams.readnoise, 8.8, 7)
        self.assertAlmostEqual(photParams.othernoise, 0, 7)
        self.assertAlmostEqual(photParams.platescale, 0.2, 7)
        self.assertAlmostEqual(photParams.sigmaSys, 0.005, 7)

    def testAssignment(self):
        """
        Test that it is impossible to set PhotometricParameters on the fly
        """
        testCase = PhotometricParameters()
        controlCase = PhotometricParameters()
        success = 0

        msg = ''
        try:
            testCase.exptime = -1.0
            success += 1
            msg += 'was able to assign exptime; '
        except:
            self.assertEqual(testCase.exptime, controlCase.exptime)

        try:
            testCase.nexp = -1.0
            success += 1
            msg += 'was able to assign nexp; '
        except:
            self.assertEqual(testCase.nexp, controlCase.nexp)

        try:
            testCase.effarea = -1.0
            success += 1
            msg += 'was able to assign effarea; '
        except:
            self.assertEqual(testCase.effarea, controlCase.effarea)

        try:
            testCase.gain = -1.0
            success += 1
            msg += 'was able to assign gain; '
        except:
            self.assertEqual(testCase.gain, controlCase.gain)

        try:
            testCase.readnoise = -1.0
            success += 1
            msg += 'was able to assign readnoise; '
        except:
            self.assertEqual(testCase.readnoise, controlCase.readnoise)

        try:
            testCase.darkcurrent = -1.0
            success += 1
            msg += 'was able to assign darkcurrent; '
        except:
            self.assertEqual(testCase.darkcurrent, controlCase.darkcurrent)

        try:
            testCase.othernoise = -1.0
            success += 1
            msg += 'was able to assign othernoise; '
        except:
            self.assertEqual(testCase.othernoise, controlCase.othernoise)

        try:
            testCase.platescale = -1.0
            success += 1
            msg += 'was able to assign platescale; '
        except:
            self.assertEqual(testCase.platescale, controlCase.platescale)

        try:
            testCase.sigmaSys = -1.0
            success += 1
            msg += 'was able to assign sigmaSys; '
        except:
            self.assertEqual(testCase.sigmaSys, controlCase.sigmaSys)

        try:
            testCase.bandpass = 'z'
            success += 1
            msg += 'was able to assign bandpass; '
        except:
            self.assertEqual(testCase.bandpass, controlCase.bandpass)

        self.assertEqual(success, 0, msg=msg)

    def testApplication(self):
        """
        Test that PhotometricParameters get properly propagated into
        Sed methods.  We will test this using Sed.calcADU, since the ADU
        scale linearly with the appropriate parameter.
        """

        testSed = Sed()
        testSed.setFlatSED()

        testBandpass = Bandpass()
        testBandpass.readThroughput(os.path.join(get_data_dir(), 'throughputs',
                                                 'baseline', 'total_g.dat'))

        control = testSed.calcADU(testBandpass,
                                  photParams=PhotometricParameters())

        testCase = PhotometricParameters(exptime=30.0)

        test = testSed.calcADU(testBandpass, photParams=testCase)

        self.assertGreater(control, 0.0)
        self.assertEqual(control, 0.5*test)


class PhysicalParametersUnitTest(unittest.TestCase):

    def testAssignment(self):
        """
        Make sure it is impossible to change the values stored in
        PhysicalParameters
        """

        pp = PhysicalParameters()
        control = PhysicalParameters()
        success = 0
        msg = ''

        try:
            pp.minwavelen = 2.0
            success += 1
            msg += 'was able to assign minwavelen; '
        except:
            self.assertEqual(pp.minwavelen, control.minwavelen)

        try:
            pp.maxwavelen = 2.0
            success += 1
            msg += 'was able to assign maxwavelen; '
        except:
            self.assertEqual(pp.maxwavelen, control.maxwavelen)

        try:
            pp.wavelenstep = 2.0
            success += 1
            msg += 'was able to assign wavelenstep; '
        except:
            self.assertEqual(pp.wavelenstep, control.wavelenstep)

        try:
            pp.lightspeed = 2.0
            success += 1
            msg += 'was able to assign lightspeed; '
        except:
            self.assertEqual(pp.lightspeed, control.lightspeed)

        try:
            pp.planck = 2.0
            success += 1
            msg += 'was able to assign planck; '
        except:
            self.assertEqual(pp.planck, control.planck)

        try:
            pp.nm2m = 2.0
            success += 1
            msg += 'was able to assign nm2m; '
        except:
            self.assertEqual(pp.nm2m, control.nm2m)

        try:
            pp.ergsetc2jansky = 2.0
            msg += 'was able to assign ergsetc2jansky; '
            success += 1
        except:
            self.assertEqual(pp.ergsetc2jansky, control.ergsetc2jansky)

        self.assertEqual(success, 0, msg=msg)


if __name__ == "__main__":
    unittest.main()
