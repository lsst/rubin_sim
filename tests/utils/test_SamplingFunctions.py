"""Tests for sims.utils.samplingFunctions.py

1.  test_raiseWraparoundError: The `samplePatchOnSphere` function  does not
 wrap around theta values near the pole. Check if the approporiate error is
 by such a call
2. test_checkWithinBounds : Check that the samples are indeed within the bounds
prescribed by ObsMetaData
3. test_samplePatchOnSphere : Check functionality by showing that binning up in
    dec results in numbers in dec bins changing with area.
 """
import numpy as np
import unittest\

from rubin_sim.utils import ObservationMetaData
from rubin_sim.utils import samplePatchOnSphere
from rubin_sim.utils import spatiallySample_obsmetadata


class SamplingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        """
        cls.obsMetaDataforCat = ObservationMetaData(boundType='circle',
                                                    boundLength=np.degrees(
                                                        0.25),
                                                    pointingRA=np.degrees(
                                                        0.13),
                                                    pointingDec=np.degrees(-1.2),
                                                    bandpassName=['r'],
                                                    mjd=49350.)
        ObsMetaData = cls.obsMetaDataforCat
        cls.samples = spatiallySample_obsmetadata(ObsMetaData, size=1000)

        cls.theta_c = -60.
        cls.phi_c = 30.
        cls.delta = 30.
        cls.size = 1000000

        cls.dense_samples = samplePatchOnSphere(phi=cls.phi_c, theta=cls.theta_c,
                                                delta=cls.delta, size=cls.size,
                                                seed=42)

    def test_raiseWraparoundError(self):
        """
        Test that appropriate errors are raised when at the poles
        """
        # thetamax to be exceeded
        deltamax = np.abs(self.theta_c - 0.05)
        # to be lower than thetamin
        deltamin = 2.67
        with self.assertRaises(ValueError):
            samplePatchOnSphere(phi=self.phi_c, theta=self.theta_c,
                                delta=deltamax, size=self.size, seed=42)
            samplePatchOnSphere(phi=self.phi_c, theta=self.theta_c,
                                delta=deltamin, size=self.size, seed=42)

    def test_checkWithinBounds(self):

        delta = self.obsMetaDataforCat.boundLength
        # delta = np.radians(delta)
        minPhi = 0.13 - delta
        maxPhi = 0.13 + delta
        minTheta = -1.2 - delta
        maxTheta = -1.2 + delta

        self.assertTrue(all(np.radians(self.samples[0]) <= maxPhi),
                        msg='samples are not <= maxPhi')
        self.assertTrue(all(np.radians(self.samples[0]) >= minPhi),
                        msg='samples are not >= minPhi')
        self.assertTrue(all(np.radians(self.samples[1]) >= minTheta),
                        msg='samples are not >= minTheta')
        self.assertTrue(all(np.radians(self.samples[1]) <= maxTheta),
                        msg='samples are not <= maxTheta')

    def test_samplePatchOnSphere(self):

        def A(theta_min, theta_max):
            return np.sin(theta_max) - np.sin(theta_min)

        theta_c = np.radians(self.theta_c)
        delta = np.radians(self.delta)

        theta_min = theta_c - delta
        theta_max = theta_c + delta
        tvals = np.arange(theta_min, theta_max, 0.001)
        tvalsShifted = np.zeros(len(tvals))
        tvalsShifted[:-1] = tvals[1:]

        area = A(tvals, tvalsShifted)

        binsize = np.unique(np.diff(tvals))
        self.assertEqual(binsize.size, 1)
        normval = np.sum(area) * binsize[0]

        theta_samps = np.radians(self.dense_samples[1])
        binnedvals = np.histogram(theta_samps, bins=tvals[:-1], density=True)[0]
        resids = area[:-2] / normval - binnedvals

        fiveSigma = np.sqrt(binnedvals) * 5.0
        np.testing.assert_array_less(resids, fiveSigma)


if __name__ == "__main__":
    unittest.main()
