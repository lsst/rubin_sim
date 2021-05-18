"""
Module to test that Approximate Bandpasses are in sync with the official LSST
Bandpasses from SYSENG

Note: While the LSST bandpasses list throughput values corresponding to
wavelengths in the range of 300.0-1150.0 nm, the `approximate_baseline`
directory of throughputs is created by a script manually. It is thus possible
for this directory to fall out of sync with the SYSENG values in `baseline`.
This module is intended to test whether this is happening.
"""
import os
import unittest
import numpy as np
from rubin_sim.photUtils import BandpassDict
from rubin_sim.data import get_data_dir


class ApproximateBandPassTest(unittest.TestCase):
    """
    Tests for the approximate Bandpasses in the throughputs directory
    """
    longMessage = True

    def setUp(self):
        """
        setup before tests
        """
        throughputsDir = os.path.join(get_data_dir(), 'throughputs')
        self.approxBandPassDir = os.path.join(throughputsDir, 'approximate_baseline')
        self.refBandPassDir = os.path.join(throughputsDir, 'baseline')
        self.refBandPassDict = BandpassDict.loadTotalBandpassesFromFiles()
        self.approxBandPassDict = \
            BandpassDict.loadTotalBandpassesFromFiles(bandpassDir=self.approxBandPassDir)
        self.errorMsg = "The failure of this test indicates that the"
        " approximate bandpasses in the lsst throughputs directory do not"
        "sync up with the baseline bandpasses is throughputs. This may require running"
        " the script : throughputs.approximate_baseline/approximateBandpasses.py"

    def test_BandPassIntegrals(self):
        """
        Test that the ratio of the quantity
        int dlambda T(lambda) = band flux for a SED proportional to $lambda$
        for the approximate bandpasses to the SYSENG band passes is 1.0 to an
        absolute tolerance hard coded to be 1.0e-4

        """
        for bn in 'ugrizy':
            refBandPass = self.refBandPassDict[bn]
            approxBandPass = self.approxBandPassDict[bn]
            refStep = np.diff(refBandPass.wavelen)
            approxStep = np.diff(approxBandPass.wavelen)

            # Currently we have uniform sampling, but the end points have
            # very slightly different steps. This accounts for 3 possible values
            # If there are more, then the steps are non-unifor
            msg = 'The step sizes in {} seem to be unequal'.format('Approximate Baseline')
            self.assertEqual(len(np.unique(approxStep)), 3, msg=msg)
            msg = 'The step sizes in {} seem to be unequal'.format('Baseline')
            self.assertEqual(len(np.unique(refStep)), 3, msg=msg)
            ratio = approxStep[1] * approxBandPass.sb.sum() / refStep[1] / refBandPass.sb.sum()
            self.assertAlmostEqual(ratio, 1.0, delta=1.0e-4, msg=self.errorMsg)

        def test_BandpassesIndiviaually(self):
            """
            Test that individual transmission values at wavelengths kept in the
            approximate bandpasses match the individual transmission values in
            the reference bandpasses
            """
            for bn in 'ugrizy':
                approxBandPass = self.approxBandPassDict[bn]
                refBandPass = self.refBandPass[bn]
                refwavelen = refBandPass.wavelen
                approxwavelen = approxBandPass.wavelen
                mask = np.zeros(len(refwavelen), dtype=bool)
                for i, wave in enumerate(refwavelen):
                    if wave in approxwavelen:
                        mask[i] = True
                # Assume that the wavelengths are in order
                msg = 'individual matches failed on band {}'.format(bn)
                self.assertAlmostEqual(refBandPass.sb[mask], approxBandPass.sb,
                                       1.0e-4, msg=msg)

        def tearDown(self):
            pass

if __name__ == "__main__":
    unittest.main()
