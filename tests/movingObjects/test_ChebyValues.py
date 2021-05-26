import unittest
import os
import numpy as np
import pandas as pd
from astropy.time import Time
import tempfile
import shutil

from rubin_sim.movingObjects import Orbits
from rubin_sim.movingObjects import PyOrbEphemerides
from rubin_sim.movingObjects import ChebyFits
from rubin_sim.movingObjects import ChebyValues
from rubin_sim.data import get_data_dir



ROOT = os.path.abspath(os.path.dirname(__file__))


class TestChebyValues(unittest.TestCase):
    def setUp(self):
        self.testdatadir = os.path.join(get_data_dir(), 'movingObjects', 'orbits_testdata')
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix='TestChebyValues-')
        self.coeffFile = os.path.join(self.scratch_dir, 'test_coeffs')
        self.residFile = os.path.join(self.scratch_dir, 'test_resids')
        self.failedFile = os.path.join(self.scratch_dir, 'test_failed')
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdatadir, 'test_orbitsNEO.s3m'), skiprows=1)
        self.pyephems = PyOrbEphemerides(os.path.join(get_data_dir(), 'movingObjects', 'de405.dat'))
        self.pyephems.setOrbits(self.orbits)
        self.tStart = self.orbits.orbits.epoch.iloc[0]
        self.interval = 30
        self.nCoeffs = 14
        self.nDecimal = 13
        self.chebyFits = ChebyFits(self.orbits, self.tStart, self.interval, ngran=64,
                                   skyTolerance=2.5, nDecimal=self.nDecimal, nCoeff_position=self.nCoeffs,
                                   obscode=807, timeScale='TAI')
        self.setLength = 0.5
        self.chebyFits.calcSegmentLength(length=self.setLength)
        self.chebyFits.calcSegments()
        self.chebyFits.write(self.coeffFile, self.residFile, self.failedFile, append=False)
        self.coeffKeys = ['objId', 'tStart', 'tEnd', 'ra', 'dec', 'geo_dist', 'vmag', 'elongation']

    def tearDown(self):
        del self.orbits
        del self.chebyFits
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def testSetCoeff(self):
        # Test setting coefficients directly from chebyFits outputs.
        chebyValues = ChebyValues()
        chebyValues.setCoefficients(self.chebyFits)
        for k in self.coeffKeys:
            self.assertTrue(k in chebyValues.coeffs)
            self.assertTrue(isinstance(chebyValues.coeffs[k], np.ndarray))
        self.assertEqual(len(np.unique(chebyValues.coeffs['objId'])), len(self.orbits))
        # This will only be true for carefully selected length/orbit type, where subdivision did not occur.
        # For the test MBAs, a len=1day will work.
        # For the test NEOs, a len=0.25 day will work (with 2.5mas skyTol).
        # self.assertEqual(len(chebyValues.coeffs['tStart']),
        #                  (self.interval / self.setLength) * len(self.orbits))
        self.assertEqual(len(chebyValues.coeffs['ra'][0]), self.nCoeffs)
        self.assertTrue('meanRA' in chebyValues.coeffs)
        self.assertTrue('meanDec' in chebyValues.coeffs)

    def testReadCoeffs(self):
        # Test reading the coefficients from disk.
        chebyValues = ChebyValues()
        chebyValues.readCoefficients(self.coeffFile)
        chebyValues2 = ChebyValues()
        chebyValues2.setCoefficients(self.chebyFits)
        for k in chebyValues.coeffs:
            if k == 'objId':
                # Can't test strings with np.test.assert_almost_equal.
                np.testing.assert_equal(chebyValues.coeffs[k], chebyValues2.coeffs[k])
            else:
                # All of these will only be accurate to 2 less decimal places than they are
                # print out with in chebyFits. Since vmag, delta and elongation only use 7
                # decimal places, this means we can test to 5 decimal places for those.
                np.testing.assert_allclose(chebyValues.coeffs[k], chebyValues2.coeffs[k], rtol=0, atol=1e-5)

    def testGetEphemerides(self):
        # Test that getEphemerides works and is accurate.
        chebyValues = ChebyValues()
        chebyValues.readCoefficients(self.coeffFile)
        # Multiple times, all objects, all within interval.
        tstep = self.interval/10.0
        time = np.arange(self.tStart, self.tStart + self.interval, tstep)
        # Test for a single time, but all the objects.
        ephemerides = chebyValues.getEphemerides(time)
        pyephemerides = self.pyephems.generateEphemerides(time, obscode=807,
                                                          timeScale='TAI', byObject=True)
        # RA and Dec should agree to 2.5mas (skyTolerance above)
        pos_residuals = np.sqrt((ephemerides['ra'] - pyephemerides['ra']) ** 2 +
                                ((ephemerides['dec'] - pyephemerides['dec']) *
                                 np.cos(np.radians(ephemerides['dec']))) ** 2)
        pos_residuals *= 3600.0 * 1000.0
        # Let's just look at the max residuals in all quantities.
        for k in ('ra', 'dec', 'dradt', 'ddecdt', 'geo_dist'):
            resids = np.abs(ephemerides[k] - pyephemerides[k])
            if k != 'geo_dist':
                resids *= 3600.0 * 1000.0
            print('max diff', k, np.max(resids))
        resids = np.abs(ephemerides['elongation'] - pyephemerides['solarelon'])
        print('max diff elongation', np.max(resids))
        resids = np.abs(ephemerides['vmag'] - pyephemerides['magV'])
        print('max diff vmag', np.max(resids))
        self.assertLessEqual(np.max(pos_residuals), 2.5)
        # Test for single time, but for a subset of the objects.
        objIds = self.orbits.orbits.objId.head(3).values
        ephemerides = chebyValues.getEphemerides(time, objIds)
        self.assertEqual(len(ephemerides['ra']), 3)
        # Test for time outside of segment range.
        ephemerides = chebyValues.getEphemerides(self.tStart + self.interval * 2, objIds, extrapolate=False)
        self.assertTrue(np.isnan(ephemerides['ra'][0]),
                        msg='Expected Nan for out of range ephemeris, got %.2e' %(ephemerides['ra'][0]))


class TestJPLValues(unittest.TestCase):
    # Test the interpolation-generated RA/Dec values against JPL generated RA/Dec values.
    # The resulting errors should be similar to the errors reported
    # from testEphemerides when testing against JPL values.
    def setUp(self):
        # Read orbits.
        self.orbits = Orbits()
        self.jplDir = os.path.join(get_data_dir(), 'movingObjects', 'jpl_testdata')
        self.orbits.readOrbits(os.path.join(self.jplDir, 'S0_n747.des'), skiprows=1)
        # Read JPL ephems.
        self.jpl = pd.read_table(os.path.join(self.jplDir, '807_n747.txt'), delim_whitespace=True)
        # Add times in TAI and UTC, because.
        t = Time(self.jpl['epoch_mjd'], format='mjd', scale='utc')
        self.jpl['mjdTAI'] = t.tai.mjd
        self.jpl['mjdUTC'] = t.utc.mjd
        self.jpl = self.jpl.to_records(index=False)
        # Generate interpolation coefficients for the time period in the JPL catalog.
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix='TestJPLValues-')
        self.coeffFile = os.path.join(self.scratch_dir, 'test_coeffs')
        self.residFile = os.path.join(self.scratch_dir, 'test_resids')
        self.failedFile = os.path.join(self.scratch_dir, 'test_failed')
        tStart = self.jpl['mjdTAI'].min() - 0.2
        tEnd = self.jpl['mjdTAI'].max() + 0.2 - self.jpl['mjdTAI'].min()
        self.chebyFits = ChebyFits(self.orbits, tStart, tEnd,
                                   ngran=64, skyTolerance=2.5,
                                   nDecimal=14, obscode=807)
        self.chebyFits.calcSegmentLength()
        self.chebyFits.calcSegments()
        self.chebyFits.write(self.coeffFile, self.residFile, self.failedFile, append=False)
        self.coeffKeys = ['objId', 'tStart', 'tEnd', 'ra', 'dec', 'geo_dist', 'vmag', 'elongation']
        self.chebyValues = ChebyValues()
        self.chebyValues.readCoefficients(self.coeffFile)

    def tearDown(self):
        del self.orbits
        del self.jpl
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def testRADec(self):
        # We won't compare Vmag, because this also needs information on trailing losses.
        times = np.unique(self.jpl['mjdTAI'])
        deltaRA = np.zeros(len(times), float)
        deltaDec = np.zeros(len(times), float)
        for i, t in enumerate(times):
            # Find the JPL objIds visible at this time.
            j = self.jpl[np.where(self.jpl['mjdTAI'] == t)]
            ephs = self.chebyValues.getEphemerides(t, j['objId'])
            ephorder = np.argsort(ephs['objId'])
            # Sometimes I've had to reorder both, sometimes just the ephs. ??
            jorder = np.argsort(j['objId'])
            jorder = np.arange(0, len(jorder))
            dRA = np.abs(ephs['ra'][ephorder][:,0] - j['ra_deg'][jorder]) * 3600.0 * 1000.0
            dDec = np.abs(ephs['dec'][ephorder][:,0] - j['dec_deg'][jorder]) * 3600.0 * 1000.0
            deltaRA[i] = dRA.max()
            deltaDec[i] = dDec.max()
            if deltaRA[i] > 20:
                print(j['objId'], ephs['objId'])
                print(j['ra_deg'])
                print(ephs['ra'])
                print(j['dec_deg'])
                print(ephs['dec'])
        # Should be (given OOrb direct prediction):
        # Much of the time we're closer than 1mas, but there are a few which hit higher values.
        # This is consistent with the errors/values reported by oorb directly in testEphemerides.
        print('max JPL errors', deltaRA.max(), deltaDec.max())
        print('std of JPL errors', np.std(deltaRA), np.std(deltaDec))
        self.assertLess(np.max(deltaRA), 25)
        self.assertLess(np.max(deltaDec), 25)
        self.assertLess(np.std(deltaRA), 3)
        self.assertLess(np.std(deltaDec), 3)


if __name__ == '__main__':
    unittest.main()
