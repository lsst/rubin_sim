import unittest
import os
import numpy as np
import pandas as pd
from astropy.time import Time
from rubin_sim.movingObjects import Orbits
from rubin_sim.movingObjects import PyOrbEphemerides
from rubin_sim.data import get_data_dir


class TestPyOrbEphemerides(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), 'tests', 'orbits_testdata')
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.orbitsKEP = Orbits()
        self.orbitsKEP.readOrbits(os.path.join(self.testdir, 'test_orbitsA.des'))
        self.ephems = PyOrbEphemerides()
        self.ephems.setOrbits(self.orbits)
        self.len_ephems_basic = 11
        self.len_ephems_full = 34

    def tearDown(self):
        del self.orbits
        del self.orbitsKEP
        del self.ephems

    def testSetOrbits(self):
        # Test that we can set orbits.
        self.ephems.setOrbits(self.orbits)
        # Test that setting with an empty orbit object fails.
        # (Avoids hard-to-interpret errors from pyoorb).
        with self.assertRaises(ValueError):
            emptyOrb = Orbits()
            empty = pd.DataFrame([], columns=self.orbits.dataCols['KEP'])
            emptyOrb.setOrbits(empty)
            self.ephems.setOrbits(emptyOrb)

    def testConvertToOorbArray(self):
        # Check that orbital elements are converted.
        self.ephems._convertToOorbElem(self.orbits.orbits, self.orbits.orb_format)
        self.assertEqual(len(self.ephems.oorbElem), len(self.orbits))
        self.assertEqual(self.ephems.oorbElem[0][7], 2)
        self.assertEqual(self.ephems.oorbElem[0][9], 3)
        self.assertEqual(self.ephems.oorbElem[0][1], self.orbits.orbits['q'][0])
        # Test that we can convert KEP orbital elements too.
        self.ephems._convertToOorbElem(self.orbitsKEP.orbits, self.orbitsKEP.orb_format)
        self.assertEqual(len(self.ephems.oorbElem), len(self.orbitsKEP))
        self.assertEqual(self.ephems.oorbElem[0][7], 3)
        self.assertEqual(self.ephems.oorbElem[0][1], self.orbitsKEP.orbits['a'][0])

    def testConvertFromOorbArray(self):
        # Check that we can convert orbital elements TO oorb format and back
        # without losing info (except ObjId -- we will lose that unless we use updateOrbits.)
        self.ephems._convertToOorbElem(self.orbits.orbits, self.orbits.orb_format)
        newOrbits = Orbits()
        newOrbits.setOrbits(self.orbits.orbits)
        newOrbits.updateOrbits(self.ephems.convertFromOorbElem())
        self.assertEqual(newOrbits, self.orbits)

    def testConvertTimes(self):
        times = np.arange(49353, 49353 + 10, 0.5)
        ephTimes = self.ephems._convertTimes(times, 'UTC')
        # Check that shape of ephTimes is correct. (times x 2)
        self.assertEqual(ephTimes.shape[0], len(times))
        self.assertEqual(ephTimes.shape[1], 2)
        # Check that 'timescale' for ephTimes is correct.
        self.assertEqual(ephTimes[0][1], 1)
        ephTimes = self.ephems._convertTimes(times, 'TAI')
        self.assertEqual(ephTimes[0][1], 4)

    def testOorbEphemeris(self):
        self.ephems.setOrbits(self.orbits)
        times = np.arange(49353, 49353 + 3, 0.25)
        ephTimes = self.ephems._convertTimes(times)
        # Basic ephemerides.
        oorbEphs = self.ephems._generateOorbEphsBasic(ephTimes, obscode=807, ephMode='N')
        # Check that it returned the right sort of array.
        self.assertEqual(oorbEphs.shape, (len(self.ephems.oorbElem), len(times), self.len_ephems_basic))
        # Full ephemerides
        oorbEphs = self.ephems._generateOorbEphsFull(ephTimes, obscode=807, ephMode='N')
        # Check that it returned the right sort of array.
        self.assertEqual(oorbEphs.shape, (len(self.ephems.oorbElem), len(times), self.len_ephems_full))

    def testEphemeris(self):
        # Calculate and convert ephemerides.
        self.ephems.setOrbits(self.orbits)
        times = np.arange(49353, 49353 + 2, 0.3)
        ephTimes = self.ephems._convertTimes(times)
        oorbEphs = self.ephems._generateOorbEphsBasic(ephTimes, obscode=807)
        # Group by object, and check grouping.
        ephs = self.ephems._convertOorbEphsBasic(oorbEphs, byObject=True)
        self.assertEqual(len(ephs), len(self.orbits))
        # Group by time, and check grouping.
        oorbEphs = self.ephems._generateOorbEphsBasic(ephTimes, obscode=807)
        ephs = self.ephems._convertOorbEphsBasic(oorbEphs, byObject=False)
        self.assertEqual(len(ephs), len(times))
        # And test all-wrapped-up method:
        ephsAll = self.ephems.generateEphemerides(times, obscode=807,
                                                  ephMode='N', ephType='basic',
                                                  timeScale='UTC', byObject=False)
        np.testing.assert_equal(ephsAll, ephs)
        # Reset ephems to use KEP Orbits, and calculate new ephemerides.
        self.ephems.setOrbits(self.orbitsKEP)
        oorbEphs = self.ephems._generateOorbEphsBasic(ephTimes, obscode=807, ephMode='N')
        ephsKEP = self.ephems._convertOorbEphsBasic(oorbEphs, byObject=True)
        self.assertEqual(len(ephsKEP), len(self.orbitsKEP))
        oorbEphs = self.ephems._generateOorbEphsBasic(ephTimes, obscode=807, ephMode='N')
        ephsKEP = self.ephems._convertOorbEphsBasic(oorbEphs, byObject=False)
        self.assertEqual(len(ephsKEP), len(times))
        # And test all-wrapped-up method:
        ephsAllKEP = self.ephems.generateEphemerides(times, obscode=807,
                                                     ephMode='N', ephType='basic',
                                                     timeScale='UTC', byObject=False)
        np.testing.assert_equal(ephsAllKEP, ephsKEP)
        # Check that ephemerides calculated from the different (COM/KEP) orbits are almost equal.
        #for column in ephs.dtype.names:
        #    np.testing.assert_allclose(ephs[column], ephsKEP[column], rtol=0, atol=1e-7)
        # Check that the wrapped method using KEP elements and the wrapped method using COM elements match.
        #for column in ephsAll.dtype.names:
        #    np.testing.assert_allclose(ephsAllKEP[column], ephsAll[column], rtol=0, atol=1e-7)


class TestJPLValues(unittest.TestCase):
    """Test the oorb generated RA/Dec values against JPL generated RA/Dec values."""
    def setUp(self):
        # Read orbits.
        self.orbits = Orbits()
        self.jplDir = os.path.join(get_data_dir(), 'tests', 'jpl_testdata')
        self.orbits.readOrbits(os.path.join(self.jplDir, 'S0_n747.des'), skiprows=1)
        # Read JPL ephems.
        self.jpl = pd.read_csv(os.path.join(self.jplDir, '807_n747.txt'), delim_whitespace=True)
        # Add times in TAI and UTC, because.
        t = Time(self.jpl['epoch_mjd'], format='mjd', scale='utc')
        self.jpl['mjdTAI'] = t.tai.mjd
        self.jpl['mjdUTC'] = t.utc.mjd

    def tearDown(self):
        del self.orbits
        del self.jpl

    def testRADec(self):
        # We won't compare Vmag, because this also needs information on trailing losses.
        times = self.jpl['mjdUTC'].unique()
        deltaRA = np.zeros(len(times), float)
        deltaDec = np.zeros(len(times), float)
        for i, t in enumerate(times):
            # Find the JPL objIds visible at this time.
            j = self.jpl.query('mjdUTC == @t').sort_values('objId')
            # Set the ephems, using the objects seen at this time.
            suborbits = self.orbits.orbits.query('objId in @j.objId').sort_values('objId')
            subOrbits = Orbits()
            subOrbits.setOrbits(suborbits)
            ephems = PyOrbEphemerides()
            ephems.setOrbits(subOrbits)
            ephs = ephems.generateEphemerides([t], timeScale='UTC', obscode=807,
                                              ephMode='N', ephType='Basic', byObject=False)
            deltaRA[i] = np.abs(ephs['ra'] - j['ra_deg'].values).max()
            deltaDec[i] = np.abs(ephs['dec'] - j['dec_deg'].values).max()
        # Convert to mas
        deltaRA *= 3600. * 1000.
        deltaDec *= 3600. * 1000.
        # Much of the time we're closer than 1mas, but there are a few which hit higher values.
        print('max JPL errors', np.max(deltaRA), np.max(deltaDec))
        print('std JPL errors', np.std(deltaRA), np.std(deltaDec))
        self.assertLess(np.max(deltaRA), 25)
        self.assertLess(np.max(deltaDec), 25)
        self.assertLess(np.std(deltaRA), 3)
        self.assertLess(np.std(deltaDec), 3)


if __name__ == '__main__':
    unittest.main()
