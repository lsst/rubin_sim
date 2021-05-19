import unittest
import os
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from rubin_sim.movingObjects import Orbits
from rubin_sim.data import get_data_dir


class TestOrbits(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), 'movingObjects/orbits_testdata')

    def testEqualNotEqual(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.assertEqual(len(orbits), 4)
        orbits2 = Orbits()
        orbits2.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.assertEqual(orbits, orbits2)
        orbits3 = Orbits()
        orbits3.readOrbits(os.path.join(self.testdir, 'test_orbitsA.des'))
        self.assertNotEqual(orbits, orbits3)

    def testIterationAndIndexing(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsNEO.s3m'))
        orbitsSingle = orbits[0]
        assert_frame_equal(orbitsSingle.orbits, orbits.orbits.query('index==0'))
        orbitsSingle = orbits[3]
        assert_frame_equal(orbitsSingle.orbits, orbits.orbits.query('index==3'))
        # Test iteration through all orbits.
        for orb, (i, orbi) in zip(orbits, orbits.orbits.iterrows()):
            self.assertEqual(orb.orbits.objId.values[0], orbi.objId)
            self.assertTrue(isinstance(orb, Orbits))
            self.assertEqual(orb.orbits.index, i)
        # Test iteration through a subset of orbits.
        orbitsSub = Orbits()
        orbitsSub.setOrbits(orbits.orbits.query('index > 4'))
        for orb, (i, orbi) in zip(orbitsSub, orbitsSub.orbits.iterrows()):
            self.assertEqual(orb.orbits.objId.values[0], orbi.objId)
            self.assertTrue(isinstance(orb, Orbits))
            self.assertEqual(orb.orbits.index, i)

    def testSlicing(self):
        """
        Test that we can slice a collection of orbits
        """
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsNEO.s3m'))
        orbit_slice = orbits[2:6]
        self.assertEqual(orbit_slice[0], orbits[2])
        self.assertEqual(orbit_slice[1], orbits[3])
        self.assertEqual(orbit_slice[2], orbits[4])
        self.assertEqual(orbit_slice[3], orbits[5])
        self.assertEqual(len(orbit_slice), 4)

        orbit_slice = orbits[1:7:2]
        self.assertEqual(orbit_slice[0], orbits[1])
        self.assertEqual(orbit_slice[1], orbits[3])
        self.assertEqual(orbit_slice[2], orbits[5])
        self.assertEqual(len(orbit_slice), 3)

    def testOffsetDataframe(self):
        """
        Test that we can slice and iterate through an orbits
        dataframe that has already been sub-selected from another
        dataframe.
        """
        orbits0 = Orbits()
        orbits0.readOrbits(os.path.join(self.testdir, 'test_orbitsNEO.s3m'))

        orbitsSub = Orbits()
        orbitsSub.setOrbits(orbits0.orbits.query('index>1'))

        self.assertEqual(len(orbitsSub), 6)

        orbit_slice = orbitsSub[2:6]
        self.assertEqual(orbit_slice[0], orbitsSub[2])
        self.assertEqual(orbit_slice[1], orbitsSub[3])
        self.assertEqual(orbit_slice[2], orbitsSub[4])
        self.assertEqual(orbit_slice[3], orbitsSub[5])
        self.assertEqual(len(orbit_slice), 4)

        orbit_slice = orbitsSub[1:5:2]
        self.assertEqual(orbit_slice[0], orbitsSub[1])
        self.assertEqual(orbit_slice[1], orbitsSub[3])
        self.assertEqual(len(orbit_slice), 2)

        for ii, oo in enumerate(orbitsSub):
            self.assertEqual(oo, orbits0[ii+2])

    def testReadOrbits(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        self.assertEqual(len(orbits), 4)
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsA.des'))
        self.assertEqual(len(orbits), 4)
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsCAR.des'))
        self.assertEqual(len(orbits), 1)
        with self.assertRaises(ValueError):
            orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsBadMix.des'))
        with self.assertRaises(ValueError):
            orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsBad.des'))

    def testSetOrbits(self):
        orbits = Orbits()
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        # Test that we can set the orbits using a dataframe.
        suborbits = orbits.orbits.head(1)
        newOrbits = Orbits()
        newOrbits.setOrbits(suborbits)
        self.assertEqual(len(newOrbits), 1)
        self.assertEqual(newOrbits.orb_format, 'COM')
        assert_frame_equal(newOrbits.orbits, suborbits)
        # Test that we can set the orbits using a Series.
        for i, sso in suborbits.iterrows():
            newOrbits = Orbits()
            newOrbits.setOrbits(sso)
            self.assertEqual(len(newOrbits), 1)
            self.assertEqual(newOrbits.orb_format, 'COM')
            assert_frame_equal(newOrbits.orbits, suborbits)
        # Test that we can set the orbits using a numpy array with many objects.
        numpyorbits = orbits.orbits.to_records(index=False)
        newOrbits = Orbits()
        newOrbits.setOrbits(numpyorbits)
        self.assertEqual(len(newOrbits), len(orbits))
        self.assertEqual(newOrbits.orb_format, 'COM')
        assert_frame_equal(newOrbits.orbits, orbits.orbits)
        # And test that this works for a single row of the numpy array.
        onenumpyorbits = numpyorbits[0]
        newOrbits = Orbits()
        newOrbits.setOrbits(onenumpyorbits)
        self.assertEqual(len(newOrbits), 1)
        self.assertEqual(newOrbits.orb_format, 'COM')
        assert_frame_equal(newOrbits.orbits, suborbits)
        # And test that it fails appropriately when columns are not correct.
        neworbits = pd.DataFrame(orbits.orbits)
        newcols = neworbits.columns.values.tolist()
        newcols[0] = 'ssmId'
        newcols[3] = 'ecc'
        neworbits.columns = newcols
        newOrbits = Orbits()
        with self.assertRaises(ValueError):
            newOrbits.setOrbits(neworbits)

    def testSetSeds(self):
        """
        Test that the self-assignment of SEDs works as expected.
        """
        orbits = Orbits()
        # Test with a range of a values.
        a = np.arange(0, 5, .05)
        orbs = pd.DataFrame(a, columns=['a'])
        seds = orbits.assignSed(orbs)
        self.assertEqual(np.unique(seds[np.where(a < 2)]), 'S.dat')
        self.assertEqual(np.unique(seds[np.where(a > 4)]), 'C.dat')
        # Test when read a values.
        orbits.readOrbits(os.path.join(self.testdir, 'test_orbitsA.des'))
        sedvals = orbits.assignSed(orbits.orbits, randomSeed=42)
        orbits2 = Orbits()
        orbits2.readOrbits(os.path.join(self.testdir, 'test_orbitsQ.des'))
        sedvals2 = orbits2.assignSed(orbits2.orbits, randomSeed=42)
        np.testing.assert_array_equal(sedvals, sedvals2)


if __name__ == "__main__":
    unittest.main()
