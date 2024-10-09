import os
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from rubin_scheduler.data import get_data_dir

from rubin_sim.moving_objects import Orbits


class TestOrbits(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), "tests", "orbits_testdata")

    def test_equal_not_equal(self):
        orbits = Orbits()
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsQ.des"))
        self.assertEqual(len(orbits), 4)
        orbits2 = Orbits()
        orbits2.read_orbits(os.path.join(self.testdir, "test_orbitsQ.des"))
        self.assertEqual(orbits, orbits2)
        orbits3 = Orbits()
        orbits3.read_orbits(os.path.join(self.testdir, "test_orbitsA.des"))
        self.assertNotEqual(orbits, orbits3)

    def test_iteration_and_indexing(self):
        orbits = Orbits()
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsNEO.s3m"))
        orbits_single = orbits[0]
        assert_frame_equal(orbits_single.orbits, orbits.orbits.query("index==0"))
        orbits_single = orbits[3]
        assert_frame_equal(orbits_single.orbits, orbits.orbits.query("index==3"))
        # Test iteration through all orbits.
        for orb, (i, orbi) in zip(orbits, orbits.orbits.iterrows()):
            self.assertEqual(orb.orbits.obj_id.values[0], orbi.obj_id)
            self.assertTrue(isinstance(orb, Orbits))
            self.assertEqual(orb.orbits.index, i)
        # Test iteration through a subset of orbits.
        orbits_sub = Orbits()
        orbits_sub.set_orbits(orbits.orbits.query("index > 4"))
        for orb, (i, orbi) in zip(orbits_sub, orbits_sub.orbits.iterrows()):
            self.assertEqual(orb.orbits.obj_id.values[0], orbi.obj_id)
            self.assertTrue(isinstance(orb, Orbits))
            self.assertEqual(orb.orbits.index, i)

    def test_slicing(self):
        """
        Test that we can slice a collection of orbits
        """
        orbits = Orbits()
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsNEO.s3m"))
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

    def test_offset_dataframe(self):
        """
        Test that we can slice and iterate through an orbits
        dataframe that has already been sub-selected from another
        dataframe.
        """
        orbits0 = Orbits()
        orbits0.read_orbits(os.path.join(self.testdir, "test_orbitsNEO.s3m"))

        orbits_sub = Orbits()
        orbits_sub.set_orbits(orbits0.orbits.query("index>1"))

        self.assertEqual(len(orbits_sub), 6)

        orbit_slice = orbits_sub[2:6]
        self.assertEqual(orbit_slice[0], orbits_sub[2])
        self.assertEqual(orbit_slice[1], orbits_sub[3])
        self.assertEqual(orbit_slice[2], orbits_sub[4])
        self.assertEqual(orbit_slice[3], orbits_sub[5])
        self.assertEqual(len(orbit_slice), 4)

        orbit_slice = orbits_sub[1:5:2]
        self.assertEqual(orbit_slice[0], orbits_sub[1])
        self.assertEqual(orbit_slice[1], orbits_sub[3])
        self.assertEqual(len(orbit_slice), 2)

        for ii, oo in enumerate(orbits_sub):
            self.assertEqual(oo, orbits0[ii + 2])

    def test_read_orbits(self):
        orbits = Orbits()
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsQ.des"))
        self.assertEqual(len(orbits), 4)
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsA.des"))
        self.assertEqual(len(orbits), 4)
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsCAR.des"))
        self.assertEqual(len(orbits), 1)
        with self.assertRaises(ValueError):
            orbits.read_orbits(os.path.join(self.testdir, "test_orbitsBadMix.des"))
        with self.assertRaises(ValueError):
            orbits.read_orbits(os.path.join(self.testdir, "test_orbitsBad.des"))

    def test_set_orbits(self):
        orbits = Orbits()
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsQ.des"))
        # Test that we can set the orbits using a dataframe.
        suborbits = orbits.orbits.head(1)
        new_orbits = Orbits()
        new_orbits.set_orbits(suborbits)
        self.assertEqual(len(new_orbits), 1)
        self.assertEqual(new_orbits.orb_format, "COM")
        assert_frame_equal(new_orbits.orbits, suborbits)
        # Test that we can set the orbits using a Series.
        for i, sso in suborbits.iterrows():
            new_orbits = Orbits()
            new_orbits.set_orbits(sso)
            self.assertEqual(len(new_orbits), 1)
            self.assertEqual(new_orbits.orb_format, "COM")
            assert_frame_equal(new_orbits.orbits, suborbits)
        # Test that we can set the orbits using a numpy array of many objects.
        numpyorbits = orbits.orbits.to_records(index=False)
        new_orbits = Orbits()
        new_orbits.set_orbits(numpyorbits)
        self.assertEqual(len(new_orbits), len(orbits))
        self.assertEqual(new_orbits.orb_format, "COM")
        assert_frame_equal(new_orbits.orbits, orbits.orbits)
        # And test that this works for a single row of the numpy array.
        onenumpyorbits = numpyorbits[0]
        new_orbits = Orbits()
        new_orbits.set_orbits(onenumpyorbits)
        self.assertEqual(len(new_orbits), 1)
        self.assertEqual(new_orbits.orb_format, "COM")
        assert_frame_equal(new_orbits.orbits, suborbits)
        # And test that it fails appropriately when columns are not correct.
        neworbits = pd.DataFrame(orbits.orbits)
        newcols = neworbits.columns.values.tolist()
        newcols[0] = "ssmId"
        newcols[3] = "ecc"
        neworbits.columns = newcols
        new_orbits = Orbits()
        with self.assertRaises(ValueError):
            new_orbits.set_orbits(neworbits)

    def test_set_seds(self):
        """
        Test that the self-assignment of SEDs works as expected.
        """
        orbits = Orbits()
        # Test with a range of a values.
        a = np.arange(0, 5, 0.05)
        orbs = pd.DataFrame(a, columns=["a"])
        seds = orbits.assign_sed(orbs)
        self.assertEqual(np.unique(seds[np.where(a < 2)]), "S.dat")
        self.assertEqual(np.unique(seds[np.where(a > 4)]), "C.dat")
        # Test when read a values.
        orbits.read_orbits(os.path.join(self.testdir, "test_orbitsA.des"))
        sedvals = orbits.assign_sed(orbits.orbits, random_seed=42)
        orbits2 = Orbits()
        orbits2.read_orbits(os.path.join(self.testdir, "test_orbitsQ.des"))
        sedvals2 = orbits2.assign_sed(orbits2.orbits, random_seed=42)
        np.testing.assert_array_equal(sedvals, sedvals2)


if __name__ == "__main__":
    unittest.main()
