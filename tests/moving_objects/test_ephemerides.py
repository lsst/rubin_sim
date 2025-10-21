import os
import unittest

import numpy as np
import pandas as pd
from astropy.time import Time
from rubin_scheduler.data import get_data_dir

from rubin_sim.moving_objects import Orbits, PyOrbEphemerides


@unittest.skip("Temporary skip until ephemerides replaced")
class TestPyOrbEphemerides(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), "tests", "orbits_testdata")
        self.orbits = Orbits()
        self.orbits.read_orbits(os.path.join(self.testdir, "test_orbitsQ.des"))
        self.orbits_kep = Orbits()
        self.orbits_kep.read_orbits(os.path.join(self.testdir, "test_orbitsA.des"))
        self.ephems = PyOrbEphemerides()
        self.ephems.set_orbits(self.orbits)
        self.len_ephems_basic = 11
        self.len_ephems_full = 34

    def tear_down(self):
        del self.orbits
        del self.orbits_kep
        del self.ephems

    def test_which_pyoorb(self):
        import pyoorb

        print(pyoorb.__file__)

    def test_set_orbits(self):
        # Test that we can set orbits.
        self.ephems.set_orbits(self.orbits)
        # Test that setting with an empty orbit object fails.
        # (Avoids hard-to-interpret errors from pyoorb).
        with self.assertRaises(ValueError):
            empty_orb = Orbits()
            empty = pd.DataFrame([], columns=self.orbits.data_cols["KEP"])
            empty_orb.set_orbits(empty)
            self.ephems.set_orbits(empty_orb)

    def test_convert_to_oorb_array(self):
        # Check that orbital elements are converted.
        self.ephems._convert_to_oorb_elem(self.orbits.orbits, self.orbits.orb_format)
        self.assertEqual(len(self.ephems.oorb_elem), len(self.orbits))
        self.assertEqual(self.ephems.oorb_elem[0][7], 2)
        self.assertEqual(self.ephems.oorb_elem[0][9], 3)
        self.assertEqual(self.ephems.oorb_elem[0][1], self.orbits.orbits["q"][0])
        # Test that we can convert KEP orbital elements too.
        self.ephems._convert_to_oorb_elem(self.orbits_kep.orbits, self.orbits_kep.orb_format)
        self.assertEqual(len(self.ephems.oorb_elem), len(self.orbits_kep))
        self.assertEqual(self.ephems.oorb_elem[0][7], 3)
        self.assertEqual(self.ephems.oorb_elem[0][1], self.orbits_kep.orbits["a"][0])

    def test_convert_from_oorb_array(self):
        # Check that we can convert orbital elements TO oorb format and back
        # without losing info
        # (except ObjId -- we will lose that unless we use updateOrbits.)
        self.ephems._convert_to_oorb_elem(self.orbits.orbits, self.orbits.orb_format)
        new_orbits = Orbits()
        new_orbits.set_orbits(self.orbits.orbits)
        new_orbits.update_orbits(self.ephems.convert_from_oorb_elem())
        self.assertEqual(new_orbits, self.orbits)

    def test_convert_times(self):
        times = np.arange(49353, 49353 + 10, 0.5)
        eph_times = self.ephems._convert_times(times, "UTC")
        # Check that shape of eph_times is correct. (times x 2)
        self.assertEqual(eph_times.shape[0], len(times))
        self.assertEqual(eph_times.shape[1], 2)
        # Check that 'timescale' for eph_times is correct.
        self.assertEqual(eph_times[0][1], 1)
        eph_times = self.ephems._convert_times(times, "TAI")
        self.assertEqual(eph_times[0][1], 4)

    def test_oorb_ephemeris(self):
        self.ephems.set_orbits(self.orbits)
        times = np.arange(49353, 49353 + 3, 0.25)
        eph_times = self.ephems._convert_times(times)
        # Basic ephemerides.
        oorb_ephs = self.ephems._generate_oorb_ephs_basic(eph_times, obscode=807, eph_mode="N")
        # Check that it returned the right sort of array.
        self.assertEqual(
            oorb_ephs.shape,
            (len(self.ephems.oorb_elem), len(times), self.len_ephems_basic),
        )
        # Full ephemerides
        oorb_ephs = self.ephems._generate_oorb_ephs_full(eph_times, obscode=807, eph_mode="N")
        # Check that it returned the right sort of array.
        self.assertEqual(
            oorb_ephs.shape,
            (len(self.ephems.oorb_elem), len(times), self.len_ephems_full),
        )

    def test_ephemeris(self):
        # Calculate and convert ephemerides.
        self.ephems.set_orbits(self.orbits)
        times = np.arange(49353, 49353 + 2, 0.3)
        eph_times = self.ephems._convert_times(times)
        oorb_ephs = self.ephems._generate_oorb_ephs_basic(eph_times, obscode=807)
        # Group by object, and check grouping.
        ephs = self.ephems._convert_oorb_ephs_basic(oorb_ephs, by_object=True)
        self.assertEqual(len(ephs), len(self.orbits))
        # Group by time, and check grouping.
        oorb_ephs = self.ephems._generate_oorb_ephs_basic(eph_times, obscode=807)
        ephs = self.ephems._convert_oorb_ephs_basic(oorb_ephs, by_object=False)
        self.assertEqual(len(ephs), len(times))
        # And test all-wrapped-up method:
        ephs_all = self.ephems.generate_ephemerides(
            times,
            obscode=807,
            eph_mode="N",
            eph_type="basic",
            time_scale="UTC",
            by_object=False,
        )
        # See https://rubinobs.atlassian.net/browse/SP-1633
        # This needs to be fixed, but on a separate ticket
        # for key in ephs_all.dtype.names:
        #    np.testing.assert_almost_equal(ephs_all[key], ephs[key])

        # Reset ephems to use KEP Orbits, and calculate new ephemerides.
        self.ephems.set_orbits(self.orbits_kep)
        oorb_ephs = self.ephems._generate_oorb_ephs_basic(eph_times, obscode=807, eph_mode="N")
        ephs_kep = self.ephems._convert_oorb_ephs_basic(oorb_ephs, by_object=True)
        self.assertEqual(len(ephs_kep), len(self.orbits_kep))
        oorb_ephs = self.ephems._generate_oorb_ephs_basic(eph_times, obscode=807, eph_mode="N")
        ephs_kep = self.ephems._convert_oorb_ephs_basic(oorb_ephs, by_object=False)
        self.assertEqual(len(ephs_kep), len(times))
        # And test all-wrapped-up method:
        ephs_all_kep = self.ephems.generate_ephemerides(
            times,
            obscode=807,
            eph_mode="N",
            eph_type="basic",
            time_scale="UTC",
            by_object=False,
        )
        # Also https://rubinobs.atlassian.net/browse/SP-1633
        # for key in ephs_all_kep.dtype.names:
        #    np.testing.assert_almost_equal(ephs_all_kep[key], ephs_kep[key])

        # Check that ephemerides calculated from the different (COM/KEP)
        # orbits are almost equal
        for column in ephs.dtype.names:
            np.testing.assert_allclose(ephs[column], ephs_kep[column], rtol=1e-5, atol=1e-4)
        # Check that the wrapped method using KEP elements and the wrapped
        # method using COM elements match.
        for column in ephs_all.dtype.names:
            np.testing.assert_allclose(ephs_all_kep[column], ephs_all[column], rtol=1e-5, atol=1e-4)


@unittest.skip("Temporary skip until ephemerides replaced")
class TestJPLValues(unittest.TestCase):
    """Test the oorb generated RA/Dec values against
    JPL generated RA/Dec values."""

    def setUp(self):
        # Read orbits.
        self.orbits = Orbits()
        self.jpl_dir = os.path.join(get_data_dir(), "tests", "jpl_testdata")
        self.orbits.read_orbits(os.path.join(self.jpl_dir, "S0_n747.des"), skiprows=1)
        # Read JPL ephems.
        self.jpl = pd.read_csv(os.path.join(self.jpl_dir, "807_n747.txt"), sep=r"\s+")
        # Temp key fix
        self.jpl["obj_id"] = self.jpl["objId"]
        # Add times in TAI and UTC, because.
        t = Time(self.jpl["epoch_mjd"], format="mjd", scale="utc")
        self.jpl["mjdTAI"] = t.tai.mjd
        self.jpl["mjdUTC"] = t.utc.mjd

    def tear_down(self):
        del self.orbits
        del self.jpl

    def test_ra_dec(self):
        # We won't compare Vmag, because this also needs information
        # on trailing losses.
        times = self.jpl["mjdUTC"].unique()
        delta_ra = np.zeros(len(times), float)
        delta_dec = np.zeros(len(times), float)
        for i, t in enumerate(times):
            # Find the JPL obj_ids visible at this time.
            j = self.jpl.query("mjdUTC == @t").sort_values("obj_id")
            # Set the ephems, using the objects seen at this time.
            suborbits = self.orbits.orbits.query("obj_id in @j.obj_id").sort_values("obj_id")
            sub_orbits = Orbits()
            sub_orbits.set_orbits(suborbits)
            ephems = PyOrbEphemerides()
            ephems.set_orbits(sub_orbits)
            ephs = ephems.generate_ephemerides(
                [t],
                time_scale="UTC",
                obscode=807,
                eph_mode="N",
                eph_type="Basic",
                by_object=False,
            )
            delta_ra[i] = np.abs(ephs["ra"] - j["ra_deg"].values).max()
            delta_dec[i] = np.abs(ephs["dec"] - j["dec_deg"].values).max()
        # Convert to mas
        delta_ra *= 3600.0 * 1000.0
        delta_dec *= 3600.0 * 1000.0
        # Much of the time we're closer than 1mas,
        # but there are a few which hit higher values.
        print("max JPL errors", np.max(delta_ra), np.max(delta_dec))
        print("std JPL errors", np.std(delta_ra), np.std(delta_dec))
        self.assertLess(np.max(delta_ra), 25)
        self.assertLess(np.max(delta_dec), 25)
        self.assertLess(np.std(delta_ra), 3)
        self.assertLess(np.std(delta_dec), 3)


if __name__ == "__main__":
    unittest.main()
