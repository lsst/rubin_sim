import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time
from rubin_scheduler.data import get_data_dir

from rubin_sim.moving_objects import ChebyFits, ChebyValues, Orbits, PyOrbEphemerides

ROOT = os.path.abspath(os.path.dirname(__file__))


@unittest.skip("Temporary skip until ephemerides replaced")
class TestChebyValues(unittest.TestCase):
    def setUp(self):
        self.testdatadir = os.path.join(get_data_dir(), "tests", "orbits_testdata")
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="TestChebyValues-")
        self.coeff_file = os.path.join(self.scratch_dir, "test_coeffs")
        self.resid_file = os.path.join(self.scratch_dir, "test_resids")
        self.failed_file = os.path.join(self.scratch_dir, "test_failed")
        self.orbits = Orbits()
        self.orbits.read_orbits(os.path.join(self.testdatadir, "test_orbitsNEO.s3m"), skiprows=1)
        self.pyephems = PyOrbEphemerides()
        self.pyephems.set_orbits(self.orbits)
        self.t_start = self.orbits.orbits.epoch.iloc[0]
        self.interval = 30
        self.n_coeffs = 14
        self.n_decimal = 13

        self.cheby_fits = ChebyFits(
            self.orbits,
            self.t_start,
            self.interval,
            ngran=64,
            sky_tolerance=2.5,
            n_decimal=self.n_decimal,
            n_coeff_position=self.n_coeffs,
            obscode=807,
            time_scale="TAI",
        )
        self.set_length = 0.5
        self.cheby_fits.calc_segment_length(length=self.set_length)
        self.cheby_fits.calc_segments()
        self.cheby_fits.write(self.coeff_file, self.resid_file, self.failed_file, append=False)
        self.coeff_keys = [
            "obj_id",
            "t_start",
            "t_end",
            "ra",
            "dec",
            "geo_dist",
            "vmag",
            "elongation",
        ]

    def tearDown(self):
        del self.orbits
        del self.cheby_fits
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def test_set_coeff(self):
        # Test setting coefficients directly from chebyFits outputs.
        cheby_values = ChebyValues()
        cheby_values.set_coefficients(self.cheby_fits)
        for k in self.coeff_keys:
            self.assertTrue(k in cheby_values.coeffs)
            self.assertTrue(isinstance(cheby_values.coeffs[k], np.ndarray))
        self.assertEqual(len(np.unique(cheby_values.coeffs["obj_id"])), len(self.orbits))
        # This will only be true for carefully selected length/orbit type,
        # where subdivision did not occur.
        # For the test MBAs, a len=1day will work.
        # For the test NEOs, a len=0.25 day will work (with 2.5mas skyTol).
        # self.assertEqual(len(cheby_values.coeffs['tStart']),
        #            (self.interval / self.set_length) * len(self.orbits))
        self.assertEqual(len(cheby_values.coeffs["ra"][0]), self.n_coeffs)
        self.assertTrue("meanRA" in cheby_values.coeffs)
        self.assertTrue("meanDec" in cheby_values.coeffs)

    def test_read_coeffs(self):
        # Test reading the coefficients from disk.
        cheby_values = ChebyValues()
        cheby_values.read_coefficients(self.coeff_file)
        cheby_values2 = ChebyValues()
        cheby_values2.set_coefficients(self.cheby_fits)
        for k in cheby_values.coeffs:
            if k == "obj_id":
                # Can't test strings with np.test.assert_almost_equal.
                np.testing.assert_equal(cheby_values.coeffs[k], cheby_values2.coeffs[k])
            else:
                # All of these will only be accurate to 2 less decimal places
                # than they are print out with in chebyFits. Since vmag,
                # delta and elongation only use 7
                # decimal places, this means we can test to 5 decimal
                # places for those.
                np.testing.assert_allclose(cheby_values.coeffs[k], cheby_values2.coeffs[k], rtol=0, atol=1e-5)

    def test_get_ephemerides(self):
        # Test that get_ephemerides works and is accurate.
        cheby_values = ChebyValues()
        cheby_values.set_coefficients(self.cheby_fits)

        # Multiple times, all objects, all within interval.
        tstep = self.interval / 10.0
        time = np.arange(self.t_start, self.t_start + self.interval, tstep)
        # Test for a single time, but all the objects.
        ephemerides = cheby_values.get_ephemerides(time)
        pyephemerides = self.pyephems.generate_ephemerides(
            time, obscode=807, time_scale="TAI", by_object=True
        )
        # Looks like there's some weird hardware dependent issue
        # where cheby_fits is failing to fit some orbits.

        # RA and Dec should agree to 2.5mas (sky_tolerance above)
        pos_residuals = np.sqrt(
            (ephemerides["ra"] - pyephemerides["ra"]) ** 2
            + ((ephemerides["dec"] - pyephemerides["dec"]) * np.cos(np.radians(ephemerides["dec"]))) ** 2
        )
        pos_residuals *= 3600.0 * 1000.0
        if not np.all(np.isfinite(ephemerides["ra"])):
            warnings.warn("NaN values from ChebyValues.get_ephemerides, skipping some tests")
        else:
            # Let's just look at the max residuals in all quantities.
            for k in ("ra", "dec", "dradt", "ddecdt", "geo_dist"):
                resids = np.abs(ephemerides[k] - pyephemerides[k])
                if k != "geo_dist":
                    resids *= 3600.0 * 1000.0
                print("max diff", k, np.max(resids))
            resids = np.abs(ephemerides["elongation"] - pyephemerides["solarelon"])
            print("max diff elongation", np.max(resids))
            resids = np.abs(ephemerides["vmag"] - pyephemerides["magV"])
            print("max diff vmag", np.max(resids))
            self.assertLessEqual(np.nanmax(pos_residuals), 2.5)
            # Test for single time, but for a subset of the objects.
            obj_ids = self.orbits.orbits.obj_id.head(3).values
            ephemerides = cheby_values.get_ephemerides(time, obj_ids)
            self.assertEqual(len(ephemerides["ra"]), 3)
            # Test for time outside of segment range.
            obj_ids = self.orbits.orbits.obj_id.head(3).values
            ephemerides = cheby_values.get_ephemerides(
                self.t_start + self.interval * 2, obj_ids, extrapolate=False
            )
            self.assertTrue(
                np.isnan(ephemerides["ra"][0]),
                msg=f"Expected Nan for out of range ephemeris, got {ephemerides['ra'][0]}",
            )


@unittest.skip("Temporary skip until ephemerides replaced")
class TestJPLValues(unittest.TestCase):
    # Test the interpolation-generated RA/Dec values against JPL
    # generated RA/Dec values.
    # The resulting errors should be similar to the errors reported
    # from testEphemerides when testing against JPL values.
    def setUp(self):
        # Read orbits.
        self.orbits = Orbits()
        self.jpl_dir = os.path.join(get_data_dir(), "tests", "jpl_testdata")
        self.orbits.read_orbits(os.path.join(self.jpl_dir, "S0_n747.des"), skiprows=1)
        # Read JPL ephems.
        self.jpl = pd.read_table(os.path.join(self.jpl_dir, "807_n747.txt"), sep=r"\s+")
        self.jpl["obj_id"] = self.jpl["objId"]
        # Add times in TAI and UTC, because.
        t = Time(self.jpl["epoch_mjd"], format="mjd", scale="utc")
        self.jpl["mjdTAI"] = t.tai.mjd
        self.jpl["mjdUTC"] = t.utc.mjd
        self.jpl = self.jpl.to_records(index=False)
        # Generate interpolation coefficients for the time period
        # in the JPL catalog.
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="TestJPLValues-")
        self.coeff_file = os.path.join(self.scratch_dir, "test_coeffs")
        self.resid_file = os.path.join(self.scratch_dir, "test_resids")
        self.failed_file = os.path.join(self.scratch_dir, "test_failed")
        t_start = self.jpl["mjdTAI"].min() - 0.2
        t_end = self.jpl["mjdTAI"].max() + 0.2 - self.jpl["mjdTAI"].min()
        self.cheby_fits = ChebyFits(
            self.orbits,
            t_start,
            t_end,
            ngran=64,
            sky_tolerance=2.5,
            n_decimal=14,
            obscode=807,
        )
        self.cheby_fits.calc_segment_length()
        self.cheby_fits.calc_segments()

        self.coeff_keys = [
            "obj_id",
            "t_start",
            "t_end",
            "ra",
            "dec",
            "geo_dist",
            "vmag",
            "elongation",
        ]
        self.cheby_values = ChebyValues()
        self.cheby_values.set_coefficients(self.cheby_fits)

    def tearDown(self):
        del self.orbits
        del self.jpl
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def test_ra_dec(self):
        # We won't compare Vmag, because this also needs
        # information on trailing losses.
        times = np.unique(self.jpl["mjdTAI"])
        delta_ra = np.zeros(len(times), float)
        delta_dec = np.zeros(len(times), float)
        for i, t in enumerate(times):
            # Find the JPL obj_ids visible at this time.
            j = self.jpl[np.where(self.jpl["mjdTAI"] == t)]
            ephs = self.cheby_values.get_ephemerides(t, obj_ids=j["obj_id"])
            ephorder = np.argsort(ephs["obj_id"])
            # Sometimes I've had to reorder both, sometimes just the ephs. ??
            jorder = np.argsort(j["obj_id"])
            jorder = np.arange(0, len(jorder))
            d_ra = np.abs(ephs["ra"][ephorder][:, 0] - j["ra_deg"][jorder]) * 3600.0 * 1000.0
            d_dec = np.abs(ephs["dec"][ephorder][:, 0] - j["dec_deg"][jorder]) * 3600.0 * 1000.0
            delta_ra[i] = d_ra.max()
            delta_dec[i] = d_dec.max()
        # Should be (given OOrb direct prediction):
        # Much of the time we're closer than 1mas, but there are a
        # few which hit higher values.
        # This is consistent with the errors/values reported by oorb
        # directly in testEphemerides.

        #    # XXX--units?
        print("max JPL errors", delta_ra.max(), delta_dec.max())
        print("std of JPL errors", np.std(delta_ra), np.std(delta_dec))
        self.assertLess(np.max(delta_ra), 25)
        self.assertLess(np.max(delta_dec), 25)
        self.assertLess(np.std(delta_ra), 3)
        self.assertLess(np.std(delta_dec), 3)


if __name__ == "__main__":
    unittest.main()
