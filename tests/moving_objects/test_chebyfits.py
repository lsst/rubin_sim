import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.moving_objects import ChebyFits, Orbits

ROOT = os.path.abspath(os.path.dirname(__file__))


@unittest.skip("Temporary skip until ephemerides replaced")
class TestChebyFits(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), "tests", "orbits_testdata")
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="TestChebyFits-")
        self.orbits = Orbits()
        self.orbits.read_orbits(os.path.join(self.testdir, "test_orbitsMBA.s3m"))
        self.cheb = ChebyFits(
            self.orbits,
            54800,
            30,
            ngran=64,
            sky_tolerance=2.5,
            n_decimal=10,
            n_coeff_position=14,
        )
        self.assertEqual(self.cheb.ngran, 64)

    def tearDown(self):
        del self.orbits
        del self.cheb
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def test_precompute_multipliers(self):
        # Precompute multipliers is done as an automatic step in __init__.
        # After setting up self.cheb, these multipliers should all exist.
        for key in self.cheb.n_coeff:
            self.assertTrue(key in self.cheb.multipliers)

    def test_set_segment_length(self):
        # Expect MBAs with standard ngran and tolerance to have
        # length ~2.0 days.
        self.cheb.calc_segment_length()
        self.assertAlmostEqual(self.cheb.length, 2.0)
        # Test that we can set it to other values which fit into
        # the 30 day window.
        self.cheb.calc_segment_length(length=1.5)
        self.assertEqual(self.cheb.length, 1.5)
        # Test that we if we try to set it to a value which does not
        # fit into the 30 day window,
        # that the actual value used is different - and smaller.
        self.cheb.calc_segment_length(length=1.9)
        self.assertTrue(self.cheb.length < 1.9)
        # Test that we get a warning about the residuals if we try
        # to set the length to be too long.
        with warnings.catch_warnings(record=True) as w:
            self.cheb.calc_segment_length(length=5.0)
            self.assertTrue(len(w), 1)
        # Now check granularity works for other orbit types
        # (which would have other standard lengths).
        # Check for multiple orbit types.
        for orbit_file in [
            "test_orbitsMBA.s3m",
            "test_orbitsOuter.s3m",
            "test_orbitsNEO.s3m",
        ]:
            self.orbits.read_orbits(os.path.join(self.testdir, orbit_file))
            t_start = self.orbits.orbits["epoch"].iloc[0]
            cheb = ChebyFits(self.orbits, t_start, 30, ngran=64, n_decimal=2)
            # And that we should converge for a variety of other tolerances.
            for sky_tolerance in (2.5, 5.0, 10.0, 100.0, 1000.0, 20000.0):
                cheb.sky_tolerance = sky_tolerance
                cheb.calc_segment_length()
                pos_resid, ratio = cheb._test_residuals(cheb.length)
                self.assertTrue(pos_resid < sky_tolerance)
                self.assertEqual((cheb.length * 100) % 1, 0)
                # print('final', orbit_file, sky_tolerance, pos_resid,
                # cheb.length, ratio)
        # And check for challenging 'impactors'.
        for orbit_file in ["test_orbitsImpactors.s3m"]:
            self.orbits.read_orbits(os.path.join(self.testdir, orbit_file))
            t_start = self.orbits.orbits["epoch"].iloc[0]
            cheb = ChebyFits(self.orbits, t_start, 30, ngran=64, n_decimal=10)
            # And that we should converge for a variety of other tolerances.
            for sky_tolerance in (2.5, 10.0, 100.0):
                cheb.sky_tolerance = sky_tolerance
                cheb.calc_segment_length()
                pos_resid, ratio = cheb._test_residuals(cheb.length)
                self.assertTrue(pos_resid < sky_tolerance)
                # print('final', orbit_file, sky_tolerance, pos_resid,
                # cheb.length, ratio)

    @unittest.skip("Skipping because it has a strange platform-dependent failure")
    def test_segments(self):
        # Test that we can create segments.
        self.cheb.calc_segment_length(length=1.0)
        times = self.cheb.make_all_times()
        self.cheb.generate_ephemerides(times)
        self.cheb.calc_segments()
        # We expect calculated coefficients to have the following keys:
        coeff_keys = [
            "obj_id",
            "t_start",
            "t_end",
            "ra",
            "dec",
            "geo_dist",
            "vmag",
            "elongation",
        ]
        for k in coeff_keys:
            self.assertTrue(k in self.cheb.coeffs.keys())
        # And in this case, we had a 30 day timespan with 1 day segments
        # (one day segments should be more than enough to meet
        # 2.5mas tolerance, so not subdivided)
        self.assertEqual(len(self.cheb.coeffs["t_start"]), 30 * len(self.orbits))
        # And we used 14 coefficients for ra and dec.
        self.assertEqual(len(self.cheb.coeffs["ra"][0]), 14)
        self.assertEqual(len(self.cheb.coeffs["dec"][0]), 14)

    def test_write(self):
        # Test that we can write the output to files.
        self.cheb.calc_segment_length()
        self.cheb.generate_ephemerides(self.cheb.make_all_times())
        self.cheb.calc_segments()
        coeff_name = os.path.join(self.scratch_dir, "coeff1.txt")
        resid_name = os.path.join(self.scratch_dir, "resid1.txt")
        failed_name = os.path.join(self.scratch_dir, "failed1.txt")
        self.cheb.write(coeff_name, resid_name, failed_name)
        self.assertTrue(os.path.isfile(coeff_name))
        self.assertTrue(os.path.isfile(resid_name))


@unittest.skip("Temporary skip until ephemerides replaced")
class TestRun(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), "tests", "orbits_testdata")
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="TestChebyFits-")
        self.orbits = Orbits()
        self.orbits.read_orbits(os.path.join(self.testdir, "test_orbitsMBA.s3m"))

    def tearDown(self):
        del self.orbits
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def test_run_through(self):
        # Set up chebyshev fitter.
        t_start = self.orbits.orbits.epoch.iloc[0]
        interval = 30
        cheb = ChebyFits(self.orbits, t_start, interval, ngran=64, sky_tolerance=2.5, n_decimal=10)
        # Set granularity. Use an value that will be too long,
        # to trigger recursion below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cheb.calc_segment_length(length=10.0)
        # Run through segments.
        cheb.calc_segments()
        self.assertEqual(len(np.unique(cheb.coeffs["obj_id"])), len(self.orbits))
        # Write outputs.
        coeff_name = os.path.join(self.scratch_dir, "coeff2.txt")
        resid_name = os.path.join(self.scratch_dir, "resid2.txt")
        failed_name = os.path.join(self.scratch_dir, "failed2.txt")
        cheb.write(coeff_name, resid_name, failed_name)
        # Test that the segments for each individual object fit
        # together start/end.
        for k in cheb.coeffs:
            cheb.coeffs[k] = np.array(cheb.coeffs[k])
        for obj_id in np.unique(cheb.coeffs["obj_id"]):
            condition = cheb.coeffs["obj_id"] == obj_id
            te_prev = t_start
            for ts, te in zip(cheb.coeffs["t_start"][condition], cheb.coeffs["t_end"][condition]):
                # Test that the start of the current interval =
                # the end of the previous interval.
                self.assertEqual(te_prev, ts)
                te_prev = te
        # Test that the end of the last interval is equal to the end
        # of the total interval
        self.assertEqual(te, t_start + interval)


if __name__ == "__main__":
    unittest.main()
