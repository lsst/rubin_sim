import unittest
import os
import warnings
import tempfile
import shutil
import numpy as np
from rubin_sim.moving_bjects import Orbits, ChebyFits
from rubin_sim.data import get_data_dir


ROOT = os.path.abspath(os.path.dirname(__file__))


class TestChebyFits(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), "tests", "orbits_testdata")
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="TestChebyFits-")
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, "test_orbitsMBA.s3m"))
        self.cheb = ChebyFits(
            self.orbits,
            54800,
            30,
            ngran=64,
            skyTolerance=2.5,
            nDecimal=10,
            nCoeff_position=14,
        )
        self.assertEqual(self.cheb.ngran, 64)

    def tearDown(self):
        del self.orbits
        del self.cheb
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def testPrecomputeMultipliers(self):
        # Precompute multipliers is done as an automatic step in __init__.
        # After setting up self.cheb, these multipliers should all exist.
        for key in self.cheb.nCoeff:
            self.assertTrue(key in self.cheb.multipliers)

    def testSetSegmentLength(self):
        # Expect MBAs with standard ngran and tolerance to have length ~2.0 days.
        self.cheb.calcSegmentLength()
        self.assertAlmostEqual(self.cheb.length, 2.0)
        # Test that we can set it to other values which fit into the 30 day window.
        self.cheb.calcSegmentLength(length=1.5)
        self.assertEqual(self.cheb.length, 1.5)
        # Test that we if we try to set it to a value which does not fit into the 30 day window,
        # that the actual value used is different - and smaller.
        self.cheb.calcSegmentLength(length=1.9)
        self.assertTrue(self.cheb.length < 1.9)
        # Test that we get a warning about the residuals if we try to set the length to be too long.
        with warnings.catch_warnings(record=True) as w:
            self.cheb.calcSegmentLength(length=5.0)
            self.assertTrue(len(w), 1)
        # Now check granularity works for other orbit types (which would have other standard lengths).
        # Check for multiple orbit types.
        for orbitFile in [
            "test_orbitsMBA.s3m",
            "test_orbitsOuter.s3m",
            "test_orbitsNEO.s3m",
        ]:
            self.orbits.readOrbits(os.path.join(self.testdir, orbitFile))
            tStart = self.orbits.orbits["epoch"].iloc[0]
            cheb = ChebyFits(self.orbits, tStart, 30, ngran=64, nDecimal=2)
            # And that we should converge for a variety of other tolerances.
            for skyTolerance in (2.5, 5.0, 10.0, 100.0, 1000.0, 20000.0):
                cheb.skyTolerance = skyTolerance
                cheb.calcSegmentLength()
                pos_resid, ratio = cheb._testResiduals(cheb.length)
                self.assertTrue(pos_resid < skyTolerance)
                self.assertEqual((cheb.length * 100) % 1, 0)
                # print('final', orbitFile, skyTolerance, pos_resid, cheb.length, ratio)
        # And check for challenging 'impactors'.
        for orbitFile in ["test_orbitsImpactors.s3m"]:
            self.orbits.readOrbits(os.path.join(self.testdir, orbitFile))
            tStart = self.orbits.orbits["epoch"].iloc[0]
            cheb = ChebyFits(self.orbits, tStart, 30, ngran=64, nDecimal=10)
            # And that we should converge for a variety of other tolerances.
            for skyTolerance in (2.5, 10.0, 100.0):
                cheb.skyTolerance = skyTolerance
                cheb.calcSegmentLength()
                pos_resid, ratio = cheb._testResiduals(cheb.length)
                self.assertTrue(pos_resid < skyTolerance)
                # print('final', orbitFile, skyTolerance, pos_resid, cheb.length, ratio)

    def testSegments(self):
        # Test that we can create segments.
        self.cheb.calcSegmentLength(length=1.0)
        times = self.cheb.makeAllTimes()
        self.cheb.generateEphemerides(times)
        self.cheb.calcSegments()
        # We expect calculated coefficients to have the following keys:
        coeffKeys = [
            "objId",
            "tStart",
            "tEnd",
            "ra",
            "dec",
            "geo_dist",
            "vmag",
            "elongation",
        ]
        for k in coeffKeys:
            self.assertTrue(k in self.cheb.coeffs.keys())
        # And in this case, we had a 30 day timespan with 1 day segments
        # (one day segments should be more than enough to meet 2.5mas tolerance, so not subdivided)
        self.assertEqual(len(self.cheb.coeffs["tStart"]), 30 * len(self.orbits))
        # And we used 14 coefficients for ra and dec.
        self.assertEqual(len(self.cheb.coeffs["ra"][0]), 14)
        self.assertEqual(len(self.cheb.coeffs["dec"][0]), 14)

    def testWrite(self):
        # Test that we can write the output to files.
        self.cheb.calcSegmentLength()
        self.cheb.generateEphemerides(self.cheb.makeAllTimes())
        self.cheb.calcSegments()
        coeff_name = os.path.join(self.scratch_dir, "coeff1.txt")
        resid_name = os.path.join(self.scratch_dir, "resid1.txt")
        failed_name = os.path.join(self.scratch_dir, "failed1.txt")
        self.cheb.write(coeff_name, resid_name, failed_name)
        self.assertTrue(os.path.isfile(coeff_name))
        self.assertTrue(os.path.isfile(resid_name))


class TestRun(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(get_data_dir(), "tests", "orbits_testdata")
        self.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="TestChebyFits-")
        self.orbits = Orbits()
        self.orbits.readOrbits(os.path.join(self.testdir, "test_orbitsMBA.s3m"))

    def tearDown(self):
        del self.orbits
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def testRunThrough(self):
        # Set up chebyshev fitter.
        tStart = self.orbits.orbits.epoch.iloc[0]
        interval = 30
        cheb = ChebyFits(
            self.orbits, tStart, interval, ngran=64, skyTolerance=2.5, nDecimal=10
        )
        # Set granularity. Use an value that will be too long, to trigger recursion below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cheb.calcSegmentLength(length=10.0)
        # Run through segments.
        cheb.calcSegments()
        self.assertEqual(len(np.unique(cheb.coeffs["objId"])), len(self.orbits))
        # Write outputs.
        coeff_name = os.path.join(self.scratch_dir, "coeff2.txt")
        resid_name = os.path.join(self.scratch_dir, "resid2.txt")
        failed_name = os.path.join(self.scratch_dir, "failed2.txt")
        cheb.write(coeff_name, resid_name, failed_name)
        # Test that the segments for each individual object fit together start/end.
        for k in cheb.coeffs:
            cheb.coeffs[k] = np.array(cheb.coeffs[k])
        for objId in np.unique(cheb.coeffs["objId"]):
            condition = cheb.coeffs["objId"] == objId
            te_prev = tStart
            for ts, te in zip(
                cheb.coeffs["tStart"][condition], cheb.coeffs["tEnd"][condition]
            ):
                # Test that the start of the current interval = the end of the previous interval.
                self.assertEqual(te_prev, ts)
                te_prev = te
        # Test that the end of the last interval is equal to the end of the total interval
        self.assertEqual(te, tStart + interval)


if __name__ == "__main__":
    unittest.main()
