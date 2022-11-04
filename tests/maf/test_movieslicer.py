import matplotlib

matplotlib.use("Agg")
import numpy as np
import warnings
import unittest
from rubin_sim.maf.slicers import MovieSlicer, UniSlicer


def makeTimes(size=100, min=0.0, max=10.0, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random > 0:
        rng = np.random.RandomState(random)
        randorder = rng.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(list(zip(datavalues)), dtype=[("times", "float")])
    return datavalues


class TestMovieSlicerSetup(unittest.TestCase):
    def setUp(self):
        self.testslicer = MovieSlicer(
            slice_col_name="times", cumulative=False, forceNoFfmpeg=True
        )

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        self.assertEqual(self.testslicer.slicerName, self.testslicer.__class__.__name__)
        self.assertEqual(self.testslicer.slicerName, "MovieSlicer")

    def testSetupSlicerBins(self):
        """Test setting up slicer using defined bins."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.1)
        dv = makeTimes(nvalues, dvmin, dvmax, random=987)
        # Used right bins?
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, forceNoFfmpeg=True
        )
        self.testslicer.setup_slicer(dv)
        np.testing.assert_equal(self.testslicer.bins, bins)
        self.assertEqual(self.testslicer.nslice, len(bins) - 1)

    def testSetupSlicerBinsize(self):
        """Test setting up slicer using binsize."""
        dvmin = 0
        dvmax = 1
        dv = makeTimes(1000, dvmin, dvmax, random=1992)
        binsize = 0.1
        for cumulative in [True, False]:
            self.testslicer = MovieSlicer(
                slice_col_name="times",
                binsize=binsize,
                cumulative=cumulative,
                forceNoFfmpeg=True,
            )
            self.testslicer.setup_slicer(dv)
            # Bins of the right size?
            bindiff = np.diff(self.testslicer.bins)
            self.assertAlmostEqual(bindiff.max(), binsize)
            self.assertAlmostEqual(bindiff.min(), binsize)
        # Test that warning works.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.testslicer = MovieSlicer(
                slice_col_name="times",
                bins=200,
                binsize=binsize,
                cumulative=False,
                forceNoFfmpeg=True,
            )
            self.testslicer.setup_slicer(dv)
            # Verify some things
            self.assertIn("binsize", str(w[-1].message))

    def testSetupSlicerNbinsZeros(self):
        """Test what happens if give slicer test data that is all single-value."""
        dv = np.zeros(100, float)
        dv = np.array(list(zip(dv)), dtype=[("times", "float")])
        nbins = 10
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=nbins, cumulative=False, forceNoFfmpeg=True
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.testslicer.setup_slicer(dv)
            self.assertIn("creasing binMax", str(w[-1].message))
        self.assertEqual(self.testslicer.nslice, nbins)

    def testSetupSlicerLimits(self):
        """Test setting up slicer using binMin/Max."""
        binMin = 0
        binMax = 1
        nbins = 10
        dvmin = -0.5
        dvmax = 1.5
        dv = makeTimes(1000, dvmin, dvmax, random=1772)
        self.testslicer = MovieSlicer(
            slice_col_name="times",
            binMin=binMin,
            binMax=binMax,
            bins=nbins,
            cumulative=False,
            forceNoFfmpeg=True,
        )
        self.testslicer.setup_slicer(dv)
        self.assertAlmostEqual(self.testslicer.bins.min(), binMin)
        self.assertAlmostEqual(self.testslicer.bins.max(), binMax)

    def testIndexing(self):
        """Test iteration and indexing."""
        dvmin = 0
        dvmax = 1
        bins = np.arange(dvmin, dvmax + 0.05, 0.05)
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, forceNoFfmpeg=True
        )
        dv = makeTimes(1000, dvmin, dvmax, random=908223)
        self.testslicer.setup_slicer(dv)
        for i, (s, b) in enumerate(zip(self.testslicer, bins)):
            self.assertEqual(s["slicePoint"]["sid"], i)
            self.assertEqual(s["slicePoint"]["binLeft"], b)
            self.assertLessEqual(s["slicePoint"]["binRight"], bins[i + 1])
        for i in [0, len(self.testslicer) // 2, len(self.testslicer) - 1]:
            self.assertEqual(self.testslicer[i]["slicePoint"]["sid"], i)
            self.assertEqual(self.testslicer[i]["slicePoint"]["binLeft"], bins[i])

    def testEquivalence(self):
        """Test equals method."""
        # Note that two Movie slicers will be considered equal if they are both the same kind of
        # slicer AND have the same bins.
        # Set up self..
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.01)
        dv = makeTimes(nvalues, dvmin, dvmax, random=72031)
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, forceNoFfmpeg=True
        )
        self.testslicer.setup_slicer(dv)
        # Set up another slicer to match (same bins, although not the same data).
        dv2 = makeTimes(nvalues + 100, dvmin, dvmax, random=56221)
        testslicer2 = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, forceNoFfmpeg=True
        )
        testslicer2.setup_slicer(dv2)
        self.assertEqual(self.testslicer, testslicer2)
        # Set up another slicer that should not match (different bins)
        dv2 = makeTimes(nvalues, dvmin + 1, dvmax + 1, random=542093)
        testslicer2 = MovieSlicer(
            slice_col_name="times", bins=len(bins), cumulative=False, forceNoFfmpeg=True
        )
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)
        # Set up a different kind of slicer that should not match.
        dv2 = makeTimes(100, 0, 1, random=16)
        testslicer2 = UniSlicer()
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)

    def testSlicing(self):
        """Test slicing."""
        dvmin = 0
        dvmax = 1
        nbins = 100
        # Test that testbinner raises appropriate error before it's set up (first time)
        self.assertRaises(NotImplementedError, self.testslicer._slice_sim_data, 0)
        for nvalues in (100, 1000):
            dv = makeTimes(nvalues, dvmin, dvmax, random=82)
            # Test differential case.
            self.testslicer = MovieSlicer(
                slice_col_name="times", bins=nbins, cumulative=False, forceNoFfmpeg=True
            )
            self.testslicer.setup_slicer(dv)
            sum = 0
            for i, s in enumerate(self.testslicer):
                idxs = s["idxs"]
                dataslice = dv["times"][idxs]
                sum += len(idxs)
                if len(dataslice) > 0:
                    self.assertEqual(len(dataslice), nvalues / float(nbins))
                else:
                    raise ValueError(
                        "Data in test case expected to always be > 0 len after slicing"
                    )
            self.assertEqual(sum, nvalues)
            # And cumulative case.
            self.testslicer = MovieSlicer(
                slice_col_name="times", bins=nbins, cumulative=True, forceNoFfmpeg=True
            )
            self.testslicer.setup_slicer(dv)
            for i, s in enumerate(self.testslicer):
                idxs = s["idxs"]
                dataslice = dv["times"][idxs]
                self.assertGreater(len(dataslice), 0)


if __name__ == "__main__":
    unittest.main()
