import unittest
import warnings

import numpy as np

from rubin_sim.maf.slicers import MovieSlicer, UniSlicer


def make_times(size=100, min=0.0, max=10.0, random=-1):
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
        self.testslicer = MovieSlicer(slice_col_name="times", cumulative=False, force_no_ffmpeg=True)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_slicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        self.assertEqual(self.testslicer.slicer_name, self.testslicer.__class__.__name__)
        self.assertEqual(self.testslicer.slicer_name, "MovieSlicer")

    def test_setup_slicer_bins(self):
        """Test setting up slicer using defined bins."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.1)
        dv = make_times(nvalues, dvmin, dvmax, random=987)
        # Used right bins?
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, force_no_ffmpeg=True
        )
        self.testslicer.setup_slicer(dv)
        np.testing.assert_equal(self.testslicer.bins, bins)
        self.assertEqual(self.testslicer.nslice, len(bins) - 1)

    def test_setup_slicer_binsize(self):
        """Test setting up slicer using bin_size."""
        dvmin = 0
        dvmax = 1
        dv = make_times(1000, dvmin, dvmax, random=1992)
        bin_size = 0.1
        for cumulative in [True, False]:
            self.testslicer = MovieSlicer(
                slice_col_name="times",
                bin_size=bin_size,
                cumulative=cumulative,
                force_no_ffmpeg=True,
            )
            self.testslicer.setup_slicer(dv)
            # Bins of the right size?
            bindiff = np.diff(self.testslicer.bins)
            self.assertAlmostEqual(bindiff.max(), bin_size)
            self.assertAlmostEqual(bindiff.min(), bin_size)
        # Test that warning works.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.testslicer = MovieSlicer(
                slice_col_name="times",
                bins=200,
                bin_size=bin_size,
                cumulative=False,
                force_no_ffmpeg=True,
            )
            self.testslicer.setup_slicer(dv)
            # Verify some things
            self.assertIn("bin_size", str(w[-1].message))

    def test_setup_slicer_nbins_zeros(self):
        """Test what happens if give slicer test data that is all
        single-value."""
        dv = np.zeros(100, float)
        dv = np.array(list(zip(dv)), dtype=[("times", "float")])
        nbins = 10
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=nbins, cumulative=False, force_no_ffmpeg=True
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.testslicer.setup_slicer(dv)
            self.assertIn("creasing bin_max", str(w[-1].message))
        self.assertEqual(self.testslicer.nslice, nbins)

    def test_setup_slicer_limits(self):
        """Test setting up slicer using bin_min/Max."""
        bin_min = 0
        bin_max = 1
        nbins = 10
        dvmin = -0.5
        dvmax = 1.5
        dv = make_times(1000, dvmin, dvmax, random=1772)
        self.testslicer = MovieSlicer(
            slice_col_name="times",
            bin_min=bin_min,
            bin_max=bin_max,
            bins=nbins,
            cumulative=False,
            force_no_ffmpeg=True,
        )
        self.testslicer.setup_slicer(dv)
        self.assertAlmostEqual(self.testslicer.bins.min(), bin_min)
        self.assertAlmostEqual(self.testslicer.bins.max(), bin_max)

    def test_indexing(self):
        """Test iteration and indexing."""
        dvmin = 0
        dvmax = 1
        bins = np.arange(dvmin, dvmax + 0.05, 0.05)
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, force_no_ffmpeg=True
        )
        dv = make_times(1000, dvmin, dvmax, random=908223)
        self.testslicer.setup_slicer(dv)
        for i, (s, b) in enumerate(zip(self.testslicer, bins)):
            self.assertEqual(s["slice_point"]["sid"], i)
            self.assertEqual(s["slice_point"]["bin_left"], b)
            self.assertLessEqual(s["slice_point"]["bin_right"], bins[i + 1])
        for i in [0, len(self.testslicer) // 2, len(self.testslicer) - 1]:
            self.assertEqual(self.testslicer[i]["slice_point"]["sid"], i)
            self.assertEqual(self.testslicer[i]["slice_point"]["bin_left"], bins[i])

    def test_equivalence(self):
        """Test equals method."""
        # Note that two Movie slicers will be considered equal
        # if they are both the same kind of
        # slicer AND have the same bins.
        # Set up self..
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.01)
        dv = make_times(nvalues, dvmin, dvmax, random=72031)
        self.testslicer = MovieSlicer(
            slice_col_name="times", bins=bins, cumulative=False, force_no_ffmpeg=True
        )
        self.testslicer.setup_slicer(dv)
        # Set up another slicer to match
        # (same bins, although not the same data).
        dv2 = make_times(nvalues + 100, dvmin, dvmax, random=56221)
        testslicer2 = MovieSlicer(slice_col_name="times", bins=bins, cumulative=False, force_no_ffmpeg=True)
        testslicer2.setup_slicer(dv2)
        self.assertEqual(self.testslicer, testslicer2)
        # Set up another slicer that should not match (different bins)
        dv2 = make_times(nvalues, dvmin + 1, dvmax + 1, random=542093)
        testslicer2 = MovieSlicer(
            slice_col_name="times",
            bins=len(bins),
            cumulative=False,
            force_no_ffmpeg=True,
        )
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)
        # Set up a different kind of slicer that should not match.
        dv2 = make_times(100, 0, 1, random=16)
        testslicer2 = UniSlicer()
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)

    def test_slicing(self):
        """Test slicing."""
        dvmin = 0
        dvmax = 1
        nbins = 100
        # Test that testbinner raises appropriate error before it's set up
        # (first time)
        self.assertRaises(NotImplementedError, self.testslicer._slice_sim_data, 0)
        for nvalues in (100, 1000):
            dv = make_times(nvalues, dvmin, dvmax, random=82)
            # Test differential case.
            self.testslicer = MovieSlicer(
                slice_col_name="times",
                bins=nbins,
                cumulative=False,
                force_no_ffmpeg=True,
            )
            self.testslicer.setup_slicer(dv)
            sum = 0
            for i, s in enumerate(self.testslicer):
                idxs = s["idxs"]
                data_slice = dv["times"][idxs]
                sum += len(idxs)
                if len(data_slice) > 0:
                    self.assertEqual(len(data_slice), nvalues / float(nbins))
                else:
                    raise ValueError("Data in test case expected to always be > 0 len after slicing")
            self.assertEqual(sum, nvalues)
            # And cumulative case.
            self.testslicer = MovieSlicer(
                slice_col_name="times",
                bins=nbins,
                cumulative=True,
                force_no_ffmpeg=True,
            )
            self.testslicer.setup_slicer(dv)
            for i, s in enumerate(self.testslicer):
                idxs = s["idxs"]
                data_slice = dv["times"][idxs]
                self.assertGreater(len(data_slice), 0)


if __name__ == "__main__":
    unittest.main()
