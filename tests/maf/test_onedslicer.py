import matplotlib

matplotlib.use("Agg")
import numpy as np
import warnings
import unittest
from rubin_sim.maf.slicers import OneDSlicer, UniSlicer


def make_data_values(size=100, min=0.0, max=1.0, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random > 0:
        rng = np.random.RandomState(random)
        randorder = rng.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(list(zip(datavalues)), dtype=[("testdata", "float")])
    return datavalues


class TestOneDSlicerSetup(unittest.TestCase):
    def setUp(self):
        self.testslicer = OneDSlicer(slice_col_name="testdata")

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_slicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        self.assertEqual(
            self.testslicer.slicer_name, self.testslicer.__class__.__name__
        )
        self.assertEqual(self.testslicer.slicer_name, "OneDSlicer")

    def test_setup_slicer_bins(self):
        """Test setting up slicer using defined bins."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.1)
        dv = make_data_values(nvalues, dvmin, dvmax, random=4)
        # Used right bins?
        self.testslicer = OneDSlicer(slice_col_name="testdata", bins=bins)
        self.testslicer.setup_slicer(dv)
        np.testing.assert_equal(self.testslicer.bins, bins)
        self.assertEqual(self.testslicer.nslice, len(bins) - 1)

    def test_setup_slicer_nbins_zeros(self):
        """Test what happens if give slicer test data that is all single-value."""
        dv = np.zeros(100, float)
        dv = np.array(list(zip(dv)), dtype=[("testdata", "float")])
        binsize = 0.1
        self.testslicer = OneDSlicer(slice_col_name="testdata", binsize=binsize)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.testslicer.setup_slicer(dv)
            self.assertIn("creasing bin_max", str(w[-1].message))
        self.assertEqual(self.testslicer.nslice, 2)

    def test_setup_slicer_limits(self):
        """Test setting up slicer using bin_min/Max."""
        bin_min = 0
        bin_max = 1
        nbins = 10
        dvmin = -0.5
        dvmax = 1.5
        dv = make_data_values(1000, dvmin, dvmax, random=342)
        self.testslicer = OneDSlicer(
            slice_col_name="testdata", bin_min=bin_min, bin_max=bin_max
        )
        self.testslicer.setup_slicer(dv)
        self.assertAlmostEqual(self.testslicer.bins.min(), bin_min)
        self.assertAlmostEqual(self.testslicer.bins.max(), bin_max)

    def test_setup_slicer_binsize(self):
        """Test setting up slicer using binsize."""
        dvmin = 0
        dvmax = 1
        dv = make_data_values(1000, dvmin, dvmax, random=8977)
        # Test basic use.
        binsize = 0.5
        self.testslicer = OneDSlicer(slice_col_name="testdata", binsize=binsize)
        self.testslicer.setup_slicer(dv)
        self.assertEqual(self.testslicer.nslice, (dvmax - dvmin) / binsize)

    def test_setup_slicer_freedman(self):
        """Test that setting up the slicer using bins=None works."""
        dvmin = 0
        dvmax = 1
        dv = make_data_values(1000, dvmin, dvmax, random=2234)
        self.testslicer = OneDSlicer(slice_col_name="testdata", bins=None)
        self.testslicer.setup_slicer(dv)
        # How many bins do you expect from optimal binsize?
        from rubin_sim.maf.utils import optimal_bins

        bins = optimal_bins(dv["testdata"])
        np.testing.assert_equal(self.testslicer.nslice, bins)


class TestOneDSlicerIteration(unittest.TestCase):
    def setUp(self):
        self.testslicer = OneDSlicer(slice_col_name="testdata")
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        self.bins = np.arange(dvmin, dvmax, 0.01)
        dv = make_data_values(nvalues, dvmin, dvmax, random=5678)
        self.testslicer = OneDSlicer(slice_col_name="testdata", bins=self.bins)
        self.testslicer.setup_slicer(dv)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_iteration(self):
        """Test iteration."""
        for i, (s, b) in enumerate(zip(self.testslicer, self.bins)):
            self.assertEqual(s["slice_point"]["sid"], i)
            self.assertEqual(s["slice_point"]["binLeft"], b)

    def test_get_item(self):
        """Test that can return an individual indexed values of the slicer."""
        for i in [0, 10, 20]:
            self.assertEqual(self.testslicer[i]["slice_point"]["sid"], i)
            self.assertEqual(self.testslicer[i]["slice_point"]["binLeft"], self.bins[i])


class TestOneDSlicerEqual(unittest.TestCase):
    def setUp(self):
        self.testslicer = OneDSlicer(slice_col_name="testdata")

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_equivalence(self):
        """Test equals method."""
        # Note that two OneD slicers will be considered equal if they are both the same kind of
        # slicer AND have the same bins.
        # Set up self..
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.01)
        dv = make_data_values(nvalues, dvmin, dvmax, random=32499)
        # Set up reference
        self.testslicer = OneDSlicer(slice_col_name="testdata", bins=bins)
        self.testslicer.setup_slicer(dv)
        # Set up another slicer to match (same bins, although not the same data).
        dv2 = make_data_values(nvalues + 100, dvmin, dvmax, random=334)
        testslicer2 = OneDSlicer(slice_col_name="testdata", bins=bins)
        testslicer2.setup_slicer(dv2)
        self.assertTrue(self.testslicer == testslicer2)
        self.assertFalse(self.testslicer != testslicer2)
        # Set up another slicer that should not match (different bins, based on data)
        dv2 = make_data_values(nvalues, dvmin + 1, dvmax + 1, random=445)
        testslicer2 = OneDSlicer(slice_col_name="testdata", bins=None)
        testslicer2.setup_slicer(dv2)
        self.assertTrue(self.testslicer != testslicer2)
        self.assertFalse(self.testslicer == testslicer2)
        # Set up a different kind of slicer that should not match.
        dv2 = make_data_values(100, 0, 1, random=12)
        testslicer2 = UniSlicer()
        testslicer2.setup_slicer(dv2)
        self.assertTrue(self.testslicer != testslicer2)
        self.assertFalse(self.testslicer == testslicer2)
        # Get another oneDslicer that is not set up, and check equivalence.
        testslicer2 = OneDSlicer(slice_col_name="testdata")
        self.assertTrue(self.testslicer != testslicer2)
        self.assertFalse(self.testslicer == testslicer2)
        testslicer2 = OneDSlicer(
            slice_col_name="testdata", bin_min=0, bin_max=1, binsize=0.5
        )
        testslicer3 = OneDSlicer(
            slice_col_name="testdata", bin_min=0, bin_max=1, binsize=0.5
        )
        self.assertTrue(testslicer2 == testslicer3)
        self.assertFalse(testslicer2 != testslicer3)
        testslicer3 = OneDSlicer(slice_col_name="testdata", bin_min=0, bin_max=1)
        self.assertFalse(testslicer2 == testslicer3)
        self.assertTrue(testslicer2 != testslicer3)
        usebins = np.arange(0, 1, 0.1)
        testslicer2 = OneDSlicer(slice_col_name="testdata", bins=usebins)
        testslicer3 = OneDSlicer(slice_col_name="testdata", bins=usebins)
        self.assertTrue(testslicer2 == testslicer3)
        self.assertFalse(testslicer2 != testslicer3)
        testslicer3 = OneDSlicer(slice_col_name="testdata", bins=usebins + 1)
        self.assertFalse(testslicer2 == testslicer3)
        self.assertTrue(testslicer2 != testslicer3)


class TestOneDSlicerSlicing(unittest.TestCase):

    long_message = True

    def setUp(self):
        self.testslicer = OneDSlicer(slice_col_name="testdata")

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_slicing(self):
        """Test slicing."""
        dvmin = 0
        dvmax = 1
        stepsize = 0.01
        bins = np.arange(dvmin, dvmax + stepsize / 2, stepsize)
        nbins = len(bins) - 1
        # Test that testbinner raises appropriate error before it's set up (first time)
        self.assertRaises(NotImplementedError, self.testslicer._slice_sim_data, 0)
        for nvalues in (1000, 10000, 100000):
            dv = make_data_values(nvalues, dvmin, dvmax, random=560)
            self.testslicer = OneDSlicer(slice_col_name="testdata", bins=bins)
            self.testslicer.setup_slicer(dv)
            sum = 0
            for i, s in enumerate(self.testslicer):
                idxs = s["idxs"]
                dataslice = dv["testdata"][idxs]
                sum += len(idxs)
                if len(dataslice) > 0:
                    self.assertEqual(len(dataslice), nvalues / float(nbins))
                else:
                    self.assertGreater(
                        len(dataslice),
                        0,
                        msg="Data in test case expected to always be > 0 len after slicing",
                    )
            self.assertTrue(sum, nvalues)


if __name__ == "__main__":
    unittest.main()
