import unittest

import numpy as np

from rubin_sim.maf.slicers import OneDSlicer, UniSlicer


def make_data_values(size=100, min=0.0, max=1.0, random=-1):
    """Generate a simple array of numbers,
    evenly arranged between min/max, but (optional) random order.
    """
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


class TestUniSlicerSetupAndSlice(unittest.TestCase):
    def setUp(self):
        self.testslicer = UniSlicer()

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_slicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        self.assertEqual(self.testslicer.slicer_name, self.testslicer.__class__.__name__)
        self.assertEqual(self.testslicer.slicer_name, "UniSlicer")

    def test_slicer_nbins(self):
        self.assertEqual(self.testslicer.nslice, 1)

    def test_setup_slicer_indices(self):
        """Test slicer returns correct indices (all) after setup.
        Note this also tests slicing.
        """
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = make_data_values(nvalues, dvmin, dvmax, random=672)
        self.testslicer.setup_slicer(dv)
        # test slicing
        self.assertEqual(len(self.testslicer.indices), len(dv["testdata"]))
        np.testing.assert_equal(dv[self.testslicer.indices], dv)


class TestUniSlicerIteration(unittest.TestCase):
    def setUp(self):
        self.testslicer = UniSlicer()

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_iteration(self):
        """Test iteration -- which is a one-step identity op for a
        unislicer.
        """
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = make_data_values(nvalues, dvmin, dvmax, random=432)
        self.testslicer.setup_slicer(dv)
        for i, b in enumerate(self.testslicer):
            pass
        self.assertEqual(i, 0)

    def test_get_item(self):
        """Test that can return an individual indexed values of the slicer."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = make_data_values(nvalues, dvmin, dvmax, random=1192)
        self.testslicer.setup_slicer(dv)
        self.assertEqual(self.testslicer[0]["slice_point"]["sid"], 0)


class TestUniSlicerEqual(unittest.TestCase):
    def setUp(self):
        self.testslicer = UniSlicer()
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = make_data_values(nvalues, dvmin, dvmax, random=3482)
        self.testslicer.setup_slicer(dv)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_equivalence(self):
        """Test equals method."""
        # Note that two uni slicers will be considered equal if they are
        # both the same kind of slicer (unislicer).
        # They will not necessarily slice data equally though (the indices are
        #  not necessarily the same!).
        # These should be the same, even though data is not the same.
        testslicer2 = UniSlicer()
        dv2 = make_data_values(100, 0, 1, random=43298)
        testslicer2.setup_slicer(dv2)
        self.assertEqual(self.testslicer, testslicer2)
        # these will not be the same, as different slicer type.
        testslicer2 = OneDSlicer(slice_col_name="testdata", bins=np.arange(0, 10, 1))
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)


if __name__ == "__main__":
    unittest.main()
