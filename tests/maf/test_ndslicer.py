import itertools
import unittest
import warnings

import numpy as np
import numpy.lib.recfunctions as rfn

from rubin_sim.maf.slicers import NDSlicer, UniSlicer


def make_data_values(size=100, min=0.0, max=1.0, nd=3, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max,
    in nd dimensions, but (optional) random order.
    """
    data = []
    for d in range(nd):
        datavalues = np.arange(0, size, dtype="float")
        datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
        datavalues += min
        if random > 0:
            rng = np.random.RandomState(random)
            randorder = rng.rand(size)
            randind = np.argsort(randorder)
            datavalues = datavalues[randind]
        datavalues = np.array(list(zip(datavalues)), dtype=[("testdata" + "%d" % (d), "float")])
        data.append(datavalues)
    data = rfn.merge_arrays(data, flatten=True, usemask=False)
    return data


class TestNDSlicerSetup(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=608)
        self.dvlist = self.dv.dtype.names

    def test_slicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        testslicer = NDSlicer(self.dvlist)
        self.assertEqual(testslicer.slicer_name, testslicer.__class__.__name__)
        self.assertEqual(testslicer.slicer_name, "NDSlicer")

    def test_setup_slicer_bins(self):
        """Test setting up slicer using defined bins."""
        # Used right bins?
        bins = np.arange(self.dvmin, self.dvmax, 0.1)
        binlist = []
        for d in range(self.nd):
            binlist.append(bins)
        testslicer = NDSlicer(self.dvlist, bins_list=binlist)
        testslicer.setup_slicer(self.dv)
        for d in range(self.nd):
            np.testing.assert_equal(testslicer.bins[d], bins)
        self.assertEqual(testslicer.nslice, (len(bins) - 1) ** self.nd)

    def test_setup_slicer_nbins(self):
        """Test setting up slicer using nbins."""
        for nvalues in (100, 1000):
            for nbins in (5, 25, 74):
                dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=-1)
                # Right number of bins?
                # expect one more 'bin' to accomodate last right edge,
                # but nbins accounts for this
                testslicer = NDSlicer(self.dvlist, bins_list=nbins)
                testslicer.setup_slicer(dv)
                self.assertEqual(testslicer.nslice, nbins**self.nd)
                # Bins of the right size?
                for i in range(self.nd):
                    bindiff = np.diff(testslicer.bins[i])
                    expectedbindiff = (self.dvmax - self.dvmin) / float(nbins)
                    np.testing.assert_allclose(bindiff, expectedbindiff)
                # Can we use a list of nbins too and get the
                # right number of bins?
                nbins_list = []
                expectednbins = 1
                for d in range(self.nd):
                    nbins_list.append(nbins + d)
                    expectednbins *= nbins + d
                testslicer = NDSlicer(self.dvlist, bins_list=nbins_list)
                testslicer.setup_slicer(dv)
                self.assertEqual(testslicer.nslice, expectednbins)

    def test_setup_slicer_nbins_zeros(self):
        """Test handling case of data being single values."""
        dv = make_data_values(100, 0, 0, self.nd, random=-1)
        nbins = 10
        testslicer = NDSlicer(self.dvlist, bins_list=nbins)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            testslicer.setup_slicer(dv)
            self.assertIn("creasing bin_max", str(w[-1].message))
        expectednbins = nbins**self.nd
        self.assertEqual(testslicer.nslice, expectednbins)

    def test_setup_slicer_equivalent(self):
        """Test setting up slicer using defined bins and nbins is
        equal where expected.
        """
        for nbins in (20, 105):
            testslicer = NDSlicer(self.dvlist, bins_list=nbins)
            bins = make_data_values(nbins + 1, self.dvmin, self.dvmax, self.nd, random=-1)
            bins_list = []
            for i in bins.dtype.names:
                bins_list.append(bins[i])
            for nvalues in (100, 10000):
                dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=64432)
                testslicer.setup_slicer(dv)
                for i in range(self.nd):
                    np.testing.assert_allclose(testslicer.bins[i], bins_list[i])


class TestNDSlicerEqual(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=20367)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist, bins_list=100)
        self.testslicer.setup_slicer(self.dv)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_equivalence(self):
        """Test equals method."""
        # Note that two ND slicers will be considered equal if
        # they are both the same kind of slicer AND have the same bins in
        # all dimensions.
        # Set up another slicer to match
        # (same bins, although not the same data).
        dv2 = make_data_values(100, self.dvmin, self.dvmax, self.nd, random=10029)
        dvlist = dv2.dtype.names
        testslicer2 = NDSlicer(slice_col_list=dvlist, bins_list=self.testslicer.bins)
        testslicer2.setup_slicer(dv2)
        self.assertEqual(self.testslicer, testslicer2)
        # Set up another slicer that should not match (different bins)
        dv2 = make_data_values(1000, self.dvmin + 1, self.dvmax + 1, self.nd, random=209837)
        testslicer2 = NDSlicer(slice_col_list=dvlist, bins_list=100)
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)
        # Set up another slicer that should not match (different dimensions)
        dv2 = make_data_values(1000, self.dvmin, self.dvmax, self.nd - 1, random=50623)
        testslicer2 = NDSlicer(dv2.dtype.names, bins_list=100)
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)
        # Set up a different kind of slicer that should not match.
        testslicer2 = UniSlicer()
        dv2 = make_data_values(100, 0, 1, random=22310098)
        testslicer2.setup_slicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)


class TestNDSlicerIteration(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=11081)
        self.dvlist = self.dv.dtype.names
        nvalues = 1000
        bins = np.arange(self.dvmin, self.dvmax, 0.1)
        bins_list = []
        self.iterlist = []
        for i in range(self.nd):
            bins_list.append(bins)
            # (remember iteration doesn't use the very last bin in 'bins')
            self.iterlist.append(bins[:-1])
        dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=17)
        self.testslicer = NDSlicer(self.dvlist, bins_list=bins_list)
        self.testslicer.setup_slicer(dv)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_iteration(self):
        """Test iteration."""
        for s, ib in zip(self.testslicer, itertools.product(*self.iterlist)):
            self.assertEqual(s["slice_point"]["bin_left"], ib)

    def test_get_item(self):
        """Test getting indexed binpoint."""
        for i, s in enumerate(self.testslicer):
            self.assertEqual(
                self.testslicer[i]["slice_point"]["bin_left"],
                s["slice_point"]["bin_left"],
            )
        self.assertEqual(self.testslicer[0]["slice_point"]["bin_left"], (0.0, 0.0, 0.0))


class TestNDSlicerSlicing(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=173)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def test_slicing(self):
        """Test slicing."""
        # Test get error if try to slice before setup.
        self.assertRaises(NotImplementedError, self.testslicer._slice_sim_data, 0)
        nbins = 10
        bin_size = (self.dvmax - self.dvmin) / (float(nbins))
        self.testslicer = NDSlicer(self.dvlist, bins_list=nbins)
        for nvalues in (1000, 10000):
            dv = make_data_values(nvalues, self.dvmin, self.dvmax, self.nd, random=1735)
            self.testslicer.setup_slicer(dv)
            sum = 0
            for i, s in enumerate(self.testslicer):
                idxs = s["idxs"]
                data_slice = dv[idxs]
                sum += len(idxs)
                if len(data_slice) > 0:
                    for i, dvname, b in zip(list(range(self.nd)), self.dvlist, s["slice_point"]["bin_left"]):
                        self.assertGreaterEqual((data_slice[dvname].min() - b), 0)
                    if i < self.testslicer.nslice - 1:
                        self.assertLessEqual((data_slice[dvname].max() - b), bin_size)
                    else:
                        self.assertAlmostEqual((data_slice[dvname].max() - b), bin_size)
                    self.assertEqual(len(data_slice), nvalues / float(nbins))
            # and check that every data value was assigned somewhere.
            self.assertEqual(sum, nvalues)


if __name__ == "__main__":
    unittest.main()
