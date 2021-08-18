import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
from rubin_sim.maf.slicers.uniSlicer import UniSlicer
from rubin_sim.maf.slicers.oneDSlicer import OneDSlicer


def makeDataValues(size=100, min=0., max=1., random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random > 0:
        rng = np.random.RandomState(random)
        randorder = rng.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(list(zip(datavalues)), dtype=[('testdata', 'float')])
    return datavalues


class TestUniSlicerSetupAndSlice(unittest.TestCase):

    def setUp(self):
        self.testslicer = UniSlicer()

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        self.assertEqual(self.testslicer.slicerName, self.testslicer.__class__.__name__)
        self.assertEqual(self.testslicer.slicerName, 'UniSlicer')

    def testSlicerNbins(self):
        self.assertEqual(self.testslicer.nslice, 1)

    def testSetupSlicerIndices(self):
        """Test slicer returns correct indices (all) after setup. Note this also tests slicing."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=672)
        self.testslicer.setupSlicer(dv)
        # test slicing
        self.assertEqual(len(self.testslicer.indices), len(dv['testdata']))
        np.testing.assert_equal(dv[self.testslicer.indices], dv)


class TestUniSlicerIteration(unittest.TestCase):

    def setUp(self):
        self.testslicer = UniSlicer()

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testIteration(self):
        """Test iteration -- which is a one-step identity op for a unislicer."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=432)
        self.testslicer.setupSlicer(dv)
        for i, b in enumerate(self.testslicer):
            pass
        self.assertEqual(i, 0)

    def testGetItem(self):
        """Test that can return an individual indexed values of the slicer."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=1192)
        self.testslicer.setupSlicer(dv)
        self.assertEqual(self.testslicer[0]['slicePoint']['sid'], 0)


class TestUniSlicerEqual(unittest.TestCase):

    def setUp(self):
        self.testslicer = UniSlicer()
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=3482)
        self.testslicer.setupSlicer(dv)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testEquivalence(self):
        """Test equals method."""
        # Note that two uni slicers will be considered equal if they are both the same kind of
        # slicer (unislicer). They will not necessarily slice data equally though (the indices are
        #  not necessarily the same!).
        # These should be the same, even though data is not the same.
        testslicer2 = UniSlicer()
        dv2 = makeDataValues(100, 0, 1, random=43298)
        testslicer2.setupSlicer(dv2)
        self.assertEqual(self.testslicer, testslicer2)
        # these will not be the same, as different slicer type.
        testslicer2 = OneDSlicer(sliceColName='testdata', bins=np.arange(0, 10, 1))
        testslicer2.setupSlicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)


if __name__ == "__main__":
    unittest.main()
