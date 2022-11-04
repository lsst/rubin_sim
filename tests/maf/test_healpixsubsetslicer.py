import matplotlib

matplotlib.use("Agg")
import os
import numpy as np
import numpy.lib.recfunctions as rfn
import unittest
import healpy as hp
from rubin_sim.data import get_data_dir
from rubin_sim.maf.slicers import HealpixSubsetSlicer


def makeDataValues(
    size=100,
    minval=0.0,
    maxval=1.0,
    ramin=0,
    ramax=2 * np.pi,
    decmin=-np.pi,
    decmax=np.pi,
    random=1172,
):
    """Generate a simple array of numbers, evenly arranged between min/max,
    in 1 dimensions (optionally sorted), together with RA/Dec values
    for each data value."""
    data = []
    # Generate data values min - max.
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(maxval) - float(minval)) / (
        datavalues.max() - datavalues.min()
    )
    datavalues += minval
    rng = np.random.RandomState(random)
    randorder = rng.rand(size)
    randind = np.argsort(randorder)
    datavalues = datavalues[randind]
    datavalues = np.array(list(zip(datavalues)), dtype=[("testdata", "float")])
    data.append(datavalues)
    # Generate RA/Dec values equally spaces on sphere between ramin/max, decmin/max.
    ra = np.arange(0, size, dtype="float")
    ra *= (float(ramax) - float(ramin)) / (ra.max() - ra.min())
    randorder = rng.rand(size)
    randind = np.argsort(randorder)
    ra = ra[randind]
    ra = np.array(list(zip(ra)), dtype=[("ra", "float")])
    data.append(ra)
    v = np.arange(0, size, dtype="float")
    v *= (
        (np.cos(decmax + np.pi) + 1.0) / 2.0 - (np.cos(decmin + np.pi) + 1.0) / 2.0
    ) / (v.max() - v.min())
    v += (np.cos(decmin + np.pi) + 1.0) / 2.0
    dec = np.arccos(2 * v - 1) - np.pi
    randorder = rng.rand(size)
    randind = np.argsort(randorder)
    dec = dec[randind]
    dec = np.array(list(zip(dec)), dtype=[("dec", "float")])
    data.append(dec)
    # Add in rotation angle
    rot = rng.rand(len(dec)) * 2 * np.pi
    data.append(np.array(rot, dtype=[("rotSkyPos", "float")]))
    mjd = np.arange(len(dec)) * 0.1
    data.append(np.array(mjd, dtype=[("observationStartMJD", "float")]))
    data = rfn.merge_arrays(data, flatten=True, usemask=False)
    return data


def calcDist_vincenty(RA1, Dec1, RA2, Dec2):
    """Calculates distance on a sphere using the Vincenty formula.
    Give this function RA/Dec values in radians. Returns angular distance(s), in radians.
    Note that since this is all numpy, you could input arrays of RA/Decs."""
    D1 = (np.cos(Dec2) * np.sin(RA2 - RA1)) ** 2 + (
        np.cos(Dec1) * np.sin(Dec2) - np.sin(Dec1) * np.cos(Dec2) * np.cos(RA2 - RA1)
    ) ** 2
    D1 = np.sqrt(D1)
    D2 = np.sin(Dec1) * np.sin(Dec2) + np.cos(Dec1) * np.cos(Dec2) * np.cos(RA2 - RA1)
    D = np.arctan2(D1, D2)
    return D


class TestHealpixSubsetSlicerSetup(unittest.TestCase):
    def setUp(self):
        self.cameraFootprintFile = os.path.join(get_data_dir(), "tests", "fov_map.npz")
        self.nside = 16
        idxs = np.arange(0, hp.nside2npix(self.nside))
        self.idxs = idxs[np.where(idxs < idxs.max() / 2)]

    def testSlicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""
        testslicer = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=self.idxs,
            verbose=False,
            camera_footprint_file=self.cameraFootprintFile,
        )
        self.assertEqual(testslicer.slicerName, testslicer.__class__.__name__)
        self.assertEqual(testslicer.slicerName, "HealpixSubsetSlicer")


class TestHealpixSlicerEqual(unittest.TestCase):
    def setUp(self):
        self.cameraFootprintFile = os.path.join(get_data_dir(), "tests", "fov_map.npz")
        self.nside = 16
        self.testslicer = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=np.arange(0, 10),
            verbose=False,
            lon_col="ra",
            lat_col="dec",
            camera_footprint_file=self.cameraFootprintFile,
        )
        nvalues = 10000
        self.dv = makeDataValues(
            size=nvalues,
            minval=0.0,
            maxval=1.0,
            ramin=0,
            ramax=2 * np.pi,
            decmin=-np.pi,
            decmax=0,
            random=22,
        )
        self.testslicer.setup_slicer(self.dv)

    def tearDown(self):
        del self.testslicer
        del self.dv
        self.testslicer = None

    def testSlicerEquivalence(self):
        """Test that slicers are marked equal when appropriate, and unequal when appropriate."""
        # Note that they are judged equal based on nsides (not on data in ra/dec spatial tree).
        testslicer2 = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=np.arange(0, 10),
            verbose=False,
            lon_col="ra",
            lat_col="dec",
            camera_footprint_file=self.cameraFootprintFile,
        )
        self.assertEqual(self.testslicer, testslicer2)
        assert (self.testslicer != testslicer2) is False
        testslicer2 = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=np.arange(10, 20),
            verbose=False,
            lon_col="ra",
            lat_col="dec",
            camera_footprint_file=self.cameraFootprintFile,
        )
        self.assertNotEqual(self.testslicer, testslicer2)
        assert (self.testslicer != testslicer2) is True


class TestHealpixSlicerIteration(unittest.TestCase):
    def setUp(self):
        self.cameraFootprintFile = os.path.join(get_data_dir(), "tests", "fov_map.npz")
        self.nside = 8
        self.hpid = np.arange(10, 20)
        self.testslicer = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=self.hpid,
            verbose=False,
            lon_col="ra",
            lat_col="dec",
            camera_footprint_file=self.cameraFootprintFile,
        )
        nvalues = 10000
        self.dv = makeDataValues(
            size=nvalues,
            minval=0.0,
            maxval=1.0,
            ramin=0,
            ramax=2 * np.pi,
            decmin=-np.pi,
            decmax=0,
            random=33,
        )
        self.testslicer.setup_slicer(self.dv)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testIteration(self):
        """Test iteration goes through expected range and ra/dec are in expected range (radians)."""
        npix = hp.nside2npix(self.nside)
        for i, s in enumerate(self.testslicer):
            # We do not expect testslicer.slicePoint['sid'] to == i ..
            # The slicer will only iterate over the subset
            self.assertEqual(self.hpid[i], s["slicePoint"]["sid"])
            ra = s["slicePoint"]["ra"]
            dec = s["slicePoint"]["dec"]
            self.assertGreaterEqual(ra, 0)
            self.assertLessEqual(ra, 2 * np.pi)
            self.assertGreaterEqual(dec, -np.pi)
            self.assertLessEqual(dec, np.pi)
        self.assertEqual(i + 1, len(self.hpid))

    def testGetItem(self):
        """Test getting indexed value."""
        for i, s in enumerate(self.testslicer):
            np.testing.assert_equal(self.testslicer[self.hpid[i]], s)


class TestHealpixSlicerSlicing(unittest.TestCase):
    # Note that this is really testing baseSpatialSlicer, as slicing is done there for healpix grid

    def setUp(self):
        self.cameraFootprintFile = os.path.join(get_data_dir(), "tests", "fov_map.npz")
        self.nside = 8
        self.radius = 1.8
        self.testslicer = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=np.arange(0, hp.nside2npix(self.nside)),
            verbose=False,
            lon_col="ra",
            lat_col="dec",
            lat_lon_deg=False,
            radius=self.radius,
            use_camera=False,
        )
        nvalues = 10000
        self.dv = makeDataValues(
            size=nvalues,
            minval=0.0,
            maxval=1.0,
            ramin=0,
            ramax=2 * np.pi,
            decmin=-np.pi,
            decmax=0,
            random=44,
        )

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicing(self):
        """Test slicing returns (all) data points which are within 'radius' of bin point."""
        # Test that slicing fails before setup_slicer
        self.assertRaises(NotImplementedError, self.testslicer._slice_sim_data, 0)
        # Set up and test actual slicing.
        self.testslicer.setup_slicer(self.dv)
        for s in self.testslicer:
            ra = s["slicePoint"]["ra"]
            dec = s["slicePoint"]["dec"]
            distances = calcDist_vincenty(ra, dec, self.dv["ra"], self.dv["dec"])
            didxs = np.where(distances <= np.radians(self.radius))
            sidxs = s["idxs"]
            self.assertEqual(len(sidxs), len(didxs[0]))
            if len(sidxs) > 0:
                didxs = np.sort(didxs[0])
                sidxs = np.sort(sidxs)
                np.testing.assert_equal(
                    self.dv["testdata"][didxs], self.dv["testdata"][sidxs]
                )

    def testSubsetSlice(self):
        testslicer = HealpixSubsetSlicer(
            nside=self.nside,
            hpid=np.arange(0, 2),
            verbose=False,
            lon_col="ra",
            lat_col="dec",
            lat_lon_deg=False,
            radius=self.radius,
            use_camera=False,
        )
        testslicer.setup_slicer(self.dv)
        self.testslicer.setup_slicer(self.dv)
        # For points which return values when used with the full set of points (self.testslicer)
        # there will be no data when used with the subset of points
        for s in self.testslicer:
            if s["slicePoint"]["sid"] > 2:
                if len(s["idxs"]) > 0:
                    islice = s["slicePoint"]["sid"]
                    self.assertTrue(len(testslicer[islice]["idxs"]) == 0)


if __name__ == "__main__":
    unittest.main()
