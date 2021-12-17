import matplotlib

matplotlib.use("Agg")
import numpy as np
import unittest
import warnings
import os
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.maps as maps
from rubin_sim.data import get_data_dir


def makeDataValues(size=100, min=0.0, max=1.0, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype="float")
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    rot_vals = datavalues * 0
    if random > 0:
        rng = np.random.RandomState(random)
        randorder = rng.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    ids = np.arange(size)
    datavalues = np.array(
        list(zip(datavalues, datavalues, ids, rot_vals)),
        dtype=[
            ("fieldRA", "float"),
            ("fieldDec", "float"),
            ("fieldId", "int"),
            ("rotSkyPos", "float"),
        ],
    )
    return datavalues


def makeFieldData(seed):
    rng = np.random.RandomState(seed)
    names = ["fieldId", "fieldRA", "fieldDec"]
    types = [int, float, float]
    fieldData = np.zeros(100, dtype=list(zip(names, types)))
    fieldData["fieldId"] = np.arange(100)
    fieldData["fieldRA"] = rng.rand(100)
    fieldData["fieldDec"] = rng.rand(100)
    return fieldData


class TestMaps(unittest.TestCase):
    def testDustMap(self):

        mapPath = os.path.join(get_data_dir(), "tests")
        nside = 8
        if os.path.isfile(os.path.join(mapPath, f"dust_nside_{nside}.npz")):

            data = makeDataValues(random=981)
            dustmap = maps.DustMap(nside=nside, mapPath=mapPath)

            slicer1 = slicers.HealpixSlicer(
                latLonDeg=False, nside=nside, useCamera=False
            )
            slicer1.setupSlicer(data)
            result1 = dustmap.run(slicer1.slicePoints)
            assert "ebv" in list(result1.keys())

            fieldData = makeFieldData(2234)

            slicer2 = slicers.UserPointsSlicer(
                fieldData["fieldRA"], fieldData["fieldDec"], latLonDeg=False
            )
            result2 = dustmap.run(slicer2.slicePoints)
            assert "ebv" in list(result2.keys())

            # Check interpolation works
            dustmap = maps.DustMap(interp=True, nside=nside, mapPath=mapPath)
            result3 = dustmap.run(slicer2.slicePoints)
            assert "ebv" in list(result3.keys())

            # Check warning gets raised
            dustmap = maps.DustMap(nside=4, mapPath=mapPath)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dustmap.run(slicer1.slicePoints)
                self.assertIn("nside", str(w[-1].message))
        else:
            warnings.warn("Did not find dustmaps, not running testMaps.py")

    def testStarMap(self):
        mapPath = os.path.join(get_data_dir(), "tests")

        if os.path.isfile(os.path.join(mapPath, "starDensity_r_nside_64.npz")):
            data = makeDataValues(random=887)
            # check that it works if nside does not match map nside of 64
            nsides = [32, 64, 128]
            for nside in nsides:
                starmap = maps.StellarDensityMap(mapDir=mapPath)
                slicer1 = slicers.HealpixSlicer(
                    nside=nside, latLonDeg=False, useCamera=False
                )
                slicer1.setupSlicer(data)
                result1 = starmap.run(slicer1.slicePoints)
                assert "starMapBins_r" in list(result1.keys())
                assert "starLumFunc_r" in list(result1.keys())
                assert np.max(result1["starLumFunc_r"] > 0)

            fieldData = makeFieldData(22)

            slicer2 = slicers.UserPointsSlicer(
                fieldData["fieldRA"], fieldData["fieldDec"], latLonDeg=False
            )
            result2 = starmap.run(slicer2.slicePoints)
            assert "starMapBins_r" in list(result2.keys())
            assert "starLumFunc_r" in list(result2.keys())
            assert np.max(result2["starLumFunc_r"] > 0)

        else:
            warnings.warn("Did not find stellar density map, skipping test.")


if __name__ == "__main__":
    unittest.main()
