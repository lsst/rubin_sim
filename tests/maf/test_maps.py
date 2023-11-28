import os
import unittest
import warnings

import numpy as np
from rubin_scheduler.data import get_data_dir

import rubin_sim.maf.maps as maps
import rubin_sim.maf.slicers as slicers


def make_data_values(size=100, min=0.0, max=1.0, random=-1):
    """Generate a simple array of numbers, evenly arranged between min/max,
    (optional) random order.
    """
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


def make_field_data(seed):
    rng = np.random.RandomState(seed)
    names = ["fieldId", "fieldRA", "fieldDec"]
    types = [int, float, float]
    field_data = np.zeros(100, dtype=list(zip(names, types)))
    field_data["fieldId"] = np.arange(100)
    field_data["fieldRA"] = rng.rand(100)
    field_data["fieldDec"] = rng.rand(100)
    return field_data


class TestMaps(unittest.TestCase):
    def test_dust_map(self):
        map_path = os.path.join(get_data_dir(), "tests")
        nside = 8
        if os.path.isfile(os.path.join(map_path, f"dust_nside_{nside}.npz")):
            data = make_data_values(random=981)
            dustmap = maps.DustMap(nside=nside, map_path=map_path)

            slicer1 = slicers.HealpixSlicer(lat_lon_deg=False, nside=nside, use_camera=False)
            slicer1.setup_slicer(data)
            result1 = dustmap.run(slicer1.slice_points)
            assert "ebv" in list(result1.keys())

            field_data = make_field_data(2234)

            slicer2 = slicers.UserPointsSlicer(
                field_data["fieldRA"], field_data["fieldDec"], lat_lon_deg=False
            )
            result2 = dustmap.run(slicer2.slice_points)
            assert "ebv" in list(result2.keys())

            # Check interpolation works
            dustmap = maps.DustMap(interp=True, nside=nside, map_path=map_path)
            result3 = dustmap.run(slicer2.slice_points)
            assert "ebv" in list(result3.keys())

            # Check warning gets raised
            dustmap = maps.DustMap(nside=4, map_path=map_path)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dustmap.run(slicer1.slice_points)
                self.assertIn("nside", str(w[-1].message))
        else:
            warnings.warn("Did not find dustmaps, not running testMaps.py")

    def test_dust_map3_d(self):
        nside = 8
        map_file = os.path.join(get_data_dir(), "tests", f"test_ebv3d_nside{nside}.fits")
        if os.path.isfile(map_file):
            data = make_data_values(random=981)
            dustmap = maps.DustMap3D(nside=nside, map_file=map_file, interp=False)

            slicer1 = slicers.HealpixSlicer(lat_lon_deg=False, nside=nside, use_camera=False)
            slicer1.setup_slicer(data)
            result1 = dustmap.run(slicer1.slice_points)
            assert "ebv3d_ebvs" in list(result1.keys())
            assert "ebv3d_dists" in list(result1.keys())

            field_data = make_field_data(2234)

            slicer2 = slicers.UserPointsSlicer(
                field_data["fieldRA"], field_data["fieldDec"], lat_lon_deg=False
            )
            result2 = dustmap.run(slicer2.slice_points)
            assert "ebv3d_ebvs" in list(result2.keys())
            assert "ebv3d_dists" in list(result2.keys())

            # Check interpolation works
            dustmap = maps.DustMap3D(interp=True, nside=nside, map_file=map_file, dist_pc=2000, d_mag=10)
            result3 = dustmap.run(slicer2.slice_points)
            assert "ebv3d_ebvs" in list(result3.keys())
            assert "ebv3d_dists" in list(result3.keys())
            assert "ebv3d_ebv_at_2000.0" in list(result3.keys())
            assert "ebv3d_dist_at_10.0" in list(result3.keys())

            # Check that we can call the distance at magnitude method
            _ = dustmap.distance_at_dmag(5, result3["ebv3d_dists"], result3["ebv3d_ebvs"], "r")
            # And call it without running the map
            # (and thus reading more info) first
            _ = maps.DustMap3D().distance_at_dmag(5, result3["ebv3d_dists"], result3["ebv3d_ebvs"], "r")
            # And call it as a static method at one point on the sky
            _ = maps.DustMap3D().distance_at_dmag(
                5, result3["ebv3d_dists"][0, :], result3["ebv3d_ebvs"][0, :], "r"
            )

            # Check warning gets raised
            dustmap = maps.DustMap3D(nside=4, map_file=map_file)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dustmap.run(slicer1.slice_points)
                self.assertIn("nside", str(w[-1].message))
        else:
            warnings.warn("Did not find dustmaps, not running testMaps.py")

    def test_star_map(self):
        map_path = os.path.join(get_data_dir(), "tests")

        if os.path.isfile(os.path.join(map_path, "starDensity_r_nside_64.npz")):
            data = make_data_values(random=887)
            # check that it works if nside does not match map nside of 64
            nsides = [32, 64, 128]
            for nside in nsides:
                starmap = maps.StellarDensityMap(map_dir=map_path)
                slicer1 = slicers.HealpixSlicer(nside=nside, lat_lon_deg=False, use_camera=False)
                slicer1.setup_slicer(data)
                result1 = starmap.run(slicer1.slice_points)
                assert "starMapBins_r" in list(result1.keys())
                assert "starLumFunc_r" in list(result1.keys())
                assert np.max(result1["starLumFunc_r"] > 0)

            field_data = make_field_data(22)

            slicer2 = slicers.UserPointsSlicer(
                field_data["fieldRA"], field_data["fieldDec"], lat_lon_deg=False
            )
            result2 = starmap.run(slicer2.slice_points)
            assert "starMapBins_r" in list(result2.keys())
            assert "starLumFunc_r" in list(result2.keys())
            assert np.max(result2["starLumFunc_r"] > 0)

        else:
            warnings.warn("Did not find stellar density map, skipping test.")

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maps")),
        "Skip the galplane priority map data unless maps data present, required for setup",
    )
    def test_galplane_priority_maps(self):
        map_path = os.path.join(get_data_dir(), "maps")
        nside = 64
        data = make_data_values(random=981)
        galplane_map = maps.GalacticPlanePriorityMap(nside=nside, map_path=map_path, interp=False)

        # Set up basic case - healpix
        slicer1 = slicers.HealpixSlicer(lat_lon_deg=False, nside=nside, use_camera=False)
        slicer1.setup_slicer(data)
        result1 = galplane_map.run(slicer1.slice_points)
        key = maps.gp_priority_map_components_to_keys("sum", "combined_map")
        assert key in list(result1.keys())

        # Set up more advanced case - random ra/dec
        field_data = make_field_data(2234)
        slicer2 = slicers.UserPointsSlicer(field_data["fieldRA"], field_data["fieldDec"], lat_lon_deg=False)
        result2 = galplane_map.run(slicer2.slice_points)
        assert key in list(result2.keys())

        # Check interpolation works
        galplane_map = maps.GalacticPlanePriorityMap(interp=True, nside=nside, map_path=map_path)
        result3 = galplane_map.run(slicer2.slice_points)
        assert key in list(result3.keys())

        # Check warning gets raised
        galplane_map = maps.GalacticPlanePriorityMap(nside=4, map_path=map_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            galplane_map.run(slicer1.slice_points)
            self.assertIn("nside", str(w[-1].message))


if __name__ == "__main__":
    unittest.main()
