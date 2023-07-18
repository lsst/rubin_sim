import os
import unittest
from tempfile import TemporaryDirectory

import healpy
import numpy as np
import palpy
import pandas as pd

import rubin_sim.skybrightness_pre.zernike as zernike
from rubin_sim.data import get_data_dir
from rubin_sim.skybrightness_pre.zernike.zernike import TELESCOPE


class TestZenikeFitDrivers(unittest.TestCase):
    test_data_base_fname = "59823_59823"

    def setUp(self):
        self.cut_pre_data_dir = os.path.join(get_data_dir(), "tests")

    #    @unittest.skip("skipping test_fit_pre")
    def test_fit_pre(self):
        npy_fname = os.path.join(self.cut_pre_data_dir, self.test_data_base_fname + ".npy")
        npz_fname = os.path.join(self.cut_pre_data_dir, self.test_data_base_fname + ".npz")
        zernike_coeffs = zernike.fit_pre(npy_fname, npz_fname)
        self.assertGreater(zernike_coeffs.shape[0], 5)
        self.assertGreater(zernike_coeffs.shape[1], 20)
        self.assertEqual(tuple(zernike_coeffs.index.names), ("band", "mjd"))

    # @unittest.skip("skipping because slow")
    def test_bulk_zernike_fit(self):
        test_out_dir = TemporaryDirectory()
        out_fname = os.path.join(test_out_dir.name, "bulk_zern_fit.h5")
        zernike_coeffs = zernike.bulk_zernike_fit(self.cut_pre_data_dir, out_fname)
        self.assertGreater(zernike_coeffs.shape[0], 5)
        self.assertGreater(zernike_coeffs.shape[1], 20)
        self.assertEqual(tuple(zernike_coeffs.index.names), ("band", "mjd"))

        reread_zernike = pd.read_hdf(out_fname, "zernike_coeffs")
        self.assertGreater(reread_zernike.shape[0], 5)
        self.assertGreater(reread_zernike.shape[1], 20)
        self.assertEqual(tuple(reread_zernike.index.names), ("band", "mjd"))

        test_out_dir.cleanup()


class TestZernikeSky(unittest.TestCase):
    def setUp(self):
        self.cut_pre_data_dir = os.path.join(get_data_dir(), "tests")
        self.fname = os.path.join(self.cut_pre_data_dir, "zernsky.h5")

    @unittest.skip("skipping because slow")
    def test_compute_sky(self):
        zsky = zernike.ZernikeSky()
        zsky.load_coeffs(self.fname, "i")

        # Test computing from alt and az
        alt = np.arange(75.1, 80.1, 1)
        az = np.arange(175.1, 180.1, 1)
        mjd = 59823.97
        brightness = zsky.compute_sky(alt[0], az[0], mjd)
        brightnesses = zsky.compute_sky(alt, az, mjd)
        self.assertEqual(len(alt), len(brightnesses))

        # Figure out healpixels to test.
        sphere_npix = healpy.nside2npix(zsky.nside)
        sphere_ipix = np.arange(sphere_npix)
        ra, decl = healpy.pix2ang(zsky.nside, sphere_ipix, lonlat=True)
        gmst_rad = palpy.gmst(mjd)
        lst_rad = gmst_rad + TELESCOPE.longitude_rad
        ha_rad = lst_rad - np.radians(ra)
        az_rad, alt_rad = palpy.de2hVector(ha_rad, np.radians(decl), TELESCOPE.latitude_rad)
        visible_ipix = sphere_ipix[np.degrees(alt_rad) > 30]
        sample_hpix = pd.Series(visible_ipix).sample(5)
        hp_brightnesses = zsky.compute_healpix(sample_hpix, mjd)


class TestSkyBrightnessPreData(unittest.TestCase):
    test_data_base_fname = "59823_59823"

    def setUp(self):
        self.cut_pre_data_dir = os.path.join(get_data_dir(), "tests")

    def test_load(self):
        pre_data = zernike.SkyBrightnessPreData(
            self.test_data_base_fname,
            ("g", "r", "z"),
            pre_data_dir=self.cut_pre_data_dir,
        )

        self.assertIsInstance(pre_data.sky, pd.DataFrame)
        self.assertIsInstance(pre_data.times, pd.DataFrame)
        self.assertIsInstance(pre_data.metadata["alt"], np.ndarray)


class TestSkyModelZernike(unittest.TestCase):
    def setUp(self):
        self.fname = os.path.join(self.cut_pre_data_dir, "zernsky.h5")

    @unittest.skip("skipping because slow")
    def test_get_mags(self):
        mjd = 59823.97
        sky_model_zern = zernike.SkyModelZernike(data_file=self.fname)
        sky = sky_model_zern.return_mags(mjd, badval=np.nan)
        self.assertEqual(set(sky.keys()), set(("u", "g", "r", "i", "z", "y")))

        nside = sky_model_zern.zernike_model["g"].nside
        npix = healpy.nside2npix(nside)
        for band in sky.keys():
            self.assertEqual(sky[band].shape, (npix,))
            notnan = ~np.isnan(sky[band])
            self.assertLess(np.count_nonzero(notnan) / npix, 0.5)
            self.assertGreater(np.count_nonzero(notnan) / npix, 0.2)
            self.assertLess(sky[band][notnan].max(), 20)
            self.assertGreater(sky[band][notnan].min(), 8)

    @unittest.skip("skipping because slow")
    def test_get_mags_day(self):
        mjd = 59824.8
        sky_model_zern = zernike.SkyModelZernike(data_file=self.fname)
        sky = sky_model_zern.return_mags(mjd, badval=np.nan)
        self.assertEqual(set(sky.keys()), set(("u", "g", "r", "i", "z", "y")))

        nside = sky_model_zern.zernike_model["g"].nside
        npix = healpy.nside2npix(nside)
        for band in sky.keys():
            self.assertEqual(sky[band].shape, (npix,))
            self.assertEqual(np.count_nonzero(np.isnan(sky[band])), npix)


if __name__ == "__main__":
    unittest.main()
