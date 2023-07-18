import unittest
import warnings

import healpy as hp
import numpy as np

import rubin_sim.skybrightness_pre as sbp
import rubin_sim.utils as utils


class TestDarkSky(unittest.TestCase):
    def test_default(self):
        dark_sky = sbp.dark_sky()

        self.assert_dark_sky(32, dark_sky)

    def test_upgrade(self):
        nside = 64
        dark_sky = sbp.dark_sky(nside=nside)

        self.assert_dark_sky(nside, dark_sky)

    def test_downgrade(self):
        nside = 16
        dark_sky = sbp.dark_sky(nside=nside)

        self.assert_dark_sky(nside, dark_sky)

    def test_downgrade_default(self):
        default_map_before_downgrade = sbp.dark_sky()

        nside = 16
        sbp.dark_sky(nside=16)

        default_map_after_downgrade = sbp.dark_sky()

        for band in self.expected_bands:
            assert np.all(
                np.equal(
                    default_map_before_downgrade[band][np.isfinite(default_map_before_downgrade[band])],
                    default_map_after_downgrade[band][np.isfinite(default_map_after_downgrade[band])],
                )
            )

    def assert_dark_sky(self, nside, dark_sky):
        for band in self.expected_bands:
            assert band in dark_sky.dtype.names
            assert hp.npix2nside(len(dark_sky[band])) == nside

    @property
    def expected_bands(self):
        return ("u", "g", "r", "i", "z", "y")
