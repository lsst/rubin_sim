import unittest

import healpy as hp
import numpy as np

import rubin_sim.utils as utils


class TestHealUtils(unittest.TestCase):
    def test_ra_decs_rad(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils._hpid2_ra_dec(nside, hpids)

        hpids_return = utils._ra_dec2_hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def test_ra_decs_deg(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2_ra_dec(nside, hpids)

        hpids_return = utils.ra_dec2_hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def test_bin_rad(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra * 0.0 + 1.0

        nside = 128
        hpid = utils._ra_dec2_hpid(nside, ra[0], dec[0])

        map1 = utils._healbin(ra, dec, values, nside=nside)
        self.assertEqual(map1[hpid], 1.0)
        self.assertEqual(hp.maptype(map1), 0)
        map2 = utils._healbin(ra, dec, values, nside=nside, reduce_func=np.sum)
        self.assertEqual(map2[hpid], 3.0)
        self.assertEqual(hp.maptype(map2), 0)
        map3 = utils._healbin(ra, dec, values, nside=nside, reduce_func=np.std)
        self.assertEqual(map3[hpid], 0.0)
        self.assertEqual(hp.maptype(map3), 0)

    def test_bin_deg(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra * 0.0 + 1.0

        nside = 128
        hpid = utils.ra_dec2_hpid(nside, ra[0], dec[0])

        map1 = utils.healbin(ra, dec, values, nside=nside)
        self.assertEqual(map1[hpid], 1.0)
        self.assertEqual(hp.maptype(map1), 0)
        map2 = utils.healbin(ra, dec, values, nside=nside, reduce_func=np.sum)
        self.assertEqual(map2[hpid], 3.0)
        self.assertEqual(hp.maptype(map2), 0)
        map3 = utils.healbin(ra, dec, values, nside=nside, reduce_func=np.std)
        self.assertEqual(map3[hpid], 0.0)
        self.assertEqual(hp.maptype(map3), 0)


if __name__ == "__main__":
    unittest.main()
