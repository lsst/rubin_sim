import numpy as np
import unittest
import healpy as hp
import rubin_sim.utils as utils


class TestHealUtils(unittest.TestCase):

    def testRaDecsRad(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils._hpid2RaDec(nside, hpids)

        hpids_return = utils._raDec2Hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def testRaDecsDeg(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2RaDec(nside, hpids)

        hpids_return = utils.raDec2Hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def testBinRad(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra * 0. + 1.

        nside = 128
        hpid = utils._raDec2Hpid(nside, ra[0], dec[0])

        map1 = utils._healbin(ra, dec, values, nside=nside)
        self.assertEqual(map1[hpid], 1.)
        self.assertEqual(hp.maptype(map1), 0)
        map2 = utils._healbin(ra, dec, values, nside=nside, reduceFunc=np.sum)
        self.assertEqual(map2[hpid], 3.)
        self.assertEqual(hp.maptype(map2), 0)
        map3 = utils._healbin(ra, dec, values, nside=nside, reduceFunc=np.std)
        self.assertEqual(map3[hpid], 0.)
        self.assertEqual(hp.maptype(map3), 0)

    def testBinDeg(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra * 0. + 1.

        nside = 128
        hpid = utils.raDec2Hpid(nside, ra[0], dec[0])

        map1 = utils.healbin(ra, dec, values, nside=nside)
        self.assertEqual(map1[hpid], 1.)
        self.assertEqual(hp.maptype(map1), 0)
        map2 = utils.healbin(ra, dec, values, nside=nside, reduceFunc=np.sum)
        self.assertEqual(map2[hpid], 3.)
        self.assertEqual(hp.maptype(map2), 0)
        map3 = utils.healbin(ra, dec, values, nside=nside, reduceFunc=np.std)
        self.assertEqual(map3[hpid], 0.)
        self.assertEqual(hp.maptype(map3), 0)


if __name__ == "__main__":
    unittest.main()
