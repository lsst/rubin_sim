import unittest

import healpy as hp
import numpy as np

import rubin_sim.utils as utils


class ApproxCoordTests(unittest.TestCase):
    """
    Test the fast approximate ra,dec to alt,az transforms
    """

    def test_degrees(self):
        nside = 16
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2_ra_dec(nside, hpids)
        mjd = 59852.0
        obs = utils.ObservationMetaData(mjd=mjd)

        alt1, az1, pa1 = utils.alt_az_pa_from_ra_dec(ra, dec, obs)

        alt2, az2 = utils.approx_ra_dec2_alt_az(ra, dec, obs.site.latitude, obs.site.longitude, mjd)

        # Check that the fast is similar to the more precice transform
        tol = 2  # Degrees
        tol_mean = 1.0
        separations = utils.angular_separation(az1, alt1, az2, alt2)
        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)

        # Check that the fast can nearly round-trip
        ra_back, dec_back = utils.approx_alt_az2_ra_dec(alt2, az2, obs.site.latitude, obs.site.longitude, mjd)
        separations = utils.angular_separation(ra, dec, ra_back, dec_back)
        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)

    def test_rad(self):
        nside = 16
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils._hpid2_ra_dec(nside, hpids)
        mjd = 59852.0
        obs = utils.ObservationMetaData(mjd=mjd)

        alt1, az1, pa1 = utils._alt_az_pa_from_ra_dec(ra, dec, obs)

        alt2, az2 = utils._approx_ra_dec2_alt_az(ra, dec, obs.site.latitude_rad, obs.site.longitude_rad, mjd)

        # Check that the fast is similar to the more precice transform
        tol = np.radians(2)
        tol_mean = np.radians(1.0)
        separations = utils._angular_separation(az1, alt1, az2, alt2)

        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)

        # Check that the fast can nearly round-trip
        ra_back, dec_back = utils._approx_alt_az2_ra_dec(
            alt2, az2, obs.site.latitude_rad, obs.site.longitude_rad, mjd
        )
        separations = utils._angular_separation(ra, dec, ra_back, dec_back)
        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)


if __name__ == "__main__":
    unittest.main()
