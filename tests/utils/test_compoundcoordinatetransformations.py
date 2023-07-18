import unittest

import numpy as np

import rubin_sim.utils as utils


def control_alt_az_from_ra_dec(ra_rad_in, dec_rad_in, long_rad, lat_rad, mjd):
    """
    Converts RA and Dec to altitude and azimuth

    Parameters
    ----------
    ra_rad : `Unknown`
        is the RA in radians
        (observed geocentric)

    Parameters
    ----------
    dec_rad : `Unknown`
        is the Dec in radians
        (observed geocentric)

    Parameters
    ----------
    long_rad : `Unknown`
        is the longitude of the observer in radians
        (positive east of the prime meridian)
    lat_rad : `Unknown`
        Latitude of the observer in radians (positive north of the equator)

    Parameters
    ----------
    mjd : `Unknown`
        is the universal time expressed as an MJD

    Parameters
    ----------
    altitude : `Unknown`
        in radians

    Returns
    -------
    azimuth : `Unknown`
        Azimuth in radians.

    see: http://www.stargazing.net/kepler/altaz.html#twig04
    """
    obs = utils.ObservationMetaData(
        mjd=utils.ModifiedJulianDate(utc=mjd),
        site=utils.Site(longitude=np.degrees(long_rad), latitude=np.degrees(lat_rad), name="LSST"),
    )

    if hasattr(ra_rad_in, "__len__"):
        ra_rad, dec_rad = utils._observed_from_icrs(
            ra_rad_in,
            dec_rad_in,
            obs_metadata=obs,
            epoch=2000.0,
            include_refraction=True,
        )
    else:
        ra_rad, dec_rad = utils._observed_from_icrs(
            ra_rad_in,
            dec_rad_in,
            obs_metadata=obs,
            epoch=2000.0,
            include_refraction=True,
        )

    lst = utils.calc_lmst_last(obs.mjd.ut1, long_rad)
    last = lst[1]
    ha_rad = np.radians(last * 15.0) - ra_rad

    sin_dec = np.sin(dec_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    sin_alt = sin_dec * sin_lat + np.cos(dec_rad) * cos_lat * np.cos(ha_rad)
    alt_rad = np.arcsin(sin_alt)
    az_rad = np.arccos((sin_dec - sin_alt * sin_lat) / (np.cos(alt_rad) * cos_lat))
    az_rad_out = np.where(np.sin(ha_rad) >= 0.0, 2.0 * np.pi - az_rad, az_rad)
    if isinstance(alt_rad, float):
        return alt_rad, float(az_rad_out)
    return alt_rad, az_rad_out


class CompoundCoordinateTransformationsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(32)
        self.mjd = 57087.0
        self.tolerance = 1.0e-5

    def test_exceptions(self):
        """
        Test to make sure that methods complain when incorrect data types are passed.
        """
        obs = utils.ObservationMetaData(pointing_ra=55.0, pointing_dec=-72.0, mjd=53467.8)

        ra_float = 1.1
        ra_list = np.array([0.2, 0.3])

        dec_float = 1.1
        dec_list = np.array([0.2, 0.3])

        self.assertRaises(RuntimeError, utils._alt_az_pa_from_ra_dec, ra_list, dec_float, obs)
        self.assertRaises(RuntimeError, utils._alt_az_pa_from_ra_dec, ra_float, dec_list, obs)
        utils._alt_az_pa_from_ra_dec(ra_float, dec_float, obs)
        utils._alt_az_pa_from_ra_dec(ra_list, dec_list, obs)

        self.assertRaises(RuntimeError, utils._ra_dec_from_alt_az, ra_list, dec_float, obs)
        self.assertRaises(RuntimeError, utils._ra_dec_from_alt_az, ra_float, dec_list, obs)
        utils._ra_dec_from_alt_az(ra_float, dec_float, obs)
        utils._ra_dec_from_alt_az(ra_list, dec_list, obs)

        self.assertRaises(RuntimeError, utils.alt_az_pa_from_ra_dec, ra_list, dec_float, obs)
        self.assertRaises(RuntimeError, utils.alt_az_pa_from_ra_dec, ra_float, dec_list, obs)
        utils.alt_az_pa_from_ra_dec(ra_float, dec_float, obs)
        utils.alt_az_pa_from_ra_dec(ra_list, dec_list, obs)

        self.assertRaises(RuntimeError, utils.ra_dec_from_alt_az, ra_list, dec_float, obs)
        self.assertRaises(RuntimeError, utils.ra_dec_from_alt_az, ra_float, dec_list, obs)
        utils.ra_dec_from_alt_az(ra_float, dec_float, obs)
        utils.ra_dec_from_alt_az(ra_list, dec_list, obs)

    def test_ra_dec_from_alt_az(self):
        """
        Test conversion of Alt, Az to Ra, Dec using data on the Sun

        This site gives the altitude and azimuth of the Sun as a function
        of time and position on the earth

        http://aa.usno.navy.mil/data/docs/AltAz.php

        This site gives the apparent geocentric RA, Dec of major celestial objects
        as a function of time

        http://aa.usno.navy.mil/data/docs/geocentric.php

        This site converts calendar dates into Julian Dates

        http://aa.usno.navy.mil/data/docs/JulianDate.php
        """

        hours = np.radians(360.0 / 24.0)
        minutes = hours / 60.0
        seconds = minutes / 60.0

        longitude_list = []
        latitude_list = []
        mjd_list = []
        alt_list = []
        az_list = []
        ra_app_list = []
        dec_app_list = []

        longitude_list.append(np.radians(-22.0 - 33.0 / 60.0))
        latitude_list.append(np.radians(11.0 + 45.0 / 60.0))
        mjd_list.append(2457364.958333 - 2400000.5)  # 8 December 2015 11:00 utc
        alt_list.append(np.radians(41.1))
        az_list.append(np.radians(134.7))
        ra_app_list.append(16.0 * hours + 59.0 * minutes + 16.665 * seconds)
        dec_app_list.append(np.radians(-22.0 - 42.0 / 60.0 - 2.94 / 3600.0))

        longitude_list.append(np.radians(-22.0 - 33.0 / 60.0))
        latitude_list.append(np.radians(11.0 + 45.0 / 60.0))
        mjd_list.append(2457368.958333 - 2400000.5)  # 12 December 2015 11:00 utc
        alt_list.append(np.radians(40.5))
        az_list.append(np.radians(134.7))
        ra_app_list.append(17.0 * hours + 16.0 * minutes + 51.649 * seconds)
        dec_app_list.append(np.radians(-23.0 - 3 / 60.0 - 50.35 / 3600.0))

        longitude_list.append(np.radians(145.0 + 23.0 / 60.0))
        latitude_list.append(np.radians(-64.0 - 5.0 / 60.0))
        mjd_list.append(2456727.583333 - 2400000.5)  # 11 March 2014, 02:00 utc
        alt_list.append(np.radians(29.5))
        az_list.append(np.radians(8.2))
        ra_app_list.append(23.0 * hours + 24.0 * minutes + 46.634 * seconds)
        dec_app_list.append(np.radians(-3.0 - 47.0 / 60.0 - 47.81 / 3600.0))

        longitude_list.append(np.radians(145.0 + 23.0 / 60.0))
        latitude_list.append(np.radians(-64.0 - 5.0 / 60.0))
        mjd_list.append(2456731.583333 - 2400000.5)  # 15 March 2014, 02:00 utc
        alt_list.append(np.radians(28.0))
        az_list.append(np.radians(7.8))
        ra_app_list.append(23.0 * hours + 39.0 * minutes + 27.695 * seconds)
        dec_app_list.append(np.radians(-2.0 - 13.0 / 60.0 - 18.32 / 3600.0))

        for longitude, latitude, mjd, alt, az, ra_app, dec_app in zip(
            longitude_list,
            latitude_list,
            mjd_list,
            alt_list,
            az_list,
            ra_app_list,
            dec_app_list,
        ):
            obs = utils.ObservationMetaData(
                site=utils.Site(
                    longitude=np.degrees(longitude),
                    latitude=np.degrees(latitude),
                    name="LSST",
                ),
                mjd=utils.ModifiedJulianDate(utc=mjd),
            )

            ra_icrs, dec_icrs = utils._ra_dec_from_alt_az(alt, az, obs)
            ra_test, dec_test = utils._app_geo_from_icrs(ra_icrs, dec_icrs, mjd=obs.mjd)

            distance = np.degrees(utils.haversine(ra_app, dec_app, ra_test, dec_test))
            # this is all the precision we have in the alt,az data taken from the USNO
            self.assertLess(distance, 0.1)

            correction = np.degrees(utils.haversine(ra_test, dec_test, ra_icrs, dec_icrs))
            self.assertLess(distance, correction)

    def test_alt_az_ra_dec_round_trip(self):
        """
        Test that alt_az_pa_from_ra_dec and ra_dec_from_alt_az really invert each other
        """

        mjd = 58350.0

        alt_in = []
        az_in = []
        for alt in np.arange(0.0, 90.0, 10.0):
            for az in np.arange(0.0, 360.0, 10.0):
                alt_in.append(alt)
                az_in.append(az)

        alt_in = np.array(alt_in)
        az_in = np.array(az_in)

        for lon in (0.0, 90.0, 135.0):
            for lat in (60.0, 30.0, -60.0, -30.0):
                obs = utils.ObservationMetaData(
                    mjd=mjd, site=utils.Site(longitude=lon, latitude=lat, name="LSST")
                )

                ra_in, dec_in = utils.ra_dec_from_alt_az(alt_in, az_in, obs)

                self.assertIsInstance(ra_in, np.ndarray)
                self.assertIsInstance(dec_in, np.ndarray)

                self.assertFalse(np.isnan(ra_in).any(), msg="there were NaNs in ra_in")
                self.assertFalse(np.isnan(dec_in).any(), msg="there were NaNs in dec_in")

                # test that passing them in one at a time gives the same answer
                for ix in range(len(alt_in)):
                    ra_f, dec_f = utils.ra_dec_from_alt_az(alt_in[ix], az_in[ix], obs)
                    self.assertIsInstance(ra_f, float)
                    self.assertIsInstance(dec_f, float)
                    self.assertAlmostEqual(ra_f, ra_in[ix], 12)
                    self.assertAlmostEqual(dec_f, dec_in[ix], 12)

                alt_out, az_out, pa_out = utils.alt_az_pa_from_ra_dec(ra_in, dec_in, obs)

                self.assertFalse(np.isnan(pa_out).any(), msg="there were NaNs in pa_out")

                for alt_c, az_c, alt_t, az_t in zip(
                    np.radians(alt_in),
                    np.radians(az_in),
                    np.radians(alt_out),
                    np.radians(az_out),
                ):
                    distance = utils.arcsec_from_radians(utils.haversine(az_c, alt_c, az_t, alt_t))
                    self.assertLess(distance, 0.2)
                    # not sure why 0.2 arcsec is the limiting precision of this test

    def test_alt_az_from_ra_dec(self):
        """
        Test conversion from RA, Dec to Alt, Az
        """

        n_samples = 100
        ra = self.rng.random_sample(n_samples) * 2.0 * np.pi
        dec = (self.rng.random_sample(n_samples) - 0.5) * np.pi
        lon_rad = 1.467
        lat_rad = -0.234
        control_alt, control_az = control_alt_az_from_ra_dec(ra, dec, lon_rad, lat_rad, self.mjd)

        obs = utils.ObservationMetaData(
            mjd=utils.ModifiedJulianDate(utc=self.mjd),
            site=utils.Site(longitude=np.degrees(lon_rad), latitude=np.degrees(lat_rad), name="LSST"),
        )

        # verify parallactic angle against an expression from
        # http://www.astro.washington.edu/groups/APO/Mirror.Motions/Feb.2000.Image.Jumps/report.html#Image%20motion%20directions
        #
        ra_obs, dec_obs = utils._observed_from_icrs(
            ra, dec, obs_metadata=obs, epoch=2000.0, include_refraction=True
        )

        lmst, last = utils.calc_lmst_last(obs.mjd.ut1, lon_rad)
        hour_angle = np.radians(last * 15.0) - ra_obs
        control_sin_pa = np.sin(hour_angle) * np.cos(lat_rad) / np.cos(control_alt)

        test_alt, test_az, test_pa = utils._alt_az_pa_from_ra_dec(ra, dec, obs)

        distance = utils.arcsec_from_radians(utils.haversine(control_az, control_alt, test_az, test_alt))
        self.assertLess(distance.max(), 0.0001)
        self.assertLess(np.abs(np.sin(test_pa) - control_sin_pa).max(), self.tolerance)

        # test non-vectorized version
        for r, d in zip(ra, dec):
            control_alt, control_az = control_alt_az_from_ra_dec(r, d, lon_rad, lat_rad, self.mjd)
            test_alt, test_az, test_pa = utils._alt_az_pa_from_ra_dec(r, d, obs)
            lmst, last = utils.calc_lmst_last(obs.mjd.ut1, lon_rad)
            r_obs, dec_obs = utils._observed_from_icrs(
                r, d, obs_metadata=obs, epoch=2000.0, include_refraction=True
            )
            hour_angle = np.radians(last * 15.0) - r_obs
            control_sin_pa = np.sin(hour_angle) * np.cos(lat_rad) / np.cos(control_alt)
            distance = utils.arcsec_from_radians(utils.haversine(control_az, control_alt, test_az, test_alt))
            self.assertLess(distance, 0.0001)
            self.assertLess(np.abs(np.sin(test_pa) - control_sin_pa), self.tolerance)

    def test_alt_az_pa_from_ra_dec_no_refraction(self):
        """
        Test that alt_az_pa_from_ra_dec gives a sane answer when you turn off
        refraction.
        """

        rng = np.random.RandomState(44)
        n_samples = 10
        n_batches = 10
        for i_batch in range(n_batches):
            # first, generate some sane RA, Dec values by generating sane
            # Alt, Az values with refraction and converting them into
            # RA, Dec
            alt_sane = rng.random_sample(n_samples) * 45.0 + 45.0
            az_sane = rng.random_sample(n_samples) * 360.0
            mjd_input = rng.random_sample(n_samples) * 10000.0 + 40000.0
            mjd_list = utils.ModifiedJulianDate.get_list(TAI=mjd_input)

            ra_sane = []
            dec_sane = []
            obs_sane = []
            for alt, az, mjd in zip(alt_sane, az_sane, mjd_list):
                obs = utils.ObservationMetaData(mjd=mjd)
                ra, dec = utils.ra_dec_from_alt_az(alt, az, obs)
                ra_sane.append(ra)
                dec_sane.append(dec)
                obs_sane.append(obs)

            # Now, loop over our refracted RA, Dec, Alt, Az values.
            # Convert from RA, Dec to unrefracted Alt, Az.  Then, apply refraction
            # with our apply_refraction method.  Check that the resulting refracted
            # zenith distance is:
            #    1) within 0.1 arcsec of the zenith distance of the already refracted
            #       alt value calculated above
            #
            #    2) closer to the zenith distance calculated above than to the
            #       unrefracted zenith distance
            for ra, dec, obs, alt_ref, az_ref in zip(ra_sane, dec_sane, obs_sane, alt_sane, az_sane):
                alt, az, pa = utils.alt_az_pa_from_ra_dec(ra, dec, obs, include_refraction=False)

                tanz, tanz3 = utils.refraction_coefficients(site=obs.site)
                refracted_zd = utils.apply_refraction(np.radians(90.0 - alt), tanz, tanz3)

                # Check that the two independently refracted zenith distances agree
                # to within 0.1 arcsec
                self.assertLess(
                    np.abs(
                        utils.arcsec_from_radians(refracted_zd)
                        - utils.arcsec_from_radians(np.radians(90.0 - alt_ref))
                    ),
                    0.1,
                )

                # Check that the two refracted zenith distances are closer to each other
                # than to the unrefracted zenith distance
                self.assertLess(
                    np.abs(np.degrees(refracted_zd) - (90.0 - alt_ref)),
                    np.abs((90.0 - alt_ref) - (90.0 - alt)),
                )

                self.assertLess(
                    np.abs(np.degrees(refracted_zd) - (90.0 - alt_ref)),
                    np.abs(np.degrees(refracted_zd) - (90.0 - alt)),
                )

    def test_ra_dec_from_alt_az_noref(self):
        """
        test that ra_dec_from_alt_az correctly inverts alt_az_pa_from_ra_dec, even when
        refraction is turned off
        """

        rng = np.random.RandomState(55)
        n_samples = 10
        n_batches = 10

        for i_batch in range(n_batches):
            d_sun = 0.0
            while d_sun < 45.0:  # because ICRS->Observed transformation breaks down close to the sun
                alt_in = rng.random_sample(n_samples) * 50.0 + 20.0
                az_in = rng.random_sample(n_samples) * 360.0
                obs = utils.ObservationMetaData(mjd=43000.0)
                ra_in, dec_in = utils.ra_dec_from_alt_az(alt_in, az_in, obs=obs, include_refraction=False)

                d_sun = utils.distance_to_sun(ra_in, dec_in, obs.mjd).min()

            alt_out, az_out, pa_out = utils.alt_az_pa_from_ra_dec(
                ra_in, dec_in, obs=obs, include_refraction=False
            )

            dd = utils.haversine(
                np.radians(alt_out),
                np.radians(az_out),
                np.radians(alt_in),
                np.radians(az_in),
            )
            self.assertLess(utils.arcsec_from_radians(dd).max(), 0.01)

    def test_ra_dec_alt_az_no_refraction_deg_vs_radians(self):
        """
        Check that ra_dec_from_alt_az and alt_az_pa_from_ra_dec are consistent in a degrees-versus-radians
        sense when refraction is turned off
        """

        rng = np.random.RandomState(34)
        n_samples = 10
        ra_in = rng.random_sample(n_samples) * 360.0
        dec_in = rng.random_sample(n_samples) * 180.0 - 90.0
        mjd = 43000.0
        obs = utils.ObservationMetaData(mjd=mjd)
        alt, az, pa = utils.alt_az_pa_from_ra_dec(ra_in, dec_in, obs, include_refraction=False)
        alt_rad, az_rad, pa_rad = utils._alt_az_pa_from_ra_dec(
            np.radians(ra_in), np.radians(dec_in), obs, include_refraction=False
        )

        distance = utils.haversine(az_rad, alt_rad, np.radians(az), np.radians(alt))
        self.assertLess(utils.arcsec_from_radians(distance).min(), 0.001)
        np.testing.assert_array_almost_equal(pa, np.degrees(pa_rad), decimal=12)

        ra, dec = utils.ra_dec_from_alt_az(alt, az, obs, include_refraction=False)
        ra_rad, dec_rad = utils._ra_dec_from_alt_az(alt_rad, az_rad, obs, include_refraction=False)
        distance = utils.haversine(ra_rad, dec_rad, np.radians(ra), np.radians(dec))
        self.assertLess(utils.arcsec_from_radians(distance).min(), 0.001)


if __name__ == "__main__":
    unittest.main()
