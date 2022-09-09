import unittest
import numpy as np
import rubin_sim.utils as utils


def controlAltAzFromRaDec(raRad_in, decRad_in, longRad, latRad, mjd):
    """
    Converts RA and Dec to altitude and azimuth

    @param [in] raRad is the RA in radians
    (observed geocentric)

    @param [in] decRad is the Dec in radians
    (observed geocentric)

    @param [in] longRad is the longitude of the observer in radians
    (positive east of the prime meridian)

    @param [in[ latRad is the latitude of the observer in radians
    (positive north of the equator)

    @param [in] mjd is the universal time expressed as an MJD

    @param [out] altitude in radians

    @param [out[ azimuth in radians

    see: http://www.stargazing.net/kepler/altaz.html#twig04
    """
    obs = utils.ObservationMetaData(
        mjd=utils.ModifiedJulianDate(UTC=mjd),
        site=utils.Site(
            longitude=np.degrees(longRad), latitude=np.degrees(latRad), name="LSST"
        ),
    )

    if hasattr(raRad_in, "__len__"):
        raRad, decRad = utils._observedFromICRS(
            raRad_in, decRad_in, obs_metadata=obs, epoch=2000.0, includeRefraction=True
        )
    else:
        raRad, decRad = utils._observedFromICRS(
            raRad_in, decRad_in, obs_metadata=obs, epoch=2000.0, includeRefraction=True
        )

    lst = utils.calcLmstLast(obs.mjd.UT1, longRad)
    last = lst[1]
    haRad = np.radians(last * 15.0) - raRad

    sinDec = np.sin(decRad)
    cosLat = np.cos(latRad)
    sinLat = np.sin(latRad)
    sinAlt = sinDec * sinLat + np.cos(decRad) * cosLat * np.cos(haRad)
    altRad = np.arcsin(sinAlt)
    azRad = np.arccos((sinDec - sinAlt * sinLat) / (np.cos(altRad) * cosLat))
    azRadOut = np.where(np.sin(haRad) >= 0.0, 2.0 * np.pi - azRad, azRad)
    if isinstance(altRad, float):
        return altRad, float(azRadOut)
    return altRad, azRadOut


class CompoundCoordinateTransformationsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(32)
        self.mjd = 57087.0
        self.tolerance = 1.0e-5

    def testExceptions(self):
        """
        Test to make sure that methods complain when incorrect data types are passed.
        """
        obs = utils.ObservationMetaData(pointingRA=55.0, pointingDec=-72.0, mjd=53467.8)

        raFloat = 1.1
        raList = np.array([0.2, 0.3])

        decFloat = 1.1
        decList = np.array([0.2, 0.3])

        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raFloat, decList, obs)
        utils._altAzPaFromRaDec(raFloat, decFloat, obs)
        utils._altAzPaFromRaDec(raList, decList, obs)

        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raFloat, decList, obs)
        utils._raDecFromAltAz(raFloat, decFloat, obs)
        utils._raDecFromAltAz(raList, decList, obs)

        self.assertRaises(RuntimeError, utils.altAzPaFromRaDec, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils.altAzPaFromRaDec, raFloat, decList, obs)
        utils.altAzPaFromRaDec(raFloat, decFloat, obs)
        utils.altAzPaFromRaDec(raList, decList, obs)

        self.assertRaises(RuntimeError, utils.raDecFromAltAz, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils.raDecFromAltAz, raFloat, decList, obs)
        utils.raDecFromAltAz(raFloat, decFloat, obs)
        utils.raDecFromAltAz(raList, decList, obs)

    def test_raDecFromAltAz(self):
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
        mjd_list.append(2457364.958333 - 2400000.5)  # 8 December 2015 11:00 UTC
        alt_list.append(np.radians(41.1))
        az_list.append(np.radians(134.7))
        ra_app_list.append(16.0 * hours + 59.0 * minutes + 16.665 * seconds)
        dec_app_list.append(np.radians(-22.0 - 42.0 / 60.0 - 2.94 / 3600.0))

        longitude_list.append(np.radians(-22.0 - 33.0 / 60.0))
        latitude_list.append(np.radians(11.0 + 45.0 / 60.0))
        mjd_list.append(2457368.958333 - 2400000.5)  # 12 December 2015 11:00 UTC
        alt_list.append(np.radians(40.5))
        az_list.append(np.radians(134.7))
        ra_app_list.append(17.0 * hours + 16.0 * minutes + 51.649 * seconds)
        dec_app_list.append(np.radians(-23.0 - 3 / 60.0 - 50.35 / 3600.0))

        longitude_list.append(np.radians(145.0 + 23.0 / 60.0))
        latitude_list.append(np.radians(-64.0 - 5.0 / 60.0))
        mjd_list.append(2456727.583333 - 2400000.5)  # 11 March 2014, 02:00 UTC
        alt_list.append(np.radians(29.5))
        az_list.append(np.radians(8.2))
        ra_app_list.append(23.0 * hours + 24.0 * minutes + 46.634 * seconds)
        dec_app_list.append(np.radians(-3.0 - 47.0 / 60.0 - 47.81 / 3600.0))

        longitude_list.append(np.radians(145.0 + 23.0 / 60.0))
        latitude_list.append(np.radians(-64.0 - 5.0 / 60.0))
        mjd_list.append(2456731.583333 - 2400000.5)  # 15 March 2014, 02:00 UTC
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
                mjd=utils.ModifiedJulianDate(UTC=mjd),
            )

            ra_icrs, dec_icrs = utils._raDecFromAltAz(alt, az, obs)
            ra_test, dec_test = utils._appGeoFromICRS(ra_icrs, dec_icrs, mjd=obs.mjd)

            distance = np.degrees(utils.haversine(ra_app, dec_app, ra_test, dec_test))
            # this is all the precision we have in the alt,az data taken from the USNO
            self.assertLess(distance, 0.1)

            correction = np.degrees(
                utils.haversine(ra_test, dec_test, ra_icrs, dec_icrs)
            )
            self.assertLess(distance, correction)

    def testAltAzRADecRoundTrip(self):
        """
        Test that altAzPaFromRaDec and raDecFromAltAz really invert each other
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

                ra_in, dec_in = utils.raDecFromAltAz(alt_in, az_in, obs)

                self.assertIsInstance(ra_in, np.ndarray)
                self.assertIsInstance(dec_in, np.ndarray)

                self.assertFalse(np.isnan(ra_in).any(), msg="there were NaNs in ra_in")
                self.assertFalse(
                    np.isnan(dec_in).any(), msg="there were NaNs in dec_in"
                )

                # test that passing them in one at a time gives the same answer
                for ix in range(len(alt_in)):
                    ra_f, dec_f = utils.raDecFromAltAz(alt_in[ix], az_in[ix], obs)
                    self.assertIsInstance(ra_f, float)
                    self.assertIsInstance(dec_f, float)
                    self.assertAlmostEqual(ra_f, ra_in[ix], 12)
                    self.assertAlmostEqual(dec_f, dec_in[ix], 12)

                alt_out, az_out, pa_out = utils.altAzPaFromRaDec(ra_in, dec_in, obs)

                self.assertFalse(
                    np.isnan(pa_out).any(), msg="there were NaNs in pa_out"
                )

                for alt_c, az_c, alt_t, az_t in zip(
                    np.radians(alt_in),
                    np.radians(az_in),
                    np.radians(alt_out),
                    np.radians(az_out),
                ):
                    distance = utils.arcsecFromRadians(
                        utils.haversine(az_c, alt_c, az_t, alt_t)
                    )
                    self.assertLess(distance, 0.2)
                    # not sure why 0.2 arcsec is the limiting precision of this test

    def testAltAzFromRaDec(self):
        """
        Test conversion from RA, Dec to Alt, Az
        """

        nSamples = 100
        ra = self.rng.random_sample(nSamples) * 2.0 * np.pi
        dec = (self.rng.random_sample(nSamples) - 0.5) * np.pi
        lon_rad = 1.467
        lat_rad = -0.234
        controlAlt, controlAz = controlAltAzFromRaDec(
            ra, dec, lon_rad, lat_rad, self.mjd
        )

        obs = utils.ObservationMetaData(
            mjd=utils.ModifiedJulianDate(UTC=self.mjd),
            site=utils.Site(
                longitude=np.degrees(lon_rad), latitude=np.degrees(lat_rad), name="LSST"
            ),
        )

        # verify parallactic angle against an expression from
        # http://www.astro.washington.edu/groups/APO/Mirror.Motions/Feb.2000.Image.Jumps/report.html#Image%20motion%20directions
        #
        ra_obs, dec_obs = utils._observedFromICRS(
            ra, dec, obs_metadata=obs, epoch=2000.0, includeRefraction=True
        )

        lmst, last = utils.calcLmstLast(obs.mjd.UT1, lon_rad)
        hourAngle = np.radians(last * 15.0) - ra_obs
        controlSinPa = np.sin(hourAngle) * np.cos(lat_rad) / np.cos(controlAlt)

        testAlt, testAz, testPa = utils._altAzPaFromRaDec(ra, dec, obs)

        distance = utils.arcsecFromRadians(
            utils.haversine(controlAz, controlAlt, testAz, testAlt)
        )
        self.assertLess(distance.max(), 0.0001)
        self.assertLess(np.abs(np.sin(testPa) - controlSinPa).max(), self.tolerance)

        # test non-vectorized version
        for r, d in zip(ra, dec):
            controlAlt, controlAz = controlAltAzFromRaDec(
                r, d, lon_rad, lat_rad, self.mjd
            )
            testAlt, testAz, testPa = utils._altAzPaFromRaDec(r, d, obs)
            lmst, last = utils.calcLmstLast(obs.mjd.UT1, lon_rad)
            r_obs, dec_obs = utils._observedFromICRS(
                r, d, obs_metadata=obs, epoch=2000.0, includeRefraction=True
            )
            hourAngle = np.radians(last * 15.0) - r_obs
            controlSinPa = np.sin(hourAngle) * np.cos(lat_rad) / np.cos(controlAlt)
            distance = utils.arcsecFromRadians(
                utils.haversine(controlAz, controlAlt, testAz, testAlt)
            )
            self.assertLess(distance, 0.0001)
            self.assertLess(np.abs(np.sin(testPa) - controlSinPa), self.tolerance)

    def test_altAzPaFromRaDec_no_refraction(self):
        """
        Test that altAzPaFromRaDec gives a sane answer when you turn off
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
                ra, dec = utils.raDecFromAltAz(alt, az, obs)
                ra_sane.append(ra)
                dec_sane.append(dec)
                obs_sane.append(obs)

            # Now, loop over our refracted RA, Dec, Alt, Az values.
            # Convert from RA, Dec to unrefracted Alt, Az.  Then, apply refraction
            # with our applyRefraction method.  Check that the resulting refracted
            # zenith distance is:
            #    1) within 0.1 arcsec of the zenith distance of the already refracted
            #       alt value calculated above
            #
            #    2) closer to the zenith distance calculated above than to the
            #       unrefracted zenith distance
            for ra, dec, obs, alt_ref, az_ref in zip(
                ra_sane, dec_sane, obs_sane, alt_sane, az_sane
            ):

                alt, az, pa = utils.altAzPaFromRaDec(
                    ra, dec, obs, includeRefraction=False
                )

                tanz, tanz3 = utils.refractionCoefficients(site=obs.site)
                refracted_zd = utils.applyRefraction(
                    np.radians(90.0 - alt), tanz, tanz3
                )

                # Check that the two independently refracted zenith distances agree
                # to within 0.1 arcsec
                self.assertLess(
                    np.abs(
                        utils.arcsecFromRadians(refracted_zd)
                        - utils.arcsecFromRadians(np.radians(90.0 - alt_ref))
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

    def test_raDecFromAltAz_noref(self):
        """
        test that raDecFromAltAz correctly inverts altAzPaFromRaDec, even when
        refraction is turned off
        """

        rng = np.random.RandomState(55)
        n_samples = 10
        n_batches = 10

        for i_batch in range(n_batches):
            d_sun = 0.0
            while (
                d_sun < 45.0
            ):  # because ICRS->Observed transformation breaks down close to the sun

                alt_in = rng.random_sample(n_samples) * 50.0 + 20.0
                az_in = rng.random_sample(n_samples) * 360.0
                obs = utils.ObservationMetaData(mjd=43000.0)
                ra_in, dec_in = utils.raDecFromAltAz(
                    alt_in, az_in, obs=obs, includeRefraction=False
                )

                d_sun = utils.distanceToSun(ra_in, dec_in, obs.mjd).min()

            alt_out, az_out, pa_out = utils.altAzPaFromRaDec(
                ra_in, dec_in, obs=obs, includeRefraction=False
            )

            dd = utils.haversine(
                np.radians(alt_out),
                np.radians(az_out),
                np.radians(alt_in),
                np.radians(az_in),
            )
            self.assertLess(utils.arcsecFromRadians(dd).max(), 0.01)

    def test_raDecAltAz_noRefraction_degVsRadians(self):
        """
        Check that raDecFromAltAz and altAzPaFromRaDec are consistent in a degrees-versus-radians
        sense when refraction is turned off
        """

        rng = np.random.RandomState(34)
        n_samples = 10
        ra_in = rng.random_sample(n_samples) * 360.0
        dec_in = rng.random_sample(n_samples) * 180.0 - 90.0
        mjd = 43000.0
        obs = utils.ObservationMetaData(mjd=mjd)
        alt, az, pa = utils.altAzPaFromRaDec(
            ra_in, dec_in, obs, includeRefraction=False
        )
        alt_rad, az_rad, pa_rad = utils._altAzPaFromRaDec(
            np.radians(ra_in), np.radians(dec_in), obs, includeRefraction=False
        )

        distance = utils.haversine(az_rad, alt_rad, np.radians(az), np.radians(alt))
        self.assertLess(utils.arcsecFromRadians(distance).min(), 0.001)
        np.testing.assert_array_almost_equal(pa, np.degrees(pa_rad), decimal=12)

        ra, dec = utils.raDecFromAltAz(alt, az, obs, includeRefraction=False)
        ra_rad, dec_rad = utils._raDecFromAltAz(
            alt_rad, az_rad, obs, includeRefraction=False
        )
        distance = utils.haversine(ra_rad, dec_rad, np.radians(ra), np.radians(dec))
        self.assertLess(utils.arcsecFromRadians(distance).min(), 0.001)


if __name__ == "__main__":
    unittest.main()
