import unittest

import numpy as np

from rubin_sim.utils import (
    ModifiedJulianDate,
    ObservationMetaData,
    _icrs_from_observed,
    _native_lon_lat_from_ra_dec,
    _observed_from_icrs,
    _observed_from_pupil_coords,
    _pupil_coords_from_ra_dec,
    _ra_dec_from_pupil_coords,
    arcsec_from_radians,
    distance_to_sun,
    haversine,
    icrs_from_observed,
    observed_from_icrs,
    observed_from_pupil_coords,
    pupil_coords_from_ra_dec,
    ra_dec_from_alt_az,
    radians_from_arcsec,
    solar_ra_dec,
)


class PupilCoordinateUnitTest(unittest.TestCase):
    long_message = True

    def test_exceptions(self):
        """
        Test that exceptions are raised when they ought to be
        """
        obs_metadata = ObservationMetaData(pointing_ra=25.0, pointing_dec=25.0, rot_sky_pos=25.0, mjd=52000.0)

        rng = np.random.RandomState(42)
        ra = rng.random_sample(10) * np.radians(1.0) + np.radians(obs_metadata.pointing_ra)
        dec = rng.random_sample(10) * np.radians(1.0) + np.radians(obs_metadata.pointing_dec)
        ra_short = np.array([1.0])
        dec_short = np.array([1.0])

        # test without obs_metadata
        self.assertRaises(RuntimeError, _pupil_coords_from_ra_dec, ra, dec, epoch=2000.0)

        # test without pointing_ra
        dummy = ObservationMetaData(
            pointing_dec=obs_metadata.pointing_dec,
            rot_sky_pos=obs_metadata.rot_sky_pos,
            mjd=obs_metadata.mjd,
        )
        self.assertRaises(
            RuntimeError,
            _pupil_coords_from_ra_dec,
            ra,
            dec,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        # test without pointing_dec
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            rot_sky_pos=obs_metadata.rot_sky_pos,
            mjd=obs_metadata.mjd,
        )
        self.assertRaises(
            RuntimeError,
            _pupil_coords_from_ra_dec,
            ra,
            dec,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        # test without rot_sky_pos
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            mjd=obs_metadata.mjd,
        )
        self.assertRaises(
            RuntimeError,
            _pupil_coords_from_ra_dec,
            ra,
            dec,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        # test without mjd
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            rot_sky_pos=obs_metadata.rot_sky_pos,
        )
        self.assertRaises(
            RuntimeError,
            _pupil_coords_from_ra_dec,
            ra,
            dec,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        # test for mismatches
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            rot_sky_pos=obs_metadata.rot_sky_pos,
            mjd=obs_metadata.mjd,
        )

        self.assertRaises(
            RuntimeError,
            _pupil_coords_from_ra_dec,
            ra,
            dec_short,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        self.assertRaises(
            RuntimeError,
            _pupil_coords_from_ra_dec,
            ra_short,
            dec,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        # test that it actually runs (and that passing in either numpy arrays or floats gives
        # the same results)
        xx_arr, yy_arr = _pupil_coords_from_ra_dec(ra, dec, obs_metadata=obs_metadata)
        self.assertIsInstance(xx_arr, np.ndarray)
        self.assertIsInstance(yy_arr, np.ndarray)

        for ix in range(len(ra)):
            xx_f, yy_f = _pupil_coords_from_ra_dec(ra[ix], dec[ix], obs_metadata=obs_metadata)
            self.assertIsInstance(xx_f, float)
            self.assertIsInstance(yy_f, float)
            self.assertAlmostEqual(xx_arr[ix], xx_f, 12)
            self.assertAlmostEqual(yy_arr[ix], yy_f, 12)
            self.assertFalse(np.isnan(xx_f))
            self.assertFalse(np.isnan(yy_f))

    def test_cardinal_directions(self):
        """
        This unit test verifies that the following conventions hold:

        if rot_sky_pos = 0, then north is +y the camera and east is +x

        if rot_sky_pos = -90, then north is -x on the camera and east is +y

        if rot_sky_pos = 90, then north is +x on the camera and east is -y

        if rot_sky_pos = 180, then north is -y on the camera and east is -x

        This is consistent with rot_sky_pos = rotTelPos - parallacticAngle

        parallacticAngle is negative when the pointing is east of the meridian.
        http://www.petermeadows.com/html/parallactic.html

        rotTelPos is the angle between up on the telescope and up on
        the camera, where positive rotTelPos goes from north to west
        (from an email sent to me by LynneJones)

        I have verified that OpSim follows the rot_sky_pos = rotTelPos - paralacticAngle
        convention.

        I have verified that alt_az_pa_from_ra_dec follows the convention that objects
        east of the meridian have a negative parallactic angle.  (alt_az_pa_from_ra_dec
        uses PALPY under the hood, so it can probably be taken as correct)

        It will verify this convention for multiple random pointings.
        """

        epoch = 2000.0
        mjd = 42350.0
        rng = np.random.RandomState(42)
        ra_list = rng.random_sample(10) * 360.0
        dec_list = rng.random_sample(10) * 180.0 - 90.0

        for rot_sky_pos in np.arange(-90.0, 181.0, 90.0):
            for ra, dec in zip(ra_list, dec_list):
                obs = ObservationMetaData(pointing_ra=ra, pointing_dec=dec, mjd=mjd, rot_sky_pos=rot_sky_pos)

                ra_obs, dec_obs = _observed_from_icrs(
                    np.radians([ra]),
                    np.radians([dec]),
                    obs_metadata=obs,
                    epoch=2000.0,
                    include_refraction=True,
                )

                # test points that are displaced just to the (E, W, N, S) of the pointing
                # in observed geocentric RA, Dec; verify that the pupil coordinates
                # change as expected
                ra_test_obs = ra_obs[0] + np.array([0.01, -0.01, 0.0, 0.0])
                dec_test_obs = dec_obs[0] + np.array([0.0, 0.0, 0.01, -0.01])
                ra_test, dec_test = _icrs_from_observed(
                    ra_test_obs,
                    dec_test_obs,
                    obs_metadata=obs,
                    epoch=2000.0,
                    include_refraction=True,
                )

                x, y = _pupil_coords_from_ra_dec(ra_test, dec_test, obs_metadata=obs, epoch=epoch)

                lon, lat = _native_lon_lat_from_ra_dec(ra_test, dec_test, obs)
                rr = np.abs(np.cos(lat) / np.sin(lat))

                if np.abs(rot_sky_pos) < 0.01:  # rot_sky_pos == 0
                    control_x = np.array([1.0 * rr[0], -1.0 * rr[1], 0.0, 0.0])
                    control_y = np.array([0.0, 0.0, 1.0 * rr[2], -1.0 * rr[3]])
                elif np.abs(rot_sky_pos + 90.0) < 0.01:  # rot_sky_pos == -90
                    control_x = np.array([0.0, 0.0, -1.0 * rr[2], 1.0 * rr[3]])
                    control_y = np.array([1.0 * rr[0], -1.0 * rr[1], 0.0, 0.0])
                elif np.abs(rot_sky_pos - 90.0) < 0.01:  # rot_sky_pos == 90
                    control_x = np.array([0.0, 0.0, 1.0 * rr[2], -1.0 * rr[3]])
                    control_y = np.array([-1.0 * rr[0], +1.0 * rr[1], 0.0, 0.0])
                elif np.abs(rot_sky_pos - 180.0) < 0.01:  # rot_sky_pos == 180
                    control_x = np.array([-1.0 * rr[0], +1.0 * rr[1], 0.0, 0.0])
                    control_y = np.array([0.0, 0.0, -1.0 * rr[2], 1.0 * rr[3]])

                msg = "failed on rot_sky_pos == %e\n" % rot_sky_pos
                msg += "control_x %s\n" % str(control_x)
                msg += "test_x %s\n" % str(x)
                msg += "control_y %s\n" % str(control_y)
                msg += "test_y %s\n" % str(y)

                dx = np.array([xx / cc if np.abs(cc) > 1.0e-10 else 1.0 - xx for xx, cc in zip(x, control_x)])
                dy = np.array([yy / cc if np.abs(cc) > 1.0e-10 else 1.0 - yy for yy, cc in zip(y, control_y)])
                self.assertLess(np.abs(dx - np.ones(4)).max(), 0.001, msg=msg)
                self.assertLess(np.abs(dy - np.ones(4)).max(), 0.001, msg=msg)

    def test_ra_dec_from_pupil(self):
        """
        Test conversion from pupil coordinates back to Ra, Dec
        """

        mjd = ModifiedJulianDate(TAI=52000.0)
        solar_ra, solar_dec = solar_ra_dec(mjd)

        # to make sure that we are more than 45 degrees from the Sun as required
        # for _icrs_from_observed to be at all accurate
        ra_center = solar_ra + 100.0
        dec_center = solar_dec - 30.0

        obs = ObservationMetaData(
            pointing_ra=ra_center,
            pointing_dec=dec_center,
            bound_type="circle",
            bound_length=0.1,
            rot_sky_pos=23.0,
            mjd=mjd,
        )

        n_samples = 1000
        rng = np.random.RandomState(42)
        ra = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(ra_center)
        dec = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(dec_center)
        xp, yp = _pupil_coords_from_ra_dec(ra, dec, obs_metadata=obs, epoch=2000.0)

        ra_test, dec_test = _ra_dec_from_pupil_coords(xp, yp, obs_metadata=obs, epoch=2000.0)

        distance = arcsec_from_radians(haversine(ra, dec, ra_test, dec_test))

        dex = np.argmax(distance)

        worst_solar_distance = distance_to_sun(np.degrees(ra[dex]), np.degrees(dec[dex]), mjd)

        msg = "_ra_dec_from_pupil_coords off by %e arcsec at distance to Sun of %e degrees" % (
            distance.max(),
            worst_solar_distance,
        )

        self.assertLess(distance.max(), 1.0e-6, msg=msg)

        # now check that passing in the xp, yp values one at a time still gives
        # the right answer
        for ix in range(len(ra)):
            ra_f, dec_f = _ra_dec_from_pupil_coords(xp[ix], yp[ix], obs_metadata=obs, epoch=2000.0)
            self.assertIsInstance(ra_f, float)
            self.assertIsInstance(dec_f, float)
            dist_f = arcsec_from_radians(haversine(ra_f, dec_f, ra_test[ix], dec_test[ix]))
            self.assertLess(dist_f, 1.0e-9)

    def test_ra_dec_from_pupil_no_refraction(self):
        """
        Test conversion from pupil coordinates back to Ra, Dec
        with include_refraction=False
        """

        mjd = ModifiedJulianDate(TAI=52000.0)
        solar_ra, solar_dec = solar_ra_dec(mjd)

        # to make sure that we are more than 45 degrees from the Sun as required
        # for _icrs_from_observed to be at all accurate
        ra_center = solar_ra + 100.0
        dec_center = solar_dec - 30.0

        obs = ObservationMetaData(
            pointing_ra=ra_center,
            pointing_dec=dec_center,
            bound_type="circle",
            bound_length=0.1,
            rot_sky_pos=23.0,
            mjd=mjd,
        )

        n_samples = 1000
        rng = np.random.RandomState(42)
        ra = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(ra_center)
        dec = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(dec_center)
        xp, yp = _pupil_coords_from_ra_dec(ra, dec, obs_metadata=obs, epoch=2000.0, include_refraction=False)

        ra_test, dec_test = _ra_dec_from_pupil_coords(
            xp, yp, obs_metadata=obs, epoch=2000.0, include_refraction=False
        )

        distance = arcsec_from_radians(haversine(ra, dec, ra_test, dec_test))

        dex = np.argmax(distance)

        worst_solar_distance = distance_to_sun(np.degrees(ra[dex]), np.degrees(dec[dex]), mjd)

        msg = "_ra_dec_from_pupil_coords off by %e arcsec at distance to Sun of %e degrees" % (
            distance.max(),
            worst_solar_distance,
        )

        self.assertLess(distance.max(), 1.0e-6, msg=msg)

        # now check that passing in the xp, yp values one at a time still gives
        # the right answer
        for ix in range(len(ra)):
            ra_f, dec_f = _ra_dec_from_pupil_coords(
                xp[ix], yp[ix], obs_metadata=obs, epoch=2000.0, include_refraction=False
            )
            self.assertIsInstance(ra_f, float)
            self.assertIsInstance(dec_f, float)
            dist_f = arcsec_from_radians(haversine(ra_f, dec_f, ra_test[ix], dec_test[ix]))
            self.assertLess(dist_f, 1.0e-9)

    def test_observed_from_pupil(self):
        """
        Test conversion from pupil coordinates to observed coordinates
        """

        mjd = ModifiedJulianDate(TAI=53000.0)
        solar_ra, solar_dec = solar_ra_dec(mjd)

        # to make sure that we are more than 45 degrees from the Sun as required
        # for _icrs_from_observed to be at all accurate
        ra_center = solar_ra + 100.0
        dec_center = solar_dec - 30.0

        obs = ObservationMetaData(
            pointing_ra=ra_center,
            pointing_dec=dec_center,
            bound_type="circle",
            bound_length=0.1,
            rot_sky_pos=23.0,
            mjd=mjd,
        )

        n_samples = 1000
        rng = np.random.RandomState(4453)
        ra = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(ra_center)
        dec = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(dec_center)
        xp, yp = _pupil_coords_from_ra_dec(ra, dec, obs_metadata=obs, epoch=2000.0, include_refraction=True)

        ra_obs, dec_obs = _observed_from_icrs(
            ra, dec, obs_metadata=obs, epoch=2000.0, include_refraction=True
        )

        ra_obs_test, dec_obs_test = _observed_from_pupil_coords(
            xp, yp, obs_metadata=obs, epoch=2000.0, include_refraction=True
        )

        dist = arcsec_from_radians(haversine(ra_obs, dec_obs, ra_obs_test, dec_obs_test))
        self.assertLess(dist.max(), 1.0e-6)

        # test output in degrees
        ra_obs_deg, dec_obs_deg = observed_from_pupil_coords(
            xp, yp, obs_metadata=obs, epoch=2000.0, include_refraction=True
        )

        np.testing.assert_array_almost_equal(ra_obs_deg, np.degrees(ra_obs_test), decimal=16)
        np.testing.assert_array_almost_equal(dec_obs_deg, np.degrees(dec_obs_test), decimal=16)

        # test one-at-a-time input
        for ii in range(len(ra_obs)):
            rr, dd = _observed_from_pupil_coords(
                xp[ii], yp[ii], obs_metadata=obs, epoch=2000.0, include_refraction=True
            )
            self.assertAlmostEqual(rr, ra_obs_test[ii], 16)
            self.assertAlmostEqual(dd, dec_obs_test[ii], 16)

            rr, dd = observed_from_pupil_coords(
                xp[ii], yp[ii], obs_metadata=obs, epoch=2000.0, include_refraction=True
            )
            self.assertAlmostEqual(rr, ra_obs_deg[ii], 16)
            self.assertAlmostEqual(dd, dec_obs_deg[ii], 16)

    def test_observed_from_pupil_no_refraction(self):
        """
        Test conversion from pupil coordinates to observed coordinates
        when include_refraction=False
        """

        mjd = ModifiedJulianDate(TAI=53000.0)
        solar_ra, solar_dec = solar_ra_dec(mjd)

        # to make sure that we are more than 45 degrees from the Sun as required
        # for _icrs_from_observed to be at all accurate
        ra_center = solar_ra + 100.0
        dec_center = solar_dec - 30.0

        obs = ObservationMetaData(
            pointing_ra=ra_center,
            pointing_dec=dec_center,
            bound_type="circle",
            bound_length=0.1,
            rot_sky_pos=23.0,
            mjd=mjd,
        )

        n_samples = 1000
        rng = np.random.RandomState(4453)
        ra = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(ra_center)
        dec = (rng.random_sample(n_samples) * 0.1 - 0.2) + np.radians(dec_center)
        xp, yp = _pupil_coords_from_ra_dec(ra, dec, obs_metadata=obs, epoch=2000.0, include_refraction=False)

        ra_obs, dec_obs = _observed_from_icrs(
            ra, dec, obs_metadata=obs, epoch=2000.0, include_refraction=False
        )

        ra_obs_test, dec_obs_test = _observed_from_pupil_coords(
            xp, yp, obs_metadata=obs, epoch=2000.0, include_refraction=False
        )

        dist = arcsec_from_radians(haversine(ra_obs, dec_obs, ra_obs_test, dec_obs_test))
        self.assertLess(dist.max(), 1.0e-6)

        # test output in degrees
        ra_obs_deg, dec_obs_deg = observed_from_pupil_coords(
            xp, yp, obs_metadata=obs, epoch=2000.0, include_refraction=False
        )

        np.testing.assert_array_almost_equal(ra_obs_deg, np.degrees(ra_obs_test), decimal=16)
        np.testing.assert_array_almost_equal(dec_obs_deg, np.degrees(dec_obs_test), decimal=16)

        # test one-at-a-time input
        for ii in range(len(ra_obs)):
            rr, dd = _observed_from_pupil_coords(
                xp[ii], yp[ii], obs_metadata=obs, epoch=2000.0, include_refraction=False
            )
            self.assertAlmostEqual(rr, ra_obs_test[ii], 16)
            self.assertAlmostEqual(dd, dec_obs_test[ii], 16)

            rr, dd = observed_from_pupil_coords(
                xp[ii], yp[ii], obs_metadata=obs, epoch=2000.0, include_refraction=False
            )
            self.assertAlmostEqual(rr, ra_obs_deg[ii], 16)
            self.assertAlmostEqual(dd, dec_obs_deg[ii], 16)

    def test_na_ns(self):
        """
        Test how _pupil_coords_from_ra_dec handles improper values
        """
        obs = ObservationMetaData(pointing_ra=42.0, pointing_dec=-28.0, rot_sky_pos=111.0, mjd=42356.0)
        n_samples = 100
        rng = np.random.RandomState(42)
        ra_list = np.radians(rng.random_sample(n_samples) * 2.0 + 42.0)
        dec_list = np.radians(rng.random_sample(n_samples) * 2.0 - 28.0)

        x_control, y_control = _pupil_coords_from_ra_dec(ra_list, dec_list, obs_metadata=obs, epoch=2000.0)

        ra_list[5] = np.NaN
        dec_list[5] = np.NaN
        ra_list[15] = np.NaN
        dec_list[20] = np.NaN
        ra_list[30] = np.radians(42.0) + np.pi

        x_test, y_test = _pupil_coords_from_ra_dec(ra_list, dec_list, obs_metadata=obs, epoch=2000.0)

        for ix, (xc, yc, xt, yt) in enumerate(zip(x_control, y_control, x_test, y_test)):
            if ix != 5 and ix != 15 and ix != 20 and ix != 30:
                self.assertAlmostEqual(xc, xt, 10)
                self.assertAlmostEqual(yc, yt, 10)
                self.assertFalse(np.isnan(xt))
                self.assertFalse(np.isnan(yt))
            else:
                np.testing.assert_equal(xt, np.NaN)
                np.testing.assert_equal(yt, np.NaN)

    def test_with_proper_motion(self):
        """
        Test that calculating pupil coordinates in the presence of proper motion, parallax,
        and radial velocity is equivalent to
        observed_from_icrs -> icrs_from_observed -> pupil_coords_from_ra_dec
        (mostly to make surethat pupil_coords_from_ra_dec is correctly calling observed_from_icrs
        with non-zero proper motion, etc.)
        """
        rng = np.random.RandomState(38442)
        is_valid = False
        while not is_valid:
            mjd_tai = 59580.0 + 10000.0 * rng.random_sample()
            obs = ObservationMetaData(mjd=mjd_tai)
            ra, dec = ra_dec_from_alt_az(78.0, 112.0, obs)
            dd = distance_to_sun(ra, dec, obs.mjd)
            if dd > 45.0:
                is_valid = True

        n_obj = 1000
        rr = rng.random_sample(n_obj) * 2.0
        theta = rng.random_sample(n_obj) * 2.0 * np.pi
        ra_list = ra + rr * np.cos(theta)
        dec_list = dec + rr * np.sin(theta)
        obs = ObservationMetaData(pointing_ra=ra, pointing_dec=dec, mjd=mjd_tai, rot_sky_pos=19.0)

        pm_ra_list = rng.random_sample(n_obj) * 100.0 - 50.0
        pm_dec_list = rng.random_sample(n_obj) * 100.0 - 50.0
        px_list = rng.random_sample(n_obj) + 0.05
        v_rad_list = rng.random_sample(n_obj) * 600.0 - 300.0

        for include_refraction in (True, False):
            ra_obs, dec_obs = observed_from_icrs(
                ra_list,
                dec_list,
                pm_ra=pm_ra_list,
                pm_dec=pm_dec_list,
                parallax=px_list,
                v_rad=v_rad_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            ra_icrs, dec_icrs = icrs_from_observed(
                ra_obs,
                dec_obs,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            xp_control, yp_control = pupil_coords_from_ra_dec(
                ra_icrs,
                dec_icrs,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            xp_test, yp_test = pupil_coords_from_ra_dec(
                ra_list,
                dec_list,
                pm_ra=pm_ra_list,
                pm_dec=pm_dec_list,
                parallax=px_list,
                v_rad=v_rad_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            distance = arcsec_from_radians(
                np.sqrt(np.power(xp_test - xp_control, 2) + np.power(yp_test - yp_control, 2))
            )
            self.assertLess(distance.max(), 0.006)

            # now test it in radians
            xp_rad, yp_rad = _pupil_coords_from_ra_dec(
                np.radians(ra_list),
                np.radians(dec_list),
                pm_ra=radians_from_arcsec(pm_ra_list),
                pm_dec=radians_from_arcsec(pm_dec_list),
                parallax=radians_from_arcsec(px_list),
                v_rad=v_rad_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            np.testing.assert_array_equal(xp_rad, xp_test)
            np.testing.assert_array_equal(yp_rad, yp_test)

            # now test it with proper motion = 0
            ra_obs, dec_obs = observed_from_icrs(
                ra_list,
                dec_list,
                parallax=px_list,
                v_rad=v_rad_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            ra_icrs, dec_icrs = icrs_from_observed(
                ra_obs,
                dec_obs,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            xp_control, yp_control = pupil_coords_from_ra_dec(
                ra_icrs,
                dec_icrs,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            xp_test, yp_test = pupil_coords_from_ra_dec(
                ra_list,
                dec_list,
                parallax=px_list,
                v_rad=v_rad_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            distance = arcsec_from_radians(
                np.sqrt(np.power(xp_test - xp_control, 2) + np.power(yp_test - yp_control, 2))
            )
            self.assertLess(distance.max(), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
