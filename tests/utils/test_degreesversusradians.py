import unittest

import numpy as np

import rubin_sim.utils as utils
from rubin_sim.utils import ModifiedJulianDate, ObservationMetaData, Site


class TestDegrees(unittest.TestCase):
    """
    Test that all the pairs of methods that deal in degrees versus
    radians agree with each other.
    """

    def setUp(self):
        self.rng = np.random.RandomState(87334)
        self.ra_list = self.rng.random_sample(100) * 2.0 * np.pi
        self.dec_list = (self.rng.random_sample(100) - 0.5) * np.pi
        self.lon = self.rng.random_sample(1)[0] * 360.0
        self.lat = (self.rng.random_sample(1)[0] - 0.5) * 180.0

    def test_unit_conversion(self):
        """
        Test that arcsec_from_radians, arcsec_from_degrees,
        radians_from_arcsec, and degrees_from_arcsec are all
        self-consistent
        """

        rad_list = self.rng.random_sample(100) * 2.0 * np.pi
        deg_list = np.degrees(rad_list)

        arcsec_rad_list = utils.arcsec_from_radians(rad_list)
        arcsec_deg_list = utils.arcsec_from_degrees(deg_list)

        np.testing.assert_array_equal(arcsec_rad_list, arcsec_deg_list)

        arcsec_list = self.rng.random_sample(100) * 1.0
        rad_list = utils.radians_from_arcsec(arcsec_list)
        deg_list = utils.degrees_from_arcsec(arcsec_list)
        np.testing.assert_array_equal(np.radians(deg_list), rad_list)

    def test_galactic_from_equatorial(self):
        ra_list = self.ra_list
        dec_list = self.dec_list

        lon_rad, lat_rad = utils._galactic_from_equatorial(ra_list, dec_list)
        lon_deg, lat_deg = utils.galactic_from_equatorial(np.degrees(ra_list), np.degrees(dec_list))

        np.testing.assert_array_almost_equal(lon_rad, np.radians(lon_deg), 10)
        np.testing.assert_array_almost_equal(lat_rad, np.radians(lat_deg), 10)

        for ra, dec in zip(ra_list, dec_list):
            lon_rad, lat_rad = utils._galactic_from_equatorial(ra, dec)
            lon_deg, lat_deg = utils.galactic_from_equatorial(np.degrees(ra), np.degrees(dec))
            self.assertAlmostEqual(lon_rad, np.radians(lon_deg), 10)
            self.assertAlmostEqual(lat_rad, np.radians(lat_deg), 10)

    def test_equaorial_from_galactic(self):
        lon_list = self.ra_list
        lat_list = self.dec_list

        ra_rad, dec_rad = utils._equatorial_from_galactic(lon_list, lat_list)
        ra_deg, dec_deg = utils.equatorial_from_galactic(np.degrees(lon_list), np.degrees(lat_list))

        np.testing.assert_array_almost_equal(ra_rad, np.radians(ra_deg), 10)
        np.testing.assert_array_almost_equal(dec_rad, np.radians(dec_deg), 10)

        for lon, lat in zip(lon_list, lat_list):
            ra_rad, dec_rad = utils._equatorial_from_galactic(lon, lat)
            ra_deg, dec_deg = utils.equatorial_from_galactic(np.degrees(lon), np.degrees(lat))
            self.assertAlmostEqual(ra_rad, np.radians(ra_deg), 10)
            self.assertAlmostEqual(dec_rad, np.radians(dec_deg), 10)

    def test_alt_az_pa_from_ra_dec(self):
        mjd = 57432.7
        obs = ObservationMetaData(mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST"))

        alt_rad, az_rad, pa_rad = utils._alt_az_pa_from_ra_dec(self.ra_list, self.dec_list, obs)

        alt_deg, az_deg, pa_deg = utils.alt_az_pa_from_ra_dec(
            np.degrees(self.ra_list), np.degrees(self.dec_list), obs
        )

        np.testing.assert_array_almost_equal(alt_rad, np.radians(alt_deg), 10)
        np.testing.assert_array_almost_equal(az_rad, np.radians(az_deg), 10)
        np.testing.assert_array_almost_equal(pa_rad, np.radians(pa_deg), 10)

        alt_rad, az_rad, pa_rad = utils._alt_az_pa_from_ra_dec(self.ra_list, self.dec_list, obs)

        alt_deg, az_deg, pa_deg = utils.alt_az_pa_from_ra_dec(
            np.degrees(self.ra_list), np.degrees(self.dec_list), obs
        )

        np.testing.assert_array_almost_equal(alt_rad, np.radians(alt_deg), 10)
        np.testing.assert_array_almost_equal(az_rad, np.radians(az_deg), 10)
        np.testing.assert_array_almost_equal(pa_rad, np.radians(pa_deg), 10)

        for (
            ra,
            dec,
        ) in zip(self.ra_list, self.dec_list):
            alt_rad, az_rad, pa_rad = utils._alt_az_pa_from_ra_dec(ra, dec, obs)
            alt_deg, az_deg, pa_deg = utils.alt_az_pa_from_ra_dec(np.degrees(ra), np.degrees(dec), obs)

            self.assertAlmostEqual(alt_rad, np.radians(alt_deg), 10)
            self.assertAlmostEqual(az_rad, np.radians(az_deg), 10)
            self.assertAlmostEqual(pa_rad, np.radians(pa_deg), 10)

    def test_ra_dec_from_alt_az(self):
        az_list = self.ra_list
        alt_list = self.dec_list
        mjd = 47895.6
        obs = ObservationMetaData(mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST"))

        ra_rad, dec_rad = utils._ra_dec_from_alt_az(alt_list, az_list, obs)

        ra_deg, dec_deg = utils.ra_dec_from_alt_az(np.degrees(alt_list), np.degrees(az_list), obs)

        np.testing.assert_array_almost_equal(ra_rad, np.radians(ra_deg), 10)
        np.testing.assert_array_almost_equal(dec_rad, np.radians(dec_deg), 10)

        ra_rad, dec_rad = utils._ra_dec_from_alt_az(alt_list, az_list, obs)

        ra_deg, dec_deg = utils.ra_dec_from_alt_az(np.degrees(alt_list), np.degrees(az_list), obs)

        np.testing.assert_array_almost_equal(ra_rad, np.radians(ra_deg), 10)
        np.testing.assert_array_almost_equal(dec_rad, np.radians(dec_deg), 10)

        for alt, az in zip(alt_list, az_list):
            ra_rad, dec_rad = utils._ra_dec_from_alt_az(alt, az, obs)
            ra_deg, dec_deg = utils.ra_dec_from_alt_az(np.degrees(alt), np.degrees(az), obs)

            self.assertAlmostEqual(ra_rad, np.radians(ra_deg), 10)
            self.assertAlmostEqual(dec_rad, np.radians(dec_deg), 10)

    def test_get_rot_sky_pos(self):
        rot_tel_list = self.rng.random_sample(len(self.ra_list)) * 2.0 * np.pi
        mjd = 56321.8

        obs_temp = ObservationMetaData(mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST"))

        rot_sky_rad = utils._get_rot_sky_pos(self.ra_list, self.dec_list, obs_temp, rot_tel_list)

        rot_sky_deg = utils.get_rot_sky_pos(
            np.degrees(self.ra_list),
            np.degrees(self.dec_list),
            obs_temp,
            np.degrees(rot_tel_list),
        )

        np.testing.assert_array_almost_equal(rot_sky_rad, np.radians(rot_sky_deg), 10)

        rot_sky_rad = utils._get_rot_sky_pos(self.ra_list, self.dec_list, obs_temp, rot_tel_list[0])

        rot_sky_deg = utils.get_rot_sky_pos(
            np.degrees(self.ra_list),
            np.degrees(self.dec_list),
            obs_temp,
            np.degrees(rot_tel_list[0]),
        )

        np.testing.assert_array_almost_equal(rot_sky_rad, np.radians(rot_sky_deg), 10)

        for ra, dec, rotTel in zip(self.ra_list, self.dec_list, rot_tel_list):
            rot_sky_rad = utils._get_rot_sky_pos(ra, dec, obs_temp, rotTel)

            rot_sky_deg = utils.get_rot_sky_pos(np.degrees(ra), np.degrees(dec), obs_temp, np.degrees(rotTel))

            self.assertAlmostEqual(rot_sky_rad, np.radians(rot_sky_deg), 10)

    def test_get_rot_tel_pos(self):
        rot_sky_list = self.rng.random_sample(len(self.ra_list)) * 2.0 * np.pi
        mjd = 56789.3
        obs_temp = ObservationMetaData(mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST"))

        rot_tel_rad = utils._get_rot_tel_pos(self.ra_list, self.dec_list, obs_temp, rot_sky_list)

        rot_tel_deg = utils.get_rot_tel_pos(
            np.degrees(self.ra_list),
            np.degrees(self.dec_list),
            obs_temp,
            np.degrees(rot_sky_list),
        )

        np.testing.assert_array_almost_equal(rot_tel_rad, np.radians(rot_tel_deg), 10)

        rot_tel_rad = utils._get_rot_tel_pos(self.ra_list, self.dec_list, obs_temp, rot_sky_list[0])

        rot_tel_deg = utils.get_rot_tel_pos(
            np.degrees(self.ra_list),
            np.degrees(self.dec_list),
            obs_temp,
            np.degrees(rot_sky_list[0]),
        )

        np.testing.assert_array_almost_equal(rot_tel_rad, np.radians(rot_tel_deg), 10)

        for ra, dec, rotSky in zip(self.ra_list, self.dec_list, rot_sky_list):
            obs_temp = ObservationMetaData(
                mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST")
            )

            rot_tel_rad = utils._get_rot_tel_pos(ra, dec, obs_temp, rotSky)

            rot_tel_deg = utils.get_rot_tel_pos(np.degrees(ra), np.degrees(dec), obs_temp, np.degrees(rotSky))

            self.assertAlmostEqual(rot_tel_rad, np.radians(rot_tel_deg), 10)


class AstrometryDegreesTest(unittest.TestCase):
    def setUp(self):
        self.n_stars = 10
        self.rng = np.random.RandomState(8273)
        self.ra_list = self.rng.random_sample(self.n_stars) * 2.0 * np.pi
        self.dec_list = (self.rng.random_sample(self.n_stars) - 0.5) * np.pi
        self.mjd_list = self.rng.random_sample(10) * 5000.0 + 52000.0
        self.pm_ra_list = utils.radians_from_arcsec(self.rng.random_sample(self.n_stars) * 10.0 - 5.0)
        self.pm_dec_list = utils.radians_from_arcsec(self.rng.random_sample(self.n_stars) * 10.0 - 5.0)
        self.px_list = utils.radians_from_arcsec(self.rng.random_sample(self.n_stars) * 2.0)
        self.v_rad_list = self.rng.random_sample(self.n_stars) * 500.0 - 250.0

    def test_apply_precession(self):
        for mjd in self.mjd_list:
            ra_rad, dec_rad = utils._apply_precession(
                self.ra_list, self.dec_list, mjd=ModifiedJulianDate(TAI=mjd)
            )

            ra_deg, dec_deg = utils.apply_precession(
                np.degrees(self.ra_list),
                np.degrees(self.dec_list),
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

            d_dec = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

    def test_apply_proper_motion(self):
        for mjd in self.mjd_list:
            ra_rad, dec_rad = utils._apply_proper_motion(
                self.ra_list,
                self.dec_list,
                self.pm_ra_list,
                self.pm_dec_list,
                self.px_list,
                self.v_rad_list,
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            ra_deg, dec_deg = utils.apply_proper_motion(
                np.degrees(self.ra_list),
                np.degrees(self.dec_list),
                utils.arcsec_from_radians(self.pm_ra_list),
                utils.arcsec_from_radians(self.pm_dec_list),
                utils.arcsec_from_radians(self.px_list),
                self.v_rad_list,
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

            d_dec = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

        for ra, dec, pm_ra, pm_dec, px, v_rad in zip(
            self.ra_list,
            self.dec_list,
            self.pm_ra_list,
            self.pm_dec_list,
            self.px_list,
            self.v_rad_list,
        ):
            ra_rad, dec_rad = utils._apply_proper_motion(
                ra,
                dec,
                pm_ra,
                pm_dec,
                px,
                v_rad,
                mjd=ModifiedJulianDate(TAI=self.mjd_list[0]),
            )

            ra_deg, dec_deg = utils.apply_proper_motion(
                np.degrees(ra),
                np.degrees(dec),
                utils.arcsec_from_radians(pm_ra),
                utils.arcsec_from_radians(pm_dec),
                utils.arcsec_from_radians(px),
                v_rad,
                mjd=ModifiedJulianDate(TAI=self.mjd_list[0]),
            )

            self.assertAlmostEqual(utils.arcsec_from_radians(ra_rad - np.radians(ra_deg)), 0.0, 9)
            self.assertAlmostEqual(utils.arcsec_from_radians(dec_rad - np.radians(dec_deg)), 0.0, 9)

    def test_app_geo_from_icrs(self):
        mjd = 42350.0
        for pm_ra_list in [self.pm_ra_list, None]:
            for pm_dec_list in [self.pm_dec_list, None]:
                for px_list in [self.px_list, None]:
                    for v_rad_list in [self.v_rad_list, None]:
                        ra_rad, dec_rad = utils._app_geo_from_icrs(
                            self.ra_list,
                            self.dec_list,
                            pm_ra_list,
                            pm_dec_list,
                            px_list,
                            v_rad_list,
                            mjd=ModifiedJulianDate(TAI=mjd),
                        )

                        ra_deg, dec_deg = utils.app_geo_from_icrs(
                            np.degrees(self.ra_list),
                            np.degrees(self.dec_list),
                            utils.arcsec_from_radians(pm_ra_list),
                            utils.arcsec_from_radians(pm_dec_list),
                            utils.arcsec_from_radians(px_list),
                            v_rad_list,
                            mjd=ModifiedJulianDate(TAI=mjd),
                        )

                        d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
                        np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

                        d_dec = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
                        np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

    def test_observed_from_app_geo(self):
        obs = ObservationMetaData(pointing_ra=35.0, pointing_dec=-45.0, mjd=43572.0)

        for include_refraction in [True, False]:
            ra_rad, dec_rad = utils._observed_from_app_geo(
                self.ra_list,
                self.dec_list,
                include_refraction=include_refraction,
                alt_az_hr=False,
                obs_metadata=obs,
            )

            ra_deg, dec_deg = utils.observed_from_app_geo(
                np.degrees(self.ra_list),
                np.degrees(self.dec_list),
                include_refraction=include_refraction,
                alt_az_hr=False,
                obs_metadata=obs,
            )

            d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

            d_dec = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

            ra_dec, alt_az = utils._observed_from_app_geo(
                self.ra_list,
                self.dec_list,
                include_refraction=include_refraction,
                alt_az_hr=True,
                obs_metadata=obs,
            )

            ra_rad = ra_dec[0]
            alt_rad = alt_az[0]
            az_rad = alt_az[1]

            ra_dec, alt_az = utils.observed_from_app_geo(
                np.degrees(self.ra_list),
                np.degrees(self.dec_list),
                include_refraction=include_refraction,
                alt_az_hr=True,
                obs_metadata=obs,
            )

            ra_deg = ra_dec[0]
            alt_deg = alt_az[0]
            az_deg = alt_az[1]

            d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

            d_dec = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

            d_az = utils.arcsec_from_radians(az_rad - np.radians(az_deg))
            np.testing.assert_array_almost_equal(d_az, np.zeros(self.n_stars), 9)

            d_alt = utils.arcsec_from_radians(alt_rad - np.radians(alt_deg))
            np.testing.assert_array_almost_equal(d_alt, np.zeros(self.n_stars), 9)

    def test_app_geo_from_observed(self):
        obs = ObservationMetaData(pointing_ra=35.0, pointing_dec=-45.0, mjd=43572.0)

        for include_refraction in (True, False):
            for wavelength in (0.5, 0.2, 0.3):
                ra_rad, dec_rad = utils._app_geo_from_observed(
                    self.ra_list,
                    self.dec_list,
                    include_refraction=include_refraction,
                    wavelength=wavelength,
                    obs_metadata=obs,
                )

                ra_deg, dec_deg = utils.app_geo_from_observed(
                    np.degrees(self.ra_list),
                    np.degrees(self.dec_list),
                    include_refraction=include_refraction,
                    wavelength=wavelength,
                    obs_metadata=obs,
                )

                d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
                np.testing.assert_array_almost_equal(d_ra, np.zeros(len(d_ra)), 9)

                d_dec = utils.arcsec_from_radians(dec_rad - np.radians(dec_deg))
                np.testing.assert_array_almost_equal(d_dec, np.zeros(len(d_dec)), 9)

    def test_icrs_from_app_geo(self):
        for mjd in (53525.0, 54316.3, 58463.7):
            for epoch in (2000.0, 1950.0, 2010.0):
                ra_rad, dec_rad = utils._icrs_from_app_geo(
                    self.ra_list,
                    self.dec_list,
                    epoch=epoch,
                    mjd=ModifiedJulianDate(TAI=mjd),
                )

                ra_deg, dec_deg = utils.icrs_from_app_geo(
                    np.degrees(self.ra_list),
                    np.degrees(self.dec_list),
                    epoch=epoch,
                    mjd=ModifiedJulianDate(TAI=mjd),
                )

                d_ra = utils.arcsec_from_radians(np.abs(ra_rad - np.radians(ra_deg)))
                self.assertLess(d_ra.max(), 1.0e-9)

                d_dec = utils.arcsec_from_radians(np.abs(dec_rad - np.radians(dec_deg)))
                self.assertLess(d_dec.max(), 1.0e-9)

    def test_observed_from_icrs(self):
        obs = ObservationMetaData(pointing_ra=35.0, pointing_dec=-45.0, mjd=43572.0)
        for pm_ra_list in [self.pm_ra_list, None]:
            for pm_dec_list in [self.pm_dec_list, None]:
                for px_list in [self.px_list, None]:
                    for v_rad_list in [self.v_rad_list, None]:
                        for include_refraction in [True, False]:
                            ra_rad, dec_rad = utils._observed_from_icrs(
                                self.ra_list,
                                self.dec_list,
                                pm_ra=pm_ra_list,
                                pm_dec=pm_dec_list,
                                parallax=px_list,
                                v_rad=v_rad_list,
                                obs_metadata=obs,
                                epoch=2000.0,
                                include_refraction=include_refraction,
                            )

                            ra_deg, dec_deg = utils.observed_from_icrs(
                                np.degrees(self.ra_list),
                                np.degrees(self.dec_list),
                                pm_ra=utils.arcsec_from_radians(pm_ra_list),
                                pm_dec=utils.arcsec_from_radians(pm_dec_list),
                                parallax=utils.arcsec_from_radians(px_list),
                                v_rad=v_rad_list,
                                obs_metadata=obs,
                                epoch=2000.0,
                                include_refraction=include_refraction,
                            )

                            d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
                            np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

                            d_dec = utils.arcsec_from_radians(dec_rad - np.radians(dec_deg))
                            np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

    def test_icrs_from_observed(self):
        obs = ObservationMetaData(pointing_ra=35.0, pointing_dec=-45.0, mjd=43572.0)

        for include_refraction in [True, False]:
            ra_rad, dec_rad = utils._icrs_from_observed(
                self.ra_list,
                self.dec_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            ra_deg, dec_deg = utils.icrs_from_observed(
                np.degrees(self.ra_list),
                np.degrees(self.dec_list),
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
            np.testing.assert_array_almost_equal(d_ra, np.zeros(self.n_stars), 9)

            d_dec = utils.arcsec_from_radians(dec_rad - np.radians(dec_deg))
            np.testing.assert_array_almost_equal(d_dec, np.zeros(self.n_stars), 9)

    def testra_dec_from_pupil_coords(self):
        obs = ObservationMetaData(pointing_ra=23.5, pointing_dec=-115.0, mjd=42351.0, rot_sky_pos=127.0)

        xp_list = self.rng.random_sample(100) * 0.25 * np.pi
        yp_list = self.rng.random_sample(100) * 0.25 * np.pi

        ra_rad, dec_rad = utils._ra_dec_from_pupil_coords(xp_list, yp_list, obs_metadata=obs, epoch=2000.0)
        ra_deg, dec_deg = utils.ra_dec_from_pupil_coords(xp_list, yp_list, obs_metadata=obs, epoch=2000.0)

        d_ra = utils.arcsec_from_radians(ra_rad - np.radians(ra_deg))
        np.testing.assert_array_almost_equal(d_ra, np.zeros(len(xp_list)), 9)

        d_dec = utils.arcsec_from_radians(dec_rad - np.radians(dec_deg))
        np.testing.assert_array_almost_equal(d_dec, np.zeros(len(xp_list)), 9)

    def testpupil_coords_from_ra_dec(self):
        obs = ObservationMetaData(pointing_ra=23.5, pointing_dec=-115.0, mjd=42351.0, rot_sky_pos=127.0)

        # need to make sure the test points are tightly distributed around the bore site, or
        # PALPY will throw an error
        ra_list = self.rng.random_sample(self.n_stars) * np.radians(1.0) + np.radians(23.5)
        dec_list = self.rng.random_sample(self.n_stars) * np.radians(1.0) + np.radians(-115.0)

        xp_control, yp_control = utils._pupil_coords_from_ra_dec(
            ra_list, dec_list, obs_metadata=obs, epoch=2000.0
        )

        xp_test, yp_test = utils.pupil_coords_from_ra_dec(
            np.degrees(ra_list), np.degrees(dec_list), obs_metadata=obs, epoch=2000.0
        )

        dx = utils.arcsec_from_radians(xp_control - xp_test)
        np.testing.assert_array_almost_equal(dx, np.zeros(self.n_stars), 9)

        dy = utils.arcsec_from_radians(yp_control - yp_test)
        np.testing.assert_array_almost_equal(dy, np.zeros(self.n_stars), 9)


if __name__ == "__main__":
    unittest.main()
