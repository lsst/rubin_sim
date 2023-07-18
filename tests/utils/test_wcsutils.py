import unittest

import numpy as np

from rubin_sim.utils import (
    ObservationMetaData,
    Site,
    _native_lon_lat_from_ra_dec,
    _ra_dec_from_native_lon_lat,
    arcsec_from_radians,
    haversine,
    icrs_from_observed,
    native_lon_lat_from_ra_dec,
    observed_from_icrs,
    ra_dec_from_alt_az,
    ra_dec_from_native_lon_lat,
)


class NativeLonLatTest(unittest.TestCase):
    def test_native_lon_lat(self):
        """
        Test that native_lon_lat_from_ra_dec works by considering stars and pointings
        at intuitive locations
        """

        mjd = 53855.0

        ra_list_obs = [0.0, 0.0, 0.0, 270.0]
        dec_list_obs = [90.0, 90.0, 0.0, 0.0]

        ra_point_list_obs = [0.0, 270.0, 270.0, 0.0]
        dec_point_list_obs = [0.0, 0.0, 0.0, 0.0]

        lon_control_list = [180.0, 180.0, 90.0, 270.0]
        lat_control_list = [0.0, 0.0, 0.0, 0.0]

        for rr_obs, dd_obs, rp_obs, dp_obs, lonc, latc in zip(
            ra_list_obs,
            dec_list_obs,
            ra_point_list_obs,
            dec_point_list_obs,
            lon_control_list,
            lat_control_list,
        ):
            obs_temp = ObservationMetaData(mjd=mjd)

            rr, dd = icrs_from_observed(
                np.array([rr_obs, rp_obs]),
                np.array([dd_obs, dp_obs]),
                obs_metadata=obs_temp,
                epoch=2000.0,
                include_refraction=True,
            )

            obs = ObservationMetaData(pointing_ra=rr[1], pointing_dec=dd[1], mjd=mjd)
            lon, lat = native_lon_lat_from_ra_dec(rr[0], dd[0], obs)
            distance = arcsec_from_radians(haversine(lon, lat, lonc, latc))
            self.assertLess(distance, 1.0)

    def test_native_long_lat_complicated(self):
        """
        Test that nativeLongLatFromRaDec works by considering stars and pointings
        at non-intuitive locations.
        """

        rng = np.random.RandomState(42)
        n_pointings = 10
        ra_pointing_list_icrs = rng.random_sample(n_pointings) * 360.0
        dec_pointing_list_icrs = rng.random_sample(n_pointings) * 180.0 - 90.0
        mjd_list = rng.random_sample(n_pointings) * 10000.0 + 43000.0

        n_stars = 10
        for ra_pointing_icrs, decPointing_icrs, mjd in zip(
            ra_pointing_list_icrs, dec_pointing_list_icrs, mjd_list
        ):
            obs = ObservationMetaData(pointing_ra=ra_pointing_icrs, pointing_dec=decPointing_icrs, mjd=mjd)
            ra_list_icrs = rng.random_sample(n_stars) * 360.0
            dec_list_icrs = rng.random_sample(n_stars) * 180.0 - 90.0
            ra_list_obs, dec_list_obs = observed_from_icrs(
                ra_list_icrs,
                dec_list_icrs,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=True,
            )

            obs_temp = ObservationMetaData(mjd=mjd)
            ra_pointing_obs, dec_pointing_obs = observed_from_icrs(
                ra_pointing_icrs,
                decPointing_icrs,
                obs_metadata=obs_temp,
                epoch=2000.0,
                include_refraction=True,
            )

            for ra_obs, dec_obs, ra_icrs, dec_icrs in zip(
                ra_list_obs, dec_list_obs, ra_list_icrs, dec_list_icrs
            ):
                ra_rad = np.radians(ra_obs)
                dec_rad = np.radians(dec_obs)
                sin_ra = np.sin(ra_rad)
                cos_ra = np.cos(ra_rad)
                sin_dec = np.sin(dec_rad)
                cos_dec = np.cos(dec_rad)

                # the three dimensional position of the star
                control_position = np.array([-cos_dec * sin_ra, cos_dec * cos_ra, sin_dec])

                # calculate the rotation matrices needed to transform the
                # x, y, and z axes into the local x, y, and z axes
                # (i.e. the axes with z lined up with ra_pointing_obs, dec_pointing_obs)
                alpha = 0.5 * np.pi - np.radians(dec_pointing_obs)
                ca = np.cos(alpha)
                sa = np.sin(alpha)
                rot_x = np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])

                cb = np.cos(np.radians(ra_pointing_obs))
                sb = np.sin(np.radians(ra_pointing_obs))
                rot_z = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])

                # rotate the coordinate axes into the local basis
                x_axis = np.dot(rot_z, np.dot(rot_x, np.array([1.0, 0.0, 0.0])))
                y_axis = np.dot(rot_z, np.dot(rot_x, np.array([0.0, 1.0, 0.0])))
                z_axis = np.dot(rot_z, np.dot(rot_x, np.array([0.0, 0.0, 1.0])))

                # calculate the local longitude and latitude of the star
                lon, lat = native_lon_lat_from_ra_dec(ra_icrs, dec_icrs, obs)
                cos_lon = np.cos(np.radians(lon))
                sin_lon = np.sin(np.radians(lon))
                cos_lat = np.cos(np.radians(lat))
                sin_lat = np.sin(np.radians(lat))

                # the x, y, z position of the star in the local coordinate
                # basis
                transformed_position = np.array([-cos_lat * sin_lon, cos_lat * cos_lon, sin_lat])

                # convert that position back into the un-rotated bases
                test_position = (
                    transformed_position[0] * x_axis
                    + transformed_position[1] * y_axis
                    + transformed_position[2] * z_axis
                )

                # assert that test_position and control_position should be equal
                distance = np.sqrt(np.power(control_position - test_position, 2).sum())
                self.assertLess(distance, 1.0e-12)

    def test_native_lon_lat_vector(self):
        """
        Test that native_lon_lat_from_ra_dec works in a vectorized way; we do this
        by performing a bunch of tansformations passing in ra and dec as numpy arrays
        and then comparing them to results computed in an element-wise way
        """

        obs = ObservationMetaData(pointing_ra=123.0, pointing_dec=43.0, mjd=53467.2)

        n_samples = 100
        rng = np.random.RandomState(42)
        ra_list = rng.random_sample(n_samples) * 360.0
        dec_list = rng.random_sample(n_samples) * 180.0 - 90.0

        lon_list, lat_list = native_lon_lat_from_ra_dec(ra_list, dec_list, obs)

        for rr, dd, lon, lat in zip(ra_list, dec_list, lon_list, lat_list):
            lon_control, lat_control = native_lon_lat_from_ra_dec(rr, dd, obs)
            distance = arcsec_from_radians(
                haversine(
                    np.radians(lon),
                    np.radians(lat),
                    np.radians(lon_control),
                    np.radians(lat_control),
                )
            )

            self.assertLess(distance, 0.0001)

    def test_ra_dec(self):
        """
        Test that ra_dec_from_native_lon_lat does invert
        native_lon_lat_from_ra_dec
        """
        rng = np.random.RandomState(42)
        n_samples = 100
        # because ra_dec_from_native_lon_lat is only good
        rr_list = rng.random_sample(n_samples) * 50.0
        # out to a zenith distance of ~ 70 degrees

        theta_list = rng.random_sample(n_samples) * 2.0 * np.pi

        rr_pointing_list = rng.random_sample(10) * 50.0
        theta_pointing_list = rng.random_sample(10) * 2.0 * np.pi
        mjd_list = rng.random_sample(n_samples) * 10000.0 + 43000.0

        for rrp, thetap, mjd in zip(rr_pointing_list, theta_pointing_list, mjd_list):
            site = Site(name="LSST")
            ra_zenith, dec_zenith = ra_dec_from_alt_az(180.0, 0.0, ObservationMetaData(mjd=mjd, site=site))

            rp = ra_zenith + rrp * np.cos(thetap)
            dp = dec_zenith + rrp * np.sin(thetap)
            obs = ObservationMetaData(pointing_ra=rp, pointing_dec=dp, mjd=mjd, site=site)

            ra_list_icrs = (ra_zenith + rr_list * np.cos(theta_list)) % 360.0
            dec_list_icrs = dec_zenith + rr_list * np.sin(theta_list)

            ra_list_obs, dec_list_obs = observed_from_icrs(
                ra_list_icrs,
                dec_list_icrs,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=True,
            )

            # calculate the distance between the ICRS position and the observed
            # geocentric position
            dd_icrs_obs_list = arcsec_from_radians(
                haversine(
                    np.radians(ra_list_icrs),
                    np.radians(dec_list_icrs),
                    np.radians(ra_list_obs),
                    np.radians(dec_list_obs),
                )
            )

            for rr, dd, dd_icrs_obs in zip(ra_list_icrs, dec_list_icrs, dd_icrs_obs_list):
                lon, lat = native_lon_lat_from_ra_dec(rr, dd, obs)
                r1, d1 = ra_dec_from_native_lon_lat(lon, lat, obs)

                # the distance between the input RA, Dec and the round-trip output
                # RA, Dec
                distance = arcsec_from_radians(
                    haversine(np.radians(r1), np.radians(d1), np.radians(rr), np.radians(dd))
                )

                rr_obs, dec_obs = observed_from_icrs(
                    rr, dd, obs_metadata=obs, epoch=2000.0, include_refraction=True
                )

                # verify that the round trip through nativeLonLat only changed
                # RA, Dec by less than an arcsecond
                self.assertLess(distance, 1.0)

                # verify that any difference in the round trip is much less
                # than the distance between the ICRS and the observed geocentric
                # RA, Dec
                self.assertLess(distance, dd_icrs_obs * 0.01)

    def test_ra_dec_vector(self):
        """
        Test that ra_dec_from_native_lon_lat does invert
        native_lon_lat_from_ra_dec (make sure it works in a vectorized way)
        """
        rng = np.random.RandomState(42)
        n_samples = 100
        lat_list = rng.random_sample(n_samples) * 360.0
        lon_list = rng.random_sample(n_samples) * 180.0 - 90.0
        ra_point = 95.0
        dec_point = 75.0

        obs = ObservationMetaData(pointing_ra=ra_point, pointing_dec=dec_point, mjd=53467.89)

        ra_list, dec_list = ra_dec_from_native_lon_lat(lon_list, lat_list, obs)

        for lon, lat, ra0, dec0 in zip(lon_list, lat_list, ra_list, dec_list):
            ra1, dec1 = ra_dec_from_native_lon_lat(lon, lat, obs)
            distance = arcsec_from_radians(
                haversine(np.radians(ra0), np.radians(dec0), np.radians(ra1), np.radians(dec1))
            )
            self.assertLess(distance, 0.1)

    def test_degrees_versus_radians(self):
        """
        Test that the radian and degree versions of native_lon_lat_from_ra_dec
        and ra_dec_from_native_lon_lat are consistent with each other
        """

        rng = np.random.RandomState(873)
        n_samples = 1000
        obs = ObservationMetaData(pointing_ra=45.0, pointing_dec=-34.5, mjd=54656.76)
        ra_list = rng.random_sample(n_samples) * 360.0
        dec_list = rng.random_sample(n_samples) * 180.0 - 90.0

        lon_deg, lat_deg = native_lon_lat_from_ra_dec(ra_list, dec_list, obs)
        lon_rad, lat_rad = _native_lon_lat_from_ra_dec(np.radians(ra_list), np.radians(dec_list), obs)
        np.testing.assert_array_almost_equal(np.radians(lon_deg), lon_rad, 15)
        np.testing.assert_array_almost_equal(np.radians(lat_deg), lat_rad, 15)

        ra_deg, dec_deg = ra_dec_from_native_lon_lat(ra_list, dec_list, obs)
        ra_rad, dec_rad = _ra_dec_from_native_lon_lat(np.radians(ra_list), np.radians(dec_list), obs)
        np.testing.assert_array_almost_equal(np.radians(ra_deg), ra_rad, 15)
        np.testing.assert_array_almost_equal(np.radians(dec_deg), dec_rad, 15)


if __name__ == "__main__":
    unittest.main()
