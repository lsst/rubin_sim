"""
Some of the data in this unit test will appear abitrary.  That is
because, in addition to testing the execution of all of the functionality
provided in the sims_coordUtils package, this unit test validates
the outputs of PALPY against the outputs of pySLALIB v 1.0.2
(it was written when we were making the transition from pySLALIB to PALPY).

There will be some difference, as the two libraries are based on slightly
different conventions (for example, the prenut routine which calculates
the matrix of precession and nutation is based on the IAU 2006/2000A
standard in PALPY and on SF2001 in pySLALIB; however, the two outputs
still agree to within one part in 10^5)

"""
import unittest

import numpy as np
import palpy as pal

# Tell astropy not to download the IERS tables
from astropy.utils import iers

from rubin_sim.utils import (
    ModifiedJulianDate,
    ObservationMetaData,
    Site,
    _angular_separation,
    _app_geo_from_icrs,
    _app_geo_from_observed,
    _apply_precession,
    _apply_proper_motion,
    _distance_to_sun,
    _get_rot_sky_pos,
    _get_rot_tel_pos,
    _icrs_from_app_geo,
    _icrs_from_observed,
    _observed_from_app_geo,
    _observed_from_icrs,
    _ra_dec_from_alt_az,
    _solar_ra_dec,
    apply_proper_motion,
    apply_refraction,
    arcsec_from_radians,
    distance_to_sun,
    haversine,
    observed_from_icrs,
    ra_dec_from_alt_az,
    radians_from_arcsec,
    refraction_coefficients,
    solar_ra_dec,
    spherical_from_cartesian,
)

iers.conf.auto_download = False


def make_observation_meta_data():
    # create a cartoon ObservationMetaData object
    mjd = 52000.0
    alt = np.pi / 2.0
    az = 0.0
    test_site = Site(
        latitude=np.degrees(0.5),
        longitude=np.degrees(1.1),
        height=3000,
        temperature=260.0,
        pressure=725.0,
        lapse_rate=0.005,
        humidity=0.4,
    )
    obs_temp = ObservationMetaData(site=test_site, mjd=mjd)
    center_ra, center_dec = _ra_dec_from_alt_az(alt, az, obs_temp)
    rot_tel = _get_rot_tel_pos(center_ra, center_dec, obs_temp, 0.0)
    rot_sky = _get_rot_sky_pos(center_ra, center_dec, obs_temp, rot_tel)

    radius = 0.1

    obs_metadata = ObservationMetaData(
        pointing_ra=np.degrees(center_ra),
        pointing_dec=np.degrees(center_dec),
        rot_sky_pos=np.degrees(rot_sky),
        mjd=mjd,
        bound_type="circle",
        bound_length=2.0 * radius,
        site=test_site,
    )

    return obs_metadata


def make_random_sample(ra_center=None, dec_center=None, radius=None):
    # create a random sample of object data

    nsamples = 100
    rng = np.random.RandomState(32)

    if ra_center is None or dec_center is None or radius is None:
        ra = rng.random_sample(nsamples) * 2.0 * np.pi
        dec = (rng.random_sample(nsamples) - 0.5) * np.pi
    else:
        rr = rng.random_sample(nsamples) * radius
        theta = rng.random_sample(nsamples) * 2.0 * np.pi
        ra = ra_center + rr * np.cos(theta)
        dec = dec_center + rr * np.cos(theta)

    pm_ra = (rng.random_sample(nsamples) - 0.5) * 0.1
    pm_dec = (rng.random_sample(nsamples) - 0.5) * 0.1
    parallax = rng.random_sample(nsamples) * 0.01
    v_rad = rng.random_sample(nsamples) * 1000.0

    return ra, dec, pm_ra, pm_dec, parallax, v_rad


class AstrometryUnitTest(unittest.TestCase):
    """
    The bulk of this unit test involves inputting a set list of input values
    and comparing the astrometric results to results derived from SLALIB run
    with the same input values.  We have to create a test catalog artificially (rather than
    querying the database) because SLALIB was originally run on values that did not correspond
    to any particular Opsim run.
    """

    def setUp(self):
        self.metadata = {}

        # these were the LSST site parameters as coded when this unit test was
        # written
        self.test_site = Site(
            longitude=np.degrees(-1.2320792),
            latitude=np.degrees(-0.517781017),
            height=2650.0,
            temperature=11.505,
            pressure=749.3,
            lapse_rate=0.0065,
            humidity=0.4,
        )

        # Inputs are consistent with the baseline SLALIB run
        # used to create this unit test
        self.obs_metadata = ObservationMetaData(
            pointing_ra=200.0,
            pointing_dec=-30.0,
            rot_sky_pos=1.0,
            mjd=50984.371741,
            bound_type="circle",
            bound_length=0.05,
            site=self.test_site,
        )

        self.tol = 1.0e-5

    def tearDown(self):
        del self.obs_metadata
        del self.metadata
        del self.tol

    def test_distance_to_sun(self):
        """
        Test _distance_to_sun using solar RA, Dec calculated from

        http://aa.usno.navy.mil/data/docs/JulianDate.php
        http://aa.usno.navy.mil/data/docs/geocentric.php
        """

        hour = np.radians(360.0 / 24.0)
        minute = hour / 60.0
        second = minute / 60.0

        mjd_list = [57026.0, 57543.625]

        sun_ra_list = [
            18.0 * hour + 56.0 * minute + 51.022 * second,
            4.0 * hour + 51.0 * minute + 22.776 * second,
        ]

        sun_dec_list = [
            np.radians(-22.0 - 47.0 / 60.0 - 40.27 / 3600.0),
            np.radians(22.0 + 30.0 / 60.0 + 0.73 / 3600.0),
        ]

        for ra_s, decS, tai in zip(sun_ra_list, sun_dec_list, mjd_list):
            mjd = ModifiedJulianDate(TAI=tai)

            # first, verify that the Sun is where we think it is to within 5
            # arc seconds
            self.assertLess(arcsec_from_radians(_distance_to_sun(ra_s, decS, mjd)), 5.0)

            # find Sun's Cartesian coordinates
            sun_x = np.cos(decS) * np.cos(ra_s)
            sun_y = np.cos(decS) * np.sin(ra_s)
            sun_z = np.sin(decS)

            # now choose positions that are a set distance away from the Sun, and make sure
            # that _distance_to_sun returns the expected result
            for theta in (np.pi / 2.0, np.pi / 4.0, -np.pi / 3.0):
                # displace by rotating about z axis
                new_x = sun_x * np.cos(theta) + sun_y * np.sin(theta)
                new_y = -sun_x * np.sin(theta) + sun_y * np.cos(theta)
                new_z = sun_z

                new_ra = np.arctan2(new_y, new_x)
                new_dec = np.arctan2(new_z, np.sqrt(new_x * new_x + new_y * new_y))

                dd = _distance_to_sun(new_ra, new_dec, mjd)
                hh = haversine(ra_s, decS, new_ra, new_dec)
                self.assertLess(np.abs(arcsec_from_radians(dd - hh)), 5.0)

                # displace by rotating about y axis
                new_x = sun_x * np.cos(theta) + sun_z * np.sin(theta)
                new_y = sun_y
                new_z = -sun_x * np.sin(theta) + sun_z * np.cos(theta)

                new_ra = np.arctan2(new_y, new_x)
                new_dec = np.arctan2(new_z, np.sqrt(new_x * new_x + new_y * new_y))
                dd = _distance_to_sun(new_ra, new_dec, mjd)
                hh = haversine(ra_s, decS, new_ra, new_dec)
                self.assertLess(np.abs(arcsec_from_radians(dd - hh)), 5.0)

                # displace by rotating about x axis
                new_x = sun_x
                new_y = sun_y * np.cos(theta) + sun_z * np.sin(theta)
                new_z = -sun_y * np.sin(theta) + sun_z * np.cos(theta)

                new_ra = np.arctan2(new_y, new_x)
                new_dec = np.arctan2(new_z, np.sqrt(new_x * new_x + new_y * new_y))
                dd = _distance_to_sun(new_ra, new_dec, mjd)
                hh = haversine(ra_s, decS, new_ra, new_dec)
                self.assertLess(np.abs(arcsec_from_radians(dd - hh)), 5.0)

        # Test passing in numpy arrays of RA, Dec
        rng = np.random.RandomState(87)
        n_samples = 100
        ra = rng.random_sample(n_samples) * 2.0 * np.pi
        dec = (rng.random_sample(n_samples) - 0.5) * np.pi
        mjd = ModifiedJulianDate(TAI=59580.0)
        control_distance = _distance_to_sun(ra, dec, mjd)
        self.assertIsInstance(control_distance, np.ndarray)
        for ix, (rr, dd) in enumerate(zip(ra, dec)):
            dd = _distance_to_sun(rr, dd, mjd)
            self.assertIsInstance(dd, float)
            self.assertAlmostEqual(dd, control_distance[ix], 12)

    def test_distance_to_sun_deg(self):
        """
        Test that distance_to_sun is consistent with _distance_to_sun
        """

        for tai, ra, dec in zip((57632.1, 45623.4, 55682.3), (112.0, 24.0, 231.2), (-25.0, 23.4, -60.0)):
            mjd = ModifiedJulianDate(TAI=tai)
            dd_deg = distance_to_sun(ra, dec, mjd)
            dd_rad = _distance_to_sun(np.radians(ra), np.radians(dec), mjd)
            self.assertAlmostEqual(np.radians(dd_deg), dd_rad, 10)

    def test_solar_ra_dec_deg(self):
        """
        Test that solar_ra_dec is consistent with _solar_ra_dec
        """

        for tai in (57664.2, 53478.9, 45672.1):
            mjd = ModifiedJulianDate(TAI=tai)
            ra_deg, dec_deg = solar_ra_dec(mjd)
            ra_rad, dec_rad = _solar_ra_dec(mjd)
            self.assertAlmostEqual(np.radians(ra_deg), ra_rad, 10)
            self.assertAlmostEqual(np.radians(dec_deg), dec_rad, 10)

    def test_distance_to_sun_array(self):
        """
        Test _distance_to_sun on numpy arrays of RA, Dec using solar RA, Dec calculated from

        http://aa.usno.navy.mil/data/docs/JulianDate.php
        http://aa.usno.navy.mil/data/docs/geocentric.php
        """

        rng = np.random.RandomState(77)
        n_stars = 100

        hour = np.radians(360.0 / 24.0)
        minute = hour / 60.0
        second = minute / 60.0

        mjd_list = [57026.0, 57543.625]

        sun_ra_list = [
            18.0 * hour + 56.0 * minute + 51.022 * second,
            4.0 * hour + 51.0 * minute + 22.776 * second,
        ]

        sun_dec_list = [
            np.radians(-22.0 - 47.0 / 60.0 - 40.27 / 3600.0),
            np.radians(22.0 + 30.0 / 60.0 + 0.73 / 3600.0),
        ]

        for tai, raS, decS in zip(mjd_list, sun_ra_list, sun_dec_list):
            mjd = ModifiedJulianDate(TAI=tai)
            ra_list = rng.random_sample(n_stars) * 2.0 * np.pi
            dec_list = (rng.random_sample(n_stars) - 0.5) * np.pi
            distance_list = _distance_to_sun(ra_list, dec_list, mjd)
            distance_control = haversine(ra_list, dec_list, raS, decS)
            np.testing.assert_array_almost_equal(distance_list, distance_control, 5)

    def test_astrometry_exceptions(self):
        """
        Test to make sure that stand-alone astrometry methods raise an exception when they are called without
        the necessary arguments
        """
        obs_metadata = make_observation_meta_data()
        ra, dec, pm_ra, pm_dec, parallax, v_rad = make_random_sample()

        ra_short = np.array([1.0])
        dec_short = np.array([1.0])

        # test refraction_coefficients
        self.assertRaises(RuntimeError, refraction_coefficients)
        site = obs_metadata.site
        x, y = refraction_coefficients(site=site)

        # test apply_refraction
        zd = 0.1
        apply_refraction(zd, x, y)

        zd = [0.1, 0.2]
        self.assertRaises(RuntimeError, apply_refraction, zd, x, y)

        zd = np.array([0.1, 0.2])
        apply_refraction(zd, x, y)

        # test _apply_precession
        # test without mjd
        self.assertRaises(RuntimeError, _apply_precession, ra, dec)

        # test mismatches
        self.assertRaises(
            RuntimeError,
            _apply_precession,
            ra_short,
            dec,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_precession,
            ra,
            dec_short,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )

        # test that it runs
        _apply_precession(ra, dec, mjd=ModifiedJulianDate(TAI=52000.0))

        # test _apply_proper_motion
        ra_list = list(ra)
        dec_list = list(dec)
        pm_ra_list = list(pm_ra)
        pm_dec_list = list(pm_dec)
        parallax_list = list(parallax)
        v_rad_list = list(v_rad)

        pm_ra_short = np.array([pm_ra[0]])
        pm_dec_short = np.array([pm_dec[0]])
        parallax_short = np.array([parallax[0]])
        v_rad_short = np.array([v_rad[0]])

        # test without mjd
        self.assertRaises(RuntimeError, _apply_proper_motion, ra, dec, pm_ra, pm_dec, parallax, v_rad)

        # test passing lists
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra_list,
            dec,
            pm_ra,
            pm_dec,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec_list,
            pm_ra,
            pm_dec,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra_list,
            pm_dec,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra,
            pm_dec_list,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra,
            pm_dec,
            parallax_list,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra,
            pm_dec,
            parallax,
            v_rad_list,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )

        # test mismatches
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra_short,
            dec,
            pm_ra,
            pm_dec,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec_short,
            pm_ra,
            pm_dec,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra_short,
            pm_dec,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra,
            pm_dec_short,
            parallax,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra,
            pm_dec,
            parallax_short,
            v_rad,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _apply_proper_motion,
            ra,
            dec,
            pm_ra,
            pm_dec,
            parallax,
            v_rad_short,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )

        # test that it actually runs
        _apply_proper_motion(ra, dec, pm_ra, pm_dec, parallax, v_rad, mjd=ModifiedJulianDate(TAI=52000.0))
        _apply_proper_motion(
            ra[0],
            dec[0],
            pm_ra[0],
            pm_dec[0],
            parallax[0],
            v_rad[0],
            mjd=ModifiedJulianDate(TAI=52000.0),
        )

        # test _app_geo_from_icrs
        # test without mjd
        self.assertRaises(RuntimeError, _app_geo_from_icrs, ra, dec)

        # test with mismatched ra, dec
        self.assertRaises(
            RuntimeError,
            _app_geo_from_icrs,
            ra,
            dec_short,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )
        self.assertRaises(
            RuntimeError,
            _app_geo_from_icrs,
            ra_short,
            dec,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )

        # test that it actually urns
        _app_geo_from_icrs(ra, dec, mjd=obs_metadata.mjd)

        # test _observed_from_app_geo
        # test without obs_metadata
        self.assertRaises(RuntimeError, _observed_from_app_geo, ra, dec)

        # test without site
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            mjd=obs_metadata.mjd,
            site=None,
        )
        self.assertRaises(RuntimeError, _observed_from_app_geo, ra, dec, obs_metadata=dummy)

        # test without mjd
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            site=Site(name="LSST"),
        )
        self.assertRaises(RuntimeError, _observed_from_app_geo, ra, dec, obs_metadata=dummy)

        # test mismatches
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            mjd=obs_metadata.mjd,
            site=Site(name="LSST"),
        )

        self.assertRaises(RuntimeError, _observed_from_app_geo, ra, dec_short, obs_metadata=dummy)
        self.assertRaises(RuntimeError, _observed_from_app_geo, ra_short, dec, obs_metadata=dummy)

        # test that it actually runs
        _observed_from_app_geo(ra, dec, obs_metadata=dummy)

        # test _observed_from_icrs
        # test without epoch
        self.assertRaises(RuntimeError, _observed_from_icrs, ra, dec, obs_metadata=obs_metadata)

        # test without obs_metadata
        self.assertRaises(RuntimeError, _observed_from_icrs, ra, dec, epoch=2000.0)

        # test without mjd
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            site=obs_metadata.site,
        )
        self.assertRaises(RuntimeError, _observed_from_icrs, ra, dec, epoch=2000.0, obs_metadata=dummy)

        # test that it actually runs
        dummy = ObservationMetaData(
            pointing_ra=obs_metadata.pointing_ra,
            pointing_dec=obs_metadata.pointing_dec,
            site=obs_metadata.site,
            mjd=obs_metadata.mjd,
        )

        # test mismatches
        self.assertRaises(
            RuntimeError,
            _observed_from_icrs,
            ra,
            dec_short,
            epoch=2000.0,
            obs_metadata=dummy,
        )
        self.assertRaises(
            RuntimeError,
            _observed_from_icrs,
            ra_short,
            dec,
            epoch=2000.0,
            obs_metadata=dummy,
        )

        # test that it actually runs
        ra_arr, dec_arr = _observed_from_icrs(ra, dec, obs_metadata=dummy, epoch=2000.0)
        self.assertIsInstance(ra_arr, np.ndarray)
        self.assertIsInstance(dec_arr, np.ndarray)

        # test passing in floats
        for ix in range(len(ra_arr)):
            ra_f, dec_f = _observed_from_icrs(ra[ix], dec[ix], obs_metadata=dummy, epoch=2000.0)
            self.assertIsInstance(ra_f, float)
            self.assertIsInstance(dec_f, float)
            self.assertAlmostEqual(ra_f, ra_arr[ix], 12)
            self.assertAlmostEqual(dec_f, dec_arr[ix], 12)

    def test_apply_precession(self):
        ra = np.zeros((3), dtype=float)
        dec = np.zeros((3), dtype=float)

        ra[0] = 2.549091039839124218e00
        dec[0] = 5.198752733024248895e-01
        ra[1] = 8.693375673649429425e-01
        dec[1] = 1.038086165642298164e00
        ra[2] = 7.740864769302191473e-01
        dec[2] = 2.758053025017753179e-01

        self.assertRaises(RuntimeError, _apply_precession, ra, dec)

        # just make sure it runs
        mjd = ModifiedJulianDate(TAI=57388.0)
        ra_arr, dec_arr = _apply_precession(ra, dec, mjd=mjd)
        self.assertIsInstance(ra_arr, np.ndarray)
        self.assertIsInstance(dec_arr, np.ndarray)

        # test that passing in floats gie the same results
        for ix, (rr, dd) in enumerate(zip(ra, dec)):
            ra_f, dec_f = _apply_precession(rr, dd, mjd=mjd)
            self.assertIsInstance(ra_f, float)
            self.assertIsInstance(dec_f, float)
            self.assertAlmostEqual(ra_f, ra_arr[ix], 12)
            self.assertAlmostEqual(dec_f, dec_arr[ix], 12)

    def test_apply_proper_motion(self):
        """
        Compare the output of _apply_proper_motion to control outputs
        generated by recreating the 'space motion' section of code
        from palMapqk.c in palpy/cextern/pal
        """
        VF = 0.21094502
        pal_das2r = 4.8481368110953599358991410235794797595635330237270e-6

        rng = np.random.RandomState(18)
        n_samples = 1000

        mjd_list = rng.random_sample(20) * 20000.0 + 45000.0

        for mjd in mjd_list:
            ra_list_icrs = rng.random_sample(n_samples) * 2.0 * np.pi
            dec_list_icrs = (rng.random_sample(n_samples) - 0.5) * np.pi

            # stars' original position in Cartesian space
            x_list_icrs = np.cos(dec_list_icrs) * np.cos(ra_list_icrs)
            y_list_icrs = np.cos(dec_list_icrs) * np.sin(ra_list_icrs)
            z_list_icrs = np.sin(dec_list_icrs)

            pm_ra = (rng.random_sample(n_samples) - 0.5) * radians_from_arcsec(1.0)
            pm_dec = (rng.random_sample(n_samples) - 0.5) * radians_from_arcsec(1.0)
            px = rng.random_sample(n_samples) * radians_from_arcsec(1.0)
            v_rad = rng.random_sample(n_samples) * 200.0

            ra_list_pm, dec_list_pm = _apply_proper_motion(
                ra_list_icrs,
                dec_list_icrs,
                pm_ra * np.cos(dec_list_icrs),
                pm_dec,
                px,
                v_rad,
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            # stars' Cartesian position after proper motion is applied
            x_list_pm = np.cos(dec_list_pm) * np.cos(ra_list_pm)
            y_list_pm = np.cos(dec_list_pm) * np.sin(ra_list_pm)
            z_list_pm = np.sin(dec_list_pm)

            ###############################################################
            # The code below is copied from palMapqk.c in palpy/cextern/pal
            params = pal.mappa(2000.0, mjd)
            pmt = params[0]
            eb = np.array([params[1], params[2], params[3]])

            pxr = px * pal_das2r

            w = VF * v_rad * pxr

            motion_per_year = np.array(
                [
                    -1.0 * pm_ra * y_list_icrs
                    - pm_dec * np.cos(ra_list_icrs) * np.sin(dec_list_icrs)
                    + w * x_list_icrs,
                    pm_ra * x_list_icrs
                    - pm_dec * np.sin(ra_list_icrs) * np.sin(dec_list_icrs)
                    + w * y_list_icrs,
                    pm_dec * np.cos(dec_list_icrs) + w * z_list_icrs,
                ]
            )

            xyz_control = np.array(
                [
                    x_list_icrs + pmt * motion_per_year[0] - pxr * eb[0],
                    y_list_icrs + pmt * motion_per_year[1] - pxr * eb[1],
                    z_list_icrs + pmt * motion_per_year[2] - pxr * eb[2],
                ]
            )

            xyz_norm = np.sqrt(
                np.power(xyz_control[0], 2) + np.power(xyz_control[1], 2) + np.power(xyz_control[2], 2)
            )

            # stars' Cartesian position after applying the control proper
            # motion method
            xyz_control[0] = xyz_control[0] / xyz_norm
            xyz_control[1] = xyz_control[1] / xyz_norm
            xyz_control[2] = xyz_control[2] / xyz_norm

            # this is the Cartesian distance between the stars' positions as found by _apply_proper_motion
            # and the distance as found by the control proper motion code above
            distance = np.sqrt(
                np.power(x_list_pm - xyz_control[0], 2)
                + np.power(y_list_pm - xyz_control[1], 2)
                + np.power(z_list_pm - xyz_control[2], 2)
            )

            # this is the Cartesian distance between the stars' original positions on the celestial sphere
            # and their positions after the control proper motion was applied
            correction = np.sqrt(
                np.power(xyz_control[0] - x_list_icrs, 2)
                + np.power(xyz_control[1] - y_list_icrs, 2)
                + np.power(xyz_control[2] - z_list_icrs, 2)
            )

            dex = np.argmax(distance)
            msg = "pm %e %e vr %e px %e; time %e; err %e arcsec; corr %e" % (
                arcsec_from_radians(pm_ra[dex]),
                arcsec_from_radians(pm_dec[dex]),
                v_rad[dex],
                arcsec_from_radians(px[dex]),
                pmt,
                arcsec_from_radians(distance[dex]),
                arcsec_from_radians(correction[dex]),
            )

            # demand that the two methods agree on the stars' new positions to
            # within one part in 100
            test_value = (distance / correction).max()
            self.assertLess(test_value, 0.01, msg=msg)

    def test_apply_proper_motion_inputs(self):
        """
        Verify that _apply_proper_motion handles both floats and numpy arrays as inputs
        """
        rng = np.random.RandomState(7134)
        n_samples = 100
        pm_ra = (rng.random_sample(n_samples) - 0.5) * radians_from_arcsec(1.0)
        pm_dec = (rng.random_sample(n_samples) - 0.5) * radians_from_arcsec(1.0)
        px = rng.random_sample(n_samples) * radians_from_arcsec(1.0)
        v_rad = rng.random_sample(n_samples) * 200.0
        mjd = ModifiedJulianDate(TAI=59580.0)

        ra_icrs = rng.random_sample(n_samples) * 2.0 * np.pi
        dec_icrs = (rng.random_sample(n_samples) - 0.5) * np.pi

        ra_arr, dec_arr = _apply_proper_motion(ra_icrs, dec_icrs, pm_ra, pm_dec, px, v_rad, mjd=mjd)

        self.assertIsInstance(ra_arr, np.ndarray)
        self.assertIsInstance(dec_arr, np.ndarray)

        for ix, (rr, dd, mura, mudec, xx, vv) in enumerate(zip(ra_icrs, dec_icrs, pm_ra, pm_dec, px, v_rad)):
            ra_f, dec_f = _apply_proper_motion(rr, dd, mura, mudec, xx, vv, mjd=mjd)
            self.assertIsInstance(ra_f, float)
            self.assertIsInstance(dec_f, float)
            distance = arcsec_from_radians(haversine(ra_f, dec_f, ra_arr[ix], dec_arr[ix]))
            self.assertLess(distance, 0.000001)

    def test_app_geo_from_icrs(self):
        """
        Test conversion between ICRS RA, Dec and apparent geocentric ICRS.

        Apparent, geocentric RA, Dec of objects will be taken from this website

        http://aa.usno.navy.mil/data/docs/geocentric.php

        dates converted to JD using this website

        http://aa.usno.navy.mil/data/docs/geocentric.php

        """

        hours = np.radians(360.0 / 24.0)
        minutes = hours / 60.0
        seconds = minutes / 60.0

        # test on Arcturus
        # data taken from
        # http://aa.usno.navy.mil/data/docs/geocentric.php
        ra_icrs = 14.0 * hours + 15.0 * minutes + 39.67207 * seconds
        dec_icrs = np.radians(19.0 + 10.0 / 60.0 + 56.673 / 3600.0)
        pm_ra = radians_from_arcsec(-1.0939)
        pm_dec = radians_from_arcsec(-2.00006)
        v_rad = -5.19
        px = radians_from_arcsec(0.08883)

        mjd_list = []
        ra_app_list = []
        dec_app_list = []

        # jd (UT)
        jd = 2457000.375000
        mjd = jd - 2400000.5

        mjd_list.append(mjd)
        ra_app_list.append(14.0 * hours + 16.0 * minutes + 19.59 * seconds)
        dec_app_list.append(np.radians(19.0 + 6.0 / 60.0 + 19.56 / 3600.0))

        jd = 2457187.208333
        mjd = jd - 2400000.5
        mjd_list.append(mjd)
        ra_app_list.append(14.0 * hours + 16.0 * minutes + 22.807 * seconds)
        dec_app_list.append(np.radians(19.0 + 6.0 / 60.0 + 18.12 / 3600.0))

        jd = 2457472.625000
        mjd = jd - 2400000.5
        mjd_list.append(mjd)
        ra_app_list.append(14.0 * hours + 16.0 * minutes + 24.946 * seconds)
        dec_app_list.append(np.radians(19.0 + 5.0 / 60.0 + 49.65 / 3600.0))

        for mjd, ra_app, dec_app in zip(mjd_list, ra_app_list, dec_app_list):
            obs = ObservationMetaData(mjd=mjd)

            ra_test, dec_test = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                mjd=obs.mjd,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                v_rad=v_rad,
                parallax=px,
                epoch=2000.0,
            )

            self.assertIsInstance(ra_test, float)
            self.assertIsInstance(dec_test, float)

            distance = arcsec_from_radians(haversine(ra_app, dec_app, ra_test, dec_test))
            self.assertLess(distance, 0.1)

        # test on Sirius
        # data taken from
        # http://simbad.u-strasbg.fr/simbad/sim-id?Ident=Sirius
        ra_icrs = 6.0 * hours + 45.0 * minutes + 8.91728 * seconds
        dec_icrs = np.radians(-16.0 - 42.0 / 60.0 - 58.0171 / 3600.0)
        pm_ra = radians_from_arcsec(-0.54601)
        pm_dec = radians_from_arcsec(-1.22307)
        px = radians_from_arcsec(0.37921)
        v_rad = -5.5

        mjd_list = []
        ra_app_list = []
        dec_app_list = []

        jd = 2457247.000000
        mjd_list.append(jd - 2400000.5)
        ra_app_list.append(6.0 * hours + 45.0 * minutes + 49.276 * seconds)
        dec_app_list.append(np.radians(-16.0 - 44.0 / 60.0 - 18.69 / 3600.0))

        jd = 2456983.958333
        mjd_list.append(jd - 2400000.5)
        ra_app_list.append(6.0 * hours + 45.0 * minutes + 49.635 * seconds)
        dec_app_list.append(np.radians(-16.0 - 44.0 / 60.0 - 17.04 / 3600.0))

        jd = 2457523.958333
        mjd_list.append(jd - 2400000.5)
        ra_app_list.append(6.0 * hours + 45.0 * minutes + 50.99 * seconds)
        dec_app_list.append(np.radians(-16.0 - 44.0 / 60.0 - 39.76 / 3600.0))

        for mjd, ra_app, dec_app in zip(mjd_list, ra_app_list, dec_app_list):
            obs = ObservationMetaData(mjd=mjd)

            ra_test, dec_test = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                mjd=obs.mjd,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                v_rad=v_rad,
                parallax=px,
                epoch=2000.0,
            )

            self.assertIsInstance(ra_test, float)
            self.assertIsInstance(dec_test, float)

            distance = arcsec_from_radians(haversine(ra_app, dec_app, ra_test, dec_test))
            self.assertLess(distance, 0.1)

    def test_app_geo_from_icrs_inputs(self):
        """
        Test that app_geo_from_icrs behaves as expected when given both numpy
        array and float inputs.
        """

        rng = np.random.RandomState(83)
        n_samples = 100
        ra_icrs = 2.0 * np.pi * rng.random_sample(n_samples)
        dec_icrs = (rng.random_sample(n_samples) - 0.5) * np.pi
        pm_ra = radians_from_arcsec((rng.random_sample(n_samples) - 0.5) * 0.02)
        pm_dec = radians_from_arcsec((rng.random_sample(n_samples) - 0.5) * 0.02)
        parallax = radians_from_arcsec(rng.random_sample(n_samples) * 0.01)
        v_rad = (rng.random_sample(n_samples) - 0.5) * 1200.0
        mjd = ModifiedJulianDate(TAI=59580.0)

        ra_control, dec_control = _app_geo_from_icrs(
            ra_icrs,
            dec_icrs,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            parallax=parallax,
            v_rad=v_rad,
            mjd=mjd,
        )

        self.assertIsInstance(ra_control, np.ndarray)
        self.assertIsInstance(dec_control, np.ndarray)

        # test that passing in floats and numpy arrays gives the same result
        for ix in range(len(ra_control)):
            ra_test, dec_test = _app_geo_from_icrs(
                ra_icrs[ix],
                dec_icrs[ix],
                pm_ra=pm_ra[ix],
                pm_dec=pm_dec[ix],
                parallax=parallax[ix],
                v_rad=v_rad[ix],
                mjd=mjd,
            )

            self.assertIsInstance(ra_test, float)
            self.assertIsInstance(dec_test, float)
            dd = arcsec_from_radians(haversine(ra_test, dec_test, ra_control[ix], dec_control[ix]))
            self.assertLess(dd, 1.0e-6)

        # next test that inputs of inappropriate types raise RuntimeErrors
        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                5.0,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("dec", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                5.0,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("dec", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=5.0,
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("pm_ra", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=5.0,
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("pm_dec", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=5.0,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("parallax", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=5.0,
                mjd=mjd,
            )

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("v_rad", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs[:2],
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertEqual(
            context.exception.args[0],
            "The arrays input to app_geo_from_icrs all need to " "have the same length",
        )

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra[:2],
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertEqual(
            context.exception.args[0],
            "The arrays input to app_geo_from_icrs all need to " "have the same length",
        )

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=pm_dec[:2],
                parallax=parallax,
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertEqual(
            context.exception.args[0],
            "The arrays input to app_geo_from_icrs all need to " "have the same length",
        )

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax[:2],
                v_rad=v_rad,
                mjd=mjd,
            )

        self.assertEqual(
            context.exception.args[0],
            "The arrays input to app_geo_from_icrs all need to " "have the same length",
        )

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _app_geo_from_icrs(
                ra_icrs,
                dec_icrs,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax,
                v_rad=v_rad[:2],
                mjd=mjd,
            )

        self.assertEqual(
            context.exception.args[0],
            "The arrays input to app_geo_from_icrs all need to " "have the same length",
        )

    def test_app_geo_from_icrs_no_motion(self):
        """
        Test that app_geo_from_icrs with parallax, proper motion, and radial velocity
        set to None behaves the same as appGeoFromICRs with parallax, proper motion
        and radial velocity set to zero.
        """
        obs = ObservationMetaData(pointing_ra=25.0, pointing_dec=-11.0, mjd=59781.2)

        rng = np.random.RandomState(88)
        n_obj = 100
        ra_list = rng.random_sample(n_obj) * 2.0 * np.pi
        dec_list = rng.random_sample(n_obj) * np.pi - 0.5 * np.pi
        px_list = np.zeros(n_obj)
        vrad_list = np.zeros(n_obj)
        pm_ra_list = np.zeros(n_obj)
        pm_dec_list = np.zeros(n_obj)

        control_ra, control_dec = _app_geo_from_icrs(
            ra_list,
            dec_list,
            mjd=obs.mjd,
            pm_ra=pm_ra_list,
            pm_dec=pm_dec_list,
            parallax=px_list,
            v_rad=vrad_list,
            epoch=2000.0,
        )

        test_ra, test_dec = _app_geo_from_icrs(ra_list, dec_list, mjd=obs.mjd, epoch=2000.0)

        dd_sun = _distance_to_sun(ra_list, dec_list, obs.mjd)
        valid = np.where(dd_sun > np.radians(20.0))
        self.assertGreater(len(valid[0]), n_obj / 3)

        dd = _angular_separation(test_ra[valid], test_dec[valid], control_ra[valid], control_dec[valid])
        self.assertLess(arcsec_from_radians(dd).max(), 0.005)

    def test_icrs_from_app_geo(self):
        """
        Test that _icrs_from_app_geo really inverts _app_geo_from_icrs.

        This test is a tricky because _app_geo_from_icrs applies
        light deflection due to the sun.  _icrs_from_app_geo does not
        account for that effect, which is fine, because it is only
        meant to map pointing RA, Decs to RA, Decs on fatboy.

        _icrs_from_app_geo should invert _app_geo_from_icrs to within
        0.01 arcsec at an angular distance greater than 45 degrees
        from the sun.
        """

        rng = np.random.RandomState(412)
        n_samples = 100

        for tai in (53000.0, 53241.6, 58504.6):
            mjd = ModifiedJulianDate(TAI=tai)

            ra_in = rng.random_sample(n_samples) * 2.0 * np.pi
            dec_in = (rng.random_sample(n_samples) - 0.5) * np.pi

            ra_app, dec_app = _app_geo_from_icrs(ra_in, dec_in, mjd=mjd)

            ra_icrs, dec_icrs = _icrs_from_app_geo(ra_app, dec_app, epoch=2000.0, mjd=mjd)

            self.assertFalse(np.isnan(ra_icrs).any(), msg="There were NaNs in ra_icrs; should not be")
            self.assertFalse(
                np.isnan(dec_icrs).any(),
                msg="There were NaNs in dec_icrs; should not be",
            )

            valid_pts = np.where(_distance_to_sun(ra_in, dec_in, mjd) > 0.25 * np.pi)[0]

            self.assertGreater(len(valid_pts), 0)

            distance = arcsec_from_radians(
                pal.dsepVector(
                    ra_in[valid_pts],
                    dec_in[valid_pts],
                    ra_icrs[valid_pts],
                    dec_icrs[valid_pts],
                )
            )

            self.assertLess(distance.max(), 0.01)

            # test passing in floats
            for ix in valid_pts:
                ra_test, dec_test = _icrs_from_app_geo(ra_app[ix], dec_app[ix], mjd=mjd)
                self.assertIsInstance(ra_test, float)
                self.assertIsInstance(dec_test, float)
                distance_f = arcsec_from_radians(pal.dsep(ra_in[ix], dec_in[ix], ra_test, dec_test))
                self.assertLess(distance_f, 0.01)

    def test_icrs_from_observed(self):
        """
        Test that _icrs_from_observed really inverts _observed_from_icrs and that
        _app_geo_from_observed really does invert _observed_from_app_geo.

        In this case, the method is only reliable at distances of more than
        45 degrees from the sun and at zenith distances less than 70 degrees.
        """

        rng = np.random.RandomState(412)
        n_samples = 100

        site = Site(name="LSST")

        for tai in (53000.0, 53241.6, 58504.6):
            for include_refraction in (True, False):
                for ra_pointing in (23.5, 256.9, 100.0):
                    for dec_pointing in (-12.0, 45.0, 66.8):
                        obs = ObservationMetaData(mjd=tai, site=site)

                        ra_zenith, dec_zenith = _ra_dec_from_alt_az(0.5 * np.pi, 0.0, obs)

                        obs = ObservationMetaData(
                            pointing_ra=ra_pointing,
                            pointing_dec=dec_pointing,
                            mjd=tai,
                            site=site,
                        )

                        rr = rng.random_sample(n_samples) * np.radians(50.0)
                        theta = rng.random_sample(n_samples) * 2.0 * np.pi

                        ra_in = ra_zenith + rr * np.cos(theta)
                        dec_in = dec_zenith + rr * np.sin(theta)

                        # test a round-trip between observed_from_icrs and
                        # icrs_from_observed
                        ra_obs, dec_obs = _observed_from_icrs(
                            ra_in,
                            dec_in,
                            obs_metadata=obs,
                            include_refraction=include_refraction,
                            epoch=2000.0,
                        )

                        ra_icrs, dec_icrs = _icrs_from_observed(
                            ra_obs,
                            dec_obs,
                            obs_metadata=obs,
                            include_refraction=include_refraction,
                            epoch=2000.0,
                        )

                        valid_pts = np.where(_distance_to_sun(ra_in, dec_in, obs.mjd) > 0.25 * np.pi)[0]

                        self.assertGreater(len(valid_pts), 0)

                        distance = arcsec_from_radians(
                            pal.dsepVector(
                                ra_in[valid_pts],
                                dec_in[valid_pts],
                                ra_icrs[valid_pts],
                                dec_icrs[valid_pts],
                            )
                        )

                        self.assertLess(distance.max(), 1.0e-6)

                        # test a round-trip between observed_from_app_geo and
                        # app_geo_from_observed
                        (
                            ra_obs_from_app_geo,
                            dec_obs_from_app_geo,
                        ) = _observed_from_app_geo(
                            ra_in,
                            dec_in,
                            obs_metadata=obs,
                            include_refraction=include_refraction,
                        )

                        ra_app, dec_app = _app_geo_from_observed(
                            ra_obs_from_app_geo,
                            dec_obs_from_app_geo,
                            obs_metadata=obs,
                            include_refraction=include_refraction,
                        )

                        distance = arcsec_from_radians(
                            pal.dsepVector(
                                ra_in[valid_pts],
                                dec_in[valid_pts],
                                ra_app[valid_pts],
                                dec_app[valid_pts],
                            )
                        )

                        self.assertLess(distance.max(), 1.0e-6)

                        # test that passing arguments in as floats gives consistent
                        # results
                        for ix in valid_pts:
                            ra_f, dec_f = _icrs_from_observed(
                                ra_obs[ix],
                                dec_obs[ix],
                                obs_metadata=obs,
                                include_refraction=include_refraction,
                                epoch=2000.0,
                            )
                            self.assertIsInstance(ra_f, float)
                            self.assertIsInstance(dec_f, float)
                            dist_f = arcsec_from_radians(pal.dsep(ra_icrs[ix], dec_icrs[ix], ra_f, dec_f))
                            self.assertLess(dist_f, 1.0e-9)

                            ra_f, dec_f = _observed_from_app_geo(
                                ra_in[ix],
                                dec_in[ix],
                                obs_metadata=obs,
                                include_refraction=include_refraction,
                            )
                            self.assertIsInstance(ra_f, float)
                            self.assertIsInstance(dec_f, float)
                            dist_f = arcsec_from_radians(
                                pal.dsep(
                                    ra_obs_from_app_geo[ix],
                                    dec_obs_from_app_geo[ix],
                                    ra_f,
                                    dec_f,
                                )
                            )
                            self.assertLess(dist_f, 1.0e-9)

                            ra_f, dec_f = _app_geo_from_observed(
                                ra_obs_from_app_geo[ix],
                                dec_obs_from_app_geo[ix],
                                obs_metadata=obs,
                                include_refraction=include_refraction,
                            )
                            self.assertIsInstance(ra_f, float)
                            self.assertIsInstance(dec_f, float)
                            dist_f = arcsec_from_radians(pal.dsep(ra_app[ix], dec_app[ix], ra_f, dec_f))
                            self.assertLess(dist_f, 1.0e-9)

    def test_icrs_from_observed_no_refraction(self):
        """
        Test that _icrs_from_observed really does invert _observed_from_icrs
        in the case of no refraction.
        """
        rng = np.random.RandomState(85)
        n_batches = 10
        n_samples = 10
        for i_batch in range(n_batches):
            _d_sun = 0.0
            while _d_sun < 0.25 * np.pi:  # because ICRS -> Observed conversion breaks close to the sun
                mjd = rng.random_sample(1)[0] * 10000.0 + 40000.0
                obs = ObservationMetaData(mjd=mjd)
                ra_in = rng.random_sample(n_samples) * np.pi * 2.0
                dec_in = rng.random_sample(n_samples) * np.pi - 0.5 * np.pi

                _d_sun = _distance_to_sun(ra_in, dec_in, obs.mjd).min()

            ra_obs, dec_obs = _observed_from_icrs(
                ra_in, dec_in, obs_metadata=obs, include_refraction=False, epoch=2000.0
            )
            ra_icrs, dec_icrs = _icrs_from_observed(
                ra_obs,
                dec_obs,
                obs_metadata=obs,
                include_refraction=False,
                epoch=2000.0,
            )
            distance = pal.dsepVector(ra_in, dec_in, ra_icrs, dec_icrs)

            self.assertLess(arcsec_from_radians(distance).max(), 0.01)

    def test_icrs_from_observed_exceptions(self):
        """
        Test that _icrs_from_observed raises exceptions when it is supposed to.
        """
        rng = np.random.RandomState(33)
        ra_in = rng.random_sample(10)
        dec_in = rng.random_sample(10)
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrs_from_observed(ra_in, dec_in, epoch=2000.0)
        self.assertEqual(
            context.exception.args[0],
            "Cannot call icrs_from_observed; obs_metadata is None",
        )

        obs = ObservationMetaData(pointing_ra=23.0, pointing_dec=-19.0)
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrs_from_observed(ra_in, dec_in, epoch=2000.0, obs_metadata=obs)
        self.assertEqual(
            context.exception.args[0],
            "Cannot call icrs_from_observed; obs_metadata.mjd is None",
        )

        obs = ObservationMetaData(pointing_ra=23.0, pointing_dec=-19.0, mjd=ModifiedJulianDate(TAI=52344.0))
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrs_from_observed(ra_in, dec_in, obs_metadata=obs)
        self.assertEqual(
            context.exception.args[0],
            "Cannot call icrs_from_observed; you have not specified an epoch",
        )

        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrs_from_observed(ra_in[:3], dec_in, obs_metadata=obs, epoch=2000.0)
        self.assertEqual(
            context.exception.args[0],
            "The arrays input to icrs_from_observed all need to have the same length",
        )

    def test_app_geo_from_observed(self):
        """
        Test that _app_geo_from_observed really does invert _observed_from_app_geo
        """
        mjd = 58350.0
        site = Site(longitude=np.degrees(0.235), latitude=np.degrees(-1.2), name="LSST")
        ra_center, dec_center = ra_dec_from_alt_az(90.0, 0.0, ObservationMetaData(mjd=mjd, site=site))

        obs = ObservationMetaData(
            pointing_ra=ra_center,
            pointing_dec=dec_center,
            mjd=ModifiedJulianDate(TAI=58350.0),
            site=site,
        )

        rng = np.random.RandomState(125543)
        n_samples = 200

        # Note: the PALPY routines in question start to become inaccurate at
        # a zenith distance of about 75 degrees, so we restrict our test points
        # to be within 50 degrees of the telescope pointing, which is at zenith
        # in a flat sky approximation
        rr = rng.random_sample(n_samples) * np.radians(50.0)
        theta = rng.random_sample(n_samples) * 2.0 * np.pi
        ra_in = np.radians(ra_center) + rr * np.cos(theta)
        dec_in = np.radians(dec_center) + rr * np.sin(theta)

        xx_in = np.cos(dec_in) * np.cos(ra_in)
        yy_in = np.cos(dec_in) * np.sin(ra_in)
        zz_in = np.sin(dec_in)

        for include_refraction in [True, False]:
            for wavelength in (0.5, 0.3, 0.7):
                ra_obs, dec_obs = _observed_from_app_geo(
                    ra_in,
                    dec_in,
                    obs_metadata=obs,
                    wavelength=wavelength,
                    include_refraction=include_refraction,
                )

                ra_out, dec_out = _app_geo_from_observed(
                    ra_obs,
                    dec_obs,
                    obs_metadata=obs,
                    wavelength=wavelength,
                    include_refraction=include_refraction,
                )

                xx_out = np.cos(dec_out) * np.cos(ra_out)
                yy_out = np.cos(dec_out) * np.sin(ra_out)
                zz_out = np.sin(dec_out)

                distance = np.sqrt(
                    np.power(xx_in - xx_out, 2) + np.power(yy_in - yy_out, 2) + np.power(zz_in - zz_out, 2)
                )

                self.assertLess(distance.max(), 1.0e-12)

    def test_app_geo_from_observed_exceptions(self):
        """
        Test that _app_geo_from_observed raises exceptions where expected
        """
        rng = np.random.RandomState(12)
        ra_in = rng.random_sample(10) * 2.0 * np.pi
        dec_in = (rng.random_sample(10) - 0.5) * np.pi

        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _app_geo_from_observed(ra_in, dec_in)
        self.assertEqual(
            context.exception.args[0],
            "Cannot call app_geo_from_observed without an obs_metadata",
        )

        obs = ObservationMetaData(
            pointing_ra=25.0,
            pointing_dec=-12.0,
            site=None,
            mjd=ModifiedJulianDate(TAI=52000.0),
        )

        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _app_geo_from_observed(ra_in, dec_in, obs_metadata=obs)
        self.assertEqual(
            context.exception.args[0],
            "Cannot call app_geo_from_observed: obs_metadata has no site info",
        )

        obs = ObservationMetaData(pointing_ra=25.0, pointing_dec=-12.0)
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _app_geo_from_observed(ra_in, dec_in, obs_metadata=obs)
        self.assertEqual(
            context.exception.args[0],
            "Cannot call app_geo_from_observed: obs_metadata has no mjd",
        )

        obs = ObservationMetaData(pointing_ra=25.0, pointing_dec=-12.0, mjd=ModifiedJulianDate(TAI=52000.0))
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _app_geo_from_observed(ra_in[:2], dec_in, obs_metadata=obs)
        self.assertEqual(
            context.exception.args[0],
            "The arrays input to app_geo_from_observed all need to have the same length",
        )

    def test_refraction_coefficients(self):
        output = refraction_coefficients(wavelength=5000.0, site=self.obs_metadata.site)

        self.assertAlmostEqual(output[0], 2.295817926320665320e-04, 6)
        self.assertAlmostEqual(output[1], -2.385964632924575670e-07, 6)

    def test_apply_refraction(self):
        coeffs = refraction_coefficients(wavelength=5000.0, site=self.obs_metadata.site)

        output = apply_refraction(0.25 * np.pi, coeffs[0], coeffs[1])

        self.assertAlmostEqual(output, 7.851689251070859132e-01, 6)

        # test that passing in a numpy array and passing in floats
        # give the same results
        rng = np.random.RandomState(712)
        zd_arr = rng.random_sample(20) * np.pi * 0.2
        control_refraction = apply_refraction(zd_arr, coeffs[0], coeffs[1])
        for ix, zz in enumerate(zd_arr):
            test_refraction = apply_refraction(zz, coeffs[0], coeffs[1])
            self.assertAlmostEqual(test_refraction, control_refraction[ix], 12)

    def test_apply_proper_motion_vs_icrs(self):
        """
        test that running:
        apply_proper_motion() -> observed_from_icrs(pm=0)
        gives the same results as running
        observed_from_icrs(pm!=0)
        """
        rng = np.random.RandomState(18293)
        n_obj = 500
        ra = 46.2
        dec = -14.2

        # generate a set of points uniformly distributed on the
        # unit sphere
        xyz_list = rng.normal(loc=0.0, scale=1.0, size=(n_obj, 3))
        ra_list, dec_list = spherical_from_cartesian(xyz_list)
        self.assertEqual(len(ra_list), n_obj)
        ra_list = np.degrees(ra_list)
        dec_list = np.degrees(dec_list)

        px_list = np.array([0.2] * n_obj)
        vrad_list = np.array([200.0] * n_obj)
        pm_ra_list = np.array([30.0] * n_obj)
        pm_dec_list = np.array([-30.0] * n_obj)

        obs = ObservationMetaData(pointing_ra=ra, pointing_dec=dec, mjd=60123.0)

        for include_refraction in (True, False):
            ra_control, dec_control = observed_from_icrs(
                ra_list,
                dec_list,
                pm_ra=pm_ra_list,
                pm_dec=pm_dec_list,
                v_rad=vrad_list,
                parallax=px_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            ra_pm, dec_pm = apply_proper_motion(
                ra_list,
                dec_list,
                pm_ra_list,
                pm_dec_list,
                parallax=px_list,
                v_rad=vrad_list,
                mjd=obs.mjd,
                epoch=2000.0,
            )

            ra_test, dec_test = observed_from_icrs(
                ra_pm,
                dec_pm,
                parallax=px_list,
                v_rad=vrad_list,
                obs_metadata=obs,
                epoch=2000.0,
                include_refraction=include_refraction,
            )

            # the distance between the test points and the control points
            dd = arcsec_from_radians(
                haversine(
                    np.radians(ra_test),
                    np.radians(dec_test),
                    np.radians(ra_control),
                    np.radians(dec_control),
                )
            )

            self.assertLess(dd.max(), 0.005)

            # the distance between the origina points and the motion-propagated points
            dd_bad = arcsec_from_radians(
                haversine(
                    np.radians(ra_control),
                    np.radians(dec_control),
                    np.radians(ra_list),
                    np.radians(dec_list),
                )
            )

            self.assertGreater(dd_bad.min(), 10.0)


if __name__ == "__main__":
    unittest.main()
