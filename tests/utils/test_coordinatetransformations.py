import unittest

import numpy as np

import rubin_sim.utils as utils


def control_equation_of_equinoxes(mjd):
    """
    Taken from http://aa.usno.navy.mil/faq/docs/GAST.php

    Parameters
    ----------
    mjd : `Unknown`
        is Terrestrial Time as a Modified Julian Date

    Parameters
    ----------
    the : `Unknown`
        equation of equinoxes in radians
    """

    JD = mjd + 2400000.5
    D = JD - 2451545.0
    omega_degrees = 125.04 - 0.052954 * D
    ldegrees = 280.47 + 0.98565 * D
    delta_psi_hours = -0.000319 * np.sin(np.radians(omega_degrees)) - 0.000024 * np.sin(
        2.0 * np.radians(ldegrees)
    )
    epsilon_degrees = 23.4393 - 0.0000004 * D
    return (delta_psi_hours / 24.0) * 2.0 * np.pi * np.cos(np.radians(epsilon_degrees))


def control_calc_gmst_gast(mjd):
    # From http://aa.usno.navy.mil/faq/docs/GAST.php Nov. 9 2013
    mjd_conv = 2400000.5
    jd2000 = 2451545.0
    mjd_o = np.floor(mjd)
    jd = mjd + mjd_conv
    jd_o = mjd_o + mjd_conv
    h = 24.0 * (jd - jd_o)
    d = jd - jd2000
    d_o = jd_o - jd2000
    t = d / 36525.0
    gmst = 6.697374558 + 0.06570982441908 * d_o + 1.00273790935 * h + 0.000026 * t**2
    gast = gmst + 24.0 * utils.equation_of_equinoxes(mjd) / (2.0 * np.pi)
    gmst %= 24.0
    gast %= 24.0
    return gmst, gast


class AngularSeparationTestCase(unittest.TestCase):
    def test_ang_sep_exceptions(self):
        """
        Test that an exception is raised when you pass
        mismatched inputs to angular_separation.
        """
        ra1 = 23.0
        dec1 = -12.0
        ra2 = 45.0
        dec2 = 33.1
        ra1_arr = np.array([11.0, 21.0, 33.1])
        dec1_arr = np.array([-11.1, 34.1, 86.2])
        ra2_arr = np.array([45.2, 112.0, 89.3])
        dec2_arr = np.array([11.1, -45.0, -71.0])

        # test that everything runs
        utils._angular_separation(np.radians(ra1), np.radians(dec1), np.radians(ra2), np.radians(dec2))
        utils.angular_separation(ra1, dec1, ra2, dec2)
        ans = utils._angular_separation(
            np.radians(ra1_arr),
            np.radians(dec1_arr),
            np.radians(ra2_arr),
            np.radians(dec2_arr),
        )
        self.assertEqual(len(ans), 3)
        ans = utils.angular_separation(ra1_arr, dec1_arr, ra2_arr, dec2_arr)
        self.assertEqual(len(ans), 3)

        ans = utils.angular_separation(ra1_arr, dec1_arr, ra2, dec2)
        self.assertEqual(len(ans), 3)
        ans = utils._angular_separation(ra1_arr, ra2_arr, ra2, dec2)
        self.assertEqual(len(ans), 3)

        ans = utils.angular_separation(ra1, dec1, ra2_arr, dec2_arr)
        self.assertEqual(len(ans), 3)
        ans = utils._angular_separation(dec1, ra1, ra2_arr, dec2_arr)
        self.assertEqual(len(ans), 3)

        # test with lists
        # Note: these exceptions will not get raised if you call
        # angular_separation() with lists instead of numpy arrays
        # because the conversion from degrees to radians with
        # np.radians will automatically convert the lists into arrays
        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(list(ra1_arr), dec1_arr, ra2_arr, dec2_arr)
        self.assertIn("number", context.exception.args[0])
        self.assertIn("numpy array", context.exception.args[0])
        self.assertIn("long1", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, list(dec1_arr), ra2_arr, dec2_arr)
        self.assertIn("number", context.exception.args[0])
        self.assertIn("numpy array", context.exception.args[0])
        self.assertIn("lat1", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1_arr, list(ra2_arr), dec2_arr)
        self.assertIn("number", context.exception.args[0])
        self.assertIn("numpy array", context.exception.args[0])
        self.assertIn("long2", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1_arr, ra2_arr, list(dec2_arr))
        self.assertIn("number", context.exception.args[0])
        self.assertIn("numpy array", context.exception.args[0])
        self.assertIn("lat2", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        # test with numbers and arrays
        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1, ra2, dec2)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1, dec1_arr, ra2, dec2)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1, dec1, ra2_arr, dec2)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1, dec1, ra2, dec2_arr)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr, dec1, ra2, dec2)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1, dec1_arr, ra2, dec2)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1, dec1, ra2_arr, dec2)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1, dec1, ra2, dec2_arr)
        self.assertIn("the same type", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        # test with mismatched arrays
        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr[:2], dec1_arr, ra2_arr, dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1_arr[:2], ra2_arr, dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1_arr, ra2_arr[:2], dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1_arr, ra2_arr, dec2_arr[:2])
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr[:2], dec1_arr, ra2_arr, dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr, dec1_arr[:2], ra2_arr, dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr, dec1_arr, ra2_arr[:2], dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr, dec1_arr, ra2_arr, dec2_arr[:2])
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr[:2], dec1_arr[:2], ra2_arr, dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1_arr, dec1_arr, ra2_arr[:2], dec2_arr[:2])
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr[:2], dec1_arr[:2], ra2_arr, dec2_arr)
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils.angular_separation(ra1_arr, dec1_arr, ra2_arr[:2], dec2_arr[:2])
        self.assertIn("same length", context.exception.args[0])
        self.assertIn("angular_separation", context.exception.args[0])

        # test that a sensible error is raised if you pass a string
        # into angular_separation
        # Note: a different error will be raised if you pass these
        # bad inputs into angular_separation().  The exception will come
        # from trying to convert a str with np.radians
        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation("a", dec1, ra2, dec2)
        self.assertIn("angular_separation", context.exception.args[0])
        self.assertIn("number", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1, "a", ra2, dec2)
        self.assertIn("angular_separation", context.exception.args[0])
        self.assertIn("number", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1, dec1, "a", dec2)
        self.assertIn("angular_separation", context.exception.args[0])
        self.assertIn("number", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            utils._angular_separation(ra1, dec1, ra2, "a")
        self.assertIn("angular_separation", context.exception.args[0])
        self.assertIn("number", context.exception.args[0])

    def test_ang_sep_results_arr(self):
        """
        Test that angular_separation gives the correct answer by comparing
        results with the dot products of Cartesian vectors.  Pass in arrays
        of arguments.
        """
        # Looks like this might be version? 15 works on older verisons, but
        # may need to go less-precice on newer versions
        precision = 14

        rng = np.random.RandomState(99421)
        n_obj = 100
        ra1 = rng.random_sample(n_obj) * 2.0 * np.pi
        dec1 = rng.random_sample(n_obj) * np.pi - 0.5 * np.pi
        ra2 = rng.random_sample(n_obj) * 2.0 * np.pi
        dec2 = rng.random_sample(n_obj) * np.pi - 0.5 * np.pi

        x1 = np.cos(dec1) * np.cos(ra1)
        y1 = np.cos(dec1) * np.sin(ra1)
        z1 = np.sin(dec1)

        x2 = np.cos(dec2) * np.cos(ra2)
        y2 = np.cos(dec2) * np.sin(ra2)
        z2 = np.sin(dec2)

        test = utils._angular_separation(ra1, dec1, ra2, dec2)
        test = np.cos(test)
        control = x1 * x2 + y1 * y2 + z1 * z2
        np.testing.assert_array_almost_equal(test, control, decimal=precision)

        test = utils.angular_separation(np.degrees(ra1), np.degrees(dec1), np.degrees(ra2), np.degrees(dec2))
        test = np.cos(np.radians(test))
        np.testing.assert_array_almost_equal(test, control, decimal=precision)

        # specifically test at the north pole
        dec1 = np.ones(n_obj) * np.pi
        x1 = np.cos(dec1) * np.cos(ra1)
        y1 = np.cos(dec1) * np.sin(ra1)
        z1 = np.sin(dec1)
        control = x1 * x2 + y1 * y2 + z1 * z2
        test = utils._angular_separation(ra1, dec1, ra2, dec2)
        test = np.cos(test)
        np.testing.assert_array_almost_equal(test, control, decimal=precision)

        test = utils.angular_separation(np.degrees(ra1), np.degrees(dec1), np.degrees(ra2), np.degrees(dec2))
        test = np.cos(np.radians(test))
        np.testing.assert_array_almost_equal(test, control, decimal=precision)

        # specifically test at the south pole
        dec1 = -1.0 * np.ones(n_obj) * np.pi
        x1 = np.cos(dec1) * np.cos(ra1)
        y1 = np.cos(dec1) * np.sin(ra1)
        z1 = np.sin(dec1)
        control = x1 * x2 + y1 * y2 + z1 * z2
        test = utils._angular_separation(ra1, dec1, ra2, dec2)
        test = np.cos(test)
        np.testing.assert_array_almost_equal(test, control, decimal=precision)

        test = utils.angular_separation(np.degrees(ra1), np.degrees(dec1), np.degrees(ra2), np.degrees(dec2))
        test = np.cos(np.radians(test))
        np.testing.assert_array_almost_equal(test, control, decimal=precision)

    def test_ang_sep_results_extreme(self):
        """
        Test that angular_separation gives the correct answer by comparing
        results with the dot products of Cartesian vectors.  Test on extremal
        values (i.e. longitudes that go beyond 360.0 and latitudes that go
        beyond 90.0)
        """
        rng = np.random.RandomState(99421)
        n_obj = 100
        for sgn in (-1.0, 1.0):
            ra1 = sgn * (rng.random_sample(n_obj) * 2.0 * np.pi + 2.0 * np.pi)
            dec1 = sgn * (rng.random_sample(n_obj) * 4.0 * np.pi + 2.0 * np.pi)
            ra2 = sgn * (rng.random_sample(n_obj) * 2.0 * np.pi + 2.0 * np.pi)
            dec2 = sgn * (rng.random_sample(n_obj) * 2.0 * np.pi + 2.0 * np.pi)

            x1 = np.cos(dec1) * np.cos(ra1)
            y1 = np.cos(dec1) * np.sin(ra1)
            z1 = np.sin(dec1)

            x2 = np.cos(dec2) * np.cos(ra2)
            y2 = np.cos(dec2) * np.sin(ra2)
            z2 = np.sin(dec2)

            test = utils._angular_separation(ra1, dec1, ra2, dec2)
            test = np.cos(test)
            control = x1 * x2 + y1 * y2 + z1 * z2
            np.testing.assert_array_almost_equal(test, control, decimal=14)

            test = utils.angular_separation(
                np.degrees(ra1), np.degrees(dec1), np.degrees(ra2), np.degrees(dec2)
            )
            test = np.cos(np.radians(test))
            np.testing.assert_array_almost_equal(test, control, decimal=14)

            # specifically test at the north pole
            dec1 = np.ones(n_obj) * np.pi
            x1 = np.cos(dec1) * np.cos(ra1)
            y1 = np.cos(dec1) * np.sin(ra1)
            z1 = np.sin(dec1)
            control = x1 * x2 + y1 * y2 + z1 * z2
            test = utils._angular_separation(ra1, dec1, ra2, dec2)
            test = np.cos(test)
            np.testing.assert_array_almost_equal(test, control, decimal=14)

            test = utils.angular_separation(
                np.degrees(ra1), np.degrees(dec1), np.degrees(ra2), np.degrees(dec2)
            )
            test = np.cos(np.radians(test))
            dd = np.abs(test - control)
            np.testing.assert_array_almost_equal(test, control, decimal=14)

            # specifically test at the south pole
            dec1 = -1.0 * np.ones(n_obj) * np.pi
            x1 = np.cos(dec1) * np.cos(ra1)
            y1 = np.cos(dec1) * np.sin(ra1)
            z1 = np.sin(dec1)
            control = x1 * x2 + y1 * y2 + z1 * z2
            test = utils._angular_separation(ra1, dec1, ra2, dec2)
            test = np.cos(test)
            np.testing.assert_array_almost_equal(test, control, decimal=14)

            test = utils.angular_separation(
                np.degrees(ra1), np.degrees(dec1), np.degrees(ra2), np.degrees(dec2)
            )
            test = np.cos(np.radians(test))
            np.testing.assert_array_almost_equal(test, control, decimal=14)

    def test_ang_sep_results_float(self):
        """
        Test that angular_separation gives the correct answer by comparing
        results with the dot products of Cartesian vectors.  Pass in floats
        as arguments.
        """

        precision = 14

        rng = np.random.RandomState(831)
        ra1 = rng.random_sample() * 2.0 * np.pi
        dec1 = rng.random_sample() * np.pi - 0.5 * np.pi
        ra2 = rng.random_sample() * 2.0 * np.pi
        dec2 = rng.random_sample() * np.pi - 0.5 * np.pi
        x1 = np.cos(dec1) * np.cos(ra1)
        y1 = np.cos(dec1) * np.sin(ra1)
        z1 = np.sin(dec1)

        x2 = np.cos(dec2) * np.cos(ra2)
        y2 = np.cos(dec2) * np.sin(ra2)
        z2 = np.sin(dec2)

        control = x1 * x2 + y1 * y2 + z1 * z2

        test = utils._angular_separation(ra1, dec1, ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(test)
        self.assertAlmostEqual(control, test, precision)

        test = utils._angular_separation(np.array([ra1]), np.array([dec1]), ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(test)
        self.assertAlmostEqual(control, test, precision)

        test = utils._angular_separation(ra1, dec1, np.array([ra2]), np.array([dec2]))
        self.assertIsInstance(test, float)
        test = np.cos(test)
        self.assertAlmostEqual(control, test, precision)

        # try north pole
        ra1 = 0.5 * np.pi
        x1 = np.cos(dec1) * np.cos(ra1)
        y1 = np.cos(dec1) * np.sin(ra1)
        z1 = np.sin(dec1)
        control = x1 * x2 + y1 * y2 + z1 * z2

        test = utils._angular_separation(ra1, dec1, ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(test)
        self.assertAlmostEqual(control, test, precision)

        test = utils._angular_separation(np.array([ra1]), np.array([dec1]), ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(test)
        self.assertAlmostEqual(control, test, precision)

        test = utils._angular_separation(ra1, dec1, np.array([ra2]), np.array([dec2]))
        self.assertIsInstance(test, float)
        test = np.cos(test)
        self.assertAlmostEqual(control, test, precision)

        # do all of that in degrees
        ra1 = rng.random_sample() * 360.0
        dec1 = rng.random_sample() * 180.0 - 90.0
        ra2 = rng.random_sample() * 360.0
        dec2 = rng.random_sample() * 180.0 - 90.0
        x1 = np.cos(np.radians(dec1)) * np.cos(np.radians(ra1))
        y1 = np.cos(np.radians(dec1)) * np.sin(np.radians(ra1))
        z1 = np.sin(np.radians(dec1))

        x2 = np.cos(np.radians(dec2)) * np.cos(np.radians(ra2))
        y2 = np.cos(np.radians(dec2)) * np.sin(np.radians(ra2))
        z2 = np.sin(np.radians(dec2))

        control = x1 * x2 + y1 * y2 + z1 * z2

        test = utils.angular_separation(ra1, dec1, ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(np.radians(test))
        self.assertAlmostEqual(control, test, precision)

        test = utils.angular_separation(np.array([ra1]), np.array([dec1]), ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(np.radians(test))
        self.assertAlmostEqual(control, test, precision)

        test = utils.angular_separation(ra1, dec1, np.array([ra2]), np.array([dec2]))
        self.assertIsInstance(test, float)
        test = np.cos(np.radians(test))
        self.assertAlmostEqual(control, test, precision)

        # try north pole
        ra1 = 90.0
        x1 = np.cos(np.radians(dec1)) * np.cos(np.radians(ra1))
        y1 = np.cos(np.radians(dec1)) * np.sin(np.radians(ra1))
        z1 = np.sin(np.radians(dec1))
        control = x1 * x2 + y1 * y2 + z1 * z2

        test = utils.angular_separation(ra1, dec1, ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(np.radians(test))
        self.assertAlmostEqual(control, test, precision)

        test = utils.angular_separation(np.array([ra1]), np.array([dec1]), ra2, dec2)
        self.assertIsInstance(test, float)
        test = np.cos(np.radians(test))
        self.assertAlmostEqual(control, test, precision)

        test = utils.angular_separation(ra1, dec1, np.array([ra2]), np.array([dec2]))
        self.assertIsInstance(test, float)
        test = np.cos(np.radians(test))
        self.assertAlmostEqual(control, test, precision)

    def test_ang_sep_results_mixed(self):
        """
        Test that angular_separation gives the correct answer by comparing
        results with the dot products of Cartesian vectors.  Pass in mixtures
        of floats and arrays as arguments.
        """
        rng = np.random.RandomState(8131)
        n_obj = 100
        ra1 = rng.random_sample(n_obj) * 2.0 * np.pi
        dec1 = rng.random_sample(n_obj) * np.pi - 0.5 * np.pi
        ra2 = rng.random_sample() * 2.0 * np.pi
        dec2 = rng.random_sample() * np.pi - 0.5 * np.pi
        self.assertIsInstance(ra1, np.ndarray)
        self.assertIsInstance(dec1, np.ndarray)
        self.assertIsInstance(ra2, float)
        self.assertIsInstance(dec2, float)

        x1 = np.cos(dec1) * np.cos(ra1)
        y1 = np.cos(dec1) * np.sin(ra1)
        z1 = np.sin(dec1)

        x2 = np.cos(dec2) * np.cos(ra2)
        y2 = np.cos(dec2) * np.sin(ra2)
        z2 = np.sin(dec2)

        control = x1 * x2 + y1 * y2 + z1 * z2
        test = utils._angular_separation(ra1, dec1, ra2, dec2)
        test = np.cos(test)
        np.testing.assert_array_almost_equal(test, control, decimal=15)
        test = utils._angular_separation(ra2, dec2, ra1, dec1)
        test = np.cos(test)
        np.testing.assert_array_almost_equal(test, control, decimal=15)

        # now do it in degrees
        ra1 = rng.random_sample(n_obj) * 360.0
        dec1 = rng.random_sample(n_obj) * 180.0 - 90.0
        ra2 = rng.random_sample() * 360.0
        dec2 = rng.random_sample() * 180.0 - 90.0
        self.assertIsInstance(ra1, np.ndarray)
        self.assertIsInstance(dec1, np.ndarray)
        self.assertIsInstance(ra2, float)
        self.assertIsInstance(dec2, float)

        x1 = np.cos(np.radians(dec1)) * np.cos(np.radians(ra1))
        y1 = np.cos(np.radians(dec1)) * np.sin(np.radians(ra1))
        z1 = np.sin(np.radians(dec1))

        x2 = np.cos(np.radians(dec2)) * np.cos(np.radians(ra2))
        y2 = np.cos(np.radians(dec2)) * np.sin(np.radians(ra2))
        z2 = np.sin(np.radians(dec2))

        control = x1 * x2 + y1 * y2 + z1 * z2
        test = utils.angular_separation(ra1, dec1, ra2, dec2)
        test = np.cos(np.radians(test))
        np.testing.assert_array_almost_equal(test, control, decimal=15)
        test = utils.angular_separation(ra2, dec2, ra1, dec1)
        test = np.cos(np.radians(test))
        np.testing.assert_array_almost_equal(test, control, decimal=15)

    def test_haversine(self):
        """
        Test that haversine() returns the same thing as _angular_separation
        """

        ra1 = 0.2
        dec1 = 1.3
        ra2 = 2.1
        dec2 = -0.5
        ra3 = np.array([1.9, 2.1, 0.3])
        dec3 = np.array([-1.1, 0.34, 0.01])
        control = utils._angular_separation(ra1, dec1, ra2, dec2)
        test = utils.haversine(ra1, dec1, ra2, dec2)
        self.assertIsInstance(test, float)
        self.assertEqual(test, control)

        control = utils._angular_separation(ra1, dec1, ra3, dec3)
        test = utils.haversine(ra1, dec1, ra3, dec3)
        np.testing.assert_array_equal(test, control)

        control = utils._angular_separation(np.array([ra1]), np.array([dec1]), ra3, dec3)
        test = utils.haversine(np.array([ra1]), np.array([dec1]), ra3, dec3)
        np.testing.assert_array_equal(test, control)

        control = utils._angular_separation(ra3, dec3, np.array([ra1]), np.array([dec1]))
        test = utils.haversine(ra3, dec3, np.array([ra1]), np.array([dec1]))
        np.testing.assert_array_equal(test, control)

        control = utils._angular_separation(ra2, dec2, np.array([ra1]), np.array([dec1]))
        test = utils.haversine(ra2, dec2, np.array([ra1]), np.array([dec1]))
        self.assertIsInstance(test, float)
        self.assertEqual(test, control)

        control = utils._angular_separation(np.array([ra1]), np.array([dec1]), ra2, dec2)
        test = utils.haversine(np.array([ra1]), np.array([dec1]), ra2, dec2)
        self.assertIsInstance(test, float)
        self.assertEqual(test, control)


class TestCoordinateTransformations(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(32)
        ntests = 100
        self.mjd = 57087.0 - 1000.0 * (self.rng.random_sample(ntests) - 0.5)
        self.tolerance = 1.0e-5

    def test_exceptions(self):
        """
        Test to make sure that methods complain when incorrect data types are passed.
        """
        mjd_float = 52000.0
        mjd2 = np.array([52000.0, 53000.0])
        mjd3 = np.array([53000.0, 53000.0, 54000.0])

        long_float = 1.2
        long_arr = np.array([1.2, 1.4])

        self.assertRaises(RuntimeError, utils.calc_lmst_last, mjd_float, long_arr)
        self.assertRaises(RuntimeError, utils.calc_lmst_last, mjd3, long_arr)
        self.assertRaises(RuntimeError, utils.calc_lmst_last, list(mjd2), long_arr)
        self.assertRaises(RuntimeError, utils.calc_lmst_last, mjd2, list(long_arr))
        self.assertRaises(RuntimeError, utils.calc_lmst_last, mjd_float, long_arr)
        utils.calc_lmst_last(mjd2, long_float)
        utils.calc_lmst_last(mjd_float, long_float)
        utils.calc_lmst_last(int(mjd_float), long_float)
        utils.calc_lmst_last(mjd_float, int(long_float))
        utils.calc_lmst_last(int(mjd_float), int(long_float))
        utils.calc_lmst_last(mjd2, long_arr)

    def test_equation_of_equinoxes(self):
        """
        Test equation of equninoxes calculation
        """

        # test vectorized version
        control = control_equation_of_equinoxes(self.mjd)
        test = utils.equation_of_equinoxes(self.mjd)
        self.assertLess(np.abs(test - control).max(), self.tolerance)

        # test non-vectorized version
        for mm in self.mjd:
            control = control_equation_of_equinoxes(mm)
            test = utils.equation_of_equinoxes(mm)
            self.assertLess(np.abs(test - control), self.tolerance)

    def test_gmst_gast(self):
        """
        Test calculation of Greenwich mean and apparent sidereal times
        """

        control_gmst, control_gast = control_calc_gmst_gast(self.mjd)
        test_gmst, test_gast = utils.calc_gmst_gast(self.mjd)
        self.assertLess(np.abs(test_gmst - control_gmst).max(), self.tolerance)
        self.assertLess(np.abs(test_gast - control_gast).max(), self.tolerance)

        # test non-vectorized version
        for mm in self.mjd:
            control_gmst, control_gast = control_calc_gmst_gast(mm)
            test_gmst, test_gast = utils.calc_gmst_gast(mm)
            self.assertLess(np.abs(test_gmst - control_gmst), self.tolerance)
            self.assertLess(np.abs(test_gast - control_gast), self.tolerance)

    def test_lmst_last(self):
        """
        Test calculation of local mean and apparent sidereal time
        """

        gmst, gast = utils.calc_gmst_gast(self.mjd)
        ll = [1.2, 2.2]

        # test passing a float for longitude and a numpy array for mjd
        for longitude in ll:
            hours = np.degrees(longitude) / 15.0
            if hours > 24.0:
                hours -= 24.0
            control_lmst = gmst + hours
            control_last = gast + hours
            control_lmst %= 24.0
            control_last %= 24.0
            test_lmst, test_last = utils.calc_lmst_last(self.mjd, longitude)
            self.assertLess(np.abs(test_lmst - control_lmst).max(), self.tolerance)
            self.assertLess(np.abs(test_last - control_last).max(), self.tolerance)
            self.assertIsInstance(test_lmst, np.ndarray)
            self.assertIsInstance(test_last, np.ndarray)

        # test passing two floats
        for longitude in ll:
            for mm in self.mjd:
                gmst, gast = utils.calc_gmst_gast(mm)
                hours = np.degrees(longitude) / 15.0
                if hours > 24.0:
                    hours -= 24.0
                control_lmst = gmst + hours
                control_last = gast + hours
                control_lmst %= 24.0
                control_last %= 24.0
                test_lmst, test_last = utils.calc_lmst_last(mm, longitude)
                self.assertLess(np.abs(test_lmst - control_lmst), self.tolerance)
                self.assertLess(np.abs(test_last - control_last), self.tolerance)
                self.assertIsInstance(test_lmst, float)
                self.assertIsInstance(test_last, float)

        # test passing two numpy arrays
        ll = self.rng.random_sample(len(self.mjd)) * 2.0 * np.pi
        test_lmst, test_last = utils.calc_lmst_last(self.mjd, ll)
        self.assertIsInstance(test_lmst, np.ndarray)
        self.assertIsInstance(test_last, np.ndarray)
        for ix, (longitude, mm) in enumerate(zip(ll, self.mjd)):
            control_lmst, control_last = utils.calc_lmst_last(mm, longitude)
            self.assertAlmostEqual(control_lmst, test_lmst[ix], 10)
            self.assertAlmostEqual(control_last, test_last[ix], 10)

    def test_galactic_from_equatorial(self):
        ra = np.zeros((3), dtype=float)
        dec = np.zeros((3), dtype=float)

        ra[0] = 2.549091039839124218e00
        dec[0] = 5.198752733024248895e-01
        ra[1] = 8.693375673649429425e-01
        dec[1] = 1.038086165642298164e00
        ra[2] = 7.740864769302191473e-01
        dec[2] = 2.758053025017753179e-01

        glon, glat = utils._galactic_from_equatorial(ra, dec)

        self.assertIsInstance(glon, np.ndarray)
        self.assertIsInstance(glat, np.ndarray)

        self.assertAlmostEqual(glon[0], 3.452036693523627964e00, 6)
        self.assertAlmostEqual(glat[0], 8.559512505657201897e-01, 6)
        self.assertAlmostEqual(glon[1], 2.455968474619387720e00, 6)
        self.assertAlmostEqual(glat[1], 3.158563770667878468e-02, 6)
        self.assertAlmostEqual(glon[2], 2.829585540991265358e00, 6)
        self.assertAlmostEqual(glat[2], -6.510790587552289788e-01, 6)

        # test passing in floats as args
        for ix, (rr, dd) in enumerate(zip(ra, dec)):
            gl, gb = utils._galactic_from_equatorial(rr, dd)
            self.assertIsInstance(rr, float)
            self.assertIsInstance(dd, float)
            self.assertIsInstance(gl, float)
            self.assertIsInstance(gb, float)
            self.assertAlmostEqual(gl, glon[ix], 10)
            self.assertAlmostEqual(gb, glat[ix], 10)

    def test_equatorial_from_galactic(self):
        lon = np.zeros((3), dtype=float)
        lat = np.zeros((3), dtype=float)

        lon[0] = 3.452036693523627964e00
        lat[0] = 8.559512505657201897e-01
        lon[1] = 2.455968474619387720e00
        lat[1] = 3.158563770667878468e-02
        lon[2] = 2.829585540991265358e00
        lat[2] = -6.510790587552289788e-01

        ra, dec = utils._equatorial_from_galactic(lon, lat)

        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)

        self.assertAlmostEqual(ra[0], 2.549091039839124218e00, 6)
        self.assertAlmostEqual(dec[0], 5.198752733024248895e-01, 6)
        self.assertAlmostEqual(ra[1], 8.693375673649429425e-01, 6)
        self.assertAlmostEqual(dec[1], 1.038086165642298164e00, 6)
        self.assertAlmostEqual(ra[2], 7.740864769302191473e-01, 6)
        self.assertAlmostEqual(dec[2], 2.758053025017753179e-01, 6)

        # test passing in floats as args
        for ix, (ll, bb) in enumerate(zip(lon, lat)):
            rr, dd = utils._equatorial_from_galactic(ll, bb)
            self.assertIsInstance(ll, float)
            self.assertIsInstance(bb, float)
            self.assertIsInstance(rr, float)
            self.assertIsInstance(dd, float)
            self.assertAlmostEqual(rr, ra[ix], 10)
            self.assertAlmostEqual(dd, dec[ix], 10)

    def test_spherical_from_cartesian(self):
        """
        Note that xyz[i][j] is the ith component of the jth vector

        Each column of xyz is a vector
        """
        nsamples = 10
        radius = self.rng.random_sample(nsamples) * 10.0
        theta = self.rng.random_sample(nsamples) * np.pi - 0.5 * np.pi
        phi = self.rng.random_sample(nsamples) * 2.0 * np.pi

        points = []
        for ix in range(nsamples):
            vv = [
                radius[ix] * np.cos(theta[ix]) * np.cos(phi[ix]),
                radius[ix] * np.cos(theta[ix]) * np.sin(phi[ix]),
                radius[ix] * np.sin(theta[ix]),
            ]

            points.append(vv)

        points = np.array(points)
        lon, lat = utils.spherical_from_cartesian(points)
        for ix in range(nsamples):
            self.assertAlmostEqual(np.cos(lon[ix]), np.cos(phi[ix]), 5)
            self.assertAlmostEqual(np.sin(lon[ix]), np.sin(phi[ix]), 5)
            self.assertAlmostEqual(np.cos(lat[ix]), np.cos(theta[ix]), 5)
            self.assertAlmostEqual(np.sin(lat[ix]), np.sin(theta[ix]), 5)

        # test passing in the points one at a time
        for pp, th, ph in zip(points, theta, phi):
            lon, lat = utils.spherical_from_cartesian(pp)
            self.assertAlmostEqual(np.cos(lon), np.cos(ph), 5)
            self.assertAlmostEqual(np.sin(lon), np.sin(ph), 5)
            self.assertAlmostEqual(np.cos(lat), np.cos(th), 5)
            self.assertAlmostEqual(np.sin(lat), np.sin(th), 5)

        # test ra_dec_from_xyz <-> spherical_from_cartesian
        np.testing.assert_array_equal(
            utils.spherical_from_cartesian(points),
            utils._ra_dec_from_xyz(points[:, 0], points[:, 1], points[:, 2]),
        )

        # now, test passing one at a time
        for pp in points:
            np.testing.assert_array_equal(
                utils.spherical_from_cartesian(pp),
                utils._ra_dec_from_xyz(pp[0], pp[1], pp[2]),
            )

    def test_cartesian_from_spherical(self):
        nsamples = 10
        theta = self.rng.random_sample(nsamples) * np.pi - 0.5 * np.pi
        phi = self.rng.random_sample(nsamples) * 2.0 * np.pi

        points = []
        for ix in range(nsamples):
            vv = [
                np.cos(theta[ix]) * np.cos(phi[ix]),
                np.cos(theta[ix]) * np.sin(phi[ix]),
                np.sin(theta[ix]),
            ]

            points.append(vv)

        points = np.array(points)
        lon, lat = utils.spherical_from_cartesian(points)
        out_points = utils.cartesian_from_spherical(lon, lat)

        for pp, oo in zip(points, out_points):
            np.testing.assert_array_almost_equal(pp, oo, decimal=6)

        # test passing in arguments as floats
        for ix, (ll, bb) in enumerate(zip(lon, lat)):
            xyz = utils.cartesian_from_spherical(ll, bb)
            self.assertIsInstance(xyz[0], float)
            self.assertIsInstance(xyz[1], float)
            self.assertIsInstance(xyz[2], float)
            self.assertAlmostEqual(xyz[0], out_points[ix][0], 12)
            self.assertAlmostEqual(xyz[1], out_points[ix][1], 12)
            self.assertAlmostEqual(xyz[2], out_points[ix][2], 12)

        # test _xyz_from_ra_dec <-> testCartesianFromSpherical
        np.testing.assert_array_equal(
            utils.cartesian_from_spherical(lon, lat),
            utils._xyz_from_ra_dec(lon, lat).transpose(),
        )

    def test_haversine(self):
        arg1 = 7.853981633974482790e-01
        arg2 = 3.769911184307751517e-01
        arg3 = 5.026548245743668986e00
        arg4 = -6.283185307179586232e-01

        output = utils.haversine(arg1, arg2, arg3, arg4)

        self.assertAlmostEqual(output, 2.162615946398791955e00, 10)

    def test_rotation_matrix_from_vectors(self):
        v1 = np.zeros((3), dtype=float)
        v2 = np.zeros((3), dtype=float)
        v3 = np.zeros((3), dtype=float)

        v1[0] = -3.044619987218469825e-01
        v2[0] = 5.982190522311925385e-01
        v1[1] = -5.473550908956383854e-01
        v2[1] = -5.573565912346714057e-01
        v1[2] = 7.795545496018386755e-01
        v2[2] = -5.757495946632366079e-01

        output = utils.rotation_matrix_from_vectors(v1, v2)

        for i in range(3):
            for j in range(3):
                v3[i] += output[i][j] * v1[j]

        for i in range(3):
            self.assertAlmostEqual(v3[i], v2[i], 7)

        v1 = np.array([1.0, 1.0, 1.0])
        self.assertRaises(RuntimeError, utils.rotation_matrix_from_vectors, v1, v2)
        self.assertRaises(RuntimeError, utils.rotation_matrix_from_vectors, v2, v1)


class RotationTestCase(unittest.TestCase):
    def setUp(self):
        self.sqrt2o2 = np.sqrt(2.0) / 2.0
        self.x_vec = np.array([1.0, 0.0, 0.0])
        self.y_vec = np.array([0.0, 1.0, 0.0])
        self.z_vec = np.array([0.0, 0.0, 1.0])
        self.px_py_vec = np.array([self.sqrt2o2, self.sqrt2o2, 0.0])
        self.px_ny_vec = np.array([self.sqrt2o2, -self.sqrt2o2, 0.0])
        self.nx_py_vec = np.array([-self.sqrt2o2, self.sqrt2o2, 0.0])
        self.nx_ny_vec = np.array([-self.sqrt2o2, -self.sqrt2o2, 0.0])

        self.px_pz_vec = np.array([self.sqrt2o2, 0.0, self.sqrt2o2])
        self.px_nz_vec = np.array([self.sqrt2o2, 0.0, -self.sqrt2o2])
        self.nx_pz_vec = np.array([-self.sqrt2o2, 0.0, self.sqrt2o2])
        self.nx_nz_vec = np.array([-self.sqrt2o2, 0.0, -self.sqrt2o2])

        self.py_pz_vec = np.array([0.0, self.sqrt2o2, self.sqrt2o2])
        self.ny_pz_vec = np.array([0.0, -self.sqrt2o2, self.sqrt2o2])
        self.py_nz_vec = np.array([0.0, self.sqrt2o2, -self.sqrt2o2])
        self.ny_nz_vec = np.array([0.0, -self.sqrt2o2, -self.sqrt2o2])

    def test_rot_z(self):
        out = utils.rot_about_z(self.x_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, self.y_vec, decimal=10)
        out = utils.rot_about_z(self.x_vec, -0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.y_vec, decimal=10)
        out = utils.rot_about_z(self.x_vec, np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.x_vec, decimal=10)
        out = utils.rot_about_z(self.x_vec, 0.25 * np.pi)
        np.testing.assert_array_almost_equal(out, self.px_py_vec, decimal=10)
        out = utils.rot_about_z(self.x_vec, -0.25 * np.pi)
        np.testing.assert_array_almost_equal(out, self.px_ny_vec, decimal=10)
        out = utils.rot_about_z(self.x_vec, 0.75 * np.pi)
        np.testing.assert_array_almost_equal(out, self.nx_py_vec, decimal=10)
        out = utils.rot_about_z(self.y_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.x_vec, decimal=10)
        out = utils.rot_about_z(self.px_py_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, self.nx_py_vec, decimal=10)
        out = utils.rot_about_z(self.px_py_vec, 0.75 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.x_vec, decimal=10)

        out = utils.rot_about_z(
            np.array([self.x_vec, self.y_vec, self.nx_py_vec, self.nx_ny_vec]),
            -0.25 * np.pi,
        )
        np.testing.assert_array_almost_equal(out[0], self.px_ny_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[1], self.px_py_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[2], self.y_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[3], -1.0 * self.x_vec, decimal=10)

    def test_rot_y(self):
        out = utils.rot_about_y(self.x_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.z_vec, decimal=10)
        out = utils.rot_about_y(self.z_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, self.x_vec, decimal=10)
        out = utils.rot_about_y(self.px_pz_vec, 0.75 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.z_vec, decimal=10)
        out = utils.rot_about_y(self.px_pz_vec, -0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, self.nx_pz_vec, decimal=10)

        out = utils.rot_about_y(
            np.array([self.px_pz_vec, self.nx_pz_vec, self.z_vec, self.x_vec]),
            -0.75 * np.pi,
        )

        np.testing.assert_array_almost_equal(out[0], -1.0 * self.x_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[1], -1.0 * self.z_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[2], self.nx_nz_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[3], self.nx_pz_vec, decimal=10)

    def test_rot_x(self):
        out = utils.rot_about_x(self.y_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, self.z_vec, decimal=10)
        out = utils.rot_about_x(self.y_vec, 0.75 * np.pi)
        np.testing.assert_array_almost_equal(out, self.ny_pz_vec, decimal=10)
        out = utils.rot_about_x(self.z_vec, 0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.y_vec, decimal=10)
        out = utils.rot_about_x(self.z_vec, -0.25 * np.pi)
        np.testing.assert_array_almost_equal(out, self.py_pz_vec, decimal=10)
        out = utils.rot_about_x(self.py_nz_vec, -0.5 * np.pi)
        np.testing.assert_array_almost_equal(out, self.ny_nz_vec, decimal=10)
        out = utils.rot_about_x(self.ny_nz_vec, 0.25 * np.pi)
        np.testing.assert_array_almost_equal(out, -1.0 * self.z_vec, decimal=10)

        out = utils.rot_about_x(
            np.array([self.z_vec, self.py_pz_vec, self.ny_pz_vec, self.y_vec]),
            0.25 * np.pi,
        )

        np.testing.assert_array_almost_equal(out[0], self.ny_pz_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[1], self.z_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[2], -1.0 * self.y_vec, decimal=10)
        np.testing.assert_array_almost_equal(out[3], self.py_pz_vec, decimal=10)


if __name__ == "__main__":
    unittest.main()
