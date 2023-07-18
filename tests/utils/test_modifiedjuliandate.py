import copy
import inspect
import os
import unittest
import warnings

import astropy
import numpy as np

import rubin_sim
from rubin_sim.data import get_data_dir
from rubin_sim.utils import ModifiedJulianDate, Utctout1Warning


class MjdTest(unittest.TestCase):
    """
    This unit test TestCase will just verify that the contents
    of ModifiedJulianDate agree with results generated 'by hand'.
    The 'by hand' transformations will have been tested by
    testTimeTransformations.py
    """

    long_message = True

    def test_tai_from_utc(self):
        """
        Load a table of utc vs. TAI (as JD) generated directly
        with ERFA.  Verify our ModifiedJulianDate wrapper against
        this data.  This is mostly so that we can catch any major
        API changes in astropy.
        """

        file_name = os.path.join(get_data_dir(), "tests")
        file_name = os.path.join(file_name, "testData", "utc_tai_comparison_data.txt")

        dtype = np.dtype([("utc", float), ("tai", float)])
        data = np.genfromtxt(file_name, dtype=dtype)

        msg = (
            "\n\nIt is possible you are using an out-of-date astropy.\n"
            + "Try running 'conda update astropy' and restarting the build."
        )

        for uu, tt in zip(data["utc"] - 2400000.5, data["tai"] - 2400000.5):
            mjd = ModifiedJulianDate(utc=uu)
            dd_sec = np.abs(mjd.TAI - tt) * 86400.0
            self.assertLess(dd_sec, 5.0e-5, msg=msg)
            self.assertAlmostEqual(mjd.utc, uu, 15, msg=msg)
            mjd = ModifiedJulianDate(TAI=tt)
            dd_sec = np.abs(mjd.utc - uu) * 86400.0
            self.assertLess(dd_sec, 5.0e-5, msg=msg)
            self.assertAlmostEqual(mjd.TAI, tt, 15, msg=msg)

    def test_tt(self):
        """
        Verify that Terrestrial Time is TAI + 32.184 seconds
        as in equation 2.223-6 of

        Explanatory Supplement to the Astrnomical Almanac
        ed. Seidelmann, Kenneth P.
        1992, University Science Books

        Mostly, this test exists to catch any major API
        changes in astropy.time
        """

        rng = np.random.RandomState(115)
        tai_list = rng.random_sample(1000) * 7000.0 + 50000.0
        for tai in tai_list:
            mjd = ModifiedJulianDate(TAI=tai)
            self.assertAlmostEqual(mjd.TT, tai + 32.184 / 86400.0, 15)

    def test_tdb(self):
        """
        Verify that TDB is within a few tens of microseconds of the value given
        by the approximation given by equation 2.222-1 of

        Explanatory Supplement to the Astrnomical Almanac
        ed. Seidelmann, Kenneth P.
        1992, University Science Books

        Mostly, this test exists to catch any major API
        changes in astropy.time
        """

        rng = np.random.RandomState(117)
        tai_list = rng.random_sample(1000) * 10000.0 + 46000.0
        for tai in tai_list:
            mjd = ModifiedJulianDate(TAI=tai)
            g = np.radians(357.53 + 0.9856003 * (np.round(tai - 51544.5)))
            tdb_test = mjd.TT + (0.001658 * np.sin(g) + 0.000014 * np.sin(2.0 * g)) / 86400.0
            dt = np.abs(tdb_test - mjd.TDB) * 8.64 * 1.0e10  # convert to microseconds
            self.assertLess(dt, 50)

    def test_dut1(self):
        """
        Test that ut1 is within 0.9 seconds of utc and that dut1 is equal
        to ut1-utc to within a microsecond.

        (Because calculating ut1-utc requires loading a lookup
        table, we will just do this somewhat gross unit test to
        make sure that the astropy.time API doesn't change out
        from under us in some weird way... for instance, returning
        dut in units of days rather than seconds, etc.)
        """

        rng = np.random.RandomState(117)

        utc_list = rng.random_sample(1000) * 10000.0 + 43000.0
        for utc in utc_list:
            mjd = ModifiedJulianDate(utc=utc)

            # first, test the self-consistency of ModifiedJulianData.dut1
            # and ModifiedJulianData.ut1-ModifiedJulianData.utc
            #
            # this only works for days on which a leap second is not applied
            dt = (mjd.ut1 - mjd.utc) * 86400.0

            self.assertLess(np.abs(dt - mjd.dut1), 1.0e-5, msg="failed on utc: %.12f" % mjd.utc)

            self.assertLess(np.abs(mjd.dut1), 0.9)

    def test_dut1_future(self):
        """
        Test that ut1 is within 0.9 seconds of utc and that dut1 is equal
        to ut1-utc to within a microsecond.  Consider times far in the future.

        (Because calculating ut1-utc requires loading a lookup
        table, we will just do this somewhat gross unit test to
        make sure that the astropy.time API doesn't change out
        from under us in some weird way... for instance, returning
        dut in units of days rather than seconds, etc.)
        """

        rng = np.random.RandomState(117)

        utc_list = rng.random_sample(1000) * 10000.0 + 63000.0
        for utc in utc_list:
            mjd = ModifiedJulianDate(utc=utc)

            # first, test the self-consistency of ModifiedJulianData.dut1
            # and ModifiedJulianData.ut1-ModifiedJulianData.utc
            #
            # this only works for days on which a leap second is not applied
            dt = (mjd.ut1 - mjd.utc) * 86400.0

            self.assertLess(np.abs(dt - mjd.dut1), 1.0e-5, msg="failed on utc %.12f" % mjd.utc)

            self.assertLess(np.abs(mjd.dut1), 0.9)

    def test_eq(self):
        mjd1 = ModifiedJulianDate(TAI=43000.0)
        mjd2 = ModifiedJulianDate(TAI=43000.0)
        self.assertEqual(mjd1, mjd2)
        self.assertTrue(mjd1 == mjd2)
        self.assertFalse(mjd1 != mjd2)
        mjd3 = ModifiedJulianDate(TAI=43000.01)
        self.assertNotEqual(mjd1, mjd3)
        self.assertFalse(mjd1 == mjd3)
        self.assertTrue(mjd1 != mjd3)

    def test_deepcopy(self):
        # make sure that deepcopy() creates identical
        # ModifiedJulianDates with different memory addresses
        mjd1 = ModifiedJulianDate(TAI=43590.0)
        mjd1.dut1
        deep_mjd2 = copy.deepcopy(mjd1)
        self.assertEqual(mjd1, deep_mjd2)
        self.assertNotEqual(mjd1.__repr__(), deep_mjd2.__repr__())
        self.assertEqual(mjd1.TAI, deep_mjd2.TAI)
        self.assertEqual(mjd1.dut1, deep_mjd2.dut1)
        equiv_mjd2 = mjd1
        self.assertEqual(mjd1, equiv_mjd2)
        self.assertEqual(mjd1.__repr__(), equiv_mjd2.__repr__())

        mjd1 = ModifiedJulianDate(utc=43590.0)
        mjd1.dut1
        deep_mjd2 = copy.deepcopy(mjd1)
        self.assertEqual(mjd1, deep_mjd2)
        self.assertEqual(mjd1.utc, deep_mjd2.utc)
        self.assertEqual(mjd1.dut1, deep_mjd2.dut1)
        self.assertNotEqual(mjd1.__repr__(), deep_mjd2.__repr__())
        equiv_mjd2 = mjd1
        self.assertEqual(mjd1, equiv_mjd2)
        self.assertEqual(mjd1.__repr__(), equiv_mjd2.__repr__())

        # make sure that deepcopy() still works, even if you have called
        # all of the original ModifiedJulianDate's properties
        mjd1 = ModifiedJulianDate(TAI=42590.0)
        mjd1.utc
        mjd1.dut1
        mjd1.ut1
        mjd1.TT
        mjd1.TDB
        mjd2 = copy.deepcopy(mjd1)
        self.assertEqual(mjd1.TAI, mjd2.TAI)
        self.assertEqual(mjd1.utc, mjd2.utc)
        self.assertEqual(mjd1.dut1, mjd2.dut1)
        self.assertEqual(mjd1.ut1, mjd2.ut1)
        self.assertEqual(mjd1.TT, mjd2.TT)
        self.assertEqual(mjd1.TDB, mjd2.TDB)
        self.assertEqual(mjd1, mjd2)
        self.assertNotEqual(mjd1.__repr__(), mjd2.__repr__())

    @unittest.skipIf(
        astropy.__version__ >= "1.2",
        "astropy 1.2 handles cases of dates too far in the future "
        "on its own in a graceful manner. Our warning classes are not needed",
    )
    def test_warnings(self):
        """
        Test that warnings raised when trying to interpolate ut1-utc
        for utc too far in the future are of the type Utctout1Warning
        """
        with warnings.catch_warnings(record=True) as w_list:
            mjd = ModifiedJulianDate(1000000.0)
            # clear the warning registry, in case a previous test raised the warnings
            # we are looking for
            if "__warningregistry__" in mjd._warn_utc_out_of_bounds.__globals__:
                mjd._warn_utc_out_of_bounds.__globals__["__warningregistry__"].clear()
            warnings.simplefilter("always")
            # Trigger a warning.
            # Note that this may also trigger astropy warnings,
            # depending on the order in which tests are run.
            mjd.ut1
        expected_mjd_warnings = 1
        mjd_warnings = 0
        for w in w_list:
            # Count the number of warnings and test we can filter by category.
            if w.category == Utctout1Warning:
                mjd_warnings += 1
                # Test that the string "ModifiedJulianDate.ut1" actually showed up in the message.
                # This indicates what method the warning occured from (ut1 vs dut).
                self.assertIn("ModifiedJulianDate.ut1", str(w.message))
        self.assertEqual(
            expected_mjd_warnings,
            mjd_warnings,
            msg="ut1 did not emit a Utctout1Warning",
        )

        expected_mjd_warnings = 1
        mjd_warnings = 0
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            mjd = ModifiedJulianDate(1000000.0)
            mjd.dut1
        for w in w_list:
            if w.category == Utctout1Warning:
                mjd_warnings += 1
                self.assertIn("ModifiedJulianDate.dut1", str(w.message))
        self.assertEqual(
            expected_mjd_warnings,
            mjd_warnings,
            msg="dut1 did not emit a Utctout1Warning",
        )

    def test_force_values(self):
        """
        Test that we can force the properties of a ModifiedJulianDate to have
        specific values
        """
        tt = ModifiedJulianDate(TAI=59580.0)
        values = np.arange(6)
        tt._force_values(values)
        self.assertEqual(tt.TAI, 0.0)
        self.assertEqual(tt.utc, 1.0)
        self.assertEqual(tt.TT, 2.0)
        self.assertEqual(tt.TDB, 3.0)
        self.assertEqual(tt.ut1, 4.0)
        self.assertEqual(tt.dut1, 5.0)

        tt = ModifiedJulianDate(utc=59580.0)
        values = 2.0 * np.arange(6)
        tt._force_values(values)
        self.assertEqual(tt.TAI, 0.0)
        self.assertEqual(tt.utc, 2.0)
        self.assertEqual(tt.TT, 4.0)
        self.assertEqual(tt.TDB, 6.0)
        self.assertEqual(tt.ut1, 8.0)
        self.assertEqual(tt.dut1, 10.0)

    def test_list(self):
        """
        Test that ModifiedJulianDate.get_list() gets results that are consistent
        with creating a list of ModifiedJulianDates by hand.
        """

        rng = np.random.RandomState(88)
        tol = 10  # decimal place tolerance

        tai_list = 40000.0 + 10000.0 * rng.random_sample(20)
        tai_list = np.append(tai_list, 59580.0 + 10000.0 * rng.random_sample(20))
        mjd_list = ModifiedJulianDate.get_list(TAI=tai_list)
        for tai, mjd in zip(tai_list, mjd_list):
            msg = "Offending TAI: %f" % tai
            control = ModifiedJulianDate(TAI=tai)
            self.assertAlmostEqual(mjd.TAI, tai, 11, msg=msg)
            self.assertAlmostEqual(mjd.TAI, control.TAI, tol, msg=msg)
            self.assertAlmostEqual(mjd.utc, control.utc, tol, msg=msg)
            self.assertAlmostEqual(mjd.ut1, control.ut1, tol, msg=msg)
            self.assertAlmostEqual(mjd.TT, control.TT, tol, msg=msg)
            self.assertAlmostEqual(mjd.TDB, control.TDB, tol, msg=msg)
            self.assertAlmostEqual(mjd.dut1, control.dut1, tol, msg=msg)

        utc_list = 40000.0 + 10000.0 * rng.random_sample(20)
        utc_list = np.append(utc_list, 59580.0 + 10000.0 * rng.random_sample(20))
        mjd_list = ModifiedJulianDate.get_list(utc=utc_list)
        for utc, mjd in zip(utc_list, mjd_list):
            msg = "Offending utc: %f" % utc
            control = ModifiedJulianDate(utc=utc)
            self.assertAlmostEqual(mjd.utc, utc, tol, msg=msg)
            self.assertAlmostEqual(mjd.TAI, control.TAI, tol, msg=msg)
            self.assertAlmostEqual(mjd.utc, control.utc, tol, msg=msg)
            self.assertAlmostEqual(mjd.ut1, control.ut1, tol, msg=msg)
            self.assertAlmostEqual(mjd.TT, control.TT, tol, msg=msg)
            self.assertAlmostEqual(mjd.TDB, control.TDB, tol, msg=msg)
            self.assertAlmostEqual(mjd.dut1, control.dut1, tol, msg=msg)

        # Now test the case where we only have dates in the future (this
        # is an edge case since good_dexes in ModifiedJulianDate._get_ut1_from_utc
        # will have len = 0
        tai_list = 60000.0 + 10000.0 * rng.random_sample(20)
        mjd_list = ModifiedJulianDate.get_list(TAI=tai_list)
        for tai, mjd in zip(tai_list, mjd_list):
            msg = "Offending TAI: %f" % tai
            control = ModifiedJulianDate(TAI=tai)
            self.assertAlmostEqual(mjd.TAI, tai, 11, msg=msg)
            self.assertAlmostEqual(mjd.TAI, control.TAI, tol, msg=msg)
            self.assertAlmostEqual(mjd.utc, control.utc, tol, msg=msg)
            self.assertAlmostEqual(mjd.ut1, control.ut1, tol, msg=msg)
            self.assertAlmostEqual(mjd.TT, control.TT, tol, msg=msg)
            self.assertAlmostEqual(mjd.TDB, control.TDB, tol, msg=msg)
            self.assertAlmostEqual(mjd.dut1, control.dut1, tol, msg=msg)


if __name__ == "__main__":
    unittest.main()
