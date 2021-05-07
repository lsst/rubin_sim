import warnings
import numpy as np
import copy

from astropy.time import Time
from astropy.utils.iers.iers import IERSRangeError

__all__ = ["ModifiedJulianDate", "MJDWarning", "UTCtoUT1Warning"]


# Filter out ERFA's complaints that we are simulating dates which
# are in the future
warnings.filterwarnings("ignore",
                        message='.*taiutc.*dubious.year.*')


class MJDWarning(Warning):
    """
    A sub-class of Warning.  All of the warnings raised by ModifiedJulianDate
    will be of this class (or its sub-classes), so that users can filter them
    out by creating a simple filter targeted at category=MJDWarning.
    """
    pass


class UTCtoUT1Warning(MJDWarning):
    """
    A sub-class of MJDWarning meant for use when astropy.Time cannot interpolate
    UT1-UTC as a function of UTC because UTC is out of bounds of the data.
    This class exists so that users can filter these warnings out by creating
    a simple filter targeted at category=UTCtoUT1Warning.
    """
    pass


class ModifiedJulianDate(object):

    @classmethod
    def _get_ut1_from_utc(cls, UTC):
        """
        Take a numpy array of UTC values and return a numpy array of UT1 and dut1 values
        """

        time_list = Time(UTC, scale='utc', format='mjd')

        try:
            dut1_out = time_list.delta_ut1_utc
            ut1_out = time_list.ut1.mjd
        except IERSRangeError:
            ut1_out = np.copy(UTC)
            dut1_out = np.zeros(len(UTC))
            warnings.warn("ModifiedJulianData.get_list() was given date values that are outside "
                          "astropy's range of interpolation for converting from UTC to UT1. "
                          "We will treat UT1=UTC for those dates, lacking a better alternative.",
                          category=UTCtoUT1Warning)
            from astropy.utils.iers import TIME_BEFORE_IERS_RANGE, TIME_BEYOND_IERS_RANGE
            dut1_test, status = time_list.get_delta_ut1_utc(return_status=True)
            good_dexes = np.where(np.logical_and(status != TIME_BEFORE_IERS_RANGE,
                                                 status != TIME_BEYOND_IERS_RANGE))

            if len(good_dexes[0]) > 0:
                time_good = Time(UTC[good_dexes], scale='utc', format='mjd')
                dut1_good = time_good.delta_ut1_utc
                ut1_good = time_good.ut1.mjd

                ut1_out[good_dexes] = ut1_good
                dut1_out[good_dexes] = dut1_good

        return ut1_out, dut1_out

    @classmethod
    def get_list(cls, TAI=None, UTC=None):
        """
        Instantiate a list of ModifiedJulianDates from a numpy array of either TAI
        or UTC values.

        @param[in] TAI (optional) a numpy array of MJD' in TAI

        @param[in] UTC (optional) a numpy array of MJDs in UTC

        @param[out] a list of ModifiedJulianDate instantiations with all of their
        properties already set (so the code does not waste time converting from TAI
        to TT, TDB, etc. when those time scales are called for).
        """

        if TAI is None and UTC is None:
            return None

        if TAI is not None and UTC is not None:
            raise RuntimeError("You should not specify both TAI and UTC in ModifiedJulianDate.get_list()")

        if TAI is not None:
            time_list = Time(TAI, scale='tai', format='mjd')
            tai_list = TAI
            utc_list = time_list.utc.mjd
        elif UTC is not None:
            time_list = Time(UTC, scale='utc', format='mjd')
            utc_list = UTC
            tai_list = time_list.tai.mjd

        tt_list = time_list.tt.mjd
        tdb_list = time_list.tdb.mjd

        ut1_list, dut1_list = cls._get_ut1_from_utc(utc_list)

        values = np.array([tai_list, utc_list, tt_list, tdb_list,
                           ut1_list, dut1_list]).transpose()

        output = []
        for vv in values:
            mjd = ModifiedJulianDate(TAI=40000.0)
            mjd._force_values(vv)
            output.append(mjd)

        return output

    def __init__(self, TAI=None, UTC=None):
        """
        Must specify either:

        @param [in] TAI = the International Atomic Time as an MJD

        or

        @param [in] UTC = Universal Coordinate Time as an MJD
        """

        if TAI is None and UTC is None:
            raise RuntimeError("You must specify either TAI or UTC to "
                               "instantiate ModifiedJulianDate")

        if TAI is not None:
            self._time = Time(TAI, scale='tai', format='mjd')
            self._tai = TAI
            self._utc = None
            self._initialized_with = 'TAI'
        else:
            self._time = Time(UTC, scale='utc', format='mjd')
            self._utc = UTC
            self._tai = None
            self._initialized_with = 'UTC'

        self._tt = None
        self._tdb = None
        self._ut1 = None
        self._dut1 = None

    def _force_values(self, values):
        """
        Force the properties of this ModifiedJulianDate to have specific values.

        values is a list of [TAI, UTC, TT, TDB, UT1, UT1-UTC] values.

        This method exists so that, when instantiating lists of ModifiedJulianDates,
        we can use astropy.time.Time's vectorized methods to quickly perform many
        conversions at once.  Users should not try to use this method by hand.
        """
        self._tai = values[0]
        self._utc = values[1]
        self._tt = values[2]
        self._tdb = values[3]
        self._ut1 = values[4]
        self._dut1 = values[5]

    def __eq__(self, other):
        return self._time == other._time

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        if self._initialized_with == 'TAI':
            new_mjd = ModifiedJulianDate(TAI=self.TAI)
        else:
            new_mjd = ModifiedJulianDate(UTC=self.UTC)

        new_mjd._tai = copy.deepcopy(self._tai, memo)
        new_mjd._utc = copy.deepcopy(self._utc, memo)
        new_mjd._tt = copy.deepcopy(self._tt, memo)
        new_mjd._tdb = copy.deepcopy(self._tdb, memo)
        new_mjd._ut1 = copy.deepcopy(self._ut1, memo)
        new_mjd._dut1 = copy.deepcopy(self._dut1, memo)

        return new_mjd

    def _warn_utc_out_of_bounds(self, method_name):
        """
        Raise a standard warning if UTC is outside of the range that can
        be interpolated on the IERS tables.

        method_name is the name of the method that caused this warning.
        """
        warnings.warn("UTC is outside of IERS table for UT1-UTC.\n"
                      "Returning UT1 = UTC for lack of a better idea\n"
                      "This warning was caused by calling ModifiedJulianDate.%s\n" % method_name,
                      category=UTCtoUT1Warning)

    @property
    def TAI(self):
        """
        International Atomic Time as an MJD
        """
        if self._tai is None:
            self._tai = self._time.tai.mjd

        return self._tai

    @property
    def UTC(self):
        """
        Universal Coordinate Time as an MJD
        """
        if self._utc is None:
            self._utc = self._time.utc.mjd

        return self._utc

    @property
    def UT1(self):
        """
        Universal Time as an MJD
        """
        if self._ut1 is None:
            try:
                self._ut1 = self._time.ut1.mjd
            except IERSRangeError:
                self._warn_utc_out_of_bounds('UT1')
                self._ut1 = self.UTC

        return self._ut1

    @property
    def dut1(self):
        """
        UT1-UTC in seconds
        """

        if self._dut1 is None:
            try:
                self._dut1 = self._time.delta_ut1_utc
            except IERSRangeError:
                self._warn_utc_out_of_bounds('dut1')
                self._dut1 = 0.0

        return self._dut1

    @property
    def TT(self):
        """
        Terrestrial Time (aka Terrestrial Dynamical Time) as an MJD
        """
        if self._tt is None:
            self._tt = self._time.tt.mjd

        return self._tt

    @property
    def TDB(self):
        """
        Barycentric Dynamical Time as an MJD
        """
        if self._tdb is None:
            self._tdb = self._time.tdb.mjd

        return self._tdb
