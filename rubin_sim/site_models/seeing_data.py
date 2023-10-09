__all__ = ("SeeingData", "ConstantSeeingData")

import os
import sqlite3
from dataclasses import dataclass

import numpy as np
from astropy.time import Time

from rubin_sim.data import get_data_dir


class SeeingData:
    """Read the seeing data from disk and return appropriate FWHM_500 value at a given time.
    This is for use in simulations only. Otherwise data would come from the EFD.

    Parameters
    ----------
    start_time : astropy.time.Time
        The time of the start of the simulation.
        The seeing database will be assumed to start on Jan 01 of the same year.
    seeing_db : str or None, optional
        The name of the seeing database.
        If None (default), this will use the simsee_pachon_58777_13.db file in the 'data' directory
        of this package.
        Other available seeing databases from sims_seeingModel include:
        seeing.db (the original, less-variable, 3 year seeing database)
        simsee_pachon_58777_13.db (the current default, 10 year, seeing database)
        simsee_pachon_58777_16.db (a similar, but slightly offset, 13 year seeing database)
        For more info on simsee_pachon_58777_*, see https://github.com/lsst/sims_seeingModel/issues/2
    offset_year : float, optional
        Offset into the cloud database by 'offset_year' years. Default 0.
    """

    def __init__(self, start_time, seeing_db=None, offset_year=0):
        self.seeing_db = seeing_db
        if self.seeing_db is None:
            self.seeing_db = os.path.join(get_data_dir(), "site_models", "simsee_pachon_58777_13.db")

        # Seeing database starts in Jan 01 of the year of the start of the simulation
        year_start = start_time.datetime.year + offset_year
        self.start_time = Time("%d-01-01" % year_start, format="isot", scale="tai")

        self.seeing_dates = None
        self.seeing_values = None
        self.read_data()

    def __call__(self, time):
        """Get the FWHM_500 value for the specified time.

        Parameters
        ----------
        time : astropy.time.Time
            Time in the simulation for which to find the 'current' zenith seeing values.
            The difference between this time and the start_time, plus the offset,
            will be used to query the seeing database.

        Returns
        -------
        float
            The FWHM_500(") closest to the specified time.
        """
        delta_time = (time - self.start_time).sec
        # Find the date to look for in the time range of the data.
        # Note that data dates should not necessarily start at zero.
        dbdate = delta_time % self.time_range + self.min_time
        idx = np.searchsorted(self.seeing_dates, dbdate)
        # searchsorted ensures that left < date < right
        # but we need to know if date is closer to left or to right
        left = self.seeing_dates[idx - 1]
        right = self.seeing_dates[idx]
        if np.size(idx) == 1:
            if dbdate - left < right - dbdate:
                idx -= 1
        else:
            d1 = dbdate - left
            d2 = right - dbdate
            to_sub = np.where(d1 < d2)
            idx[to_sub] -= 1
        return self.seeing_values[idx]

    def read_data(self):
        """Read the seeing information from disk.

        The default behavior is to use the module stored database. However, an
        alternate database file can be provided. The alternate database file needs to have a
        table called *Seeing* with the following columns:

        seeingId
            int : A unique index for each seeing entry.
        s_date
            int : The time (in seconds) from the start of the simulation, for the seeing observation.
        seeing
            float : The FWHM of the atmospheric PSF (in arcseconds) at zenith.
        """
        with sqlite3.connect(self.seeing_db) as conn:
            cur = conn.cursor()
            query = "select s_date, seeing from Seeing order by s_date;"
            cur.execute(query)
            results = np.array(cur.fetchall())
            self.seeing_dates = np.hsplit(results, 2)[0].flatten()
            self.seeing_values = np.hsplit(results, 2)[1].flatten()
            cur.close()
        # Make sure seeing dates are ordered appropriately (monotonically increasing).
        ordidx = self.seeing_dates.argsort()
        self.seeing_dates = self.seeing_dates[ordidx]
        self.seeing_values = self.seeing_values[ordidx]
        self.min_time = self.seeing_dates[0]
        self.max_time = self.seeing_dates[-1]
        self.time_range = self.max_time - self.min_time

    def config_info(self):
        """Report information about configuration of this data.

        Returns
        -------
        OrderedDict
        """
        config_info = {}
        config_info["Start time for db"] = self.start_time
        config_info["Seeing database"] = self.seeing_db
        return config_info


@dataclass
class ConstantSeeingData:
    fwhm_500: float = 0.7

    def __call__(self, time):
        """A constant FWHM_500 value

        Parameters
        ----------
        time : `astropy.time.Time`
            It principle the time for which the seeing is returned,
            in practice this argumnet is ignored, and included for
            compatibility.

        Returns
        -------
        fwhm_500 : `float`
            The FWHM at 500nm, in arcseconds.
        """
        return self.fwhm_500
