__all__ = ("CloudData", "ConstantCloudData")

import os
import sqlite3
from dataclasses import dataclass

import numpy as np
from astropy.time import Time

from rubin_sim.data import get_data_dir


class CloudData:
    """Handle the cloud information.

    This class deals with the cloud information that was previously produced for
    OpSim version 3.

    Parameters
    ----------
    start_time : astropy.time.Time
        The time of the start of the simulation.
        The cloud database will be assumed to start on Jan 01 of the same year.
    cloud_db : str, optional
        The full path name for the cloud database. Default None,
        which will use the database stored in the module (site_models/clouds_ctio_1975_2022.db).
    offset_year : float, optional
        Offset into the cloud database by 'offset_year' years. Default 0.
    scale : float (1e6)
        Enforce machine precision for cross-platform repeatability by scaling and rounding date values.
    """

    def __init__(self, start_time, cloud_db=None, offset_year=0, scale=1e6):
        self.cloud_db = cloud_db
        if self.cloud_db is None:
            self.cloud_db = os.path.join(get_data_dir(), "site_models", "clouds_ctio_1975_2022.db")

        # Cloud database starts in Jan 01 of the year of the start of the simulation.
        year_start = start_time.datetime.year + offset_year
        self.start_time = Time("%d-01-01" % year_start, format="isot", scale="tai")

        self.cloud_dates = None
        self.cloud_values = None
        self.scale = scale
        self.read_data()

    def __call__(self, time):
        """Get the cloud for the specified time.

        Parameters
        ----------
        time : astropy.time.Time
            Time in the simulation for which to find the current cloud coverage.
            The difference between this time and the start_time, plus the offset,
            will be used to query the cloud database for the 'current' conditions.

        Returns
        -------
        float
            The fraction of the sky that is cloudy (measured in steps of 8ths) closest to the specified time.
        """
        delta_time = (time - self.start_time).sec
        dbdate = delta_time % self.time_range + self.min_time
        if self.scale is not None:
            dbdate = np.round(dbdate * self.scale).astype(int)
        idx = np.searchsorted(self.cloud_dates, dbdate)
        # searchsorted ensures that left < date < right
        # but we need to know if date is closer to left or to right
        left = self.cloud_dates[idx - 1]
        right = self.cloud_dates[idx]
        # If we are only doing one time
        if np.size(left) == 1:
            if dbdate - left < right - dbdate:
                idx -= 1
        # If we have an array of times
        else:
            d1 = dbdate - left
            d2 = right - dbdate
            to_sub = np.where(d1 < d2)
            idx[to_sub] -= 1

        return self.cloud_values[idx]

    def read_data(self):
        """Read the cloud data from disk.

        The default behavior is to use the module stored database. However, an
        alternate database file can be provided. The alternate database file needs to have a
        table called *Cloud* with the following columns:

        cloudId
            int : A unique index for each cloud entry.
        c_date
            int : The time (units=seconds) since the start of the simulation for the cloud observation.
        cloud
            float : The cloud coverage (in steps of 8ths) of the sky.
        """
        with sqlite3.connect(self.cloud_db) as conn:
            cur = conn.cursor()
            query = "select c_date, cloud from Cloud order by c_date;"
            cur.execute(query)
            results = np.array(cur.fetchall())
            self.cloud_dates = np.hsplit(results, 2)[0].flatten()
            self.cloud_values = np.hsplit(results, 2)[1].flatten()
            cur.close()
        # Make sure seeing dates are ordered appropriately (monotonically increasing).
        ordidx = self.cloud_dates.argsort()
        self.cloud_dates = self.cloud_dates[ordidx]
        # Record this information, in case the cloud database does not start at t=0.
        self.min_time = self.cloud_dates[0]
        self.max_time = self.cloud_dates[-1]
        self.time_range = self.max_time - self.min_time
        if self.scale is not None:
            self.cloud_dates = np.round(self.cloud_dates * self.scale).astype(int)
        self.cloud_values = self.cloud_values[ordidx]


@dataclass
class ConstantCloudData:
    """Generate constant cloud information.

    Parameters
    ----------
    cloud_fraction : `float`
        The fraction of the sky that is cloudy.
    """

    cloud_fraction: float = 0.0

    def __call__(self, time):
        """Get the cloud for the specified time.

        Parameters
        ----------
        time : `astropy.time.Time`
            In principle, the time in the simulation for which to find the
            current cloud coverage. In practice, this argument is ignored.

        Returns
        -------
        cloud_faction : `float`
            The fraction of the sky that is cloudy.
        """
        return self.cloud_fraction
