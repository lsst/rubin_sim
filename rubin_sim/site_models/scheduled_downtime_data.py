__all__ = ("ScheduledDowntimeData",)

import os
import sqlite3
import warnings

import numpy as np
from astropy.time import Time, TimeDelta

from rubin_sim.data import get_data_dir


class ScheduledDowntimeData:
    """Read the scheduled downtime data.

    This class deals with the scheduled downtime information that was previously produced for
    OpSim version 3.

    Parameters
    ----------
    start_time : `astropy.time.Time`
        The time of the start of the simulation.
        The cloud database will be assumed to start on Jan 01 of the same year.
    cloud_db : `str`, optional
        The full path name for the cloud database. Default None,
        which will use the database stored in the module ($SIMS_CLOUDMODEL_DIR/data/cloud.db).
    start_of_night_offset : `float`, optional
        The fraction of a day to offset from MJD.0 to reach the defined start of a night ('noon' works).
        Default 0.16 (UTC midnight in Chile) - 0.5 (minus half a day) = -0.34
    """

    def __init__(self, start_time, scheduled_downtime_db=None, start_of_night_offset=-0.34):
        self.scheduled_downtime_db = scheduled_downtime_db
        if self.scheduled_downtime_db is None:
            self.scheduled_downtime_db = os.path.join(get_data_dir(), "site_models", "scheduled_downtime.db")

        # downtime database starts in Jan 01 of the year of the start of the simulation.
        year_start = start_time.datetime.year
        self.night0 = Time("%d-01-01" % year_start, format="isot", scale="tai") + TimeDelta(
            start_of_night_offset, format="jd"
        )

        # Scheduled downtime data is a np.ndarray of start / end / activity for each scheduled downtime.
        self.downtime = None
        self.read_data()

    def __call__(self):
        """Return the current (if any) and any future scheduled downtimes.

        Parameters
        ----------
        time : `astropy.time.Time`
            Time in the simulation for which to find the current downtime.

        Returns
        -------
        downtime : `np.ndarray`
            The array of all unscheduled downtimes, with keys for 'start', 'end', 'activity',
            corresponding to astropy.time.Time, astropy.time.Time, and str.
        """
        return self.downtime

    def _downtime_status(self, time):
        """Look behind the scenes at the downtime status/next values"""
        next_start = self.downtime["start"].searchsorted(time, side="right")
        next_end = self.downtime["end"].searchsorted(time, side="right")
        if next_start > next_end:
            current = self.downtime[next_end]
        else:
            current = None
        future = self.downtime[next_start:]
        return current, future

    def read_data(self):
        """Read the scheduled downtime information from disk and translate to astropy.time.Times.

        This function gets the appropriate database file and creates the set of
        scheduled downtimes from it. The default behavior is to use the module stored
        database. However, an alternate database file can be provided. The alternate
        database file needs to have a table called *Downtime* with the following columns:

        night : `int`
            The night (from start of simulation) the downtime occurs.
        duration : `int`
            The duration (units=days) of the downtime.
        activity : `str`
            A description of the activity involved.
        """
        # Read from database.
        starts = []
        ends = []
        acts = []
        with sqlite3.connect(self.scheduled_downtime_db) as conn:
            cur = conn.cursor()
            cur.execute("select * from Downtime;")
            for row in cur:
                start_night = int(row[0])
                start_night = self.night0 + TimeDelta(start_night, format="jd")
                n_down = int(row[1])
                end_night = start_night + TimeDelta(n_down, format="jd")
                activity = row[2]
                starts.append(start_night)
                ends.append(end_night)
                acts.append(activity)
            cur.close()
        self.downtime = np.array(
            list(zip(starts, ends, acts)),
            dtype=[("start", "O"), ("end", "O"), ("activity", "O")],
        )

    def total_downtime(self):
        """Return total downtime (in days).

        Returns
        -------
        total : `int`
            Total number of downtime days.
        """
        total = 0
        for td in self.downtime["end"] - self.downtime["start"]:
            total += td.jd
        return total
