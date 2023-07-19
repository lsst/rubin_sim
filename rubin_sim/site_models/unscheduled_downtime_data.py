__all__ = ("UnscheduledDowntimeData",)

import random
import warnings

import numpy as np
from astropy.time import Time, TimeDelta


class UnscheduledDowntimeData:
    """Handle (and create) the unscheduled downtime information.

    Parameters
    ----------
    start_time : `astropy.time.Time`
        The time of the start of the simulation.
        The cloud database will be assumed to start on Jan 01 of the same year.
    seed : `int`, optional
        The random seed for creating the random nights of unscheduled downtime. Default 1516231120.
    start_of_night_offset : `float`, optional
        The fraction of a day to offset from MJD.0 to reach the defined start of a night ('noon' works).
        Default 0.16 (UTC midnight in Chile) - 0.5 (minus half a day) = -0.34
    survey_length : `int`, optional
        The number of nights in the total survey. Default 3650*2.
    """

    MINOR_EVENT = {"P": 0.0137, "length": 1, "level": "minor event"}
    INTERMEDIATE_EVENT = {"P": 0.00548, "length": 3, "level": "intermediate event"}
    MAJOR_EVENT = {"P": 0.00137, "length": 7, "level": "major event"}
    CATASTROPHIC_EVENT = {"P": 0.000274, "length": 14, "level": "catastrophic event"}

    def __init__(
        self,
        start_time,
        seed=1516231120,
        start_of_night_offset=-0.34,
        survey_length=3650 * 2,
    ):
        self.seed = seed
        self.survey_length = survey_length
        year_start = start_time.datetime.year
        self.night0 = Time("%d-01-01" % year_start, format="isot", scale="tai") + TimeDelta(
            start_of_night_offset, format="jd"
        )

        # Scheduled downtime data is a np.ndarray of start / end / activity for each scheduled downtime.
        self.downtime = None
        self.make_data()

    def __call__(self):
        """Return the array of unscheduled downtimes.

        Parameters
        ----------
        time : `astropy.time.Time`
            Time in the simulation for which to find the current downtime.

        Returns
        -------
        downtime : `np.ndarray`
            The array of all unscheduled downtimes, with keys for 'start', 'end', 'activity',
            corresponding to `astropy.time.Time`, `astropy.time.Time`, and `str`.
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

    def make_data(self):
        """Configure the set of unscheduled downtimes.

        This function creates the unscheduled downtimes based on a set of probabilities
        of the downtime type occurance.

        The random downtime is calculated using the following probabilities:

        minor event : remainder of night and next day = 5/365 days e.g. power supply failure
        intermediate : 3 nights = 2/365 days e.g. repair filter mechanism, rotator, hexapod, or shutter
        major event : 7 nights = 1/2*365 days
        catastrophic event : 14 nights = 1/3650 days e.g. replace a raft
        """
        random.seed(self.seed)

        starts = []
        ends = []
        acts = []
        night = 0
        while night < self.survey_length:
            prob = random.random()
            if prob < self.CATASTROPHIC_EVENT["P"]:
                start_night = self.night0 + TimeDelta(night, format="jd")
                starts.append(start_night)
                end_night = start_night + TimeDelta(self.CATASTROPHIC_EVENT["length"], format="jd")
                ends.append(end_night)
                acts.append(self.CATASTROPHIC_EVENT["level"])
                night += self.CATASTROPHIC_EVENT["length"] + 1
                continue
            else:
                prob = random.random()
                if prob < self.MAJOR_EVENT["P"]:
                    start_night = self.night0 + TimeDelta(night, format="jd")
                    starts.append(start_night)
                    end_night = start_night + TimeDelta(self.MAJOR_EVENT["length"], format="jd")
                    ends.append(end_night)
                    acts.append(self.MAJOR_EVENT["level"])
                    night += self.MAJOR_EVENT["length"] + 1
                    continue
                else:
                    prob = random.random()
                    if prob < self.INTERMEDIATE_EVENT["P"]:
                        start_night = self.night0 + TimeDelta(night, format="jd")
                        starts.append(start_night)
                        end_night = start_night + TimeDelta(self.INTERMEDIATE_EVENT["length"], format="jd")
                        ends.append(end_night)
                        acts.append(self.INTERMEDIATE_EVENT["level"])
                        night += self.INTERMEDIATE_EVENT["length"] + 1
                        continue
                    else:
                        prob = random.random()
                        if prob < self.MINOR_EVENT["P"]:
                            start_night = self.night0 + TimeDelta(night, format="jd")
                            starts.append(start_night)
                            end_night = start_night + TimeDelta(self.MINOR_EVENT["length"], format="jd")
                            ends.append(end_night)
                            acts.append(self.MINOR_EVENT["level"])
                            night += self.MINOR_EVENT["length"] + 1
            night += 1
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
