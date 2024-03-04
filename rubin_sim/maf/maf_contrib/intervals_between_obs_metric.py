# Example for IntervalsBetweenObsMetric
# Somayeh Khakpash - Lehigh University
# Last edited : 10/21/2020
# Calculates statistics (mean or median or standard deviation) of intervals
# between observations during simultaneous windows/Inter-seasonal gap of
# another survey.
# SurveyIntervals is the list of the survey observing window/Inter-seasonal
# gap intervals. It should be in the format:
# SurveyIntervals = [ [YYYY-MM-DD, YYYY-MM-DD] , [YYYY-MM-DD, YYYY-MM-DD] ,
# ... , [YYYY-MM-DD, YYYY-MM-DD] ]
# We are interested in calculating this metric in each of the LSST passbands.
# The difference between this metric and the VisitGapMetric metric is that
# VisitGapMetric calculates reduceFunc of gaps between observations of a
# data_slice throughout the whole
# baseline, but IntervalsBetweenObsMetric calculates the gaps between
# observations during another survey observing window.
# This metric combined with surveys footprint
# overlap can determine how many often another survey footprint is
# observed by LSST during specific time intervals.
__all__ = ("IntervalsBetweenObsMetric",)

import numpy as np
from astropy.time import Time

from rubin_sim.maf.metrics import BaseMetric


class IntervalsBetweenObsMetric(BaseMetric):
    def __init__(
        self,
        survey_intervals,
        stat,
        metric_name="IntervalsBetweenObsMetric",
        time_col="observationStartMJD",
        **kwargs,
    ):
        self.time_col = time_col
        self.metric_name = metric_name
        self.survey_intervals = survey_intervals
        self.stat = stat
        super(IntervalsBetweenObsMetric, self).__init__(col=time_col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        obs_diff = []

        for interval in self.survey_intervals:
            start_interval = Time(interval[0] + " 00:00:00")
            end_interval = Time(interval[1] + " 00:00:00")
            index = data_slice[self.time_col][
                np.where(
                    (data_slice[self.time_col] > start_interval.mjd)
                    & (data_slice[self.time_col] < end_interval.mjd)
                )[0]
            ]
            obs_diff_per_interval = np.diff(index)
            obs_diff = obs_diff + obs_diff_per_interval.tolist()

        if self.stat == "mean":
            result = np.mean(obs_diff)

        elif self.stat == "median":
            result = np.median(obs_diff)

        elif self.stat == "std":
            result = np.std(obs_diff)

        return result
