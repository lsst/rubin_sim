# Example for numObsInSurveyTimeOverlap
# Somayeh Khakpash - Lehigh University
# Last edited : 10/21/2020
# Calculates number of observations during simultaneous windows of another
# survey.
# SurveyObsWin is the list of the survey observing window/inter-seasonal
# gap intervals. It should be in the format:
# SurveyObsWin = [ [YYYY-MM-DD, YYYY-MM-DD] ,
# [YYYY-MM-DD, YYYY-MM-DD] , ... , [YYYY-MM-DD, YYYY-MM-DD] ]

__all__ = ("NumObsInSurveyTimeOverlapMetric",)

import numpy as np
from astropy.time import Time

from rubin_sim.maf.metrics import BaseMetric


class NumObsInSurveyTimeOverlapMetric(BaseMetric):
    def __init__(
        self,
        survey_obs_win,
        time_col="observationStartMJD",
        metric_name="NumObsInSurveyTimeOverlapMetric",
        **kwargs,
    ):
        self.time_col = time_col
        self.metric_name = metric_name
        self.survey_obs_win = survey_obs_win
        super(NumObsInSurveyTimeOverlapMetric, self).__init__(col=time_col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        n__obs = 0
        for interval in self.survey_obs_win:
            start_interval = Time(interval[0] + " 00:00:00")
            end_interval = Time(interval[1] + " 00:00:00")
            index = np.where(
                (data_slice[self.time_col] > start_interval.mjd)
                & (data_slice[self.time_col] < end_interval.mjd)
            )[0]
            n__obs = n__obs + np.size(index)

        return n__obs
