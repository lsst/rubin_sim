"""Metrics for scheduler monitoring and progress."""

__all__ = ["AgeMetric"]

import numpy as np

from .base_metric import BaseMetric


class AgeMetric(BaseMetric):
    def __init__(
        self, mjd, mjd_col="observationStartMJD", long_limit=30, metric_name="age", mask_val=np.nan, **kwargs
    ):
        """Metric that shows the time since the previous visit in each slice,
        as of a given time

        Parameters
        ----------
        mjd : `float`
            Reference time for the age.
        mjd_col : `str`
            Column with the time of visit, by default "observationStartMJD"
        long_limit : `int`
            The age past which to mask values, by default 30
        metric_name : `str`
            The metric name, by default 'age'
        mask_val : `object`
            Name for masked values, by default np.nan
        """
        self.mjd = mjd
        self.mjd_col = mjd_col
        self.long_limit = long_limit
        super().__init__(col=[self.mjd_col], metric_name=metric_name, mask_val=mask_val, **kwargs)

    def run(self, data_slice, slice_point=None):
        age = self.mjd - np.max(data_slice[self.mjd_col])
        if age > self.long_limit:
            age = self.mask_val
        return age
