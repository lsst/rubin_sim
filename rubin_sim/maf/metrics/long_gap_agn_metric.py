__all__ = ("LongGapAGNMetric",)

import numpy as np

from .base_metric import BaseMetric


class LongGapAGNMetric(BaseMetric):
    """max delta-t and average of the top-10 longest gaps."""

    def __init__(
        self,
        metric_name="longGapAGNMetric",
        mjdcol="observationStartMJD",
        units="days",
        xgaps=10,
        badval=-666,
        **kwargs,
    ):
        """Instantiate metric.
        mjdcol = column name for exposure time dates
        """
        cols = [mjdcol]
        super(LongGapAGNMetric, self).__init__(cols, metric_name, units=units, **kwargs)
        self.badval = badval
        self.mjdcol = mjdcol
        self.xgaps = xgaps
        self.units = units

    def run(self, data_slice, slice_point=None):
        metricval = np.diff(data_slice[self.mjdcol])
        return metricval

    def reduce_max_gap(self, metricval):
        if metricval.size > 0:
            result = np.max(metricval)
        else:
            result = self.badval
        return result

    def reduce_average_longest_x_gaps(self, metricval):
        if np.size(metricval) - self.xgaps > 0:
            return np.average(np.sort(metricval)[np.size(metricval) - self.xgaps :])
        else:
            return self.badval
