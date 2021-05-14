import numpy as np
from .baseMetric import BaseMetric

__all__ = ['LongGapAGNMetric']


class LongGapAGNMetric(BaseMetric):
    """max delta-t and average of the top-10 longest gaps.
    """

    def __init__(self, metricName='longGapAGNMetric',
                 mjdcol='observationStartMJD', units='days', xgaps=10, badval=-666,
                 **kwargs):
        """ Instantiate metric.
        mjdcol = column name for exposure time dates
        """
        cols = [mjdcol]
        super(LongGapAGNMetric, self).__init__(cols, metricName, units=units, **kwargs)
        self.badval = badval
        self.mjdcol = mjdcol
        self.xgaps = xgaps
        self.units = units

    def run(self, dataslice, slicePoint=None):
        metricval = np.diff(dataslice[self.mjdcol])
        return metricval

    def reduceMaxGap(self, metricval):
        if metricval.size > 0:
            result = np.max(metricval)
        else:
            result = self.badval
        return result

    def reduceAverageLongestXGaps(self, metricval):
        if np.size(metricval)-self.xgaps > 0:
            return np.average(np.sort(metricval)[np.size(metricval)-self.xgaps:])
        else:
            return self.badval
