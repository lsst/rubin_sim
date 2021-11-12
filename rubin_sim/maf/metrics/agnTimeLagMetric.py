import numpy as np
from .baseMetric import BaseMetric

__all__ = ['AGN_TimeLagMetric']


class AGN_TimeLagMetric(BaseMetric):
    def __init__(self, lag=100, z=1, log=False, threshold=2.2, calcType='mean',
                 mjdCol='observationStartMJD', filterCol='filter',
                 metricName=None, **kwargs):
        self.lag = lag
        self.z = z
        self.log = log
        self.threshold = threshold
        self.calcType = calcType
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        if metricName is None:
            metricName = f'AGN_TimeLag_{lag}_days'
        super().__init__(col=[self.mjdCol, self.filterCol], metricName=metricName, **kwargs)

    # Calculate NQUIST value for time-lag and sampling time (redshift is included in formula if desired)
    def _getNquistValue(self, caden, lag, z):
        return (lag / ((1 + z) * caden))

    def run(self, dataSlice, slicePoint=None):
        # Calculate differences in time between visits
        mv = np.sort(dataSlice[self.mjdCol])
        val = np.diff(mv)
        # If there was only one visit; bail out now.
        if len(val) == 0:
            return self.badval

        # Otherwise summarize the time differences as:
        if self.calcType == 'mean':
            val = np.mean(val)
        elif self.calcType == 'min':
            val = np.min(val)
        elif self.calcType == 'max':
            val = np.max(val)
        else:
            # find the greatest common divisor
            val = np.rint(val).astype(int)
            val = np.gcd.reduce(val)

        # Will always have a value at this point
        nquist = self._getNquistValue(val, self.lag, self.z)
        if self.log:
            nquist = np.log(nquist)

        # Threshold nquist value is 2.2,
        # hence we are aiming to show values higher than threshold (2.2) value
        threshold = self.threshold
        if self.log:
            threshold = np.log(threshold)

        if nquist < threshold:
            nquist = self.badval

        return nquist
