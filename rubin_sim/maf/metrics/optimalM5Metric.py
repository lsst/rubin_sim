from builtins import str
from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
import numpy as np
import warnings

__all__ = ['OptimalM5Metric']


class OptimalM5Metric(BaseMetric):
    """Compare the co-added depth of the survey to one where
    all the observations were taken on the meridian.

    Parameters
    ----------
    m5Col : str ('fiveSigmaDepth')
        Column name that contains the five-sigma limiting depth of
        each observation
    optM5Col : str ('m5Optimal')
        The column name of the five-sigma-limiting depth if the
        observation had been taken on the meridian.
    normalize : bool (False)
        If False, metric returns how many more observations would need
        to be taken to reach the optimal depth.  If True, the number
        is normalized by the total number of observations already taken
        at that position.
    magDiff : bool (False)
        If True, metric returns the magnitude difference between the
        achieved coadded depth and the optimal coadded depth.

    Returns
    --------
    numpy.array

    If magDiff is True, returns the magnitude difference between the
    optimal and actual coadded depth.  If normalize is False
    (default), the result is the number of additional observations
    (taken at the median depth) the survey needs to catch up to
    optimal.  If normalize is True, the result is divided by the
    number of observations already taken. So if a 10-year survey
    returns 20%, it would need to run for 12 years to reach the same
    depth as a 10-year meridian survey.

    """

    def __init__(self, m5Col='fiveSigmaDepth', optM5Col='m5Optimal',
                 filterCol='filter', magDiff=False, normalize=False, **kwargs):

        if normalize:
            self.units = '% behind'
        else:
            self.units = 'N visits behind'
        if magDiff:
                self.units = 'mags'
        super(OptimalM5Metric, self).__init__(col=[m5Col, optM5Col,filterCol],
                                              units=self.units, **kwargs)
        self.m5Col = m5Col
        self.optM5Col = optM5Col
        self.normalize = normalize
        self.filterCol = filterCol
        self.magDiff = magDiff
        self.coaddRegular = Coaddm5Metric(m5Col=m5Col)
        self.coaddOptimal = Coaddm5Metric(m5Col=optM5Col)

    def run(self, dataSlice, slicePoint=None):

        filters = np.unique(dataSlice[self.filterCol])
        if np.size(filters) > 1:
            warnings.warn("OptimalM5Metric does not make sense mixing filters. Currently using filters " +
                          str(filters))
        regularDepth = self.coaddRegular.run(dataSlice)
        optimalDepth = self.coaddOptimal.run(dataSlice)
        if self.magDiff:
            return optimalDepth-regularDepth

        medianSingle = np.median(dataSlice[self.m5Col])

        # Number of additional median observations to get as deep as optimal
        result = (10.**(0.8 * optimalDepth)-10.**(0.8 * regularDepth)) / \
                 (10.**(0.8 * medianSingle))

        if self.normalize:
            result = result/np.size(dataSlice)*100.

        return result
