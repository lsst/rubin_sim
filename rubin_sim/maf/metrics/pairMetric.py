import numpy as np
from .baseMetric import BaseMetric

__all__ = ['PairMetric']


class PairMetric(BaseMetric):
    """
    Count the number of pairs that could be used for Solar System object detection
    """
    def __init__(self, mjdCol='observationStartMJD', metricName='Pairs', match_min=20., match_max=40.,
                 binsize=5., **kwargs):
        """
        Parameters
        ----------
        match_min : float (20.)
            Minutes after first observation to count something as a match
        match_max : float (40.)
            Minutes after first observation to count something as a match
        binsize : float (5.)
            Binsize to use (minutes)
        """
        self.mjdCol = mjdCol
        self.binsize = binsize/60./24.
        self.match_min = match_min/60./24.
        self.match_max = match_max/60./24.
        super(PairMetric, self).__init__(col=mjdCol, metricName=metricName,
                                         units='N Pairs', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        bins = np.arange(dataSlice[self.mjdCol].min(),
                         dataSlice[self.mjdCol].max()+self.binsize, self.binsize)

        hist, bin_edges = np.histogram(dataSlice[self.mjdCol], bins=bins)
        nbin_min = np.round(self.match_min / self.binsize)
        nbin_max = np.round(self.match_max / self.binsize)
        bins_to_check = np.arange(nbin_min, nbin_max+1, 1)
        bins_w_obs = np.where(hist > 0)[0]
        # now, for each bin with an observation, need to check if there is a bin
        # far enough ahead that is also populated.
        result = 0
        for binadd in bins_to_check:
            result += np.size(np.intersect1d(bins_w_obs, bins_w_obs+binadd))
        if result == 0:
            result = self.badval
        return result


