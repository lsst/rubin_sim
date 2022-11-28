import numpy as np
from .base_metric import BaseMetric

__all__ = ["PairMetric"]


class PairMetric(BaseMetric):
    """
    Count the number of pairs that could be used for Solar System object detection
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        metric_name="Pairs",
        match_min=20.0,
        match_max=40.0,
        binsize=5.0,
        **kwargs
    ):
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
        self.mjd_col = mjd_col
        self.binsize = binsize / 60.0 / 24.0
        self.match_min = match_min / 60.0 / 24.0
        self.match_max = match_max / 60.0 / 24.0
        super(PairMetric, self).__init__(
            col=mjd_col, metric_name=metric_name, units="N Pairs", **kwargs
        )

    def run(self, data_slice, slice_point=None):
        bins = np.arange(
            data_slice[self.mjd_col].min(),
            data_slice[self.mjd_col].max() + self.binsize,
            self.binsize,
        )

        hist, bin_edges = np.histogram(data_slice[self.mjd_col], bins=bins)
        nbin_min = np.round(self.match_min / self.binsize)
        nbin_max = np.round(self.match_max / self.binsize)
        bins_to_check = np.arange(nbin_min, nbin_max + 1, 1)
        bins_w_obs = np.where(hist > 0)[0]
        # now, for each bin with an observation, need to check if there is a bin
        # far enough ahead that is also populated.
        result = 0
        for binadd in bins_to_check:
            result += np.size(np.intersect1d(bins_w_obs, bins_w_obs + binadd))
        if result == 0:
            result = self.badval
        return result
