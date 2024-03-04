__all__ = ("PairMetric",)

import numpy as np

from .base_metric import BaseMetric


class PairMetric(BaseMetric):
    """Count the number of pairs of visits that could be used for
    Solar System object detection.

    Parameters
    ----------
    mjd_col : `str`, opt
        Name of the MJD column in the observations.
    metric_name : `str`, opt
        Name for the resulting metric. If None, one is constructed from
        the class name.
    match_min : `float`, opt
        Minutes after first observation to count something as a match.
    match_max : `float`, opt
        Minutes after first observation to count something as a match.
    bin_size : `float`, opt
        bin_size to use (minutes).
        Note that bin_size should be considerably smaller than the difference
        between match_min and match_max.

    Result
    ------
    num_pairs : `float`
        The number of pairs of visits within the min and max time range.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        metric_name="Pairs",
        match_min=20.0,
        match_max=40.0,
        bin_size=5.0,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.bin_size = bin_size / 60.0 / 24.0
        self.match_min = match_min / 60.0 / 24.0
        self.match_max = match_max / 60.0 / 24.0
        super(PairMetric, self).__init__(col=mjd_col, metric_name=metric_name, units="N Pairs", **kwargs)

    def run(self, data_slice, slice_point=None):
        bins = np.arange(
            data_slice[self.mjd_col].min(),
            data_slice[self.mjd_col].max() + self.bin_size,
            self.bin_size,
        )

        hist, bin_edges = np.histogram(data_slice[self.mjd_col], bins=bins)
        nbin_min = np.round(self.match_min / self.bin_size)
        nbin_max = np.round(self.match_max / self.bin_size)
        bins_to_check = np.arange(nbin_min, nbin_max + 1, 1)
        bins_w_obs = np.where(hist > 0)[0]
        # now, for each bin with an observation,
        # need to check if there is a bin
        # far enough ahead that is also populated.
        result = 0
        for binadd in bins_to_check:
            result += np.size(np.intersect1d(bins_w_obs, bins_w_obs + binadd))
        if result == 0:
            result = self.badval
        return result
