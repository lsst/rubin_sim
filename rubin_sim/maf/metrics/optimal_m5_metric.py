__all__ = ("OptimalM5Metric",)

import warnings

import numpy as np

from .base_metric import BaseMetric
from .simple_metrics import Coaddm5Metric


class OptimalM5Metric(BaseMetric):
    """Compare the co-added depth of the survey to one where
    all the observations were taken on the meridian.

    Parameters
    ----------
    m5_col : str ('fiveSigmaDepth')
        Column name that contains the five-sigma limiting depth of
        each observation
    opt_m5_col : str ('m5Optimal')
        The column name of the five-sigma-limiting depth if the
        observation had been taken on the meridian.
    normalize : bool (False)
        If False, metric returns how many more observations would need
        to be taken to reach the optimal depth.  If True, the number
        is normalized by the total number of observations already taken
        at that position.
    mag_diff : bool (False)
        If True, metric returns the magnitude difference between the
        achieved coadded depth and the optimal coadded depth.

    Returns
    --------
    numpy.array

    If mag_diff is True, returns the magnitude difference between the
    optimal and actual coadded depth.  If normalize is False
    (default), the result is the number of additional observations
    (taken at the median depth) the survey needs to catch up to
    optimal.  If normalize is True, the result is divided by the
    number of observations already taken. So if a 10-year survey
    returns 20%, it would need to run for 12 years to reach the same
    depth as a 10-year meridian survey.

    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        opt_m5_col="m5Optimal",
        filter_col="filter",
        mag_diff=False,
        normalize=False,
        **kwargs,
    ):
        if normalize:
            self.units = "% behind"
        else:
            self.units = "N visits behind"
        if mag_diff:
            self.units = "mags"
        super(OptimalM5Metric, self).__init__(
            col=[m5_col, opt_m5_col, filter_col], units=self.units, **kwargs
        )
        self.m5_col = m5_col
        self.opt_m5_col = opt_m5_col
        self.normalize = normalize
        self.filter_col = filter_col
        self.mag_diff = mag_diff
        self.coadd_regular = Coaddm5Metric(m5_col=m5_col)
        self.coadd_optimal = Coaddm5Metric(m5_col=opt_m5_col)

    def run(self, data_slice, slice_point=None):
        filters = np.unique(data_slice[self.filter_col])
        if np.size(filters) > 1:
            warnings.warn(
                "OptimalM5Metric does not make sense mixing filters. Currently using filters " + str(filters)
            )
        regular_depth = self.coadd_regular.run(data_slice)
        optimal_depth = self.coadd_optimal.run(data_slice)
        if self.mag_diff:
            return optimal_depth - regular_depth

        median_single = np.median(data_slice[self.m5_col])

        # Number of additional median observations to get as deep as optimal
        result = (10.0 ** (0.8 * optimal_depth) - 10.0 ** (0.8 * regular_depth)) / (
            10.0 ** (0.8 * median_single)
        )

        if self.normalize:
            result = result / np.size(data_slice) * 100.0

        return result
