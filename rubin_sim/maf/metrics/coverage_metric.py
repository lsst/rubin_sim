__all__ = ("YearCoverageMetric",)

import numpy as np

from .base_metric import BaseMetric


class YearCoverageMetric(BaseMetric):
    """Count the number of bins covered by night_col -- default bins are 'years'.
    Handy for checking that a point on the sky gets observed every year, as the default settings
    result in the metric returning the number years in the data_slice (when used with a HealpixSlicer).

    Parameters
    ----------
    night_col: str, optional
        Data column to histogram. Default 'night'.
    bins: numpy.ndarray, optional
        Bins to use in the histogram. Default corresponds to years 0-10 (with 365.25 nights per year).
    units: str, optional
        Units to use for the metric result. Default 'N years'.

    Returns
    -------
    integer
        Number of histogram bins where the histogram value is greater than 0.
        Typically this will be the number of years in the 'night_col'.
    """

    def __init__(self, night_col="night", bins=None, units=None, **kwargs):
        self.night_col = night_col
        if bins is None:
            self.bins = np.arange(0, np.ceil(365.25 * 10.0), 365.25) - 0.5
        else:
            self.bins = bins

        if units is None:
            units = "N years"

        super().__init__([night_col], units=units)

    def run(self, data_slice, slice_point):
        hist, be = np.histogram(data_slice[self.night_col], bins=self.bins)
        result = np.where(hist > 0)[0].size
        return result
