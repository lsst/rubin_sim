import numpy as np
from .baseMetric import BaseMetric

__all__ = ['YearCoverageMetric']


class YearCoverageMetric(BaseMetric):
    """Count the number of bins covered by nightCol -- default bins are 'years'.
    Handy for checking that a point on the sky gets observed every year, as the default settings
    result in the metric returning the number years in the dataslice (when used with a HealpixSlicer).

    Parameters
    ----------
    nightCol: str, opt
        Data column to histogram. Default 'night'.
    bins: numpy.ndarray, opt
        Bins to use in the histogram. Default corresponds to years 0-10 (with 365.25 nights per year).
    units: str, opt
        Units to use for the metric result. Default 'N years'.

    Returns
    -------
    integer
        Number of histogram bins where the histogram value is greater than 0.
        Typically this will be the number of years in the 'nightCol'.
    """

    def __init__(self, nightCol='night', bins=None, units=None, **kwargs):
        self.nightCol = nightCol
        if bins is None:
            self.bins = np.arange(0, np.ceil(365.25*10.), 365.25) - 0.5
        else:
            self.bins = bins

        if units is None:
            units = 'N years'

        super().__init__([nightCol], units=units)

    def run(self, dataSlice, slicePoint):
        hist, be = np.histogram(dataSlice[self.nightCol], bins=self.bins)
        result = np.where(hist > 0)[0].size
        return result
