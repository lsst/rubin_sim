import numpy as np
from .baseMetric import BaseMetric
import healpy as hp

__all__ = ['AreaSummaryMetric']


class AreaSummaryMetric(BaseMetric):
    """
    Find the min/max of a value in the best area. This is a handy substitute for when
    users want to know "the WFD value".

    Parameters
    ----------
    area : float (18000)
        The area to consider (sq degrees)
    decreasing : bool (True)
        Should the values be sorted by increasing or decreasing order. For values where
        "larger is better", decreasing is probably what you want. For metrics where
        "smaller is better" (e.g., astrometric precission), set decreasing to False.
    reduce_func : None
        The function to reduce the clipped values by. Will default to min/max depending on
        the bool val of the decreasing kwarg.

    """
    def __init__(self, col='metricdata', metricName='AreaSummary', area=18000., decreasing=True,
                 reduce_func=None, **kwargs):
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.area = area
        self.decreasing = decreasing
        self.reduce_func = reduce_func
        self.maskVal = np.nan  # Include so all values get passed
        self.col = col
        if reduce_func is None:
            if decreasing:
                self.reduce_func = np.min
            else:
                self.reduce_func = np.max
        else:
            self.reduce_func = reduce_func

    def run(self, dataSlice, slicePoint=None):
        # find out what nside we have
        nside = hp.npix2nside(dataSlice.size)
        pix_area = hp.nside2pixarea(nside, degrees=True)
        n_pix_needed = int(np.ceil(self.area/pix_area))

        # Only use the finite data
        data = dataSlice[self.col][np.isfinite(dataSlice[self.col].astype(float))]
        order = np.argsort(data)
        if self.decreasing:
            order = order[::-1]
        result = self.reduce_func(data[order][0:n_pix_needed])
        return result


class AreaThresholdMetric(BaseMetric):
    """
    Find the amount of area on the sky that meets a given threshold value.

    The area per pixel is determined from the size of the metricValues array passed to the summary metric.
    This assumes that both all values are passed and that the metric was calculated with a healpix slicer.

    Parameters
    ----------
    upper_threshold : `float` or None
        The metric value must be below this threshold to count toward the area.
        Default None implies no upper bound.
    lower_threshold : `float` or None, opt
        The metric value must be above this threshold to count toward the area.
        Default None implies no lower bound.
    """
    def __init__(self, col='metricdata', metricName='AreaThreshold',
                 upper_threshold=None, lower_threshold=None, **kwargs):
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.nside = nside
        self.areaPerSlice = hp.nside2pixarea(nside, degrees=True)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.maskVal = np.nan  # Include so all values get passed
        self.col = col

    def run(self, dataSlice, slicePoint=None):
        # find out what nside we have
        nside = hp.npix2nside(dataSlice.size)
        pix_area = hp.nside2pixarea(nside, degrees=True)
        # Look for pixels which match the critera for the thresholds
        if self.upper_threshold is None and self.lower_threshold is None:
            npix = len(dataSlice)
        elif self.upper_threshold is None:
            npix = len(np.where(dataSlice[self.col] > self.lower_threshold)[0])
        elif self.lower_threshold is None:
            npix = len(np.where(dataSlice[self.col] < self.upper_threshold)[0])
        else:
            npix = len(np.where((dataSlice[self.col] > self.lower_threshold) and
                                (dataSlice[self.col] < self.upper_threshold))[0])
        area = pix_area * npix
        return area