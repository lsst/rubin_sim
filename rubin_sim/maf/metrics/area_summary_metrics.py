__all__ = ("AreaSummaryMetric", "AreaThresholdMetric")

import healpy as hp
import numpy as np

from .base_metric import BaseMetric


class AreaSummaryMetric(BaseMetric):
    """
    Find the min/max of a value over the area with the 'best' results
    in the metric.
    This is a handy substitute for when users want to know "the WFD value".

    Parameters
    ----------
    area : `float`
        The area to consider (sq degrees)
    decreasing : `bool`
        Should the values be sorted by increasing or decreasing order.
        For values where "larger is better", decreasing (True) is probably
        what you want. For metrics where "smaller is better"
        (e.g., astrometric precission), set decreasing to False.
    reduce_func : None
        The function to reduce the clipped values by.
        Will default to min/max depending on the bool val of the decreasing
        kwarg.
    """

    def __init__(
        self,
        col="metricdata",
        metric_name="AreaSummary",
        area=18000.0,
        decreasing=True,
        reduce_func=None,
        **kwargs,
    ):
        super().__init__(col=col, metric_name=metric_name, **kwargs)
        self.area = area
        self.decreasing = decreasing
        self.reduce_func = reduce_func
        self.mask_val = np.nan  # Include so all values get passed
        self.col = col
        if reduce_func is None:
            if decreasing:
                self.reduce_func = np.min
            else:
                self.reduce_func = np.max
        else:
            self.reduce_func = reduce_func

    def run(self, data_slice, slice_point=None):
        # find out what nside we have
        nside = hp.npix2nside(data_slice.size)
        pix_area = hp.nside2pixarea(nside, degrees=True)
        n_pix_needed = int(np.ceil(self.area / pix_area))

        # Only use the finite data
        data = data_slice[self.col][np.isfinite(data_slice[self.col].astype(float))]
        order = np.argsort(data)
        if self.decreasing:
            order = order[::-1]
        result = self.reduce_func(data[order][0:n_pix_needed])
        return result


class AreaThresholdMetric(BaseMetric):
    """Find the amount of area on the sky that meets a given threshold value.

    The area per pixel is determined from the size of the metric_values
    array passed to the summary metric.
    This assumes that both all values are passed and that the metric was
    calculated with a healpix slicer.

    Parameters
    ----------
    upper_threshold : `float` or None
        The metric value must be below this threshold to count toward the area.
        Default None implies no upper bound.
    lower_threshold : `float` or None, opt
        The metric value must be above this threshold to count toward the area.
        Default None implies no lower bound.
    """

    def __init__(
        self,
        col="metricdata",
        metric_name="AreaThreshold",
        upper_threshold=None,
        lower_threshold=None,
        **kwargs,
    ):
        super().__init__(col=col, metric_name=metric_name, **kwargs)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.mask_val = np.nan  # Include so all values get passed
        self.col = col
        self.units = "degrees"

    def run(self, data_slice, slice_point=None):
        # find out what nside we have
        nside = hp.npix2nside(data_slice.size)
        pix_area = hp.nside2pixarea(nside, degrees=True)
        # Look for pixels which match the critera for the thresholds
        if self.upper_threshold is None and self.lower_threshold is None:
            npix = len(data_slice)
        elif self.upper_threshold is None:
            npix = len(np.where(data_slice[self.col] > self.lower_threshold)[0])
        elif self.lower_threshold is None:
            npix = len(np.where(data_slice[self.col] < self.upper_threshold)[0])
        else:
            npix = len(
                np.where(
                    (data_slice[self.col] > self.lower_threshold)
                    and (data_slice[self.col] < self.upper_threshold)
                )[0]
            )
        area = pix_area * npix
        return area
