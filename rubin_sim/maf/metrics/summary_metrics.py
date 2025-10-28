__all__ = (
    "FootprintFractionMetric",
    "FOArea",
    "FONv",
    "IdentityMetric",
    "NormalizeMetric",
    "ZeropointMetric",
)


import healpy as hp
import numpy as np

from .base_metric import BaseMetric

# Metrics which are primarily intended to be used as summary statistics.


class FootprintFractionMetric(BaseMetric):
    """Calculate fraction of a desired footprint got covered.
    Helpful to check if everything was covered in first year

    Parameters
    ----------
    footprint : `np.ndarray`, (N,)
        The HEALpix footprint to compare to.
        Nside of the footprint should match nside of the slicer.
    n_min : `int`
        The number of visits to require to consider an area covered
    """

    def __init__(self, footprint=None, n_min=1, **kwargs):
        super().__init__(**kwargs)
        self.footprint = footprint
        self.nside = hp.npix2nside(footprint.size)
        self.npix = np.where(self.footprint > 0)[0].size
        # get whole array passed
        self.mask_val = 0
        self.n_min = n_min

    def run(self, data_slice, slice_point=None):
        overlap = np.where((self.footprint > 0) & (data_slice["metricdata"] >= self.n_min))[0]
        result = overlap.size / self.npix
        return result


class FONv(BaseMetric):
    """Given asky area, what is the minimum and median NVISITS obtained over
    that area?
    (chooses the portion of the sky with the highest number of visits first).

    Parameters
    ----------
    col : `str` or `list` of `strs`, optional
        Name of the column in the numpy recarray passed to the summary metric.
    asky : `float`, optional
        Area of the sky to base the evaluation of number of visits over.
    nside : `int`, optional
        Nside parameter from healpix slicer, used to set the physical
        relationship between on-sky area and number of healpixels.
    n_visit : `int`, optional
        Number of visits to use as the benchmark value, if choosing to return
        a normalized n_visit value.
    norm : `bool`, optional
        Normalize the returned "n_visit" (min / median) values by n_visit,
        if true.
    metric_name : `str`, optional
        Name of the summary metric. Default FONv.
    """

    def __init__(self, col="metricdata", asky=18000.0, nside=128, n_visit=825, norm=False, **kwargs):
        """asky = square degrees"""
        super().__init__(col=col, **kwargs)
        self.nvisit = n_visit
        self.nside = nside
        # Determine how many healpixels are included in asky sq deg.
        self.asky = asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.npix__asky = int(np.ceil(self.asky / self.scale))
        self.norm = norm

    def run(self, data_slice, slice_point=None):
        result = np.empty(2, dtype=[("name", np.str_, 20), ("value", float)])
        result["name"][0] = "MedianNvis"
        result["name"][1] = "MinNvis"
        # If there is not even as much data as needed to cover Asky:
        if len(data_slice) < self.npix__asky:
            result["value"][0] = self.badval
            result["value"][1] = self.badval
            return result
        # Otherwise, calculate median and mean Nvis:
        nvis_sorted = np.sort(data_slice[self.colname])
        # Find the Asky's worth of healpixels with the largest # of visits.
        nvis__asky = nvis_sorted[-self.npix__asky :]
        result["value"][0] = np.median(nvis__asky)
        result["value"][1] = np.min(nvis__asky)
        if self.norm:
            result["value"] /= float(self.nvisit)
        return result


class FOArea(BaseMetric):
    """Given an n_visit threshold, how much AREA receives at least that many
    visits?

    Parameters
    ----------
    col : `str` or `list` of `strs`, optional
        Name of the column in the numpy recarray passed to the summary metric.
    n_visit : `int`, optional
        Number of visits to use as the minimum required --
        metric calculated area that has this many visits.
    asky : `float`, optional
        Area to use as the benchmark area value,
        if choosing to return a normalized Area value.
    nside : `int`, optional
        Nside parameter from healpix slicer, used to set the physical
        relationship between on-sky area and number of healpixels.
    norm : `bool`, optional
        If true, normalize the returned area value by asky.
    """

    def __init__(
        self,
        col="metricdata",
        n_visit=825,
        asky=18000.0,
        nside=128,
        norm=False,
        **kwargs,
    ):
        """asky = square degrees"""
        super().__init__(col=col, **kwargs)
        self.nvisit = n_visit
        self.nside = nside
        self.asky = asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.norm = norm

    def run(self, data_slice, slice_point=None):
        nvis_sorted = np.sort(data_slice[self.colname])
        # Identify the healpixels with more than Nvisits.
        nvis_min = nvis_sorted[np.where(nvis_sorted >= self.nvisit)]
        if len(nvis_min) == 0:
            result = self.badval
        else:
            result = nvis_min.size * self.scale
            if self.norm:
                result /= float(self.asky)
        return result


class IdentityMetric(BaseMetric):
    """Return the metric value.

    This is primarily useful as a summary statistic for UniSlicer metrics,
    to propagate the ~MetricBundle.metric_value into the results database.
    """

    def run(self, data_slice, slice_point=None):
        if len(data_slice[self.colname]) == 1:
            result = data_slice[self.colname][0]
        else:
            result = data_slice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """
    Return a metric values divided by 'norm_val'.
    Useful for turning summary statistics into fractions.
    """

    def __init__(self, col="metricdata", norm_val=1, **kwargs):
        super(NormalizeMetric, self).__init__(col=col, **kwargs)
        self.norm_val = float(norm_val)

    def run(self, data_slice, slice_point=None):
        result = data_slice[self.colname] / self.norm_val
        if len(result) == 1:
            return result[0]
        else:
            return result


class ZeropointMetric(BaseMetric):
    """
    Return a metric values with the addition of 'zp'.
    Useful for altering the zeropoint for summary statistics.
    """

    def __init__(self, col="metricdata", zp=0, **kwargs):
        super(ZeropointMetric, self).__init__(col=col, **kwargs)
        self.zp = zp

    def run(self, data_slice, slice_point=None):
        result = data_slice[self.colname] + self.zp
        if len(result) == 1:
            return result[0]
        else:
            return result
