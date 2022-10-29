import warnings
import numpy as np
import healpy as hp
from scipy import interpolate
from .base_metric import BaseMetric

# A collection of metrics which are primarily intended to be used as summary statistics.

__all__ = [
    "FOArea",
    "FONv",
    "IdentityMetric",
    "NormalizeMetric",
    "ZeropointMetric",
    "TotalPowerMetric",
    "StaticProbesFoMEmulatorMetricSimple",
]


class FONv(BaseMetric):
    """
    Metrics based on a specified area, but returning NVISITS related to area:
    given asky, what is the minimum and median number of visits obtained over that much area?
    (choose the portion of the sky with the highest number of visits first).

    Parameters
    ----------
    col : str or list of strs, optional
        Name of the column in the numpy recarray passed to the summary metric.
    asky : float, optional
        Area of the sky to base the evaluation of number of visits over.
        Default 18,0000 sq deg.
    nside : int, optional
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    nvisit : int, optional
        Number of visits to use as the benchmark value, if choosing to return a normalized nvisit value.
    norm : `bool`, optional
        Normalize the returned "nvisit" (min / median) values by nvisit, if true.
        Default False.
    metricName : str, optional
        Name of the summary metric. Default FONv.
    """

    def __init__(
        self,
        col="metricdata",
        asky=18000.0,
        nside=128,
        nvisit=825,
        norm=False,
        metric_name="FONv",
        **kwargs
    ):
        """asky = square degrees"""
        super().__init__(col=col, metric_name=metric_name, **kwargs)
        self.nvisit = nvisit
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
            # Return the same type of metric value, to make it easier downstream.
            result["value"][0] = self.badval
            result["value"][1] = self.badval
            return result
        # Otherwise, calculate median and mean Nvis:
        name = data_slice.dtype.names[0]
        nvis_sorted = np.sort(data_slice[name])
        # Find the Asky's worth of healpixels with the largest # of visits.
        nvis__asky = nvis_sorted[-self.npix__asky :]
        result["value"][0] = np.median(nvis__asky)
        result["value"][1] = np.min(nvis__asky)
        if self.norm:
            result["value"] /= float(self.nvisit)
        return result


class FOArea(BaseMetric):
    """
    Metrics based on a specified number of visits, but returning AREA related to Nvisits:
    given nvisit, what amount of sky is covered with at least that many visits?

    Parameters
    ----------
    col : str or list of strs, optional
        Name of the column in the numpy recarray passed to the summary metric.
    nvisit : int, optional
        Number of visits to use as the minimum required -- metric calculated area that has this many visits.
        Default 825.
    asky : float, optional
        Area to use as the benchmark value, if choosing to returned a normalized Area value.
        Default 18,0000 sq deg.
    nside : int, optional
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    norm : `bool`, optional
        Normalize the returned "area" (area with minimum nvisit visits) value by asky, if true.
        Default False.
    metricName : str, optional
        Name of the summary metric. Default FOArea.
    """

    def __init__(
        self,
        col="metricdata",
        nvisit=825,
        asky=18000.0,
        nside=128,
        norm=False,
        metric_name="FOArea",
        **kwargs
    ):
        """asky = square degrees"""
        super().__init__(col=col, metric_name=metric_name, **kwargs)
        self.nvisit = nvisit
        self.nside = nside
        self.asky = asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.norm = norm

    def run(self, data_slice, slice_point=None):
        name = data_slice.dtype.names[0]
        nvis_sorted = np.sort(data_slice[name])
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
    """
    Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics.
    """

    def run(self, data_slice, slice_point=None):
        if len(data_slice[self.colname]) == 1:
            result = data_slice[self.colname][0]
        else:
            result = data_slice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """
    Return a metric values divided by 'norm_val'. Useful for turning summary statistics into fractions.
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
    Return a metric values with the addition of 'zp'. Useful for altering the zeropoint for summary statistics.
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


class TotalPowerMetric(BaseMetric):
    """
    Calculate the total power in the angular power spectrum between lmin/lmax.
    """

    def __init__(
        self,
        col="metricdata",
        lmin=100.0,
        lmax=300.0,
        remove_dipole=True,
        mask_val=hp.UNSEEN,
        **kwargs
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.remove_dipole = remove_dipole
        super(TotalPowerMetric, self).__init__(col=col, mask_val=mask_val, **kwargs)

    def run(self, data_slice, slice_point=None):
        # Calculate the power spectrum.
        if self.remove_dipole:
            cl = hp.anafast(hp.remove_dipole(data_slice[self.colname], verbose=False))
        else:
            cl = hp.anafast(data_slice[self.colname])
        ell = np.arange(np.size(cl))
        condition = np.where((ell <= self.lmax) & (ell >= self.lmin))[0]
        totalpower = np.sum(cl[condition] * (2 * ell[condition] + 1))
        return totalpower


class StaticProbesFoMEmulatorMetricSimple(BaseMetric):
    """This calculates the Figure of Merit for the combined
    static probes (3x2pt, i.e., Weak Lensing, LSS, Clustering).
    This FoM is purely statistical and does not factor in systematics.

    This version of the emulator was used to generate the results in
    https://ui.adsabs.harvard.edu/abs/2018arXiv181200515L/abstract

    A newer version is being created. This version has been renamed
    Simple in anticipation of the newer, more sophisticated metric
    replacing it.

    Note that this is truly a summary metric and should be run on the output of
    Exgalm5_with_cuts.
    """

    def __init__(self, nside=128, year=10, col=None, **kwargs):

        """
        Args:
            nside (int): healpix resolution
            year (int): year of the FoM emulated values,
                can be one of [1, 3, 6, 10]
            col (str): column name of metric data.
        """
        self.nside = nside
        super().__init__(col=col, **kwargs)
        if col is None:
            self.col = "metricdata"
        self.year = year

    def run(self, data_slice, slice_point=None):
        """
        Args:
            data_slice (ndarray): Values passed to metric by the slicer,
                which the metric will use to calculate metric values
                at each slice_point.
            slice_point (Dict): Dictionary of slice_point metadata passed
                to each metric.
        Returns:
             float: Interpolated static-probe statistical Figure-of-Merit.
        Raises:
             ValueError: If year is not one of the 4 for which a FoM is calculated
        """
        # Chop off any outliers
        good_pix = np.where(data_slice[self.col] > 0)[0]

        # Calculate area and med depth from
        area = hp.nside2pixarea(self.nside, degrees=True) * np.size(good_pix)
        median_depth = np.median(data_slice[self.col][good_pix])

        # FoM is calculated at the following values
        if self.year == 1:
            areas = [7500, 13000, 16000]
            depths = [24.9, 25.2, 25.5]
            fom_arr = [
                [1.212257e02, 1.462689e02, 1.744913e02],
                [1.930906e02, 2.365094e02, 2.849131e02],
                [2.316956e02, 2.851547e02, 3.445717e02],
            ]
        elif self.year == 3:
            areas = [10000, 15000, 20000]
            depths = [25.5, 25.8, 26.1]
            fom_arr = [
                [1.710645e02, 2.246047e02, 2.431472e02],
                [2.445209e02, 3.250737e02, 3.516395e02],
                [3.173144e02, 4.249317e02, 4.595133e02],
            ]

        elif self.year == 6:
            areas = [10000, 15000, 2000]
            depths = [25.9, 26.1, 26.3]
            fom_arr = [
                [2.346060e02, 2.414678e02, 2.852043e02],
                [3.402318e02, 3.493120e02, 4.148814e02],
                [4.452766e02, 4.565497e02, 5.436992e02],
            ]

        elif self.year == 10:
            areas = [10000, 15000, 20000]
            depths = [26.3, 26.5, 26.7]
            fom_arr = [
                [2.887266e02, 2.953230e02, 3.361616e02],
                [4.200093e02, 4.292111e02, 4.905306e02],
                [5.504419e02, 5.624697e02, 6.441837e02],
            ]
        else:
            warnings.warn("FoMEmulator is not defined for this year")
            return self.badval

        # Interpolate FoM to the actual values for this sim
        areas = [[i] * 3 for i in areas]
        depths = [depths] * 3
        f = interpolate.interp2d(areas, depths, fom_arr, bounds_error=False)
        fom = f(area, median_depth)[0]
        return fom
