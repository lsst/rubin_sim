import numpy as np
import healpy as hp
from scipy import interpolate
from .baseMetric import BaseMetric

# A collection of metrics which are primarily intended to be used as summary statistics.

__all__ = ['fOArea', 'fONv', 'TableFractionMetric', 'IdentityMetric',
           'NormalizeMetric', 'ZeropointMetric', 'TotalPowerMetric',
           'StaticProbesFoMEmulatorMetricSimple']


class fONv(BaseMetric):
    """
    Metrics based on a specified area, but returning NVISITS related to area:
    given Asky, what is the minimum and median number of visits obtained over that much area?
    (choose the portion of the sky with the highest number of visits first).

    Parameters
    ----------
    col : str or list of strs, opt
        Name of the column in the numpy recarray passed to the summary metric.
    Asky : float, opt
        Area of the sky to base the evaluation of number of visits over.
        Default 18,0000 sq deg.
    nside : int, opt
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    Nvisit : int, opt
        Number of visits to use as the benchmark value, if choosing to return a normalized Nvisit value.
    norm : boolean, opt
        Normalize the returned "nvisit" (min / median) values by Nvisit, if true.
        Default False.
    metricName : str, opt
        Name of the summary metric. Default fONv.
    """
    def __init__(self, col='metricdata', Asky=18000., nside=128, Nvisit=825,
                 norm=False, metricName='fONv',  **kwargs):
        """Asky = square degrees """
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.Nvisit = Nvisit
        self.nside = nside
        # Determine how many healpixels are included in Asky sq deg.
        self.Asky = Asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.npix_Asky = int(np.ceil(self.Asky / self.scale))
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        result = np.empty(2, dtype=[('name', np.str_, 20), ('value', float)])
        result['name'][0] = "MedianNvis"
        result['name'][1] = "MinNvis"
        # If there is not even as much data as needed to cover Asky:
        if len(dataSlice) < self.npix_Asky:
            # Return the same type of metric value, to make it easier downstream.
            result['value'][0] = self.badval
            result['value'][1] = self.badval
            return result
        # Otherwise, calculate median and mean Nvis:
        name = dataSlice.dtype.names[0]
        nvis_sorted = np.sort(dataSlice[name])
        # Find the Asky's worth of healpixels with the largest # of visits.
        nvis_Asky = nvis_sorted[-self.npix_Asky:]
        result['value'][0] = np.median(nvis_Asky)
        result['value'][1] = np.min(nvis_Asky)
        if self.norm:
            result['value'] /= float(self.Nvisit)
        return result


class fOArea(BaseMetric):
    """
    Metrics based on a specified number of visits, but returning AREA related to Nvisits:
    given Nvisit, what amount of sky is covered with at least that many visits?

    Parameters
    ----------
    col : str or list of strs, opt
        Name of the column in the numpy recarray passed to the summary metric.
    Nvisit : int, opt
        Number of visits to use as the minimum required -- metric calculated area that has this many visits.
        Default 825.
    Asky : float, opt
        Area to use as the benchmark value, if choosing to returned a normalized Area value.
        Default 18,0000 sq deg.
    nside : int, opt
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    norm : boolean, opt
        Normalize the returned "area" (area with minimum Nvisit visits) value by Asky, if true.
        Default False.
    metricName : str, opt
        Name of the summary metric. Default fOArea.
    """
    def __init__(self, col='metricdata', Nvisit=825, Asky = 18000.0, nside=128,
                  norm=False, metricName='fOArea',  **kwargs):
        """Asky = square degrees """
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.Nvisit = Nvisit
        self.nside = nside
        self.Asky = Asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        name = dataSlice.dtype.names[0]
        nvis_sorted = np.sort(dataSlice[name])
        # Identify the healpixels with more than Nvisits.
        nvis_min = nvis_sorted[np.where(nvis_sorted >= self.Nvisit)]
        if len(nvis_min) == 0:
            result = self.badval
        else:
            result = nvis_min.size * self.scale
            if self.norm:
                result /= float(self.Asky)
        return result


class TableFractionMetric(BaseMetric):
    """
    Count the completeness (for many fields) and summarize how many fields have given completeness levels
    (within a series of bins). Works with completenessMetric only.

    This metric is meant to be used as a summary statistic on something like the completeness metric.
    The output is DIFFERENT FROM SSTAR and is:
    element   matching values
    0         0 == P
    1         0 < P < .1
    2         .1 <= P < .2
    3         .2 <= P < .3
    ...
    10        .9 <= P < 1
    11        1 == P
    12        1 < P
    Note the 1st and last elements do NOT obey the numpy histogram conventions.
    """
    def __init__(self, col='metricdata',  nbins=10, maskVal=0.):
        """
        colname = the column name in the metric data (i.e. 'metricdata' usually).
        nbins = number of bins between 0 and 1. Should divide evenly into 100.
        """
        super(TableFractionMetric, self).__init__(col=col, maskVal=maskVal, metricDtype='float')
        self.nbins = nbins

    def run(self, dataSlice, slicePoint=None):
        # Calculate histogram of completeness values that fall between 0-1.
        goodVals = np.where((dataSlice[self.colname] > 0) & (dataSlice[self.colname] < 1)  )
        bins = np.arange(self.nbins+1.)/self.nbins
        hist, b = np.histogram(dataSlice[self.colname][goodVals], bins=bins)
        # Fill in values for exact 0, exact 1 and >1.
        zero = np.size(np.where(dataSlice[self.colname] == 0)[0])
        one = np.size(np.where(dataSlice[self.colname] == 1)[0])
        overone = np.size(np.where(dataSlice[self.colname] > 1)[0])
        hist = np.concatenate((np.array([zero]), hist, np.array([one]), np.array([overone])))
        # Create labels for each value
        binNames = ['0 == P']
        binNames.append('0 < P < 0.1')
        for i in np.arange(1, self.nbins):
            binNames.append('%.2g <= P < %.2g'%(b[i], b[i+1]) )
        binNames.append('1 == P')
        binNames.append('1 < P')
        # Package the names and values up
        result = np.empty(hist.size, dtype=[('name', np.str_, 20), ('value', float)])
        result['name'] = binNames
        result['value'] = hist
        return result


class IdentityMetric(BaseMetric):
    """
    Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics.
    """
    def run(self, dataSlice, slicePoint=None):
        if len(dataSlice[self.colname]) == 1:
            result = dataSlice[self.colname][0]
        else:
            result = dataSlice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """
    Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions.
    """
    def __init__(self, col='metricdata', normVal=1, **kwargs):
        super(NormalizeMetric, self).__init__(col=col, **kwargs)
        self.normVal = float(normVal)
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname]/self.normVal
        if len(result) == 1:
            return result[0]
        else:
            return result

class ZeropointMetric(BaseMetric):
    """
    Return a metric values with the addition of 'zp'. Useful for altering the zeropoint for summary statistics.
    """
    def __init__(self, col='metricdata', zp=0, **kwargs):
        super(ZeropointMetric, self).__init__(col=col, **kwargs)
        self.zp = zp
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname] + self.zp
        if len(result) == 1:
            return result[0]
        else:
            return result

class TotalPowerMetric(BaseMetric):
    """
    Calculate the total power in the angular power spectrum between lmin/lmax.
    """
    def __init__(self, col='metricdata', lmin=100., lmax=300., removeDipole=True, maskVal=hp.UNSEEN, **kwargs):
        self.lmin = lmin
        self.lmax = lmax
        self.removeDipole = removeDipole
        super(TotalPowerMetric, self).__init__(col=col, maskVal=maskVal, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Calculate the power spectrum.
        if self.removeDipole:
            cl = hp.anafast(hp.remove_dipole(dataSlice[self.colname], verbose=False))
        else:
            cl = hp.anafast(dataSlice[self.colname])
        ell = np.arange(np.size(cl))
        condition = np.where((ell <= self.lmax) & (ell >= self.lmin))[0]
        totalpower = np.sum(cl[condition]*(2*ell[condition]+1))
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
            self.col = 'metricdata'
        self.year = year

    def run(self, dataSlice, slicePoint=None):
        """
        Args:
            dataSlice (ndarray): Values passed to metric by the slicer, 
                which the metric will use to calculate metric values 
                at each slicePoint.
            slicePoint (Dict): Dictionary of slicePoint metadata passed
                to each metric.
        Returns:
             float: Interpolated static-probe statistical Figure-of-Merit.
        Raises:
             ValueError: If year is not one of the 4 for which a FoM is calculated
        """
        # Chop off any outliers
        good_pix = np.where(dataSlice[self.col] > 0)[0]
        
        # Calculate area and med depth from
        area = hp.nside2pixarea(self.nside, degrees=True) * np.size(good_pix)
        median_depth = np.median(dataSlice[self.col][good_pix])

        # FoM is calculated at the following values
        if self.year == 1:
            areas = [7500, 13000, 16000]
            depths = [24.9, 25.2, 25.5]
            fom_arr = [
                [1.212257e+02, 1.462689e+02, 1.744913e+02],
                [1.930906e+02, 2.365094e+02, 2.849131e+02],
                [2.316956e+02, 2.851547e+02, 3.445717e+02]
            ]
        elif self.year == 3:
            areas = [10000, 15000, 20000]
            depths = [25.5, 25.8, 26.1]
            fom_arr = [
                [1.710645e+02, 2.246047e+02, 2.431472e+02],
                [2.445209e+02, 3.250737e+02, 3.516395e+02],
                [3.173144e+02, 4.249317e+02, 4.595133e+02]
            ]

        elif self.year == 6:
            areas = [10000, 15000, 2000]
            depths = [25.9, 26.1, 26.3]
            fom_arr = [
                [2.346060e+02, 2.414678e+02, 2.852043e+02],
                [3.402318e+02, 3.493120e+02, 4.148814e+02],
                [4.452766e+02, 4.565497e+02, 5.436992e+02]
            ]

        elif self.year == 10:
            areas = [10000, 15000, 20000]
            depths = [26.3, 26.5, 26.7]
            fom_arr = [
                [2.887266e+02, 2.953230e+02, 3.361616e+02],
                [4.200093e+02, 4.292111e+02, 4.905306e+02],
                [5.504419e+02, 5.624697e+02, 6.441837e+02]
            ]
        else:
            raise ValueError('FoMEmulator is not defined for this year')

        # Interpolate FoM to the actual values for this sim
        areas = [[i]*3 for i in areas]
        depths = [depths]*3
        f = interpolate.interp2d(areas, depths, fom_arr, bounds_error=False)
        fom = f(area, median_depth)[0]
        return fom

