import numpy as np
import healpy as hp
import warnings
from scipy.stats import binned_statistic
from rubin_sim.utils import int_binned_stat


__all__ = ['optimalBins', 'percentileClipping',
           'gnomonic_project_toxy', 'radec2pix', 'collapse_night']


def coaddM5(mags):
    """Coadded depth, assuming Gaussian noise
    """
    return 1.25 * np.log10(np.sum(10.**(0.8*np.array(mags))))


def collapse_night(dataSlice, nightCol='night', filterCol='filter',
                   m5Col='fiveSigmaDepth', mjdCol='observationStartMJD'):
    """Collapse a dataSlice into per-filter, per-night values for
    the 'night', 'filter', 'median observationStartMJD', and 'fiveSigmaDepth'.
    """
    filters = np.unique(dataSlice[filterCol])
    # Find the per-night, per-filter values
    nightSlice = {}
    for filtername in filters:
        infilt = np.where(dataSlice[filterCol] == filtername)[0]
        unight = np.unique(dataSlice[nightCol][infilt])
        right = unight+0.5
        bins = [unight[0]-0.5] + right.tolist()
        coadds, be, bn = binned_statistic(dataSlice[nightCol][infilt],
                                          dataSlice[m5Col][infilt], bins=bins,
                                          statistic=coaddM5)

        unights, median_mjd_per_night = int_binned_stat(dataSlice[nightCol][infilt],
                                                        dataSlice[mjdCol][infilt],
                                                        statistic=np.median)

        nightSlice[filtername] = np.array(list(zip(unight, median_mjd_per_night,
                                                   coadds, filtername * len(unight))),
                                          dtype=[(nightCol, int), (mjdCol, float),
                                                 (m5Col, float), (filterCol, 'U1')])

    nightSlice = np.concatenate([nightSlice[f] for f in nightSlice])
    nightSlice.sort(order=['observationStartMJD'])

    return nightSlice


def optimalBins(datain, binmin=None, binmax=None, nbinMax=200, nbinMin=1):
    """
    Set an 'optimal' number of bins using the Freedman-Diaconis rule.

    Parameters
    ----------
    datain : numpy.ndarray or numpy.ma.MaskedArray
        The data for which we want to set the binsize.
    binmin : float
        The minimum bin value to consider (if None, uses minimum data value).
    binmax : float
        The maximum bin value to consider (if None, uses maximum data value).
    nbinMax : int
        The maximum number of bins to create. Sometimes the 'optimal binsize' implies
        an unreasonably large number of bins, if the data distribution is unusual.
    nbinMin : int
        The minimum number of bins to create. Default is 1.

    Returns
    -------
    int
        The number of bins.
    """
    # if it's a masked array, only use unmasked values
    if hasattr(datain, 'compressed'):
        data = datain.compressed()
    else:
        data = datain
    # Check that any good data values remain.
    if data.size == 0:
        nbins = nbinMax
        warnings.warn('No unmasked data available for calculating optimal bin size: returning %i bins' %(nbins))
    # Else proceed.
    else:
        if binmin is None:
            binmin = np.nanmin(data)
        if binmax is None:
            binmax = np.nanmax(data)
        cond = np.where((data >= binmin)  & (data <= binmax))[0]
        # Check if any data points remain within binmin/binmax.
        if np.size(data[cond]) == 0:
            nbins = nbinMax
            warnings.warn('No data available for calculating optimal bin size within range of %f, %f'
                          %(binmin, binmax) + ': returning %i bins' %(nbins))
        else:
            iqr = np.percentile(data[cond], 75) - np.percentile(data[cond], 25)
            binwidth = 2 * iqr * (np.size(data[cond])**(-1./3.))
            nbins = (binmax - binmin) / binwidth
            if nbins > nbinMax:
                warnings.warn('Optimal bin calculation tried to make %.0f bins, returning %i'%(nbins, nbinMax))
                nbins = nbinMax
            if nbins < nbinMin:
                warnings.warn('Optimal bin calculation tried to make %.0f bins, returning %i'%(nbins, nbinMin))
                nbins = nbinMin
    if np.isnan(nbins):
        warnings.warn('Optimal bin calculation calculated NaN: returning %i' %(nbinMax))
        nbins = nbinMax
    return int(nbins)


def percentileClipping(data, percentile=95.):
    """
    Calculate the minimum and maximum values of a distribution of points, after
    discarding data more than 'percentile' from the median.
    This is useful for determining useful data ranges for plots.
    Note that 'percentile' percent of the data is retained.

    Parameters
    ----------
    data : numpy.ndarray
        The data to clip.
    percentile : float
        Retain values within percentile of the median.

    Returns
    -------
    float, float
        The minimum and maximum values of the clipped data.
    """
    lower_percentile = (100 - percentile) / 2.0
    upper_percentile = 100 - lower_percentile
    min_value = np.percentile(data, lower_percentile)
    max_value = np.percentile(data, upper_percentile)
    return  min_value, max_value

def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """
    Calculate the x/y values of RA1/Dec1 in a gnomonic projection with center at RAcen/Deccen.

    Parameters
    ----------
    RA1 : numpy.ndarray
        RA values of the data to be projected, in radians.
    Dec1 : numpy.ndarray
        Dec values of the data to be projected, in radians.
    RAcen: float
        RA value of the center of the projection, in radians.
    Deccen : float
        Dec value of the center of the projection, in radians.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The x/y values of the projected RA1/Dec1 positions.
    """
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y


def radec2pix(nside, ra, dec):
    """
    Calculate the nearest healpixel ID of an RA/Dec array, assuming nside.

    Parameters
    ----------
    nside : int
        The nside value of the healpix grid.
    ra : numpy.ndarray
        The RA values to be converted to healpix ids, in radians.
    dec : numpy.ndarray
        The Dec values to be converted to healpix ids, in radians.

    Returns
    -------
    numpy.ndarray
        The healpix ids.
    """
    lat = np.pi/2. - dec
    hpid = hp.ang2pix(nside, lat, ra )
    return hpid
