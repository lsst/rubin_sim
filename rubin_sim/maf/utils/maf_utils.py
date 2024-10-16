__all__ = (
    "optimal_bins",
    "percentile_clipping",
    "radec2pix",
    "coadd_m5",
    "collapse_night",
    "load_inst_zeropoints",
)

import os
import warnings

import healpy as hp
import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SysEngVals, int_binned_stat
from scipy.stats import binned_statistic

from rubin_sim.phot_utils import Bandpass, PhotometricParameters


def load_inst_zeropoints():
    """Load up and return instrumental zeropoints and atmospheric
    extinctions.
    """
    zp_inst = {}
    datadir = get_data_dir()
    for filtername in "ugrizy":
        # set gain and exptime to 1 so the instrumental zeropoint will be in
        # photoelectrons and per second
        phot_params = PhotometricParameters(nexp=1, gain=1, exptime=1, bandpass=filtername)
        bp = Bandpass()
        bp.read_throughput(os.path.join(datadir, "throughputs/baseline/", "total_%s.dat" % filtername))
        zp_inst[filtername] = bp.calc_zp_t(phot_params)

    syseng = SysEngVals()
    k_atm = syseng.k_atm

    return zp_inst, k_atm


def coadd_m5(mags):
    """Coadded depth, assuming Gaussian noise."""
    return 1.25 * np.log10(np.sum(10.0 ** (0.8 * np.array(mags))))


def collapse_night(
    data_slice,
    night_col="night",
    filter_col="filter",
    m5_col="fiveSigmaDepth",
    mjd_col="observationStartMJD",
):
    """Collapse a data_slice into per-filter, per-night values for
    the 'night', 'filter', 'median observationStartMJD', and 'fiveSigmaDepth'.
    """
    filters = np.unique(data_slice[filter_col])
    # Find the per-night, per-filter values
    night_slice = {}
    for filtername in filters:
        infilt = np.where(data_slice[filter_col] == filtername)[0]
        unight = np.unique(data_slice[night_col][infilt])
        right = unight + 0.5
        bins = [unight[0] - 0.5] + right.tolist()
        coadds, be, bn = binned_statistic(
            data_slice[night_col][infilt],
            data_slice[m5_col][infilt],
            bins=bins,
            statistic=coadd_m5,
        )

        unights, median_mjd_per_night = int_binned_stat(
            data_slice[night_col][infilt],
            data_slice[mjd_col][infilt],
            statistic=np.median,
        )

        night_slice[filtername] = np.array(
            list(zip(unight, median_mjd_per_night, coadds, filtername * len(unight))),
            dtype=[
                (night_col, int),
                (mjd_col, float),
                (m5_col, float),
                (filter_col, "U1"),
            ],
        )

    night_slice = np.concatenate([night_slice[f] for f in night_slice])
    night_slice.sort(order=["observationStartMJD"])

    return night_slice


def optimal_bins(datain, binmin=None, binmax=None, nbin_max=200, nbin_min=1, verbose=False):
    """
    Set an 'optimal' number of bins using the Freedman-Diaconis rule.

    Parameters
    ----------
    datain : `numpy.ndarray` or `numpy.ma.MaskedArray`
        The data for which we want to set the bin_size.
    binmin : `float`
        The minimum bin value to consider (if None, uses minimum data value).
    binmax : `float`
        The maximum bin value to consider (if None, uses maximum data value).
    nbin_max : `int`
        The maximum number of bins to create.
        Sometimes the 'optimal bin_size' implies an unreasonably large number
        of bins, if the data distribution is unusual.
    nbin_min : `int`
        The minimum number of bins to create. Default is 1.
    verbose : `bool`
        Turn off warning messages. This utility very often raises warnings
        and these should likely be logging messages at a lower logging level,
        but for now - just use the verbose flag to turn these off or on.

    Returns
    -------
    nbins : `int`
        The number of bins.
    """
    # if it's a masked array, only use unmasked values
    if hasattr(datain, "compressed"):
        data = datain.compressed()
    else:
        data = datain
    # Check that any good data values remain.
    if data.size == 0:
        nbins = nbin_max
        if verbose:
            warnings.warn(
                f"No unmasked data available for calculating optimal bin size: returning {nbins} bins"
            )
    # Else proceed.
    else:
        if binmin is None:
            binmin = np.nanmin(data)
        if binmax is None:
            binmax = np.nanmax(data)
        cond = np.where((data >= binmin) & (data <= binmax))[0]
        # Check if any data points remain within binmin/binmax.
        if np.size(data[cond]) == 0:
            nbins = nbin_max
            warnings.warn(
                "No data available for calculating optimal bin size within range of %f, %f" % (binmin, binmax)
                + ": returning %i bins" % (nbins)
            )
        else:
            iqr = np.percentile(data[cond], 75) - np.percentile(data[cond], 25)
            binwidth = 2 * iqr * (np.size(data[cond]) ** (-1.0 / 3.0))
            nbins = (binmax - binmin) / binwidth
            if nbins > nbin_max:
                warnings.warn(
                    "Optimal bin calculation tried to make %.0f bins, returning %i" % (nbins, nbin_max)
                )
                nbins = nbin_max
            if nbins < nbin_min:
                warnings.warn(
                    "Optimal bin calculation tried to make %.0f bins, returning %i" % (nbins, nbin_min)
                )
                nbins = nbin_min
    if np.isnan(nbins):
        warnings.warn("Optimal bin calculation calculated NaN: returning %i" % (nbin_max))
        nbins = nbin_max
    return int(nbins)


def percentile_clipping(data, percentile=95.0):
    """Calculate the minimum and maximum values of a distribution of points,
    after discarding data more than 'percentile' from the median.
    This is useful for determining useful data ranges for plots.
    Note that 'percentile' percent of the data is retained.

    Parameters
    ----------
    data : `numpy.ndarray`
        The data to clip.
    percentile : `float`
        Retain values within percentile of the median.

    Returns
    -------
    minimum, maximum : `float`, `float`
        The minimum and maximum values of the clipped data.
    """
    lower_percentile = (100 - percentile) / 2.0
    upper_percentile = 100 - lower_percentile
    min_value = np.percentile(data, lower_percentile)
    max_value = np.percentile(data, upper_percentile)
    return min_value, max_value


def radec2pix(nside, ra, dec):
    """Calculate the nearest healpixel ID of an RA/Dec array, assuming nside.

    Parameters
    ----------
    nside : `int`
        The nside value of the healpix grid.
    ra : `numpy.ndarray`, (N,)
        The RA values to be converted to healpix ids, in radians.
    dec : `numpy.ndarray`, (N,)
        The Dec values to be converted to healpix ids, in radians.

    Returns
    -------
    hpid : `numpy.ndarray`, (N,)
        The healpix ids.
    """
    lat = np.pi / 2.0 - dec
    hpid = hp.ang2pix(nside, lat, ra)
    return hpid
