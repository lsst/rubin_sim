__all__ = (
    "optimal_bins",
    "gen_summary_row",
    "open_shutter_fraction",
    "count_value_changes",
    "osf_visit_array",
    "fO_calcs",
    "percentile_clipping",
    "fO_time_calcs",
)

import copy
import warnings

import healpy as hp
import numpy as np


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


def fO_calcs(nvis_hp, asky=18000.0, n_visit=750):
    """ """
    nside = hp.npix2nside(nvis_hp.size)
    scale = hp.nside2pixarea(nside, degrees=True)
    npix_asky = int(np.ceil(asky / scale))

    nvis_sorted = np.sort(nvis_hp)
    nvis_sorted = nvis_sorted[np.isfinite(nvis_sorted)]
    if nvis_sorted.size >= npix_asky:
        nvis_asky = nvis_sorted[-npix_asky:]
    else:
        nvis_asky = 0

    result = {"Median N visits in top %ik sq deg" % (asky / 1e3): np.median(nvis_asky)}
    result["Min N visits in top %ik sq deg" % (asky / 1e3)] = np.min(nvis_asky)

    nvis_min = np.where(nvis_sorted >= n_visit)[0]
    result["Area above %i (sq deg)" % n_visit] = nvis_min.size * scale

    return result


def fO_time_calcs(nvis_hp_time, asky=18000.0, n_visit=750, stat=np.median):
    """Given n visits in time array, convert to fO vs time."""
    n_pix_heal = nvis_hp_time[:, 0].size
    nside = hp.npix2nside(n_pix_heal)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    n_pix_needed = int(np.ceil(asky / pix_area))
    # sort by value
    data = nvis_hp_time
    data.sort(axis=0)
    # Crop down to the desired sky area
    data = data[n_pix_heal - n_pix_needed :, :]
    result = stat(data, axis=0)
    return result


def count_value_changes(inarr):
    return np.sum(inarr[1:].values != inarr[:-1].values)


def open_shutter_fraction(exposure_start_mjd, exposure_times, max_gap=10.0):
    max_gap = max_gap / 60.0 / 24.0  # convert from min to days
    times = np.sort(exposure_start_mjd)
    diff = np.diff(times)
    good = np.where(diff < max_gap)
    open_time = np.sum(diff[good]) * 24.0 * 3600.0
    result = np.sum(exposure_times) / float(open_time)
    return result


def osf_visit_array(in_array):
    """Wrapper to make it easy to pandas groupby"""
    return open_shutter_fraction(in_array["observationStartMJD"], in_array["visitExposureTime"])


def gen_summary_row(info, summary_name, value):
    summary = copy.copy(info)
    summary["summary_name"] = summary_name
    summary["value"] = value
    return summary


def optimal_bins(datain, binmin=None, binmax=None, nbin_max=200, nbin_min=1, verbose=False):
    """
    Set an 'optimal' number of bins using the Freedman-Diaconis rule.

    Parameters
    ----------
    datain : `numpy.ndarray` or `numpy.ma.MaskedArray`
        The data for which we want to set the bin_size.
    binmin : `float`
        The minimum bin value to consider. Default None uses
        minimum data value.
    binmax : `float`
        The maximum bin value to consider. Default None uses
        maximum data value.
    nbin_max : `int`
        The maximum number of bins to create.
        Sometimes the 'optimal bin_size' implies an unreasonably
        large number of bins, if the data distribution is
        unusual. Default 200.
    nbin_min : `int`
        The minimum number of bins to create. Default is 1.
    verbose : `bool`
        Turn on warning messages. Default False.

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
                if verbose:
                    warnings.warn(
                        "Optimal bin calculation tried to make %.0f bins, returning %i" % (nbins, nbin_max)
                    )
                nbins = nbin_max
            if nbins < nbin_min:
                if verbose:
                    warnings.warn(
                        "Optimal bin calculation tried to make %.0f bins, returning %i" % (nbins, nbin_min)
                    )
                nbins = nbin_min
    if np.isnan(nbins):
        if verbose:
            warnings.warn("Optimal bin calculation calculated NaN: returning %i" % (nbin_max))
        nbins = nbin_max
    return int(nbins)
