__all__ = ("optimal_bins",)

import warnings

import numpy as np


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
