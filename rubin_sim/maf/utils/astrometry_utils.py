__all__ = ("sigma_slope", "m52snr", "astrom_precision")

import numpy as np

"""Some simple functions that are useful for astrometry calculations. """


def sigma_slope(x, sigma_y):
    """
    Calculate the uncertainty in fitting a line, as
    given by the spread in x values and the uncertainties
    in the y values.

    Parameters
    ----------
    x : numpy.ndarray
        The x values of the data
    sigma_y : numpy.ndarray
        The uncertainty in the y values

    Returns
    -------
    float
        The uncertainty in the line fit
    """
    w = 1.0 / sigma_y**2
    denom = np.sum(w) * np.sum(w * x**2) - np.sum(w * x) ** 2
    if denom <= 0:
        return np.nan
    else:
        result = np.sqrt(np.sum(w) / denom)
        return result


def m52snr(m, m5, gamma=0.04):
    """
    Calculate the SNR for a star of magnitude m in an
    observation with 5-sigma limiting magnitude depth m5.
    Assumes gaussian distribution of photons and might not be
    strictly due in bluer filters. See table 2 and equation 5
    in astroph/0805.2366.

    Parameters
    ----------
    m : `float` or `np.ndarray` (N,)
        The magnitude of the star
    m5 : `float` or `np.ndarray` (N,)
        The m5 limiting magnitude of the observation
    gamma : `float` or None
        The 'gamma' value used when calculating photometric or
        astrometric errors and weighting SNR accordingly.
        See equation 5 of the LSST Overview paper.
        Use "None" to discount the gamma factor completely
        and use standard 5*10^(0.4 * (m5-m)).

    Returns
    -------
    snr : `float` or `np.ndarray` (N,)
        The SNR
    """
    # gamma varies per band, but is fairly close to 0.04

    if gamma is None:
        snr = 5.0 * 10.0 ** (-0.4 * (m - m5))
    else:
        xval = np.power(10, 0.4 * (m - m5))
        snr = 1 / np.sqrt((0.04 - gamma) * xval + gamma * xval * xval)
    return snr


def astrom_precision(fwhm, snr, systematic_floor=0.00):
    """
    Calculate the approximate precision of astrometric measurements,
    given a particular seeing and SNR value.

    Parameters
    ----------
    fwhm : `float` or `np.ndarray` (N,)
        The seeing (FWHMgeom) of the observation.
    snr : float` or `np.ndarray` (N,)
        The SNR of the object.
    systematic_floor : `float`
        Systematic noise floor for astrometric error, in arcseconds.
        Default here is 0, for backwards compatibility.
        General Rubin use should be 0.01.

    Returns
    -------
    astrom_err : `float` or `numpy.ndarray` (N,)
        The astrometric precision, in arcseconds.
    """
    astrom_err = np.sqrt((fwhm / snr) ** 2 + systematic_floor**2)
    return astrom_err
