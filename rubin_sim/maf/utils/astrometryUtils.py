import numpy as np
"""Some simple functions that are useful for astrometry calculations. """

__all__ = ['sigma_slope', 'm52snr', 'astrom_precision']

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
    w = 1./sigma_y**2
    denom = np.sum(w)*np.sum(w*x**2)-np.sum(w*x)**2
    if denom <= 0:
        return np.nan
    else:
        result = np.sqrt(np.sum(w)/denom )
        return result

def m52snr(m, m5):
    """
    Calculate the SNR for a star of magnitude m in an
    observation with 5-sigma limiting magnitude depth m5.
    Assumes gaussian distribution of photons and might not be
    strictly due in bluer filters. See table 2 and equation 5
    in astroph/0805.2366.

    Parameters
    ----------
    m : float or numpy.ndarray
        The magnitude of the star
    m5 : float or numpy.ndarray
        The m5 limiting magnitude of the observation

    Returns
    -------
    float or numpy.ndarray
        The SNR
    """
    snr = 5.*10.**(-0.4*(m-m5))
    return snr

def astrom_precision(fwhm, snr):
    """
    Calculate the approximate precision of astrometric measurements,
    given a particular seeing and SNR value.

    Parameters
    ----------
    fwhm : float or numpy.ndarray
        The seeing (FWHMgeom) of the observation.
    snr : float or numpy.ndarray
        The SNR of the object.

    Returns
    -------
    float or numpy.ndarray
        The astrometric precision.
    """
    result = fwhm/(snr)
    return result
