import numpy as np


def wrapRA(ra):
    """
    Wrap only RA values into 0-2pi (using mod).
    """
    ra = ra % (2.0*np.pi)
    return ra


def robustRMS(array, missing=0.):
    """
    Use the interquartile range to compute a robust approximation of the RMS.
    if passed an array smaller than 2 elements, return missing value
    """
    if np.size(array) < 2:
        rms = missing
    else:
        iqr = np.percentile(array, 75)-np.percentile(array, 25)
        rms = iqr/1.349  # approximation
    return rms
