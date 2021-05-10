import numpy as np
import ephem


def wrapRA(ra):
    """
    Wrap only RA values into 0-2pi (using mod).
    """
    ra = ra % (2.0*np.pi)
    return ra


def mjd2djd(inDate):
    """
    Convert Modified Julian Date to Dublin Julian Date (what pyephem uses).
    """
    if not hasattr(mjd2djd, 'doff'):
        mjd2djd.doff = ephem.Date(0)-ephem.Date('1858/11/17')
    djd = inDate-mjd2djd.doff
    return djd


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


def ut2Mjd(dateString):
    obs = ephem.Observer()
    obs.date = dateString
    doff = ephem.Date(0)-ephem.Date('1858/11/17')
    mjd = obs.date+doff
    return mjd


def mjd2ut(mjd):
    obs = ephem.Observer()
    doff = ephem.Date(0)-ephem.Date('1858/11/17')
    djd = mjd-doff
    obs.date = djd
    return obs.date
