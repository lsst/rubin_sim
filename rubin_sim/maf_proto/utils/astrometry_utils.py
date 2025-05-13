__all__ = ("sigma_slope", "m52snr", "astrom_precision", "parallax_amplitude", "dcr_amplitude")

import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, SkyCoord
from astropy.time import Time
from rubin_scheduler.utils import gnomonic_project_toxy


def dcr_amplitude(zenith_distance, parallactic_angle, filters, dcr_magnitudes=None, degrees=True):
    """ """

    if dcr_magnitudes is None:
        # DCR amplitudes are in arcseconds.
        dcr_magnitudes = {
            "u": 0.07,
            "g": 0.07,
            "r": 0.050,
            "i": 0.045,
            "z": 0.042,
            "y": 0.04,
        }

    if degrees:
        zenith_tan = np.tan(np.radians(zenith_distance))
        parallactic_angle = np.radians(parallactic_angle)
    else:
        zenith_tan = np.tan(zenith_distance)
    dcr_in_ra = zenith_tan * np.sin(parallactic_angle)
    dcr_in_dec = zenith_tan * np.cos(parallactic_angle)
    for filtername in np.unique(filters):
        fmatch = np.where(filters == filtername)
        dcr_in_ra[fmatch] = dcr_magnitudes[filtername] * dcr_in_ra[fmatch]
        dcr_in_dec[fmatch] = dcr_magnitudes[filtername] * dcr_in_dec[fmatch]
    return dcr_in_ra, dcr_in_dec


def parallax_amplitude(ra, dec, mjd, degrees=True):
    """Compute the parallax amplitude for a visit

    Parameters
    ----------
    ra : `float`
        RA of the points
    dec : `float`
        Dec of the points
    mjd : `float`
        MJD for the points
    degrees : `bool`
        If True, ra and dec are considered to be degrees.
        If False, assumed to be in radians. Default True.

    Returns
    -------
    ra_pi_amp : `float`
        Amplitude of parallax in RA direction (arcseconds)
    dec_pi_amp : `float`
        Amplitude of parallax in Dec direction (arcseconds)
    """
    if degrees:
        ra = np.radians(ra)
        dec = np.radians(dec)

    times = Time(mjd, format="mjd")
    c = SkyCoord(ra * u.rad, dec * u.rad, obstime=times)
    geo_far = c.transform_to(GCRS)
    c_near = SkyCoord(ra * u.rad, dec * u.rad, distance=1 * u.pc, obstime=times)
    geo_near = c_near.transform_to(GCRS)

    x_geo1, y_geo1 = gnomonic_project_toxy(geo_near.ra.rad, geo_near.dec.rad, ra, dec)
    x_geo, y_geo = gnomonic_project_toxy(geo_far.ra.rad, geo_far.dec.rad, ra, dec)

    # Return ra_pi_amp and dec_pi_amp in arcseconds.
    ra_pi_amp = np.degrees(x_geo1 - x_geo) * 3600.0
    dec_pi_amp = np.degrees(y_geo1 - y_geo) * 3600.0
    return ra_pi_amp, dec_pi_amp


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
