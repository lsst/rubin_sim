import numpy as np
from .sysEngVals import SysEngVals

__all__ = ['m5_flat_sed', 'm5_scale']


def m5_scale(expTime, nexp, airmass, FWHMeff, musky, darkSkyMag, Cm, dCm_infinity, kAtm,
             tauCloud=0, baseExpTime=15):
    """ Return m5 (scaled) value for all filters.

    Parameters
    ----------
    expTime : float
        Exposure time (in seconds) for each exposure
    nexp : int
        Number of exposures
    airmass : float
        Airmass of the observation
    FWHMeff : np.ndarray or pd.DataFrame
        FWHM (in arcseconds) per filter
    musky : np.ndarray or pd.DataFrame
        Sky background (in magnitudes/sq arcsecond) per filter of the observation
    darkSkyMag : np.ndarray or pd.DataFrame
        Dark Sky, zenith magnitude/sq arcsecond - to scale musky. per filter
    Cm : np.ndarray or pd.DataFrame
        Cm value for the throughputs per filter
    dCm_infinity : np.ndarray or pd.DataFrame
        dCm_infinity values for the throughputs, per filter
    kAtm : np.ndarray or pd.DataFrame
        Atmospheric extinction values, per filter
    tauCloud : float, opt
        Extinction due to clouds
    baseExpTime : float, opt
        The exposure time used to calculate Cm / dCm_infinity. Used to scale expTime.
        This is the individual exposure exposure time.

    Returns
    -------
    np.ndarray or pd.DataFrame
        m5 values scaled for the visit conditions

    Note: The columns required as input for m5_scale can be calculated using
    the makeM5 function in lsst.syseng.throughputs.
    """
    # Calculate adjustment if readnoise is significant for exposure time
    # (see overview paper, equation 7)
    Tscale = expTime / baseExpTime * np.power(10.0, -0.4 * (musky - darkSkyMag))
    dCm = 0.
    dCm += dCm_infinity
    dCm -= 1.25 * np.log10(1 + (10**(0.8 * dCm_infinity) - 1)/Tscale)
    # Calculate m5 for 1 exp - constants here come from definition of Cm/dCm_infinity
    m5 = (Cm + dCm + 0.50 * (musky - 21.0) + 2.5 * np.log10(0.7 / FWHMeff) +
          1.25 * np.log10(expTime / 30.0) - kAtm * (airmass - 1.0) - 1.1 * tauCloud)
    if nexp > 1:
        m5 = 1.25 * np.log10(nexp * 10**(0.8 * m5))
    return m5


def m5_flat_sed(visitFilter, musky, FWHMeff, expTime, airmass, nexp=1, tauCloud=0):
    """Calculate the m5 value, using photometric scaling.  Note, does not include shape of the object SED.

    Parameters
    ----------
    visitFilter : str
         One of u,g,r,i,z,y
    musky : float
        Surface brightness of the sky in mag/sq arcsec
    FWHMeff : float
        The seeing effective FWHM (arcsec)
    expTime : float
        Exposure time for each exposure in the visit.
    airmass : float
        Airmass of the observation (unitless)
    nexp : int, opt
        The number of exposures. Default 1.  (total on-sky time = expTime * nexp)
    tauCloud : float (0.)
        Any extinction from clouds in magnitudes (positive values = more extinction)

    Output
    ------
    m5 : float
        The five-sigma limiting depth of a point source observed in the given conditions.
    """

    # Set up expected extinction (kAtm) and m5 normalization values (Cm) for each filter.
    # The Cm values must be changed when telescope and site parameters are updated.
    #
    # These values are calculated using $SYSENG_THROUGHPUTS/python/calcM5.py.
    # This set of values are calculated using v1.2 of the SYSENG_THROUGHPUTS repo.
    # The exposure time scaling depends on knowing the value of the exposure time used to calculate Cm/etc.

    # Only define the dicts once on initial call
    if not hasattr(m5_flat_sed, 'Cm'):
        # Using Cm / dCm_infinity values calculated for a 1x30s visit.
        # This results in an error of about 0.01 mag in u band for 2x15s visits (< in other bands)
        # See https://github.com/lsst-pst/survey_strategy/blob/master/fbs_1.3/m5FlatSed%20update.ipynb
        # for a more in-depth evaluation.
        sev = SysEngVals()

        m5_flat_sed.baseExpTime = sev.exptime
        m5_flat_sed.Cm = sev.Cm
        m5_flat_sed.dCm_infinity = sev.dCm_infinity
        m5_flat_sed.kAtm = sev.kAtm
        m5_flat_sed.msky = sev.skyMag
    # Calculate adjustment if readnoise is significant for exposure time
    # (see overview paper, equation 7)
    Tscale = expTime / m5_flat_sed.baseExpTime * np.power(10.0, -0.4 * (musky - m5_flat_sed.msky[visitFilter]))
    dCm = 0.
    dCm += m5_flat_sed.dCm_infinity[visitFilter]
    dCm -= 1.25 * np.log10(1 + (10**(0.8 * m5_flat_sed.dCm_infinity[visitFilter]) - 1) / Tscale)
    # Calculate m5 for 1 exp - 30s and other constants here come from definition of Cm/dCm_infinity
    m5 = (m5_flat_sed.Cm[visitFilter] + dCm + 0.50 * (musky - 21.0) + 2.5 * np.log10(0.7 / FWHMeff) +
          1.25 * np.log10(expTime / 30.0) - m5_flat_sed.kAtm[visitFilter] * (airmass - 1.0) - 1.1 * tauCloud)
    # Then combine with coadd if >1 exposure
    if nexp > 1:
        m5 = 1.25 * np.log10(nexp * 10**(0.8 * m5))
    return m5
