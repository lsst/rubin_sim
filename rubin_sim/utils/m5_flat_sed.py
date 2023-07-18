__all__ = ("m5_flat_sed", "m5_scale")

import numpy as np

from .sys_eng_vals import SysEngVals


def m5_scale(
    exp_time,
    nexp,
    airmass,
    fwhm_eff,
    musky,
    dark_sky_mag,
    cm,
    d_cm_infinity,
    k_atm,
    tau_cloud=0,
    base_exp_time=15,
):
    """Return m5 (scaled) value for all filters.

    Parameters
    ----------
    exp_time : float
        Exposure time (in seconds) for each exposure
    nexp : int
        Number of exposures
    airmass : float
        Airmass of the observation
    fwhm_eff : np.ndarray or pd.DataFrame
        FWHM (in arcseconds) per filter
    musky : np.ndarray or pd.DataFrame
        Sky background (in magnitudes/sq arcsecond) per filter of the observation
    dark_sky_mag : np.ndarray or pd.DataFrame
        Dark Sky, zenith magnitude/sq arcsecond - to scale musky. per filter
    cm : np.ndarray or pd.DataFrame
        cm value for the throughputs per filter
    d_cm_infinity : np.ndarray or pd.DataFrame
        d_cm_infinity values for the throughputs, per filter
    k_atm : np.ndarray or pd.DataFrame
        Atmospheric extinction values, per filter
    tau_cloud : float, optional
        Extinction due to clouds
    base_exp_time : float, optional
        The exposure time used to calculate cm / d_cm_infinity. Used to scale exp_time.
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
    tscale = exp_time / base_exp_time * np.power(10.0, -0.4 * (musky - dark_sky_mag))
    d_cm = 0.0
    d_cm += d_cm_infinity
    d_cm -= 1.25 * np.log10(1 + (10 ** (0.8 * d_cm_infinity) - 1) / tscale)
    # Calculate m5 for 1 exp - constants here come from definition of cm/d_cm_infinity
    m5 = (
        cm
        + d_cm
        + 0.50 * (musky - 21.0)
        + 2.5 * np.log10(0.7 / fwhm_eff)
        + 1.25 * np.log10(exp_time / 30.0)
        - k_atm * (airmass - 1.0)
        - 1.1 * tau_cloud
    )
    if nexp > 1:
        m5 = 1.25 * np.log10(nexp * 10 ** (0.8 * m5))
    return m5


def m5_flat_sed(visit_filter, musky, fwhm_eff, exp_time, airmass, nexp=1, tau_cloud=0):
    """Calculate the m5 value, using photometric scaling.  Note, does not include shape of the object SED.

    Parameters
    ----------
    visit_filter : str
         One of u,g,r,i,z,y
    musky : float
        Surface brightness of the sky in mag/sq arcsec
    fwhm_eff : float
        The seeing effective FWHM (arcsec)
    exp_time : float
        Exposure time for each exposure in the visit.
    airmass : float
        Airmass of the observation (unitless)
    nexp : int, optional
        The number of exposures. Default 1.  (total on-sky time = exp_time * nexp)
    tau_cloud : float (0.)
        Any extinction from clouds in magnitudes (positive values = more extinction)

    Returns
    -------
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
    if not hasattr(m5_flat_sed, "Cm"):
        # Using Cm / d_cm_infinity values calculated for a 1x30s visit.
        # This results in an error of about 0.01 mag in u band for 2x15s visits (< in other bands)
        # See https://github.com/lsst-pst/survey_strategy/blob/master/fbs_1.3/m5FlatSed%20update.ipynb
        # for a more in-depth evaluation.
        sev = SysEngVals()

        m5_flat_sed.base_exp_time = sev.exptime
        m5_flat_sed.cm = sev.cm
        m5_flat_sed.d_cm_infinity = sev.d_cm_infinity
        m5_flat_sed.k_atm = sev.k_atm
        m5_flat_sed.msky = sev.sky_mag
    # Calculate adjustment if readnoise is significant for exposure time
    # (see overview paper, equation 7)
    tscale = (
        exp_time / m5_flat_sed.base_exp_time * np.power(10.0, -0.4 * (musky - m5_flat_sed.msky[visit_filter]))
    )
    d_cm = 0.0
    d_cm += m5_flat_sed.d_cm_infinity[visit_filter]
    d_cm -= 1.25 * np.log10(1 + (10 ** (0.8 * m5_flat_sed.d_cm_infinity[visit_filter]) - 1) / tscale)
    # Calculate m5 for 1 exp - 30s and other constants here come from definition of Cm/d_cm_infinity
    m5 = (
        m5_flat_sed.cm[visit_filter]
        + d_cm
        + 0.50 * (musky - 21.0)
        + 2.5 * np.log10(0.7 / fwhm_eff)
        + 1.25 * np.log10(exp_time / 30.0)
        - m5_flat_sed.k_atm[visit_filter] * (airmass - 1.0)
        - 1.1 * tau_cloud
    )
    # Then combine with coadd if >1 exposure
    if nexp > 1:
        m5 = 1.25 * np.log10(nexp * 10 ** (0.8 * m5))
    return m5
