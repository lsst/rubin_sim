__all__ = ("saturation_limit",)

import numpy as np
from rubin_sim.maf.utils import load_inst_zeropoints


def saturation_limit(
    visits,
    zeropoints=None,
    km=None,
    band_col="band",
    pixscale=0.2,
    saturation_e=150e3,
    seeing_col="seeingFwhmEff",
    skybrightness_col="skyBrightness",
    exptime_col="visitExposureTime",
    nexp_col="numExposures",
    airmass_col="airmass",
):
    """Calculate point-source saturation limit for each visit.

    Assumes Gaussian PSF.

    Parameters
    ----------
    visits : `np.array`
        The visits to calculate
    pixscale : `float`, optional
        Arcsec per pixel.
    saturation_e : `float`, optional
        The saturation level in electrons.
    zeropoints : dict-like, optional
        The zeropoints for the telescope.
        Keys should be str with filter names, values in mags.
        Default of None, will use Rubin calculated zeropoints.
    km : dict-like, optional
        Atmospheric extinction values.
        Keys should be str with filter names.
        If None, will use Rubin calculated atmospheric extinction values.
    """

    if zeropoints is None:
        zp_inst, k_atm = load_inst_zeropoints()
        zeropoints = zp_inst
    if km is None:
        km = k_atm

    saturation_mag = np.empty(visits.size)

    for filtername in np.unique(visits[band_col]):
        in_filt = np.where(visits[band_col] == filtername)[0]
        # Calculate the length of the on-sky time per EXPOSURE
        exptime = visits[exptime_col][in_filt] / visits[nexp_col][in_filt]
        # Calculate sky counts per pixel per second
        # from skybrightness + zeropoint (e/1s)
        sky_counts = (
            10.0 ** (0.4 * (zeropoints[filtername] - visits[skybrightness_col][in_filt])) * pixscale**2
        )
        # Total sky counts in each exposure
        sky_counts = sky_counts * exptime
        # The counts available to the source (at peak) in each exposure is
        # the difference between saturation and sky
        remaining_counts_peak = saturation_e - sky_counts
        # Now to figure out how many counts there would be total, if there
        # are that many in the peak
        sigma = visits[seeing_col][in_filt] / 2.354
        source_counts = remaining_counts_peak * 2.0 * np.pi * (sigma / pixscale) ** 2
        # source counts = counts per exposure (expTimeCol / nexp)
        # Translate to counts per second, to apply zeropoint
        count_rate = source_counts / exptime
        saturation_mag[in_filt] = -2.5 * np.log10(count_rate) + zeropoints[filtername]
        # Airmass correction
        saturation_mag[in_filt] -= km[filtername] * (visits[airmass_col][in_filt] - 1.0)
        # Explicitly make sure if sky has saturated we return NaN
        saturation_mag[np.where(remaining_counts_peak < 0)] = np.nan

    return saturation_mag
