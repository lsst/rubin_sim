__all__ = ("SurfaceBrightLimitMetric",)

import warnings

import numpy as np

from rubin_sim.maf.utils import load_inst_zeropoints

from .base_metric import BaseMetric


def surface_brightness_limit_approx(
    zp,
    k,
    airmass,
    mu_sky,
    rn=8.8,
    pixscale=0.2,
    nsigma=3.0,
    t_exp=30.0,
    tot_area=100.0,
    mag_diff_warn=0.1,
):
    """Compute surface brightness limit in 3 limiting cases, return the
    brightest.

    Algerbra worked out in this technote:
    https://github.com/lsst-sims/smtn-016

    Parameters
    ---------
    zp : `float`
        Telescope zeropoint (mags)
    k : `float`
        Atmospheric extinction term
    airmass : `float`
        Airmass
    mu_sky : `float`
        Surface brightness of the sky (mag/sq arcsec)
    rn : `float` (8.8)
        Readnoise in electrons
    pixscale : `float` (0.2)
        Arcseconds per pixel
    nsigma : `float` (3)
        The SNR to demand
    t_exp : `float` (30.)
        Exposure time (seconds)
    tot_area : `float` (100)
        The total area measuring over (sq arcsec)
    mag_diff_warn : `float` (0.1)
        If the limiting cases are within mag_diff_warn, throw a warning
        that the surface brightness limit may be an overestimate.

    Returns
    -------
    surface brightness limit in mags/sq arcsec, aka the surface brightness that
    reaches SNR=nsigma when measured over tot_area.
    """

    a_pix = pixscale**2

    n_pix = tot_area / a_pix

    # Sky limited case
    mu_sky_lim = -1.25 * np.log10(nsigma**2 / (a_pix * t_exp * n_pix)) + 0.5 * mu_sky + 0.5 * zp - k * airmass

    # Source limited case
    # XXX--double check this algerbra. Pretty sure it's right now.
    mu_source_lim = -5 * np.log10(nsigma) + 2.5 * np.log10(n_pix * a_pix * t_exp) + zp - k * airmass

    # Readnoise limited case
    mu_rn_lim = -2.5 * np.log10(nsigma * rn / (t_exp * a_pix * n_pix**0.5)) + zp - k * airmass

    d1 = np.min(np.abs(mu_sky_lim - mu_source_lim))
    d2 = np.min(np.abs(mu_sky_lim - mu_rn_lim))
    d3 = np.min(np.abs(mu_rn_lim - mu_source_lim))

    if np.min([d1, d2, d3]) < mag_diff_warn:
        warnings.warn(
            "Limiting magnitudes in different cases are within %.3f mags, \
            result may be too optimistic by up 0.38 mags/sq arcsec."
            % mag_diff_warn
        )

    result = np.vstack([mu_sky_lim, mu_source_lim, mu_rn_lim])
    result = np.min(result, axis=0)

    return result


class SurfaceBrightLimitMetric(BaseMetric):
    """Gaussian limit, ignoring systematic errors in photometry

    Parameters
    ----------
    pixscale : `float` (0.2)
        Pixelscale, Arcseconds per pixel
    nsigma : `float` (3)
        The detection limit (usuall 3 or 5)
    tot_area : `float` (100)
        Total sky area summed over, square arcseconds
    zpt : `dict` of `float` (None)
        telescope zeropoints. If None, computed from phot_utils
    k_atm : `dict` of `float` (None)
        Atmospheric extinction parameters. If None, computed from phot_utils
    readnoise : `float` (8.8)
        Readnoise in electrons
    """

    def __init__(
        self,
        pixscale=0.2,
        nsigma=3.0,
        tot_area=100.0,
        filter_col="filter",
        units="mag/sq arcsec",
        airmass_col="airmass",
        exptime_col="visitExposureTime",
        metric_name="SurfaceBrightLimit",
        skybrightness_col="skyBrightness",
        nexp_col="numExposures",
        zpt=None,
        k_atm=None,
        readnoise=8.8,
        **kwargs,
    ):
        super().__init__(
            col=[filter_col, airmass_col, exptime_col, skybrightness_col, nexp_col],
            units=units,
            metric_name=metric_name,
            **kwargs,
        )
        self.filter_col = filter_col
        self.airmass_col = airmass_col
        self.exptime_col = exptime_col
        self.skybrightness_col = skybrightness_col
        self.nexp_col = nexp_col

        self.readnoise = readnoise
        self.pixscale = pixscale
        self.nsigma = nsigma
        self.tot_area = tot_area

        # Compute default zeropoints
        if zpt is None:
            zp_inst, k_atm = load_inst_zeropoints()
            self.zpt = zp_inst
            self.k_atm = k_atm
        else:
            self.zpt = zpt
            self.k_atm = k_atm

    def run(self, data_slice, slice_point):
        filtername = np.unique(data_slice[self.filter_col])
        if np.size(filtername) > 1:
            ValueError("Can only coadd depth in single filter, got filters %s" % filtername)
        filtername = filtername[0]

        # Scale up readnoise if the visit was split into multiple snaps
        readnoise = self.readnoise * np.sqrt(data_slice[self.nexp_col])

        sb_per_visit = surface_brightness_limit_approx(
            self.zpt[filtername],
            self.k_atm[filtername],
            data_slice[self.airmass_col],
            data_slice[self.skybrightness_col],
            rn=readnoise,
            pixscale=self.pixscale,
            nsigma=self.nsigma,
            t_exp=data_slice[self.exptime_col],
            tot_area=self.tot_area,
        )

        coadd_sb_limit = 1.25 * np.log10(np.sum(10 ** (0.8 * sb_per_visit)))
        return coadd_sb_limit
