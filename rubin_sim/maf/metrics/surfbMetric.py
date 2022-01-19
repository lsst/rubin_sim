import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.photUtils import Bandpass, PhotometricParameters
import rubin_sim.utils as rsUtils
from rubin_sim.data import get_data_dir
import os
import warnings

__all__ = ["SurfaceBrightLimitMetric"]


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
    """Compute surface brightness limit in 3 limiting cases, return the brightest.

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
    surface brightness limit in mags/sq arcsec
    aka the surface brightness that reaches SNR=nsigma when measured over tot_area.
    """

    A_pix = pixscale ** 2

    n_pix = tot_area / A_pix

    # Sky limited case
    mu_sky_lim = (
        -1.25 * np.log10(nsigma ** 2 / (A_pix * t_exp * n_pix))
        + 0.5 * mu_sky
        + 0.5 * zp
        - k * airmass
    )

    # Source limited case
    # XXX--double check this algerbra. Pretty sure it's right now.
    mu_source_lim = (
        -1.25 * np.log10(nsigma)
        + 1.25 / 2.0 * np.log10(n_pix * A_pix * t_exp)
        + zp
        - k * airmass
    )

    # Readnoise limited case
    mu_rn_lim = (
        -2.5 * np.log10(nsigma * rn / (t_exp * A_pix * n_pix ** 0.5)) + zp - k * airmass
    )

    d1 = np.min(np.abs(mu_sky_lim - mu_source_lim))
    d2 = np.min(np.abs(mu_sky_lim - mu_rn_lim))
    d3 = np.min(np.abs(mu_rn_lim - mu_source_lim))

    if np.min([d1, d2, d3]) < mag_diff_warn:
        warnings.warn(
            "Limiting magnitudes in different cases are within %.3f mags, result may be too optimistic by up 0.38 mags/sq arcsec."
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
        telescope zeropoints. If None, computed from photUtils
    kAtm : `dict` of `float` (None)
        Atmospheric extinction parameters. If None, computed from photUtils
    readnoise : `float` (8.8)
        Readnoise in electrons
    """

    def __init__(
        self,
        pixscale=0.2,
        nsigma=3.0,
        tot_area=100.0,
        filterCol="filter",
        units="mag/sq arcsec",
        airmassCol="airmass",
        exptimeCol="visitExposureTime",
        metricName="SurfaceBrightLimit",
        skybrightnessCol="skyBrightness",
        nexpCol="numExposures",
        zpt=None,
        kAtm=None,
        readnoise=8.8,
        **kwargs
    ):
        super().__init__(
            col=[filterCol, airmassCol, exptimeCol, skybrightnessCol, nexpCol],
            units=units,
            metricName=metricName,
            **kwargs
        )
        self.filterCol = filterCol
        self.airmassCol = airmassCol
        self.exptimeCol = exptimeCol
        self.skybrightnessCol = skybrightnessCol
        self.nexpCol = nexpCol

        self.readnoise = readnoise
        self.pixscale = pixscale
        self.nsigma = nsigma
        self.tot_area = tot_area

        # Compute default zeropoints
        if zpt is None:
            zp_inst = {}
            datadir = get_data_dir()
            for filtername in "ugrizy":
                # set gain and exptime to 1 so the instrumental zeropoint will be in photoelectrons and per second
                phot_params = PhotometricParameters(
                    nexp=1, gain=1, exptime=1, bandpass=filtername
                )
                bp = Bandpass()
                bp.readThroughput(
                    os.path.join(
                        datadir, "throughputs/baseline/", "total_%s.dat" % filtername
                    )
                )
                zp_inst[filtername] = bp.calcZP_t(phot_params)
            self.zpt = zp_inst
        else:
            self.zpt = zpt
        if kAtm is None:
            syseng = rsUtils.SysEngVals()
            self.kAtm = syseng.kAtm
        else:
            self.kAtm = kAtm

    def run(self, dataSlice, slicePoint):
        filtername = np.unique(dataSlice[self.filterCol])
        if np.size(filtername) > 1:
            ValueError(
                "Can only coadd depth in single filter, got filters %s" % filtername
            )
        filtername = filtername[0]

        # Scale up readnoise if the visit was split into multiple snaps
        readnoise = self.readnoise * np.sqrt(dataSlice[self.nexpCol])

        sb_per_visit = surface_brightness_limit_approx(
            self.zpt[filtername],
            self.kAtm[filtername],
            dataSlice[self.airmassCol],
            dataSlice[self.skybrightnessCol],
            rn=readnoise,
            pixscale=self.pixscale,
            nsigma=self.nsigma,
            t_exp=dataSlice[self.exptimeCol],
            tot_area=self.tot_area,
        )

        coadd_sb_limit = 1.25 * np.log10(np.sum(10 ** (0.8 * sb_per_visit)))
        return coadd_sb_limit
