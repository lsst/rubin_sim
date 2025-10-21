__all__ = ["NoOffset", "OffsetSNR", "BaseOffset"]

import numpy as np

from ..maf.utils.astrometry_utils import m52snr

# from lsst.sims.selfcal.clouds.Arma import ArmaSf, Clouds


class BaseOffset:
    """Base class for how to make offset classes"""

    def __init__(self, **kwargs):
        self.newkey = "dmag_keyname"
        pass

    def __call__(self, stars, visit, **kwargs):
        pass


class NoOffset(BaseOffset):
    def __init__(self):
        """Make no changes to the mags"""
        self.newkey = "dmag_zero"

    def __call__(self, stars, visits, **kwargs):
        dmag = np.zeros(stars.size, dtype=list(zip([self.newkey], [float])))
        return dmag


class OffsetSys(BaseOffset):
    def __init__(self, error_sys=0.003):
        """Systematic error floor for photometry"""
        self.error_sys = error_sys
        self.newkey = "dmag_sys"

    def __call__(self, stars, visits, **kwargs):
        nstars = np.size(stars)
        dmag = np.random.randn(nstars) * self.error_sys
        return dmag


class OffsetClouds(BaseOffset):
    """Offset based on cloud structure.
    Not used, as not fully implemented in this version (ArmaSf).
    """

    def __init__(self, sampling=256, fov=3.5):
        self.fov = fov
        self.newkey = "dmag_cloud"
        # self.SF = ArmaSf()
        self.SF = None
        # self.cloud = Clouds()
        self.cloud = None

    def __call__(self, stars, visits, **kwargs):
        # XXX-Double check extinction is close to the Opsim transparency
        extinc_mags = visits["transparency"]
        if extinc_mags != 0.0:
            sf_theta, sf_sf = self.SF.CloudSf(500.0, 300.0, 5.0, extinc_mags, 0.55)
            # Call the Clouds
            self.cloud.makeCloudImage(sf_theta, sf_sf, extinc_mags, fov=self.fov)
            # Interpolate clouds to correct position.
            # Nearest neighbor for speed?
            nim = self.cloud.cloudimage[0, :].size
            # calc position in cloud image of each star
            starx_interp = (np.degrees(stars["x"]) + self.fov / 2.0) * 3600.0 / self.cloud.pixscale
            stary_interp = (np.degrees(stars["y"]) + self.fov / 2.0) * 3600.0 / self.cloud.pixscale

            # Round off position and make it an int
            starx_interp = np.round(starx_interp).astype(int)
            stary_interp = np.round(stary_interp).astype(int)

            # Handle any stars that are out of the field for some reason
            starx_interp[np.where(starx_interp < 0)] = 0
            starx_interp[np.where(starx_interp > nim - 1)] = nim - 1
            stary_interp[np.where(stary_interp < 0)] = 0
            stary_interp[np.where(stary_interp > nim - 1)] = nim - 1

            dmag = self.cloud.cloudimage[starx_interp, stary_interp]
        else:
            dmag = np.zeros(stars.size)
        return dmag


class OffsetSNR(BaseOffset):
    """Generate offsets based on the 5-sigma limiting depth of an observation
    and the brightness of the star.

    Note that this takes into account previous offsets that have been applied
    (so run this after things like vignetting).
    """

    def __init__(self, lsst_filter="r"):
        self.lsst_filter = lsst_filter
        self.newkey = "dmag_snr"

    def calc_mag_errors(self, magnitudes, m5, err_only=False):
        """ """
        snr = m52snr(magnitudes, m5)
        # via good old https://www.eso.org/~ohainaut/ccd/sn.html
        magnitude_errors = 2.5 * np.log10(1.0 + 1.0 / snr)
        if err_only:
            dmag = magnitude_errors
        else:
            dmag = np.random.randn(len(magnitudes)) * magnitude_errors
        return dmag

    def __call__(self, stars, visit, dmags=None):
        if dmags is None:
            dmags = {}
        temp_mag = stars[self.lsst_filter + "mag"].copy()
        # calc what magnitude the star has when it hits the silicon.
        # Thus we compute the SNR noise
        # AFTER things like cloud extinction and vignetting.
        for key in list(dmags.keys()):
            temp_mag = temp_mag + dmags[key]
        dmag = self.calc_mag_errors(temp_mag, visit["fiveSigmaDepth"])
        return dmag
