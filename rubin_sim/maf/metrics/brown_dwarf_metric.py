__all__ = ("BDParallaxMetric", "VolumeSumMetric")

import healpy as hp
import numpy as np
from scipy import interpolate

import rubin_sim.maf.utils as mafUtils

from .base_metric import BaseMetric


def bd_colors(spec_type):
    """Returns dictionary of potential Brown Dwarf colors

    Parameters
    ----------
    spec_type : `str`
        The desired spetral type of the Brown Dwarf.

    """
    result = {}
    result["L0"] = {"i": 16.00, "z": 14.52, "y": 13.58}
    result["L1"] = {"i": 16.41, "z": 14.93, "y": 13.97}
    result["L2"] = {"i": 16.73, "z": 15.30, "y": 14.33}
    result["L3"] = ({"i": 17.4, "z": 15.88, "y": 14.89},)
    result["L4"] = {"i": 18.35, "z": 16.68, "y": 15.66}
    result["L5"] = {"i": 18.71, "z": 16.94, "y": 15.87}
    result["L6"] = {"i": 19.27, "z": 17.35, "y": 16.27}
    result["L7"] = {"i": 20.09, "z": 18.18, "y": 17.13}
    result["L8"] = {"i": 20.38, "z": 18.10, "y": 17.04}
    result["L9"] = {"i": 20.09, "z": 17.69, "y": 16.57}
    result["T0"] = {"i": 20.22, "z": 17.98, "y": 16.77}
    result["T1"] = {"i": 21.10, "z": 18.84, "y": 17.45}
    result["T2"] = {"i": 21.97, "z": 18.26, "y": 16.75}
    result["T3"] = {"i": 22.50, "z": 18.08, "y": 16.50}
    result["T4"] = {"i": 22.50, "z": 18.02, "y": 16.32}
    result["T5"] = {"i": 22.69, "z": 19.20, "y": 17.43}
    result["T6"] = {"i": 23.00, "z": 19.82, "y": 18.06}
    result["T7"] = {"z": 21.17, "y": 19.34}
    result["T8"] = {"z": 21.52, "y": 19.75}
    result["T9"] = {"z": 21.82, "y": 20.37}
    return result[spec_type]


class BDParallaxMetric(BaseMetric):
    """Calculate the distance to which one could reach a parallax SNR for a
    given object

    Modification of ParallaxMetric, illustrated in
    https://github.com/jgizis/
    LSST-BD-Cadence/blob/main/bd_allLT_baseline_17.ipynb

    Uses columns ra_pi_amp and dec_pi_amp,
    calculated by the ParallaxFactorStacker.

    Parameters
    ----------
    metricName : `str`, opt
        Default 'parallax'.
    m5_col : `str`, opt
        The default column name for m5 information in the input data.
        Default fiveSigmaDepth.
    filter_col : `str`, opt
        The column name for the filter information. Default filter.
    seeing_col : `str`, opt
        The column name for the seeing information.
        Since the astrometry errors are based on the physical
        size of the PSF, this should be the FWHM of the physical psf.
        Default seeingFwhmGeom.
    mags : `dict` or None
        The absolute magnitude of the object in question.
        Keys of filter name, values in mags.
        Defaults to an L7 spectral type if None.
    distances : `np.array`, (N,)
        Distances to try putting the object at (pc).
    atm_err : `float`, opt
        The expected centroiding error due to the atmosphere, in arcseconds.
        Default 0.01.
    badval : `float`, opt
        The value to return when the metric value cannot be calculated.
        Default 0.
    """

    def __init__(
        self,
        metric_name="bdParallax",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        seeing_col="seeingFwhmGeom",
        badval=0,
        mags=None,
        parallax_snr=10.0,
        distances=np.arange(10, 200, 10),
        atm_err=0.01,
        normalize=False,
        **kwargs,
    ):
        cols = [m5_col, filter_col, seeing_col, "ra_pi_amp", "dec_pi_amp"]

        units = "pc"
        super().__init__(cols, metric_name=metric_name, units=units, badval=badval, **kwargs)
        # set return types
        self.m5_col = m5_col
        self.seeing_col = seeing_col
        self.filter_col = filter_col
        self.distances = distances
        self.mags = {}
        distance_mod = 5.0 * np.log10(distances) - 5.0
        if mags is None:
            mags = bd_colors("L7")
        for key in mags:
            self.mags[key] = mags[key] + distance_mod
        self.parallax_snr_goal = parallax_snr
        self.atm_err = atm_err
        self.filters = list(self.mags.keys())
        self.parallaxes = 1000.0 / distances  # mas

    def _final_sigma(self, position_errors, ra_pi_amp, dec_pi_amp):
        """Assume parallax in RA and DEC are fit independently, then combined.
        All inputs assumed to be arcsec.
        """
        sigma_a = position_errors / ra_pi_amp
        sigma_b = position_errors / dec_pi_amp
        sigma_ra = np.sqrt(1.0 / np.sum(1.0 / sigma_a**2, axis=1))
        sigma_dec = np.sqrt(1.0 / np.sum(1.0 / sigma_b**2, axis=1))
        # Combine RA and Dec uncertainties, convert to mas
        sigma = np.sqrt(1.0 / (1.0 / sigma_ra**2 + 1.0 / sigma_dec**2)) * 1e3
        return sigma

    def run(self, data_slice, slice_point=None):
        snr = np.zeros((np.size(self.mags[self.filters[0]]), len(data_slice)), dtype="float")
        # compute SNR for all observations
        for filt in self.filters:
            good = np.where(data_slice[self.filter_col] == filt)[0]
            if np.size(good) > 0:
                snr[:, good] = mafUtils.m52snr(
                    self.mags[str(filt)][:, np.newaxis], data_slice[self.m5_col][good]
                )

        position_errors = mafUtils.astrom_precision(data_slice[self.seeing_col], snr, self.atm_err)
        # uncertainty in the parallax in mas
        sigma = self._final_sigma(position_errors, data_slice["ra_pi_amp"], data_slice["dec_pi_amp"])
        fitted_parallax_snr = self.parallaxes / sigma
        result = self.badval
        # Let's interpolate to the distance where we would get our target SNR
        if np.max(fitted_parallax_snr) >= self.parallax_snr_goal:
            f = interpolate.interp1d(fitted_parallax_snr, self.distances, fill_value="extrapolate")
            result = f(self.parallax_snr_goal)
        return result


class VolumeSumMetric(BaseMetric):
    """Compute the total volume assuming a metric has values of distance."""

    def __init__(self, col=None, metric_name="VolumeSum", nside=None, **kwargs):
        super(VolumeSumMetric, self).__init__(col=col, metric_name=metric_name, **kwargs)
        self.pix_area = hp.nside2pixarea(nside)

    def run(self, data_slice, slice_point=None):
        # volume of sphere, times ratio of pixel area divided by area of sphere
        vols = 1.0 / 3.0 * data_slice[self.colname] ** 3 * self.pix_area
        return np.sum(vols)
