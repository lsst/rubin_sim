__all__ = ("GalaxyCountsMetric",)

import healpy as hp
import numpy as np
import scipy

from rubin_sim.maf.metrics import BaseMetric, ExgalM5


class GalaxyCountsMetric(BaseMetric):
    """Estimate the number of galaxies expected at a particular (extragalactic)
    coadded depth.
    """

    def __init__(self, m5_col="fiveSigmaDepth", nside=128, metric_name="GalaxyCounts", **kwargs):
        self.m5_col = m5_col
        super(GalaxyCountsMetric, self).__init__(col=self.m5_col, metric_name=metric_name, **kwargs)
        # Use the extinction corrected coadded depth metric to calculate
        # the depth at each point.
        self.coaddmetric = ExgalM5(m5_col=self.m5_col)
        # Total of 41253.0 galaxies across the sky (at what magnitude?).
        # This didn't seem to work quite right for me..
        self.scale = 41253.0 / hp.nside2npix(nside) / 5000.0
        # Reset units (otherwise uses magnitudes).
        self.units = "Galaxy Counts"

    def _gal_count(self, apparent_mag, coaddm5):
        # Order for galCount must be apparent mag, then coaddm5,
        # for scipy.integrate method.
        dn_gal = np.power(10.0, -3.52) * np.power(10.0, 0.34 * apparent_mag)
        completeness = 0.5 * scipy.special.erfc(apparent_mag - coaddm5)
        return dn_gal * completeness

    def run(self, data_slice, slice_point=None):
        # Calculate the coadded depth.
        coaddm5 = self.coaddmetric.run(data_slice)
        # Calculate the number of galaxies.
        # From Carroll et al, 2014 SPIE (http://arxiv.org/abs/1501.04733)
        # I'm not entirely certain this gives a properly calibrated number
        # of galaxy counts, however it is proportional to the expected number
        # at least (and should be within an order of magnitude)
        num_gal, int_err = scipy.integrate.quad(self._gal_count, -np.inf, 32, args=coaddm5)
        num_gal *= self.scale
        return num_gal
