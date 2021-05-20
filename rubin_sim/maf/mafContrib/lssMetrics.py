import numpy as np
import healpy as hp
import scipy

from rubin_sim.maf.metrics import BaseMetric, Coaddm5Metric

__all__ = ['GalaxyCountsMetric']


class GalaxyCountsMetric(BaseMetric):
    """ Estimate the number of galaxies expected at a particular coadded depth.
    """
    def __init__(self, m5Col = 'fiveSigmaDepth', nside=128, metricName='GalaxyCounts', **kwargs):
        self.m5Col = m5Col
        super(GalaxyCountsMetric, self).__init__(col=self.m5Col, metricName=metricName, **kwargs)
        # Use the coadded depth metric to calculate the coadded depth at each point.
        self.coaddmetric = Coaddm5Metric(m5Col=self.m5Col)
        # Total of 41253.0 galaxies across the sky (at what magnitude?).
        # This didn't seem to work quite right for me..
        self.scale = 41253.0 / hp.nside2npix(nside) / 5000.
        # Reset units (otherwise uses magnitudes).
        self.units = 'Galaxy Counts'

    def _galCount(self, apparent_mag, coaddm5):
        # Order for galCount must be apparent mag, then coaddm5, for scipy.integrate method.
        dn_gal = np.power(10., -3.52) * np.power(10., 0.34*apparent_mag)
        completeness = 0.5*scipy.special.erfc(apparent_mag-coaddm5)
        return dn_gal*completeness

    def run(self, dataSlice, slicePoint=None):
        # Calculate the coadded depth.
        coaddm5 = self.coaddmetric.run(dataSlice)
        # Calculate the number of galaxies.
        # From Carroll et al, 2014 SPIE (http://arxiv.org/abs/1501.04733)
        # I'm not entirely certain this gives a properly calibrated number of galaxy counts,
        # however it is proportional to the expected number at least (and should be within an order of magnitude)
        num_gal, intErr = scipy.integrate.quad(self._galCount, -np.inf, 32, args=coaddm5)
        num_gal *= self.scale
        return num_gal
