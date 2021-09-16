import numpy as np
import healpy as hp
from BaseMetric import BaseMetric
import rubin_sim.maf.utils as mafUtils
from scipy import interpolate


class BDParallaxMetric(BaseMetric):
    """Calculate the distance to which one could reach a parallax SNR for a given object
    Modification of ParallaxMetric. Taken from:
    https://github.com/jgizis/LSST-BD-Cadence/blob/main/bd_allLT_baseline_17.ipynb

    Uses columns ra_pi_amp and dec_pi_amp, calculated by the ParallaxFactorStacker.

    Parameters
    ----------
    metricName : str, opt
        Default 'parallax'.
    m5Col : str, opt
        The default column name for m5 information in the input data. Default fiveSigmaDepth.
    filterCol : str, opt
        The column name for the filter information. Default filter.
    seeingCol : str, opt
        The column name for the seeing information. Since the astrometry errors are based on the physical
        size of the PSF, this should be the FWHM of the physical psf. Default seeingFwhmGeom.
    mags : dict
        The absolute magnitude of the obeject in question. Keys of filter name, values in mags.
    distances : np.array
        Distances to try putting the object at (pc).
    atm_err : float, opt
        The expected centroiding error due to the atmosphere, in arcseconds. Default 0.01.
    badval : float, opt
        The value to return when the metric value cannot be calculated. Default 0.

    Good values for mags:
    mags={'i': 16.00, 'z': 14.52, 'y': 13.58}, metricName='L0'
    mags={'i': 16.41, 'z': 14.93, 'y': 13.97}, metricName='L1'
    mags={'i': 16.73, 'z': 15.30, 'y': 14.33}, metricName='L2'
    mags={'i': 17.4, 'z': 15.88, 'y': 14.89}, metricName='L3'
    mags={'i': 18.35, 'z': 16.68, 'y': 15.66}, metricName='L4'
    mags={'i': 18.71, 'z': 16.94, 'y': 15.87}, metricName='L5'
    mags={'i': 19.27, 'z': 17.35, 'y': 16.27}, metricName='L6'
    mags={'i': 20.09, 'z': 18.18, 'y': 17.13}, metricName='L7'
    mags={'i': 20.38, 'z': 18.10, 'y': 17.04}, metricName='L8'
    mags={'i': 20.09, 'z': 17.69, 'y': 16.57}, metricName='L9'
    mags={'i': 20.22, 'z': 17.98, 'y': 16.77}, metricName='T0'
    mags={'i': 21.10, 'z': 18.84, 'y': 17.45}, metricName='T1'
    mags={'i': 21.97, 'z': 18.26, 'y': 16.75}, metricName='T2'
    mags={'i': 22.50, 'z': 18.08, 'y': 16.50}, metricName='T3'
    mags={'i': 22.50, 'z': 18.02, 'y': 16.32}, metricName='T4'
    mags={'i': 22.69, 'z': 19.20, 'y': 17.43}, metricName='T5'
    mags={'i': 23.00, 'z': 19.82, 'y': 18.06}, metricName='T6'
    mags={'z': 21.17, 'y': 19.34}, metricName='T7'
    mags={'z': 21.52, 'y': 19.75}, metricName='T8'
    mags={'z': 21.82, 'y': 20.37}, metricName='T9'
    """
    def __init__(self, metricName='bdParallax', m5Col='fiveSigmaDepth',
                 filterCol='filter', seeingCol='seeingFwhmGeom',
                 badval=0, mags={'i': 20.09, 'z': 18.18, 'y': 17.13}, parallax_snr=10.,
                 distances=np.arange(10, 200, 10),
                 atm_err=0.01, normalize=False, **kwargs):
        Cols = [m5Col, filterCol, seeingCol, 'ra_pi_amp', 'dec_pi_amp']
        
        units = 'pc'
        super(BDParallaxMetric, self).__init__(Cols, metricName=metricName, units=units,
                                               badval=badval, **kwargs)
        # set return types
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.distances = distances
        self.mags = {}
        distance_mod = 5.0*np.log10(distances)-5.0
        for key in mags:
            self.mags[key] = mags[key] + distance_mod
        self.parallax_snr_goal = parallax_snr
        self.atm_err = atm_err
        self.filters = list(self.mags.keys())
        self.parallaxes = 1000.0/distances  # mas
        
    def _final_sigma(self, position_errors, ra_pi_amp, dec_pi_amp):
        """Assume parallax in RA and DEC are fit independently, then combined.
        All inputs assumed to be arcsec """
        sigma_A = position_errors/ra_pi_amp
        sigma_B = position_errors/dec_pi_amp
        sigma_ra = np.sqrt(1./np.sum(1./sigma_A**2, axis=1))
        sigma_dec = np.sqrt(1./np.sum(1./sigma_B**2, axis=1))
        # Combine RA and Dec uncertainties, convert to mas
        sigma = np.sqrt(1./(1./sigma_ra**2+1./sigma_dec**2))*1e3
        return sigma

    def run(self, dataslice, slicePoint=None):
        snr = np.zeros((np.size(self.mags[self.filters[0]]), len(dataslice)), dtype='float')
        # compute SNR for all observations
        for filt in self.filters:
            good = np.where(dataslice[self.filterCol] == filt)[0]
            if np.size(good) > 0:
                snr[:, good] = mafUtils.m52snr(self.mags[str(filt)][:, np.newaxis],
                                               dataslice[self.m5Col][good])
                
        position_errors = np.sqrt(mafUtils.astrom_precision(dataslice[self.seeingCol],
                                                            snr)**2+self.atm_err**2)
        # uncertainty in the parallax in mas
        sigma = self._final_sigma(position_errors, dataslice['ra_pi_amp'], dataslice['dec_pi_amp'])
        fitted_parallax_snr = self.parallaxes/sigma
        result = self.badval
        # Let's interpolate to the distance where we would get our target SNR
        if np.max(fitted_parallax_snr) >= self.parallax_snr_goal:
            f = interpolate.interp1d(fitted_parallax_snr, self.distances, fill_value="extrapolate")
            result = f(self.parallax_snr_goal)
        return result


class VolumeSumMetric(BaseMetric):
    """Compute the total volume assuming a metric has values of distance
    """
    def __init__(self, col=None, metricName='VolumeSum', nside=None, **kwargs):
        super(VolumeSumMetric, self).__init__(col=col, metricName=metricName, **kwargs)
        self.pix_area = hp.nside2pixarea(nside)
        
    def run(self, dataSlice, slicePoint=None):
        # volume of sphere, times ratio of pixel area divided by area of sphere
        vols = 1./3. * dataSlice[self.colname]**3 * self.pix_area
        return np.sum(vols)
