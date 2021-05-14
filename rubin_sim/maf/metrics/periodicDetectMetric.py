import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.maf.utils import m52snr
import rubin_sim.utils as utils
import scipy

__all__ = ['PeriodicDetectMetric']


class PeriodicDetectMetric(BaseMetric):
    """Determine if we would be able to classify an object as periodic/non-uniform, using an F-test
    The idea here is that if a periodic source is aliased, it will be indistinguishable from a constant source,
    so we can find a best-fit constant, and if the reduced chi-squared is ~1, we know we are aliased.

    Parameters
    ----------

    period : float (2) or array
        The period of the star (days). Can be a single value, or an array. If an array, amplitude and starMag
        should be arrays of equal length.
    amplitude : floar (0.1)
        The amplitude of the stellar variablility (mags).
    starMag : float (20.)
        The mean magnitude of the star in r (mags).
    sig_level : float (0.05)
        The value to use to compare to the p-value when deciding if we can reject the null hypothesis.
    SedTemplate : str ('F')
        The stellar SED template to use to generate realistic colors (default is an F star, so RR Lyrae-like)

    Returns
    -------

    1 if we would detect star is variable, 0 if it is well-fit by a constant value. If using arrays to test multiple
    period-amplitude-mag combinations, will be the sum of the number of detected stars.
    """
    def __init__(self, mjdCol='observationStartMJD', periods=2., amplitudes=0.1, m5Col='fiveSigmaDepth',
                 metricName='PeriodicDetectMetric', filterCol='filter', starMags=20, sig_level=0.05, 
                 SedTemplate='F', **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        if np.size(periods) == 1:
            self.periods = [periods]
            # Using the same magnitude for all filters. Could expand to fit the mean in each filter.
            self.starMags = [starMags]
            self.amplitudes = [amplitudes]
        else:
            self.periods = periods
            self.starMags = starMags
            self.amplitudes = amplitudes
        self.sig_level = sig_level
        self.SedTemplate = SedTemplate

        super(PeriodicDetectMetric, self).__init__([mjdCol, m5Col, filterCol], metricName=metricName,
                                                   units='N Detected (0, %i)' % np.size(periods), **kwargs)

    def run(self, dataSlice, slicePoint=None):
        result = 0
        n_pts = np.size(dataSlice[self.mjdCol])
        n_filt = np.size(np.unique(dataSlice[self.filterCol]))

        # If we had a correct model with phase, amplitude, period, mean_mags, then chi_squared/DoF would be ~1 with 3+n_filt free parameters.
        # The mean is one free parameter
        p1 = n_filt
        p2 = 3.+n_filt
        chi_sq_2 = 1.*(n_pts-p2)

        u_filters = np.unique(dataSlice[self.filterCol])

        if n_pts > p2:
            for period, starMag, amplitude in zip(self.periods, self.starMags, self.amplitudes):
                chi_sq_1 = 0
                mags = utils.stellarMags(self.SedTemplate, rmag=starMag)
                for filtername in u_filters:
                    in_filt = np.where(dataSlice[self.filterCol] == filtername)[0]
                    lc = amplitude*np.sin(dataSlice[self.mjdCol][in_filt]*(np.pi*2)/period) + mags[filtername]
                    snr = m52snr(lc, dataSlice[self.m5Col][in_filt])
                    delta_m = 2.5*np.log10(1.+1./snr)
                    weights = 1./(delta_m**2)
                    weighted_mean = np.sum(weights*lc)/np.sum(weights)
                    chi_sq_1 += np.sum(((lc - weighted_mean)**2/delta_m**2))
                # Yes, I'm fitting magnitudes rather than flux. At least I feel kinda bad about it.
                # F-test for nested models Regression problems:  https://en.wikipedia.org/wiki/F-test
                f_numerator = (chi_sq_1 - chi_sq_2)/(p2-p1)
                f_denom = 1.  # This is just reduced chi-squared for the more complicated model, so should be 1.
                f_val = f_numerator/f_denom
                # Has DoF (p2-p1, n-p2)
                # https://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python/21503346
                p_value = scipy.stats.f.sf(f_val, p2-p1, n_pts-p2)
                if np.isfinite(p_value):
                    if p_value < self.sig_level:
                        result += 1

        return result
