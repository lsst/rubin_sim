from __future__ import print_function
# Example of a *very* simple variabiilty metric
# krughoff@uw.edu, ebellm, ljones

import numpy as np
from scipy.signal import lombscargle

from rubin_sim.maf.metrics import BaseMetric

__all__ = ['PeriodDeviationMetric']


def find_period_LS(times, mags, minperiod=2., maxperiod=35., nbinmax=10**5, verbose=False):
    """Find the period of a lightcurve using scipy's lombscargle method.
    The parameters used here imply magnitudes but there is no reason this would not work if fluxes are passed.

    :param times: A list of times for the given observations
    :param mags: A list of magnitudes for the object at the given times
    :param minperiod: Minimum period to search
    :param maxperiod: Maximum period to search
    :param nbinmax: Maximum number of frequency bins to use in the search
    :returns: Period in the same units as used in times.  This is simply
              the max value in the Lomb-Scargle periodogram
    """
    if minperiod < 0:
        minperiod = 0.01
    nbins = int((times.max() - times.min())/minperiod * 1000)
    if nbins > nbinmax:
        if verbose:
            print('lowered nbins')
        nbins = nbinmax

    # Recenter the magnitude measurements about zero
    dmags = mags - np.median(mags)
    # Create frequency bins
    f = np.linspace(1./maxperiod, 1./minperiod, nbins)

    # Calculate periodogram
    pgram = lombscargle(times, dmags, f)

    idx = np.argmax(pgram)
    # Return period of the bin with the max value in the periodogram
    return 1./f[idx]


class PeriodDeviationMetric(BaseMetric):
    """Measure the percentage deviation of recovered periods for pure sine wave variability (in magnitude).
    """
    def __init__(self, col='observationStartMJD', periodMin=3., periodMax=35., nPeriods=5,
                 meanMag=21., amplitude=1., metricName='Period Deviation', periodCheck=None,
                 **kwargs):
        """
        Construct an instance of a PeriodDeviationMetric class

        :param col: Name of the column to use for the observation times, commonly 'expMJD'
        :param periodMin: Minimum period to test (days)
        :param periodMax: Maximimum period to test (days)
        :param periodCheck: Period to use in the reduce function (days)
        :param meanMag: Mean value of the lightcurve
        :param amplitude: Amplitude of the variation (mags)
        """
        self.periodMin = periodMin
        self.periodMax = periodMax
        self.periodCheck = periodCheck
        self.guessPMin = np.min([self.periodMin*0.8, self.periodMin-1])
        self.guessPMax = np.max([self.periodMax*1.20, self.periodMax+1])
        self.nPeriods = nPeriods
        self.meanMag = meanMag
        self.amplitude = amplitude
        super(PeriodDeviationMetric, self).__init__(col, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """
        Run the PeriodDeviationMetric
        :param dataSlice : Data for this slice.
        :param slicePoint: Metadata for the slice. (optional)
        :return: The error in the period estimated from a Lomb-Scargle periodogram
        """

        # Make sure the observation times are sorted
        data = np.sort(dataSlice[self.colname])

        # Create 'nPeriods' random periods within range of min to max.
        if self.periodCheck is not None:
            periods = [self.periodCheck]
        else:
            periods = self.periodMin + np.random.random(self.nPeriods)*(self.periodMax - self.periodMin)
        # Make sure the period we want to check is in there
        periodsdev = np.zeros(np.size(periods), dtype='float')
        for i, period in enumerate(periods):
            omega = 1./period
            # Calculate up the amplitude.
            lc = self.meanMag + self.amplitude*np.sin(omega*data)
            # Try to recover the period given a window buffered by min of a day or 20% of period value.
            if len(lc) < 3:
                # Too few points to find a period
                return self.badval

            pguess = find_period_LS(data, lc, minperiod=self.guessPMin, maxperiod=self.guessPMax)
            periodsdev[i] = (pguess - period) / period

        return {'periods': periods, 'periodsdev': periodsdev}

    def reducePDev(self, metricVal):
        """
        At a particular slicepoint, return the period deviation for self.periodCheck.
        If self.periodCheck is None, just return a random period in the range.
        """
        result = metricVal['periodsdev'][0]
        return result

    def reduceWorstPeriod(self, metricVal):
        """
        At each slicepoint, return the period with the worst period deviation.
        """
        worstP = np.array(metricVal['periods'])[np.where(metricVal['periodsdev'] == metricVal['periodsdev'].max())[0]]
        return worstP

    def reduceWorstPDev(self, metricVal):
        """
        At each slicepoint, return the largest period deviation.
        """
        worstPDev = np.array(metricVal['periodsdev'])[np.where(metricVal['periodsdev'] == metricVal['periodsdev'].max())[0]]
        return worstPDev
