import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.maf.utils import m52snr

__all__ = ['PhaseGapMetric', 'PeriodicQualityMetric']

class PhaseGapMetric(BaseMetric):
    """
    Measure the maximum gap in phase coverage for observations of periodic variables.

    Parameters
    ----------
    col: str, optional
        Name of the column to use for the observation times (MJD)
    nPeriods: int, optional
        Number of periods to test
    periodMin: float, optional
        Minimum period to test, in days.
    periodMax: float, optional
        Maximum period to test, in days
    nVisitsMin: int, optional
        Minimum number of visits necessary before looking for the phase gap.
    """
    def __init__(self, col='observationStartMJD', nPeriods=5, periodMin=3., periodMax=35., nVisitsMin=3,
                 metricName='Phase Gap', **kwargs):
        self.periodMin = periodMin
        self.periodMax = periodMax
        self.nPeriods = nPeriods
        self.nVisitsMin = nVisitsMin
        super(PhaseGapMetric, self).__init__(col, metricName=metricName, units='Fraction, 0-1', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        if len(dataSlice) < self.nVisitsMin:
            return self.badval
        # Create 'nPeriods' evenly spaced periods within range of min to max.
        step = (self.periodMax-self.periodMin)/self.nPeriods
        if step == 0:
            periods = np.array([self.periodMin])
        else:
            periods = np.arange(self.nPeriods)
            periods = periods/np.max(periods)*(self.periodMax-self.periodMin)+self.periodMin
        maxGap = np.zeros(self.nPeriods, float)

        for i, period in enumerate(periods):
            # For each period, calculate the phases.
            phases = (dataSlice[self.colname] % period)/period
            phases = np.sort(phases)
            # Find the largest gap in coverage.
            gaps = np.diff(phases)
            start_to_end = np.array([1.0 - phases[-1] + phases[0]], float)
            gaps = np.concatenate([gaps, start_to_end])
            maxGap[i] = np.max(gaps)

        return {'periods':periods, 'maxGaps':maxGap}

    def reduceMeanGap(self, metricVal):
        """
        At each slicepoint, return the mean gap value.
        """
        return np.mean(metricVal['maxGaps'])

    def reduceMedianGap(self, metricVal):
        """
        At each slicepoint, return the median gap value.
        """
        return np.median(metricVal['maxGaps'])

    def reduceWorstPeriod(self, metricVal):
        """
        At each slicepoint, return the period with the largest phase gap.
        """
        worstP = metricVal['periods'][np.where(metricVal['maxGaps'] == metricVal['maxGaps'].max())]
        return worstP

    def reduceLargestGap(self, metricVal):
        """
        At each slicepoint, return the largest phase gap value.
        """
        return np.max(metricVal['maxGaps'])


#  To fit a periodic source well, you need to cover the full phase, and fit the amplitude.
class PeriodicQualityMetric(BaseMetric):
    def __init__(self, mjdCol='observationStartMJD', period=2., m5Col='fiveSigmaDepth',
                 metricName='PhaseCoverageMetric', starMag=20, **kwargs):
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.period = period
        self.starMag = starMag
        super(PeriodicQualityMetric, self).__init__([mjdCol, m5Col], metricName=metricName,
                                                    units='Fraction, 0-1', **kwargs)

    def _calc_phase(self, dataSlice):
        """1 is perfectly balanced phase coverage, 0 is no effective coverage.
        """
        angles = dataSlice[self.mjdCol] % self.period
        angles = angles/self.period * 2.*np.pi
        x = np.cos(angles)
        y = np.sin(angles)

        snr = m52snr(self.starMag, dataSlice[self.m5Col])
        x_ave = np.average(x, weights=snr)
        y_ave = np.average(y, weights=snr)

        vector_off = np.sqrt(x_ave**2+y_ave**2)
        return 1.-vector_off

    def _calc_amp(self, dataSlice):
        """Fractional SNR on the amplitude, testing for a variety of possible phases
        """
        phases = np.arange(0, np.pi, np.pi/8.)
        snr = m52snr(self.starMag, dataSlice[self.m5Col])
        amp_snrs = np.sin(dataSlice[self.mjdCol]/self.period*2*np.pi + phases[:, np.newaxis])*snr
        amp_snr = np.min(np.sqrt(np.sum(amp_snrs**2, axis=1)))

        max_snr = np.sqrt(np.sum(snr**2))
        return amp_snr/max_snr

    def run(self, dataSlice, slicePoint=None):
        amplitude_fraction = self._calc_amp(dataSlice)
        phase_fraction = self._calc_phase(dataSlice)
        return amplitude_fraction * phase_fraction
