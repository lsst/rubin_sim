__all__ = ("PhaseGapMetric", "PeriodicQualityMetric")

import numpy as np

from rubin_sim.maf.utils import m52snr

from .base_metric import BaseMetric


class PhaseGapMetric(BaseMetric):
    """Measure the maximum gap in phase coverage for
    observations of periodic variables.

    Parameters
    ----------
    col : `str`, optional
        Name of the column to use for the observation times (MJD)
    n_periods : `int`, optional
        Number of periods to test
    period_min : `float`, optional
        Minimum period to test, in days.
    period_max : `float`, optional
        Maximum period to test, in days
    n_visits_min : `int`, optional
        Minimum number of visits necessary before looking for the phase gap.

    Returns
    -------
    metric_value : `dict` {`periods`: `float`, `maxGaps` : `float`}
        Calculates a dictionary of max gap in phase coverage for each period.
    """

    def __init__(
        self,
        col="observationStartMJD",
        n_periods=5,
        period_min=3.0,
        period_max=35.0,
        n_visits_min=3,
        metric_name="Phase Gap",
        **kwargs,
    ):
        self.period_min = period_min
        self.period_max = period_max
        self.n_periods = n_periods
        self.n_visits_min = n_visits_min
        super(PhaseGapMetric, self).__init__(col, metric_name=metric_name, units="Fraction, 0-1", **kwargs)

    def run(self, data_slice, slice_point=None):
        if len(data_slice) < self.n_visits_min:
            return self.badval
        # Create 'nPeriods' evenly spaced periods within range of min to max.
        step = (self.period_max - self.period_min) / self.n_periods
        if step == 0:
            periods = np.array([self.period_min])
        else:
            periods = np.arange(self.n_periods)
            periods = periods / np.max(periods) * (self.period_max - self.period_min) + self.period_min
        max_gap = np.zeros(self.n_periods, float)

        for i, period in enumerate(periods):
            # For each period, calculate the phases.
            phases = (data_slice[self.colname] % period) / period
            phases = np.sort(phases)
            # Find the largest gap in coverage.
            gaps = np.diff(phases)
            start_to_end = np.array([1.0 - phases[-1] + phases[0]], float)
            gaps = np.concatenate([gaps, start_to_end])
            max_gap[i] = np.max(gaps)

        return {"periods": periods, "maxGaps": max_gap}

    def reduce_mean_gap(self, metric_val):
        """
        At each slice_point, return the mean gap value.
        """
        return np.mean(metric_val["maxGaps"])

    def reduce_median_gap(self, metric_val):
        """
        At each slice_point, return the median gap value.
        """
        return np.median(metric_val["maxGaps"])

    def reduce_worst_period(self, metric_val):
        """
        At each slice_point, return the period with the largest phase gap.
        """
        worst_p = metric_val["periods"][np.where(metric_val["maxGaps"] == metric_val["maxGaps"].max())]
        return worst_p

    def reduce_largest_gap(self, metric_val):
        """
        At each slice_point, return the largest phase gap value.
        """
        return np.max(metric_val["maxGaps"])


#  To fit a periodic source well, you need to cover the full phase,
#  and fit the amplitude.
class PeriodicQualityMetric(BaseMetric):
    """Evaluate phase coverage over a given period.

    Parameters
    ----------
    mjd_col : `str`, opt
        Name of the MJD column in the observations.
    period : `float`, opt
        Period to check.
    m5_col : `str`, opt
        Name of the m5 column in the observations.
    metric_name : `str`, opt
        Name of the metric.
    star_mag : `float`, opt
        Magnitude of the star to simulate coverage for.

    Returns
    -------
    value : `float`
        Value representing phase_coverage * amplitude_snr.
        Ranges from 0 (poor) to 1.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        period=2.0,
        m5_col="fiveSigmaDepth",
        metric_name="PhaseCoverageMetric",
        star_mag=20,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.period = period
        self.star_mag = star_mag
        super(PeriodicQualityMetric, self).__init__(
            [mjd_col, m5_col], metric_name=metric_name, units="Fraction, 0-1", **kwargs
        )

    def _calc_phase(self, data_slice):
        """1 is perfectly balanced phase coverage,
        0 is no effective coverage.
        """
        angles = data_slice[self.mjd_col] % self.period
        angles = angles / self.period * 2.0 * np.pi
        x = np.cos(angles)
        y = np.sin(angles)

        snr = m52snr(self.star_mag, data_slice[self.m5_col])
        x_ave = np.average(x, weights=snr)
        y_ave = np.average(y, weights=snr)

        vector_off = np.sqrt(x_ave**2 + y_ave**2)
        return 1.0 - vector_off

    def _calc_amp(self, data_slice):
        """Fractional SNR on the amplitude,
        testing for a variety of possible phases.
        """
        phases = np.arange(0, np.pi, np.pi / 8.0)
        snr = m52snr(self.star_mag, data_slice[self.m5_col])
        amp_snrs = np.sin(data_slice[self.mjd_col] / self.period * 2 * np.pi + phases[:, np.newaxis]) * snr
        amp_snr = np.min(np.sqrt(np.sum(amp_snrs**2, axis=1)))

        max_snr = np.sqrt(np.sum(snr**2))
        return amp_snr / max_snr

    def run(self, data_slice, slice_point=None):
        amplitude_fraction = self._calc_amp(data_slice)
        phase_fraction = self._calc_phase(data_slice)
        return amplitude_fraction * phase_fraction
