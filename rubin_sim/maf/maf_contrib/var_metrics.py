__all__ = ("PeriodDeviationMetric",)

import numpy as np
from scipy.signal import lombscargle

from rubin_sim.maf.metrics import BaseMetric

# Example of a *very* simple variabiilty metric
# krughoff@uw.edu, ebellm, ljones


def find_period_ls(times, mags, minperiod=2.0, maxperiod=35.0, nbinmax=10**5, verbose=False):
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
    nbins = int((times.max() - times.min()) / minperiod * 1000)
    if nbins > nbinmax:
        if verbose:
            print("lowered nbins")
        nbins = nbinmax

    # Recenter the magnitude measurements about zero
    dmags = mags - np.median(mags)
    # Create frequency bins
    f = np.linspace(1.0 / maxperiod, 1.0 / minperiod, nbins)

    # Calculate periodogram
    pgram = lombscargle(times, dmags, f)

    idx = np.argmax(pgram)
    # Return period of the bin with the max value in the periodogram
    return 1.0 / f[idx]


class PeriodDeviationMetric(BaseMetric):
    """Measure the percentage deviation of recovered periods for pure sine wave variability (in magnitude)."""

    def __init__(
        self,
        col="observationStartMJD",
        period_min=3.0,
        period_max=35.0,
        n_periods=5,
        mean_mag=21.0,
        amplitude=1.0,
        metric_name="Period Deviation",
        period_check=None,
        **kwargs,
    ):
        """
        Construct an instance of a PeriodDeviationMetric class

        :param col: Name of the column to use for the observation times, commonly 'expMJD'
        :param period_min: Minimum period to test (days)
        :param period_max: Maximimum period to test (days)
        :param period_check: Period to use in the reduce function (days)
        :param mean_mag: Mean value of the lightcurve
        :param amplitude: Amplitude of the variation (mags)
        """
        self.period_min = period_min
        self.period_max = period_max
        self.period_check = period_check
        self.guess_p_min = np.min([self.period_min * 0.8, self.period_min - 1])
        self.guess_p_max = np.max([self.period_max * 1.20, self.period_max + 1])
        self.n_periods = n_periods
        self.mean_mag = mean_mag
        self.amplitude = amplitude
        super(PeriodDeviationMetric, self).__init__(col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        """
        Run the PeriodDeviationMetric
        :param data_slice : Data for this slice.
        :param slice_point: Metadata for the slice. (optional)
        :return: The error in the period estimated from a Lomb-Scargle periodogram
        """

        # Make sure the observation times are sorted
        data = np.sort(data_slice[self.colname])

        # Create 'nPeriods' random periods within range of min to max.
        if self.period_check is not None:
            periods = [self.period_check]
        else:
            periods = self.period_min + np.random.random(self.n_periods) * (self.period_max - self.period_min)
        # Make sure the period we want to check is in there
        periodsdev = np.zeros(np.size(periods), dtype="float")
        for i, period in enumerate(periods):
            omega = 1.0 / period
            # Calculate up the amplitude.
            lc = self.mean_mag + self.amplitude * np.sin(omega * data)
            # Try to recover the period given a window buffered by min of a day or 20% of period value.
            if len(lc) < 3:
                # Too few points to find a period
                return self.badval

            pguess = find_period_ls(data, lc, minperiod=self.guess_p_min, maxperiod=self.guess_p_max)
            periodsdev[i] = (pguess - period) / period

        return {"periods": periods, "periodsdev": periodsdev}

    def reduce_p_dev(self, metric_val):
        """
        At a particular slice_point, return the period deviation for self.period_check.
        If self.period_check is None, just return a random period in the range.
        """
        result = metric_val["periodsdev"][0]
        return result

    def reduce_worst_period(self, metric_val):
        """
        At each slice_point, return the period with the worst period deviation.
        """
        worst_p = np.array(metric_val["periods"])[
            np.where(metric_val["periodsdev"] == metric_val["periodsdev"].max())[0]
        ]
        return worst_p

    def reduce_worst_p_dev(self, metric_val):
        """
        At each slice_point, return the largest period deviation.
        """
        worst_p_dev = np.array(metric_val["periodsdev"])[
            np.where(metric_val["periodsdev"] == metric_val["periodsdev"].max())[0]
        ]
        return worst_p_dev
