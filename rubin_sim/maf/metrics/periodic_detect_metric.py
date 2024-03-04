__all__ = ("PeriodicDetectMetric",)

import numpy as np
import scipy

from rubin_sim.maf.utils import m52snr, stellar_mags

from .base_metric import BaseMetric


class PeriodicDetectMetric(BaseMetric):
    """Determine if we would be able to classify an object as
    periodic/non-uniform, using an F-test.

    The idea here is that if a periodic source is aliased, it will be
    indistinguishable from a constant source,
    so we can find a best-fit constant, and if the reduced chi-squared is ~1,
    we know we are aliased.

    Parameters
    ----------
    mjd_col : `str`, opt
        Name of the MJD column in the observations.
    periods : `float` or `np.ndarray`, (N,), opt
        The period of the star (days).
        Can be a single value, or an array.
        If an array, amplitude and starMag should be arrays of equal length.
    amplitudes : `float`, opt
        The amplitude of the stellar variability, (mags).
    m5_col : `str`, opt
        The name of the m5 limiting magnitude column in the observations.
    metric_name : `str`, opt
        The name for the metric.
    starMags : `float`, opt
        The mean magnitude of the star in r (mags).
    sig_level : `float`, opt
        The value to use to compare to the p-value when deciding
        if we can reject the null hypothesis.
    sed_template : `str`, opt
        The stellar SED template to use to generate realistic colors
        (default is an F star, so RR Lyrae-like)

    Returns
    -------
    flag : `int`
        Returns 1 if we would detect star is variable,
        0 if it is well-fit by a constant value.
        If using arrays to test multiple period-amplitude-mag combinations,
        will be the sum of the number of detected stars.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        periods=2.0,
        amplitudes=0.1,
        m5_col="fiveSigmaDepth",
        metric_name="PeriodicDetectMetric",
        filter_col="filter",
        star_mags=20,
        sig_level=0.05,
        sed_template="F",
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        if np.size(periods) == 1:
            self.periods = [periods]
            # Using the same magnitude for all filters.
            # Could expand to fit the mean in each filter.
            self.star_mags = [star_mags]
            self.amplitudes = [amplitudes]
        else:
            self.periods = periods
            self.star_mags = star_mags
            self.amplitudes = amplitudes
        self.sig_level = sig_level
        self.sed_template = sed_template

        super(PeriodicDetectMetric, self).__init__(
            [mjd_col, m5_col, filter_col],
            metric_name=metric_name,
            units="N Detected (0, %i)" % np.size(periods),
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        result = 0
        n_pts = np.size(data_slice[self.mjd_col])
        n_filt = np.size(np.unique(data_slice[self.filter_col]))

        # If we had a correct model with phase, amplitude, period, mean_mags,
        # then chi_squared/DoF would be ~1 with 3+n_filt free parameters.
        # The mean is one free parameter
        p1 = n_filt
        p2 = 3.0 + n_filt
        chi_sq_2 = 1.0 * (n_pts - p2)

        u_filters = np.unique(data_slice[self.filter_col])

        if n_pts > p2:
            for period, starMag, amplitude in zip(self.periods, self.star_mags, self.amplitudes):
                chi_sq_1 = 0
                mags = stellar_mags(self.sed_template, rmag=starMag)
                for filtername in u_filters:
                    in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
                    lc = (
                        amplitude * np.sin(data_slice[self.mjd_col][in_filt] * (np.pi * 2) / period)
                        + mags[filtername]
                    )
                    snr = m52snr(lc, data_slice[self.m5_col][in_filt])
                    delta_m = 2.5 * np.log10(1.0 + 1.0 / snr)
                    weights = 1.0 / (delta_m**2)
                    weighted_mean = np.sum(weights * lc) / np.sum(weights)
                    chi_sq_1 += np.sum(((lc - weighted_mean) ** 2 / delta_m**2))
                # Yes, I'm fitting magnitudes rather than flux.
                # At least I feel kinda bad about it.
                # F-test for nested models Regression problems:
                # https://en.wikipedia.org/wiki/F-test
                f_numerator = (chi_sq_1 - chi_sq_2) / (p2 - p1)
                f_denom = 1.0
                # This is just reduced chi-squared for the more
                # complicated model, so should be 1.
                f_val = f_numerator / f_denom
                # Has DoF (p2-p1, n-p2)
                # https://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python/21503346
                p_value = scipy.stats.f.sf(f_val, p2 - p1, n_pts - p2)
                if np.isfinite(p_value):
                    if p_value < self.sig_level:
                        result += 1

        return result
