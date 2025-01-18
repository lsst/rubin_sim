__all__ = ("PeriodicStarModulationMetric",)

import random
import warnings

import numpy as np
from scipy.optimize import curve_fit

from rubin_sim.maf.metrics.base_metric import BaseMetric
from rubin_sim.maf.utils import m52snr

from .periodic_star_metric import PeriodicStar

""" This metric is based on the PeriodicStar metric
    It was modified in a way to reproduce attempts to identify
    phase modulation (Blazhko effect) in RR Lyrae stars.
    We are not implementing a period/ phase modulation in the light curve,
    but rather use short baselines (e.g.: 20 days) of observations to test
    how well we can recover the period, phase and amplitude.
    We do this as such an attempt is also useful for other purposes,
    i.e. if we want to test whether we can just recover period, phase
    and amplitude from short baselines at all, without necessarily having
    in mind to look for period/ phase modulations.
    Like in the PeriodicStar metric, the light curve of an RR Lyrae star,
    or a periodic star in general, is approximated as a simple sin wave.
    Other solutions might make use of light curve templates
    to generate light curves.
    Two other modifications we introduced are:
    In contrast to the PeriodicStar metric, we allow for a random phase
    offset to mimic observation starting at random phase.
    Also, we vary the periods and amplitudes within +/- 10 % to allow
    for a more realistic sample of variable stars.

    This metric is based on the cadence note:
    N. Hernitschek, K. Stassun, LSST Cadence Note:
    "Cadence impacts on reliable classification of standard-candle
    variable stars (2021)"
     https://docushare.lsst.org/docushare/dsweb/Get/Document-37673
"""


class PeriodicStarModulationMetric(BaseMetric):
    """Evaluate how well a periodic source can be fit on a short baseline,
    using a Monte Carlo simulation.

    At each slice_point, run a Monte Carlo simulation to see how well a
    periodic source can be fit.
    Assumes a simple sin-wave light-curve, and generates Gaussain noise
    based in the 5-sigma limiting depth of each observation.
    Light curves are evaluated piecewise to test how well we can recover
    the period, phase and amplitude from shorter baselines.
    We allow for a random phase offset to mimic observation starting
    at random phase. Also, we vary the periods and amplitudes
    within +/- 10 % to allow for a more realistic sample of variable stars.

    Parameters
    ----------
    period : `float`, opt
        days (default 10)
    amplitude : `float`, opt
        mags (default 0.5)
    phase : `float`, opt
        days (default 2.)
    random_phase : `bool`, opt
        a random phase is assigned (default False)
    time_interval : `float`, opt
        days (default 50);
        the interval over which we want to evaluate the light curve
    n_monte : `int`, opt
        number of noise realizations to make in the Monte Carlo (default 1000)
    period_tol : `float`, opt
        fractional tolerance on the period to demand
        for a star to be considered well-fit (default 0.05)
    amp_tol : `float`, opt
        fractional tolerance on the amplitude to demand (default 0.10)
    means : `list` of `float`, opt
        mean magnitudes for ugrizy (default all 20)
    mag_tol : `float`, opt
        Mean magnitude tolerance (mags) (default 0.1)
    n_bands : `int`, opt
        Number of bands that must be within mag_tol (default 3)
    seed : `int`, opt
        random number seed (default 42)
    """

    def __init__(
        self,
        metric_name="PeriodicStarModulationMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        period=10.0,
        amplitude=0.5,
        phase=2.0,
        random_phase=False,
        time_interval=50,
        n_monte=1000,
        period_tol=0.05,
        amp_tol=0.10,
        means=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        mag_tol=0.10,
        n_bands=3,
        seed=42,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        super(PeriodicStarModulationMetric, self).__init__(
            col=[self.mjd_col, self.m5_col, self.filter_col],
            units="Fraction Detected",
            metric_name=metric_name,
            **kwargs,
        )
        self.period = period
        self.amplitude = amplitude
        self.time_interval = time_interval
        if random_phase:
            self.phase = np.nan
        else:
            self.phase = phase
        self.n_monte = n_monte
        self.period_tol = period_tol
        self.amp_tol = amp_tol
        self.means = np.array(means)
        self.mag_tol = mag_tol
        self.n_bands = n_bands
        np.random.seed(seed)
        self.filter2index = {"u": 3, "g": 4, "r": 5, "i": 6, "z": 7, "y": 8}

    def run(self, data_slice, slice_point=None):
        # Bail if we don't have enough points
        # (need to fit mean magnitudes in each of the available bands
        # - self.means and for a period, amplitude, and phase)
        if data_slice.size < self.means.size + 3:
            return self.badval

        # Generate input for true light curve

        lightcurvelength = data_slice.size

        t = np.empty(lightcurvelength, dtype=list(zip(["time", "filter"], [float, "|U1"])))
        t["time"] = data_slice[self.mjd_col] - data_slice[self.mjd_col].min()
        t["filter"] = data_slice[self.filter_col]
        m5 = data_slice[self.m5_col]

        lightcurvelength_days = self.time_interval

        # evaluate light curves piecewise in subruns
        subruns = int(np.max(t["time"]) / lightcurvelength_days)

        # print('number of subruns: ', subruns)
        frac_recovered_list = []

        for subrun_idx in range(0, subruns):
            good = (t["time"] >= (lightcurvelength_days * (subrun_idx))) & (
                t["time"] <= (lightcurvelength_days * (subrun_idx + 1))
            )
            t_subrun = t[good]
            m5_subrun = m5[good]
            if t_subrun["time"].size > 0:
                # If we are adding a distance modulus to the magnitudes
                if "distMod" in list(slice_point.keys()):
                    mags = self.means + slice_point["distMod"]
                else:
                    mags = self.means
                # slightly different periods and amplitudes (+/- 10 %)
                # to mimic true stars. random phase offsets to mimic
                # observation starting at random phase
                true_period = random.uniform(0.9, 1.1) * self.period
                true_amplitude = random.uniform(0.9, 1.1) * self.amplitude
                if np.isnan(self.phase):
                    # a random phase (in days) should be assigned
                    true_phase = random.uniform(0, 1) * self.period
                else:
                    true_phase = self.phase

                true_params = np.append(np.array([true_period, true_phase, true_amplitude]), mags)
                true_obj = PeriodicStar(t_subrun["filter"])
                true_lc = true_obj(t_subrun["time"], *true_params)

                # Array to hold the fit results
                fits = np.zeros((self.n_monte, true_params.size), dtype=float)
                for i in np.arange(self.n_monte):
                    snr = m52snr(true_lc, m5_subrun)
                    dmag = 2.5 * np.log10(1.0 + 1.0 / snr)
                    noise = np.random.randn(true_lc.size) * dmag
                    # Suppress warnings about failing on covariance
                    fit_obj = PeriodicStar(t_subrun["filter"])
                    # check if we have enough points
                    if np.size(true_params) >= np.size(fit_obj):
                        parm_vals = true_params * 0 + np.inf
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # If it fails to converge,
                            # save values that should fail later
                            try:
                                parm_vals, pcov = curve_fit(
                                    fit_obj,
                                    t_subrun["time"],
                                    true_lc + noise,
                                    p0=true_params,
                                    sigma=dmag,
                                )
                            except RuntimeError:
                                parm_vals = true_params * 0 + np.inf
                    fits[i, :] = parm_vals

                # Throw out any magnitude fits if there are no
                # observations in that filter
                ufilters = np.unique(data_slice[self.filter_col])
                if ufilters.size < 9:
                    for key in list(self.filter2index.keys()):
                        if key not in ufilters:
                            fits[:, self.filter2index[key]] = -np.inf

                # Find the fraction of fits that meet the "well-fit" criteria
                period_frac_err = np.abs((fits[:, 0] - true_params[0]) / true_params[0])
                amp_frac_err = np.abs((fits[:, 2] - true_params[2]) / true_params[2])
                mag_err = np.abs(fits[:, 3:] - true_params[3:])
                n_bands = np.zeros(mag_err.shape, dtype=int)
                n_bands[np.where(mag_err <= self.mag_tol)] = 1
                n_bands = np.sum(n_bands, axis=1)
                n_recovered = np.size(
                    np.where(
                        (period_frac_err <= self.period_tol)
                        & (amp_frac_err <= self.amp_tol)
                        & (n_bands >= self.n_bands)
                    )[0]
                )
                frac_recovered = float(n_recovered) / self.n_monte
                frac_recovered_list.append(frac_recovered)

        frac_recovered = np.sum(frac_recovered_list) / (len(frac_recovered_list))
        return frac_recovered
