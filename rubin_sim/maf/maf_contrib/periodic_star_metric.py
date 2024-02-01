__all__ = ("PeriodicStar", "PeriodicStarMetric")

import warnings

import numpy as np
from scipy.optimize import curve_fit

from rubin_sim.maf.metrics.base_metric import BaseMetric
from rubin_sim.maf.utils import m52snr


class PeriodicStar:
    def __init__(self, filternames):
        self.filternames = filternames

    def __call__(self, t, x0, x1, x2, x3, x4, x5, x6, x7, x8):
        """Approximate a periodic star as a simple sin wave.
        t: array with "time" in days, and "filter" dtype names.
        x0: Period (days)
        x1: Phase (days)
        x2: Amplitude (mag)
        x3: mean u mag
        x4: mean g mag
        x5: mean r mag
        x6: mean i mag
        x7: mean z mag
        x8: mean y mag
        """
        filter2index = {"u": 3, "g": 4, "r": 5, "i": 6, "z": 7, "y": 8}
        filter_names = np.unique(self.filternames)
        mags = x2 * np.sin((t + x1) / x0 * 2.0 * np.pi)
        x = [x0, x1, x2, x3, x4, x5, x6, x7, x8]
        for f in filter_names:
            good = np.where(self.filternames == f)
            mags[good] += x[filter2index[f]]
        return mags


class PeriodicStarMetric(BaseMetric):
    """At each slice_point, run a Monte Carlo simulation to see how
    well a periodic source can be fit. Assumes a simple sin-wave light-curve,
    and generates Gaussain noise based in the 5-sigma limiting depth
    of each observation.

    Parameters
    ----------
    period : `float`
        The period to check, in days.
    amplitude : `float`
        The amplitude of the sinusoidal light curve, in mags.
    phase : `float`
        The phase of the lightcurve at the time of the first observation.
    n_monte : `int`
        The number of noise realizations to make in the Monte Carlo.
    period_tol : `float`
        The fractional tolerance on the period to require in order for a star
        to be considered well-fit
    amp_tol : `float`
        The fractional tolerance on the amplitude.
    means : `list` [`float`]
        The mean magnitudes in ugrizy of the star.
    mag_tol : `float`
        The mean magnitude tolerance, in magnitudes, for the star to be
        considered well-fit.
    n_bands : `int`
        Number of bands that must be within mag_tol.
    seed : `int`
        Random number seed for the noise realizations.
    """

    def __init__(
        self,
        metric_name="PeriodicStarMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        period=10.0,
        amplitude=0.5,
        phase=2.0,
        n_monte=1000,
        period_tol=0.05,
        amp_tol=0.10,
        means=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        mag_tol=0.10,
        n_bands=3,
        seed=42,
        **kwargs,
    ):
        """
        period: days (default 10)
        amplitude: mags (default 1)
        n_monte: number of noise realizations to make in the Monte Carlo
        period_tol: fractional tolerance on the period to demand for a star
        to be considered well-fit
        amp_tol: fractional tolerance on the amplitude to demand
        means: mean magnitudes for ugrizy
        mag_tol: Mean magnitude tolerance (mags)
        n_bands: Number of bands that must be within mag_tol
        seed: random number seed
        """
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        super(PeriodicStarMetric, self).__init__(
            col=[self.mjd_col, self.m5_col, self.filter_col],
            units="Fraction Detected",
            metric_name=metric_name,
            **kwargs,
        )
        self.period = period
        self.amplitude = amplitude
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
        # (need to fit mean magnitudes in each of the available bands -
        # self.means and for a period, amplitude, and phase)
        if data_slice.size < self.means.size + 3:
            return self.badval

        # Generate input for true light curve
        t = np.empty(data_slice.size, dtype=list(zip(["time", "filter"], [float, "|U1"])))
        t["time"] = data_slice[self.mjd_col] - data_slice[self.mjd_col].min()
        t["filter"] = data_slice[self.filter_col]

        # If we are adding a distance modulus to the magnitudes
        if "distMod" in list(slice_point.keys()):
            mags = self.means + slice_point["distMod"]
        else:
            mags = self.means
        true_params = np.append(np.array([self.period, self.phase, self.amplitude]), mags)
        true_obj = PeriodicStar(t["filter"])
        true_lc = true_obj(t["time"], *true_params)

        # Array to hold the fit results
        fits = np.zeros((self.n_monte, true_params.size), dtype=float)
        for i in np.arange(self.n_monte):
            snr = m52snr(true_lc, data_slice[self.m5_col])
            dmag = 2.5 * np.log10(1.0 + 1.0 / snr)
            noise = np.random.randn(true_lc.size) * dmag
            # Suppress warnings about failing on covariance
            fit_obj = PeriodicStar(t["filter"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # If it fails to converge, save values that should fail later
                try:
                    parm_vals, pcov = curve_fit(
                        fit_obj, t["time"], true_lc + noise, p0=true_params, sigma=dmag
                    )
                except RuntimeError:
                    parm_vals = true_params * 0 + np.inf
            fits[i, :] = parm_vals

        # Throw out any magnitude fits if there are no observations
        # in that filter
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

        return frac_recovered
