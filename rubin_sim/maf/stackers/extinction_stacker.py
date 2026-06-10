__all__ = ("ExtinctionStacker", "CloudExtinctionStacker")

import logging
import warnings

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor

from rubin_sim.phot_utils import predicted_zeropoint

from .base_stacker import BaseStacker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _make_default_band_priors():
    """Build the default per-band prior dictionary from the throughput model.

    Derives ``expected_k`` and ``expected_zp_X0`` for each of the six
    Rubin bands by evaluating `~rubin_sim.phot_utils.predicted_zeropoint`
    at two airmasses on a 1-second exposure time scale.

    Returns
    -------
    band_priors : `dict`
        Keys are single-character band names; values are dicts with keys
        ``expected_k``, ``k_fraction_tolerance``, ``expected_zp_X0``,
        and ``zp_window``.  See `ExtinctionStacker` for a description of
        each key.
    """
    band_priors = {}
    for band in "ugrizy":
        zp_at_X1 = predicted_zeropoint(band, airmass=1.0, exptime=1.0)
        zp_at_X2 = predicted_zeropoint(band, airmass=2.0, exptime=1.0)
        k_predicted = zp_at_X1 - zp_at_X2  # positive extinction coefficient
        zp_at_X0 = zp_at_X1 + k_predicted
        band_priors[band] = {
            "expected_k": k_predicted,
            "k_fraction_tolerance": 0.5,
            "expected_zp_X0": zp_at_X0,
            "zp_window": 0.5,
        }
    return band_priors


DEFAULT_BAND_PRIORS = _make_default_band_priors()


def _limits_from_prior(prior):
    """Compute ``k_min``, ``k_max``, ``zp_min``, ``zp_max`` from a prior dict.

    Parameters
    ----------
    prior : `dict`
        Must contain ``expected_k``, ``k_fraction_tolerance``,
        ``expected_zp_X0``, and ``zp_window``.

    Returns
    -------
    k_min, k_max, zp_min, zp_max : `float`
    """
    k = prior["expected_k"]
    tol = prior["k_fraction_tolerance"]
    zp0 = prior["expected_zp_X0"]
    win = prior["zp_window"]
    k_min = max(0.0, (1.0 - tol) * k)
    k_max = (1.0 + tol) * k
    zp_min = zp0 - win
    zp_max = zp0 + win
    return k_min, k_max, zp_min, zp_max


def _fit_extinction_one_group(
    airmass,
    zero_point,
    k_min,
    k_max,
    zp_min,
    zp_max,
    zp_prior,
    residual_threshold,
    min_inliers,
    min_inlier_fraction,
    premask=True,
):
    """Fit the extinction coefficient for a single band/dayObs group.

    Fits the linear model::

        zero_point = fitted_zeropoint - extinction_k * airmass

    Three stages are attempted in order, stopping as soon as one succeeds:

    **Stage 1 — RANSAC** free two-parameter linear regression.  The fit is
    accepted if both ``extinction_k`` and ``fitted_zeropoint`` fall within
    their physical bounds and the inlier count passes the quality thresholds.

    **Stage 2 — Theil-Sen** robust regression, used as a fallback when RANSAC
    fails or its result is out of bounds.  Subject to the same acceptance
    criteria.

    **Stage 3 — Anchored-clip + bounded Huber regression**.  Used when both
    free fits fail.  ``zp_prior`` is used as a fixed anchor to identify
    inliers via a median-based k estimate; bounded
    ``scipy.optimize.least_squares`` with a Huber loss then fits both
    parameters jointly within their physical bounds, so an out-of-range
    intercept can never cause rejection.

    Parameters
    ----------
    airmass : `np.ndarray`, shape (N,)
        Airmass values for the group.
    zero_point : `np.ndarray`, shape (N,)
        Photometric zeropoints for the group, normalized to a 1-second
        exposure time (i.e., ``raw_zp - 2.5 * log10(exptime)``).
    k_min : `float`
        Minimum physically plausible extinction coefficient (mag/airmass).
    k_max : `float`
        Maximum physically plausible extinction coefficient (mag/airmass).
    zp_min : `float`
        Minimum physically plausible zenith zeropoint (mag).
    zp_max : `float`
        Maximum physically plausible zenith zeropoint (mag).
    zp_prior : `float`
        Expected zenith zeropoint used as the fixed anchor in Stage 3
        (typically ``expected_zp_X0`` from the band prior, i.e. the
        midpoint of ``[zp_min, zp_max]``).
    residual_threshold : `float`
        RANSAC inlier residual threshold in magnitudes.  Also used as the
        Huber ``f_scale`` in Stage 3; the Stage 3 clip threshold is
        ``2 * residual_threshold``.
    min_inliers : `int`
        Minimum number of inlier visits required for a valid fit.
    min_inlier_fraction : `float`
        Minimum fraction of the original visit count that must be inliers.
    premask : `bool`, optional
        If True (default), remove visits whose zeropoint is implausibly
        low (i.e., consistent with extinction greater than ``k_max``)
        before fitting.

    Returns
    -------
    extinction_k : `float`
        Fitted extinction coefficient, or `np.nan` if all stages failed.
    fitted_zeropoint : `float`
        Fitted zenith zeropoint, or `np.nan` if all stages failed.
    """
    orig_npts = len(airmass)
    if orig_npts == 0:
        logger.debug("Extinction fit: no data points supplied; returning NaN.")
        return np.nan, np.nan

    X = airmass.reshape(-1, 1)
    y = zero_point.copy()

    # Pre-mask visits whose raw zeropoint is too low to be plausible even
    # under maximum allowed extinction (proxy for heavily clouded exposures).
    if premask:
        min_expected_zp = zp_min - k_max * airmass
        keep = y > min_expected_zp
        X = X[keep]
        y = y[keep]

    npts = len(y)
    if npts < min_inliers or (npts / orig_npts) < min_inlier_fraction:
        logger.debug(
            "Extinction fit: too few points after pre-masking (%d of %d survive; "
            "min_inliers=%d, min_inlier_fraction=%.2f); returning NaN.",
            npts,
            orig_npts,
            min_inliers,
            min_inlier_fraction,
        )
        return np.nan, np.nan

    # --- Stage 1: RANSAC robust linear regression ---
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=residual_threshold,
        max_trials=10,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ransac.fit(X, y)
        num_inliers = int(ransac.inlier_mask_.sum())
        k = float(-ransac.estimator_.coef_[0])
        zp = float(ransac.estimator_.intercept_)

        ok = (
            (k_min <= k <= k_max)
            and (zp_min <= zp <= zp_max)
            and (num_inliers >= min_inliers)
            and (num_inliers >= min_inlier_fraction * orig_npts)
        )
        if ok:
            logger.debug(
                "Extinction fit stage 1 (RANSAC) succeeded: k=%.4f, zp=%.4f, "
                "inliers=%d/%d.",
                k,
                zp,
                num_inliers,
                orig_npts,
            )
            return k, zp
    except Exception as exc:
        logger.debug("Extinction fit stage 1 (RANSAC) raised an exception: %s.", exc)

    # --- Stage 2: Theil-Sen fallback ---
    try:
        ts = TheilSenRegressor(max_subpopulation=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts.fit(X, y)
        k = float(-ts.coef_[0])
        zp = float(ts.intercept_)
        if (k_min <= k <= k_max) and (zp_min <= zp <= zp_max):
            logger.debug(
                "Extinction fit stage 2 (Theil-Sen) succeeded: k=%.4f, zp=%.4f.",
                k,
                zp,
            )
            return k, zp
    except Exception as exc:
        logger.debug("Extinction fit stage 2 (Theil-Sen) raised an exception: %s.", exc)

    # --- Stage 3: Anchored-clip + bounded Huber regression ---
    #
    # Fix the intercept to zp_prior and compute the implied k for each visit,
    # then take the median as a robust anchor estimate.  Visits whose residual
    # against the anchored model exceeds 2 * residual_threshold are clipped.
    # The surviving inlier set is passed to a bounded least_squares fit with a
    # Huber loss that jointly optimises both k and zp within their physical
    # bounds, so an out-of-range intercept can never cause rejection.

    airmass_pm = X.ravel()  # post-premask 1-D array
    zp_pm = y

    # Guard against any airmass == 0 entries (shouldn't occur in practice).
    nonzero = airmass_pm > 0.0
    if nonzero.sum() < min_inliers:
        logger.debug(
            "Extinction fit stage 3: too few non-zero airmass points (%d); "
            "returning NaN.",
            int(nonzero.sum()),
        )
        return np.nan, np.nan

    k_implied = (zp_prior - zp_pm[nonzero]) / airmass_pm[nonzero]
    k_anchor = float(np.clip(np.median(k_implied), k_min, k_max))

    # Clip outliers relative to the anchored model.
    clip_threshold = 2.0 * residual_threshold
    residuals = zp_pm - (zp_prior - k_anchor * airmass_pm)
    inlier_mask = np.abs(residuals) <= clip_threshold

    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_inliers or (n_inliers / orig_npts) < min_inlier_fraction:
        logger.debug(
            "Extinction fit stage 3: too few inliers after anchor-clip (%d of %d; "
            "min_inliers=%d, min_inlier_fraction=%.2f); returning NaN.",
            n_inliers,
            orig_npts,
            min_inliers,
            min_inlier_fraction,
        )
        return np.nan, np.nan

    airmass_in = airmass_pm[inlier_mask]
    zp_in = zp_pm[inlier_mask]

    def _residuals(params):
        k_fit, zp_fit = params
        return zp_in - (zp_fit - k_fit * airmass_in)

    try:
        result = least_squares(
            _residuals,
            x0=[k_anchor, zp_prior],
            bounds=([k_min, zp_min], [k_max, zp_max]),
            loss="huber",
            f_scale=residual_threshold,
            method="trf",
        )
        if result.success or result.cost < np.inf:
            k_fit, zp_fit = float(result.x[0]), float(result.x[1])
            logger.debug(
                "Extinction fit stage 3 (anchored-clip + bounded Huber) succeeded: "
                "k=%.4f, zp=%.4f, inliers=%d/%d.",
                k_fit,
                zp_fit,
                n_inliers,
                orig_npts,
            )
            return k_fit, zp_fit
        logger.debug(
            "Extinction fit stage 3: least_squares converged to infinite cost; "
            "returning NaN."
        )
    except Exception as exc:
        logger.debug(
            "Extinction fit stage 3 (anchored-clip + bounded Huber) raised an "
            "exception: %s.",
            exc,
        )

    logger.debug("Extinction fit: all three stages failed; returning NaN.")
    return np.nan, np.nan


class ExtinctionStacker(BaseStacker):
    """Fit atmospheric extinction per band/dayObs and add to each visit.

    For each unique (band, dayObs) combination, normalizes the raw
    zeropoints to a 1-second exposure time and fits the linear model::

        zp_1s = fitted_zeropoint - extinction_k * airmass

    where ``zp_1s = zero_point_median - 2.5 * log10(visitExposureTime)``.

    Three fitting stages are attempted in order for each group:

    1. **RANSAC** free two-parameter regression with physical bounds check.
    2. **Theil-Sen** robust regression with physical bounds check.
    3. **Anchored-clip + bounded Huber regression**: the expected zenith
       zeropoint ``expected_zp_X0`` from the band prior is used as a fixed
       anchor to identify inliers via a median-based slope estimate;
       `scipy.optimize.least_squares` then fits both parameters with hard
       physical bounds so an out-of-range intercept can never cause
       rejection.

    Parameters
    ----------
    zero_point_col : `str`, optional
        Name of the raw photometric zeropoint column.
        Default ``'zero_point_median'``.
    airmass_col : `str`, optional
        Name of the airmass column.  Default ``'airmass'``.
    band_col : `str`, optional
        Name of the band (filter) column.  Default ``'band'``.
    exptime_col : `str`, optional
        Name of the exposure time column (seconds).
        Default ``'visitExposureTime'``.
    day_obs_col : `str`, optional
        Name of the dayObs column (integer YYYYMMDD, as defined by
        SITCOMTN-32).  Default ``'dayObs'``.
    band_priors : `dict` or `None`, optional
        Per-band prior parameters used to derive physical limits for the
        fit (on a 1-second zeropoint scale).  Keys are single-character
        band names (``'u'``, ``'g'``, ``'r'``, ``'i'``, ``'z'``,
        ``'y'``); values are dicts with keys:

        ``expected_k`` : `float`
            Expected atmospheric extinction coefficient (mag/airmass).
        ``k_fraction_tolerance`` : `float`
            Allowed fractional deviation from ``expected_k``; the fit
            bounds become
            ``[(1 - tol) * expected_k, (1 + tol) * expected_k]``
            (floored at 0).
        ``expected_zp_X0`` : `float`
            Expected zenith zeropoint at airmass 0, 1-second exposure
            (mag).  Used as both the centre of the zeropoint window and
            the fixed anchor in Stage 3.
        ``zp_window`` : `float`
            Half-width of the allowed zeropoint range (mag); the fit
            bounds become
            ``[expected_zp_X0 - zp_window, expected_zp_X0 + zp_window]``.

        If ``None``, the Rubin Observatory defaults
        (`DEFAULT_BAND_PRIORS`) are used.
    residual_threshold : `float`, optional
        RANSAC inlier residual threshold in magnitudes.  Also used as
        the Huber ``f_scale`` in Stage 3.  Default 0.05.
    min_inliers : `int`, optional
        Minimum number of inlier visits required for a valid fit.
        Default 10.
    min_inlier_fraction : `float`, optional
        Minimum fraction of visits (before pre-masking) that must be
        inliers.  Default 0.1.
    premask : `bool`, optional
        If True (default), visits with implausibly low zeropoints are
        removed before fitting to reduce the impact of heavily clouded
        frames.

    Notes
    -----
    The fitted ``extinction_k`` and ``fitted_zeropoint`` (at 1-second) are
    assigned to every visit belonging to that group.  Visits from groups
    where all fitting stages failed (too few data points, or no physically
    plausible solution found) receive `NaN` in both columns.

    This stacker is intended for use with consdb visit tables, which
    contain the ``zero_point_median`` column (raw, uncorrected photometric
    zeropoint) that is absent from opsim output.

    Two columns are added to the data:

    ``extinction_k``
        Atmospheric extinction coefficient fitted for the visit's band
        and night (mag/airmass).
    ``fitted_zeropoint``
        Instrument+atmosphere zenith zeropoint fitted for the visit's
        band and night, normalized to a 1-second exposure time (mag).

    The ``dayObs`` column (integer YYYYMMDD) must already be present in
    the data.  If it is not, run `DayObsStacker` first.
    """

    cols_added = ["extinction_k", "fitted_zeropoint"]

    def __init__(
        self,
        zero_point_col="zero_point_median",
        airmass_col="airmass",
        band_col="band",
        exptime_col="visitExposureTime",
        day_obs_col="dayObs",
        band_priors=None,
        residual_threshold=0.05,
        min_inliers=10,
        min_inlier_fraction=0.1,
        premask=True,
    ):
        self.zero_point_col = zero_point_col
        self.airmass_col = airmass_col
        self.band_col = band_col
        self.exptime_col = exptime_col
        self.day_obs_col = day_obs_col
        self.band_priors = band_priors if band_priors is not None else DEFAULT_BAND_PRIORS
        self.residual_threshold = residual_threshold
        self.min_inliers = min_inliers
        self.min_inlier_fraction = min_inlier_fraction
        self.premask = premask

        self.cols_req = [zero_point_col, airmass_col, band_col, exptime_col, day_obs_col]
        self.units = ["mag/airmass", "mag"]
        self.cols_added_dtypes = [float, float]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data

        # Check that the required source columns are actually present.
        # They will be absent when this stacker is exercised against opsim
        # data that does not carry consdb columns such as zero_point_median.
        col_names = sim_data.dtype.names if hasattr(sim_data, "dtype") else list(sim_data.keys())
        for col in self.cols_req:
            if col not in col_names:
                logger.warning(
                    "ExtinctionStacker: required column '%s' not found in data; "
                    "extinction_k and fitted_zeropoint will be NaN.",
                    col,
                )
                sim_data["extinction_k"] = np.nan
                sim_data["fitted_zeropoint"] = np.nan
                return sim_data

        sim_data["extinction_k"] = np.nan
        sim_data["fitted_zeropoint"] = np.nan

        for band in np.unique(sim_data[self.band_col]):
            prior = self.band_priors.get(band)
            if prior is None:
                logger.warning(
                    "No band priors defined for band '%s'; skipping extinction fit.",
                    band,
                )
                continue

            k_min, k_max, zp_min, zp_max = _limits_from_prior(prior)
            zp_prior = prior["expected_zp_X0"]

            band_mask = sim_data[self.band_col] == band

            for day_obs in np.unique(sim_data[self.day_obs_col][band_mask]):
                group_mask = band_mask & (sim_data[self.day_obs_col] == day_obs)
                airmass = sim_data[self.airmass_col][group_mask].astype(float)
                raw_zp = sim_data[self.zero_point_col][group_mask].astype(float)
                exptime = sim_data[self.exptime_col][group_mask].astype(float)

                # Normalize to 1-second exposure time
                zp_1s = raw_zp - 2.5 * np.log10(exptime)

                k, zp = _fit_extinction_one_group(
                    airmass,
                    zp_1s,
                    k_min=k_min,
                    k_max=k_max,
                    zp_min=zp_min,
                    zp_max=zp_max,
                    zp_prior=zp_prior,
                    residual_threshold=self.residual_threshold,
                    min_inliers=self.min_inliers,
                    min_inlier_fraction=self.min_inlier_fraction,
                    premask=self.premask,
                )

                if np.isnan(k):
                    logger.debug(
                        "Extinction fit failed for band=%s dayObs=%s (%d visits).",
                        band,
                        day_obs,
                        int(group_mask.sum()),
                    )

                sim_data["extinction_k"][group_mask] = k
                sim_data["fitted_zeropoint"][group_mask] = zp

        return sim_data


class CloudExtinctionStacker(BaseStacker):
    """Estimate extinction due to clouds for each visit.

    Computes the difference between the expected clear-sky zeropoint
    (accounting for atmospheric extinction along the line of sight) and
    the observed raw zeropoint::

        cloud_extinction = (fitted_zeropoint - extinction_k * airmass)
                           - zero_point_median

    A positive ``cloud_extinction`` means the visit was dimmer than the
    photometric model predicts (i.e., clouds were present).  A value of
    zero means the visit was consistent with a clear sky.

    When the `ExtinctionStacker` was unable to produce a photometric
    solution for a given visit (``extinction_k`` is NaN), the predicted
    zeropoint from `rubin_sim.phot_utils.predicted_zeropoint` (plus an
    optional per-band offset) is used as the fallback expected zeropoint
    for that visit.

    Parameters
    ----------
    zero_point_col : `str`, optional
        Name of the raw photometric zeropoint column.
        Default ``'zero_point_median'``.
    airmass_col : `str`, optional
        Name of the airmass column.  Default ``'airmass'``.
    band_col : `str`, optional
        Name of the band (filter) column.  Default ``'band'``.
    exptime_col : `str`, optional
        Name of the exposure time column (seconds).
        Default ``'visitExposureTime'``.
    extinction_k_col : `str`, optional
        Name of the fitted extinction coefficient column (output of
        `ExtinctionStacker`).  Default ``'extinction_k'``.
    fitted_zeropoint_col : `str`, optional
        Name of the fitted zenith zeropoint column (output of
        `ExtinctionStacker`).  Default ``'fitted_zeropoint'``.
    zeropoint_offsets : `dict` or `None`, optional
        Per-band offsets (in magnitudes) to apply to the
        ``predicted_zeropoint`` fallback values, accounting for
        recently measured differences between the throughput model and
        the actual instrument.  Keys are single-character band names;
        values are floats added to the predicted zeropoint.  Bands not
        present in the dict receive no offset.  If ``None`` (default),
        no offsets are applied.

    Notes
    -----
    This stacker depends on the columns produced by `ExtinctionStacker`
    (``extinction_k`` and ``fitted_zeropoint``).  It should be run after
    `ExtinctionStacker`, or the ``extinction_k`` and ``fitted_zeropoint``
    columns should already be present in the data.

    One column is added:

    ``cloud_extinction``
        Estimated extinction due to clouds in magnitudes.  Positive values
        indicate cloud absorption; values near zero indicate photometric
        conditions.  Negative values (visit brighter than the model) are
        possible due to noise or model imperfections.
    """

    cols_added = ["cloud_extinction"]

    def __init__(
        self,
        zero_point_col="zero_point_median",
        airmass_col="airmass",
        band_col="band",
        exptime_col="visitExposureTime",
        extinction_k_col="extinction_k",
        fitted_zeropoint_col="fitted_zeropoint",
        zeropoint_offsets=None,
    ):
        self.zero_point_col = zero_point_col
        self.airmass_col = airmass_col
        self.band_col = band_col
        self.exptime_col = exptime_col
        self.extinction_k_col = extinction_k_col
        self.fitted_zeropoint_col = fitted_zeropoint_col
        self.zeropoint_offsets = zeropoint_offsets if zeropoint_offsets is not None else {}

        self.cols_req = [
            zero_point_col,
            airmass_col,
            band_col,
            exptime_col,
            extinction_k_col,
            fitted_zeropoint_col,
        ]
        self.units = ["mag"]
        self.cols_added_dtypes = [float]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data

        # Check that the required source columns are actually present.
        col_names = sim_data.dtype.names if hasattr(sim_data, "dtype") else list(sim_data.keys())
        for col in self.cols_req:
            if col not in col_names:
                logger.warning(
                    "CloudExtinctionStacker: required column '%s' not found "
                    "in data; cloud_extinction will be NaN.",
                    col,
                )
                sim_data["cloud_extinction"] = np.nan
                return sim_data

        airmass = sim_data[self.airmass_col].astype(float)
        zero_point = sim_data[self.zero_point_col].astype(float)
        exptime = sim_data[self.exptime_col].astype(float)
        extinction_k = sim_data[self.extinction_k_col].astype(float)
        fitted_zp = sim_data[self.fitted_zeropoint_col].astype(float)

        # fitted_zeropoint is on a 1-second scale, so the expected raw
        # zeropoint at the actual exposure time and airmass is:
        #   expected = fitted_zeropoint - k * airmass + 2.5 * log10(exptime)
        expected_zp = fitted_zp - extinction_k * airmass + 2.5 * np.log10(exptime)

        # For visits where the extinction fit failed (NaN), use the
        # predicted_zeropoint function (plus any configured offset) as
        # fallback.  predicted_zeropoint already accounts for airmass
        # and exptime.
        nan_mask = np.isnan(extinction_k)
        if np.any(nan_mask):
            for band in np.unique(sim_data[self.band_col][nan_mask]):
                band_nan_mask = nan_mask & (sim_data[self.band_col] == band)
                band_airmass = airmass[band_nan_mask]
                band_exptime = exptime[band_nan_mask]
                try:
                    fallback_zp = predicted_zeropoint(band, band_airmass, band_exptime)
                    fallback_zp += self.zeropoint_offsets.get(band, 0.0)
                    expected_zp[band_nan_mask] = fallback_zp
                except KeyError:
                    logger.warning(
                        "CloudExtinctionStacker: predicted_zeropoint does not "
                        "support band '%s'; cloud_extinction will be NaN for "
                        "those visits.",
                        band,
                    )
                    expected_zp[band_nan_mask] = np.nan

        # cloud_extinction = expected_zp - observed_zp
        sim_data["cloud_extinction"] = expected_zp - zero_point

        return sim_data
