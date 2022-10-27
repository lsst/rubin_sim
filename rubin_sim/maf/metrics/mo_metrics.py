from builtins import zip
import numpy as np

from .base_metric import BaseMetric

__all__ = [
    "BaseMoMetric",
    "NObsMetric",
    "NObsNoSinglesMetric",
    "NNightsMetric",
    "ObsArcMetric",
    "DiscoveryMetric",
    "DiscoveryNChancesmetric",
    "DiscoveryNObsmetric",
    "DiscoveryTimemetric",
    "DiscoveryDistancemetric",
    "DiscoveryRadecmetric",
    "DiscoveryEclonlatmetric",
    "DiscoveryVelocitymetric",
    "ActivityOverTimeMetric",
    "ActivityOverPeriodMetric",
    "MagicDiscoveryMetric",
    "HighVelocityMetric",
    "HighVelocityNightsMetric",
    "LightcurveinversionAsteroidmetric",
    "ColorAsteroidmetric",
    "InstantaneousColorMetric",
    "LightcurvecolorOutermetric",
    "PeakVMagMetric",
    "KnownObjectsMetric",
]


def _set_vis(sso_obs, snr_limit, snr_col, vis_col):
    if snr_limit is not None:
        vis = np.where(sso_obs[snr_col] >= snr_limit)[0]
    else:
        vis = np.where(sso_obs[vis_col] > 0)[0]
    return vis


class BaseMoMetric(BaseMetric):
    """Base class for the moving object metrics.
    Intended to be used with the Moving Object Slicer."""

    def __init__(
        self,
        cols=None,
        metric_name=None,
        units="#",
        badval=0,
        comment=None,
        child_metrics=None,
        app_mag_col="appMag",
        app_mag_v_col="appMagV",
        m5_col="fiveSigmaDepth",
        night_col="night",
        mjd_col="observationStartMJD",
        snr_col="SNR",
        vis_col="vis",
        ra_col="ra",
        dec_col="dec",
        seeing_col="seeingFwhmGeom",
        exp_time_col="visitExposureTime",
        filter_col="filter",
    ):
        # Set metric name.
        self.name = metric_name
        if self.name is None:
            self.name = self.__class__.__name__.replace("Metric", "", 1)
        # Set badval and units, leave space for 'comment' (tied to displayDict).
        self.badval = badval
        self.units = units
        self.comment = comment
        # Set some commonly used column names.
        self.m5_col = m5_col
        self.app_mag_col = app_mag_col
        self.app_mag_v_col = app_mag_v_col
        self.night_col = night_col
        self.mjd_col = mjd_col
        self.snr_col = snr_col
        self.vis_col = vis_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.seeing_col = seeing_col
        self.exp_time_col = exp_time_col
        self.filter_col = filter_col
        self.cols_req = [
            self.app_mag_col,
            self.m5_col,
            self.night_col,
            self.mjd_col,
            self.snr_col,
            self.vis_col,
        ]
        if cols is not None:
            for col in cols:
                self.cols_req.append(col)

        if child_metrics is None:
            try:
                if not isinstance(self.child_metrics, dict):
                    raise ValueError(
                        "self.child_metrics must be a dictionary (possibly empty)"
                    )
            except AttributeError:
                self.child_metrics = {}
                self.metric_dtype = "float"
        else:
            if not isinstance(child_metrics, dict):
                raise ValueError("childmetrics must be provided as a dictionary.")
            self.child_metrics = child_metrics
            self.metric_dtype = "object"
        self.shape = 1

    def run(self, sso_obs, orb, hval):
        """Calculate the metric value.

        Parameters
        ----------
        sso_obs: np.ndarray
            The input data to the metric (same as the parent metric).
        orb: np.ndarray
            The information about the orbit for which the metric is being calculated.
        hval : float
            The H value for which the metric is being calculated.

        Returns
        -------
        float or np.ndarray or dict
        """
        raise NotImplementedError


class BaseChildMetric(BaseMoMetric):
    """Base class for child metrics.

    Parameters
    ----------
    parentDiscoveryMetric: BaseMoMetric
        The 'parent' metric which generated the metric data used to calculate this 'child' metric.
    badval: float, optional
        Value to return when metric cannot be calculated.
    """

    def __init__(self, parent_discovery_metric, badval=0, **kwargs):
        super().__init__(badval=badval, **kwargs)
        self.parent_metric = parent_discovery_metric
        self.child_metrics = {}
        if "metricDtype" in kwargs:
            self.metric_dtype = kwargs["metricDtype"]
        else:
            self.metric_dtype = "float"

    def run(self, sso_obs, orb, hval, metric_values):
        """Calculate the child metric value.

        Parameters
        ----------
        sso_obs: np.ndarray
            The input data to the metric (same as the parent metric).
        orb: np.ndarray
            The information about the orbit for which the metric is being calculated.
        hval : float
            The H value for which the metric is being calculated.
        metric_values : dict or np.ndarray
            The return value from the parent metric.

        Returns
        -------
        float
        """
        raise NotImplementedError


class NObsMetric(BaseMoMetric):
    """
    Count the total number of observations where an SSobject was 'visible'.
    """

    def __init__(self, snr_limit=None, **kwargs):
        """
        @ snr_limit .. if snr_limit is None, this uses the _calcVis method/completeness
                      if snr_limit is not None, this uses that value as a cutoff instead.
        """
        super().__init__(**kwargs)
        self.snr_limit = snr_limit

    def run(self, sso_obs, orb, hval):
        if self.snr_limit is not None:
            vis = np.where(sso_obs[self.snr_col] >= self.snr_limit)[0]
            return vis.size
        else:
            vis = np.where(sso_obs[self.vis_col] > 0)[0]
            return vis.size


class NObsNoSinglesMetric(BaseMoMetric):
    """
    Count the number of observations for an SSobject, without singles.
    Don't include any observations where it was a single observation on a night.
    """

    def __init__(self, snr_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.snr_limit = snr_limit

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return 0
        nights = sso_obs[self.night_col][vis]
        nights = nights.astype("int")
        ncounts = np.bincount(nights)
        nobs = ncounts[np.where(ncounts > 1)].sum()
        return nobs


class NNightsMetric(BaseMoMetric):
    """Count the number of distinct nights an SSobject is observed."""

    def __init__(self, snr_limit=None, **kwargs):
        """
        @ snr_limit : if SNRlimit is None, this uses _calcVis method/completeness
                     else if snr_limit is not None, it uses that value as a cutoff.
        """
        super().__init__(**kwargs)
        self.snr_limit = snr_limit

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return 0
        nights = len(np.unique(sso_obs[self.night_col][vis]))
        return nights


class ObsArcMetric(BaseMoMetric):
    """Calculate the difference between the first and last observation of an SSobject."""

    def __init__(self, snr_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.snr_limit = snr_limit

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return 0
        arc = sso_obs[self.mjd_col][vis].max() - sso_obs[self.mjd_col][vis].min()
        return arc


class DiscoveryMetric(BaseMoMetric):
    """Identify the discovery opportunities for an SSobject.

    Parameters
    ----------
    n_obs_per_night : int, optional
        Number of observations required within a single night. Default 2.
    t_min : float, optional
        Minimum time span between observations in a single night, in days.
        Default 5 minutes (5/60/24).
    t_max : float, optional
        Maximum time span between observations in a single night, in days.
        Default 90 minutes.
    n_nights_per_window : int, optional
        Number of nights required with observations, within the track window. Default 3.
    t_window : int, optional
        Number of nights included in the track window. Default 15.
    snr_limit : None or float, optional
        SNR limit to use for observations. If snr_limit is None, (default), then it uses
        the completeness calculation added to the 'vis' column (probabilistic visibility,
        based on 5-sigma limit). If snr_limit is not None, it uses this SNR value as a cutoff.
    metricName : str, optional
        The metric name to use.
        Default will be to construct Discovery_nObsPerNightxnNightsPerWindowintWindow.
    """

    def __init__(
        self,
        n_obs_per_night=2,
        t_min=5.0 / 60.0 / 24.0,
        t_max=90.0 / 60.0 / 24.0,
        n_nights_per_window=3,
        t_window=15,
        snr_limit=None,
        badval=None,
        **kwargs
    ):
        # Define anything needed by the child metrics first.
        self.snr_limit = snr_limit
        self.child_metrics = {
            "N_Chances": DiscoveryNChancesmetric(self),
            "N_Obs": DiscoveryNObsmetric(self),
            "Time": DiscoveryTimemetric(self),
            "Distance": DiscoveryDistancemetric(self),
            "RADec": DiscoveryRadecmetric(self),
            "EcLonLat": DiscoveryEclonlatmetric(self),
        }
        if "metric_name" in kwargs:
            metric_name = kwargs.get("metric_name")
            del kwargs["metric_name"]
        else:
            metric_name = "Discovery_%.0fx%.0fin%.0f" % (
                n_obs_per_night,
                n_nights_per_window,
                t_window,
            )
        # Set up for inheriting from __init__.
        super().__init__(
            metric_name=metric_name,
            childMetrics=self.child_metrics,
            badval=badval,
            **kwargs
        )
        # Define anything needed for this metric.
        self.n_obs_per_night = n_obs_per_night
        self.t_min = t_min
        self.t_max = t_max
        self.n_nights_per_window = n_nights_per_window
        self.t_window = t_window

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        # Identify discovery opportunities.
        #  Identify visits where the 'night' changes.
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        nights = sso_obs[self.night_col][vis][vis_sort]
        # print 'all nights', nights
        n = np.unique(nights)
        # Identify all the indexes where the night changes in value.
        n_idx = np.searchsorted(nights, n)
        # print 'nightchanges', nights[n_idx]
        # Count the number of observations per night (except last night)
        obs_per_night = (n_idx - np.roll(n_idx, 1))[1:]
        # Add the number of observations on the last night.
        obs_last_night = np.array([len(nights) - n_idx[-1]])
        obs_per_night = np.concatenate((obs_per_night, obs_last_night))
        # Find the nights with more than nObsPerNight.
        n_with_x_obs = n[np.where(obs_per_night >= self.n_obs_per_night)]
        n_idx_many = np.searchsorted(nights, n_with_x_obs)
        n_idx_many_end = np.searchsorted(nights, n_with_x_obs, side="right") - 1
        # Check that nObsPerNight observations are within tMin/tMax
        times_start = sso_obs[self.mjd_col][vis][vis_sort][n_idx_many]
        times_end = sso_obs[self.mjd_col][vis][vis_sort][n_idx_many_end]
        # Identify the nights with 'clearly good' observations.
        good = np.where(
            (times_end - times_start >= self.t_min) & (times_end - times_start <= self.t_max),
            1,
            0,
        )
        # Identify the nights where we need more investigation
        # (a subset of the visits may be within the interval).
        check = np.where(
            (good == 0)
            & (n_idx_many_end + 1 - n_idx_many > self.n_obs_per_night)
            & (times_end - times_start > self.t_max)
        )[0]
        for i, j, c in zip(
            vis_sort[n_idx_many][check], vis_sort[n_idx_many_end][check], check
        ):
            t = sso_obs[self.mjd_col][vis][vis_sort][i : j + 1]
            dtimes = (np.roll(t, 1 - self.n_obs_per_night) - t)[:-1]
            tidx = np.where((dtimes >= self.t_min) & (dtimes <= self.t_max))[0]
            if len(tidx) > 0:
                good[c] = 1
        # 'good' provides mask for observations which could count as 'good to make tracklets'
        # against sso_obs[vis_sort][n_idx_many].  Now identify tracklets which can make tracks.
        good_idx = vis_sort[n_idx_many][good == 1]
        good_idx_ends = vis_sort[n_idx_many_end][good == 1]
        # print 'good tracklets', nights[good_idx]
        if len(good_idx) < self.n_nights_per_window:
            return self.badval
        delta_nights = (
            np.roll(sso_obs[self.night_col][vis][good_idx], 1 - self.n_nights_per_window)
            - sso_obs[self.night_col][vis][good_idx]
        )
        # Identify the index in sso_obs[vis][good_idx] (sorted by mjd) where the discovery opportunity starts.
        start_idxs = np.where((delta_nights >= 0) & (delta_nights <= self.t_window))[0]
        # Identify the index where the discovery opportunity ends.
        end_idxs = np.zeros(len(start_idxs), dtype="int")
        for i, sIdx in enumerate(start_idxs):
            in_window = np.where(
                sso_obs[self.night_col][vis][good_idx]
                - sso_obs[self.night_col][vis][good_idx][sIdx]
                <= self.t_window
            )[0]
            end_idxs[i] = np.array([in_window.max()])
        # Convert back to index based on sso_obs[vis] (sorted by expMJD).
        start_idxs = good_idx[start_idxs]
        end_idxs = good_idx_ends[end_idxs]
        # print 'start', start_idxs,  nights[start_idxs]#, orb['objId'], hval
        # print 'end', end_idxs, nights[end_idxs]#, orb['objId'], hval
        return {
            "start": start_idxs,
            "end": end_idxs,
            "trackletNights": sso_obs[self.night_col][vis][good_idx],
        }


class DiscoveryNChancesmetric(BaseChildMetric):
    """Calculate total number of discovery opportunities for an SSobject.

    Calculates total number of discovery opportunities between night_start / night_end.
    Child metric to be used with the Discovery Metric.
    """

    def __init__(
        self, parent_discovery_metric, night_start=None, night_end=None, badval=0, **kwargs
    ):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        self.night_start = night_start
        self.night_end = night_end
        self.snr_limit = parent_discovery_metric.snr_limit
        # Update the metric name to use the night_start/night_end values, if an overriding name is not given.
        if "metricName" not in kwargs:
            if night_start is not None:
                self.name = self.name + "_n%d" % (night_start)
            if night_end is not None:
                self.name = self.name + "_n%d" % (night_end)

    def run(self, sso_obs, orb, hval, metric_values):
        """Return the number of different discovery chances we had for each object/H combination."""
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        if self.night_start is None and self.night_end is None:
            return len(metric_values["start"])
        # Otherwise, we have to sort out what night the discovery chances happened on.
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        nights = sso_obs[self.night_col][vis][vis_sort]
        start_nights = nights[metric_values["start"]]
        end_nights = nights[metric_values["end"]]
        if self.night_end is None and self.night_start is not None:
            valid = np.where(start_nights >= self.night_start)[0]
        elif self.night_start is None and self.night_end is not None:
            valid = np.where(end_nights <= self.night_end)[0]
        else:
            # And we only end up here if both were not None.
            valid = np.where(
                (start_nights >= self.night_start) & (end_nights <= self.night_end)
            )[0]
        return len(valid)


class DiscoveryNObsmetric(BaseChildMetric):
    """Calculates the number of observations in the i-th discovery track of an SSobject."""

    def __init__(self, parent_discovery_metric, i=0, badval=0, **kwargs):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        # The number of the discovery chance to use.
        self.i = i

    def run(self, sso_obs, orb, hval, metric_values):
        if self.i >= len(metric_values["start"]):
            return 0
        start_idx = metric_values["start"][self.i]
        end_idx = metric_values["end"][self.i]
        nobs = end_idx - start_idx
        return nobs


class DiscoveryTimemetric(BaseChildMetric):
    """Returns the time of the i-th discovery track of an SSobject."""

    def __init__(self, parent_discovery_metric, i=0, t_start=None, badval=-999, **kwargs):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        self.i = i
        self.t_start = t_start
        self.snr_limit = parent_discovery_metric.snr_limit

    def run(self, sso_obs, orb, hval, metric_values):
        if self.i >= len(metric_values["start"]):
            return self.badval
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        times = sso_obs[self.mjd_col][vis][vis_sort]
        start_idx = metric_values["start"][self.i]
        t_disc = times[start_idx]
        if self.t_start is not None:
            t_disc = t_disc - self.t_start
        return t_disc


class DiscoveryDistancemetric(BaseChildMetric):
    """Returns the distance of the i-th discovery track of an SSobject."""

    def __init__(
        self, parent_discovery_metric, i=0, distance_col="geo_dist", badval=-999, **kwargs
    ):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        self.i = i
        self.distance_col = distance_col
        self.snr_limit = parent_discovery_metric.snr_limit

    def run(self, sso_obs, orb, hval, metric_values):
        if self.i >= len(metric_values["start"]):
            return self.badval
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        dists = sso_obs[self.distance_col][vis][vis_sort]
        start_idx = metric_values["start"][self.i]
        dist_disc = dists[start_idx]
        return dist_disc


class DiscoveryRadecmetric(BaseChildMetric):
    """Returns the RA/Dec of the i-th discovery track of an SSobject."""

    def __init__(self, parent_discovery_metric, i=0, badval=None, **kwargs):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        self.i = i
        self.snr_limit = parent_discovery_metric.snr_limit
        self.metric_dtype = "object"

    def run(self, sso_obs, orb, hval, metric_values):
        if self.i >= len(metric_values["start"]):
            return self.badval
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        ra = sso_obs[self.ra_col][vis][vis_sort]
        dec = sso_obs[self.dec_col][vis][vis_sort]
        start_idx = metric_values["start"][self.i]
        return (ra[start_idx], dec[start_idx])


class DiscoveryEclonlatmetric(BaseChildMetric):
    """Returns the ecliptic lon/lat and solar elong of the i-th discovery track of an SSobject."""

    def __init__(self, parent_discovery_metric, i=0, badval=None, **kwargs):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        self.i = i
        self.snr_limit = parent_discovery_metric.snr_limit
        self.metric_dtype = "object"

    def run(self, sso_obs, orb, hval, metric_values):
        if self.i >= len(metric_values["start"]):
            return self.badval
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        ec_lon = sso_obs["ec_lon"][vis][vis_sort]
        ec_lat = sso_obs["ec_lat"][vis][vis_sort]
        solar_elong = sso_obs["solar_elong"][vis][vis_sort]
        start_idx = metric_values["start"][self.i]
        return (ec_lon[start_idx], ec_lat[start_idx], solar_elong[start_idx])


class DiscoveryVelocitymetric(BaseChildMetric):
    """Returns the sky velocity of the i-th discovery track of an SSobject."""

    def __init__(self, parent_discovery_metric, i=0, badval=-999, **kwargs):
        super().__init__(parent_discovery_metric, badval=badval, **kwargs)
        self.i = i
        self.snr_limit = parent_discovery_metric.snrLimit

    def run(self, sso_obs, orb, hval, metric_values):
        if self.i >= len(metric_values["start"]):
            return self.badval
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        vis_sort = np.argsort(sso_obs[self.mjd_col][vis])
        velocity = sso_obs["velocity"][vis][vis_sort]
        start_idx = metric_values["start"][self.i]
        return velocity[start_idx]


class ActivityOverTimeMetric(BaseMoMetric):
    """Count fraction of survey we could identify activity for an SSobject.

    Counts the time periods where we would have a chance to detect activity on
    a moving object.
    Splits observations into time periods set by 'window', then looks for observations within each window,
    and reports what fraction of the total windows receive 'nObs' visits.
    """

    def __init__(self, window, snr_limit=5, survey_years=10.0, metric_name=None, **kwargs):
        if metric_name is None:
            metric_name = "Chance of detecting activity lasting %.0f days" % (window)
        super().__init__(metric_name=metric_name, **kwargs)
        self.snr_limit = snr_limit
        self.window = window
        self.survey_years = survey_years
        self.window_bins = np.arange(
            0, self.survey_years * 365 + self.window / 2.0, self.window
        )
        self.n_windows = len(self.window_bins)
        self.units = "%.1f Day Windows" % (self.window)

    def run(self, sso_obs, orb, hval):
        # For cometary activity, expect activity at the same point in its orbit at the same time, mostly
        # For collisions, expect activity at random times
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        n, b = np.histogram(sso_obs[vis][self.night_col], bins=self.window_bins)
        activity_windows = np.where(n > 0)[0].size
        return activity_windows / float(self.n_windows)


class ActivityOverPeriodMetric(BaseMoMetric):
    """Count fraction of object period we could identify activity for an SSobject.

    Count the fraction of the orbit (when split into n_bins) that receive
    observations, in order to have a chance to detect activity.
    """

    def __init__(
        self,
        binsize,
        snr_limit=5,
        q_col="q",
        e_col="e",
        a_col="a",
        t_peri_col="tPeri",
        anomaly_col="meanAnomaly",
        metric_name=None,
        **kwargs
    ):
        """
        @ binsize : size of orbit slice, in degrees.
        """
        if metric_name is None:
            metric_name = "Chance of detecting activity covering %.1f of the orbit" % (
                binsize
            )
        super().__init__(metric_name=metric_name, **kwargs)
        self.q_col = q_col
        self.e_col = e_col
        self.a_col = a_col
        self.t_peri_col = t_peri_col
        self.anomaly_col = anomaly_col
        self.snr_limit = snr_limit
        self.binsize = np.radians(binsize)
        self.anomaly_bins = np.arange(0, 2 * np.pi, self.binsize)
        self.anomaly_bins = np.concatenate([self.anomaly_bins, np.array([2 * np.pi])])
        self.n_bins = len(self.anomaly_bins) - 1
        self.units = "%.1f deg" % (np.degrees(self.binsize))

    def run(self, sso_obs, orb, hval):
        # For cometary activity, expect activity at the same point in its orbit at the same time, mostly
        # For collisions, expect activity at random times
        if self.a_col in orb.keys():
            a = orb[self.a_col]
        elif self.q_col in orb.keys():
            a = orb[self.q_col] / (1 - orb[self.e_col])
        else:
            return self.badval

        period = np.power(a, 3.0 / 2.0) * 365.25  # days

        if self.anomaly_col in orb.keys():
            curranomaly = np.radians(
                orb[self.anomaly_col]
                + (sso_obs[self.mjd_col] - orb["epoch"]) / period * 360.0
            ) % (2 * np.pi)
        elif self.t_peri_col in orb.keys():
            curranomaly = ((sso_obs[self.mjd_col] - orb[self.t_peri_col]) / period) % (
                2 * np.pi
            )
        else:
            return self.badval

        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        n, b = np.histogram(curranomaly[vis], bins=self.anomaly_bins)
        activity_windows = np.where(n > 0)[0].size
        return activity_windows / float(self.n_bins)


class MagicDiscoveryMetric(BaseMoMetric):
    """Count the number of nights with discovery opportunities with very good software for an SSobject."""

    def __init__(self, n_obs=6, t_window=60, snr_limit=None, **kwargs):
        """
        @ n_obs = the total number of observations required for 'discovery'
        @ t_window = the timespan of the discovery window.
        @ snr_limit .. if snr_limit is None then uses 'completeness' calculation,
                   .. if snr_limit is not None, then uses this value as a cutoff.
        """
        super().__init__(**kwargs)
        self.snr_limit = snr_limit
        self.n_obs = n_obs
        self.t_window = t_window
        self.badval = 0

    def run(self, sso_obs, orb, hval):
        """SsoObs = Dataframe, orb=Dataframe, hval=single number."""
        # Calculate visibility for this orbit at this H.
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) < self.n_obs:
            return self.badval
        t_nights = np.sort(sso_obs[self.night_col])[vis]
        u_nights = np.unique(t_nights)
        indx_w = np.searchsorted(t_nights, u_nights, "left")
        indx_p = np.searchsorted(t_nights - self.t_window, u_nights, "left")
        n_disc = np.where(indx_p - indx_w >= self.n_obs)[0].size
        return n_disc


class HighVelocityMetric(BaseMoMetric):
    """Count number of times an SSobject appears trailed.

    Count the number of times an asteroid is observed with a velocity high enough to make it appear
    trailed by a factor of (psf_factor)*PSF - i.e. velocity >= psf_factor * seeing / visitExpTime.
    Simply counts the total number of observations with high velocity.
    """

    def __init__(self, psf_factor=2.0, snr_limit=None, velocity_col="velocity", **kwargs):
        """
        @ psf_factor = factor to multiply seeing/visitExpTime by
        (velocity(deg/day) >= 24*psf_factor*seeing(")/visitExptime(s))
        """
        super().__init__(**kwargs)
        self.velocity_col = velocity_col
        self.snr_limit = snr_limit
        self.psf_factor = psf_factor
        self.badval = 0

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        high_velocity_obs = np.where(
            sso_obs[self.velocity_col][vis]
            >= (
                24.0
                * self.psf_factor
                * sso_obs[self.seeing_col][vis]
                / sso_obs[self.exp_time_col][vis]
            )
        )[0]
        return high_velocity_obs.size


class HighVelocityNightsMetric(BaseMoMetric):
    """Count the number of discovery opportunities (via trailing) for an SSobject.

    Determine the first time an asteroid is observed is observed with a velocity high enough to make
    it appear trailed by a factor of psf_factor*PSF with n_obs_per_night observations within a given night.

    Parameters
    ----------
    psf_factor: float, optional
        Object velocity (deg/day) must be >= 24 * psf_factor * seeingGeom (") / visitExpTime (s).
        Default is 2 (i.e. object trailed over 2 psf's).
    n_obs_per_night: int, optional
        Number of observations per night required. Default 2.
    snr_limit: float or None
        If snr_limit is set as a float, then requires object to be above snr_limit SNR in the image.
        If snr_limit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    velocity_col: str, optional
        Name of the velocity column in the obs file. Default 'velocity'. (note this is deg/day).

    Returns
    -------
    float
        The time of the first detection where the conditions are satisifed.
    """

    def __init__(
        self,
        psf_factor=2.0,
        n_obs_per_night=2,
        snr_limit=None,
        velocity_col="velocity",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.velocity_col = velocity_col
        self.snr_limit = snr_limit
        self.psf_factor = psf_factor
        self.n_obs_per_night = n_obs_per_night

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return self.badval
        high_velocity_obs = np.where(
            sso_obs[self.velocity_col][vis]
            >= (
                24.0
                * self.psf_factor
                * sso_obs[self.seeing_col][vis]
                / sso_obs[self.exp_time_col][vis]
            )
        )[0]
        if len(high_velocity_obs) == 0:
            return 0
        nights = sso_obs[self.night_col][vis][high_velocity_obs]
        n = np.unique(nights)
        n_idx = np.searchsorted(nights, n)
        # Count the number of observations per night (except last night)
        obs_per_night = (n_idx - np.roll(n_idx, 1))[1:]
        # Add the number of observations on the last night.
        obs_last_night = np.array([len(nights) - n_idx[-1]])
        obs_per_night = np.concatenate((obs_per_night, obs_last_night))
        # Find the nights with at least nObsPerNight visits
        # (this is already looking at only high velocity observations).
        n_with_x_obs = n[np.where(obs_per_night >= self.n_obs_per_night)]
        if len(n_with_x_obs) > 0:
            found = sso_obs[np.where(sso_obs[self.night_col] == n_with_x_obs[0])][
                self.mjd_col
            ][0]
        else:
            found = self.badval
        return found


class LightcurveinversionAsteroidmetric(BaseMoMetric):
    """
    This metric is generally applicable to NEOs and MBAs - inner solar system objects.

    Determine if the cumulative sum of observations of a target are enough to enable lightcurve
    inversion for shape modeling. For this to be true, multiple conditions need to be
    satisfied:

    1) The SNR-weighted number of observations (each observation is weighted by its SNR, up to a max of 100)
    must be larger than the threshhold weight_det (default 50)
    2) Ecliptic longitudinal coverage needs to be at least 90 degrees, and the absolute deviation
    needs to be at least 1/8th the longitudinal coverage.
    3) The phase angle coverage needs to span at least 5 degrees.

    For evaluation of condition 2, the median ecliptic longitude is subtracted from all longitudes,
    and the modulo 360 of those values is taken. This ensures that the wrap around 360 is handled
    correctly.

    For more information on the above conditions, please see
    https://docs.google.com/document/d/1GAriM7trpTS08uanjUF7PyKALB2JBTjVT7Y6R30i0-8/edit?usp=sharing
    Contributed by Steve Chesley, Wes Fraser, Josef Durech, and the inner solar system working group.

    Parameters
    ----------
    weight_det: float, optional
        The SNR-weighted number of detections required (per bandpass in any ONE of the filters in filterlist).
        Default 50.
    snr_limit: float or None, optional
        If snr_limit is set as a float, then requires object to be above snr_limit SNR in the image.
        If snr_limit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    snr_max: float, optional
        Maximum value toward the SNR-weighting to consider. Default 100.
    filterlist: list of str, optional
        The filters which the lightcurve inversion could be based on. Requirements must be met in one of
        these filters.

    Returns
    -------
    int
        0 (could not perform lightcurve inversion) or 1 (could)
    """

    def __init__(
        self,
        weight_det=50,
        snr_limit=None,
        snr_max=100,
        filterlist=("u", "g", "r", "i", "z", "y"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.snr_limit = snr_limit
        self.snr_max = snr_max
        self.weight_det = weight_det
        self.filterlist = filterlist

    def run(self, sso_obs, orb, hval):
        # Calculate the clipped SNR - ranges from snrLimit / SNR+vis to snrMax.
        clip_snr = np.minimum(sso_obs[self.snr_col], self.snr_max)
        if self.snr_limit is not None:
            clip_snr = np.where(sso_obs[self.snr_col] <= self.snr_limit, 0, clip_snr)
        else:
            clip_snr = np.where(sso_obs[self.vis_col] == 0, 0, clip_snr)
        if len(np.where(clip_snr > 0)[0]) == 0:
            return 0
        # Check each filter in filterlist:
        # stop as soon as find a filter that matches requirements.
        inversion_possible = 0
        for f in self.filterlist:
            # Is the SNR-weight sum of observations in this filter high enough?
            match = np.where(sso_obs[self.filter_col] == f)
            snr_sum = np.sum(clip_snr[match]) / self.snr_max
            if snr_sum < self.weight_det:
                # Do not have enough SNR-weighted observations, so skip on to the next filter.
                continue
            # Is the ecliptic longitude coverage for the visible observations sufficient?
            # Is the phase coverage sufficient?
            vis = np.where(clip_snr[match] > 0)
            ec_l = sso_obs["ecLon"][match][vis]
            phase_angle = sso_obs["phase"][match][vis]
            # Calculate the absolute deviation and range of ecliptic longitude.
            ec_l_centred = (ec_l - np.median(ec_l)) % 360.0
            a_dev = np.sum(np.abs(ec_l_centred - np.mean(ec_l_centred))) / len(ec_l_centred)
            d_l = np.max(ec_l) - np.min(ec_l)
            # Calculate the range of the phase angle
            dp = np.max(phase_angle) - np.min(phase_angle)
            # Metric requirement is that d_l >= 90 deg, absolute deviation is greater than d_l/8
            # and then that the phase coverage is more than 5 degrees.
            # Stop as soon as find a case where this is true.
            if d_l >= 90.0 and a_dev >= d_l / 8 and dp >= 5:
                inversion_possible += 1
                break
        return inversion_possible


class ColorAsteroidmetric(BaseMoMetric):
    """
    This metric is appropriate for MBAs and NEOs, and other inner solar system objects.

    The metric evaluates if the SNR-weighted number of observations are enough to
    determine an approximate lightcurve and phase function -- and from this,
    then a color for the asteroid can be determined.
    The assumption is that you must fit the lightcurve/phase function in each bandpass,
    and could do this well-enough if you have at least weight_det SNR-weighted observations
    in the bandpass.
    e.g. to find a g-r color, you must have 10 (SNR-weighted) obs in g and 10 in r.

    For more details, see
    https://docs.google.com/document/d/1GAriM7trpTS08uanjUF7PyKALB2JBTjVT7Y6R30i0-8/edit?usp=sharing
    Contributed by Wes Fraser, Steven Chesley & the inner solar system working group.

    Parameters
    ----------
    weight_det: float, optional
        The SNR-weighted number of detections required (per bandpass in any ONE of the filters in filterlist).
        Default 10.
    snr_limit: float or None, optional
        If snr_limit is set as a float, then requires object to be above snr_limit SNR in the image.
        If snr_limit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    snr_max: float, optional
        Maximum value toward the SNR-weighting to consider. Default 20.

    Returns
    -------
    int
        An integer 'flag' that indicates whether the mean magnitude (and thus a color) was determined in:
        0 = no bands
        1 = g and (r or i) and (z or y). i.e. obtain colors g-r or g-i PLUS g-z or g-y
        2 = Any 4 different filters (from grizy). i.e. colors = g-r, r-i, i-z, OR r-i, i-z, z-y..
        3 = All 5 from grizy. i.e. colors g-r, r-i, i-z, z-y.
        4 = All 6 filters (ugrizy) -- best possible! add u-g.
    """

    def __init__(self, weight_det=10, snr_max=20, snr_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_det = weight_det
        self.snr_limit = snr_limit
        self.snr_max = snr_max
        self.filterlist = ("u", "g", "r", "i", "z", "y")

    def run(self, sso_obs, orb, hval):
        clip_snr = np.minimum(sso_obs[self.snr_col], self.snr_max)
        if self.snr_limit is not None:
            clip_snr = np.where(sso_obs[self.snr_col] <= self.snr_limit, 0, clip_snr)
        else:
            clip_snr = np.where(sso_obs[self.vis_col] == 0, 0, clip_snr)
        if len(np.where(clip_snr > 0)[0]) == 0:
            return self.badval

        # Evaluate SNR-weighted number of observations in each filter.
        filter_weight = {}
        for f in self.filterlist:
            match = np.where(sso_obs[self.filter_col] == f)
            snrweight = np.sum(clip_snr[match]) / self.snr_max
            # If the snrweight exceeds the weightDet, add it to the dictionary.
            if snrweight > self.weight_det:
                filter_weight[f] = snrweight

        # Now assign a flag:
        # 0 = no bands
        # 1 = g and (r or i) and (z or y). i.e. obtain colors g-r or g-i PLUS g-z or g-y
        # 2 = Any 4 different filters (from grizy). i.e. colors = g-r, r-i, i-z, OR r-i, i-z, z-y..
        # 3 = All 5 from grizy. i.e. colors g-r, r-i, i-z, z-y.
        # 4 = All 6 filters (ugrizy) -- best possible! add u-g.
        all_six = set(self.filterlist)
        good_five = set(["g", "r", "i", "z", "y"])

        if len(filter_weight) == 0:  # this lets us stop evaluating here if possible.
            flag = 0
        elif all_six.intersection(filter_weight) == all_six:
            flag = 4
        elif good_five.intersection(filter_weight) == good_five:
            flag = 3
        elif len(good_five.intersection(filter_weight)) == 4:
            flag = 2
        elif "g" in filter_weight:
            # Have 'g' - do we have (r or i) and (z or y)
            if ("r" in filter_weight or "i" in filter_weight) and (
                "z" in filter_weight or "y" in filter_weight
            ):
                flag = 1
            else:
                flag = 0
        else:
            flag = 0

        return flag


class LightcurvecolorOutermetric(BaseMoMetric):
    """
    This metric is appropriate for outer solar system objects, such as TNOs and SDOs.

    This metric evaluates whether the number of observations is sufficient to fit a lightcurve
    in a primary and secondary bandpass. The primary bandpass requires more observations than
    the secondary. Essentially, it's a complete lightcurve in one or both bandpasses, with at
    least a semi-complete lightcurve in the secondary band.

    The lightcurve/color can be calculated with any two of the bandpasses in filterlist.
    Contributed by Wes Fraser.

    Parameters
    ----------
    snr_limit: float or None, optional
        If snr_limit is set as a float, then requires object to be above snr_limit SNR in the image.
        If snr_limit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    num_req: int, optional
        Number of observations required for a lightcurve fitting. Default 30.
    num_sec_filt: int, optional
        Number of observations required in a secondary band for color only. Default 20.
    filterlist: list of str, optional
        Filters that the primary/secondary measurements can be in.

    Returns
    -------
    int
        A flag that indicates whether a color/lightcurve was generated in:
        0 = no lightcurve (although may have had 'color' in one or more band)
        1 = a lightcurve in a single filter (but no additional color information)
        2+ = lightcurves in more than one filter  (or lightcurve + color)
        e.g. lightcurve in 2 bands, with additional color information in another = 3.
    """

    def __init__(
        self,
        snr_limit=None,
        num_req=30,
        num_sec_filt=20,
        filterlist=("u", "g", "r", "i", "z", "y"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.snr_limit = snr_limit
        self.num_req = num_req
        self.num_sec_filt = num_sec_filt
        self.filterlist = filterlist

    def run(self, sso_obs, orb, hval):
        vis = _set_vis(sso_obs, self.snr_limit, self.snr_col, self.vis_col)
        if len(vis) == 0:
            return 0

        lightcurves = set()
        colors = set()
        for f in self.filterlist:
            nmatch = np.where(sso_obs[vis][self.filter_col] == f)[0]
            if len(nmatch) >= self.num_req:
                lightcurves.add(f)
            if len(nmatch) >= self.num_sec_filt:
                colors.add(f)

        # Set the flags - first the number of filters with lightcurves.
        flag = len(lightcurves)
        # And check if there were extra filters which had enough for a color
        # but not enough for a full lightcurve.
        if len(colors.difference(lightcurves)) > 0:
            # If there was no lightcurve available to match against:
            if len(lightcurves) == 0:
                flag = 0
            else:
                # We had a lightcurve and now can add a color.
                flag += 1
        return flag


class InstantaneousColorMetric(BaseMoMetric):
    """Identify SSobjects which could have observations suitable to determine colors.

    Generally, this is not the mode LSST would work in - the lightcurves of the objects
    mean that the time interval would have to be quite short.

    This is roughly defined as objects which have more than n_pairs pairs of observations
    with SNR greater than snr_limit, in bands bandOne and bandTwo, within n_hours.

    Parameters
    ----------
    n_pairs: int, optional
        The number of pairs of observations (in each band) that must be within n_hours
        Default 1
    snr_limit: float, optional
        The SNR limit for the observations. Default 10.
    n_hours: float, optional
        The time interval between observations in the two bandpasses (hours). Default 0.5 hours.
    b_one: str, optional
        The first bandpass for the color. Default 'g'.
    b_two: str, optional
        The second bandpass for the color. Default 'r'.

    Returns
    -------
    int
        0 (no color possible under these constraints) or 1 (color possible).
    """

    def __init__(self, n_pairs=1, snr_limit=10, n_hours=0.5, b_one="g", b_two="r", **kwargs):
        super().__init__(**kwargs)
        self.n_pairs = n_pairs
        self.snr_limit = snr_limit
        self.n_hours = n_hours
        self.b_one = b_one
        self.b_two = b_two
        self.badval = -666

    def run(self, sso_obs, orb, hval):
        vis = np.where(sso_obs[self.snr_col] >= self.snr_limit)[0]
        if len(vis) < self.n_pairs * 2:
            return 0
        b_one_obs = np.where(sso_obs[self.filter_col][vis] == self.b_one)[0]
        b_two_obs = np.where(sso_obs[self.filter_col][vis] == self.b_two)[0]
        timesb_one = sso_obs[self.mjd_col][vis][b_one_obs]
        timesb_two = sso_obs[self.mjd_col][vis][b_two_obs]
        if len(timesb_one) == 0 or len(timesb_two) == 0:
            return 0
        d_time = self.n_hours / 24.0
        # Calculate the time between the closest pairs of observations.
        in_order = np.searchsorted(timesb_one, timesb_two, "right")
        in_order = np.where(in_order - 1 > 0, in_order - 1, 0)
        dt_pairs = timesb_two - timesb_one[in_order]
        if len(np.where(dt_pairs < d_time)[0]) >= self.n_pairs:
            found = 1
        else:
            found = 0
        return found


class PeakVMagMetric(BaseMoMetric):
    """Pull out the peak V magnitude of all observations of the SSobject."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, sso_obs, orb, hval):
        peak_vmag = np.min(sso_obs[self.app_mag_v_col])
        return peak_vmag


class KnownObjectsMetric(BaseMoMetric):
    """Identify SSobjects which could be classified as 'previously known' based on their peak V magnitude.
    This is most appropriate for NEO surveys, where most of the sky has been covered so the exact location
    (beyond being in the visible sky) is not as important.

    Default parameters tuned to match NEO survey capabilities.
    Returns the time at which each first reached that threshold V magnitude.
    The default values are calibrated using the NEOs larger than 140m discovered in the last 20 years
    and assuming a 30% completeness in 2017.

    Parameters
    -----------
    elong_thresh : float, optional
        The cutoff in solar elongation to consider an object 'visible'. Default 100 deg.
    v_mag_thresh1 : float, optional
        The magnitude threshold for previously known objects. Default 20.0.
    eff1 : float, optional
        The likelihood of actually achieving each individual input observation.
        If the input observations include one observation per day, an 'eff' value of 0.3 would
        mean that (on average) only one third of these observations would be achieved.
        This is similar to the level for LSST, which can cover the visible sky every 3-4 days.
        Default 0.1
    t_switch1 : float, optional
        The (MJD) time to switch between v_mag_thresh1 + eff1 to v_mag_thresh2 + eff2, e.g.
        the end of the first period.
        Default 53371 (2005).
    v_mag_thresh2 : float, optional
        The magnitude threshhold for previously known objects. Default 22.0.
        This is based on assuming PS and other surveys will be efficient down to V=22.
    eff2 : float, optional
        The efficiency of observations during the second period of time. Default 0.1
    t_switch2 : float, optional
        The (MJD) time to switch between v_mag_thresh2 + eff2 to v_mag_thresh3 + eff3.
        Default 57023 (2015).
    v_mag_thresh3 : float, optional
        The magnitude threshold during the third period. Default 22.0, based on PS1 + Catalina.
    eff3 : float, optional
        The efficiency of observations during the third period. Default 0.1
    t_switch3 : float, optional
        The (MJD) time to switch between v_mag_thresh3 + eff3 to v_mag_thresh4 + eff4.
        Default 59580 (2022).
    v_mag_thresh4 : float, optional
        The magnitude threshhold during the fourth (last) period. Default 22.0, based on PS1 + Catalina.
    eff4 : float, optional
        The efficiency of observations during the fourth (last) period. Default 0.2
    """

    def __init__(
        self,
        elong_thresh=100.0,
        v_mag_thresh1=20.0,
        eff1=0.1,
        t_switch1=53371,  # XXX--maybe swap to survey_start_mjd and then delta_t's
        v_mag_thresh2=21.5,
        eff2=0.1,
        t_switch2=57023,
        v_mag_thresh3=22.0,
        eff3=0.1,
        t_switch3=59580,
        v_mag_thresh4=22.0,
        eff4=0.2,
        elong_col="Elongation",
        mjd_col="MJD(UTC)",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.elong_thresh = elong_thresh
        self.elong_col = elong_col
        self.v_mag_thresh1 = v_mag_thresh1
        self.eff1 = eff1
        self.t_switch1 = t_switch1
        self.v_mag_thresh2 = v_mag_thresh2
        self.eff2 = eff2
        self.t_switch2 = t_switch2
        self.v_mag_thresh3 = v_mag_thresh3
        self.eff3 = eff3
        self.t_switch3 = t_switch3
        self.v_mag_thresh4 = v_mag_thresh4
        self.eff4 = eff4
        self.mjd_col = mjd_col
        self.badval = int(t_switch3) + 365 * 1000

    def _pick_obs(self, potential_obs_times, eff):
        # From a set of potential observations, apply an efficiency
        # And return the minimum time (if any)
        rand_pick = np.random.rand(len(potential_obs_times))
        picked = np.where(rand_pick <= eff)[0]
        if len(picked) > 0:
            disc_time = potential_obs_times[picked].min()
        else:
            disc_time = None
        return disc_time

    def run(self, sso_obs, orb, hval):
        visible = np.where(sso_obs[self.elong_col] >= self.elong_thresh, 1, 0)
        discovery_time = None
        # Look for discovery in any of the three periods.
        # First period.
        obs1 = np.where((sso_obs[self.mjd_col] < self.t_switch1) & visible)[0]
        over_peak = np.where(sso_obs[self.app_mag_v_col][obs1] <= self.v_mag_thresh1)[0]
        if len(over_peak) > 0:
            discovery_time = self._pick_obs(
                sso_obs[self.mjd_col][obs1][over_peak], self.eff1
            )
        # Second period.
        if discovery_time is None:
            obs2 = np.where(
                (sso_obs[self.mjd_col] >= self.t_switch1)
                & (sso_obs[self.mjd_col] < self.t_switch2)
                & visible
            )[0]
            over_peak = np.where(sso_obs[self.app_mag_v_col][obs2] <= self.v_mag_thresh2)[0]
            if len(over_peak) > 0:
                discovery_time = self._pick_obs(
                    sso_obs[self.mjd_col][obs2][over_peak], self.eff2
                )
        # Third period.
        if discovery_time is None:
            obs3 = np.where(
                (sso_obs[self.mjd_col] >= self.t_switch2)
                & (sso_obs[self.mjd_col] < self.t_switch3)
                & visible
            )[0]
            over_peak = np.where(sso_obs[self.app_mag_v_col][obs3] <= self.v_mag_thresh3)[0]
            if len(over_peak) > 0:
                discovery_time = self._pick_obs(
                    sso_obs[self.mjd_col][obs3][over_peak], self.eff3
                )
        # Fourth period.
        if discovery_time is None:
            obs4 = np.where((sso_obs[self.mjd_col] >= self.t_switch3) & visible)[0]
            over_peak = np.where(sso_obs[self.app_mag_v_col][obs4] <= self.v_mag_thresh4)[0]
            if len(over_peak) > 0:
                discovery_time = self._pick_obs(
                    sso_obs[self.mjd_col][obs4][over_peak], self.eff4
                )
        if discovery_time is None:
            discovery_time = self.badval
        return discovery_time
