###################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
###################################################################
__all__ = (
    "calc_interval_decay",
    "GalPlaneVisitIntervalsTimescaleMetric",
    "GalPlaneSeasonGapsTimescaleMetric",
)

from types import MethodType

import numpy as np
from rubin_scheduler.utils import calc_season

from rubin_sim.maf.maps.galactic_plane_priority_maps import gp_priority_map_components_to_keys

from .base_metric import BaseMetric
from .galactic_plane_metrics import galplane_priority_map_thresholds
from .season_metrics import find_season_edges

TAU_OBS = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])


def calc_interval_decay(delta_tobs, tau):
    # Decay constant for metric value relationship as function of obs interval
    K = 1.0 / tau
    m = np.exp(-K * (delta_tobs - tau))
    # But where observation interval is <= tau, replace with 1
    m[np.where(delta_tobs <= tau)] = 1.0
    return m


# this is a bit of a hack .. it helps us use a variety of tau_obs values,
# and dynamically set up reduce functions
def help_set_reduce_func(obj, metricval, tau):
    def _reduce_tau(obj, metricval):
        return metricval[tau]

    return _reduce_tau


class GalPlaneVisitIntervalsTimescaleMetric(BaseMetric):
    """Evaluate the intervals between sequential observations in a
    lightcurve relative to the scientifically desired sampling interval.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column
        headers contained in the priority_GalPlane_footprint_map_data tables.
    tau_obs : `np.ndarray` or `list` of `float`, opt
        Timescales of minimum-required observations intervals for various
        classes of time variability.
        Default (None), uses TAU_OBS. In general, this should be left as the
        default and consistent across all galactic-plane oriented metrics.
    mag_limit : `float`, opt
        Magnitude limit to use as a cutoff for various observations.
        Default 22.0.
    mjd_col : `str`, opt
        The name of the observation start MJD column.
        Default 'observationStartMJD'.
    m5_col : `str', opt
        The name of the five sigma depth column. Default 'fiveSigmaDepth'.
    """

    def __init__(
        self,
        science_map,
        tau_obs=None,
        mag_limit=22.0,
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        **kwargs,
    ):
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        # tau_obs is an array of minimum-required observation intervals for
        # four categories of time variability
        if tau_obs is not None:
            self.tau_obs = tau_obs
        else:
            self.tau_obs = TAU_OBS
        # Create reduce functions for the class that are return the metric
        # for each value in tau_obs

        self.mag_limit = mag_limit
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        maps = ["GalacticPlanePriorityMap"]
        if "metric_name" not in kwargs:
            metric_name = f"GalPlaneVisitIntervalsTimescales_{self.science_map}"
        else:
            metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        for tau in self.tau_obs:
            tau_reduce_name = f"reduce_Tau_{tau:.1f}".replace(".", "_")
            newmethod = help_set_reduce_func(self, None, tau)
            setattr(self, tau_reduce_name, MethodType(newmethod, tau_reduce_name))
        super().__init__(
            col=[self.mjd_col, self.m5_col],
            metric_name=metric_name,
            maps=maps,
            **kwargs,
        )
        for i, tau in enumerate(self.tau_obs):
            self.reduce_order[f"reduceTau_{tau:.1f}".replace(".", "_").replace("reduce", "")] = i

    def run(self, data_slice, slice_point=None):
        # Check if we want to evaluate this part of the sky,
        # or if the weight is below threshold.
        if (
            slice_point[gp_priority_map_components_to_keys("sum", self.science_map)]
            <= self.priority_map_threshold
        ):
            return self.badval
        # Select observations in the time sequence that fulfill the
        # S/N requirements:
        match = np.where(data_slice[self.m5_col] >= self.mag_limit)[0]
        # We need at least two visits which match these requirements
        # to calculate visit gaps
        if len(match) < 2:
            return self.badval
        # Find the time gaps between visits (in any filter)
        times = data_slice[self.mjd_col][match]
        times.sort()
        delta_tobs = np.diff(times)
        # Compare the time gap distribution to the time gap required
        # to characterize variability
        metric_data = {}
        for tau in self.tau_obs:
            # Normalize
            metric_data[tau] = calc_interval_decay(delta_tobs, tau).sum() / len(delta_tobs)
        return metric_data


class GalPlaneSeasonGapsTimescaleMetric(BaseMetric):
    """Evaluate the gap between sequential seasonal gaps in
    observations in a lightcurve relative to the scientifically desired
    sampling interval.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column
        headers contained in the
        priority_GalPlane_footprint_map_data tables.
    tau_var : `np.ndarray` or `list` of `float`, opt
        Timescales of variability for various classes of time variability.
        Default (None), uses TAU_OBS * 5. In general, this should be left
        as the default and consistent
        across all galactic-plane oriented metrics.
    mag_limit : `float`, opt
        Magnitude limit to use as a cutoff for various observations.
        Default 22.0.
    expected_season_gap : `float`, opt
        The typical season gap expected for a galactic plane field in days.
        The default, 145 days, is typical for a bulge field.
    mjd_col : `str`, opt
        The name of the observation start MJD column.
        Default 'observationStartMJD'.
    m5_col : `str', opt
        The name of the five sigma depth column. Default 'fiveSigmaDepth'.
    """

    def __init__(
        self,
        science_map,
        tau_var=None,
        mag_limit=22.0,
        expected_season_gap=145,
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        **kwargs,
    ):
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        # tau_obs is an array of minimum-required observation intervals for
        # four categories of time variability; tau_var is the related timescale
        # for the variability (tau_var is approximately 5*tau_obs, in general)
        if tau_var is not None:
            self.tau_var = tau_var
        else:
            self.tau_var = TAU_OBS * 5
        ### NOTE: I would recommend dropping tau_var 10 and 25 from this
        # analysis unless the metric is changed
        # these intervals are so short they will *always* be dropped
        # during the season gap
        self.mag_limit = mag_limit
        self.expected_season_gap = expected_season_gap
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        if "metric_name" not in kwargs:
            metric_name = f"GalPlaneSeasonGapsTimescales_{self.science_map}"
        else:
            metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        for tau in self.tau_var:
            tau_reduce_name = f"reduce_Tau_{tau:.1f}".replace(".", "_")
            newmethod = help_set_reduce_func(self, None, tau)
            setattr(self, tau_reduce_name, MethodType(newmethod, tau_reduce_name))
        super().__init__(col=[self.mjd_col, self.m5_col], metric_name=metric_name, **kwargs)
        for i, tau in enumerate(self.tau_var):
            self.reduce_order[f"reduce_Tau_{tau:.1f}".replace(".", "_").replace("reduce", "")] = i

    def run(self, data_slice, slice_point):
        # Check if we want to evaluate this part of the sky,
        # or if the weight is below threshold.
        if (
            slice_point[gp_priority_map_components_to_keys("sum", self.science_map)]
            <= self.priority_map_threshold
        ):
            return self.badval
        # Find the length of the gaps between each season
        times = data_slice[self.mjd_col]
        times.sort()
        # data = np.sort(data_slice[self.mjd_col], order=self.mjd_col)
        # SlicePoints ra/dec are always in radians -
        # convert to degrees to calculate season
        seasons = calc_season(np.degrees(slice_point["ra"]), times)
        first_of_season, last_of_season = find_season_edges(seasons)
        # season_lengths = times[last_of_season] -  times[first_of_season]
        # would this match interval calc better?
        season_gaps = times[first_of_season][1:] - times[last_of_season][:-1]
        if len(season_gaps) == 0:
            return self.badval
        metric_data = {}
        for i, tau in enumerate(self.tau_var):
            metric_data[tau] = calc_interval_decay(season_gaps, tau)
            # if the season gap is shorter than the expected season gap,
            # count this as 'good'
            good_season_gaps = np.where(season_gaps <= self.expected_season_gap)
            metric_data[tau][good_season_gaps] = 1
            metric_data[tau] = metric_data[tau].sum() / len(season_gaps)
        return metric_data
