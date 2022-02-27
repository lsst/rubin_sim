################################################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
from types import MethodType
import numpy as np
from rubin_sim.utils import calcSeason
from rubin_sim.maf.maps.galacticPlanePriorityMaps import (
    gp_priority_map_components_to_keys,
)
from .galacticPlaneMetrics import galplane_priority_map_thresholds
from .seasonMetrics import findSeasonEdges
from .baseMetric import BaseMetric

__all__ = [
    "calc_interval_decay",
    "GalPlaneVisitIntervalsTimescaleMetric",
    "GalPlaneSeasonGapsTimescaleMetric",
]

SCIENCE_MAP_OPTIONS = [
    "combined_map",
    "galactic_plane_map",
    "magellenic_clouds_map",
    "galactic_bulge_map",
    "clementini_stellarpops_map",
    "bonito_sfr_map",
    "globular_clusters_map",
    "open_clusters_map",
    "zucker_sfr_map",
    "pencilbeams_map",
    "xrb_priority_map",
]

TAU_OBS = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])


def calc_interval_decay(delta_tobs, tau):
    # Decay constant for metric value relationship as function of obs interval
    K = 1.0 / tau
    m = np.exp(-K * (delta_tobs - tau))
    # But where observation interval is <= tau, replace with 1
    m[np.where(delta_tobs <= tau)] = 1.0
    return m.sum()


# this is a bit of a hack .. it helps us use a variety of tau_obs values,
# and dynamically set up reduce functions
def help_set_reduce_func(obj, metricval, tau):
    def _reduceTau(obj, metricval):
        return metricval[tau]

    return _reduceTau


class GalPlaneVisitIntervalsTimescaleMetric(BaseMetric):
    """Evaluate the intervals between sequential observations in a
    lightcurve relative to the scientifically desired sampling interval.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column headers contained in the
        priority_GalPlane_footprint_map_data tables.
    tau_obs : `np.ndarray` or `list` of `float`, opt
        Timescales of minimum-required observations intervals for various classes of time variability.
        Default (None), uses TAU_OBS. In general, this should be left as the default and consistent
        across all galactic-plane oriented metrics.
    mag_limit : `float`, opt
        Magnitude limit to use as a cutoff for various observations.
        Default 22.0.
    mjdCol : `str`, opt
        The name of the observation start MJD column. Default 'observationStartMJD'.
    m5Col : `str', opt
        The name of the five sigma depth column. Default 'fiveSigmaDepth'.
    """

    def __init__(
        self,
        science_map,
        tau_obs=None,
        mag_limit=22.0,
        mjdCol="observationStartMJD",
        m5Col="fiveSigmaDepth",
        **kwargs,
    ):
        if science_map not in SCIENCE_MAP_OPTIONS:
            message = f"Must specify scienceMap from the options expected in the map: {SCIENCE_MAP_OPTIONS}"
            message += f" but received {science_map}"
            raise ValueError(message)
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        # tau_obs is an array of minimum-required observation intervals for
        # four categories of time variability
        if tau_obs is not None:
            self.tau_obs = tau_obs
        else:
            self.tau_obs = TAU_OBS
        # Create reduce functions for the class that are return the metric for each value in tau_obs

        self.mag_limit = mag_limit
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        maps = ["GalacticPlanePriorityMap"]
        if "metricName" not in kwargs:
            metricName = f"GalPlaneVisitIntervalsTimescales_{self.science_map}"
        else:
            metricName = kwargs["metricName"]
            del kwargs["metricName"]
        for tau in self.tau_obs:
            tauReduceName = f"reduceTau_{tau:.1f}".replace(".", "_")
            newmethod = help_set_reduce_func(self, None, tau)
            setattr(self, tauReduceName, MethodType(newmethod, tauReduceName))
        super().__init__(
            col=[self.mjdCol, self.m5Col], metricName=metricName, maps=maps, **kwargs
        )
        for i, tau in enumerate(self.tau_obs):
            self.reduceOrder[
                f"reduceTau_{tau:.1f}".replace(".", "_").replace("reduce", "")
            ] = i

    def run(self, dataSlice, slicePoint=None):
        # Check if we want to evaluate this part of the sky, or if the weight is below threshold.
        if (
            slicePoint[gp_priority_map_components_to_keys("sum", self.science_map)]
            <= self.priority_map_threshold
        ):
            return self.badval
        # Select observations in the time sequence that fulfill the
        # S/N requirements:
        match = np.where(dataSlice[self.m5Col] >= self.mag_limit)[0]
        # We need at least two visits which match these requirements to calculate visit gaps
        if len(match) < 2:
            return self.badval
        # Find the time gaps between visits (in any filter)
        times = dataSlice[self.mjdCol][match]
        times.sort()
        delta_tobs = np.diff(times)
        # Compare the time gap distribution to the time gap required to characterize variability
        metric_data = {}
        for tau in self.tau_obs:
            # Normalize
            metric_data[tau] = calc_interval_decay(delta_tobs, tau) / len(delta_tobs)
        return metric_data


class GalPlaneSeasonGapsTimescaleMetric(BaseMetric):
    """Metric to evaluate the gap between sequential seasonal gaps in
    observations in a lightcurve relative to the scientifically desired
    sampling interval.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column headers contained in the
        priority_GalPlane_footprint_map_data tables.
    tau_var : `np.ndarray` or `list` of `float`, opt
        Timescales of variability for various classes of time variability.
        Default (None), uses TAU_OBS * 5. In general, this should be left as the default and consistent
        across all galactic-plane oriented metrics.
    mag_limit : `float`, opt
        Magnitude limit to use as a cutoff for various observations.
        Default 22.0.
    mjdCol : `str`, opt
        The name of the observation start MJD column. Default 'observationStartMJD'.
    m5Col : `str', opt
        The name of the five sigma depth column. Default 'fiveSigmaDepth'.
    """

    def __init__(
        self,
        science_map,
        tau_var=None,
        mag_limit=22.0,
        mjdCol="observationStartMJD",
        m5Col="fiveSigmaDepth",
        **kwargs,
    ):
        if science_map not in SCIENCE_MAP_OPTIONS:
            message = f"Must specify scienceMap from the options expected in the map: {SCIENCE_MAP_OPTIONS}"
            message += f" but received {science_map}"
            raise ValueError(message)
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        # tau_obs is an array of minimum-required observation intervals for
        # four categories of time variability; tau_var is the related timescale for the variability
        # (tau_var is approximately 5 * tau_obs, in general)
        if tau_var is not None:
            self.tau_var = tau_var
        else:
            self.tau_var = TAU_OBS * 5
        ### NOTE: I would recommend dropping tau_var 10 and 25 from this analysis unless the metric is changed
        ### these intervals are so short they will *always* be dropped during the season gap
        self.mag_limit = mag_limit
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        if "metricName" not in kwargs:
            metricName = f"GalPlaneSeasonGapsTimescales_{self.science_map}"
        else:
            metricName = kwargs["metricName"]
            del kwargs["metricName"]
        for tau in self.tau_var:
            tauReduceName = f"reduceTau_{tau:.1f}".replace(".", "_")
            newmethod = help_set_reduce_func(self, None, tau)
            setattr(self, tauReduceName, MethodType(newmethod, tauReduceName))
        super().__init__(col=[self.mjdCol, self.m5Col], metricName=metricName, **kwargs)
        for i, tau in enumerate(self.tau_var):
            self.reduceOrder[
                f"reduceTau_{tau:.1f}".replace(".", "_").replace("reduce", "")
            ] = i

    def run(self, dataSlice, slicePoint):
        # Check if we want to evaluate this part of the sky, or if the weight is below threshold.
        if (
            slicePoint[gp_priority_map_components_to_keys("sum", self.science_map)]
            <= self.priority_map_threshold
        ):
            return self.badval
        # Find the length of the gaps between each season
        times = dataSlice[self.mjdCol]
        times.sort()
        # data = np.sort(dataSlice[self.mjdCol], order=self.mjdCol)
        # SlicePoints ra/dec are always in radians - convert to degrees to calculate season
        seasons = calcSeason(np.degrees(slicePoint["ra"]), times)
        firstOfSeason, lastOfSeason = findSeasonEdges(seasons)
        # season_lengths = times[lastOfSeason] - times[firstOfSeason]  # would this match interval calc better?
        season_gaps = times[firstOfSeason][1:] - times[lastOfSeason][:-1]
        if len(season_gaps) == 0:
            return self.badval
        metric_data = {}
        metric_data["seasons_start"] = times[firstOfSeason]
        metric_data["seasons_end"] = times[lastOfSeason]
        metric_data["season_gaps"] = season_gaps
        for i, tau in enumerate(self.tau_var):
            metric_data[tau] = calc_interval_decay(season_gaps, tau)
            metric_data[tau] /= len(season_gaps)
        return metric_data
