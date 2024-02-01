__all__ = (
    "galplane_nvisits_thresholds",
    "galplane_priority_map_thresholds",
    "GalPlaneFootprintMetric",
    "GalPlaneTimePerFilterMetric",
)

from types import MethodType

import numpy as np

from rubin_sim.maf.maps.galactic_plane_priority_maps import gp_priority_map_components_to_keys

from .base_metric import BaseMetric

# These are a suite of metrics aimed at evaluating high-level
# quantities regarding galactic plane coverage.
# The metrics here evaluate the coverage
# (just number of visits and exposure time per filter)
# in relation to the desired coverage from the galactic plane priority map.
# There is a related metric in transientTimeSampling which
# evaluates the cadence weighted by this same map.

TAU_OBS = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])


def galplane_nvisits_thresholds(tau_obs, nyears=10):
    """Return estimated nvisits required to well-sample lightcurves
    that need sampling every tau_obs (days).

    This does a very basic estimate, just counting how many visits you
    would have if you distributed them at tau_obs intervals for a period
    of nyears, assuming a season length of 6.5 years and that visits in
    each night are in pairs.

    Parameters
    ----------
    tau_obs : `np.ndarray` or `list` of `float`
        Timescale that variability must be sampled at, in days.
    nyears : `float`, opt
        Number of years in the survey (as sampled).  Default 10.

    Returns
    -------
    n_visits_thresholds : `np.ndarray`
        Estimated number of visits required to well sample lightcurves
        which require sampling on tau_obs
    """
    # How many nights in the survey
    nnights_total = 365.25 * nyears
    # Estimate that the observing season is 6.5 months long, not a full year
    # And account for the fact that each 'night' will (currently) need 2 visits
    nnights = nnights_total * (6.5 / 12) * 2
    n_visits_thresholds = nnights / tau_obs
    return n_visits_thresholds


def galplane_priority_map_thresholds(science_map):
    """Return minimum threshold for priority maps,
    when considering filter balance.

    Parameters
    ----------
    science_map : `str`
        The name of the science map in the galactic plane priority map.

    Returns
    -------
    priority_threshold : `float`
        The minimum threshold to consider from the priority map
    """
    if science_map == "galactic_plane":
        priority_threshold = 0.4
    elif science_map == "combined_map":
        priority_threshold = 0.001
    else:
        priority_threshold = 0.0
    return priority_threshold


# this is a bit of a hack .. it helps us use a variety of tau_obs values,
# and dynamically set up reduce functions
def help_set_reduce_func(obj, metricval, nvisits_thresh):
    def _nvisits_cut(obj, metricval):
        if metricval["n_observations"] >= nvisits_thresh:
            return metricval["map_priority"]
        else:
            return 0

    return _nvisits_cut


class GalPlaneFootprintMetric(BaseMetric):
    """Evaluate the survey overlap with desired regions in the
    Galactic Plane and Magellanic Clouds, by referencing the
    pre-computed priority maps provided.
    These priority maps are keyed by science area (science_map) and per filter.
    The returned metric values are summed over all filters.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column
        headers contained in the priority_GalPlane_footprint_map_data tables.
    tau_obs : `np.ndarray` or `list` of `float`, opt
        Timescales of minimum-required observations intervals for
        various classes of time variability.
        Default (None), uses TAU_OBS. In general, this should be left as
        the default and consistent across all galactic-plane oriented metrics.
    mag_cuts : `dict` of `float`, opt
        Magnitudes to use as cutoffs for individual image depths.
        Default None uses a default set of values which correspond
        roughly to the 50th percentile.
    filter_col : `str`, opt
        Name of the filter column. Default 'filter'.
    m5_col : `str`, opt
        Name of the five-sigma depth column. Default 'fiveSigmaDepth'.
    filterlist : `list` of `str`, opt
        The filters to consider from the priority map and observations.
        Default None uses u, g, r, i, z, and y.
    metricName : `str`, opt
        Name for the metric. Default 'GalPlaneFootprintMetric_{scienceMap}
    """

    def __init__(
        self,
        science_map,
        tau_obs=None,
        mag_cuts=None,
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        filterlist=None,
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
        self.nvisits_threshold = galplane_nvisits_thresholds(self.tau_obs)
        self.filter_col = filter_col
        self.m5_col = m5_col
        if filterlist is not None:
            self.filterlist = filterlist
        else:
            self.filterlist = ["u", "g", "r", "i", "z", "y"]
        if mag_cuts is not None:
            self.mag_cuts = mag_cuts
        else:
            self.mag_cuts = {
                "u": 22.7,
                "g": 24.1,
                "r": 23.7,
                "i": 23.1,
                "z": 22.2,
                "y": 21.4,
            }
        maps = ["GalacticPlanePriorityMap"]
        if "metric_name" not in kwargs:
            metric_name = f"GalplaneFootprintMetric_{self.science_map}"
        else:
            metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        for tau, nvisits in zip(self.tau_obs, self.nvisits_threshold):
            tau_reduce_name = f"reduce_Tau_{tau:.1f}".replace(".", "_")
            rfunc = help_set_reduce_func(self, None, nvisits)
            # MethodType(newmethod, self) is how this next line SHOULD go
            # but that doesn't work .. the scope isn't correct somehow.
            # Using this alternate string of tau_reduce_name *does* work.
            setattr(self, tau_reduce_name, MethodType(rfunc, tau_reduce_name))
        super().__init__(
            col=[self.filter_col, self.m5_col],
            metric_name=metric_name,
            maps=maps,
            **kwargs,
        )
        self.reduce_order = {"n_obs": 0, "n_obs_priority": 1}
        for i, tau in enumerate(self.tau_obs):
            r_name = f"Tau_{tau:.1f}".replace(".", "_")
            self.reduce_order[r_name] = i + 2

    def run(self, data_slice, slice_point):
        """Calculate the number of observations that meet the mag_cut values
        at each slice_point.

        Also calculate the number of observations * the priority map summed
        over all filter. Return both of these values as a dictionary.
        """
        # Check if we want to evaluate this part of the sky,
        # or if the weight is below threshold.
        mapkey = gp_priority_map_components_to_keys("sum", self.science_map)
        priority = slice_point[mapkey]
        if priority <= self.priority_map_threshold:
            return self.badval
        # Count the number of observations per filter, above the mag cuts
        n_obs = 0
        n_obs_priority = 0
        for f in self.filterlist:
            obs_in_filter = np.where(data_slice[self.filter_col] == f)
            above_cut = np.where(data_slice[obs_in_filter][self.m5_col] >= self.mag_cuts[f])
            n_obs += len(above_cut[0])
            mapkey = gp_priority_map_components_to_keys(f, self.science_map)
            n_obs_priority += len(above_cut[0]) * slice_point[mapkey]

        return {
            "n_observations": n_obs,
            "n_obs_priority": n_obs_priority,
            "map_priority": priority,
        }

    def reduce_n_obs(self, metricval):
        return metricval["n_observations"]

    def reduce_n_obs_priority(self, metricval):
        return metricval["n_obs_priority"]


class GalPlaneTimePerFilterMetric(BaseMetric):
    """Evaluate the fraction of exposure time spent in each filter as a
    fraction of the total exposure time dedicated to that healpix in the
    weighted galactic plane priority maps.

    Parameters
    ----------
    scienceMap : `str`
        Name of the priority footprint map key to use from the column
        headers contained in the
        priority_GalPlane_footprint_map_data tables.
    magCuts : `dict` of `float`, opt
        Magnitudes to use as cutoffs for individual image depths.
        Default None uses a default set of values which correspond
        roughly to the 50th percentile.
    mjd_col : `str`, opt
        Name of the observation start MJD column.
        Default 'observationStartMJD'.
    exp_time_col : `str`, opt
        Name of the exposure time column. Default 'visitExposureTime'.
    filter_col : `str`, opt
        Name of the filter column. Default 'filter'.
    m5_col : `str`, opt
        Name of the five-sigma depth column. Default 'fiveSigmaDepth'.
    filterlist : `list` of `str`, opt
        The filters to consider from the priority map and observations.
        Default None uses u, g, r, i, z, and y.
    metricName : `str`, opt
        Name for the metric. Default 'GalPlaneFootprintMetric_{scienceMap}
    """

    def __init__(
        self,
        science_map,
        mag_cuts=None,
        mjd_col="observationStartMJD",
        exp_time_col="visitExposureTime",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        filterlist=None,
        **kwargs,
    ):
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        self.filter_col = filter_col
        self.m5_col = m5_col
        self.mjd_col = mjd_col
        self.exp_time_col = exp_time_col
        if filterlist is not None:
            self.filterlist = filterlist
        else:
            self.filterlist = ["u", "g", "r", "i", "z", "y"]
        if mag_cuts is not None:
            self.mag_cuts = mag_cuts
        else:
            self.mag_cuts = {
                "u": 22.7,
                "g": 24.1,
                "r": 23.7,
                "i": 23.1,
                "z": 22.2,
                "y": 21.4,
            }
        maps = ["GalacticPlanePriorityMap"]
        if "metric_name" not in kwargs:
            metric_name = f"GalplaneTimePerFilter_{self.science_map}"
        else:
            metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        super().__init__(
            col=[self.filter_col, self.m5_col, self.mjd_col, self.exp_time_col],
            maps=maps,
            metric_name=metric_name,
            **kwargs,
        )
        # Put the reduce functions into filter order
        for i, f in enumerate(self.filterlist):
            self.reduce_order[f"{f}"] = i

    def run(self, data_slice, slice_point):
        """Calculate the ratio of the actual on-sky exposure time per filter
        compared to the ideal on-sky exposure time per filter at this point
        on the sky across all filters.
        """
        # Check if we want to evaluate this part of the sky,
        # or if the weight is below threshold.
        weight_all_filters = slice_point[gp_priority_map_components_to_keys("sum", self.science_map)]
        if weight_all_filters <= self.priority_map_threshold:
            return self.badval

        # Calculate the ideal weighting per filter compared to all
        # filters at this point in the sky
        relative_filter_weight = {}
        for f in self.filterlist:
            mapkey = gp_priority_map_components_to_keys(f, self.science_map)
            relative_filter_weight[f] = slice_point[mapkey] / weight_all_filters

        exp_time_per_filter = {}
        for f in self.filterlist:
            # Select observations within the OpSim for the current filter
            # which match the S/N requirement, and extract the exposure times
            # for those observations
            idx1 = np.where(data_slice[self.filter_col] == f)[0]
            idx2 = np.where(data_slice[self.m5_col] >= self.mag_cuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            # Now calculate the actual fraction of exposure time spent
            # in this filter for the current slice_point, relative to the total
            # exposure time spent on this slice_point.
            # Note that this includes dithered observations.
            # If no exposures are expected in this filter, this returns 1
            # on the principle that 100% of the expected observations are
            # provided, and additional data in other filters is usually welcome
            exp_time_per_filter[f] = data_slice[self.exp_time_col][match].sum()

        # Calculate the time on-sky in each filter that overlaps this point,
        # and meets mag_cuts
        total_expt_mag_cut = 0
        for f in self.filterlist:
            total_expt_mag_cut += exp_time_per_filter[f].sum()

        # normalize by the relative filter weight.
        # Ideally metric results are close to 1.
        normalized_exp_time = {}
        for f in self.filterlist:
            if total_expt_mag_cut == 0:
                normalized_exp_time[f] = 0
            else:
                # If no exposures are expected in this filter for this
                # location, metric returns the mask val for this filter only.
                if relative_filter_weight[f] > 0:
                    normalized_exp_time[f] = (
                        exp_time_per_filter[f] / total_expt_mag_cut / relative_filter_weight[f]
                    )
                else:
                    normalized_exp_time[f] = self.badval
        return normalized_exp_time

    def reduce_u(self, metricval):
        return metricval["u"]

    def reduce_g(self, metricval):
        return metricval["g"]

    def reduce_r(self, metricval):
        return metricval["r"]

    def reduce_i(self, metricval):
        return metricval["i"]

    def reduce_z(self, metricval):
        return metricval["z"]

    def reduce_y(self, metricval):
        return metricval["y"]
