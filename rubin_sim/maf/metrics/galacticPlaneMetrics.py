from types import MethodType
import numpy as np
from rubin_sim.maf.maps.galacticPlanePriorityMaps import (
    gp_priority_map_components_to_keys,
)
from .baseMetric import BaseMetric

__all__ = [
    "galplane_nvisits_thresholds",
    "galplane_priority_map_thresholds",
    "GalPlaneFootprintMetric",
    "GalPlaneTimePerFilterMetric",
]

# These are a suite of metrics aimed at evaluating high-level quantities regarding galactic plane
# coverage. The metrics here evaluate the coverage (just number of visits and exposure time per filter)
# in relation to the desired coverage from the galactic plane priority map.
# There is a related metric in transientTimeSampling which evaluates the cadence weighted by this same map.

TAU_OBS = np.array([2.0, 5.0, 11.0, 20.0, 46.5, 73.0])


def galplane_nvisits_thresholds(tau_obs, nyears=10):
    """ "Return estimated nvisits required to well-sample lightcurves that need sampling every tau_obs (days).

    This does a very basic estimate, just counting how many visits you would have if you distributed them
    at tau_obs intervals for a period of nyears, assuming a season length of 6.5 years and that visits in
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
        Estimated number of visits required to well sample lightcurves which require sampling on tau_obs
    """
    # How many nights in the survey
    nnights_total = 365.25 * nyears
    # Estimate that the observing season is 6.5 months long, not a full year
    # And account for the fact that each 'night' will (currently) need 2 visits
    nnights = nnights_total * (6.5 / 12) * 2
    n_visits_thresholds = nnights / tau_obs
    return n_visits_thresholds


def galplane_priority_map_thresholds(science_map):
    """Return minimum threshold for priority maps, when considering filter balance.

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
        if metricval["nObservations"] >= nvisits_thresh:
            return metricval["map_priority"]
        else:
            return 0

    return _nvisits_cut


class GalPlaneFootprintMetric(BaseMetric):
    """Evaluate the survey overlap with desired regions in the Galactic Plane
    and Magellanic Clouds, by referencing the pre-computed priority maps provided.
    These priority maps are keyed by science area (science_map) and per filter.
    The returned metric values are summed over all filters.

    Parameters
    ----------
    science_map : `str`
        Name of the priority footprint map key to use from the column headers contained in the
        priority_GalPlane_footprint_map_data tables.
    tau_obs : `np.ndarray` or `list` of `float`, opt
        Timescales of minimum-required observations intervals for various classes of time variability.
        Default (None), uses TAU_OBS. In general, this should be left as the default and consistent
        across all galactic-plane oriented metrics.
    mag_cuts : `dict` of `float`, opt
        Magnitudes to use as cutoffs for individual image depths.
        Default None uses a default set of values which correspond roughly to the 50th percentile.
    filterCol : `str`, opt
        Name of the filter column. Default 'filter'.
    m5Col : `str`, opt
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
        filterCol="filter",
        m5Col="fiveSigmaDepth",
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
        self.filterCol = filterCol
        self.m5Col = m5Col
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
        if "metricName" not in kwargs:
            metricName = f"GalplaneFootprintMetric_{self.science_map}"
        else:
            metricName = kwargs["metricName"]
            del kwargs["metricName"]
        for tau, nvisits in zip(self.tau_obs, self.nvisits_threshold):
            tauReduceName = f"reduceTau_{tau:.1f}".replace(".", "_")
            rfunc = help_set_reduce_func(self, None, nvisits)
            # MethodType(newmethod, self) is how this next line SHOULD go
            # but that doesn't work .. the scope isn't correct somehow.
            # Using this alternate string of tauReduceName *does* work.
            setattr(self, tauReduceName, MethodType(rfunc, tauReduceName))
        super().__init__(
            col=[self.filterCol, self.m5Col],
            metricName=metricName,
            maps=maps,
            **kwargs,
        )
        self.reduceOrder = {"NObs": 0, "NObsPriority": 1}
        for i, tau in enumerate(self.tau_obs):
            rName = f"Tau_{tau:.1f}".replace(".", "_")
            self.reduceOrder[rName] = i + 2

    def run(self, dataSlice, slicePoint):
        """Calculate the number of observations that meet the mag_cut values at each slicePoint.
        Also calculate the number of observations * the priority map summed over all filter.
        Return both of these values as a dictionary.
        """
        # Check if we want to evaluate this part of the sky, or if the weight is below threshold.
        mapkey = gp_priority_map_components_to_keys("sum", self.science_map)
        priority = slicePoint[mapkey]
        if priority <= self.priority_map_threshold:
            return self.badval
        # Count the number of observations per filter, above the mag cuts
        nObs = 0
        nObsPriority = 0
        for f in self.filterlist:
            obs_in_filter = np.where(dataSlice[self.filterCol] == f)
            above_cut = np.where(
                dataSlice[obs_in_filter][self.m5Col] >= self.mag_cuts[f]
            )
            nObs += len(above_cut[0])
            mapkey = gp_priority_map_components_to_keys(f, self.science_map)
            nObsPriority += len(above_cut[0]) * slicePoint[mapkey]

        return {
            "nObservations": nObs,
            "nObsPriority": nObsPriority,
            "map_priority": priority,
        }

    def reduceNObs(self, metricval):
        return metricval["nObservations"]

    def reduceNObsPriority(self, metricval):
        return metricval["nObsPriority"]


class GalPlaneTimePerFilterMetric(BaseMetric):
    """Evaluate the fraction of exposure time spent in each filter as a fraction of the
    total exposure time dedicated to that healpix in the weighted galactic plane priority maps.

    Parameters
    ----------
    scienceMap : `str`
        Name of the priority footprint map key to use from the column headers contained in the
        priority_GalPlane_footprint_map_data tables.
    magCuts : `dict` of `float`, opt
        Magnitudes to use as cutoffs for individual image depths.
        Default None uses a default set of values which correspond roughly to the 50th percentile.
    mjdCol : `str`, opt
        Name of the observation start MJD column. Default 'observationStartMJD'.
    expTimeCol : `str`, opt
        Name of the exposure time column. Default 'visitExposureTime'.
    filterCol : `str`, opt
        Name of the filter column. Default 'filter'.
    m5Col : `str`, opt
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
        mjdCol="observationStartMJD",
        expTimeCol="visitExposureTime",
        filterCol="filter",
        m5Col="fiveSigmaDepth",
        filterlist=None,
        **kwargs,
    ):
        self.science_map = science_map
        self.priority_map_threshold = galplane_priority_map_thresholds(self.science_map)
        self.filterCol = filterCol
        self.m5Col = m5Col
        self.mjdCol = mjdCol
        self.expTimeCol = expTimeCol
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
        if "metricName" not in kwargs:
            metricName = f"GalplaneTimePerFilter_{self.science_map}"
        else:
            metricName = kwargs["metricName"]
            del kwargs["metricName"]
        super().__init__(
            col=[self.filterCol, self.m5Col, self.mjdCol, self.expTimeCol],
            maps=maps,
            metricName=metricName,
            **kwargs,
        )
        # Put the reduce functions into filter order
        for i, f in enumerate(self.filterlist):
            self.reduceOrder[f"{f}"] = i

    def run(self, dataSlice, slicePoint):
        """Calculate the ratio of the actual on-sky exposure time per filter
        compared to the ideal on-sky exposure time per filter at this point on the sky across all filters.
        """
        # Check if we want to evaluate this part of the sky, or if the weight is below threshold.
        weight_all_filters = slicePoint[
            gp_priority_map_components_to_keys("sum", self.science_map)
        ]
        if weight_all_filters <= self.priority_map_threshold:
            return self.badval

        # Calculate the ideal weighting per filter compared to all filters at this point in the sky
        relative_filter_weight = {}
        for f in self.filterlist:
            mapkey = gp_priority_map_components_to_keys(f, self.science_map)
            relative_filter_weight[f] = slicePoint[mapkey] / weight_all_filters

        exp_time_per_filter = {}
        for f in self.filterlist:
            # Select observations within the OpSim for the current filter
            # which match the S/N requirement, and extract the exposure times
            # for those observations
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.mag_cuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            # Now calculate the actual fraction of exposure time spent
            # in this filter for the current slicePoint, relative to the total
            # exposure time spent on this slicePoint.
            # Note that this includes dithered observations.
            # If no exposures are expected in this filter, this returns 1
            # on the principle that 100% of the expected observations are
            # provided, and additional data in other filters is usually welcome
            exp_time_per_filter[f] = dataSlice[self.expTimeCol][match].sum()

        # Calculate the time on-sky in each filter that overlaps this point, and meets mag_cuts
        total_expt_mag_cut = 0
        for f in self.filterlist:
            total_expt_mag_cut += exp_time_per_filter[f].sum()

        # normalize by the relative filter weight. Ideally metric results are close to 1.
        normalized_exp_time = {}
        for f in self.filterlist:
            if total_expt_mag_cut == 0:
                normalized_exp_time[f] = 0
            else:
                # If no exposures are expected in this filter for this location,
                # thais metric returns the mask val for this filter only.
                if relative_filter_weight[f] > 0:
                    normalized_exp_time[f] = (
                        exp_time_per_filter[f]
                        / total_expt_mag_cut
                        / relative_filter_weight[f]
                    )
                else:
                    normalized_exp_time[f] = self.badval
        return normalized_exp_time

    def reduceu(self, metricval):
        return metricval["u"]

    def reduceg(self, metricval):
        return metricval["g"]

    def reducer(self, metricval):
        return metricval["r"]

    def reducei(self, metricval):
        return metricval["i"]

    def reducez(self, metricval):
        return metricval["z"]

    def reducey(self, metricval):
        return metricval["y"]
