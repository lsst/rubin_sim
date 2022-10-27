from builtins import zip
import numpy as np
from .base_metric import BaseMetric

__all__ = [
    "NChangesMetric",
    "MinTimeBetweenStatesMetric",
    "NStateChangesFasterThanMetric",
    "MaxStateChangesWithinMetric",
    "TeffMetric",
    "OpenShutterFractionMetric",
    "BruteOSFMetric",
]


class NChangesMetric(BaseMetric):
    """
    Compute the number of times a column value changes.
    (useful for filter changes in particular).
    """

    def __init__(self, col="filter", order_by="observationStartMJD", **kwargs):
        self.col = col
        self.order_by = order_by
        super(NChangesMetric, self).__init__(col=[col, order_by], units="#", **kwargs)

    def run(self, data_slice, slice_point=None):
        idxs = np.argsort(data_slice[self.order_by])
        diff = data_slice[self.col][idxs][1:] != data_slice[self.col][idxs][:-1]
        return np.size(np.where(diff == True)[0])


class MinTimeBetweenStatesMetric(BaseMetric):
    """
    Compute the minimum time between changes of state in a column value.
    (useful for calculating fastest time between filter changes in particular).
    Returns delta time in minutes!
    """

    def __init__(
        self,
        change_col="filter",
        time_col="observationStartMJD",
        metric_name=None,
        **kwargs,
    ):
        """
        change_col = column that changes state
        time_col = column tracking time of each visit
        """
        self.change_col = change_col
        self.time_col = time_col
        if metric_name is None:
            metric_name = "Minimum time between %s changes (minutes)" % (change_col)
        super(MinTimeBetweenStatesMetric, self).__init__(
            col=[change_col, time_col], metric_name=metric_name, units="", **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # Sort on time, to be sure we've got filter (or other col) changes in the right order.
        idxs = np.argsort(data_slice[self.time_col])
        changes = (
            data_slice[self.change_col][idxs][1:] != data_slice[self.change_col][idxs][:-1]
        )
        condition = np.where(changes == True)[0]
        changetimes = data_slice[self.time_col][idxs][1:][condition]
        prevchangetime = np.concatenate(
            (
                np.array([data_slice[self.time_col][idxs][0]]),
                data_slice[self.time_col][idxs][1:][condition][:-1],
            )
        )
        dtimes = changetimes - prevchangetime
        dtimes *= 24 * 60
        if dtimes.size == 0:
            return self.badval
        return dtimes.min()


class NStateChangesFasterThanMetric(BaseMetric):
    """
    Compute the number of changes of state that happen faster than 'cutoff'.
    (useful for calculating time between filter changes in particular).
    'cutoff' should be in minutes.
    """

    def __init__(
        self,
        change_col="filter",
        time_col="observationStartMJD",
        metric_name=None,
        cutoff=20,
        **kwargs,
    ):
        """
        col = column tracking changes in
        time_col = column keeping the time of each visit
        cutoff = the cutoff value for the reduce method 'NBelow'
        """
        if metric_name is None:
            metric_name = "Number of %s changes faster than <%.1f minutes" % (
                change_col,
                cutoff,
            )
        self.change_col = change_col
        self.time_col = time_col
        self.cutoff = cutoff / 24.0 / 60.0  # Convert cutoff from minutes to days.
        super(NStateChangesFasterThanMetric, self).__init__(
            col=[change_col, time_col], metric_name=metric_name, units="#", **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # Sort on time, to be sure we've got filter (or other col) changes in the right order.
        idxs = np.argsort(data_slice[self.time_col])
        changes = (
            data_slice[self.change_col][idxs][1:] != data_slice[self.change_col][idxs][:-1]
        )
        condition = np.where(changes == True)[0]
        changetimes = data_slice[self.time_col][idxs][1:][condition]
        prevchangetime = np.concatenate(
            (
                np.array([data_slice[self.time_col][idxs][0]]),
                data_slice[self.time_col][idxs][1:][condition][:-1],
            )
        )
        dtimes = changetimes - prevchangetime
        return np.where(dtimes < self.cutoff)[0].size


class MaxStateChangesWithinMetric(BaseMetric):
    """
    Compute the maximum number of changes of state that occur within a given timespan.
    (useful for calculating time between filter changes in particular).
    'timespan' should be in minutes.
    """

    def __init__(
        self,
        change_col="filter",
        time_col="observationStartMJD",
        metric_name=None,
        timespan=20,
        **kwargs,
    ):
        """
        col = column tracking changes in
        time_col = column keeping the time of each visit
        timespan = the timespan to count the number of changes within (in minutes)
        """
        if metric_name is None:
            metric_name = "Max number of %s changes within %.1f minutes" % (
                change_col,
                timespan,
            )
        self.change_col = change_col
        self.time_col = time_col
        self.timespan = timespan / 24.0 / 60.0  # Convert timespan from minutes to days.
        super(MaxStateChangesWithinMetric, self).__init__(
            col=[change_col, time_col], metric_name=metric_name, units="#", **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # This operates slightly differently from the metrics above; those calculate only successive times
        # between changes, but here we must calculate the actual times of each change.
        # Check if there was only one observation (and return 0 if so).
        if data_slice[self.change_col].size == 1:
            return 0
        # Sort on time, to be sure we've got filter (or other col) changes in the right order.
        idxs = np.argsort(data_slice[self.time_col])
        changes = (
            data_slice[self.change_col][idxs][:-1] != data_slice[self.change_col][idxs][1:]
        )
        condition = np.where(changes == True)[0]
        changetimes = data_slice[self.time_col][idxs][1:][condition]
        # If there are 0 filter changes ...
        if changetimes.size == 0:
            return 0
        # Otherwise ..
        ct_plus = changetimes + self.timespan
        indx2 = np.searchsorted(changetimes, ct_plus, side="right")
        indx1 = np.arange(changetimes.size)
        nchanges = indx2 - indx1
        return nchanges.max()


class TeffMetric(BaseMetric):
    """
    Effective time equivalent for a given set of visits.
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        metric_name="tEff",
        fiducial_depth=None,
        teff_base=30.0,
        normed=False,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col
        if fiducial_depth is None:
            # From reference von Karman 500nm zenith seeing of 0.69"
            # median zenith dark seeing from sims_skybrightness_pre
            # airmass = 1
            # 2 "snaps" of 15 seconds each
            # m5_flat_sed sysEngVals from rubin_sim
            #   commit 6d03bd49550972e48648503ed60784a4e6775b82 (2021-05-18)
            # These include constants from:
            #   https://github.com/lsst-pst/syseng_throughputs/blob/master/notebooks/generate_sims_values.ipynb
            #   commit 7abb90951fcbc70d9c4d0c805c55a67224f9069f (2021-05-05)
            # See https://github.com/lsst-sims/smtn-002/blob/master/notebooks/teff_fiducial.ipynb
            self.depth = {
                "u": 23.71,
                "g": 24.67,
                "r": 24.24,
                "i": 23.82,
                "z": 23.21,
                "y": 22.40,
            }
        else:
            if isinstance(fiducial_depth, dict):
                self.depth = fiducial_depth
            else:
                raise ValueError("fiducial_depth should be None or dictionary")
        self.teff_base = teff_base
        self.normed = normed
        if self.normed:
            units = ""
        else:
            units = "seconds"
        super(TeffMetric, self).__init__(
            col=[m5_col, filter_col], metric_name=metric_name, units=units, **kwargs
        )
        if self.normed:
            self.comment = "Normalized effective time"
        else:
            self.comment = "Effect time"
        self.comment += (
            " of a series of observations, evaluating the equivalent amount of time"
        )
        self.comment += (
            " each observation would require if taken at a fiducial limiting magnitude."
        )
        self.comment += " Fiducial depths are : %s" % self.depth
        if self.normed:
            self.comment += " Normalized by the total amount of time actual on-sky."

    def run(self, data_slice, slice_point=None):
        filters = np.unique(data_slice[self.filter_col])
        teff = 0.0
        for f in filters:
            match = np.where(data_slice[self.filter_col] == f)[0]
            teff += (
                10.0 ** (0.8 * (data_slice[self.m5_col][match] - self.depth[f]))
            ).sum()
        teff *= self.teff_base
        if self.normed:
            # Normalize by the t_eff if each observation was at the fiducial depth.
            teff = teff / (self.teff_base * data_slice[self.m5_col].size)
        return teff


class OpenShutterFractionMetric(BaseMetric):
    """
    Compute the fraction of time the shutter is open compared to the total time spent observing.
    """

    def __init__(
        self,
        metric_name="OpenShutterFraction",
        slew_time_col="slewTime",
        exp_time_col="visitExposureTime",
        visit_time_col="visitTime",
        **kwargs,
    ):
        self.exp_time_col = exp_time_col
        self.visit_time_col = visit_time_col
        self.slew_time_col = slew_time_col
        super(OpenShutterFractionMetric, self).__init__(
            col=[self.exp_time_col, self.visit_time_col, self.slew_time_col],
            metric_name=metric_name,
            units="OpenShutter/TotalTime",
            **kwargs,
        )
        self.comment = (
            "Open shutter time (%s total) divided by total visit time "
            "(%s) + slewtime (%s)."
            % (self.exp_time_col, self.visit_time_col, self.slew_time_col)
        )

    def run(self, data_slice, slice_point=None):
        result = np.sum(data_slice[self.exp_time_col]) / np.sum(
            data_slice[self.slew_time_col] + data_slice[self.visit_time_col]
        )
        return result


class BruteOSFMetric(BaseMetric):
    """Assume I can't trust the slewtime or visittime colums.
    This computes the fraction of time the shutter is open, with no penalty for the first exposure
    after a long gap (e.g., 1st exposure of the night). Presumably, the telescope will need to focus,
    so there's not much a scheduler could do to optimize keeping the shutter open after a closure.
    """

    def __init__(
        self,
        metric_name="BruteOSFMetric",
        exp_time_col="visitExposureTime",
        mjd_col="observationStartMJD",
        maxgap=10.0,
        fudge=0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        maxgap : float (10.)
            The maximum gap between observations. Assume anything longer the dome has closed.
        fudge : float (0.)
            Fudge factor if a constant has to be added to the exposure time values (like in OpSim 3.61).
        exp_time_col : str ('expTime')
            The name of the exposure time column. Assumed to be in seconds.
        mjd_col : str ('observationStartMJD')
            The name of the start of the exposures. Assumed to be in units of days.
        """
        self.exp_time_col = exp_time_col
        self.maxgap = maxgap / 60.0 / 24.0  # convert from min to days
        self.mjd_col = mjd_col
        self.fudge = fudge
        super(BruteOSFMetric, self).__init__(
            col=[self.exp_time_col, mjd_col],
            metric_name=metric_name,
            units="OpenShutter/TotalTime",
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        times = np.sort(data_slice[self.mjd_col])
        diff = np.diff(times)
        good = np.where(diff < self.maxgap)
        open_time = np.sum(diff[good]) * 24.0 * 3600.0
        result = np.sum(data_slice[self.exp_time_col] + self.fudge) / float(open_time)
        return result
