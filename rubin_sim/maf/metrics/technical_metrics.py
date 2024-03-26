__all__ = (
    "NChangesMetric",
    "MinTimeBetweenStatesMetric",
    "NStateChangesFasterThanMetric",
    "MaxStateChangesWithinMetric",
    "OpenShutterFractionMetric",
    "BruteOSFMetric",
)

import numpy as np

from .base_metric import BaseMetric


class NChangesMetric(BaseMetric):
    """Compute the number of times a column value changes.
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
    """Compute the minimum time between changes of state in a column value.
    (useful for calculating fastest time between filter changes in particular).
    Returns delta time in minutes!

    Parameters
    ----------
    change_col : `str`
        Column that we are tracking changes in.
    time_col : str
        Column with the time of each visit
    """

    def __init__(
        self,
        change_col="filter",
        time_col="observationStartMJD",
        metric_name=None,
        **kwargs,
    ):
        self.change_col = change_col
        self.time_col = time_col
        if metric_name is None:
            metric_name = "Minimum time between %s changes (minutes)" % (change_col)
        super(MinTimeBetweenStatesMetric, self).__init__(
            col=[change_col, time_col], metric_name=metric_name, units="", **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # Sort on time, to be sure we've got changes in the right order.
        idxs = np.argsort(data_slice[self.time_col])
        changes = data_slice[self.change_col][idxs][1:] != data_slice[self.change_col][idxs][:-1]
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

    Parameters
    ----------
    change_col : `str`
        Column that we are tracking changes in.
    time_col : str
        Column with the time of each visit
    cutoff : `float`
        The cutoff value for the time between changes (in minutes).
    """

    def __init__(
        self,
        change_col="filter",
        time_col="observationStartMJD",
        metric_name=None,
        cutoff=20,
        **kwargs,
    ):
        if metric_name is None:
            metric_name = "Number of %s changes faster than <%.1f minutes" % (
                change_col,
                cutoff,
            )
        self.change_col = change_col
        self.time_col = time_col
        # Convert cutoff from minutes to days.
        self.cutoff = cutoff / 24.0 / 60.0
        super(NStateChangesFasterThanMetric, self).__init__(
            col=[change_col, time_col], metric_name=metric_name, units="#", **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # Sort on time, to be sure we've got changes in the right order.
        idxs = np.argsort(data_slice[self.time_col])
        changes = data_slice[self.change_col][idxs][1:] != data_slice[self.change_col][idxs][:-1]
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
    """Compute the maximum number of changes of state that occur
    within a given timespan.
    (useful for calculating time between filter changes in particular).

    Parameters
    ----------
    change_col : `str`
        Column that we are tracking changes in.
    time_col : str
        Column with the time of each visit
    timespan : `float`
        The timespan to count the number of changes within (in minutes).
    """

    def __init__(
        self,
        change_col="filter",
        time_col="observationStartMJD",
        metric_name=None,
        timespan=20,
        **kwargs,
    ):
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
        # This operates slightly differently from the metrics above;
        # those calculate only successive times between changes, but here
        # we must calculate the actual times of each change.
        # Check if there was only one observation (and return 0 if so).
        if data_slice[self.change_col].size == 1:
            return 0
        # Sort on time, to be sure we've got changes in the right order.
        idxs = np.argsort(data_slice[self.time_col])
        changes = data_slice[self.change_col][idxs][:-1] != data_slice[self.change_col][idxs][1:]
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


class OpenShutterFractionMetric(BaseMetric):
    """Compute the fraction of time the shutter is open
    compared to the total time spent observing.
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
        self.comment = "Open shutter time (%s total) divided by total visit time " "(%s) + slewtime (%s)." % (
            self.exp_time_col,
            self.visit_time_col,
            self.slew_time_col,
        )

    def run(self, data_slice, slice_point=None):
        result = np.sum(data_slice[self.exp_time_col]) / np.sum(
            data_slice[self.slew_time_col] + data_slice[self.visit_time_col]
        )
        return result


class BruteOSFMetric(BaseMetric):
    """Assume I can't trust the slewtime or visittime colums.
    This computes the fraction of time the shutter is open,
    with no penalty for the first exposure after a long gap
    (e.g., 1st exposure of the night).
    Presumably, the telescope will need to focus, so there's not much a
    scheduler could do to optimize keeping the shutter open after a closure.

    Parameters
    ----------
    maxgap : `float`
        The maximum gap between observations, in minutes.
        Assume anything longer the dome has closed.
    fudge : `float`
        Fudge factor if a constant has to be added to the exposure time values.
        This time (in seconds) is added to the exposure time.
    exp_time_col : `str`
        The name of the exposure time column. Assumed to be in seconds.
    mjd_col : `str`
        The name of the start of the exposures. Assumed to be in units of days.
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
