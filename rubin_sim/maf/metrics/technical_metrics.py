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

    def __init__(self, col="filter", orderBy="observationStartMJD", **kwargs):
        self.col = col
        self.orderBy = orderBy
        super(NChangesMetric, self).__init__(col=[col, orderBy], units="#", **kwargs)

    def run(self, dataSlice, slicePoint=None):
        idxs = np.argsort(dataSlice[self.orderBy])
        diff = dataSlice[self.col][idxs][1:] != dataSlice[self.col][idxs][:-1]
        return np.size(np.where(diff == True)[0])


class MinTimeBetweenStatesMetric(BaseMetric):
    """
    Compute the minimum time between changes of state in a column value.
    (useful for calculating fastest time between filter changes in particular).
    Returns delta time in minutes!
    """

    def __init__(
        self,
        changeCol="filter",
        timeCol="observationStartMJD",
        metricName=None,
        **kwargs,
    ):
        """
        changeCol = column that changes state
        timeCol = column tracking time of each visit
        """
        self.changeCol = changeCol
        self.timeCol = timeCol
        if metricName is None:
            metricName = "Minimum time between %s changes (minutes)" % (changeCol)
        super(MinTimeBetweenStatesMetric, self).__init__(
            col=[changeCol, timeCol], metricName=metricName, units="", **kwargs
        )

    def run(self, dataSlice, slicePoint=None):
        # Sort on time, to be sure we've got filter (or other col) changes in the right order.
        idxs = np.argsort(dataSlice[self.timeCol])
        changes = (
            dataSlice[self.changeCol][idxs][1:] != dataSlice[self.changeCol][idxs][:-1]
        )
        condition = np.where(changes == True)[0]
        changetimes = dataSlice[self.timeCol][idxs][1:][condition]
        prevchangetime = np.concatenate(
            (
                np.array([dataSlice[self.timeCol][idxs][0]]),
                dataSlice[self.timeCol][idxs][1:][condition][:-1],
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
        changeCol="filter",
        timeCol="observationStartMJD",
        metricName=None,
        cutoff=20,
        **kwargs,
    ):
        """
        col = column tracking changes in
        timeCol = column keeping the time of each visit
        cutoff = the cutoff value for the reduce method 'NBelow'
        """
        if metricName is None:
            metricName = "Number of %s changes faster than <%.1f minutes" % (
                changeCol,
                cutoff,
            )
        self.changeCol = changeCol
        self.timeCol = timeCol
        self.cutoff = cutoff / 24.0 / 60.0  # Convert cutoff from minutes to days.
        super(NStateChangesFasterThanMetric, self).__init__(
            col=[changeCol, timeCol], metricName=metricName, units="#", **kwargs
        )

    def run(self, dataSlice, slicePoint=None):
        # Sort on time, to be sure we've got filter (or other col) changes in the right order.
        idxs = np.argsort(dataSlice[self.timeCol])
        changes = (
            dataSlice[self.changeCol][idxs][1:] != dataSlice[self.changeCol][idxs][:-1]
        )
        condition = np.where(changes == True)[0]
        changetimes = dataSlice[self.timeCol][idxs][1:][condition]
        prevchangetime = np.concatenate(
            (
                np.array([dataSlice[self.timeCol][idxs][0]]),
                dataSlice[self.timeCol][idxs][1:][condition][:-1],
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
        changeCol="filter",
        timeCol="observationStartMJD",
        metricName=None,
        timespan=20,
        **kwargs,
    ):
        """
        col = column tracking changes in
        timeCol = column keeping the time of each visit
        timespan = the timespan to count the number of changes within (in minutes)
        """
        if metricName is None:
            metricName = "Max number of %s changes within %.1f minutes" % (
                changeCol,
                timespan,
            )
        self.changeCol = changeCol
        self.timeCol = timeCol
        self.timespan = timespan / 24.0 / 60.0  # Convert timespan from minutes to days.
        super(MaxStateChangesWithinMetric, self).__init__(
            col=[changeCol, timeCol], metricName=metricName, units="#", **kwargs
        )

    def run(self, dataSlice, slicePoint=None):
        # This operates slightly differently from the metrics above; those calculate only successive times
        # between changes, but here we must calculate the actual times of each change.
        # Check if there was only one observation (and return 0 if so).
        if dataSlice[self.changeCol].size == 1:
            return 0
        # Sort on time, to be sure we've got filter (or other col) changes in the right order.
        idxs = np.argsort(dataSlice[self.timeCol])
        changes = (
            dataSlice[self.changeCol][idxs][:-1] != dataSlice[self.changeCol][idxs][1:]
        )
        condition = np.where(changes == True)[0]
        changetimes = dataSlice[self.timeCol][idxs][1:][condition]
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
        m5Col="fiveSigmaDepth",
        filterCol="filter",
        metricName="tEff",
        fiducialDepth=None,
        teffBase=30.0,
        normed=False,
        **kwargs,
    ):
        self.m5Col = m5Col
        self.filterCol = filterCol
        if fiducialDepth is None:
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
            if isinstance(fiducialDepth, dict):
                self.depth = fiducialDepth
            else:
                raise ValueError("fiducialDepth should be None or dictionary")
        self.teffBase = teffBase
        self.normed = normed
        if self.normed:
            units = ""
        else:
            units = "seconds"
        super(TeffMetric, self).__init__(
            col=[m5Col, filterCol], metricName=metricName, units=units, **kwargs
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

    def run(self, dataSlice, slicePoint=None):
        filters = np.unique(dataSlice[self.filterCol])
        teff = 0.0
        for f in filters:
            match = np.where(dataSlice[self.filterCol] == f)[0]
            teff += (
                10.0 ** (0.8 * (dataSlice[self.m5Col][match] - self.depth[f]))
            ).sum()
        teff *= self.teffBase
        if self.normed:
            # Normalize by the t_eff if each observation was at the fiducial depth.
            teff = teff / (self.teffBase * dataSlice[self.m5Col].size)
        return teff


class OpenShutterFractionMetric(BaseMetric):
    """
    Compute the fraction of time the shutter is open compared to the total time spent observing.
    """

    def __init__(
        self,
        metricName="OpenShutterFraction",
        slewTimeCol="slewTime",
        expTimeCol="visitExposureTime",
        visitTimeCol="visitTime",
        **kwargs,
    ):
        self.expTimeCol = expTimeCol
        self.visitTimeCol = visitTimeCol
        self.slewTimeCol = slewTimeCol
        super(OpenShutterFractionMetric, self).__init__(
            col=[self.expTimeCol, self.visitTimeCol, self.slewTimeCol],
            metricName=metricName,
            units="OpenShutter/TotalTime",
            **kwargs,
        )
        self.comment = (
            "Open shutter time (%s total) divided by total visit time "
            "(%s) + slewtime (%s)."
            % (self.expTimeCol, self.visitTimeCol, self.slewTimeCol)
        )

    def run(self, dataSlice, slicePoint=None):
        result = np.sum(dataSlice[self.expTimeCol]) / np.sum(
            dataSlice[self.slewTimeCol] + dataSlice[self.visitTimeCol]
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
        metricName="BruteOSFMetric",
        expTimeCol="visitExposureTime",
        mjdCol="observationStartMJD",
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
        expTimeCol : str ('expTime')
            The name of the exposure time column. Assumed to be in seconds.
        mjdCol : str ('observationStartMJD')
            The name of the start of the exposures. Assumed to be in units of days.
        """
        self.expTimeCol = expTimeCol
        self.maxgap = maxgap / 60.0 / 24.0  # convert from min to days
        self.mjdCol = mjdCol
        self.fudge = fudge
        super(BruteOSFMetric, self).__init__(
            col=[self.expTimeCol, mjdCol],
            metricName=metricName,
            units="OpenShutter/TotalTime",
            **kwargs,
        )

    def run(self, dataSlice, slicePoint=None):
        times = np.sort(dataSlice[self.mjdCol])
        diff = np.diff(times)
        good = np.where(diff < self.maxgap)
        openTime = np.sum(diff[good]) * 24.0 * 3600.0
        result = np.sum(dataSlice[self.expTimeCol] + self.fudge) / float(openTime)
        return result
