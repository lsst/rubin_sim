from builtins import zip
import numpy as np
from .baseMetric import BaseMetric

__all__ = [
    "NChangesMetric",
    "MinTimeBetweenStatesMetric",
    "NStateChangesFasterThanMetric",
    "MaxStateChangesWithinMetric",
    "TeffMetric",
    "OpenShutterFractionMetric",
    "CompletenessMetric",
    "FilterColorsMetric",
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


class CompletenessMetric(BaseMetric):
    """Compute the completeness and joint completeness """

    def __init__(
        self,
        filterColName="filter",
        metricName="Completeness",
        u=0,
        g=0,
        r=0,
        i=0,
        z=0,
        y=0,
        **kwargs,
    ):
        """
        Compute the completeness for the each of the given filters and the
        joint completeness across all filters.

        Completeness calculated in any filter with a requested 'nvisits' value greater than 0, range is 0-1.
        """
        self.filterCol = filterColName
        super(CompletenessMetric, self).__init__(
            col=self.filterCol, metricName=metricName, **kwargs
        )
        self.nvisitsRequested = np.array([u, g, r, i, z, y])
        self.filters = np.array(["u", "g", "r", "i", "z", "y"])
        # Remove filters from consideration where number of visits requested is zero.
        good = np.where(self.nvisitsRequested > 0)
        self.nvisitsRequested = self.nvisitsRequested[good]
        self.filters = self.filters[good]
        # Raise exception if number of visits wasn't changed from the default, for at least one filter.
        if len(self.filters) == 0:
            raise ValueError(
                "Please set the requested number of visits for at least one filter."
            )
        # Set reduce order, for display purposes.
        for i, f in enumerate(["u", "g", "r", "i", "z", "y", "Joint"]):
            self.reduceOrder[f] = i
        self.comment = "Completeness fraction for each filter (and joint across all filters), calculated"
        self.comment += " as the number of visits compared to a benchmark value of :"
        for i, f in enumerate(self.filters):
            self.comment += " %s: %d" % (f, self.nvisitsRequested[i])
        self.comment += "."

    def run(self, dataSlice, slicePoint=None):
        """
        Compute the completeness for each filter, and then the minimum (joint) completeness for each slice.
        """
        allCompleteness = []
        for f, nVis in zip(self.filters, self.nvisitsRequested):
            filterVisits = np.size(np.where(dataSlice[self.filterCol] == f)[0])
            allCompleteness.append(filterVisits / float(nVis))
        allCompleteness.append(np.min(np.array(allCompleteness)))
        return np.array(allCompleteness)

    def reduceu(self, completeness):
        if "u" in self.filters:
            return completeness[np.where(self.filters == "u")[0]]
        else:
            return 1

    def reduceg(self, completeness):
        if "g" in self.filters:
            return completeness[np.where(self.filters == "g")[0]]
        else:
            return 1

    def reducer(self, completeness):
        if "r" in self.filters:
            return completeness[np.where(self.filters == "r")[0]]
        else:
            return 1

    def reducei(self, completeness):
        if "i" in self.filters:
            return completeness[np.where(self.filters == "i")[0]]
        else:
            return 1

    def reducez(self, completeness):
        if "z" in self.filters:
            return completeness[np.where(self.filters == "z")[0]]
        else:
            return 1

    def reducey(self, completeness):
        if "y" in self.filters:
            return completeness[np.where(self.filters == "y")[0]]
        else:
            return 1

    def reduceJoint(self, completeness):
        """
        The joint completeness is just the minimum completeness for a point/field.
        """
        return completeness[-1]


class FilterColorsMetric(BaseMetric):
    """
    Generate a map of [filter number], [alpha] that simply shows the *most recent* visit filter
    (in filter number) and an alpha map that scales the transparency according to how long ago
    the visit occurred.

    Parameters
    ----------
    tNow : `float`
        Current time (implying the metric must be set up for each 'current time' the metric should be
        run for -- because the results will be different for each time slice)
    nightNow : `int`
        Identify visits within this night, vs. earlier nights (for alpha map)
    filterCol : `str`, opt
        Name of the filter column. Default 'filter'.
        This is translated into an integer between 0-6 for the 'color', which then is used in combination
        with the colormap for the plot to generate colors in the visit plot.
    timeCol : `str`, opt
        Name of the time column. Default 'observationStartMJD'.
    tStep : `float`, opt
        Time gap to consider an observation occuring 'now'. This is set to 40s by default.
    alpha_night : `float`, opt
        Alpha value for visits occuring in the current night. Default 0.7.
    alpha_prev : `float`, opt
        Alpha value for visits occuring in the previous night. Default 0.3.
    """

    def __init__(
        self,
        tNow,
        nightNow,
        filterCol="filter",
        timeCol="observationStartMJD",
        nightCol="night",
        tStep=40.0 / 60.0 / 60.0 / 24.0,
        alpha_night=0.7,
        alpha_prev=0.3,
        metricName="FilterColors",
        **kwargs,
    ):
        self.filterCol = filterCol
        self.timeCol = timeCol
        self.nightCol = nightCol
        self.tNow = tNow
        self.nightNow = nightNow
        self.tStep = tStep
        self.alpha_night = alpha_night
        self.alpha_prev = alpha_prev
        super().__init__(
            col=[self.filterCol, self.timeCol, self.nightCol],
            metricName=metricName,
            **kwargs,
        )
        self.metricDtype = "object"
        self.comment = "Metric to generate color/alpha for the FBS healpix movie"
        self.filter_color_map = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def run(self, dataSlice, slicePoint=None):
        # Only count visits which occured prior to "now" --
        # this is necessary because often this metric will be run with a data set which
        # includes more visits than just 'now' and prior.
        visits = dataSlice[np.where(dataSlice[self.timeCol] < self.tNow)]
        if len(visits) == 0:
            return self.badval
        deltaNightMin = (self.nightNow - visits[self.nightCol]).min()
        # Figure out if there are any visits we should consider as "now" (or now-ish)
        deltaT = self.tNow - dataSlice[self.timeCol]
        visitNow = np.where(deltaT <= self.tStep)[0]
        if len(visitNow) > 0:
            # We have exact matches to this timestep, so use their colors directly and set alpha to >1.
            fNow = dataSlice[visitNow[0]][self.filterCol]
            colorVal = self.filter_color_map[fNow]
            alpha = 1.0
        else:
            # This part of the sky has only older exposures.
            # Are they this night or previous night?
            if deltaNightMin == 0:
                alpha = self.alpha_night
            else:
                alpha = self.alpha_prev
            # And then decide what color to use - the most recent visit
            nearest = np.where(visits[self.timeCol] == visits[self.timeCol].max())[0]
            fRecent = visits[nearest][self.filterCol]
            colorVal = self.filter_color_map[fRecent]
        return (colorVal, alpha)

    def reduceColor(self, metricValues):
        return metricValues[0]

    def reduceAlpha(self, metricValues):
        return metricValues[1]


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
