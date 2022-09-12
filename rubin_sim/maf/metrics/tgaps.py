import numpy as np
from .base_metric import BaseMetric

__all__ = [
    "TgapsMetric",
    "TgapsPercentMetric",
    "NightgapsMetric",
    "NVisitsPerNightMetric",
    "MaxGapMetric",
    "NightTimespanMetric",
]


class TgapsMetric(BaseMetric):
    """Histogram the times of the gaps between observations.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If allGaps is set to true, all gaps are
    computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
    return a histogram of [10,10,10] while allGaps returns a histogram of [10,10,10,20,20,30])

    Parameters
    ----------
    timesCol : `str`, optional
        The column name for the exposure times.  Values assumed to be in days.
        Default observationStartMJD.
    allGaps : `bool`, optional
        Histogram the gaps between all observations (True) or just successive observations (False)?
        Default is False. If all gaps are used, this metric can become significantly slower.
    bins : `np.ndarray`, optional
        The bins to use for the histogram of time gaps (in days, or same units as timesCol).
        Default values are bins from 0 to 2 hours, in 5 minute intervals.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the tgaps at each slice point;
        these histograms can be combined and plotted using the 'SummaryHistogram plotter'.
    """

    def __init__(
        self,
        timesCol="observationStartMJD",
        allGaps=False,
        bins=np.arange(0, 120.0, 5.0) / 60.0 / 24.0,
        units="days",
        **kwargs,
    ):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.timesCol = timesCol
        super().__init__(
            col=[self.timesCol], metricDtype="object", units=units, **kwargs
        )
        self.allGaps = allGaps

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        times = np.sort(dataSlice[self.timesCol])
        if self.allGaps:
            allDiffs = []
            for i in np.arange(1, times.size, 1):
                allDiffs.append((times - np.roll(times, i))[i:])
            dts = np.concatenate(allDiffs)
        else:
            dts = np.diff(times)
        result, bins = np.histogram(dts, self.bins)
        return result


class TgapsPercentMetric(BaseMetric):
    """Compute the fraction of the time gaps between observations that occur in a given time range.

    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If allGaps is set to true, all gaps are
    computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
    Compute the percent of gaps between specified endpoints.

    This is different from the TgapsMetric in that this only looks at what percent of intervals fall
    into the specified range, rather than histogramming the entire set of tgaps.

    Parameters
    ----------
    timesCol : `str`, opt
        The column name for the exposure times.  Values assumed to be in days.
        Default observationStartMJD.
    allGaps : `bool`, opt
        Histogram the gaps between all observations (True) or just successive observations (False)?
        Default is False. If all gaps are used, this metric can become significantly slower.
    minTime : `float`, opt
        Minimum time of gaps to include (days). Default 2/24 (2 hours).
    maxTime : `float`, opt
        Max time of gaps to include (days). Default 14/24 (14 hours).

    Returns
    -------
    percent : `float`
        Returns a float percent of the CDF between cdfMinTime and cdfMaxTime -
        (# of tgaps within minTime/maxTime / # of all tgaps).
    """

    def __init__(
        self,
        timesCol="observationStartMJD",
        allGaps=False,
        minTime=2.0 / 24,
        maxTime=14.0 / 24,
        units="percent",
        **kwargs,
    ):
        self.timesCol = timesCol
        assert minTime <= maxTime
        self.minTime = minTime
        self.maxTime = maxTime
        super().__init__(
            col=[self.timesCol], metricDtype="float", units=units, **kwargs
        )
        self.allGaps = allGaps

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        times = np.sort(dataSlice[self.timesCol])
        if self.allGaps:
            allDiffs = []
            for i in np.arange(1, times.size, 1):
                allDiffs.append((times - np.roll(times, i))[i:])
            dts = np.concatenate(allDiffs)
        else:
            dts = np.diff(times)
        nInWindow = np.sum((dts >= self.minTime) & (dts <= self.maxTime))
        return nInWindow / len(dts) * 100.0


class NightgapsMetric(BaseMetric):
    """Histogram the number of nights between observations.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If allGaps is set to true, all gaps are
    computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
    histogram [10,10,10] while allGaps histograms [10,10,10,20,20,30])

    Parameters
    ----------
    nightCol : `str`, optional
        The column name for the night of each observation.
        Default 'night'.
    allGaps : `bool`, optional
        Histogram the gaps between all observations (True) or just successive observations (False)?
        Default is False. If all gaps are used, this metric can become significantly slower.
    bins : `np.ndarray`, optional
        The bins to use for the histogram of time gaps (in days, or same units as timesCol).
        Default values are bins from 0 to 10 days, in 1 day intervals.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the deltaT between nights at each slice point;
        these histograms can be combined and plotted using the 'SummaryHistogram plotter'.
    """

    def __init__(
        self,
        nightCol="night",
        allGaps=False,
        bins=np.arange(0, 10, 1),
        units="nights",
        **kwargs,
    ):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.nightCol = nightCol
        super().__init__(
            col=[self.nightCol], metricDtype="object", units=units, **kwargs
        )
        self.allGaps = allGaps

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        nights = np.sort(np.unique(dataSlice[self.nightCol]))
        if self.allGaps:
            allDiffs = []
            for i in np.arange(1, nights.size, 1):
                allDiffs.append((nights - np.roll(nights, i))[i:])
            dnights = np.concatenate(allDiffs)
        else:
            dnights = np.diff(nights)
        result, bins = np.histogram(dnights, self.bins)
        return result


class NVisitsPerNightMetric(BaseMetric):
    """Histogram the number of visits in each night.

    Splits the visits by night, then histograms how many visits occur in each night.

    Parameters
    ----------
    nightCol : `str`, optional
        The column name for the night of each observation.
        Default 'night'.
    bins : `np.ndarray`, optional
        The bins to use for the histogram of time gaps (in days, or same units as timesCol).
        Default values are bins from 0 to 5 visits, in steps of 1.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the number of visits per night at each slice point;
        these histograms can be combined and plotted using the 'SummaryHistogram plotter'.
    """

    def __init__(self, nightCol="night", bins=np.arange(0, 10, 1), units="#", **kwargs):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.nightCol = nightCol
        super().__init__(
            col=[self.nightCol], metricDtype="object", units=units, **kwargs
        )

    def run(self, dataSlice, slicePoint=None):
        n, counts = np.unique(dataSlice[self.nightCol], return_counts=True)
        result, bins = np.histogram(counts, self.bins)
        return result


class MaxGapMetric(BaseMetric):
    """Find the maximum gap (in days) in between successive observations.

    Useful for making sure there is an image within the last year that would make a good template image.

    Parameters
    ----------
    mjdCol : `str`, opt
        The column name of the night of each observation.

    Returns
    -------
    maxGap : `float`
        The maximum gap (in days) between visits.
    """

    def __init__(self, mjdCol="observationStartMJD", **kwargs):
        self.mjdCol = mjdCol
        units = "Days"
        super(MaxGapMetric, self).__init__(col=[self.mjdCol], units=units, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        gaps = np.diff(np.sort(dataSlice[self.mjdCol]))
        if np.size(gaps) > 0:
            result = np.max(gaps)
        else:
            result = self.badval
        return result


class NightTimespanMetric(BaseMetric):
    """Calculate the maximum time span covered in each night, report the `percentile` value of all timespans.

    Parameters
    ----------
    percentile : `float`, opt
        Percentile value to report. Default 75th percentile.
    nightCol : `str`, opt
        Name of the night column. Default 'night'.
    mjdCol : `str`, opt
        Name of the MJD visit column. Default 'observationStartMJD'.
    """

    def __init__(
        self, percentile=75, nightCol="night", mjdCol="observationStartMJD", **kwargs
    ):
        self.percentile = percentile
        self.nightCol = nightCol
        self.mjdCol = mjdCol
        if "metricName" in kwargs:
            metricName = kwargs["metricName"]
            del kwargs["metricName"]
        else:
            metricName = f"{percentile}th Percentile Intranight Timespan"
        super().__init__(
            col=[self.nightCol, self.mjdCol],
            units="minutes",
            metricName=metricName,
            **kwargs,
        )

    def run(self, dataSlice, slicePoint=None):
        data = np.sort(dataSlice, order=self.mjdCol)
        unights, counts = np.unique(data[self.nightCol], return_counts=True)
        unights = unights[np.where(counts > 1)]
        if len(unights) == 0:
            result = self.badval
        else:
            nstart = np.searchsorted(data[self.nightCol], unights, side="left")
            nend = np.searchsorted(data[self.nightCol], unights, side="right") - 1
            tspans = (data[self.mjdCol][nend] - data[self.mjdCol][nstart]) * 24.0 * 60.0
            result = np.percentile(tspans, self.percentile)
        return result
