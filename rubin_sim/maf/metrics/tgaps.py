import numpy as np
from .baseMetric import BaseMetric

__all__ = ['TgapsMetric', 'NightgapsMetric', 'NVisitsPerNightMetric', 'MaxGapMetric']


class TgapsMetric(BaseMetric):
    """Histogram the times of the gaps between observations.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If allGaps is set to true, all gaps are
    computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
    return a histogram of [10,10,10] while allGaps returns a histogram of [10,10,10,20,20,30])

    Parameters
    ----------
    timesCol : str, opt
        The column name for the exposure times.  Values assumed to be in days.
        Default observationStartMJD.
    allGaps : bool, opt
        Histogram the gaps between all observations (True) or just successive observations (False)?
        Default is False. If all gaps are used, this metric can become significantly slower.
    bins : np.ndarray, opt
        The bins to use for the histogram of time gaps (in days, or same units as timesCol).
        Default values are bins from 0 to 2 hours, in 5 minute intervals.

    Returns a histogram at each slice point; these histograms can be combined and plotted using the
    'SummaryHistogram plotter'.
     """

    def __init__(self, timesCol='observationStartMJD', allGaps=False, bins=np.arange(0, 120.0, 5.0)/60./24.,
                 units='days', **kwargs):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.timesCol = timesCol
        super(TgapsMetric, self).__init__(col=[self.timesCol], metricDtype='object', units=units, **kwargs)
        self.allGaps = allGaps

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        times = np.sort(dataSlice[self.timesCol])
        if self.allGaps:
            allDiffs = []
            for i in np.arange(1,times.size,1):
                allDiffs.append((times-np.roll(times,i))[i:])
            dts = np.concatenate(allDiffs)
        else:
            dts = np.diff(times)
        result, bins = np.histogram(dts, self.bins)
        return result


class NightgapsMetric(BaseMetric):
    """Histogram the number of nights between observations.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If allGaps is set to true, all gaps are
    computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
    histogram [10,10,10] while allGaps histograms [10,10,10,20,20,30])

    Parameters
    ----------
    nightCol : str, opt
        The column name for the night of each observation.
        Default 'night'.
    allGaps : bool, opt
        Histogram the gaps between all observations (True) or just successive observations (False)?
        Default is False. If all gaps are used, this metric can become significantly slower.
    bins : np.ndarray, opt
        The bins to use for the histogram of time gaps (in days, or same units as timesCol).
        Default values are bins from 0 to 10 days, in 1 day intervals.

    Returns a histogram at each slice point; these histograms can be combined and plotted using the
    'SummaryHistogram plotter'.
     """

    def __init__(self, nightCol='night', allGaps=False, bins=np.arange(0, 10, 1),
                 units='nights', **kwargs):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.nightCol = nightCol
        super(NightgapsMetric, self).__init__(col=[self.nightCol], metricDtype='object',
                                              units=units, **kwargs)
        self.allGaps = allGaps

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        nights = np.sort(np.unique(dataSlice[self.nightCol]))
        if self.allGaps:
            allDiffs = []
            for i in np.arange(1, nights.size,1):
                allDiffs.append((nights-np.roll(nights,i))[i:])
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
    nightCol : str, opt
        The column name for the night of each observation.
        Default 'night'.
    bins : np.ndarray, opt
        The bins to use for the histogram of time gaps (in days, or same units as timesCol).
        Default values are bins from 0 to 5 visits, in steps of 1.

    Returns a histogram at each slice point; these histograms can be combined and plotted using the
    'SummaryHistogram plotter'.
     """

    def __init__(self, nightCol='night', bins=np.arange(0, 10, 1), units='#', **kwargs):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.nightCol = nightCol
        super(NVisitsPerNightMetric, self).__init__(col=[self.nightCol], metricDtype='object',
                                                    units=units, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        n, counts = np.unique(dataSlice[self.nightCol], return_counts=True)
        result, bins = np.histogram(counts, self.bins)
        return result


class MaxGapMetric(BaseMetric):
    """Find the maximum gap in observations. Useful for making sure there is an image within the last year that would
    make a good template image.
    """

    def __init__(self, mjdCol='observationStartMJD', **kwargs):
        self.mjdCol = mjdCol
        units = 'Days'
        super(MaxGapMetric, self).__init__(col=[self.mjdCol], units=units, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        gaps = np.diff(np.sort(dataSlice[self.mjdCol]))
        if np.size(gaps) > 0:
            result = np.max(gaps)
        else:
            result = self.badval
        return result

