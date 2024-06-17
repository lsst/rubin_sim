__all__ = (
    "GapsMetric",
    "TgapsMetric",
    "TgapsPercentMetric",
    "NightgapsMetric",
    "NVisitsPerNightMetric",
    "CoaddM5PerNightMetric",
    "MaxGapMetric",
    "NightTimespanMetric",
)

import numpy as np
from scipy.stats import binned_statistic

from ..utils import coadd_m5
from .base_metric import BaseMetric


class GapsMetric(BaseMetric):
    """Compute the number of times a gaps of a specified timescale
    (+/- 1/2 timescale) is sampled.

    Parameters
    ----------
    times_col : `str`, optional
        The column name for the exposure times.  Values assumed to be in days.
        Default is 'observationStartMJD'.
    time_scale : `float`, optional
        Time scale to see how well it is sampled (hours).
        For example, the default of 2 hours means
        observations spaced anywhere between 1 and 3 hours apart
        will count as a sample. Default 2 hours.
    units : `str`, optional
        Default units to use for plots, etc. Default is "N" (for number).
    """

    def __init__(
        self,
        times_col="observationStartMJD",
        time_scale=2.0,
        units="N",
        **kwargs,
    ):
        self.times_col = times_col
        # Convert to days and divide by two so we bin at the Nyquist frequency
        self.bin_size = (time_scale / 24.0) / 2.0
        super().__init__(col=[self.times_col], metric_dtype="float", units=units, **kwargs)

    def run(self, data_slice, slice_point=None):
        if data_slice.size < 2:
            return self.badval
        times = np.sort(data_slice[self.times_col])
        bins = np.arange(times.min() - self.bin_size, times.max() + self.bin_size, self.bin_size)
        vals, _be = np.histogram(times, bins)
        # Bins are at 1/2 desired timescale, so compare bin 0 to bin 2 to see
        # if the timescale has been sampled.
        mult = vals[2:] * vals[0:-2]
        result = np.size(np.where(mult > 0)[0])

        return result


class TgapsMetric(BaseMetric):
    """Histogram the times of the gaps between observations.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If all_gaps is set to true,
    all gaps are computed (i.e., if there are observations at 10, 20, 30
    and 40 the default will return a histogram of [10,10,10] while
    all_gaps returns a histogram of [10,10,10,20,20,30])

    Parameters
    ----------
    times_col : `str`, optional
        The column name for the exposure times.  Values assumed to be in days.
        Default observationStartMJD.
    all_gaps : `bool`, optional
        Histogram the gaps between all observations (True) or just successive
        observations (False)?
        Default is False.
        If all gaps are used, this metric can become significantly slower.
    bins : `np.ndarray`, optional
        The bins to use for the histogram of time gaps
        (in days, or same units as times_col).
        Default values are bins from 0 to 2 hours, in 5 minute intervals.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the tgaps at each slice point;
        these histograms can be combined and plotted using the
        'SummaryHistogram plotter'.
    """

    def __init__(
        self,
        times_col="observationStartMJD",
        all_gaps=False,
        bins=np.arange(0, 120.0, 5.0) / 60.0 / 24.0,
        units="days",
        **kwargs,
    ):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.times_col = times_col
        super().__init__(col=[self.times_col], metric_dtype="object", units=units, **kwargs)
        self.all_gaps = all_gaps

    def run(self, data_slice, slice_point=None):
        if data_slice.size < 2:
            return self.badval
        times = np.sort(data_slice[self.times_col])
        if self.all_gaps:
            all_diffs = []
            for i in np.arange(1, times.size, 1):
                all_diffs.append((times - np.roll(times, i))[i:])
            dts = np.concatenate(all_diffs)
        else:
            dts = np.diff(times)
        result, bins = np.histogram(dts, self.bins)
        return result


class TgapsPercentMetric(BaseMetric):
    """Compute the fraction of the time gaps between observations that occur
    in a given time range.

    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If all_gaps is set to true,
    all gaps are computed (i.e., if there are observations at 10, 20, 30 and
    40 the default will compute the percent of gaps between
    specified endpoints.

    This is different from the TgapsMetric in that this only looks at
    what percent of intervals fall into the specified range, rather than
    histogramming the entire set of tgaps.

    This metric has drawbacks in that the result is tied to the overall
    number of tgaps (i.e. a result of 100 could still be worse than
    a different simulation with a result of 50, in terms of how often a
    particular Tgap is sampled).

    Parameters
    ----------
    times_col : `str`, opt
        The column name for the exposure times.  Values assumed to be in days.
        Default observationStartMJD.
    all_gaps : `bool`, opt
        Histogram the gaps between all observations (True) or
        just successive observations (False)?
        Default is False. If all gaps are used, this metric
        can become significantly slower.
    min_time : `float`, opt
        Minimum time of gaps to include (days). Default 2/24 (2 hours).
    max_time : `float`, opt
        Max time of gaps to include (days). Default 14/24 (14 hours).

    Returns
    -------
    percent : `float`
        Returns a float percent of the CDF between cdfMinTime and cdfMaxTime -
        (# of tgaps within min_time/max_time / # of all tgaps).
    """

    def __init__(
        self,
        times_col="observationStartMJD",
        all_gaps=False,
        min_time=2.0 / 24,
        max_time=14.0 / 24,
        units="percent",
        **kwargs,
    ):
        self.times_col = times_col
        assert min_time <= max_time
        self.min_time = min_time
        self.max_time = max_time
        super().__init__(col=[self.times_col], metric_dtype="float", units=units, **kwargs)
        self.all_gaps = all_gaps

    def run(self, data_slice, slice_point=None):
        if data_slice.size < 2:
            return self.badval
        times = np.sort(data_slice[self.times_col])
        if self.all_gaps:
            all_diffs = []
            for i in np.arange(1, times.size, 1):
                all_diffs.append((times - np.roll(times, i))[i:])
            dts = np.concatenate(all_diffs)
        else:
            dts = np.diff(times)
        n_in_window = np.sum((dts >= self.min_time) & (dts <= self.max_time))
        return n_in_window / len(dts) * 100.0


class NightgapsMetric(BaseMetric):
    """Histogram the number of nights between observations.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If all_gaps is set to true,
    all gaps are computed (i.e., if there are observations at 10, 20, 30 and
    40 the default will
    histogram [10,10,10] while all_gaps histograms [10,10,10,20,20,30])

    Parameters
    ----------
    night_col : `str`, optional
        The column name for the night of each observation.
        Default 'night'.
    all_gaps : `bool`, optional
        Histogram the gaps between all observations (True) or just successive
        observations (False)?
        Default is False. If all gaps are used, this metric can become
        significantly slower.
    bins : `np.ndarray`, optional
        The bins to use for the histogram of time gaps (in days, or same
        units as timesCol).
        Default values are bins from 0 to 10 days, in 1 day intervals.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the deltaT between nights at each slice point;
        these histograms can be combined and plotted using the
        'SummaryHistogram plotter'.
    """

    def __init__(
        self,
        night_col="night",
        all_gaps=False,
        bins=np.arange(0, 10, 1),
        units="nights",
        **kwargs,
    ):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.night_col = night_col
        super().__init__(col=[self.night_col], metric_dtype="object", units=units, **kwargs)
        self.all_gaps = all_gaps

    def run(self, data_slice, slice_point=None):
        if data_slice.size < 2:
            return self.badval
        nights = np.sort(np.unique(data_slice[self.night_col]))
        if self.all_gaps:
            all_diffs = []
            for i in np.arange(1, nights.size, 1):
                all_diffs.append((nights - np.roll(nights, i))[i:])
            dnights = np.concatenate(all_diffs)
        else:
            dnights = np.diff(nights)
        result, bins = np.histogram(dnights, self.bins)
        return result


class NVisitsPerNightMetric(BaseMetric):
    """Histogram the number of visits in each night.

    Splits the visits by night, then histograms how many visits occur
    in each night.

    Parameters
    ----------
    night_col : `str`, optional
        The column name for the night of each observation.
        Default 'night'.
    bins : `np.ndarray`, optional
        The bins to use for the histogram of time gaps (in days, or same
        units as timesCol).
        Default values are bins from 0 to 5 visits, in steps of 1.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the number of visits per night at each
        slice point; these histograms can be combined and plotted using the
        'SummaryHistogram plotter'.
    """

    def __init__(self, night_col="night", bins=np.arange(0, 10, 1), units="#", **kwargs):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.night_col = night_col
        super().__init__(col=[self.night_col], metric_dtype="object", units=units, **kwargs)

    def run(self, data_slice, slice_point=None):
        n, counts = np.unique(data_slice[self.night_col], return_counts=True)
        result, bins = np.histogram(counts, self.bins)
        return result


class CoaddM5PerNightMetric(BaseMetric):
    """Histogram the coadded depth of visits in each night.

    Splits the visits by night, then histograms the coadded depth.

    Parameters
    ----------
    night_col : `str`, optional
        The column name for the night of each observation.
        Default 'night'.
    m5_col : `str`, optional
        The column name for the five sigma depth of each observation.
    bins : `np.ndarray`, (N,) optional
        The bins to use for the histogram of magnitude values.

    Returns
    -------
    histogram : `np.ndarray`
        Returns a histogram of the coadded depth per night at each slice point;
        these histograms can be combined and plotted using the
        'SummaryHistogram plotter'.
    """

    def __init__(
        self, night_col="night", m5_col="fiveSigmaDepth", bins=np.arange(20, 25, 0.05), units="mag", **kwargs
    ):
        # Pass the same bins to the plotter.
        self.night_col = night_col
        self.m5_col = m5_col
        self.bins = bins
        super().__init__(col=[self.night_col, self.m5_col], metric_dtype="object", units=units, **kwargs)

    def run(self, data_slice, slice_point=None):
        m5_per_night, be, bn = binned_statistic(
            data_slice[self.night_col],
            data_slice[self.m5_col],
            statistic=coadd_m5,
            bins=np.arange(0, 3653, 1),
        )
        # Drop the nights with no observations (-inf)
        m5_per_night = m5_per_night[np.where(m5_per_night > 0)]
        result, bins = np.histogram(m5_per_night, self.bins)
        return result


class MaxGapMetric(BaseMetric):
    """Find the maximum gap (in days) in between successive observations.

    Useful for making sure there is an image within the last year that
    would make a good template image.

    Parameters
    ----------
    mjd_col : `str`, opt
        The column name of the night of each observation.

    Returns
    -------
    maxGap : `float`
        The maximum gap (in days) between visits.
    """

    def __init__(self, mjd_col="observationStartMJD", **kwargs):
        self.mjd_col = mjd_col
        units = "Days"
        super(MaxGapMetric, self).__init__(col=[self.mjd_col], units=units, **kwargs)

    def run(self, data_slice, slice_point=None):
        gaps = np.diff(np.sort(data_slice[self.mjd_col]))
        if np.size(gaps) > 0:
            result = np.max(gaps)
        else:
            result = self.badval
        return result


class NightTimespanMetric(BaseMetric):
    """Calculate the maximum time span covered in each night,
    report the `percentile` value of all timespans.

    Parameters
    ----------
    percentile : `float`, opt
        Percentile value to report. Default 75th percentile.
    night_col : `str`, opt
        Name of the night column. Default 'night'.
    mjd_col : `str`, opt
        Name of the MJD visit column. Default 'observationStartMJD'.
    """

    def __init__(self, percentile=75, night_col="night", mjd_col="observationStartMJD", **kwargs):
        self.percentile = percentile
        self.night_col = night_col
        self.mjd_col = mjd_col
        if "metric_name" in kwargs:
            metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        else:
            metric_name = f"{percentile}th Percentile Intranight Timespan"
        super().__init__(
            col=[self.night_col, self.mjd_col],
            units="minutes",
            metric_name=metric_name,
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        data = np.sort(data_slice, order=self.mjd_col)
        unights, counts = np.unique(data[self.night_col], return_counts=True)
        unights = unights[np.where(counts > 1)]
        if len(unights) == 0:
            result = self.badval
        else:
            nstart = np.searchsorted(data[self.night_col], unights, side="left")
            nend = np.searchsorted(data[self.night_col], unights, side="right") - 1
            tspans = (data[self.mjd_col][nend] - data[self.mjd_col][nstart]) * 24.0 * 60.0
            result = np.percentile(tspans, self.percentile)
        return result
