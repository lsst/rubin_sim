__all__ = (
    "TemplateExistsMetric",
    "UniformityMetric",
    "GeneralUniformityMetric",
    "RapidRevisitUniformityMetric",
    "RapidRevisitMetric",
    "NRevisitsMetric",
    "IntraNightGapsMetric",
    "InterNightGapsMetric",
    "VisitGapMetric",
)

import numpy as np

from .base_metric import BaseMetric


class FSMetric(BaseMetric):
    """Calculate the fS value (Nvisit-weighted delta(M5-M5srd))."""

    def __init__(self, filter_col="filter", metric_name="fS", **kwargs):
        self.filter_col = filter_col
        cols = [self.filter_col]
        super().__init__(cols=cols, metric_name=metric_name, units="fS", **kwargs)

    def run(self, data_slice, slice_point=None):
        # We could import this from the m5_flat_sed values,
        # but it makes sense to calculate the m5
        # directly from the throughputs. This is easy enough to do and
        # will allow variation of
        # the throughput curves and readnoise and visit length, etc.
        pass


class TemplateExistsMetric(BaseMetric):
    """Calculate the fraction of images with a previous template
    image of desired quality."""

    def __init__(
        self,
        seeing_col="seeingFwhmGeom",
        observation_start_mjd_col="observationStartMJD",
        metric_name="TemplateExistsMetric",
        **kwargs,
    ):
        cols = [seeing_col, observation_start_mjd_col]
        super(TemplateExistsMetric, self).__init__(
            col=cols, metric_name=metric_name, units="fraction", **kwargs
        )
        self.seeing_col = seeing_col
        self.observation_start_mjd_col = observation_start_mjd_col

    def run(self, data_slice, slice_point=None):
        # Check that data is sorted in observationStartMJD order
        data_slice.sort(order=self.observation_start_mjd_col)
        # Find the minimum seeing up to a given time
        seeing_mins = np.minimum.accumulate(data_slice[self.seeing_col])
        # Find the difference between the seeing and the minimum seeing
        # at the previous visit
        seeing_diff = data_slice[self.seeing_col] - np.roll(seeing_mins, 1)
        # First image never has a template; check how many others do
        good = np.where(seeing_diff[1:] >= 0.0)[0]
        frac = (good.size) / float(data_slice[self.seeing_col].size)
        return frac


class UniformityMetric(BaseMetric):
    """Calculate how uniformly the observations are spaced in time.

    This is based on how a KS-test works:
    look at the cumulative distribution of observation dates,
    and compare to a perfectly uniform cumulative distribution.
    Perfectly uniform observations = 0, perfectly non-uniform = 1.

    Parameters
    ----------
    mjd_col : `str`, optional
        The column containing time for each observation.
        Default "observationStartMJD".
    survey_length : `float`, optional
        The overall duration of the survey. Default 10.
    """

    def __init__(self, mjd_col="observationStartMJD", units="", survey_length=10.0, **kwargs):
        """survey_length = time span of survey (years)"""
        self.mjd_col = mjd_col
        super(UniformityMetric, self).__init__(col=self.mjd_col, units=units, **kwargs)
        self.survey_length = survey_length

    def run(self, data_slice, slice_point=None):
        # If only one observation, there is no uniformity
        if data_slice[self.mjd_col].size == 1:
            return 1
        # Scale dates to lie between 0 and 1,
        # where 0 is the first observation date and 1 is surveyLength
        dates = (data_slice[self.mjd_col] - data_slice[self.mjd_col].min()) / (self.survey_length * 365.25)
        dates.sort()  # Just to be sure
        n_cum = np.arange(1, dates.size + 1) / float(dates.size)
        d_max = np.max(np.abs(n_cum - dates - dates[1]))
        return d_max


class GeneralUniformityMetric(BaseMetric):
    """Calculate how uniformly any values are distributed.

    This is based on how a KS-test works:
    look at the cumulative distribution of data,
    and compare to a perfectly uniform cumulative distribution.
    Perfectly uniform observations = 0, perfectly non-uniform = 1.
    To be "perfectly uniform" here, the endpoints need to be included.

    Parameters
    ----------
    col : `str`, optional
        The column of data to use for the metric.
        The default is "observationStartMJD" as this is most
        typically used with time.
    min_value : `float`, optional
        The minimum value expected for the data.
        Default None will calculate use the minimum value in this dataslice
        (which may not cover the full range).
    max_value : `float`, optional
        The maximum value expected for the data.
        Default None will calculate use the maximum value in this dataslice
        (which may not cover the full range).
    """

    def __init__(self, col="observationStartMJD", units="", min_value=None, max_value=None, **kwargs):
        self.col = col
        super().__init__(col=self.col, units=units, **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def run(self, data_slice, slice_point=None):
        # If only one observation, there is no uniformity
        if data_slice[self.col].size == 1:
            return 1
        # Scale values to lie between 0 and 1,
        # where 0 is the min_value and 1 is max_value
        if self.min_value is None:
            min_value = data_slice[self.col].min()
        else:
            min_value = self.min_value
        if self.max_value is None:
            max_value = data_slice[self.col].max()
        else:
            max_value = self.max_value
        scaled_values = (data_slice[self.col] - min_value) / max_value
        scaled_values.sort()  # Just to be sure
        n_cum = np.arange(0, scaled_values.size) / float(scaled_values.size - 1)
        d_max = np.max(np.abs(n_cum - scaled_values))
        return d_max


class RapidRevisitUniformityMetric(BaseMetric):
    """Calculate uniformity of time between consecutive visits on
    short timescales (for RAV1).

    Uses the same 'uniformity' calculation as the UniformityMetric,
    based on the KS-test.
    A value of 0 is perfectly uniform; a value of 1 is purely non-uniform.

    Parameters
    ----------
    mjd_col : `str`, optional
        The column containing the 'time' value. Default observationStartMJD.
    min_nvisits : `int`, optional
        The minimum number of visits required within the
        time interval (d_tmin to d_tmax).
        Default 100.
    d_tmin : `float`, optional
        The minimum dTime to consider (in days). Default 40 seconds.
    d_tmax : `float`, optional
        The maximum dTime to consider (in days). Default 30 minutes.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        min_nvisits=100,
        d_tmin=40.0 / 60.0 / 60.0 / 24.0,
        d_tmax=30.0 / 60.0 / 24.0,
        metric_name="RapidRevisitUniformity",
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.min_nvisits = min_nvisits
        self.d_tmin = d_tmin
        self.d_tmax = d_tmax
        super().__init__(col=self.mjd_col, metric_name=metric_name, **kwargs)
        # Update min_nvisits, as 0 visits will crash algorithm
        # and 1 is nonuniform by definition.
        if self.min_nvisits <= 1:
            self.min_nvisits = 2

    def run(self, data_slice, slice_point=None):
        # Calculate consecutive visit time intervals
        dtimes = np.diff(np.sort(data_slice[self.mjd_col]))
        # Identify dtimes within interval from dTmin/dTmax.
        good = np.where((dtimes >= self.d_tmin) & (dtimes <= self.d_tmax))[0]
        # If there are not enough visits in this time range, return bad value.
        if good.size < self.min_nvisits:
            return self.badval
        # Throw out dtimes outside desired range, and sort, then scale to 0-1.
        dtimes = np.sort(dtimes[good])
        dtimes = (dtimes - dtimes.min()) / float(self.d_tmax - self.d_tmin)
        # Set up a uniform distribution between 0-1 (to match dtimes).
        uniform_dtimes = np.arange(1, dtimes.size + 1, 1) / float(dtimes.size)
        # Look at the differences between our times and the uniform times.
        dmax = np.max(np.abs(uniform_dtimes - dtimes - dtimes[1]))
        return dmax


class RapidRevisitMetric(BaseMetric):
    def __init__(
        self,
        mjd_col="observationStartMJD",
        metric_name="RapidRevisit",
        d_tmin=40.0 / 60.0 / 60.0 / 24.0,
        d_tpairs=20.0 / 60.0 / 24.0,
        d_tmax=30.0 / 60.0 / 24.0,
        min_n1=28,
        min_n2=82,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.d_tmin = d_tmin
        self.d_tpairs = d_tpairs
        self.d_tmax = d_tmax
        self.min_n1 = min_n1
        self.min_n2 = min_n2
        super().__init__(col=self.mjd_col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        dtimes = np.diff(np.sort(data_slice[self.mjd_col]))
        n1 = len(np.where((dtimes >= self.d_tmin) & (dtimes <= self.d_tpairs))[0])
        n2 = len(np.where((dtimes >= self.d_tmin) & (dtimes <= self.d_tmax))[0])
        if (n1 >= self.min_n1) and (n2 >= self.min_n2):
            val = 1
        else:
            val = 0
        return val


class NRevisitsMetric(BaseMetric):
    """Calculate the number of consecutive visits with
    time differences less than d_t.

    Parameters
    ----------
    d_t : `float`, optional
        The time interval to consider (in minutes). Default 30.
    normed : `bool`, optional
        Flag to indicate whether to return the total number of
        consecutive visits with time differences less than d_t (False),
        or the fraction of overall visits (True).
        Note that we would expect (if all visits occur in pairs within d_t)
        this fraction would be 0.5!
    """

    def __init__(self, mjd_col="observationStartMJD", d_t=30.0, normed=False, metric_name=None, **kwargs):
        units = ""
        if metric_name is None:
            if normed:
                metric_name = "Fraction of revisits faster than %.1f minutes" % (d_t)
            else:
                metric_name = "Number of revisits faster than %.1f minutes" % (d_t)
                units = "#"
        self.mjd_col = mjd_col
        self.d_t = d_t / 60.0 / 24.0  # convert to days
        self.normed = normed
        super(NRevisitsMetric, self).__init__(
            col=self.mjd_col, units=units, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        dtimes = np.diff(np.sort(data_slice[self.mjd_col]))
        n_fast_revisits = np.size(np.where(dtimes <= self.d_t)[0])
        if self.normed:
            n_fast_revisits = n_fast_revisits / float(np.size(data_slice[self.mjd_col]))
        return n_fast_revisits


class IntraNightGapsMetric(BaseMetric):
    """
    Calculate the (reduce_func) of the gap between consecutive
    observations within a night, in hours.

    Parameters
    ----------
    reduce_func : function, optional
        Function that can operate on array-like structures.
        Typically numpy function.
        Default np.median.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        night_col="night",
        reduce_func=np.median,
        metric_name="Median Intra-Night Gap",
        **kwargs,
    ):
        units = "hours"
        self.mjd_col = mjd_col
        self.night_col = night_col
        self.reduce_func = reduce_func
        super(IntraNightGapsMetric, self).__init__(
            col=[self.mjd_col, self.night_col], units=units, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.mjd_col)
        dt = np.diff(data_slice[self.mjd_col])
        dn = np.diff(data_slice[self.night_col])

        good = np.where(dn == 0)
        if np.size(good[0]) == 0:
            result = self.badval
        else:
            result = self.reduce_func(dt[good]) * 24
        return result


class InterNightGapsMetric(BaseMetric):
    """Calculate the (reduce_func) of the gap between consecutive
    observations in different nights, in days.

    Parameters
    ----------
    reduce_func : function, optional
        Function that can operate on array-like structures.
        Typically numpy function.
        Default np.median.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        night_col="night",
        reduce_func=np.median,
        metric_name="Median Inter-Night Gap",
        **kwargs,
    ):
        units = "days"
        self.mjd_col = mjd_col
        self.night_col = night_col
        self.reduce_func = reduce_func
        super().__init__(col=[self.mjd_col, self.night_col], units=units, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.mjd_col)
        unights = np.unique(data_slice[self.night_col])
        if np.size(unights) < 2:
            result = self.badval
        else:
            # Find the first and last observation of each night
            first_of_night = np.searchsorted(data_slice[self.night_col], unights)
            last_of_night = np.searchsorted(data_slice[self.night_col], unights, side="right") - 1
            diff = data_slice[self.mjd_col][first_of_night[1:]] - data_slice[self.mjd_col][last_of_night[:-1]]
            result = self.reduce_func(diff)
        return result


class VisitGapMetric(BaseMetric):
    """Calculate the (reduce_func) of the gap between any
    consecutive observations, in hours, regardless of night boundaries.

    Different from inter-night and intra-night gaps,
    because this is really just counting all of the times between consecutive
    observations (not time between nights or time within a night).

    Parameters
    ----------
    reduce_func : function, optional
        Function that can operate on array-like structures.
        Typically numpy function.
        Default np.median.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        night_col="night",
        reduce_func=np.median,
        metric_name="VisitGap",
        **kwargs,
    ):
        units = "hours"
        self.mjd_col = mjd_col
        self.night_col = night_col
        self.reduce_func = reduce_func
        super().__init__(col=[self.mjd_col, self.night_col], units=units, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.mjd_col)
        diff = np.diff(data_slice[self.mjd_col])
        result = self.reduce_func(diff) * 24.0
        return result
