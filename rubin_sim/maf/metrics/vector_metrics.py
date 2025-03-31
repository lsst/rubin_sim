__all__ = (
    "HistogramMetric",
    "AccumulateMetric",
    "AccumulateCountMetric",
    "HistogramM5Metric",
    "AccumulateM5Metric",
    "AccumulateUniformityMetric",
)

import numpy as np
from scipy import stats

from .base_metric import BaseMetric


class VectorMetric(BaseMetric):
    """
    Base for metrics that return a vector
    """

    def __init__(self, times=None, time_col="night", col="night", units=None, metric_dtype=float, **kwargs):

        if isinstance(col, str):
            cols = [col, time_col]
        else:
            cols = list(col) + [time_col]

        super(VectorMetric, self).__init__(col=cols, units=units, metric_dtype=metric_dtype, **kwargs)
        self.times = times
        self.time_col = time_col
        self.shape = np.size(times)


class HistogramMetric(VectorMetric):
    """
    A wrapper to stats.binned_statistic
    """

    def __init__(
        self,
        bins=None,
        bin_col="night",
        col="night",
        units="Count",
        statistic="count",
        metric_dtype=float,
        **kwargs,
    ):
        self.statistic = statistic
        self.col = col
        super(HistogramMetric, self).__init__(
            col=col, times=bins, time_col=bin_col, units=units, metric_dtype=metric_dtype, **kwargs
        )
        self.shape = np.size(self.times) - 1

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        result, bin_edges, bin_number = stats.binned_statistic(
            data_slice[self.time_col],
            data_slice[self.col],
            bins=self.times,
            statistic=self.statistic,
        )
        return result


class AccumulateMetric(VectorMetric):
    """
    Calculate the accumulated stat
    """

    def __init__(
        self, col="night", times=None, time_col="night", function=np.add, metric_dtype=float, **kwargs
    ):
        self.function = function
        super(AccumulateMetric, self).__init__(
            col=col, time_col=time_col, times=times, metric_dtype=metric_dtype, **kwargs
        )
        self.col = col

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)

        result = self.function.accumulate(data_slice[self.col])
        indices = np.searchsorted(data_slice[self.time_col], self.times, side="right")
        indices[np.where(indices >= np.size(result))] = np.size(result) - 1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval
        return result


class AccumulateCountMetric(AccumulateMetric):
    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        to_count = np.ones(data_slice.size, dtype=int)
        result = self.function.accumulate(to_count)
        indices = np.searchsorted(data_slice[self.time_col], self.times, side="right")
        indices[np.where(indices >= np.size(result))] = np.size(result) - 1
        result = result[indices]
        return result


class HistogramM5Metric(HistogramMetric):
    """
    Calculate the coadded depth for each bin (e.g., per night).
    """

    def __init__(
        self,
        bins=None,
        bin_col="night",
        m5_col="fiveSigmaDepth",
        units="mag",
        metric_name="HistogramM5Metric",
        **kwargs,
    ):
        super(HistogramM5Metric, self).__init__(
            col=m5_col, bin_col=bin_col, bins=bins, metric_name=metric_name, units=units, **kwargs
        )
        self.m5_col = m5_col

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        flux = 10.0 ** (0.8 * data_slice[self.m5_col])
        result, bin_edges, bin_number = stats.binned_statistic(
            data_slice[self.time_col], flux, bins=self.times, statistic="sum"
        )
        no_flux = np.where(result == 0.0)
        result = 1.25 * np.log10(result)
        result[no_flux] = self.badval
        return result


class AccumulateM5Metric(AccumulateMetric):
    def __init__(
        self,
        times=None,
        time_col="night",
        m5_col="fiveSigmaDepth",
        metric_name="AccumulateM5Metric",
        **kwargs,
    ):
        self.m5_col = m5_col
        super(AccumulateM5Metric, self).__init__(
            times=times, time_col=time_col, col=m5_col, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        flux = 10.0 ** (0.8 * data_slice[self.m5_col])

        result = np.add.accumulate(flux)
        indices = np.searchsorted(data_slice[self.time_col], self.times, side="right")
        indices[np.where(indices >= np.size(result))] = np.size(result) - 1
        result = result[indices]
        result = 1.25 * np.log10(result)
        result[np.where(indices == 0)] = self.badval
        return result


class AccumulateUniformityMetric(AccumulateMetric):
    """
    Make a 2D version of UniformityMetric
    """

    def __init__(
        self,
        times=None,
        time_col="night",
        exp_mjd_col="observationStartMJD",
        metric_name="AccumulateUniformityMetric",
        survey_length=10.0,
        units="Fraction",
        **kwargs,
    ):
        self.exp_mjd_col = exp_mjd_col
        if times is None:
            times = np.arange(0, np.ceil(survey_length * 365.25))
        super(AccumulateUniformityMetric, self).__init__(
            times=times, time_col=time_col, col=exp_mjd_col, metric_name=metric_name, units=units, **kwargs
        )
        self.survey_length = survey_length

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        if data_slice.size == 1:
            return np.ones(self.times.size, dtype=float)

        visits_per_night, blah = np.histogram(data_slice[self.time_col], bins=self.times)
        visits_per_night = np.add.accumulate(visits_per_night)
        expected_per_night = np.arange(0.0, self.times.size - 1) / (self.times.size - 2) * data_slice.size

        d_max = np.abs(visits_per_night - expected_per_night)
        d_max = np.maximum.accumulate(d_max)
        result = d_max / expected_per_night.max()
        return result
