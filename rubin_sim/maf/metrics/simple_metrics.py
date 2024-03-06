__all__ = (
    "PassMetric",
    "Coaddm5Metric",
    "MaxMetric",
    "AbsMaxMetric",
    "MeanMetric",
    "AbsMeanMetric",
    "MedianMetric",
    "AbsMedianMetric",
    "MinMetric",
    "FullRangeMetric",
    "RmsMetric",
    "RelRmsMetric",
    "SumMetric",
    "CountUniqueMetric",
    "CountMetric",
    "CountRatioMetric",
    "CountSubsetMetric",
    "CountBeyondThreshold",
    "RobustRmsMetric",
    "MaxPercentMetric",
    "AbsMaxPercentMetric",
    "BinaryMetric",
    "FracAboveMetric",
    "FracBelowMetric",
    "PercentileMetric",
    "NoutliersNsigmaMetric",
    "UniqueRatioMetric",
    "MeanAngleMetric",
    "RmsAngleMetric",
    "FullRangeAngleMetric",
    "CountExplimMetric",
    "AngularSpreadMetric",
    "RealMeanMetric",
)

import numpy as np

from .base_metric import BaseMetric

# A collection of commonly used simple metrics,
# operating on a single column and returning a float.

twopi = 2.0 * np.pi


class PassMetric(BaseMetric):
    """Pass the entire dataslice array back to the MetricBundle.

    This is most likely useful while prototyping metrics and wanting to
    just 'get the data at a point in the sky', while using a HealpixSlicer
    or a UserPointSlicer.
    """

    def __init__(self, cols=None, **kwargs):
        if cols is None:
            cols = []
        super(PassMetric, self).__init__(col=cols, metric_dtype="object", **kwargs)

    def run(self, data_slice, slice_point=None):
        return data_slice


class Coaddm5Metric(BaseMetric):
    """Calculate the coadded m5 value.

    Parameters
    ----------
    m5_col : `str`, optional
        Name of the m5 column. Default fiveSigmaDepth.
    metric_name : `str`, optional
        Name to associate with the metric output. Default "CoaddM5".
    filter_name : `str`, optional
        Optionally specify a filter to sub-select visits.
        Default None, does no sub-selection or checking.
    filter_col : `str`, optional
        Name of the filter column.
    units : `str`, optional
        Units for the metric. Default "mag".
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        metric_name="CoaddM5",
        filter_name=None,
        filter_col="Filter",
        units="mag",
        **kwargs,
    ):
        self.filter_name = filter_name
        self.filter_col = filter_col
        self.m5_col = m5_col
        super().__init__(col=m5_col, metric_name=metric_name, units=units, **kwargs)

    @staticmethod
    def coadd(single_visit_m5s):
        return 1.25 * np.log10(np.sum(10.0 ** (0.8 * single_visit_m5s)))

    def run(self, data_slice, slice_point=None):
        if len(data_slice) == 0:
            return self.badval
        if self.filter_name is not None:
            matched = np.where(data_slice[self.filter_col] == self.filter_name)
            coadd = self.coadd(data_slice[matched][self.m5_col])
        else:
            coadd = self.coadd(data_slice[self.m5_col])
        return coadd


class MaxMetric(BaseMetric):
    """Calculate the maximum of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.max(data_slice[self.colname])


class AbsMaxMetric(BaseMetric):
    """Calculate the max of the absolute value of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.max(np.abs(data_slice[self.colname]))


class MeanMetric(BaseMetric):
    """Calculate the mean of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.mean(data_slice[self.colname])


class AbsMeanMetric(BaseMetric):
    """Calculate the mean of the absolute value of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.mean(np.abs(data_slice[self.colname]))


class MedianMetric(BaseMetric):
    """Calculate the median of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.median(data_slice[self.colname])


class AbsMedianMetric(BaseMetric):
    """Calculate the median of the absolute value of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.median(np.abs(data_slice[self.colname]))


class MinMetric(BaseMetric):
    """Calculate the minimum of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.min(data_slice[self.colname])


class FullRangeMetric(BaseMetric):
    """Calculate the range of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.max(data_slice[self.colname]) - np.min(data_slice[self.colname])


class RmsMetric(BaseMetric):
    """Calculate the standard deviation of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.std(data_slice[self.colname])


class RelRmsMetric(BaseMetric):
    """Calculate the relative scatter metric (RMS divided by median)."""

    def run(self, data_slice, slice_point=None):
        return np.std(data_slice[self.colname]) / np.median(data_slice[self.colname])


class SumMetric(BaseMetric):
    """Calculate the sum of a simData column slice."""

    def run(self, data_slice, slice_point=None):
        return np.sum(data_slice[self.colname])


class CountUniqueMetric(BaseMetric):
    """Return the number of unique values."""

    def run(self, data_slice, slice_point=None):
        return np.size(np.unique(data_slice[self.colname]))


class UniqueRatioMetric(BaseMetric):
    """Return the number of unique values divided by the
    total number of values."""

    def run(self, data_slice, slice_point=None):
        ntot = float(np.size(data_slice[self.colname]))
        result = np.size(np.unique(data_slice[self.colname])) / ntot
        return result


class CountMetric(BaseMetric):
    """Count the length of a simData column slice."""

    def __init__(self, col=None, units="#", **kwargs):
        super().__init__(col=col, units=units, **kwargs)
        self.metric_dtype = "int"

    def run(self, data_slice, slice_point=None):
        return len(data_slice[self.colname])


class CountExplimMetric(BaseMetric):
    """Count the number of x second visits.
    Useful for rejecting very short exposures
    and counting 60s exposures as 2 visits.

    Parameters
    ----------
    min_exp : `float`, optional
        Minimum exposure time to consider as a "visit".
        Exposures shorter than this will not be counted (count as 0).
    expected_exp : `float`, optional
        Typical exposure time to expect.
        Exposures longer than this will be counted as
        round(visit exposure time / expected_exp). (i.e. 40s = 1, 50s = 2).
    exp_col : `str`, optional
        Column name to use for visit exposure time.

    Returns
    -------
    value : `int`
        The number of visits longer than min_exp and weighted by expected_exp.
    """

    def __init__(self, min_exp=20.0, expected_exp=30.0, exp_col="visitExposureTime", **kwargs):
        self.min_exp = min_exp
        self.expected_exp = expected_exp
        self.exp_col = exp_col
        if "col" in kwargs:
            del kwargs["col"]
        super().__init__(col=[exp_col], **kwargs)
        self.metric_dtype = "int"

    def run(self, data_slice, slice_point=None):
        nv = data_slice[self.exp_col] / self.expected_exp
        nv[np.where(data_slice[self.exp_col] < self.min_exp)[0]] = 0
        nv = np.round(nv)
        return int(np.sum(nv))


class CountRatioMetric(BaseMetric):
    """Count the length of a column slice, then divide by `norm_val`."""

    def __init__(self, col=None, norm_val=1.0, metric_name=None, units="", **kwargs):
        self.norm_val = float(norm_val)
        if metric_name is None:
            metric_name = "CountRatio %s div %.1f" % (col, norm_val)
        super(CountRatioMetric, self).__init__(col=col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        return len(data_slice[self.colname]) / self.norm_val


class CountSubsetMetric(BaseMetric):
    """Count the length of a column slice which matches `subset`."""

    def __init__(self, col=None, subset=None, units="#", **kwargs):
        super(CountSubsetMetric, self).__init__(col=col, units=units, **kwargs)
        self.metric_dtype = "int"
        self.badval = 0
        self.subset = subset

    def run(self, data_slice, slice_point=None):
        count = len(np.where(data_slice[self.colname] == self.subset)[0])
        return count


class CountBeyondThreshold(BaseMetric):
    """Count the number of entries in a data column above or below
    the `threshold`."""

    def __init__(self, col=None, lower_threshold=None, upper_threshold=None, **kwargs):
        super().__init__(col=col, **kwargs)
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def run(self, data_slice, slice_point=None):
        # Look for data values which match the criteria for the thresholds
        if self.upper_threshold is None and self.lower_threshold is None:
            count = len(data_slice)
        elif self.upper_threshold is None:
            count = len(np.where(data_slice[self.colname] > self.lower_threshold)[0])
        elif self.lower_threshold is None:
            count = len(np.where(data_slice[self.colname] < self.upper_threshold)[0])
        else:
            count = len(
                np.where(
                    (data_slice[self.colname] > self.lower_threshold)
                    and (data_slice[self.colname] < self.upper_threshold)
                )[0]
            )
        return count


class RobustRmsMetric(BaseMetric):
    """Use the inter-quartile range of the data to estimate the RMS.
    Robust, as this calculation does not include outliers in the distribution.
    """

    def run(self, data_slice, slice_point=None):
        iqr = np.percentile(data_slice[self.colname], 75) - np.percentile(data_slice[self.colname], 25)
        rms = iqr / 1.349  # approximation
        return rms


class MaxPercentMetric(BaseMetric):
    """Return the percent of data which matches the maximum value
    of the data.
    """

    def run(self, data_slice, slice_point=None):
        n_max = np.size(np.where(data_slice[self.colname] == np.max(data_slice[self.colname]))[0])
        percent = n_max / float(data_slice[self.colname].size) * 100.0
        return percent


class AbsMaxPercentMetric(BaseMetric):
    """Return the percent of data which matches the absolute value of the
    max value of the data.
    """

    def run(self, data_slice, slice_point=None):
        max_val = np.abs(np.max(data_slice[self.colname]))
        n_max = np.size(np.where(np.abs(data_slice[self.colname]) == max_val)[0])
        percent = n_max / float(data_slice[self.colname].size) * 100.0
        return percent


class BinaryMetric(BaseMetric):
    """Return 1 if there is data, `badval` otherwise."""

    def run(self, data_slice, slice_point=None):
        if data_slice.size > 0:
            return 1
        else:
            return self.badval


class FracAboveMetric(BaseMetric):
    """Find the fraction of data values above a given `cutoff`."""

    def __init__(self, col=None, cutoff=0.5, scale=1, metric_name=None, **kwargs):
        # Col could just get passed in bundle with kwargs,
        # by explicitly pulling it out first, we support use cases where
        # class instantiated without explicit 'col=').
        if metric_name is None:
            metric_name = "FracAbove %.2f in %s" % (cutoff, col)
        super(FracAboveMetric, self).__init__(col, metric_name=metric_name, **kwargs)
        self.cutoff = cutoff
        self.scale = scale

    def run(self, data_slice, slice_point=None):
        good = np.where(data_slice[self.colname] >= self.cutoff)[0]
        frac_above = np.size(good) / float(np.size(data_slice[self.colname]))
        frac_above = frac_above * self.scale
        return frac_above


class FracBelowMetric(BaseMetric):
    """Find the fraction of data values below a given `cutoff`."""

    def __init__(self, col=None, cutoff=0.5, scale=1, metric_name=None, **kwargs):
        if metric_name is None:
            metric_name = "FracBelow %.2f %s" % (cutoff, col)
        super(FracBelowMetric, self).__init__(col, metric_name=metric_name, **kwargs)
        self.cutoff = cutoff
        self.scale = scale

    def run(self, data_slice, slice_point=None):
        good = np.where(data_slice[self.colname] <= self.cutoff)[0]
        frac_below = np.size(good) / float(np.size(data_slice[self.colname]))
        frac_below = frac_below * self.scale
        return frac_below


class PercentileMetric(BaseMetric):
    """Find the value of a column at a given `percentile`."""

    def __init__(self, col=None, percentile=90, metric_name=None, **kwargs):
        if metric_name is None:
            metric_name = "%.0fth%sile %s" % (percentile, "%", col)
        super(PercentileMetric, self).__init__(col=col, metric_name=metric_name, **kwargs)
        self.percentile = percentile

    def run(self, data_slice, slice_point=None):
        pval = np.percentile(data_slice[self.colname], self.percentile)
        return pval


class NoutliersNsigmaMetric(BaseMetric):
    """Calculate the # of visits less than n_sigma below the mean (n_sigma<0)
    or more than n_sigma above the mean.
    """

    def __init__(self, col=None, n_sigma=3.0, metric_name=None, **kwargs):
        self.n_sigma = n_sigma
        self.col = col
        if metric_name is None:
            metric_name = "Noutliers %.1f %s" % (self.n_sigma, self.col)
        super(NoutliersNsigmaMetric, self).__init__(col=col, metric_name=metric_name, **kwargs)
        self.metric_dtype = "int"

    def run(self, data_slice, slice_point=None):
        med = np.mean(data_slice[self.colname])
        std = np.std(data_slice[self.colname])
        boundary = med + self.n_sigma * std
        # If nsigma is positive, look for outliers above median.
        if self.n_sigma >= 0:
            outsiders = np.where(data_slice[self.colname] > boundary)
        # Else look for outliers below median.
        else:
            outsiders = np.where(data_slice[self.colname] < boundary)
        return len(data_slice[self.colname][outsiders])


def _rotate_angles(angles):
    """Private utility for the '*Angle' Metrics below.

    This takes a series of angles between 0-2pi and rotates them so that the
    first angle is at 0, ensuring the biggest 'gap' is at the end of the
    series.
    This simplifies calculations like the 'mean' and 'rms' or 'fullrange',
    removing the discontinuity at 0/2pi.
    """
    angleidx = np.argsort(angles)
    diffangles = np.diff(angles[angleidx])
    start_to_end = np.array([twopi - angles[angleidx][-1] + angles[angleidx][0]], float)
    if start_to_end < -2.0 * np.pi:
        raise ValueError("Angular metrics expect radians, this seems to be in degrees")
    diffangles = np.concatenate([diffangles, start_to_end])
    maxdiff = np.where(diffangles == diffangles.max())[0]
    if len(maxdiff) > 1:
        maxdiff = maxdiff[-1:]
    if maxdiff == (len(angles) - 1):
        rotation = angles[angleidx][0]
    else:
        rotation = angles[angleidx][maxdiff + 1][0]
    return (rotation, (angles - rotation) % twopi)


class MeanAngleMetric(BaseMetric):
    """Calculate the mean of an angular (degree) column slice.

    'MeanAngle' differs from 'Mean' in that it accounts for wraparound at 2pi.
    """

    def run(self, data_slice, slice_point=None):
        """Calculate mean angle via unit vectors.
        If unit vector 'strength' is less than 0.1,
        then just set mean to 180 degrees
        (as this indicates nearly uniformly distributed angles).
        """
        x = np.cos(np.radians(data_slice[self.colname]))
        y = np.sin(np.radians(data_slice[self.colname]))
        meanx = np.mean(x)
        meany = np.mean(y)
        angle = np.arctan2(meany, meanx)
        radius = np.sqrt(meanx**2 + meany**2)
        mean = angle % twopi
        if radius < 0.1:
            mean = np.pi
        return np.degrees(mean)


class RmsAngleMetric(BaseMetric):
    """Calculate the standard deviation of an angular (degrees) column slice.

    'RmsAngle' differs from 'Rms' in that it accounts for wraparound at 2pi.
    """

    def run(self, data_slice, slice_point=None):
        rotation, angles = _rotate_angles(np.radians(data_slice[self.colname]))
        return np.std(np.degrees(angles))


class FullRangeAngleMetric(BaseMetric):
    """Calculate the full range of an angular (degrees) column slice.

    'FullRangeAngle' differs from 'FullRange' in that it accounts for
    wraparound at 2pi.
    """

    def run(self, data_slice, slice_point=None):
        rotation, angles = _rotate_angles(np.radians(data_slice[self.colname]))
        return np.degrees(angles.max() - angles.min())


class AngularSpreadMetric(BaseMetric):
    """Compute the angular spread statistic which measures
    uniformity of a distribution angles accounting for 2pi periodicity.

    The strategy is to first map angles into unit vectors on the unit circle,
    and then compute the 2D centroid of those vectors.
    A uniform distribution of angles will lead to a distribution of
    unit vectors with mean that approaches the origin.
    In contrast, a delta function distribution of angles leads to a
    delta function distribution of unit vectors with a mean that lands on the
    unit circle.

    The angular spread statistic is then defined as 1 - R,
    where R is the radial offset of the mean
    of the unit vectors derived from the input angles.
    R approaches 1 for a uniform distribution
    of angles and 0 for a delta function distribution of angles.

    The optional parameter `period` may be used to specificy periodicity
    other than 2 pi.
    """

    def __init__(self, col=None, period=2.0 * np.pi, **kwargs):
        # https://en.wikipedia.org/wiki/Directional_statistics
        # #Measures_of_location_and_spread
        # jmeyers314@gmail.com
        self.period = period
        super(AngularSpreadMetric, self).__init__(col=col, **kwargs)

    def run(self, data_slice, slice_point=None):
        # Unit vectors; unwrapped at specified period
        x = np.cos(data_slice[self.colname] * 2.0 * np.pi / self.period)
        y = np.sin(data_slice[self.colname] * 2.0 * np.pi / self.period)
        meanx = np.mean(x)
        meany = np.mean(y)
        # radial offset (i.e., length) of the mean unit vector
        R = np.hypot(meanx, meany)
        return 1.0 - R


class RealMeanMetric(BaseMetric):
    """Calculate the mean of a column with no nans or infs."""

    def run(self, data_slice, slice_point=None):
        return np.mean(data_slice[self.colname][np.isfinite(data_slice[self.colname])])
