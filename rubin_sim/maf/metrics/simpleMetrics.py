import numpy as np
from .baseMetric import BaseMetric

# A collection of commonly used simple metrics, operating on a single column and returning a float.

__all__ = ['PassMetric', 'Coaddm5Metric', 'MaxMetric', 'AbsMaxMetric', 'MeanMetric', 'AbsMeanMetric',
           'MedianMetric', 'AbsMedianMetric', 'MinMetric', 'FullRangeMetric', 'RmsMetric', 'SumMetric',
           'CountUniqueMetric', 'CountMetric', 'CountRatioMetric', 'CountSubsetMetric', 'RobustRmsMetric',
           'MaxPercentMetric', 'AbsMaxPercentMetric', 'BinaryMetric', 'FracAboveMetric', 'FracBelowMetric',
           'PercentileMetric', 'NoutliersNsigmaMetric', 'UniqueRatioMetric',
           'MeanAngleMetric', 'RmsAngleMetric', 'FullRangeAngleMetric', 'CountExplimMetric']

twopi = 2.0*np.pi


class PassMetric(BaseMetric):
    """
    Just pass the entire array through
    """
    def __init__(self, cols=None, **kwargs):
        if cols is None:
            cols= []
        super(PassMetric, self).__init__(col=cols, metricDtype='object', **kwargs)
    def run(self, dataSlice, slicePoint=None):
        return dataSlice


class Coaddm5Metric(BaseMetric):
    """Calculate the coadded m5 value at this gridpoint.
    """
    def __init__(self, m5Col='fiveSigmaDepth', metricName='CoaddM5', **kwargs):
        """Instantiate metric.

        m5col = the column name of the individual visit m5 data."""
        super(Coaddm5Metric, self).__init__(col=m5Col, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname])))

class MaxMetric(BaseMetric):
    """Calculate the maximum of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.max(dataSlice[self.colname])

class AbsMaxMetric(BaseMetric):
    """Calculate the max of the absolute value of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.max(np.abs(dataSlice[self.colname]))

class MeanMetric(BaseMetric):
    """Calculate the mean of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.mean(dataSlice[self.colname])

class AbsMeanMetric(BaseMetric):
    """Calculate the mean of the absolute value of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.mean(np.abs(dataSlice[self.colname]))

class MedianMetric(BaseMetric):
    """Calculate the median of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.median(dataSlice[self.colname])

class AbsMedianMetric(BaseMetric):
    """Calculate the median of the absolute value of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.median(np.abs(dataSlice[self.colname]))

class MinMetric(BaseMetric):
    """Calculate the minimum of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.min(dataSlice[self.colname])

class FullRangeMetric(BaseMetric):
    """Calculate the range of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.max(dataSlice[self.colname])-np.min(dataSlice[self.colname])

class RmsMetric(BaseMetric):
    """Calculate the standard deviation of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.std(dataSlice[self.colname])

class SumMetric(BaseMetric):
    """Calculate the sum of a simData column slice.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.sum(dataSlice[self.colname])

class CountUniqueMetric(BaseMetric):
    """Return the number of unique values.
    """
    def run(self, dataSlice, slicePoint=None):
        return np.size(np.unique(dataSlice[self.colname]))

class UniqueRatioMetric(BaseMetric):
    """Return the number of unique values divided by the total number of values.
    """
    def run(self, dataSlice, slicePoint=None):
        ntot = float(np.size(dataSlice[self.colname]))
        result = np.size(np.unique(dataSlice[self.colname])) / ntot
        return result

class CountMetric(BaseMetric):
    """Count the length of a simData column slice. """
    def __init__(self, col=None, **kwargs):
        super(CountMetric, self).__init__(col=col, **kwargs)
        self.metricDtype = 'int'

    def run(self, dataSlice, slicePoint=None):
        return len(dataSlice[self.colname])


class CountExplimMetric(BaseMetric):
    """Count the number of x second visits.  Useful for rejecting very short exposures
    and counting 60s exposures as 2 visits."""
    def __init__(self, col=None, minExp=20., expectedExp=30., expCol='visitExposureTime', **kwargs):
        self.minExp = minExp
        self.expectedExp = expectedExp
        self.expCol = expCol
        super().__init__(col=[col, expCol], **kwargs)
        self.metricDtype = 'int'

    def run(self, dataSlice, slicePoint=None):
        nv = dataSlice[self.expCol] / self.expectedExp
        nv[np.where(dataSlice[self.expCol] < self.minExp)[0]] = 0
        nv = np.round(nv)
        return int(np.sum(nv))

class CountRatioMetric(BaseMetric):
    """Count the length of a simData column slice, then divide by 'normVal'. 
    """
    def __init__(self, col=None, normVal=1., metricName=None, **kwargs):
        self.normVal = float(normVal)
        if metricName is None:
            metricName = 'CountRatio %s div %.1f'%(col, normVal)
        super(CountRatioMetric, self).__init__(col=col, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        return len(dataSlice[self.colname])/self.normVal

class CountSubsetMetric(BaseMetric):
    """Count the length of a simData column slice which matches 'subset'. 
    """
    def __init__(self, col=None, subset=None, **kwargs):
        super(CountSubsetMetric, self).__init__(col=col, **kwargs)
        self.metricDtype = 'int'
        self.badval = 0
        self.subset = subset

    def run(self, dataSlice, slicePoint=None):
        count = len(np.where(dataSlice[self.colname] == self.subset)[0])
        return count

class RobustRmsMetric(BaseMetric):
    """Use the inter-quartile range of the data to estimate the RMS.  
    Robust since this calculation does not include outliers in the distribution.
    """
    def run(self, dataSlice, slicePoint=None):
        iqr = np.percentile(dataSlice[self.colname],75)-np.percentile(dataSlice[self.colname],25)
        rms = iqr/1.349 #approximation
        return rms

class MaxPercentMetric(BaseMetric):
    """Return the percent of the data which has the maximum value.
    """
    def run(self, dataSlice, slicePoint=None):
        nMax = np.size(np.where(dataSlice[self.colname] == np.max(dataSlice[self.colname]))[0])
        percent = nMax / float(dataSlice[self.colname].size) * 100.
        return percent

class AbsMaxPercentMetric(BaseMetric):
    """Return the percent of the data which has the absolute value of the max value of the data.
    """
    def run(self, dataSlice, slicePoint=None):
        maxVal = np.abs(np.max(dataSlice[self.colname]))
        nMax = np.size(np.where(np.abs(dataSlice[self.colname]) == maxVal)[0])
        percent = nMax / float(dataSlice[self.colname].size) * 100.0
        return percent

class BinaryMetric(BaseMetric):
    """Return 1 if there is data. 
    """
    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size > 0:
            return 1
        else:
            return self.badval

class FracAboveMetric(BaseMetric):
    """Find the fraction of data values above a given value.
    """
    def __init__(self, col=None, cutoff=0.5, scale=1, metricName=None, **kwargs):
        # Col could just get passed in bundle with kwargs, but by explicitly pulling it out
        #  first, we support use cases where class instantiated without explicit 'col=').
        if metricName is None:
            metricName = 'FracAbove %.2f in %s' %(cutoff, col)
        super(FracAboveMetric, self).__init__(col, metricName=metricName, **kwargs)
        self.cutoff = cutoff
        self.scale = scale
    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.colname] >= self.cutoff)[0]
        fracAbove = np.size(good)/float(np.size(dataSlice[self.colname]))
        fracAbove = fracAbove * self.scale
        return fracAbove

class FracBelowMetric(BaseMetric):
    """Find the fraction of data values below a given value.
    """
    def __init__(self, col=None, cutoff=0.5, scale=1, metricName=None, **kwargs):
        if metricName is None:
            metricName = 'FracBelow %.2f %s' %(cutoff, col)
        super(FracBelowMetric, self).__init__(col, metricName=metricName, **kwargs)
        self.cutoff = cutoff
        self.scale = scale
    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.colname] <= self.cutoff)[0]
        fracBelow = np.size(good)/float(np.size(dataSlice[self.colname]))
        fracBelow = fracBelow * self.scale
        return fracBelow

class PercentileMetric(BaseMetric):
    """Find the value of a column at a given percentile.
    """
    def __init__(self, col=None, percentile=90, metricName=None, **kwargs):
        if metricName is None:
            metricName = '%.0fth%sile %s' %(percentile, '%', col)
        super(PercentileMetric, self).__init__(col=col, metricName=metricName, **kwargs)
        self.percentile = percentile
    def run(self, dataSlice, slicePoint=None):
        pval = np.percentile(dataSlice[self.colname], self.percentile)
        return pval

class NoutliersNsigmaMetric(BaseMetric):
    """Calculate the # of visits less than nSigma below the mean (nSigma<0) or
    more than nSigma above the mean of 'col'.
    """
    def __init__(self, col=None, nSigma=3., metricName=None, **kwargs):
        self.nSigma = nSigma
        self.col = col
        if metricName is None:
            metricName = 'Noutliers %.1f %s' %(self.nSigma, self.col)
        super(NoutliersNsigmaMetric, self).__init__(col=col, metricName=metricName, **kwargs)
        self.metricDtype = 'int'

    def run(self, dataSlice, slicePoint=None):
        med = np.mean(dataSlice[self.colname])
        std = np.std(dataSlice[self.colname])
        boundary = med + self.nSigma*std
        # If nsigma is positive, look for outliers above median.
        if self.nSigma >=0:
            outsiders = np.where(dataSlice[self.colname] > boundary)
        # Else look for outliers below median.
        else:
            outsiders = np.where(dataSlice[self.colname] < boundary)
        return len(dataSlice[self.colname][outsiders])

def _rotateAngles(angles):
    """Private utility for the '*Angle' Metrics below.

    This takes a series of angles between 0-2pi and rotates them so that the
    first angle is at 0, ensuring the biggest 'gap' is at the end of the series.
    This simplifies calculations like the 'mean' and 'rms' or 'fullrange', removing
    the discontinuity at 0/2pi.
    """
    angleidx = np.argsort(angles)
    diffangles = np.diff(angles[angleidx])
    start_to_end = np.array([twopi-angles[angleidx][-1] + angles[angleidx][0]], float)
    if start_to_end < -2.*np.pi:
        raise ValueError('Angular metrics expect radians, this seems to be in degrees')
    diffangles = np.concatenate([diffangles, start_to_end])
    maxdiff = np.where(diffangles == diffangles.max())[0]
    if len(maxdiff) > 1:
        maxdiff = maxdiff[-1:]
    if maxdiff == (len(angles)-1):
        rotation = angles[angleidx][0]
    else:
        rotation = angles[angleidx][maxdiff+1][0]
    return (rotation, (angles - rotation) % twopi)

class MeanAngleMetric(BaseMetric):
    """Calculate the mean of an angular (degree) simData column slice.

    'MeanAngle' differs from 'Mean' in that it accounts for wraparound at 2pi.
    """
    def run(self, dataSlice, slicePoint=None):
        """Calculate mean angle via unit vectors.
        If unit vector 'strength' is less than 0.1, then just set mean to 180 degrees
        (as this indicates nearly uniformly distributed angles). 
        """
        x = np.cos(np.radians(dataSlice[self.colname]))
        y = np.sin(np.radians(dataSlice[self.colname]))
        meanx = np.mean(x)
        meany = np.mean(y)
        angle = np.arctan2(meany, meanx)
        radius = np.sqrt(meanx**2 + meany**2)
        mean = angle % twopi
        if radius < 0.1:
            mean = np.pi
        return np.degrees(mean)

class RmsAngleMetric(BaseMetric):
    """Calculate the standard deviation of an angular (degrees) simData column slice.

    'RmsAngle' differs from 'Rms' in that it accounts for wraparound at 2pi.
    """
    def run(self, dataSlice, slicePoint=None):
        rotation, angles = _rotateAngles(np.radians(dataSlice[self.colname]))
        return np.std(np.degrees(angles))

class FullRangeAngleMetric(BaseMetric):
    """Calculate the full range of an angular (degrees) simData column slice.

    'FullRangeAngle' differs from 'FullRange' in that it accounts for wraparound at 2pi.
    """
    def run(self, dataSlice, slicePoint=None):
        rotation, angles = _rotateAngles(np.radians(dataSlice[self.colname]))
        return np.degrees(angles.max() - angles.min())
