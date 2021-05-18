import numpy as np
from .baseMetric import BaseMetric
from scipy import stats

__all__ = ['HistogramMetric','AccumulateMetric', 'AccumulateCountMetric',
           'HistogramM5Metric', 'AccumulateM5Metric', 'AccumulateUniformityMetric']


class VectorMetric(BaseMetric):
    """
    Base for metrics that return a vector
    """
    def __init__(self, bins=None, binCol='night', col='night', units=None, metricDtype=float, **kwargs):
        super(VectorMetric,self).__init__(col=[col,binCol],units=units,metricDtype=metricDtype,**kwargs)
        self.bins = bins
        self.binCol = binCol
        self.shape = np.size(bins)-1

class HistogramMetric(VectorMetric):
    """
    A wrapper to stats.binned_statistic
    """
    def __init__(self, bins=None, binCol='night', col='night', units='Count', statistic='count',
                 metricDtype=float, **kwargs):
        self.statistic = statistic
        self.col=col
        super(HistogramMetric,self).__init__(col=col, bins=bins, binCol=binCol, units=units,
                                              metricDtype=metricDtype,**kwargs)

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.binCol)
        result, binEdges,binNumber = stats.binned_statistic(dataSlice[self.binCol],
                                                            dataSlice[self.col],
                                                            bins=self.bins,
                                                            statistic=self.statistic)
        return result

class AccumulateMetric(VectorMetric):
    """
    Calculate the accumulated stat
    """
    def __init__(self, col='night', bins=None, binCol='night', function=np.add,
                 metricDtype=float, **kwargs):
        self.function = function
        super(AccumulateMetric,self).__init__(col=col,binCol=binCol, bins=bins,
                                              metricDtype=metricDtype,**kwargs)
        self.col=col

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.binCol)

        result = self.function.accumulate(dataSlice[self.col])
        indices = np.searchsorted(dataSlice[self.binCol], self.bins[1:], side='right')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval
        return result

class AccumulateCountMetric(AccumulateMetric):
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.binCol)
        toCount = np.ones(dataSlice.size, dtype=int)
        result = self.function.accumulate(toCount)
        indices = np.searchsorted(dataSlice[self.binCol], self.bins[1:], side='right')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval
        return result

class HistogramM5Metric(HistogramMetric):
    """
    Calculate the coadded depth for each bin (e.g., per night).
    """
    def __init__(self, bins=None, binCol='night', m5Col='fiveSigmaDepth', units='mag',
                metricName='HistogramM5Metric',**kwargs):

        super(HistogramM5Metric,self).__init__(col=m5Col,binCol=binCol, bins=bins,
                                               metricName=metricName,
                                               units=units,**kwargs)
        self.m5Col=m5Col

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.binCol)
        flux = 10.**(.8*dataSlice[self.m5Col])
        result, binEdges,binNumber = stats.binned_statistic(dataSlice[self.binCol],
                                                            flux,
                                                            bins=self.bins,
                                                            statistic='sum')
        noFlux = np.where(result == 0.)
        result = 1.25*np.log10(result)
        result[noFlux] = self.badval
        return result

class AccumulateM5Metric(AccumulateMetric):
    def __init__(self, bins=None, binCol='night', m5Col='fiveSigmaDepth',
                metricName='AccumulateM5Metric',**kwargs):
        self.m5Col = m5Col
        super(AccumulateM5Metric,self).__init__(bins=bins, binCol=binCol,col=m5Col,
                                                metricName=metricName,**kwargs)


    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.binCol)
        flux = 10.**(.8*dataSlice[self.m5Col])

        result = np.add.accumulate(flux)
        indices = np.searchsorted(dataSlice[self.binCol], self.bins[1:], side='right')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result = 1.25*np.log10(result)
        result[np.where(indices == 0)] = self.badval
        return result


class AccumulateUniformityMetric(AccumulateMetric):
    """
    Make a 2D version of UniformityMetric
    """
    def __init__(self, bins=None, binCol='night', expMJDCol='observationStartMJD',
                 metricName='AccumulateUniformityMetric',surveyLength=10.,
                 units='Fraction', **kwargs):
        self.expMJDCol = expMJDCol
        if bins is None:
            bins = np.arange(0,np.ceil(surveyLength*365.25))-.5
        super(AccumulateUniformityMetric,self).__init__(bins=bins, binCol=binCol,col=expMJDCol,
                                                        metricName=metricName,units=units,**kwargs)
        self.surveyLength = surveyLength

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.binCol)
        if dataSlice.size == 1:
            return np.ones(self.bins.size-1, dtype=float)

        visitsPerNight, blah = np.histogram(dataSlice[self.binCol], bins=self.bins)
        visitsPerNight = np.add.accumulate(visitsPerNight)
        expectedPerNight = np.arange(0.,self.bins.size-1)/(self.bins.size-2) * dataSlice.size

        D_max = np.abs(visitsPerNight-expectedPerNight)
        D_max = np.maximum.accumulate(D_max)
        result = D_max/expectedPerNight.max()
        return result
