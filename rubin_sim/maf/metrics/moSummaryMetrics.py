import numpy as np
import warnings

from .moMetrics import BaseMoMetric

__all__ = ['integrateOverH', 'ValueAtHMetric', 'MeanValueAtHMetric',
           'MoCompletenessMetric', 'MoCompletenessAtTimeMetric']


def integrateOverH(Mvalues, Hvalues, Hindex = 0.33):
    """Function to calculate a metric value integrated over an Hrange, assuming a power-law distribution.

    Parameters
    ----------
    Mvalues : numpy.ndarray
        The metric values at each H value.
    Hvalues : numpy.ndarray
        The H values corresponding to each Mvalue (must be the same length).
    Hindex : float, opt
        The power-law index expected for the H value distribution.
        Default is 0.33  (dN/dH = 10^(Hindex * H) ).

    Returns
    --------
    numpy.ndarray
       The integrated or cumulative metric values.
    """
    # Set expected H distribution.
    # dndh = differential size distribution (number in this bin)
    dndh = np.power(10., Hindex*(Hvalues-Hvalues.min()))
    # dn = cumulative size distribution (number in this bin and brighter)
    intVals = np.cumsum(Mvalues*dndh)/np.cumsum(dndh)
    return intVals


class ValueAtHMetric(BaseMoMetric):
    """Return the metric value at a given H value.

    Requires the metric values to be one-dimensional (typically, completeness values).

    Parameters
    ----------
    Hmark : float, opt
        The H value at which to look up the metric value. Default = 22.
    """
    def __init__(self, Hmark=22, **kwargs):
        metricName = 'Value At H=%.1f' %(Hmark)
        super(ValueAtHMetric, self).__init__(metricName=metricName, **kwargs)
        self.Hmark = Hmark

    def run(self, metricVals, Hvals):
        # Check if desired H value is within range of H values.
        if (self.Hmark < Hvals.min()) or (self.Hmark > Hvals.max()):
            warnings.warn('Desired H value of metric outside range of provided H values.')
            return None
        if metricVals.shape[0] != 1:
            warnings.warn('This is not an appropriate summary statistic for this data - need 1d values.')
            return None
        value = np.interp(self.Hmark, Hvals, metricVals[0])
        return value


class MeanValueAtHMetric(BaseMoMetric):
    """Return the mean value of a metric at a given H.

    Allows the metric values to be multi-dimensional (i.e. use a cloned H distribution).

    Parameters
    ----------
    Hmark : float, opt
        The H value at which to look up the metric value. Default = 22.
    """
    def __init__(self, Hmark=22, reduceFunc=np.mean, metricName=None, **kwargs):
        if metricName is None:
            metricName = 'Mean Value At H=%.1f' %(Hmark)
        super(MeanValueAtHMetric, self).__init__(metricName=metricName, **kwargs)
        self.Hmark = Hmark
        self.reduceFunc = reduceFunc

    def run(self, metricVals, Hvals):
        # Check if desired H value is within range of H values.
        if (self.Hmark < Hvals.min()) or (self.Hmark > Hvals.max()):
            warnings.warn('Desired H value of metric outside range of provided H values.')
            return None
        value = np.interp(self.Hmark, Hvals, self.reduceFunc(metricVals.swapaxes(0, 1), axis=1))
        return value


class MoCompletenessMetric(BaseMoMetric):
    """Calculate the fraction of the population that meets `threshold` value or higher.
    This is equivalent to calculating the completeness (relative to the entire population) given
    the output of a Discovery_N_Chances metric, or the fraction of the population that meets a given cutoff
    value for Color determination metrics.

    Any moving object metric that outputs a float value can thus have the 'fraction of the population'
    with greater than X value calculated here, as a summary statistic.

    Parameters
    ----------
    threshold : int, opt
        Count the fraction of the population that exceeds this value. Default = 1.
    nbins : int, opt
        If the H values for the metric are not a cloned distribution, then split up H into this many bins.
        Default 20.
    minHrange : float, opt
        If the H values for the metric are not a cloned distribution, then split up H into at least this
        range (otherwise just use the min/max of the H values). Default 1.0
    cumulative : bool, opt
        If False, simply report the differential fractional value (or differential completeness).
        If True, integrate over the H distribution (using IntegrateOverH) to report a cumulative fraction.
        Default None which becomes True;
        if metricName is set and starts with 'Differential' this will then set to False.
    Hindex : float, opt
        Use Hindex as the power law to integrate over H, if cumulative is True. Default 0.3.
    """
    def __init__(self, threshold=1, nbins=20, minHrange=1.0, cumulative=None, Hindex=0.33, **kwargs):
        if cumulative is None:  # if metricName does not start with 'differential', then cumulative->True
            if 'metricName' not in kwargs:
                self.cumulative = True
                metricName = 'CumulativeCompleteness'
            else:  #  'metricName' in kwargs:
                metricName = kwargs.pop('metricName')
                if metricName.lower().startswith('differential'):
                    self.cumulative=False
                else:
                    self.cumulative=True
        else: # cumulative was set
            self.cumulative = cumulative
            if 'metricName' in kwargs:
                metricName = kwargs.pop('metricName')
                if metricName.lower().startswith('differential') and self.cumulative:
                    warnings.warn(f'Completeness metricName is {metricName} but cumulative is True')
            else:
                if self.cumulative:
                    metricName = 'CumulativeCompleteness'
                else:
                    metricName = 'DifferentialCompleteness'
        if self.cumulative:
            units = "<=H"
        else:
            units = "@H"
        super().__init__(metricName=metricName, units=units, **kwargs)
        self.threshold = threshold
        # If H is not a cloned distribution, then we need to specify how to bin these values.
        self.nbins = nbins
        self.minHrange = minHrange
        self.Hindex = Hindex

    def run(self, metricValues, Hvals):
        nSsos = metricValues.shape[0]
        nHval = len(Hvals)
        metricValH = metricValues.swapaxes(0, 1)
        if nHval == metricValues.shape[1]:
            # Hvals array is probably the same as the cloned H array.
            completeness = np.zeros(len(Hvals), float)
            for i, H in enumerate(Hvals):
                completeness[i] = np.where(metricValH[i].filled(0) >= self.threshold)[0].size
            completeness = completeness / float(nSsos)
        else:
            # The Hvals are spread more randomly among the objects (we probably used one per object).
            hrange = Hvals.max() - Hvals.min()
            minH = Hvals.min()
            if hrange < self.minHrange:
                hrange = self.minHrange
                minH = Hvals.min() - hrange/2.0
            stepsize = hrange / float(self.nbins)
            bins = np.arange(minH, minH + hrange + stepsize/2.0, stepsize)
            Hvals = bins[:-1]
            n_all, b = np.histogram(metricValH[0], bins)
            condition = np.where(metricValH[0] >= self.threshold)[0]
            n_found, b = np.histogram(metricValH[0][condition], bins)
            completeness = n_found.astype(float) / n_all.astype(float)
            completeness = np.where(n_all==0, 0, completeness)
        if self.cumulative:
            completenessInt = integrateOverH(completeness, Hvals, self.Hindex)
            summaryVal = np.empty(len(completenessInt), dtype=[('name', np.str_, 20), ('value', float)])
            summaryVal['value'] = completenessInt
            for i, Hval in enumerate(Hvals):
                summaryVal['name'][i] = 'H <= %f' % (Hval)
        else:
            summaryVal = np.empty(len(completeness), dtype=[('name', np.str_, 20), ('value', float)])
            summaryVal['value'] = completeness
            for i, Hval in enumerate(Hvals):
                summaryVal['name'][i] = 'H = %f' % (Hval)
        return summaryVal

class MoCompletenessAtTimeMetric(BaseMoMetric):
    """Calculate the completeness (relative to the entire population) <= a given H as a function of time,
    given the times of each discovery.

    Input values of the discovery times can come from the Discovery_Time (child) metric or the
    KnownObjects metric.

    Parameters
    ----------
    times : numpy.ndarray like
        The bins to distribute the discovery times into. Same units as the discovery time (typically MJD).
    Hval : float, opt
        The value of H to count completeness at (or cumulative completeness to).
        Default None, in which case a value halfway through Hvals (the slicer H range) will be chosen.
    cumulative : bool, opt
        If True, calculate the cumulative completeness (completeness <= H).
        If False, calculate the differential completeness (completeness @ H).
        Default None which becomes 'True' unless metricName starts with 'differential'.
    Hindex : float, opt
        Use Hindex as the power law to integrate over H, if cumulative is True. Default 0.3.
    """

    def __init__(self, times, Hval=None, cumulative=None, Hindex=0.33, **kwargs):
        self.Hval = Hval
        self.times = times
        self.Hindex = Hindex
        if cumulative is None:  # if metricName does not start with 'differential', then cumulative->True
            if 'metricName' not in kwargs:
                self.cumulative = True
                metricName = 'CumulativeCompleteness@Time@H=%.2f' % self.Hval
            else:  #  'metricName' in kwargs:
                metricName = kwargs.pop('metricName')
                if metricName.lower().startswith('differential'):
                    self.cumulative=False
                else:
                    self.cumulative=True
        else: # cumulative was set
            self.cumulative = cumulative
            if 'metricName' in kwargs:
                metricName = kwargs.pop('metricName')
                if metricName.lower().startswith('differential') and self.cumulative:
                    warnings.warn(f'Completeness metricName is {metricName} but cumulative is True')
            else:
                if self.cumulative:
                    metricName = 'CumulativeCompleteness@Time@H=%.2f' % self.Hval
                else:
                    metricName = 'DifferentialCompleteness@Time@H=%.2f' % self.Hval
        self._setLabels()
        super().__init__(metricName=metricName, units=self.units, **kwargs)

    def _setLabels(self):
        if self.Hval is not None:
            if self.cumulative:
                self.units = 'H <=%.1f' % (self.Hval)
            else:
                self.units = 'H = %.1f' % (self.Hval)
        else:
            self.units = 'H'

    def run(self, discoveryTimes, Hvals):
        if len(Hvals) != discoveryTimes.shape[1]:
            warnings.warn("This summary metric expects cloned H distribution. Cannot calculate summary.")
            return
        nSsos = discoveryTimes.shape[0]
        timesinH = discoveryTimes.swapaxes(0, 1)
        completenessH = np.empty([len(Hvals), len(self.times)], float)
        for i, H in enumerate(Hvals):
            n, b = np.histogram(timesinH[i].compressed(), bins=self.times)
            completenessH[i][0] = 0
            completenessH[i][1:] = n.cumsum()
        completenessH = completenessH / float(nSsos)
        completeness = completenessH.swapaxes(0, 1)
        if self.cumulative:
            for i, t in enumerate(self.times):
                completeness[i] = integrateOverH(completeness[i], Hvals)
        # To save the summary statistic, we must pick out a given H value.
        if self.Hval is None:
            Hidx = len(Hvals) // 2
            self.Hval = Hvals[Hidx]
            self._setLabels()
        else:
            Hidx = np.where(np.abs(Hvals - self.Hval) == np.abs(Hvals - self.Hval).min())[0][0]
            self.Hval = Hvals[Hidx]
            self._setLabels()
        summaryVal = np.empty(len(self.times), dtype=[('name', np.str_, 20), ('value', float)])
        summaryVal['value'] = completeness[:, Hidx]
        for i, time in enumerate(self.times):
            summaryVal['name'][i] = '%s @ %.2f' % (self.units, time)
        return summaryVal


