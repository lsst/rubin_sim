import numpy as np
from .baseMetric import BaseMetric

__all__ = ['TemplateExistsMetric', 'UniformityMetric',
           'RapidRevisitUniformityMetric', 'RapidRevisitMetric','NRevisitsMetric', 'IntraNightGapsMetric',
           'InterNightGapsMetric', 'VisitGapMetric']


class fSMetric(BaseMetric):
    """Calculate the fS value (Nvisit-weighted delta(M5-M5srd)).
    """
    def __init__(self, filterCol='filter', metricName='fS', **kwargs):
        self.filterCol = filterCol
        cols = [self.filterCol]
        super().__init__(cols=cols, metricName=metricName, units='fS', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """"Calculate the fS (reserve above/below the m5 values from the LSST throughputs)

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
            The fS value.
        """
        # We could import this from the m5_flat_sed values, but it makes sense to calculate the m5
        # directly from the throughputs. This is easy enough to do and will allow variation of
        # the throughput curves and readnoise and visit length, etc.
        pass


class TemplateExistsMetric(BaseMetric):
    """Calculate the fraction of images with a previous template image of desired quality.
    """
    def __init__(self, seeingCol='seeingFwhmGeom', observationStartMJDCol='observationStartMJD',
                 metricName='TemplateExistsMetric', **kwargs):
        cols = [seeingCol, observationStartMJDCol]
        super(TemplateExistsMetric, self).__init__(col=cols, metricName=metricName,
                                                   units='fraction', **kwargs)
        self.seeingCol = seeingCol
        self.observationStartMJDCol = observationStartMJDCol

    def run(self, dataSlice, slicePoint=None):
        """"Calculate the fraction of images with a previous template image of desired quality.

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
            The fraction of images with a 'good' previous template image.
        """
        # Check that data is sorted in observationStartMJD order
        dataSlice.sort(order=self.observationStartMJDCol)
        # Find the minimum seeing up to a given time
        seeing_mins = np.minimum.accumulate(dataSlice[self.seeingCol])
        # Find the difference between the seeing and the minimum seeing at the previous visit
        seeing_diff = dataSlice[self.seeingCol] - np.roll(seeing_mins, 1)
        # First image never has a template; check how many others do
        good = np.where(seeing_diff[1:] >= 0.)[0]
        frac = (good.size) / float(dataSlice[self.seeingCol].size)
        return frac


class UniformityMetric(BaseMetric):
    """Calculate how uniformly the observations are spaced in time.
    Returns a value between -1 and 1.
    A value of zero means the observations are perfectly uniform.

    Parameters
    ----------
    surveyLength : float, optional
        The overall duration of the survey. Default 10.
    """
    def __init__(self, mjdCol='observationStartMJD', units='',
                 surveyLength=10., **kwargs):
        """surveyLength = time span of survey (years) """
        self.mjdCol = mjdCol
        super(UniformityMetric, self).__init__(col=self.mjdCol, units=units, **kwargs)
        self.surveyLength = surveyLength

    def run(self, dataSlice, slicePoint=None):
        """"Calculate the survey uniformity.

        This is based on how a KS-test works: look at the cumulative distribution of observation dates,
        and compare to a perfectly uniform cumulative distribution.
        Perfectly uniform observations = 0, perfectly non-uniform = 1.

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
            Uniformity of 'observationStartMJDCol'.
        """
        # If only one observation, there is no uniformity
        if dataSlice[self.mjdCol].size == 1:
            return 1
        # Scale dates to lie between 0 and 1, where 0 is the first observation date and 1 is surveyLength
        dates = (dataSlice[self.mjdCol] - dataSlice[self.mjdCol].min()) / \
                (self.surveyLength * 365.25)
        dates.sort()  # Just to be sure
        n_cum = np.arange(1, dates.size + 1) / float(dates.size)
        D_max = np.max(np.abs(n_cum - dates - dates[1]))
        return D_max


class RapidRevisitUniformityMetric(BaseMetric):
    """Calculate uniformity of time between consecutive visits on short timescales (for RAV1).

    Parameters
    ----------
    mjdCol : str, optional
        The column containing the 'time' value. Default observationStartMJD.
    minNvisits : int, optional
        The minimum number of visits required within the time interval (dTmin to dTmax).
        Default 100.
    dTmin : float, optional
        The minimum dTime to consider (in days). Default 40 seconds.
    dTmax : float, optional
        The maximum dTime to consider (in days). Default 30 minutes.
    """
    def __init__(self, mjdCol='observationStartMJD', minNvisits=100,
                 dTmin=40.0 / 60.0 / 60.0 / 24.0, dTmax=30.0 / 60.0 / 24.0,
                 metricName='RapidRevisitUniformity', **kwargs):
        self.mjdCol = mjdCol
        self.minNvisits = minNvisits
        self.dTmin = dTmin
        self.dTmax = dTmax
        super().__init__(col=self.mjdCol, metricName=metricName, **kwargs)
        # Update minNvisits, as 0 visits will crash algorithm and 1 is nonuniform by definition.
        if self.minNvisits <= 1:
            self.minNvisits = 2

    def run(self, dataSlice, slicePoint=None):
        """Calculate the uniformity of visits within dTmin to dTmax.

        Uses a the same 'uniformity' calculation as the UniformityMetric, based on the KS-test.
        A value of 0 is perfectly uniform; a value of 1 is purely non-uniform.

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
           The uniformity measurement of the visits within time interval dTmin to dTmax.
        """
        # Calculate consecutive visit time intervals
        dtimes = np.diff(np.sort(dataSlice[self.mjdCol]))
        # Identify dtimes within interval from dTmin/dTmax.
        good = np.where((dtimes >= self.dTmin) & (dtimes <= self.dTmax))[0]
        # If there are not enough visits in this time range, return bad value.
        if good.size < self.minNvisits:
            return self.badval
        # Throw out dtimes outside desired range, and sort, then scale to 0-1.
        dtimes = np.sort(dtimes[good])
        dtimes = (dtimes - dtimes.min()) / float(self.dTmax - self.dTmin)
        # Set up a uniform distribution between 0-1 (to match dtimes).
        uniform_dtimes = np.arange(1, dtimes.size + 1, 1) / float(dtimes.size)
        # Look at the differences between our times and the uniform times.
        dmax = np.max(np.abs(uniform_dtimes - dtimes - dtimes[1]))
        return dmax


class RapidRevisitMetric(BaseMetric):
    def __init__(self, mjdCol='observationStartMJD', metricName='RapidRevisit',
                 dTmin=40.0 / 60.0 / 60.0 / 24.0, dTpairs = 20.0 / 60.0 / 24.0,
                 dTmax = 30.0 / 60.0 / 24.0, minN1 = 28, minN2 = 82, **kwargs):
        self.mjdCol = mjdCol
        self.dTmin = dTmin
        self.dTpairs = dTpairs
        self.dTmax = dTmax
        self.minN1 = minN1
        self.minN2 = minN2
        super().__init__(col=self.mjdCol, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        dtimes = np.diff(np.sort(dataSlice[self.mjdCol]))
        N1 = len(np.where((dtimes >= self.dTmin) & (dtimes <= self.dTpairs))[0])
        N2 = len(np.where((dtimes >= self.dTmin) & (dtimes <= self.dTmax))[0])
        if (N1 >= self.minN1) and (N2 >= self.minN2):
            val = 1
        else:
            val = 0
        return val


class NRevisitsMetric(BaseMetric):
    """Calculate the number of consecutive visits with time differences less than dT.

    Parameters
    ----------
    dT : float, optional
       The time interval to consider (in minutes). Default 30.
    normed : bool, optional
       Flag to indicate whether to return the total number of consecutive visits with time
       differences less than dT (False), or the fraction of overall visits (True).
       Note that we would expect (if all visits occur in pairs within dT) this fraction would be 0.5!
    """
    def __init__(self, mjdCol='observationStartMJD', dT=30.0, normed=False, metricName=None, **kwargs):
        units = ''
        if metricName is None:
            if normed:
                metricName = 'Fraction of revisits faster than %.1f minutes' % (dT)
            else:
                metricName = 'Number of revisits faster than %.1f minutes' % (dT)
                units = '#'
        self.mjdCol = mjdCol
        self.dT = dT / 60. / 24.  # convert to days
        self.normed = normed
        super(NRevisitsMetric, self).__init__(col=self.mjdCol, units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """Count the number of consecutive visits occuring within time intervals dT.

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
           Either the total number of consecutive visits within dT or the fraction compared to overall visits.
        """
        dtimes = np.diff(np.sort(dataSlice[self.mjdCol]))
        nFastRevisits = np.size(np.where(dtimes <= self.dT)[0])
        if self.normed:
            nFastRevisits = nFastRevisits / float(np.size(dataSlice[self.mjdCol]))
        return nFastRevisits


class IntraNightGapsMetric(BaseMetric):
    """
    Calculate the gap between consecutive observations within a night, in hours.

    Parameters
    ----------
    reduceFunc : function, optional
        Function that can operate on array-like structures. Typically numpy function.
        Default np.median.
    """

    def __init__(self, mjdCol='observationStartMJD', nightCol='night', reduceFunc=np.median,
                 metricName='Median Intra-Night Gap', **kwargs):
        units = 'hours'
        self.mjdCol = mjdCol
        self.nightCol = nightCol
        self.reduceFunc = reduceFunc
        super(IntraNightGapsMetric, self).__init__(col=[self.mjdCol, self.nightCol],
                                                   units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """Calculate the (reduceFunc) of the gap between consecutive obervations within a night.

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
           The (reduceFunc) value of the gap, in hours.
        """
        dataSlice.sort(order=self.mjdCol)
        dt = np.diff(dataSlice[self.mjdCol])
        dn = np.diff(dataSlice[self.nightCol])

        good = np.where(dn == 0)
        if np.size(good[0]) == 0:
            result = self.badval
        else:
            result = self.reduceFunc(dt[good]) * 24
        return result


class InterNightGapsMetric(BaseMetric):
    """
    Calculate the gap between consecutive observations in different nights, in days.

    Parameters
    ----------
    reduceFunc : function, optional
       Function that can operate on array-like structures. Typically numpy function.
       Default np.median.
    """
    def __init__(self, mjdCol='observationStartMJD', nightCol='night', reduceFunc=np.median,
                 metricName='Median Inter-Night Gap', **kwargs):
        units = 'days'
        self.mjdCol = mjdCol
        self.nightCol = nightCol
        self.reduceFunc = reduceFunc
        super(InterNightGapsMetric, self).__init__(col=[self.mjdCol, self.nightCol],
                                                   units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """Calculate the (reduceFunc) of the gap between consecutive nights of observations.
        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
            The (reduceFunc) of the gap between consecutive nights of observations, in days.
        """
        dataSlice.sort(order=self.mjdCol)
        unights = np.unique(dataSlice[self.nightCol])
        if np.size(unights) < 2:
            result = self.badval
        else:
            # Find the first and last observation of each night
            firstOfNight = np.searchsorted(dataSlice[self.nightCol], unights)
            lastOfNight = np.searchsorted(dataSlice[self.nightCol], unights, side='right') - 1
            diff = dataSlice[self.mjdCol][firstOfNight[1:]] - dataSlice[self.mjdCol][lastOfNight[:-1]]
            result = self.reduceFunc(diff)
        return result


class VisitGapMetric(BaseMetric):
    """
    Calculate the gap between any consecutive observations, in hours, regardless of night boundaries.

    Parameters
    ----------
    reduceFunc : function, optional
       Function that can operate on array-like structures. Typically numpy function.
       Default np.median.
    """
    def __init__(self, mjdCol='observationStartMJD', nightCol='night', reduceFunc=np.median,
                 metricName='VisitGap', **kwargs):
        units = 'hours'
        self.mjdCol = mjdCol
        self.nightCol = nightCol
        self.reduceFunc = reduceFunc
        super().__init__(col=[self.mjdCol, self.nightCol],
                         units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """Calculate the (reduceFunc) of the gap between consecutive observations.

        Different from inter-night and intra-night gaps, between this is really just counting
        all of the times between consecutive observations (not time between nights or time within a night).

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
           The (reduceFunc) of the time between consecutive observations, in hours.
        """
        dataSlice.sort(order=self.mjdCol)
        diff = np.diff(dataSlice[self.mjdCol])
        result = self.reduceFunc(diff) * 24.
        return result
