from builtins import zip
import numpy as np

from .baseMetric import BaseMetric

__all__ = ['BaseMoMetric', 'NObsMetric', 'NObsNoSinglesMetric',
           'NNightsMetric', 'ObsArcMetric',
           'DiscoveryMetric', 'Discovery_N_ChancesMetric', 'Discovery_N_ObsMetric',
           'Discovery_TimeMetric', 'Discovery_DistanceMetric',
           'Discovery_RADecMetric', 'Discovery_EcLonLatMetric',
           'Discovery_VelocityMetric',
           'ActivityOverTimeMetric', 'ActivityOverPeriodMetric',
           'MagicDiscoveryMetric',
           'HighVelocityMetric', 'HighVelocityNightsMetric',
           'LightcurveInversion_AsteroidMetric', 'Color_AsteroidMetric',
           'InstantaneousColorMetric', 'LightcurveColor_OuterMetric',
           'PeakVMagMetric', 'KnownObjectsMetric']


def _setVis(ssoObs, snrLimit, snrCol, visCol):
    if snrLimit is not None:
        vis = np.where(ssoObs[snrCol] >= snrLimit)[0]
    else:
        vis = np.where(ssoObs[visCol] > 0)[0]
    return vis


class BaseMoMetric(BaseMetric):
    """Base class for the moving object metrics.
    Intended to be used with the Moving Object Slicer."""

    def __init__(self, cols=None, metricName=None, units='#', badval=0,
                 comment=None, childMetrics=None,
                 appMagCol='appMag', appMagVCol='appMagV', m5Col='fiveSigmaDepth',
                 nightCol='night', mjdCol='observationStartMJD',
                 snrCol='SNR',  visCol='vis',
                 raCol='ra', decCol='dec', seeingCol='seeingFwhmGeom',
                 expTimeCol='visitExposureTime', filterCol='filter'):
        # Set metric name.
        self.name = metricName
        if self.name is None:
            self.name = self.__class__.__name__.replace('Metric', '', 1)
        # Set badval and units, leave space for 'comment' (tied to displayDict).
        self.badval = badval
        self.units = units
        self.comment = comment
        # Set some commonly used column names.
        self.m5Col = m5Col
        self.appMagCol = appMagCol
        self.appMagVCol = appMagVCol
        self.nightCol = nightCol
        self.mjdCol = mjdCol
        self.snrCol = snrCol
        self.visCol = visCol
        self.raCol = raCol
        self.decCol = decCol
        self.seeingCol = seeingCol
        self.expTimeCol = expTimeCol
        self.filterCol = filterCol
        self.colsReq = [self.appMagCol, self.m5Col,
                        self.nightCol, self.mjdCol,
                        self.snrCol, self.visCol]
        if cols is not None:
            for col in cols:
                self.colsReq.append(col)

        if childMetrics is None:
            try:
                if not isinstance(self.childMetrics, dict):
                    raise ValueError('self.childMetrics must be a dictionary (possibly empty)')
            except AttributeError:
                self.childMetrics = {}
                self.metricDtype = 'float'
        else:
            if not isinstance(childMetrics, dict):
                raise ValueError('childmetrics must be provided as a dictionary.')
            self.childMetrics = childMetrics
            self.metricDtype = 'object'
        self.shape = 1

    def run(self, ssoObs, orb, Hval):
        """Calculate the metric value.

        Parameters
        ----------
        ssoObs: np.ndarray
            The input data to the metric (same as the parent metric).
        orb: np.ndarray
            The information about the orbit for which the metric is being calculated.
        Hval : float
            The H value for which the metric is being calculated.

        Returns
        -------
        float or np.ndarray or dict
        """
        raise NotImplementedError


class BaseChildMetric(BaseMoMetric):
    """Base class for child metrics.

    Parameters
    ----------
    parentDiscoveryMetric: BaseMoMetric
        The 'parent' metric which generated the metric data used to calculate this 'child' metric.
    badval: float, opt
        Value to return when metric cannot be calculated.
    """
    def __init__(self, parentDiscoveryMetric, badval=0, **kwargs):
        super().__init__(badval=badval, **kwargs)
        self.parentMetric = parentDiscoveryMetric
        self.childMetrics = {}
        if 'metricDtype' in kwargs:
            self.metricDtype = kwargs['metricDtype']
        else:
            self.metricDtype = 'float'

    def run(self, ssoObs, orb, Hval, metricValues):
        """Calculate the child metric value.

        Parameters
        ----------
        ssoObs: np.ndarray
            The input data to the metric (same as the parent metric).
        orb: np.ndarray
            The information about the orbit for which the metric is being calculated.
        Hval : float
            The H value for which the metric is being calculated.
        metricValues : dict or np.ndarray
            The return value from the parent metric.

        Returns
        -------
        float
        """
        raise NotImplementedError


class NObsMetric(BaseMoMetric):
    """
    Count the total number of observations where an SSobject was 'visible'.
    """
    def __init__(self, snrLimit=None, **kwargs):
        """
        @ snrLimit .. if snrLimit is None, this uses the _calcVis method/completeness
                      if snrLimit is not None, this uses that value as a cutoff instead.
        """
        super().__init__(**kwargs)
        self.snrLimit = snrLimit

    def run(self, ssoObs, orb, Hval):
        if self.snrLimit is not None:
            vis = np.where(ssoObs[self.snrCol] >= self.snrLimit)[0]
            return vis.size
        else:
            vis = np.where(ssoObs[self.visCol] > 0)[0]
            return vis.size


class NObsNoSinglesMetric(BaseMoMetric):
    """
    Count the number of observations for an SSobject, without singles.
    Don't include any observations where it was a single observation on a night.
    """
    def __init__(self, snrLimit=None, **kwargs):
        super().__init__(**kwargs)
        self.snrLimit = snrLimit

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return 0
        nights = ssoObs[self.nightCol][vis]
        nights = nights.astype('int')
        ncounts = np.bincount(nights)
        nobs = ncounts[np.where(ncounts > 1)].sum()
        return nobs


class NNightsMetric(BaseMoMetric):
    """Count the number of distinct nights an SSobject is observed.
    """
    def __init__(self, snrLimit=None, **kwargs):
        """
        @ snrLimit : if SNRlimit is None, this uses _calcVis method/completeness
                     else if snrLimit is not None, it uses that value as a cutoff.
        """
        super().__init__(**kwargs)
        self.snrLimit = snrLimit

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return 0
        nights = len(np.unique(ssoObs[self.nightCol][vis]))
        return nights


class ObsArcMetric(BaseMoMetric):
    """Calculate the difference between the first and last observation of an SSobject.
    """
    def __init__(self, snrLimit=None, **kwargs):
        super().__init__(**kwargs)
        self.snrLimit = snrLimit

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return 0
        arc = ssoObs[self.mjdCol][vis].max() - ssoObs[self.mjdCol][vis].min()
        return arc


class DiscoveryMetric(BaseMoMetric):
    """Identify the discovery opportunities for an SSobject.

    Parameters
    ----------
    nObsPerNight : int, opt
        Number of observations required within a single night. Default 2.
    tMin : float, opt
        Minimum time span between observations in a single night, in days.
        Default 5 minutes (5/60/24).
    tMax : float, opt
        Maximum time span between observations in a single night, in days.
        Default 90 minutes.
    nNightsPerWindow : int, opt
        Number of nights required with observations, within the track window. Default 3.
    tWindow : int, opt
        Number of nights included in the track window. Default 15.
    snrLimit : None or float, opt
        SNR limit to use for observations. If snrLimit is None, (default), then it uses
        the completeness calculation added to the 'vis' column (probabilistic visibility,
        based on 5-sigma limit). If snrLimit is not None, it uses this SNR value as a cutoff.
    metricName : str, opt
        The metric name to use.
        Default will be to construct Discovery_nObsPerNightxnNightsPerWindowintWindow.
    """
    def __init__(self, nObsPerNight=2,
                 tMin=5./60.0/24.0, tMax=90./60./24.0,
                 nNightsPerWindow=3, tWindow=15,
                 snrLimit=None, badval=None, **kwargs):
        # Define anything needed by the child metrics first.
        self.snrLimit = snrLimit
        self.childMetrics = {'N_Chances': Discovery_N_ChancesMetric(self),
                             'N_Obs': Discovery_N_ObsMetric(self),
                             'Time': Discovery_TimeMetric(self),
                             'Distance': Discovery_DistanceMetric(self),
                             'RADec': Discovery_RADecMetric(self),
                             'EcLonLat': Discovery_EcLonLatMetric(self)}
        if 'metricName' in kwargs:
            metricName = kwargs.get('metricName')
            del kwargs['metricName']
        else:
            metricName = 'Discovery_%.0fx%.0fin%.0f' % (nObsPerNight, nNightsPerWindow, tWindow)
        # Set up for inheriting from __init__.
        super().__init__(metricName=metricName, childMetrics=self.childMetrics,
                                              badval=badval, **kwargs)
        # Define anything needed for this metric.
        self.nObsPerNight = nObsPerNight
        self.tMin = tMin
        self.tMax = tMax
        self.nNightsPerWindow = nNightsPerWindow
        self.tWindow = tWindow

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        # Identify discovery opportunities.
        #  Identify visits where the 'night' changes.
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        nights = ssoObs[self.nightCol][vis][visSort]
        #print 'all nights', nights
        n = np.unique(nights)
        # Identify all the indexes where the night changes in value.
        nIdx = np.searchsorted(nights, n)
        #print 'nightchanges', nights[nIdx]
        # Count the number of observations per night (except last night)
        obsPerNight = (nIdx - np.roll(nIdx, 1))[1:]
        # Add the number of observations on the last night.
        obsLastNight = np.array([len(nights) - nIdx[-1]])
        obsPerNight = np.concatenate((obsPerNight, obsLastNight))
        # Find the nights with more than nObsPerNight.
        nWithXObs = n[np.where(obsPerNight >= self.nObsPerNight)]
        nIdxMany = np.searchsorted(nights, nWithXObs)
        nIdxManyEnd = np.searchsorted(nights, nWithXObs, side='right') - 1
        # Check that nObsPerNight observations are within tMin/tMax
        timesStart = ssoObs[self.mjdCol][vis][visSort][nIdxMany]
        timesEnd = ssoObs[self.mjdCol][vis][visSort][nIdxManyEnd]
        # Identify the nights with 'clearly good' observations.
        good = np.where((timesEnd - timesStart >= self.tMin) & (timesEnd - timesStart <= self.tMax), 1, 0)
        # Identify the nights where we need more investigation
        # (a subset of the visits may be within the interval).
        check = np.where((good==0) & (nIdxManyEnd + 1 - nIdxMany > self.nObsPerNight)
                         & (timesEnd-timesStart > self.tMax))[0]
        for i, j, c in zip(visSort[nIdxMany][check], visSort[nIdxManyEnd][check], check):
            t = ssoObs[self.mjdCol][vis][visSort][i:j+1]
            dtimes = (np.roll(t, 1- self.nObsPerNight) - t)[:-1]
            tidx = np.where((dtimes >= self.tMin) & (dtimes <= self.tMax))[0]
            if len(tidx) > 0:
                good[c] = 1
        # 'good' provides mask for observations which could count as 'good to make tracklets'
        # against ssoObs[visSort][nIdxMany].  Now identify tracklets which can make tracks.
        goodIdx = visSort[nIdxMany][good == 1]
        goodIdxEnds = visSort[nIdxManyEnd][good == 1]
        #print 'good tracklets', nights[goodIdx]
        if len(goodIdx) < self.nNightsPerWindow:
            return self.badval
        deltaNights = np.roll(ssoObs[self.nightCol][vis][goodIdx], 1 - self.nNightsPerWindow) \
                      - ssoObs[self.nightCol][vis][goodIdx]
        # Identify the index in ssoObs[vis][goodIdx] (sorted by mjd) where the discovery opportunity starts.
        startIdxs = np.where((deltaNights >= 0) & (deltaNights <= self.tWindow))[0]
        # Identify the index where the discovery opportunity ends.
        endIdxs = np.zeros(len(startIdxs), dtype='int')
        for i, sIdx in enumerate(startIdxs):
            inWindow = np.where(ssoObs[self.nightCol][vis][goodIdx]
                                - ssoObs[self.nightCol][vis][goodIdx][sIdx] <= self.tWindow)[0]
            endIdxs[i] = np.array([inWindow.max()])
        # Convert back to index based on ssoObs[vis] (sorted by expMJD).
        startIdxs = goodIdx[startIdxs]
        endIdxs = goodIdxEnds[endIdxs]
        #print 'start', startIdxs,  nights[startIdxs]#, orb['objId'], Hval
        #print 'end', endIdxs, nights[endIdxs]#, orb['objId'], Hval
        return {'start':startIdxs, 'end':endIdxs, 'trackletNights':ssoObs[self.nightCol][vis][goodIdx]}


class Discovery_N_ChancesMetric(BaseChildMetric):
    """Calculate total number of discovery opportunities for an SSobject.

    Calculates total number of discovery opportunities between nightStart / nightEnd.
    Child metric to be used with the Discovery Metric.
    """
    def __init__(self, parentDiscoveryMetric, nightStart=None, nightEnd=None, badval=0, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        self.nightStart = nightStart
        self.nightEnd = nightEnd
        self.snrLimit = parentDiscoveryMetric.snrLimit
        # Update the metric name to use the nightStart/nightEnd values, if an overriding name is not given.
        if 'metricName' not in kwargs:
            if nightStart is not None:
                self.name = self.name + '_n%d' % (nightStart)
            if nightEnd is not None:
                self.name = self.name + '_n%d' % (nightEnd)

    def run(self, ssoObs, orb, Hval, metricValues):
        """Return the number of different discovery chances we had for each object/H combination.
        """
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        if self.nightStart is None and self.nightEnd is None:
            return len(metricValues['start'])
        # Otherwise, we have to sort out what night the discovery chances happened on.
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        nights = ssoObs[self.nightCol][vis][visSort]
        startNights = nights[metricValues['start']]
        endNights = nights[metricValues['end']]
        if self.nightEnd is None and self.nightStart is not None:
            valid = np.where(startNights >= self.nightStart)[0]
        elif self.nightStart is None and self.nightEnd is not None:
            valid = np.where(endNights <= self.nightEnd)[0]
        else:
            # And we only end up here if both were not None.
            valid = np.where((startNights >= self.nightStart) & (endNights <= self.nightEnd))[0]
        return len(valid)


class Discovery_N_ObsMetric(BaseChildMetric):
    """Calculates the number of observations in the i-th discovery track of an SSobject.
    """
    def __init__(self, parentDiscoveryMetric, i=0, badval=0, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        # The number of the discovery chance to use.
        self.i = i

    def run(self, ssoObs, orb, Hval, metricValues):
        if self.i >= len(metricValues['start']):
            return 0
        startIdx = metricValues['start'][self.i]
        endIdx = metricValues['end'][self.i]
        nobs = endIdx - startIdx
        return nobs


class Discovery_TimeMetric(BaseChildMetric):
    """Returns the time of the i-th discovery track of an SSobject.
    """
    def __init__(self, parentDiscoveryMetric, i=0, tStart=None, badval=-999, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        self.i = i
        self.tStart = tStart
        self.snrLimit = parentDiscoveryMetric.snrLimit

    def run(self, ssoObs, orb, Hval, metricValues):
        if self.i>=len(metricValues['start']):
            return self.badval
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        times = ssoObs[self.mjdCol][vis][visSort]
        startIdx = metricValues['start'][self.i]
        tDisc = times[startIdx]
        if self.tStart is not None:
            tDisc = tDisc - self.tStart
        return tDisc


class Discovery_DistanceMetric(BaseChildMetric):
    """Returns the distance of the i-th discovery track of an SSobject.
    """
    def __init__(self, parentDiscoveryMetric, i=0, distanceCol='geo_dist', badval=-999, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        self.i = i
        self.distanceCol = distanceCol
        self.snrLimit = parentDiscoveryMetric.snrLimit

    def run(self, ssoObs, orb, Hval, metricValues):
        if self.i>=len(metricValues['start']):
            return self.badval
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        dists = ssoObs[self.distanceCol][vis][visSort]
        startIdx = metricValues['start'][self.i]
        distDisc = dists[startIdx]
        return distDisc


class Discovery_RADecMetric(BaseChildMetric):
    """Returns the RA/Dec of the i-th discovery track of an SSobject.
    """
    def __init__(self, parentDiscoveryMetric, i=0, badval=None, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        self.i = i
        self.snrLimit = parentDiscoveryMetric.snrLimit
        self.metricDtype = 'object'

    def run(self, ssoObs, orb, Hval, metricValues):
        if self.i>=len(metricValues['start']):
            return self.badval
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        ra = ssoObs[self.raCol][vis][visSort]
        dec = ssoObs[self.decCol][vis][visSort]
        startIdx = metricValues['start'][self.i]
        return (ra[startIdx], dec[startIdx])


class Discovery_EcLonLatMetric(BaseChildMetric):
    """Returns the ecliptic lon/lat and solar elong of the i-th discovery track of an SSobject.
    """
    def __init__(self, parentDiscoveryMetric, i=0, badval=None, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        self.i = i
        self.snrLimit = parentDiscoveryMetric.snrLimit
        self.metricDtype = 'object'

    def run(self, ssoObs, orb, Hval, metricValues):
        if self.i>=len(metricValues['start']):
            return self.badval
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        ecLon = ssoObs['ecLon'][vis][visSort]
        ecLat = ssoObs['ecLat'][vis][visSort]
        solarElong = ssoObs['solarElong'][vis][visSort]
        startIdx = metricValues['start'][self.i]
        return (ecLon[startIdx], ecLat[startIdx], solarElong[startIdx])


class Discovery_VelocityMetric(BaseChildMetric):
    """Returns the sky velocity of the i-th discovery track of an SSobject.
    """
    def __init__(self, parentDiscoveryMetric, i=0, badval=-999, **kwargs):
        super().__init__(parentDiscoveryMetric, badval=badval, **kwargs)
        self.i = i
        self.snrLimit = parentDiscoveryMetric.snrLimit

    def run(self, ssoObs, orb, Hval, metricValues):
        if self.i>=len(metricValues['start']):
            return self.badval
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        visSort = np.argsort(ssoObs[self.mjdCol][vis])
        velocity = ssoObs['velocity'][vis][visSort]
        startIdx = metricValues['start'][self.i]
        return velocity[startIdx]

class ActivityOverTimeMetric(BaseMoMetric):
    """Count fraction of survey we could identify activity for an SSobject.

    Counts the time periods where we would have a chance to detect activity on
    a moving object.
    Splits observations into time periods set by 'window', then looks for observations within each window,
    and reports what fraction of the total windows receive 'nObs' visits.
    """
    def __init__(self, window, snrLimit=5, surveyYears=10.0, metricName=None, **kwargs):
        if metricName is None:
            metricName = 'Chance of detecting activity lasting %.0f days' %(window)
        super().__init__(metricName=metricName, **kwargs)
        self.snrLimit = snrLimit
        self.window = window
        self.surveyYears = surveyYears
        self.windowBins = np.arange(0, self.surveyYears*365 + self.window/2.0, self.window)
        self.nWindows = len(self.windowBins)
        self.units = '%.1f Day Windows' %(self.window)

    def run(self, ssoObs, orb,  Hval):
        # For cometary activity, expect activity at the same point in its orbit at the same time, mostly
        # For collisions, expect activity at random times
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        n, b = np.histogram(ssoObs[vis][self.nightCol], bins=self.windowBins)
        activityWindows = np.where(n>0)[0].size
        return activityWindows / float(self.nWindows)


class ActivityOverPeriodMetric(BaseMoMetric):
    """Count fraction of object period we could identify activity for an SSobject.

    Count the fraction of the orbit (when split into nBins) that receive
    observations, in order to have a chance to detect activity.
    """
    def __init__(self, binsize, snrLimit=5,
                 qCol='q', eCol='e', aCol='a', tPeriCol='tPeri', anomalyCol='meanAnomaly',
                 metricName=None, **kwargs):
        """
        @ binsize : size of orbit slice, in degrees.
        """
        if metricName is None:
            metricName = 'Chance of detecting activity covering %.1f of the orbit' %(binsize)
        super().__init__(metricName=metricName, **kwargs)
        self.qCol = qCol
        self.eCol = eCol
        self.aCol = aCol
        self.tPeriCol = tPeriCol
        self.anomalyCol = anomalyCol
        self.snrLimit = snrLimit
        self.binsize = np.radians(binsize)
        self.anomalyBins = np.arange(0, 2 * np.pi, self.binsize)
        self.anomalyBins = np.concatenate([self.anomalyBins, np.array([2 * np.pi])])
        self.nBins = len(self.anomalyBins) - 1
        self.units = '%.1f deg' %(np.degrees(self.binsize))

    def run(self, ssoObs, orb, Hval):
        # For cometary activity, expect activity at the same point in its orbit at the same time, mostly
        # For collisions, expect activity at random times
        if self.aCol in orb.keys():
            a = (orb[self.aCol])
        elif self.qCol in orb.keys():
            a = orb[self.qCol] / (1 - orb[self.eCol])
        else:
            return self.badval

        period = np.power(a, 3./2.) * 365.25  # days

        if self.anomalyCol in orb.keys():
            curranomaly = np.radians(orb[self.anomalyCol] + \
                          (ssoObs[self.mjdCol] - orb['epoch'])/ period * 360.0) % (2 * np.pi)
        elif self.tPeriCol in orb.keys():
            curranomaly = ((ssoObs[self.mjdCol] - orb[self.tPeriCol]) / period) % (2 * np.pi)
        else:
            return self.badval

        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        n, b = np.histogram(curranomaly[vis], bins=self.anomalyBins)
        activityWindows = np.where(n>0)[0].size
        return activityWindows / float(self.nBins)


class MagicDiscoveryMetric(BaseMoMetric):
    """Count the number of discovery opportunities with very good software for an SSobject.
    """
    def __init__(self, nObs=6, tWindow=60, snrLimit=None, **kwargs):
        """
        @ nObs = the total number of observations required for 'discovery'
        @ tWindow = the timespan of the discovery window.
        @ snrLimit .. if snrLimit is None then uses 'completeness' calculation,
                   .. if snrLimit is not None, then uses this value as a cutoff.
        """
        super().__init__(**kwargs)
        self.snrLimit = snrLimit
        self.nObs = nObs
        self.tWindow = tWindow
        self.badval = 0

    def run(self, ssoObs, orb, Hval):
        """SsoObs = Dataframe, orb=Dataframe, Hval=single number."""
        # Calculate visibility for this orbit at this H.
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        tNights = np.sort(ssoObs[self.nightCol][vis])
        deltaNights = np.roll(tNights, 1-self.nObs) - tNights
        nDisc = np.where((deltaNights < self.tWindow) & (deltaNights >= 0))[0].size
        return nDisc


class HighVelocityMetric(BaseMoMetric):
    """Count number of times an SSobject appears trailed.

    Count the number of times an asteroid is observed with a velocity high enough to make it appear
    trailed by a factor of (psfFactor)*PSF - i.e. velocity >= psfFactor * seeing / visitExpTime.
    Simply counts the total number of observations with high velocity.
    """
    def __init__(self, psfFactor=2.0,  snrLimit=None, velocityCol='velocity', **kwargs):
        """
        @ psfFactor = factor to multiply seeing/visitExpTime by
        (velocity(deg/day) >= 24*psfFactor*seeing(")/visitExptime(s))
        """
        super().__init__(**kwargs)
        self.velocityCol = velocityCol
        self.snrLimit = snrLimit
        self.psfFactor = psfFactor
        self.badval = 0

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        highVelocityObs = np.where(ssoObs[self.velocityCol][vis] >=
                                   (24.*  self.psfFactor * ssoObs[self.seeingCol][vis] /
                                    ssoObs[self.expTimeCol][vis]))[0]
        return highVelocityObs.size


class HighVelocityNightsMetric(BaseMoMetric):
    """Count the number of discovery opportunities (via trailing) for an SSobject.

    Determine the first time an asteroid is observed is observed with a velocity high enough to make
    it appear trailed by a factor of psfFactor*PSF with nObsPerNight observations within a given night.

    Parameters
    ----------
    psfFactor: float, opt
        Object velocity (deg/day) must be >= 24 * psfFactor * seeingGeom (") / visitExpTime (s).
        Default is 2 (i.e. object trailed over 2 psf's).
    nObsPerNight: int, opt
        Number of observations per night required. Default 2.
    snrLimit: float or None
        If snrLimit is set as a float, then requires object to be above snrLimit SNR in the image.
        If snrLimit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    velocityCol: str, opt
        Name of the velocity column in the obs file. Default 'velocity'. (note this is deg/day).

    Returns
    -------
    float
        The time of the first detection where the conditions are satisifed.
    """
    def __init__(self, psfFactor=2.0, nObsPerNight=2, snrLimit=None, velocityCol='velocity', **kwargs):
        super().__init__(**kwargs)
        self.velocityCol = velocityCol
        self.snrLimit = snrLimit
        self.psfFactor = psfFactor
        self.nObsPerNight = nObsPerNight

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return self.badval
        highVelocityObs = np.where(ssoObs[self.velocityCol][vis] >=
                                   (24. *  self.psfFactor * ssoObs[self.seeingCol][vis]
                                    / ssoObs[self.expTimeCol][vis]))[0]
        if len(highVelocityObs) == 0:
            return 0
        nights = ssoObs[self.nightCol][vis][highVelocityObs]
        n = np.unique(nights)
        nIdx = np.searchsorted(nights, n)
        # Count the number of observations per night (except last night)
        obsPerNight = (nIdx - np.roll(nIdx, 1))[1:]
        # Add the number of observations on the last night.
        obsLastNight = np.array([len(nights) - nIdx[-1]])
        obsPerNight = np.concatenate((obsPerNight, obsLastNight))
        # Find the nights with at least nObsPerNight visits
        # (this is already looking at only high velocity observations).
        nWithXObs = n[np.where(obsPerNight >= self.nObsPerNight)]
        if len(nWithXObs) > 0:
            found = ssoObs[np.where(ssoObs[self.nightCol] == nWithXObs[0])][self.mjdCol][0]
        else:
            found = self.badval
        return found


class LightcurveInversion_AsteroidMetric(BaseMoMetric):
    """
    This metric is generally applicable to NEOs and MBAs - inner solar system objects.

    Determine if the cumulative sum of observations of a target are enough to enable lightcurve
    inversion for shape modeling. For this to be true, multiple conditions need to be
    satisfied:

    1) The SNR-weighted number of observations (each observation is weighted by its SNR, up to a max of 100)
    must be larger than the threshhold weightDet (default 50)
    2) Ecliptic longitudinal coverage needs to be at least 90 degrees, and the absolute deviation
    needs to be at least 1/8th the longitudinal coverage.
    3) The phase angle coverage needs to span at least 5 degrees.

    For evaluation of condition 2, the median ecliptic longitude is subtracted from all longitudes,
    and the modulo 360 of those values is taken. This ensures that the wrap around 360 is handled
    correctly.

    For more information on the above conditions, please see
    https://docs.google.com/document/d/1GAriM7trpTS08uanjUF7PyKALB2JBTjVT7Y6R30i0-8/edit?usp=sharing
    Contributed by Steve Chesley, Wes Fraser, Josef Durech, and the inner solar system working group.

    Parameters
    ----------
    weightDet: float, opt
        The SNR-weighted number of detections required (per bandpass in any ONE of the filters in filterlist).
        Default 50.
    snrLimit: float or None, opt
        If snrLimit is set as a float, then requires object to be above snrLimit SNR in the image.
        If snrLimit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    snrMax: float, opt
        Maximum value toward the SNR-weighting to consider. Default 100.
    filterlist: list of str, opt
        The filters which the lightcurve inversion could be based on. Requirements must be met in one of
        these filters.

    Returns
    -------
    int
        0 (could not perform lightcurve inversion) or 1 (could)
    """

    def __init__(self, weightDet=50, snrLimit=None, snrMax=100,
                 filterlist=('u', 'g', 'r', 'i', 'z', 'y'), **kwargs):
        super().__init__(**kwargs)
        self.snrLimit = snrLimit
        self.snrMax = snrMax
        self.weightDet = weightDet
        self.filterlist = filterlist

    def run(self, ssoObs, orb, Hval):
        # Calculate the clipped SNR - ranges from snrLimit / SNR+vis to snrMax.
        clipSnr = np.minimum(ssoObs[self.snrCol], self.snrMax)
        if self.snrLimit is not None:
            clipSnr = np.where(ssoObs[self.snrCol] <= self.snrLimit, 0, clipSnr)
        else:
            clipSnr = np.where(ssoObs[self.visCol] == 0, 0, clipSnr)
        if len(np.where(clipSnr > 0)[0]) == 0:
            return 0
        # Check each filter in filterlist:
        # stop as soon as find a filter that matches requirements.
        inversion_possible = 0
        for f in self.filterlist:
            # Is the SNR-weight sum of observations in this filter high enough?
            match = np.where(ssoObs[self.filterCol] == f)
            snrSum = np.sum(clipSnr[match]) / self.snrMax
            if snrSum < self.weightDet:
                # Do not have enough SNR-weighted observations, so skip on to the next filter.
                continue
            # Is the ecliptic longitude coverage for the visible observations sufficient?
            # Is the phase coverage sufficient?
            vis = np.where(clipSnr[match] > 0)
            ecL = ssoObs['ecLon'][match][vis]
            phaseAngle = ssoObs['phase'][match][vis]
            # Calculate the absolute deviation and range of ecliptic longitude.
            ecL_centred = (ecL - np.median(ecL)) % 360.0
            aDev = np.sum(np.abs(ecL_centred - np.mean(ecL_centred))) / len(ecL_centred)
            dL = np.max(ecL) - np.min(ecL)
            # Calculate the range of the phase angle
            dp = np.max(phaseAngle) - np.min(phaseAngle)
            # Metric requirement is that dL >= 90 deg, absolute deviation is greater than dL/8
            # and then that the phase coverage is more than 5 degrees.
            # Stop as soon as find a case where this is true.
            if dL >= 90.0 and aDev >= dL / 8 and dp >= 5:
                inversion_possible += 1
                break
        return inversion_possible


class Color_AsteroidMetric(BaseMoMetric):
    """
    This metric is appropriate for MBAs and NEOs, and other inner solar system objects.

    The metric evaluates if the SNR-weighted number of observations are enough to
    determine an approximate lightcurve and phase function -- and from this,
    then a color for the asteroid can be determined.
    The assumption is that you must fit the lightcurve/phase function in each bandpass,
    and could do this well-enough if you have at least weightDet SNR-weighted observations
    in the bandpass.
    e.g. to find a g-r color, you must have 10 (SNR-weighted) obs in g and 10 in r.

    For more details, see
    https://docs.google.com/document/d/1GAriM7trpTS08uanjUF7PyKALB2JBTjVT7Y6R30i0-8/edit?usp=sharing
    Contributed by Wes Fraser, Steven Chesley & the inner solar system working group.

    Parameters
    ----------
    weightDet: float, opt
        The SNR-weighted number of detections required (per bandpass in any ONE of the filters in filterlist).
        Default 10.
    snrLimit: float or None, opt
        If snrLimit is set as a float, then requires object to be above snrLimit SNR in the image.
        If snrLimit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    snrMax: float, opt
        Maximum value toward the SNR-weighting to consider. Default 20.

    Returns
    -------
    int
        An integer 'flag' that indicates whether the mean magnitude (and thus a color) was determined in:
        0 = no bands
        1 = g and (r or i) and (z or y). i.e. obtain colors g-r or g-i PLUS g-z or g-y
        2 = Any 4 different filters (from grizy). i.e. colors = g-r, r-i, i-z, OR r-i, i-z, z-y..
        3 = All 5 from grizy. i.e. colors g-r, r-i, i-z, z-y.
        4 = All 6 filters (ugrizy) -- best possible! add u-g.
    """

    def __init__(self, weightDet=10, snrMax=20, snrLimit=None, **kwargs):
        super().__init__(**kwargs)
        self.weightDet = weightDet
        self.snrLimit = snrLimit
        self.snrMax = snrMax
        self.filterlist = ('u', 'g', 'r', 'i', 'z', 'y')

    def run(self, ssoObs, orb, Hval):
        clipSnr = np.minimum(ssoObs[self.snrCol], self.snrMax)
        if self.snrLimit is not None:
            clipSnr = np.where(ssoObs[self.snrCol] <= self.snrLimit, 0, clipSnr)
        else:
            clipSnr = np.where(ssoObs[self.visCol] == 0, 0, clipSnr)
        if len(np.where(clipSnr > 0)[0]) == 0:
            return self.badval

        # Evaluate SNR-weighted number of observations in each filter.
        filterWeight = {}
        for f in self.filterlist:
            match = np.where(ssoObs[self.filterCol] == f)
            snrweight = np.sum(clipSnr[match]) / self.snrMax
            # If the snrweight exceeds the weightDet, add it to the dictionary.
            if snrweight > self.weightDet:
                filterWeight[f] = snrweight

        # Now assign a flag:
        # 0 = no bands
        # 1 = g and (r or i) and (z or y). i.e. obtain colors g-r or g-i PLUS g-z or g-y
        # 2 = Any 4 different filters (from grizy). i.e. colors = g-r, r-i, i-z, OR r-i, i-z, z-y..
        # 3 = All 5 from grizy. i.e. colors g-r, r-i, i-z, z-y.
        # 4 = All 6 filters (ugrizy) -- best possible! add u-g.
        all_six = set(self.filterlist)
        good_five = set(['g', 'r', 'i', 'z', 'y'])

        if len(filterWeight) == 0:  # this lets us stop evaluating here if possible.
            flag = 0
        elif all_six.intersection(filterWeight) == all_six:
            flag = 4
        elif good_five.intersection(filterWeight) == good_five:
            flag = 3
        elif len(good_five.intersection(filterWeight)) == 4:
            flag = 2
        elif 'g' in filterWeight:
            # Have 'g' - do we have (r or i) and (z or y)
            if ('r' in filterWeight or 'i' in filterWeight) and ('z' in filterWeight or 'y' in filterWeight):
                flag = 1
            else:
                flag = 0
        else:
            flag = 0

        return flag


class LightcurveColor_OuterMetric(BaseMoMetric):
    """
    This metric is appropriate for outer solar system objects, such as TNOs and SDOs.

    This metric evaluates whether the number of observations is sufficient to fit a lightcurve
    in a primary and secondary bandpass. The primary bandpass requires more observations than
    the secondary. Essentially, it's a complete lightcurve in one or both bandpasses, with at
    least a semi-complete lightcurve in the secondary band.

    The lightcurve/color can be calculated with any two of the bandpasses in filterlist.
    Contributed by Wes Fraser.

    Parameters
    ----------
    snrLimit: float or None, opt
        If snrLimit is set as a float, then requires object to be above snrLimit SNR in the image.
        If snrLimit is None, this uses the probabilistic 'visibility' calculated by the vis stacker,
        which means SNR ~ 5.   Default is None.
    numReq: int, opt
        Number of observations required for a lightcurve fitting. Default 30.
    numSecFilt: int, opt
        Number of observations required in a secondary band for color only. Default 20.
    filterlist: list of str, opt
        Filters that the primary/secondary measurements can be in.

    Returns
    -------
    int
        A flag that indicates whether a color/lightcurve was generated in:
        0 = no lightcurve (although may have had 'color' in one or more band)
        1 = a lightcurve in a single filter (but no additional color information)
        2+ = lightcurves in more than one filter  (or lightcurve + color)
        e.g. lightcurve in 2 bands, with additional color information in another = 3.
    """

    def __init__(self, snrLimit=None, numReq=30, numSecFilt=20,
                 filterlist=('u', 'g', 'r', 'i', 'z', 'y'), **kwargs):
        super().__init__(**kwargs)
        self.snrLimit = snrLimit
        self.numReq = numReq
        self.numSecFilt = numSecFilt
        self.filterlist = filterlist

    def run(self, ssoObs, orb, Hval):
        vis = _setVis(ssoObs, self.snrLimit, self.snrCol, self.visCol)
        if len(vis) == 0:
            return 0

        lightcurves = set()
        colors = set()
        for f in self.filterlist:
            nmatch = np.where(ssoObs[vis][self.filterCol] == f)[0]
            if len(nmatch) >= self.numReq:
                lightcurves.add(f)
            if len(nmatch) >= self.numSecFilt:
                colors.add(f)

        # Set the flags - first the number of filters with lightcurves.
        flag = len(lightcurves)
        # And check if there were extra filters which had enough for a color
        # but not enough for a full lightcurve.
        if len(colors.difference(lightcurves)) > 0:
            # If there was no lightcurve available to match against:
            if len(lightcurves) == 0:
                flag = 0
            else:
                # We had a lightcurve and now can add a color.
                flag += 1
        return flag


class InstantaneousColorMetric(BaseMoMetric):
    """Identify SSobjects which could have observations suitable to determine colors.

    Generally, this is not the mode LSST would work in - the lightcurves of the objects
    mean that the time interval would have to be quite short.

    This is roughly defined as objects which have more than nPairs pairs of observations
    with SNR greater than snrLimit, in bands bandOne and bandTwo, within nHours.

    Parameters
    ----------
    nPairs: int, opt
        The number of pairs of observations (in each band) that must be within nHours
        Default 1
    snrLimit: float, opt
        The SNR limit for the observations. Default 10.
    nHours: float, opt
        The time interval between observations in the two bandpasses (hours). Default 0.5 hours.
    bOne: str, opt
        The first bandpass for the color. Default 'g'.
    bTwo: str, opt
        The second bandpass for the color. Default 'r'.

    Returns
    -------
    int
        0 (no color possible under these constraints) or 1 (color possible).
    """
    def __init__(self, nPairs=1, snrLimit=10, nHours=0.5, bOne='g', bTwo='r', **kwargs):
        super().__init__(**kwargs)
        self.nPairs = nPairs
        self.snrLimit = snrLimit
        self.nHours = nHours
        self.bOne = bOne
        self.bTwo = bTwo
        self.badval = -666

    def run(self, ssoObs, orb, Hval):
        vis = np.where(ssoObs[self.snrCol] >= self.snrLimit)[0]
        if len(vis) < self.nPairs * 2:
            return 0
        bOneObs = np.where(ssoObs[self.filterCol][vis] == self.bOne)[0]
        bTwoObs = np.where(ssoObs[self.filterCol][vis] == self.bTwo)[0]
        timesbOne = ssoObs[self.mjdCol][vis][bOneObs]
        timesbTwo = ssoObs[self.mjdCol][vis][bTwoObs]
        if len(timesbOne) == 0 or len(timesbTwo) == 0:
            return 0
        dTime = self.nHours / 24.0
        # Calculate the time between the closest pairs of observations.
        inOrder = np.searchsorted(timesbOne, timesbTwo, 'right')
        inOrder = np.where(inOrder - 1 > 0, inOrder - 1, 0)
        dtPairs = timesbTwo - timesbOne[inOrder]
        if len(np.where(dtPairs < dTime)[0]) >= self.nPairs:
            found = 1
        else:
            found = 0
        return found


class PeakVMagMetric(BaseMoMetric):
    """Pull out the peak V magnitude of all observations of the SSobject.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, ssoObs, orb, Hval):
        peakVmag = np.min(ssoObs[self.appMagVCol])
        return peakVmag


class KnownObjectsMetric(BaseMoMetric):
    """Identify SSobjects which could be classified as 'previously known' based on their peak V magnitude.
    This is most appropriate for NEO surveys, where most of the sky has been covered so the exact location
    (beyond being in the visible sky) is not as important.

    Default parameters tuned to match NEO survey capabilities.
    Returns the time at which each first reached that threshold V magnitude.
    The default values are calibrated using the NEOs larger than 140m discovered in the last 20 years
    and assuming a 30% completeness in 2017.

    Parameters
    -----------
    elongThresh : float, opt
        The cutoff in solar elongation to consider an object 'visible'. Default 100 deg.
    vMagThresh1 : float, opt
        The magnitude threshold for previously known objects. Default 20.0.
    eff1 : float, opt
        The likelihood of actually achieving each individual input observation.
        If the input observations include one observation per day, an 'eff' value of 0.3 would
        mean that (on average) only one third of these observations would be achieved.
        This is similar to the level for LSST, which can cover the visible sky every 3-4 days.
        Default 0.1
    tSwitch1 : float, opt
        The (MJD) time to switch between vMagThresh1 + eff1 to vMagThresh2 + eff2, e.g.
        the end of the first period.
        Default 53371 (2005).
    vMagThresh2 : float, opt
        The magnitude threshhold for previously known objects. Default 22.0.
        This is based on assuming PS and other surveys will be efficient down to V=22.
    eff2 : float, opt
        The efficiency of observations during the second period of time. Default 0.1
    tSwitch2 : float, opt
        The (MJD) time to switch between vMagThresh2 + eff2 to vMagThresh3 + eff3.
        Default 57023 (2015).
    vMagThresh3 : float, opt
        The magnitude threshold during the third period. Default 22.0, based on PS1 + Catalina.
    eff3 : float, opt
        The efficiency of observations during the third period. Default 0.1
    tSwitch3 : float, opt
        The (MJD) time to switch between vMagThresh3 + eff3 to vMagThresh4 + eff4.
        Default 59580 (2022).
    vMagThresh4 : float, opt
        The magnitude threshhold during the fourth (last) period. Default 22.0, based on PS1 + Catalina.
    eff4 : float, opt
        The efficiency of observations during the fourth (last) period. Default 0.2
    """
    def __init__(self, elongThresh=100., vMagThresh1=20.0, eff1=0.1, tSwitch1=53371,
                 vMagThresh2=21.5, eff2=0.1, tSwitch2=57023,
                 vMagThresh3=22.0, eff3=0.1, tSwitch3=59580,
                 vMagThresh4=22.0, eff4=0.2,
                 elongCol='Elongation', mjdCol='MJD(UTC)', **kwargs):
        super().__init__(**kwargs)
        self.elongThresh = elongThresh
        self.elongCol = elongCol
        self.vMagThresh1 = vMagThresh1
        self.eff1 = eff1
        self.tSwitch1 = tSwitch1
        self.vMagThresh2 = vMagThresh2
        self.eff2 = eff2
        self.tSwitch2 = tSwitch2
        self.vMagThresh3 = vMagThresh3
        self.eff3 = eff3
        self.tSwitch3 = tSwitch3
        self.vMagThresh4 = vMagThresh4
        self.eff4 = eff4
        self.mjdCol = mjdCol
        self.badval = int(tSwitch3) + 365*1000

    def _pickObs(self, potentialObsTimes, eff):
        # From a set of potential observations, apply an efficiency
        # And return the minimum time (if any)
        randPick = np.random.rand(len(potentialObsTimes))
        picked = np.where(randPick <= eff)[0]
        if len(picked) > 0:
            discTime = potentialObsTimes[picked].min()
        else:
            discTime = None
        return discTime

    def run(self, ssoObs, orb, Hval):
        visible = np.where(ssoObs[self.elongCol] >= self.elongThresh, 1, 0)
        discoveryTime = None
        # Look for discovery in any of the three periods.
        # First period.
        obs1 = np.where((ssoObs[self.mjdCol] < self.tSwitch1) & visible)[0]
        overPeak = np.where(ssoObs[self.appMagVCol][obs1] <= self.vMagThresh1)[0]
        if len(overPeak) > 0:
            discoveryTime = self._pickObs(ssoObs[self.mjdCol][obs1][overPeak], self.eff1)
        # Second period.
        if discoveryTime is None:
            obs2 = np.where((ssoObs[self.mjdCol] >= self.tSwitch1) &
                            (ssoObs[self.mjdCol] < self.tSwitch2) & visible)[0]
            overPeak = np.where(ssoObs[self.appMagVCol][obs2] <= self.vMagThresh2)[0]
            if len(overPeak) > 0:
                discoveryTime = self._pickObs(ssoObs[self.mjdCol][obs2][overPeak], self.eff2)
        # Third period.
        if discoveryTime is None:
            obs3 = np.where((ssoObs[self.mjdCol] >= self.tSwitch2) &
                            (ssoObs[self.mjdCol] < self.tSwitch3) & visible)[0]
            overPeak = np.where(ssoObs[self.appMagVCol][obs3] <= self.vMagThresh3)[0]
            if len(overPeak) > 0:
                discoveryTime = self._pickObs(ssoObs[self.mjdCol][obs3][overPeak], self.eff3)
        # Fourth period.
        if discoveryTime is None:
            obs4 = np.where((ssoObs[self.mjdCol] >= self.tSwitch3) & visible)[0]
            overPeak = np.where(ssoObs[self.appMagVCol][obs4] <= self.vMagThresh4)[0]
            if len(overPeak) > 0:
                discoveryTime = self._pickObs(ssoObs[self.mjdCol][obs4][overPeak], self.eff4)
        if discoveryTime is None:
            discoveryTime = self.badval
        return discoveryTime
