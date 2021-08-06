"""A group of metrics that work together to evaluate season characteristics (length, number, etc).
In addition, these supports the time delay metric calculation for strong lensing.
"""

import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.photUtils import Dust_values
from rubin_sim.utils import calcSeason

__all__ = ['findSeasonEdges',
           'SeasonLengthMetric', 'CampaignLengthMetric',
           'MeanCampaignFrequencyMetric', 'TdcMetric']


def findSeasonEdges(seasons):
    """Given the seasons, return the indexes of each start/end of the season.

    Parameters
    ----------
    seasons: np.ndarray
        Seasons, such as calculated by calcSeason.
        Note that seasons should be sorted!!

    Returns
    -------
    np.ndarray, np.ndarray
        The indexes of the first and last date in the season.
    """
    intSeasons = np.floor(seasons)
    # Get the unique seasons, so that we can separate each one
    season_list = np.unique(intSeasons)
    # Find the first and last observation of each season.
    firstOfSeason = np.searchsorted(intSeasons, season_list)
    lastOfSeason = np.searchsorted(intSeasons, season_list, side='right') - 1
    return firstOfSeason, lastOfSeason


class SeasonLengthMetric(BaseMetric):
    """
    Calculate the length of LSST seasons, in days.

    Parameters
    ----------
    minExpTime: float, optional
        Minimum visit exposure time to count for a 'visit', in seconds. Default 20.
    reduceFunc : function, optional
       Function that can operate on array-like structures. Typically numpy function.
       This reduces the season length in each season from 10 separate values to a single value.
       Default np.median.
    """
    def __init__(self, mjdCol='observationStartMJD', expTimeCol='visitExposureTime', minExpTime=20,
                 reduceFunc=np.median, metricName='SeasonLength', **kwargs):
        units = 'days'
        self.mjdCol = mjdCol
        self.expTimeCol = expTimeCol
        self.minExpTime = minExpTime
        self.reduceFunc = reduceFunc
        super().__init__(col=[self.mjdCol, self.expTimeCol],
                         units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint):
        """Calculate the (reduceFunc) of the length of each season.
        Uses the slicePoint RA/Dec to calculate the position in question, then uses the times of the visits
        to assign them into seasons (based on where the sun is relative to the slicePoint RA).

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
           The (reduceFunc) of the length of each season, in days.
        """
        # Order data Slice/times and exclude visits which are too short.
        long = np.where(dataSlice[self.expTimeCol] > self.minExpTime)
        if len(long[0]) == 0:
            return self.badval
        data = np.sort(dataSlice[long], order=self.mjdCol)
        # SlicePoints ra/dec are always in radians - convert to degrees to calculate season
        seasons = calcSeason(np.degrees(slicePoint['ra']), data[self.mjdCol])
        firstOfSeason, lastOfSeason = findSeasonEdges(seasons)
        seasonlengths = data[self.mjdCol][lastOfSeason] - data[self.mjdCol][firstOfSeason]
        result = self.reduceFunc(seasonlengths)
        return result


class CampaignLengthMetric(BaseMetric):
    """Calculate the number of seasons (roughly, years) a pointing is observed for.
    This corresponds to the 'campaign length' for lensed quasar time delays.
    """
    def __init__(self, mjdCol='observationStartMJD', expTimeCol='visitExposureTime', minExpTime=20, **kwargs):
        units = ''
        self.expTimeCol = expTimeCol
        self.minExpTime = minExpTime
        self.mjdCol = mjdCol
        super().__init__(col=[self.mjdCol, self.expTimeCol], units=units, **kwargs)

    def run(self, dataSlice, slicePoint):
        # Order data Slice/times and exclude visits which are too short.
        long = np.where(dataSlice[self.expTimeCol] > self.minExpTime)
        if len(long[0]) == 0:
            return self.badval
        data = np.sort(dataSlice[long], order=self.mjdCol)
        seasons = calcSeason(np.degrees(slicePoint['ra']), data[self.mjdCol])
        intSeasons = np.floor(seasons)
        count = len(np.unique(intSeasons))
        return count


class MeanCampaignFrequencyMetric(BaseMetric):
    """Calculate the mean separation between nights, within a season - then the mean over the campaign.
    Calculate per season, to avoid any influence from season gaps.
    """
    def __init__(self, mjdCol='observationStartMJD', expTimeCol='visitExposureTime', minExpTime=20,
                 nightCol='night', **kwargs):
        self.mjdCol = mjdCol
        self.expTimeCol = expTimeCol
        self.minExpTime = minExpTime
        self.nightCol = nightCol
        units = 'nights'
        super().__init__(col=[self.mjdCol, self.expTimeCol, self.nightCol], units=units, **kwargs)

    def run(self, dataSlice, slicePoint):
        # Order data Slice/times and exclude visits which are too short.
        long = np.where(dataSlice[self.expTimeCol] > self.minExpTime)
        if len(long[0]) == 0:
            return self.badval
        data = np.sort(dataSlice[long], order=self.mjdCol)
        # SlicePoints ra/dec are always in radians - convert to degrees to calculate season
        seasons = calcSeason(np.degrees(slicePoint['ra']), data[self.mjdCol])
        firstOfSeason, lastOfSeason = findSeasonEdges(seasons)
        seasonMeans = np.zeros(len(firstOfSeason), float)
        for i, (first, last) in enumerate(zip(firstOfSeason, lastOfSeason)):
            if first < last:
                n = data[self.nightCol][first:last+1]
                deltaNights = np.diff(np.unique(n))
                if len(deltaNights) > 0:
                    seasonMeans[i] = np.mean(deltaNights)
        return np.mean(seasonMeans)


class TdcMetric(BaseMetric):
    """Calculate the Time Delay Challenge metric, as described in Liao et al 2015
    (https://arxiv.org/pdf/1409.1254.pdf).

    This combines the MeanCampaignFrequency/MeanNightSeparation, the SeasonLength, and the CampaignLength
    metrics above, but rewritten to calculate season information only once.

    cadNorm = in units of days
    seaNorm = in units of months
    campNorm = in units of years

    This metric also adds a requirement to achieve limiting magnitudes after galactic dust extinction,
    in various bandpasses, in order to exclude visits which are not useful for detecting quasars
    (due to being short or having high sky brightness, etc.) and to reject regions with
    high galactic dust extinction.

    Parameters
    ----------
    mjdCol: str, optional
        Column name for mjd. Default observationStartMJD.
    nightCol: str, optional
        Column name for night. Default night.
    filterCol: str, optional
        Column name for filter. Default filter.
    m5Col: str, optional
        Column name for five-sigma depth. Default fiveSigmaDepth.
    magCuts: dict, optional
        Dictionary with filtername:mag limit (after dust extinction). Default None in kwarg.
        Defaults set within metric: {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}
    metricName: str, optional
        Metric Name. Default TDC.
    cadNorm: float, optional
        Cadence normalization constant, in units of days. Default 3.
    seaNorm: float, optional
        Season normalization constant, in units of months. Default 4.
    campNorm: float, optional
        Campaign length normalization constant, in units of years. Default 5.
    badval: float, optional
        Return this value instead of the dictionary for bad points.

    Returns
    -------
    dictionary
        Dictionary of values for {'rate', 'precision', 'accuracy'} at this point in the sky.
    """
    def __init__(self, mjdCol='observationStartMJD', nightCol='night', filterCol='filter',
                 m5Col='fiveSigmaDepth', magCuts=None,
                 metricName = 'TDC', cadNorm=3., seaNorm=4., campNorm=5., badval=-999, **kwargs):
        # Save the normalization values.
        self.cadNorm = cadNorm
        self.seaNorm = seaNorm
        self.campNorm = campNorm
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.nightCol = nightCol
        self.filterCol = filterCol
        if magCuts is None:
            self.magCuts = {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}
        else:
            self.magCuts = magCuts
            if not isinstance(self.magCuts, dict):
                raise Exception('magCuts should be a dictionary')
        # Set up dust map requirement
        maps = ['DustMap']
        # Set the default wavelength limits for the lsst filters. These are approximately correct.
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1
        super().__init__(col=[self.mjdCol, self.m5Col, self.nightCol, self.filterCol],
                         badval=badval, maps=maps,
                         metricName = metricName, units = '%s' %('%'), **kwargs)

    def run(self, dataSlice, slicePoint):
        # Calculate dust-extinction limiting magnitudes for each visit.
        filterlist = np.unique(dataSlice[self.filterCol])
        m5Dust = np.zeros(len(dataSlice), float)
        for f in filterlist:
            match = np.where(dataSlice[self.filterCol] == f)
            A_x = self.Ax1[f] * slicePoint['ebv']
            m5Dust[match] = dataSlice[self.m5Col][match] - A_x
            m5Dust[match] = np.where(m5Dust[match] > self.magCuts[f], m5Dust[match], -999)
        idxs = np.where(m5Dust > -998)
        if len(idxs[0]) == 0:
            return self.badval
        data = np.sort(dataSlice[idxs], order=self.mjdCol)
        # SlicePoints ra/dec are always in radians - convert to degrees to calculate season
        seasons = calcSeason(np.degrees(slicePoint['ra']), data[self.mjdCol])
        intSeasons = np.floor(seasons)
        firstOfSeason, lastOfSeason = findSeasonEdges(seasons)
        # Campaign length
        camp = len(np.unique(intSeasons))
        # Season length
        seasonlengths = data[self.mjdCol][lastOfSeason] - data[self.mjdCol][firstOfSeason]
        sea = np.median(seasonlengths)
        # Convert to months
        sea = sea / 30.0
        # Campaign frequency / mean night separation
        seasonMeans = np.zeros(len(firstOfSeason), float)
        for i, (first, last) in enumerate(zip(firstOfSeason, lastOfSeason)):
            n = data[self.nightCol][first:last+1]
            deltaNights = np.diff(np.unique(n))
            if len(deltaNights) > 0:
                seasonMeans[i] = np.mean(deltaNights)
        cad = np.mean(seasonMeans)
        # Evaluate precision and accuracy for TDC
        if sea == 0 or cad == 0 or camp == 0:
            return self.badval
        else:
            accuracy = 0.06 * (self.seaNorm / sea) * \
                       (self.campNorm / camp)**(1.1)
            precision = 4.0 * (cad / self.cadNorm)**(0.7) * \
                        (self.seaNorm/sea)**(0.3) * \
                        (self.campNorm / camp)**(0.6)
            rate = 30. * (self.cadNorm / cad)**(0.4) * \
                   (sea / self.seaNorm)**(0.8) * \
                   (self.campNorm / camp)**(0.2)
        return {'accuracy':accuracy, 'precision':precision, 'rate':rate,
                'cadence':cad, 'season':sea, 'campaign':camp}

    def reduceAccuracy(self, metricValue):
        return metricValue['accuracy']

    def reducePrecision(self, metricValue):
        return metricValue['precision']

    def reduceRate(self, metricValue):
        return metricValue['rate']

    def reduceCadence(self, metricValue):
        return metricValue['cadence']

    def reduceSeason(self, metricValue):
        return metricValue['season']

    def reduceCampaign(self, metricValue):
        return metricValue['campaign']
