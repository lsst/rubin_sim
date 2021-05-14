from builtins import zip
import numpy as np
from .baseMetric import BaseMetric

__all__ = ['TransientMetric']

class TransientMetric(BaseMetric):
    """
    Calculate what fraction of the transients would be detected. Best paired with a spatial slicer.
    We are assuming simple light curves with no color evolution.

    Parameters
    ----------
    transDuration : float, optional
        How long the transient lasts (days). Default 10.
    peakTime : float, optional
        How long it takes to reach the peak magnitude (days). Default 5.
    riseSlope : float, optional
        Slope of the light curve before peak time (mags/day).
        This should be negative since mags are backwards (magnitudes decrease towards brighter fluxes).
        Default 0.
    declineSlope : float, optional
        Slope of the light curve after peak time (mags/day).
        This should be positive since mags are backwards. Default 0.
    uPeak : float, optional
        Peak magnitude in u band. Default 20.
    gPeak : float, optional
        Peak magnitude in g band. Default 20.
    rPeak : float, optional
        Peak magnitude in r band. Default 20.
    iPeak : float, optional
        Peak magnitude in i band. Default 20.
    zPeak : float, optional
        Peak magnitude in z band. Default 20.
    yPeak : float, optional
        Peak magnitude in y band. Default 20.
    surveyDuration : float, optional
        Length of survey (years).
        Default 10.
    surveyStart : float, optional
        MJD for the survey start date.
        Default None (uses the time of the first observation).
    detectM5Plus : float, optional
        An observation will be used if the light curve magnitude is brighter than m5+detectM5Plus.
        Default 0.
    nPrePeak : int, optional
        Number of observations (in any filter(s)) to demand before peakTime,
        before saying a transient has been detected.
        Default 0.
    nPerLC : int, optional
        Number of sections of the light curve that must be sampled above the detectM5Plus theshold
        (in a single filter) for the light curve to be counted.
        For example, setting nPerLC = 2 means a light curve  is only considered detected if there
        is at least 1 observation in the first half of the LC, and at least one in the second half of the LC.
        nPerLC = 4 means each quarter of the light curve must be detected to count.
        Default 1.
    nFilters : int, optional
        Number of filters that need to be observed for an object to be counted as detected.
        Default 1.
    nPhaseCheck : int, optional
        Sets the number of phases that should be checked.
        One can imagine pathological cadences where many objects pass the detection criteria,
        but would not if the observations were offset by a phase-shift.
        Default 1.
    countMethod : {'full' 'partialLC'}, defaults to 'full'
        Sets the method of counting max number of transients. if 'full', the
        only full light curves that fit the survey duration are counted. If
        'partialLC', then the max number of possible transients is taken to be
        the integer floor
    """
    def __init__(self, metricName='TransientDetectMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter',
                 transDuration=10., peakTime=5., riseSlope=0., declineSlope=0.,
                 surveyDuration=10., surveyStart=None, detectM5Plus=0.,
                 uPeak=20, gPeak=20, rPeak=20, iPeak=20, zPeak=20, yPeak=20,
                 nPrePeak=0, nPerLC=1, nFilters=1, nPhaseCheck=1, countMethod='full',
                 **kwargs):
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(TransientMetric, self).__init__(col=[self.mjdCol, self.m5Col, self.filterCol],
                                              units='Fraction Detected',
                                              metricName=metricName, **kwargs)
        self.peaks = {'u': uPeak, 'g': gPeak, 'r': rPeak, 'i': iPeak, 'z': zPeak, 'y': yPeak}
        self.transDuration = transDuration
        self.peakTime = peakTime
        self.riseSlope = riseSlope
        self.declineSlope = declineSlope
        self.surveyDuration = surveyDuration
        self.surveyStart = surveyStart
        self.detectM5Plus = detectM5Plus
        self.nPrePeak = nPrePeak
        self.nPerLC = nPerLC
        self.nFilters = nFilters
        self.nPhaseCheck = nPhaseCheck
        self.countMethod = countMethod

    def lightCurve(self, time, filters):
        """
        Calculate the magnitude of the object at each time, in each filter.

        Parameters
        ----------
        time : numpy.ndarray
            The times of the observations.
        filters : numpy.ndarray
            The filters of the observations.

        Returns
        -------
        numpy.ndarray
            The magnitudes of the object at each time, in each filter.
        """
        lcMags = np.zeros(time.size, dtype=float)
        rise = np.where(time <= self.peakTime)
        lcMags[rise] += self.riseSlope * time[rise] - self.riseSlope * self.peakTime
        decline = np.where(time > self.peakTime)
        lcMags[decline] += self.declineSlope * (time[decline] - self.peakTime)
        for key in self.peaks:
            fMatch = np.where(filters == key)
            lcMags[fMatch] += self.peaks[key]
        return lcMags

    def run(self, dataSlice, slicePoint=None):
        """"
        Calculate the detectability of a transient with the specified lightcurve.

        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.

        Returns
        -------
        float
            The total number of transients that could be detected.
        """
        # Total number of transients that could go off back-to-back
        if self.countMethod == 'partialLC':
            _nTransMax = np.ceil(self.surveyDuration / (self.transDuration / 365.25))
        else:
            _nTransMax = np.floor(self.surveyDuration / (self.transDuration / 365.25))
        tshifts = np.arange(self.nPhaseCheck) * self.transDuration / float(self.nPhaseCheck)
        nDetected = 0
        nTransMax = 0
        for tshift in tshifts:
            # Compute the total number of back-to-back transients are possible to detect
            # given the survey duration and the transient duration.
            nTransMax += _nTransMax
            if tshift != 0:
                nTransMax -= 1
            if self.surveyStart is None:
                surveyStart = dataSlice[self.mjdCol].min()
            time = (dataSlice[self.mjdCol] - surveyStart + tshift) % self.transDuration

            # Which lightcurve does each point belong to
            lcNumber = np.floor((dataSlice[self.mjdCol] - surveyStart) / self.transDuration)

            lcMags = self.lightCurve(time, dataSlice[self.filterCol])

            # How many criteria needs to be passed
            detectThresh = 0

            # Flag points that are above the SNR limit
            detected = np.zeros(dataSlice.size, dtype=int)
            detected[np.where(lcMags < dataSlice[self.m5Col] + self.detectM5Plus)] += 1
            detectThresh += 1

            # If we demand points on the rise
            if self.nPrePeak > 0:
                detectThresh += 1
                ord = np.argsort(dataSlice[self.mjdCol])
                dataSlice = dataSlice[ord]
                detected = detected[ord]
                lcNumber = lcNumber[ord]
                time = time[ord]
                ulcNumber = np.unique(lcNumber)
                left = np.searchsorted(lcNumber, ulcNumber)
                right = np.searchsorted(lcNumber, ulcNumber, side='right')
                # Note here I'm using np.searchsorted to basically do a 'group by'
                # might be clearer to use scipy.ndimage.measurements.find_objects or pandas, but
                # this numpy function is known for being efficient.
                for le, ri in zip(left, right):
                    # Number of points where there are a detection
                    good = np.where(time[le:ri] < self.peakTime)
                    nd = np.sum(detected[le:ri][good])
                    if nd >= self.nPrePeak:
                        detected[le:ri] += 1

            # Check if we need multiple points per light curve or multiple filters
            if (self.nPerLC > 1) | (self.nFilters > 1):
                # make sure things are sorted by time
                ord = np.argsort(dataSlice[self.mjdCol])
                dataSlice = dataSlice[ord]
                detected = detected[ord]
                lcNumber = lcNumber[ord]
                time = time[ord]
                ulcNumber = np.unique(lcNumber)
                left = np.searchsorted(lcNumber, ulcNumber)
                right = np.searchsorted(lcNumber, ulcNumber, side='right')
                detectThresh += self.nFilters

                for le, ri in zip(left, right):
                    points = np.where(detected[le:ri] > 0)
                    ufilters = np.unique(dataSlice[self.filterCol][le:ri][points])
                    phaseSections = np.floor(time[le:ri][points] / self.transDuration * self.nPerLC)
                    for filtName in ufilters:
                        good = np.where(dataSlice[self.filterCol][le:ri][points] == filtName)
                        if np.size(np.unique(phaseSections[good])) >= self.nPerLC:
                            detected[le:ri] += 1

            # Find the unique number of light curves that passed the required number of conditions
            nDetected += np.size(np.unique(lcNumber[np.where(detected >= detectThresh)]))

        # Rather than keeping a single "detected" variable, maybe make a mask for each criteria, then
        # reduce functions like: reduce_singleDetect, reduce_NDetect, reduce_PerLC, reduce_perFilter.
        # The way I'm running now it would speed things up.

        return float(nDetected) / nTransMax
