from builtins import zip
# Gamma-ray burst afterglow metric
# ebellm@caltech.edu

import rubin_sim.maf.metrics as metrics
import numpy as np

__all__ = ['GRBTransientMetric'] 

class GRBTransientMetric(metrics.BaseMetric):
    """Detections for on-axis GRB afterglows decaying as 
	F(t) = F(1min)((t-t0)/1min)^-alpha.  No jet break, for now.

	Derived from TransientMetric, but calculated with reduce functions to 
    enable-band specific counts. 
	Burst parameters taken from 2011PASP..123.1034J.

	Simplifications: 
	no color variation or evolution encoded yet.
	no jet breaks.
	not treating off-axis events.
    
    Parameters
    ----------
    alpha : float, 
        temporal decay index 
        Default = 1.0
    apparent_mag_1min_mean : float, 
        mean magnitude at 1 minute after burst 
        Default = 15.35
    apparent_mag_1min_sigma : float, 
        std of magnitudes at 1 minute after burst 
        Default = 1.59
    transDuration : float, optional
        How long the transient lasts (days). Default 10.
    surveyDuration : float, optional
        Length of survey (years).
        Default 10.
    surveyStart : float, optional
        MJD for the survey start date.
        Default None (uses the time of the first observation).
    detectM5Plus : float, optional
        An observation will be used if the light curve magnitude is brighter than m5+detectM5Plus.
        Default 0.
    nPerFilter : int, optional
        Number of separate detections of the light curve above the 
        detectM5Plus theshold (in a single filter) for the light curve 
        to be counted.
        Default 1.
    nFilters : int, optional
        Number of filters that need to be observed nPerFilter times,
        with differences minDeltaMag,
        for an object to be counted as detected.
        Default 1.  
    minDeltaMag : float, optional
       magnitude difference between detections in the same filter required
       for second+ detection to be counted.
       For example, if minDeltaMag = 0.1 mag and two consecutive observations
       differ only by 0.05 mag, those two detections will only count as one.
       (Better would be a SNR-based discrimination of lightcurve change.)
       Default 0.
    nPhaseCheck : int, optional
        Sets the number of phases that should be checked.
        One can imagine pathological cadences where many objects pass the detection criteria,
        but would not if the observations were offset by a phase-shift.
        Default 1.
    """
    def __init__(self, alpha=1, apparent_mag_1min_mean=15.35, 
                 apparent_mag_1min_sigma=1.59, metricName='GRBTransientMetric', 
                 mjdCol='expMJD', m5Col='fiveSigmaDepth', filterCol='filter',
                 transDuration=10., 
                 surveyDuration=10., surveyStart=None, detectM5Plus=0.,
                 nPerFilter=1, nFilters=1, minDeltaMag=0., nPhaseCheck=1,
                 **kwargs):
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super( GRBTransientMetric, self).__init__(
                col=[self.mjdCol, self.m5Col, self.filterCol],
                units='Fraction Detected',
                metricName=metricName,**kwargs)
        self.alpha = alpha
        self.apparent_mag_1min_mean = apparent_mag_1min_mean
        self.apparent_mag_1min_sigma = apparent_mag_1min_sigma
        self.transDuration = transDuration
        self.surveyDuration = surveyDuration
        self.surveyStart = surveyStart
        self.detectM5Plus = detectM5Plus
        self.nPerFilter = nPerFilter
        self.nFilters = nFilters
        self.minDeltaMag = minDeltaMag
        self.nPhaseCheck = nPhaseCheck
        self.peakTime = 0.
        self.reduceOrder = {'Bandu':0, 'Bandg':1, 'Bandr':2, 'Bandi':3, 'Bandz':4, 'Bandy':5,'Band1FiltAvg':6,'BandanyNfilters':7}
        
    def lightCurve(self, time, filters):
        """
        given the times and filters of an observation, return the magnitudes.
        """

        lcMags = np.zeros(time.size, dtype=float)

        decline = np.where(time > self.peakTime)
        apparent_mag_1min = np.random.randn()*self.apparent_mag_1min_sigma + self.apparent_mag_1min_mean
        lcMags[decline] += apparent_mag_1min + self.alpha * 2.5 * np.log10((time[decline]-self.peakTime)*24.*60.)

        #for key in self.peaks.keys():
        #    fMatch = np.where(filters == key)
        #    lcMags[fMatch] += self.peaks[key]

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
        nTransMax = np.floor(self.surveyDuration / (self.transDuration / 365.25))
        tshifts = np.arange(self.nPhaseCheck) * self.transDuration / float(self.nPhaseCheck)
        nDetected = 0
        nTransMax = 0
        for tshift in tshifts:
            # Compute the total number of back-to-back transients are possible to detect
            # given the survey duration and the transient duration.
            nTransMax += np.floor(self.surveyDuration / (self.transDuration / 365.25))
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

            bandcounter={'u':0, 'g':0, 'r':0, 'i':0, 'z':0, 'y':0, 'any':0} #define zeroed out counter

            # make sure things are sorted by time
            ord = np.argsort(dataSlice[self.mjdCol])
            dataSlice = dataSlice[ord]
            detected = detected[ord]
            lcNumber = lcNumber[ord]
            lcMags = lcMags[ord]
            ulcNumber = np.unique(lcNumber)
            left = np.searchsorted(lcNumber, ulcNumber)
            right = np.searchsorted(lcNumber, ulcNumber, side='right')
            detectThresh += self.nFilters

            # iterate over the lightcurves
            for le, ri in zip(left, right):
                wdet = np.where(detected[le:ri] > 0)
                ufilters = np.unique(dataSlice[self.filterCol][le:ri][wdet])
                nfilts_lci = 0
                for filtName in ufilters:
                    wdetfilt = np.where(
                        (dataSlice[self.filterCol][le:ri] == filtName) &
                        detected[le:ri])

                    lcPoints = lcMags[le:ri][wdetfilt]
                    dlc = np.abs(np.diff(lcPoints))

                    # number of detections in band, requring that for
                    # nPerFilter > 1 that points have more than minDeltaMag
                    # change
                    nbanddet = np.sum(dlc > self.minDeltaMag) + 1
                    if nbanddet >= self.nPerFilter:
                        bandcounter[filtName] += 1
                        nfilts_lci += 1
                if nfilts_lci >= self.nFilters:
                    bandcounter['any'] += 1

        bandfraction = {}
        for band in bandcounter.keys():
            bandfraction[band] = float(bandcounter[band]) / nTransMax

        return bandfraction


    def reduceBand1FiltAvg(self, bandfraction):
        "Average fraction detected in single filter" 
        return np.mean(list(bandfraction.values()))
      
    def reduceBandanyNfilters(self, bandfraction):
        "Fraction of events detected in Nfilters or more" 
        return bandfraction['any']

    def reduceBandu(self, bandfraction):
        return bandfraction['u']

    def reduceBandg(self, bandfraction):
        return bandfraction['g']

    def reduceBandr(self, bandfraction):
        return bandfraction['r']

    def reduceBandi(self, bandfraction):
        return bandfraction['i']

    def reduceBandz(self, bandfraction):
        return bandfraction['z']

    def reduceBandy(self, bandfraction):
        return bandfraction['y']
