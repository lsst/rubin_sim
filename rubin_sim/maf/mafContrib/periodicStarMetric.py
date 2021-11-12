import warnings
import numpy as np
from scipy.optimize import curve_fit
from rubin_sim.maf.metrics.baseMetric import BaseMetric
from rubin_sim.maf.utils import m52snr


__all__ = ['periodicStar', 'PeriodicStarMetric']


class periodicStar(object):
    def __init__(self, filternames):
        self.filternames = filternames

    def __call__(self, t,x0,x1,x2,x3,x4,x5,x6,x7,x8):
        """ Approximate a periodic star as a simple sin wave.
        t: array with "time" in days, and "filter" dtype names.
        x0: Period (days)
        x1: Phase (days)
        x2: Amplitude (mag)
        x3: mean u mag
        x4: mean g mag
        x5: mean r mag
        x6: mean i mag
        x7: mean z mag
        x8: mean y mag
        """
        filter2index = {'u':3, 'g':4, 'r':5, 'i':6,
                        'z':7,'y':8}
        filterNames = np.unique(self.filternames)
        mags = x2*np.sin((t+x1)/x0*2.*np.pi)
        x=[x0,x1,x2,x3,x4,x5,x6,x7,x8]
        for f in filterNames:
            good = np.where(self.filternames == f)
            mags[good] += x[filter2index[f]]
        return mags


class PeriodicStarMetric(BaseMetric):
    """ At each slicePoint, run a Monte Carlo simulation to see how well a periodic source can be fit.
    Assumes a simple sin-wave light-curve, and generates Gaussain noise based in the 5-sigma limiting depth
    of each observation.
    """

    def __init__(self, metricName='PeriodicStarMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', period=10., amplitude=0.5,
                 phase=2.,
                 nMonte=1000, periodTol=0.05, ampTol=0.10, means=[20.,20.,20.,20.,20.,20.],
                 magTol=0.10, nBands=3, seed=42, **kwargs):
        """
        period: days (default 10)
        amplitude: mags (default 1)
        nMonte: number of noise realizations to make in the Monte Carlo
        periodTol: fractional tolerance on the period to demand for a star to be considered well-fit
        ampTol: fractional tolerance on the amplitude to demand
        means: mean magnitudes for ugrizy
        magTol: Mean magnitude tolerance (mags)
        nBands: Number of bands that must be within magTol
        seed: random number seed
        """
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(PeriodicStarMetric, self).__init__(col=[self.mjdCol, self.m5Col,self.filterCol],
                                                 units='Fraction Detected',
                                                 metricName=metricName,**kwargs)
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        self.nMonte = nMonte
        self.periodTol = periodTol
        self.ampTol = ampTol
        self.means = np.array(means)
        self.magTol = magTol
        self.nBands = nBands
        np.random.seed(seed)
        self.filter2index = {'u':3, 'g':4, 'r':5, 'i':6, 'z':7,'y':8}

    def run(self, dataSlice, slicePoint=None):

        # Bail if we don't have enough points
        # (need to fit mean magnitudes in each of the available bands - self.means
        # and for a period, amplitude, and phase)
        if dataSlice.size < self.means.size+3:
            return self.badval

        # Generate input for true light curve
        t = np.empty(dataSlice.size, dtype=list(zip(['time','filter'],[float,'|U1'])))
        t['time'] = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        t['filter'] = dataSlice[self.filterCol]


        # If we are adding a distance modulus to the magnitudes
        if 'distMod' in list(slicePoint.keys()):
            mags = self.means + slicePoint['distMod']
        else:
            mags = self.means
        trueParams = np.append(np.array([self.period, self.phase, self.amplitude]), mags)
        true_obj = periodicStar(t['filter'])
        trueLC = true_obj(t['time'], *trueParams)

        # Array to hold the fit results
        fits = np.zeros((self.nMonte,trueParams.size),dtype=float)
        for i in np.arange(self.nMonte):
            snr = m52snr(trueLC,dataSlice[self.m5Col])
            dmag = 2.5*np.log10(1.+1./snr)
            noise = np.random.randn(trueLC.size)*dmag
            # Suppress warnings about failing on covariance
            fit_obj = periodicStar(t['filter'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # If it fails to converge, save values that should fail later
                try:
                    parmVals, pcov = curve_fit(fit_obj, t['time'], trueLC+noise, p0=trueParams,
                                               sigma=dmag)
                except:
                    parmVals = trueParams*0+np.inf
            fits[i,:] = parmVals

        # Throw out any magnitude fits if there are no observations in that filter
        ufilters = np.unique(dataSlice[self.filterCol])
        if ufilters.size < 9:
            for key in list(self.filter2index.keys()):
                if key not in ufilters:
                    fits[:,self.filter2index[key]] = -np.inf

        # Find the fraction of fits that meet the "well-fit" criteria
        periodFracErr = np.abs((fits[:,0]-trueParams[0])/trueParams[0])
        ampFracErr = np.abs((fits[:,2]-trueParams[2])/trueParams[2])
        magErr = np.abs(fits[:,3:]-trueParams[3:])
        nBands = np.zeros(magErr.shape,dtype=int)
        nBands[np.where(magErr <= self.magTol)] = 1
        nBands = np.sum(nBands, axis=1)
        nRecovered = np.size(np.where( (periodFracErr <= self.periodTol) &
                                       (ampFracErr <= self.ampTol) &
                                       (nBands >= self.nBands) )[0])
        fracRecovered = float(nRecovered)/self.nMonte


        return fracRecovered
