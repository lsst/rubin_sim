import numpy as np
from .baseStacker import BaseStacker
import warnings

__all__ = ['BaseMoStacker', 'MoMagStacker', 'CometMagVStacker', 'EclStacker']


class BaseMoStacker(BaseStacker):
    """Base class for moving object (SSobject)  stackers. Relevant for MoSlicer ssObs (pd.dataframe).

    Provided to add moving-object specific API for 'run' method of moving object stackers."""
    def run(self, ssoObs, Href, Hval=None):
        # Redefine this here, as the API does not match BaseStacker.
        if Hval is None:
            Hval = Href
        if len(ssoObs) == 0:
            return ssoObs
        # Add the columns.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ssoObs, cols_present = self._addStackerCols(ssoObs)
        # Here we don't really care about cols_present, because almost every time we will be readding
        # columns anymore (for different H values).
        return self._run(ssoObs, Href, Hval)


class MoMagStacker(BaseMoStacker):
    """Add columns relevant to SSobject apparent magnitudes and visibility to the slicer ssoObs
    dataframe, given a particular Href and current Hval.

    Specifically, this stacker adds magLimit, appMag, SNR, and vis.
    magLimit indicates the appropriate limiting magnitude to consider for a particular object in a particular
    observation, when combined with the losses due to detection (dmagDetect) or trailing (dmagTrail).
    appMag adds the apparent magnitude in the filter of the current object, at the current Hval.
    SNR adds the SNR of this object, given the magLimit.
    vis adds a flag (0/1) indicating whether an object was visible (assuming a 5sigma threshhold including
    some probabilistic determination of visibility).

    Parameters
    ----------
    m5Col : str, opt
        Name of the column describing the 5 sigma depth of each visit. Default fiveSigmaDepth.
    lossCol : str, opt
        Name of the column describing the magnitude losses,
        due to trailing (dmagTrail) or detection (dmagDetect). Default dmagDetect.
    gamma : float, opt
        The 'gamma' value for calculating SNR. Default 0.038.
        LSST range under normal conditions is about 0.037 to 0.039.
    sigma : float, opt
        The 'sigma' value for probabilistic prediction of whether or not an object is visible at 5sigma.
        Default 0.12.
        The probabilistic prediction of visibility is based on Fermi-Dirac completeness formula (see SDSS,
        eqn 24, Stripe82 analysis: http://iopscience.iop.org/0004-637X/794/2/120/pdf/apj_794_2_120.pdf).
    randomSeed: int or None, optional
        If set, then used as the random seed for the numpy random number
        generation for the dither offsets.
        Default: None.
    """
    colsAdded = ['appMagV', 'appMag', 'SNR', 'vis']

    def __init__(self, vMagCol='magV', colorCol='dmagColor',
                 lossCol='dmagDetect', m5Col='fiveSigmaDepth', gamma=0.038, sigma=0.12,
                 randomSeed=None):
        self.vMagCol = vMagCol
        self.colorCol = colorCol
        self.m5Col = m5Col
        self.lossCol = lossCol
        self.gamma = gamma
        self.sigma = sigma
        self.randomSeed = randomSeed
        self.colsReq = [self.m5Col, self.vMagCol, self.colorCol, self.lossCol]
        self.units = ['mag', 'mag', 'SNR', '']

    def _run(self, ssoObs, Href, Hval):
        # Hval = current H value (useful if cloning over H range), Href = reference H value from orbit.
        # Without cloning, Href = Hval.
        ssoObs['appMagV'] = ssoObs[self.vMagCol] + ssoObs[self.lossCol] + Hval - Href
        ssoObs['appMag'] = ssoObs[self.vMagCol] + ssoObs[self.colorCol] + ssoObs[self.lossCol] + Hval - Href
        xval = np.power(10, 0.5 * (ssoObs['appMag'] - ssoObs[self.m5Col]))
        ssoObs['SNR'] = 1.0 / np.sqrt((0.04 - self.gamma) * xval + self.gamma * xval * xval)
        completeness = 1.0 / (1 + np.exp((ssoObs['appMag'] - ssoObs[self.m5Col])/self.sigma))
        if not hasattr(self, '_rng'):
            if self.randomSeed is not None:
                self._rng = np.random.RandomState(self.randomSeed)
            else:
                self._rng = np.random.RandomState(734421)

        probability = self._rng.random_sample(len(ssoObs['appMag']))
        ssoObs['vis'] = np.where(probability <= completeness, 1, 0)
        return ssoObs


class CometMagVStacker(BaseMoStacker):
    """Add an base V magnitude using a cometary magnitude model.

    The cometV magnitude is intended to replace the 'magV' column coming from sims_movingObjects,
    thus it does NOT take into account Hval, only Href. The full 'apparent magnitude' is calculated
    with the MoMagStacker, configured for the appropriate 'vMagCol'.
    
    m = M + 5 log10(Δ) + (5 + K) log10(rh)
        
    Parameters
    ----------
    k : float, opt
        Activity / intrinsic brightness dependence on heliocentric distance: rh**k.
        Note the default here is k = 2.
    rhCol : str, opt
        The column name for the heliocentric distance. Default 'helio_dist'.
    deltaCol : str, opt
        The column name for the geocentric distance. Default 'geo_dist'.
    """
    colsAdded = ['cometV']

    def __init__(self, k=2, rhCol='helio_dist', deltaCol='geo_dist'):
        self.units = ['mag']  # new column units
        self.k = k
        self.rhCol = rhCol
        self.deltaCol = deltaCol
        self.colsReq = [self.rhCol, self.deltaCol]  # names of required columns
        
    def _run(self, ssObs, Href, Hval):
        # comet apparent mag, use Href here and H-mag cloning will work later with MoMagStacker
        ssObs['cometV'] = (Href + 5 * np.log10(ssObs[self.deltaCol]) 
                           + (5 + self.k) * np.log10(ssObs[self.rhCol]))
        return ssObs


class EclStacker(BaseMoStacker):
    """
    Add ecliptic latitude/longitude (ecLat/ecLon) to the slicer ssoObs (in degrees).

    Parameters
    -----------
    raCol : str, opt
        Name of the RA column to convert to ecliptic lat/long. Default 'ra'.
    decCol : str, opt
        Name of the Dec column to convert to ecliptic lat/long. Default 'dec'.
    inDeg : bool, opt
        Flag indicating whether RA/Dec are in degrees. Default True.
    """
    colsAdded = ['ecLat', 'ecLon']

    def __init__(self, raCol='ra', decCol='dec', inDeg=True):
        self.raCol = raCol
        self.decCol = decCol
        self.inDeg = inDeg
        self.colsReq = [self.raCol, self.decCol]
        self.units = ['deg', 'deg']
        self.ecnode = 0.0
        self.ecinc = np.radians(23.439291)

    def _run(self, ssoObs, Href, Hval):
        ra = ssoObs[self.raCol]
        dec = ssoObs[self.decCol]
        if self.inDeg:
            ra = np.radians(ra)
            dec = np.radians(dec)
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
        xp = x
        yp = np.cos(self.ecinc)*y + np.sin(self.ecinc)*z
        zp = -np.sin(self.ecinc)*y + np.cos(self.ecinc)*z
        ssoObs['ecLat'] = np.degrees(np.arcsin(zp))
        ssoObs['ecLon'] = np.degrees(np.arctan2(yp, xp))
        ssoObs['ecLon'] = ssoObs['ecLon'] % 360
        return ssoObs
