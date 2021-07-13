import numpy as np
from .baseStacker import BaseStacker
import warnings

__all__ = ['BaseMoStacker', 'AppMagStacker', 'CometAppMagStacker', 'SNRStacker', 'EclStacker']

# Willmer 2018, ApJS 236, 47
# VEGA V mag and AB mag of sun (LSST-equivalent bandpasses)
VMAG_SUN = -26.76  # Vega mag
AB_SUN = { 'u': -25.30,
           'g': -26.52,
           'r': -26.93,
           'i': -27.05,
           'z': -27.07,
           'y': -27.07
           }
KM_PER_AU = 149597870.7


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


class AppMagNullStacker(BaseMoStacker):
    """Do nothing to calculate an apparent magnitude.

    This assumes an apparent magnitude was part of the input data and does not need to be modified (no
    cloning, color terms, trailing losses, etc). Just return the appMag column.

    This would not be necessary in general, but appMag is treated as a special column (because we must have an
    apparent magnitude for most of the basic moving object metrics, and it must be calculated before SNR
    if that is also needed).
    """
    colsAdded = ['appMag']

    def __init__(self, appMagCol='appMag'):
        self.appMagCol = appMagCol
        self.units = ['mag',]
        self.colsReq = [self.appMagCol]

    def _run(self, ssoObs, Href, Hval):
        ssoObs['appMag'] = ssoObs[self.appMagCol]
        return ssoObs


class AppMagStacker(BaseMoStacker):
    """Add apparent magnitude of an object for the current Hval (compared to Href in the orbit file),
    incorporating the magnitude losses due to trailing/detection, as well as the color of the object.

    This is calculated from the reported magV in the input observation file (calculated assuming Href) as:
    ssoObs['appMag'] = ssoObs[self.vMagCol] + ssoObs[self.colorCol] + ssoObs[self.lossCol] + Hval - Href

    Using the vMag reported in the input observations implicitly uses the phase curve coded in at that point;
    for Oorb this is an H/G phase curve, with G=0.15 unless otherwise specified in the orbit file.
    See sims_movingObjects for more details on the color and loss quantities.

    Parameters
    ----------
    vMagCol : str, opt
        Name of the column containing the base V magnitude for the object at H=Href.
    lossCol : str, opt
        Name of the column describing the magnitude losses,
        due to trailing (dmagTrail) or detection (dmagDetect). Default dmagDetect.
    colorCol : str, opt
        Name of the column describing the color correction (into the observation filter, from V).
        Default dmagColor.
    """
    colsAdded = ['appMag']

    def __init__(self, vMagCol='magV', colorCol='dmagColor', lossCol='dmagDetect'):
        self.vMagCol = vMagCol
        self.colorCol = colorCol
        self.lossCol = lossCol
        self.colsReq = [self.vMagCol, self.colorCol, self.lossCol]
        self.units = ['mag',]

    def _run(self, ssoObs, Href, Hval):
        # Hval = current H value (useful if cloning over H range), Href = reference H value from orbit.
        # Without cloning, Href = Hval.
        ssoObs['appMag'] = ssoObs[self.vMagCol] + ssoObs[self.colorCol] + ssoObs[self.lossCol] + Hval - Href
        return ssoObs


class CometAppMagStacker(BaseMoStacker):
    """Add a cometary apparent magnitude, including nucleus and coma, based on a calculation of
    Afrho (using the current Hval) and a Halley-Marcus phase curve for the coma brightness.

    Parameters
    ----------
    cometType : str, opt
        Type of comet - short, oort, or mbc. This setting also sets the value of Afrho1 and k:
        short = Afrho1 / R^2 = 100 cm/km2, k = -4
        oort = Afrho1 / R^2 = 1000 cm/km2, k = -2
        mbc = Afrho1 / R^2 = 4000 cm/km2, k = -6.
        Default = 'oort'.
        It is also possible to pass this a dictionary instead: the dictionary should contain 'k' and
        'Afrho1_const' keys, which will be used to set these values directly.
        (e.g. cometType = {'k': -3.5, 'Afrho1_const': 1500}).
    Ap : float, opt
        The albedo for calculating the object's size. Default 0.04
    rhCol : str, opt
        The column name for the heliocentric distance (in AU). Default 'helio_dist'.
    deltaCol : str, opt
        The column name for the geocentric distance (in AU). Default 'geo_dist'.
    phaseCol : str, opt
        The column name for the phase value (in degrees). Default 'phase'.
    """
    colsAdded = ['appMag']

    def __init__(self, cometType='oort', Ap=0.04,
                 rhCol='helio_dist', deltaCol='geo_dist', phaseCol='phase',
                 seeingCol='FWHMgeom', apScale=1, filterCol='filter',
                 vMagCol='magV', colorCol='dmagColor', lossCol='dmagDetect'):
        self.units = ['mag']  # new column units
        # Set up k and Afrho1 constant values.
        cometTypes = {'short': {'k': -4, 'Afrho1_const': 100},
                      'oort': {'k': -2, 'Afrho1_const': 1000},
                      'mbc': {'k': -6, 'Afrho1_const': 4000}}
        self.k = None
        self.Afrho1_const = None
        if isinstance(cometType, str):
            if cometType in cometTypes:
                self.k = cometTypes[cometType]['k']
                self.Afrho1_const = cometTypes[cometType]['Afrho1_const']
        if isinstance(cometType, dict):
            if 'k' in cometType:
                self.k = cometType['k']
            if 'Afrho1_const' in cometType:
                self.Afrho1_const = cometType['Afrho1_const']
        if self.k is None or self.Afrho1_const is None:
                raise ValueError(f'cometType must be a string {cometTypes} or '
                                 f'dict containing "k" and "Afrho1_const" - but received {cometType}')
        # Phew, now set the simple stuff.
        self.Ap = Ap
        self.rhCol = rhCol
        self.deltaCol = deltaCol
        self.phaseCol = phaseCol
        self.seeingCol = seeingCol
        self.apScale = apScale
        self.filterCol = filterCol
        self.vMagCol = vMagCol
        self.colorCol = colorCol
        self.lossCol = lossCol
        # names of required columns
        self.colsReq = [self.rhCol, self.deltaCol, self.phaseCol, self.seeingCol, self.filterCol,
                        self.vMagCol, self.colorCol, self.lossCol]

    def _run(self, ssObs, Href, Hval):
        # Calculate radius from the current H value (Hval).
        radius = 10 ** (0.2 * (VMAG_SUN - Hval)) / np.sqrt(self.Ap) * KM_PER_AU
        # Calculate expected Afrho - this is a value that describes how the brightness of the coma changes
        afrho1 = self.Afrho1_const * radius**2
        phase_val = phase_HalleyMarcus(ssoObs[self.phaseCol])
        # afrho is equivalent to a sort of 'absolute' magnitude of the coma
        afrho = afrho1 * ssoObs[self.rhCol]**self.k * phase_val
        # rho describes the projected area on the sky (project the aperture into cm on-sky)
        # Using the seeing * apScale as the aperture
        radius_aperture = ssoObs[self.seeingCol] * self.apScale
        rho = 725e5 * ssoObs[self.deltaCol] * radius_aperture
        # Calculate the expected apparent of the comet coma = sun + correction
        delta = ssoObs[self.deltaCol] * KM_PER_AU * 1000  # delta in cm
        dm = -2.5 * np.log10(afrho * rho / (2 * ssoObs[self.rhCol] * delta)**2)
        coma = np.zeros(len(ssoObs), float)
        # This calculates a coma mag that scales with the sun's color in each bandpass, but the coma
        # modification is gray (i.e. it's just reflecting sunlight)
        for f in ssoObs[self.filterCol]:
            match = np.where(ssoObs[self.filterCol] == f)
            coma[match] = AB_SUN[f] + dm[match]
        # Calculate cometary nucleus magnitude -- we'll use the apparent V mag adapted from OOrb as well as
        # the object's color - these are generally assumed to be D type (which was used in sims_movingObjects)
        nucleus = ssoObs[self.vMagCol] + ssoObs[self.colorCol] + ssoObs[self.lossCol] + Hval - Href
        # comet apparent mag - ready for calculation of SNR, etc.
        ssObs['appMag'] = coma + nucleus
        return ssObs


class SNRStacker(BaseMoStacker):
    """Add SNR and visibility for a particular object, given the five sigma depth of the image and the
    apparent magnitude (whether from AppMagStacker or CometAppMagStacker, etc).

    The SNR simply calculates the SNR based on the five sigma depth and the apparent magnitude.
    The 'vis' column is a probabilistic flag (0/1) indicating whether the object was detected, assuming
    a 5-sigma SNR threshold and then applying a probabilistic cut on whether it was detected or not (i.e.
    there is a gentle roll-over in 'vis' from 1 to 0 depending on the SNR of the object).
    This is based on the Fermi-Dirac completeness formula as described in equation 24 of the Stripe 82 SDSS
    analysis here: http://iopscience.iop.org/0004-637X/794/2/120/pdf/apj_794_2_120.pdf.

    Parameters
    ----------
    appMagCol : str, opt
        Name of the column describing the apparent magnitude of the object. Default 'appMag'.
    m5Col : str, opt
        Name of the column describing the 5 sigma depth of each visit. Default fiveSigmaDepth.
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
        generation for the probability of detection. Default: None.
    """
    colsAdded = ['SNR', 'vis']

    def __init__(self, appMagCol='appMag', m5Col='fiveSigmaDepth',
                 gamma=0.038, sigma=0.12, randomSeed=None):
        self.appMagCol = appMagCol
        self.m5Col = m5Col
        self.gamma = gamma
        self.sigma = sigma
        self.randomSeed = randomSeed
        self.colsReq = [self.appMagCol, self.m5Col]
        self.units = ['SNR', '']

    def _run(self, ssoObs, Href, Hval):
        # Hval = current H value (useful if cloning over H range), Href = reference H value from orbit.
        # Without cloning, Href = Hval.
        xval = np.power(10, 0.5 * (ssoObs[self.appMagCol] - ssoObs[self.m5Col]))
        ssoObs['SNR'] = 1.0 / np.sqrt((0.04 - self.gamma) * xval + self.gamma * xval * xval)
        completeness = 1.0 / (1 + np.exp((ssoObs[self.appMagCol] - ssoObs[self.m5Col])/self.sigma))
        if not hasattr(self, '_rng'):
            if self.randomSeed is not None:
                self._rng = np.random.RandomState(self.randomSeed)
            else:
                self._rng = np.random.RandomState(734421)
        probability = self._rng.random_sample(len(ssoObs[self.appMagCol]))
        ssoObs['vis'] = np.where(probability <= completeness, 1, 0)
        return ssoObs


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
