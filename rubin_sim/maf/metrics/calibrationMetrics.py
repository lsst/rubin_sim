import numpy as np
from .baseMetric import BaseMetric
import rubin_sim.maf.utils as mafUtils
import rubin_sim.utils as utils
from scipy.optimize import curve_fit
from builtins import str

__all__ = ['ParallaxMetric', 'ProperMotionMetric', 'RadiusObsMetric',
           'ParallaxCoverageMetric', 'ParallaxDcrDegenMetric']


class ParallaxMetric(BaseMetric):
    """Calculate the uncertainty in a parallax measurement given a series of observations.

    Uses columns ra_pi_amp and dec_pi_amp, calculated by the ParallaxFactorStacker.

    Parameters
    ----------
    metricName : str, optional
        Default 'parallax'.
    m5Col : str, optional
        The default column name for m5 information in the input data. Default fiveSigmaDepth.
    filterCol : str, optional
        The column name for the filter information. Default filter.
    seeingCol : str, optional
        The column name for the seeing information. Since the astrometry errors are based on the physical
        size of the PSF, this should be the FWHM of the physical psf. Default seeingFwhmGeom.
    rmag : float, optional
        The r magnitude of the fiducial star in r band. Other filters are sclaed using sedTemplate keyword.
        Default 20.0
    SedTemplate : str, optional
        The template to use. This can be 'flat' or 'O','B','A','F','G','K','M'. Default flat.
    atm_err : float, optional
        The expected centroiding error due to the atmosphere, in arcseconds. Default 0.01.
    normalize : `bool`, optional
        Compare the astrometric uncertainty to the uncertainty that would result if half the observations
        were taken at the start and half at the end. A perfect survey will have a value close to 1, while
        a poorly scheduled survey will be close to 0. Default False.
    badval : float, optional
        The value to return when the metric value cannot be calculated. Default -666.
    """
    def __init__(self, metricName='parallax', m5Col='fiveSigmaDepth',
                 filterCol='filter', seeingCol='seeingFwhmGeom', rmag=20.,
                 SedTemplate='flat', badval=-666,
                 atm_err=0.01, normalize=False, **kwargs):
        Cols = [m5Col, filterCol, seeingCol, 'ra_pi_amp', 'dec_pi_amp']
        if normalize:
            units = 'ratio'
        else:
            units = 'mas'
        super(ParallaxMetric, self).__init__(Cols, metricName=metricName, units=units,
                                             badval=badval, **kwargs)
        # set return type
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.mags = {}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(SedTemplate, rmag=rmag)
        self.atm_err = atm_err
        self.normalize = normalize
        self.comment = 'Estimated uncertainty in parallax measurement ' \
                       '(assuming no proper motion or that proper motion '
        self.comment += 'is well fit). Uses measurements in all bandpasses, ' \
                        'and estimates astrometric error based on SNR '
        self.comment += 'in each visit. '
        if SedTemplate == 'flat':
            self.comment += 'Assumes a flat SED. '
        if self.normalize:
            self.comment += 'This normalized version of the metric displays the ' \
                            'estimated uncertainty in the parallax measurement, '
            self.comment += 'divided by the minimum parallax uncertainty possible ' \
                            '(if all visits were six '
            self.comment += 'months apart). Values closer to 1 indicate more optimal ' \
                            'scheduling for parallax measurement.'

    def _final_sigma(self, position_errors, ra_pi_amp, dec_pi_amp):
        """Assume parallax in RA and DEC are fit independently, then combined.
        All inputs assumed to be arcsec """
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_A = position_errors/ra_pi_amp
            sigma_B = position_errors/dec_pi_amp
            sigma_ra = np.sqrt(1./np.sum(1./sigma_A**2))
            sigma_dec = np.sqrt(1./np.sum(1./sigma_B**2))
            # Combine RA and Dec uncertainties, convert to mas
            sigma = np.sqrt(1./(1./sigma_ra**2+1./sigma_dec**2))*1e3
        return sigma

    def run(self, dataslice, slicePoint=None):
        filters = np.unique(dataslice[self.filterCol])
        if hasattr(filters[0], 'decode'):
            filters = [str(f.decode('utf-8')) for f in filters]
        snr = np.zeros(len(dataslice), dtype='float')
        # compute SNR for all observations
        for filt in filters:
            good = np.where(dataslice[self.filterCol] == filt)
            snr[good] = mafUtils.m52snr(self.mags[str(filt)], dataslice[self.m5Col][good])
        position_errors = np.sqrt(mafUtils.astrom_precision(dataslice[self.seeingCol],
                                                            snr)**2+self.atm_err**2)
        sigma = self._final_sigma(position_errors, dataslice['ra_pi_amp'], dataslice['dec_pi_amp'])
        if self.normalize:
            # Leave the dec parallax as zero since one can't have ra and dec maximized at the same time.
            sigma = self._final_sigma(position_errors,
                                      dataslice['ra_pi_amp']*0+1., dataslice['dec_pi_amp']*0)/sigma
        return sigma


class ProperMotionMetric(BaseMetric):
    """Calculate the uncertainty in the returned proper motion.

    This metric assumes gaussian errors in the astrometry measurements.

    Parameters
    ----------
    metricName : str, optional
        Default 'properMotion'.
    m5Col : str, optional
        The default column name for m5 information in the input data. Default fiveSigmaDepth.
    mjdCol : str, optional
        The column name for the exposure time. Default observationStartMJD.
    filterCol : str, optional
        The column name for the filter information. Default filter.
    seeingCol : str, optional
        The column name for the seeing information. Since the astrometry errors are based on the physical
        size of the PSF, this should be the FWHM of the physical psf. Default seeingFwhmGeom.
    rmag : float, optional
        The r magnitude of the fiducial star in r band. Other filters are sclaed using sedTemplate keyword.
        Default 20.0
    SedTemplate : str, optional
        The template to use. This can be 'flat' or 'O','B','A','F','G','K','M'. Default flat.
    atm_err : float, optional
        The expected centroiding error due to the atmosphere, in arcseconds. Default 0.01.
    normalize : `bool`, optional
        Compare the astrometric uncertainty to the uncertainty that would result if half the observations
        were taken at the start and half at the end. A perfect survey will have a value close to 1, while
        a poorly scheduled survey will be close to 0. Default False.
    baseline : float, optional
        The length of the survey used for the normalization, in years. Default 10.
    badval : float, optional
        The value to return when the metric value cannot be calculated. Default -666.
    """
    def __init__(self, metricName='properMotion',
                 m5Col='fiveSigmaDepth', mjdCol='observationStartMJD',
                 filterCol='filter', seeingCol='seeingFwhmGeom', rmag=20.,
                 SedTemplate='flat', badval= -666,
                 atm_err=0.01, normalize=False,
                 baseline=10., **kwargs):
        cols = [m5Col, mjdCol, filterCol, seeingCol]
        if normalize:
            units = 'ratio'
        else:
            units = 'mas/yr'
        super(ProperMotionMetric, self).__init__(col=cols, metricName=metricName, units=units,
                                                 badval=badval, **kwargs)
        # set return type
        self.mjdCol = mjdCol
        self.seeingCol = seeingCol
        self.m5Col = m5Col
        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.mags = {}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(SedTemplate, rmag=rmag)
        self.atm_err = atm_err
        self.normalize = normalize
        self.baseline = baseline
        self.comment = 'Estimated uncertainty of the proper motion fit ' \
                       '(assuming no parallax or that parallax is well fit). '
        self.comment += 'Uses visits in all bands, and generates approximate ' \
                        'astrometric errors using the SNR in each visit. '
        if SedTemplate == 'flat':
            self.comment += 'Assumes a flat SED. '
        if self.normalize:
            self.comment += 'This normalized version of the metric represents ' \
                            'the estimated uncertainty in the proper '
            self.comment += 'motion divided by the minimum uncertainty possible ' \
                            '(if all visits were '
            self.comment += 'obtained on the first and last days of the survey). '
            self.comment += 'Values closer to 1 indicate more optimal scheduling.'

    def run(self, dataslice, slicePoint=None):
        filters = np.unique(dataslice['filter'])
        filters = [str(f) for f in filters]
        precis = np.zeros(dataslice.size, dtype='float')
        for f in filters:
            observations = np.where(dataslice['filter'] == f)
            if np.size(observations[0]) < 2:
                precis[observations] = self.badval
            else:
                snr = mafUtils.m52snr(self.mags[f],
                                      dataslice[self.m5Col][observations])
                precis[observations] = mafUtils.astrom_precision(
                    dataslice[self.seeingCol][observations], snr)
                precis[observations] = np.sqrt(precis[observations]**2 + self.atm_err**2)
        good = np.where(precis != self.badval)
        result = mafUtils.sigma_slope(dataslice[self.mjdCol][good], precis[good])
        result = result*365.25*1e3  # Convert to mas/yr
        if (self.normalize) & (good[0].size > 0):
            new_dates = dataslice[self.mjdCol][good]*0
            nDates = new_dates.size
            new_dates[nDates//2:] = self.baseline*365.25
            result = (mafUtils.sigma_slope(new_dates, precis[good])*365.25*1e3)/result
        # Observations that are very close together can still fail
        if np.isnan(result):
            result = self.badval
        return result


class ParallaxCoverageMetric(BaseMetric):
    """
    Check how well the parallax factor is distributed. Subtracts the weighted mean position of the
    parallax offsets, then computes the weighted mean radius of the points.
    If points are well distributed, the mean radius will be near 1. If phase coverage is bad,
    radius will be close to zero.

    For points on the Ecliptic, uniform sampling should result in a metric value of ~0.5.
    At the poles, uniform sampling would result in a metric value of ~1.
    Conceptually, it is helpful to remember that the parallax motion of a star at the pole is
    a (nearly circular) ellipse while the motion of a star on the ecliptic is a straight line. Thus, any
    pair of observations separated by 6 months will give the full parallax range for a star on the pole
    but only observations on very specific dates will give the full range for a star on the ecliptic.

    Optionally also demand that there are observations above the snrLimit kwarg spanning thetaRange radians.

    Parameters
    ----------
    m5Col: str, optional
        Column name for individual visit m5. Default fiveSigmaDepth.
    mjdCol: str, optional
        Column name for exposure time dates. Default observationStartMJD.
    filterCol: str, optional
        Column name for filter. Default filter.
    seeingCol: str, optional
        Column name for seeing (assumed FWHM). Default seeingFwhmGeom.
    rmag: float, optional
        Magnitude of fiducial star in r filter.  Other filters are scaled using sedTemplate keyword.
        Default 20.0
    sedTemplate: str, optional
        Template to use (can be 'flat' or 'O','B','A','F','G','K','M'). Default 'flat'.
    atm_err: float, optional
        Centroiding error due to atmosphere in arcsec. Default 0.01 (arcseconds).
    thetaRange: float, optional
        Range of parallax offset angles to demand (in radians). Default=0 (means no range requirement).
    snrLimit: float, optional
        Only include points above the snrLimit when computing thetaRange. Default 5.

    Returns
    --------
    metricValue: float
        Returns a weighted mean of the length of the parallax factor vectors.
        Values near 1 imply that the points are well distributed.
        Values near 0 imply that the parallax phase coverage is bad.
        Near the ecliptic, uniform sampling results in metric values of about 0.5.

    Notes
    -----
    Uses the ParallaxFactor stacker to calculate ra_pi_amp and dec_pi_amp.
    """
    def __init__(self, metricName='ParallaxCoverageMetric', m5Col='fiveSigmaDepth',
                 mjdCol='observationStartMJD', filterCol='filter', seeingCol='seeingFwhmGeom',
                 rmag=20., SedTemplate='flat',
                 atm_err=0.01, thetaRange=0., snrLimit=5, **kwargs):
        cols = ['ra_pi_amp', 'dec_pi_amp', m5Col, mjdCol, filterCol, seeingCol]
        units = 'ratio'
        super(ParallaxCoverageMetric, self).__init__(cols,
                                                     metricName=metricName, units=units,
                                                     **kwargs)
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.mjdCol = mjdCol

        # Demand the range of theta values
        self.thetaRange = thetaRange
        self.snrLimit = snrLimit

        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.mags = {}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(SedTemplate, rmag=rmag)
        self.atm_err = atm_err
        caption = "Parallax factor coverage for an r=%.2f star (0 is bad, 0.5-1 is good). " % (rmag)
        caption += "One expects the parallax factor coverage to vary because stars on the ecliptic "
        caption += "can be observed when they have no parallax offset while stars at the pole are always "
        caption += "offset by the full parallax offset."""
        self.comment = caption

    def _thetaCheck(self, ra_pi_amp, dec_pi_amp, snr):
        good = np.where(snr >= self.snrLimit)
        theta = np.arctan2(dec_pi_amp[good], ra_pi_amp[good])
        # Make values between 0 and 2pi
        theta = theta-np.min(theta)
        result = 0.
        if np.max(theta) >= self.thetaRange:
            # Check that things are in differnet quadrants
            theta = (theta+np.pi) % 2.*np.pi
            theta = theta-np.min(theta)
            if np.max(theta) >= self.thetaRange:
                result = 1
        return result

    def _computeWeights(self, dataSlice, snr):
        # Compute centroid uncertainty in each visit
        position_errors = np.sqrt(mafUtils.astrom_precision(dataSlice[self.seeingCol],
                                                            snr)**2+self.atm_err**2)
        weights = 1./position_errors**2
        return weights

    def _weightedR(self, dec_pi_amp, ra_pi_amp, weights):
        ycoord = dec_pi_amp-np.average(dec_pi_amp, weights=weights)
        xcoord = ra_pi_amp-np.average(ra_pi_amp, weights=weights)
        radius = np.sqrt(xcoord**2+ycoord**2)
        aveRad = np.average(radius, weights=weights)
        return aveRad

    def run(self, dataSlice, slicePoint=None):
        if np.size(dataSlice) < 2:
            return self.badval

        filters = np.unique(dataSlice[self.filterCol])
        filters = [str(f) for f in filters]
        snr = np.zeros(len(dataSlice), dtype='float')
        # compute SNR for all observations
        for filt in filters:
            inFilt = np.where(dataSlice[self.filterCol] == filt)
            snr[inFilt] = mafUtils.m52snr(self.mags[str(filt)], dataSlice[self.m5Col][inFilt])

        weights = self._computeWeights(dataSlice, snr)
        aveR = self._weightedR(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], weights)
        if self.thetaRange > 0:
            thetaCheck = self._thetaCheck(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], snr)
        else:
            thetaCheck = 1.
        result = aveR*thetaCheck
        return result


class ParallaxDcrDegenMetric(BaseMetric):
    """Use the full parallax and DCR displacement vectors to find if they are degenerate.

    Parameters
    ----------
    metricName : str, optional
        Default 'ParallaxDcrDegenMetric'.
    seeingCol : str, optional
        Default 'FWHMgeom'
    m5Col : str, optional
        Default 'fiveSigmaDepth'
    filterCol : str
        Default 'filter'
    atm_err : float
        Minimum error in photometry centroids introduced by the atmosphere (arcseconds). Default 0.01.
    rmag : float
        r-band magnitude of the fiducual star that is being used (mag).
    SedTemplate : str
        The SED template to use for fiducia star colors, passed to rubin_sim.utils.stellarMags.
        Default 'flat'
    tol : float
        Tolerance for how well curve_fit needs to work before believing the covariance result.
        Default 0.05.

    Returns
    -------
    metricValue : float
        Returns the correlation coefficient between the best-fit parallax amplitude and DCR amplitude.
        The RA and Dec offsets are fit simultaneously. Values close to zero are good, values close to +/- 1
        are bad. Experience with fitting Monte Carlo simulations suggests the astrometric fits start
        becoming poor around a correlation of 0.7.
    """
    def __init__(self, metricName='ParallaxDcrDegenMetric', seeingCol='seeingFwhmGeom',
                 m5Col='fiveSigmaDepth', atm_err=0.01, rmag=20., SedTemplate='flat',
                 filterCol='filter', tol=0.05, **kwargs):
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.tol = tol
        units = 'Correlation'
        # just put all the columns that all the stackers will need here?
        cols = ['ra_pi_amp', 'dec_pi_amp', 'ra_dcr_amp', 'dec_dcr_amp',
                seeingCol, m5Col]
        super(ParallaxDcrDegenMetric, self).__init__(cols, metricName=metricName, units=units,
                                                     **kwargs)
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        self.mags = {}
        if SedTemplate == 'flat':
            for f in self.filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(SedTemplate, rmag=rmag)
        self.atm_err = atm_err

    def _positions(self, x, a, b):
        """
        Function to find parallax and dcr amplitudes

        x should be a vector with [[parallax_x1, parallax_x2..., parallax_y1, parallax_y2...],
        [dcr_x1, dcr_x2..., dcr_y1, dcr_y2...]]
        """
        result = a*x[0, :] + b*x[1, :]
        return result

    def run(self, dataSlice, slicePoint=None):
        # The idea here is that we calculate position errors (in RA and Dec) for all observations.
        # Then we generate arrays of the parallax offsets (delta RA parallax = ra_pi_amp, etc)
        #  and the DCR offsets (delta RA DCR = ra_dcr_amp, etc), and just add them together into one
        #  RA  (and Dec) offset. Then, we try to fit for how we combined these offsets, but while
        #  considering the astrometric noise. If we can figure out that we just added them together
        # (i.e. the curve_fit result is [a=1, b=1] for the function _positions above)
        # then we should be able to disentangle the parallax and DCR offsets when fitting 'for real'.
        # compute SNR for all observations
        snr = np.zeros(len(dataSlice), dtype='float')
        for filt in self.filters:
            inFilt = np.where(dataSlice[self.filterCol] == filt)
            snr[inFilt] = mafUtils.m52snr(self.mags[filt], dataSlice[self.m5Col][inFilt])
        # Compute the centroiding uncertainties
        # Note that these centroiding uncertainties depend on the physical size of the PSF, thus
        # we are using seeingFwhmGeom for these metrics, not seeingFwhmEff.
        position_errors = np.sqrt(mafUtils.astrom_precision(dataSlice[self.seeingCol], snr)**2 +
                                  self.atm_err**2)
        # Construct the vectors of RA/Dec offsets. xdata is the "input data". ydata is the "output".
        xdata = np.empty((2, dataSlice.size * 2), dtype=float)
        xdata[0, :] = np.concatenate((dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp']))
        xdata[1, :] = np.concatenate((dataSlice['ra_dcr_amp'], dataSlice['dec_dcr_amp']))
        ydata = np.sum(xdata, axis=0)
        # Use curve_fit to compute covariance between parallax and dcr amplitudes
        # Set the initial guess slightly off from the correct [1,1] to make sure it iterates.
        popt, pcov = curve_fit(self._positions, xdata, ydata, p0=[1.1, 0.9],
                               sigma=np.concatenate((position_errors, position_errors)),
                               absolute_sigma=True)
        # Catch if the fit failed to converge on the correct solution.
        if np.max(np.abs(popt - np.array([1., 1.]))) > self.tol:
            return self.badval
        # Covariance between best fit parallax amplitude and DCR amplitude.
        cov = pcov[1, 0]
        # Convert covarience between parallax and DCR amplitudes to normalized correlation
        perr = np.sqrt(np.diag(pcov))
        correlation = cov/(perr[0]*perr[1])
        result = correlation
        # This can throw infs.
        if np.isinf(result):
            result = self.badval
        return result


def calcDist_cosines(RA1, Dec1, RA2, Dec2):
    # Taken from simSelfCalib.py
    """Calculates distance on a sphere using spherical law of cosines.

    Give this function RA/Dec values in radians. Returns angular distance(s), in radians.
    Note that since this is all numpy, you could input arrays of RA/Decs."""
    # This formula can have rounding errors for case where distances are small.
    # Oh, the joys of wikipedia - http://en.wikipedia.org/wiki/Great-circle_distance
    # For the purposes of these calculations, this is probably accurate enough.
    D = np.sin(Dec2)*np.sin(Dec1) + np.cos(Dec1)*np.cos(Dec2)*np.cos(RA2-RA1)
    D = np.arccos(D)
    return D


class RadiusObsMetric(BaseMetric):
    """find the radius in the focal plane. returns things in degrees."""

    def __init__(self, metricName='radiusObs', raCol='fieldRA', decCol='fieldDec',
                 units='radians', **kwargs):
        self.raCol = raCol
        self.decCol = decCol
        super(RadiusObsMetric, self).__init__(col=[self.raCol, self.decCol],
                                              metricName=metricName, units=units, **kwargs)

    def run(self, dataSlice, slicePoint):
        ra = slicePoint['ra']
        dec = slicePoint['dec']
        distances = calcDist_cosines(ra, dec, np.radians(dataSlice[self.raCol]),
                                     np.radians(dataSlice[self.decCol]))
        distances = np.degrees(distances)
        return distances

    def reduceMean(self, distances):
        return np.mean(distances)

    def reduceRMS(self, distances):
        return np.std(distances)

    def reduceFullRange(self, distances):
        return np.max(distances)-np.min(distances)
