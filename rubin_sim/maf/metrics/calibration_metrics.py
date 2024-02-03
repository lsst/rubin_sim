__all__ = (
    "ParallaxMetric",
    "ProperMotionMetric",
    "RadiusObsMetric",
    "ParallaxCoverageMetric",
    "ParallaxDcrDegenMetric",
)

import numpy as np
import rubin_scheduler.utils as utils
from scipy.optimize import curve_fit

import rubin_sim.maf.utils as mafUtils

from .base_metric import BaseMetric


class ParallaxMetric(BaseMetric):
    """Calculate the uncertainty in a parallax measurement
    given a series of observations.

    Uses columns ra_pi_amp and dec_pi_amp,
    calculated by the ParallaxFactorStacker.

    Parameters
    ----------
    m5_col : `str`, optional
        The default column name for m5 information in the input data.
    filter_col : `str`, optional
        The column name for the filter information.
    seeing_col : `str`, optional
        The column name for the seeing information.
        Since the astrometry errors are based on the physical size of the PSF,
        this should be the FWHM of the physical psf, e.g. seeingFwhmGeom.
    rmag : `float`, optional
        The r magnitude of the fiducial star in r band.
        Other filters are scaled using sedTemplate keyword.
    SedTemplate : `str`, optional
        The template to use. This can be 'flat' or 'O','B','A','F','G','K','M'.
    atm_err : `float`, optional
        The expected centroiding error due to the atmosphere, in arcseconds.
        Default 0.01.
    normalize : `bool`, optional
        Compare the astrometric uncertainty to the uncertainty
        that would result if half the observations were taken at the start
        and half at the end.
        A perfect survey will have a value close to 1, while
        a poorly scheduled survey will be close to 0.
    badval : `float`, optional
        The value to return when the metric value cannot be calculated.
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        seeing_col="seeingFwhmGeom",
        rmag=20.0,
        sed_template="flat",
        badval=-666,
        atm_err=0.01,
        normalize=False,
        **kwargs,
    ):
        cols = [m5_col, filter_col, seeing_col, "ra_pi_amp", "dec_pi_amp"]
        if normalize:
            units = "ratio"
        else:
            units = "mas"
        super().__init__(cols, units=units, badval=badval, **kwargs)
        # set return type
        self.m5_col = m5_col
        self.seeing_col = seeing_col
        self.filter_col = filter_col
        filters = ["u", "g", "r", "i", "z", "y"]
        self.mags = {}
        if sed_template == "flat":
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(sed_template, rmag=rmag)
        self.atm_err = atm_err
        self.normalize = normalize
        self.comment = (
            "Estimated uncertainty in parallax measurement "
            "(assuming no proper motion or that proper motion "
        )
        self.comment += (
            "is well fit). Uses measurements in all bandpasses, "
            "and estimates astrometric error based on SNR "
        )
        self.comment += "in each visit. "
        if sed_template == "flat":
            self.comment += "Assumes a flat SED. "
        if self.normalize:
            self.comment += (
                "This normalized version of the metric displays the "
                "estimated uncertainty in the parallax measurement, "
            )
            self.comment += "divided by the minimum parallax uncertainty possible "
            self.comment += "(if all visits were six months apart). "
            self.comment += "Values closer to 1 indicate more optimal " "scheduling for parallax measurement."

    def _final_sigma(self, position_errors, ra_pi_amp, dec_pi_amp):
        """Assume parallax in RA and DEC are fit independently, then combined.
        All inputs assumed to be arcsec"""
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_a = position_errors / ra_pi_amp
            sigma_b = position_errors / dec_pi_amp
            sigma_ra = np.sqrt(1.0 / np.sum(1.0 / sigma_a**2))
            sigma_dec = np.sqrt(1.0 / np.sum(1.0 / sigma_b**2))
            # Combine RA and Dec uncertainties, convert to mas
            sigma = np.sqrt(1.0 / (1.0 / sigma_ra**2 + 1.0 / sigma_dec**2)) * 1e3
        return sigma

    def run(self, data_slice, slice_point=None):
        filters = np.unique(data_slice[self.filter_col])
        if hasattr(filters[0], "decode"):
            filters = [str(f.decode("utf-8")) for f in filters]
        snr = np.zeros(len(data_slice), dtype="float")
        # compute SNR for all observations
        for filt in filters:
            good = np.where(data_slice[self.filter_col] == filt)
            snr[good] = mafUtils.m52snr(self.mags[str(filt)], data_slice[self.m5_col][good])
        position_errors = mafUtils.astrom_precision(data_slice[self.seeing_col], snr, self.atm_err)
        sigma = self._final_sigma(position_errors, data_slice["ra_pi_amp"], data_slice["dec_pi_amp"])
        if self.normalize:
            # Leave the dec parallax as zero since one can't have
            # ra and dec maximized at the same time.
            sigma = (
                self._final_sigma(
                    position_errors,
                    data_slice["ra_pi_amp"] * 0 + 1.0,
                    data_slice["dec_pi_amp"] * 0,
                )
                / sigma
            )
        return sigma


class ProperMotionMetric(BaseMetric):
    """Calculate the uncertainty in the returned proper motion.

    This metric assumes gaussian errors in the astrometry measurements.

    Parameters
    ----------
    metricName : `str`, optional
        Default 'properMotion'.
    m5_col : `str`, optional
        The default column name for m5 information in the input data.
        Default fiveSigmaDepth.
    mjd_col : `str`, optional
        The column name for the exposure time. Default observationStartMJD.
    filterCol : `str`, optional
        The column name for the filter information. Default filter.
    seeing_col : `str`, optional
        The column name for the seeing information.
        Since the astrometry errors are based on the physical
        size of the PSF, this should be the FWHM of the physical psf.
        Default seeingFwhmGeom.
    rmag : `float`, optional
        The r magnitude of the fiducial star in r band.
        Other filters are sclaed using sedTemplate keyword.
        Default 20.0
    SedTemplate : `str`, optional
        The template to use. This can be 'flat' or 'O','B','A','F','G','K','M'.
        Default flat.
    atm_err : `float`, optional
        The expected centroiding error due to the atmosphere, in arcseconds.
        Default 0.01.
    normalize : `bool`, optional
        Compare the astrometric uncertainty to the uncertainty that would
        result if half the observations were taken at the start and half
        at the end. A perfect survey will have a value close to 1, while
        a poorly scheduled survey will be close to 0. Default False.
    baseline : `float`, optional
        The length of the survey used for the normalization, in years.
        Default 10.
    badval : `float`, optional
        The value to return when the metric value cannot be calculated.
        Default -666.
    """

    def __init__(
        self,
        metric_name="properMotion",
        m5_col="fiveSigmaDepth",
        mjd_col="observationStartMJD",
        filter_col="filter",
        seeing_col="seeingFwhmGeom",
        rmag=20.0,
        sed_template="flat",
        badval=-666,
        atm_err=0.01,
        normalize=False,
        baseline=10.0,
        **kwargs,
    ):
        cols = [m5_col, mjd_col, filter_col, seeing_col]
        if normalize:
            units = "ratio"
        else:
            units = "mas/yr"
        super(ProperMotionMetric, self).__init__(
            col=cols, metric_name=metric_name, units=units, badval=badval, **kwargs
        )
        # set return type
        self.mjd_col = mjd_col
        self.seeing_col = seeing_col
        self.m5_col = m5_col
        filters = ["u", "g", "r", "i", "z", "y"]
        self.mags = {}
        if sed_template == "flat":
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(sed_template, rmag=rmag)
        self.atm_err = atm_err
        self.normalize = normalize
        self.baseline = baseline
        self.comment = (
            "Estimated uncertainty of the proper motion fit "
            "(assuming no parallax or that parallax is well fit). "
        )
        self.comment += (
            "Uses visits in all bands, and generates approximate "
            "astrometric errors using the SNR in each visit. "
        )
        if sed_template == "flat":
            self.comment += "Assumes a flat SED. "
        if self.normalize:
            self.comment += (
                "This normalized version of the metric represents " "the estimated uncertainty in the proper "
            )
            self.comment += "motion divided by the minimum uncertainty possible "
            self.comment += "(if all visits were obtained on the first and last days of the survey). "
            self.comment += "Values closer to 1 indicate more optimal scheduling."

    def run(self, data_slice, slice_point=None):
        filters = np.unique(data_slice["filter"])
        filters = [str(f) for f in filters]
        precis = np.zeros(data_slice.size, dtype="float")
        for f in filters:
            observations = np.where(data_slice["filter"] == f)
            if np.size(observations[0]) < 2:
                precis[observations] = self.badval
            else:
                snr = mafUtils.m52snr(self.mags[f], data_slice[self.m5_col][observations])
                precis[observations] = mafUtils.astrom_precision(
                    data_slice[self.seeing_col][observations], snr, self.atm_err
                )
        good = np.where(precis != self.badval)
        result = mafUtils.sigma_slope(data_slice[self.mjd_col][good], precis[good])
        result = result * 365.25 * 1e3  # Convert to mas/yr
        if (self.normalize) & (good[0].size > 0):
            new_dates = data_slice[self.mjd_col][good] * 0
            n_dates = new_dates.size
            new_dates[n_dates // 2 :] = self.baseline * 365.25
            result = (mafUtils.sigma_slope(new_dates, precis[good]) * 365.25 * 1e3) / result
        # Observations that are very close together can still fail
        if np.isnan(result):
            result = self.badval
        return result


class ParallaxCoverageMetric(BaseMetric):
    """Check how well the parallax factor is distributed.

    Subtracts the weighted mean position of the
    parallax offsets, then computes the weighted mean radius of the points.
    If points are well distributed, the mean radius will be near 1.
    If phase coverage is bad, radius will be close to zero.

    For points on the Ecliptic, uniform sampling should result in a
    metric value of ~0.5.
    At the poles, uniform sampling would result in a metric value of ~1.
    Conceptually, it is helpful to remember that the parallax motion of a
    star at the pole is a (nearly circular) ellipse while the motion of a
    star on the ecliptic is a straight line. Thus, any pair of observations
    separated by 6 months will give the full parallax range for a star on
    the pole but only observations on very specific dates will give the
    full range for a star on the ecliptic.

    Optionally also demand that there are observations above the snr_limit
    kwarg spanning theta_range radians.

    Parameters
    ----------
    m5_col : `str`, optional
        Column name for individual visit m5. Default fiveSigmaDepth.
    mjd_col : `str`, optional
        Column name for exposure time dates. Default observationStartMJD.
    filter_col : `str`, optional
        Column name for filter. Default filter.
    seeing_col : `str`, optional
        Column name for seeing (assumed FWHM). Default seeingFwhmGeom.
    rmag : `float`, optional
        Magnitude of fiducial star in r filter.
        Other filters are scaled using sedTemplate keyword.
        Default 20.0
    sedTemplate : `str`, optional
        Template to use (can be 'flat' or 'O','B','A','F','G','K','M').
        Default 'flat'.
    atm_err : `float`, optional
        Centroiding error due to atmosphere in arcsec.
        Default 0.01 (arcseconds).
    theta_range : `float`, optional
        Range of parallax offset angles to demand (in radians).
        Default=0 (means no range requirement).
    snr_limit : `float`, optional
        Only include points above the snr_limit when computing theta_range.
        Default 5.

    Returns
    --------
    metricValu e: `float`
        Returns a weighted mean of the length of the parallax factor vectors.
        Values near 1 imply that the points are well distributed.
        Values near 0 imply that the parallax phase coverage is bad.
        Near the ecliptic, uniform sampling results in metric values
        of about 0.5.

    Notes
    -----
    Uses the ParallaxFactor stacker to calculate ra_pi_amp and dec_pi_amp.
    """

    def __init__(
        self,
        metric_name="ParallaxCoverageMetric",
        m5_col="fiveSigmaDepth",
        mjd_col="observationStartMJD",
        filter_col="filter",
        seeing_col="seeingFwhmGeom",
        rmag=20.0,
        sed_template="flat",
        atm_err=0.01,
        theta_range=0.0,
        snr_limit=5,
        **kwargs,
    ):
        cols = ["ra_pi_amp", "dec_pi_amp", m5_col, mjd_col, filter_col, seeing_col]
        units = "ratio"
        super(ParallaxCoverageMetric, self).__init__(cols, metric_name=metric_name, units=units, **kwargs)
        self.m5_col = m5_col
        self.seeing_col = seeing_col
        self.filter_col = filter_col
        self.mjd_col = mjd_col

        # Demand the range of theta values
        self.theta_range = theta_range
        self.snr_limit = snr_limit

        filters = ["u", "g", "r", "i", "z", "y"]
        self.mags = {}
        if sed_template == "flat":
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(sed_template, rmag=rmag)
        self.atm_err = atm_err
        caption = "Parallax factor coverage for an r=%.2f star (0 is bad, 0.5-1 is good). " % (rmag)
        caption += "One expects the parallax factor coverage to vary because stars on the ecliptic "
        caption += "can be observed when they have no parallax offset while stars at the pole are always "
        caption += "offset by the full parallax offset." ""
        self.comment = caption

    def _theta_check(self, ra_pi_amp, dec_pi_amp, snr):
        good = np.where(snr >= self.snr_limit)
        theta = np.arctan2(dec_pi_amp[good], ra_pi_amp[good])
        # Make values between 0 and 2pi
        theta = theta - np.min(theta)
        result = 0.0
        if np.max(theta) >= self.theta_range:
            # Check that things are in different quadrants
            theta = (theta + np.pi) % 2.0 * np.pi
            theta = theta - np.min(theta)
            if np.max(theta) >= self.theta_range:
                result = 1
        return result

    def _compute_weights(self, data_slice, snr):
        # Compute centroid uncertainty in each visit
        position_errors = mafUtils.astrom_precision(data_slice[self.seeing_col], snr, self.atm_err)
        weights = 1.0 / position_errors**2
        return weights

    def _weighted_r(self, dec_pi_amp, ra_pi_amp, weights):
        ycoord = dec_pi_amp - np.average(dec_pi_amp, weights=weights)
        xcoord = ra_pi_amp - np.average(ra_pi_amp, weights=weights)
        radius = np.sqrt(xcoord**2 + ycoord**2)
        ave_rad = np.average(radius, weights=weights)
        return ave_rad

    def run(self, data_slice, slice_point=None):
        if np.size(data_slice) < 2:
            return self.badval

        filters = np.unique(data_slice[self.filter_col])
        filters = [str(f) for f in filters]
        snr = np.zeros(len(data_slice), dtype="float")
        # compute SNR for all observations
        for filt in filters:
            in_filt = np.where(data_slice[self.filter_col] == filt)
            snr[in_filt] = mafUtils.m52snr(self.mags[str(filt)], data_slice[self.m5_col][in_filt])

        weights = self._compute_weights(data_slice, snr)
        ave_r = self._weighted_r(data_slice["ra_pi_amp"], data_slice["dec_pi_amp"], weights)
        if self.theta_range > 0:
            theta_check = self._theta_check(data_slice["ra_pi_amp"], data_slice["dec_pi_amp"], snr)
        else:
            theta_check = 1.0
        result = ave_r * theta_check
        return result


class ParallaxDcrDegenMetric(BaseMetric):
    """Use the full parallax and DCR displacement vectors to find if they
    are degenerate.

    Parameters
    ----------
    metricName : `str`, optional
        Default 'ParallaxDcrDegenMetric'.
    seeing_col : `str`, optional
        Default 'FWHMgeom'
    m5_col : `str`, optional
        Default 'fiveSigmaDepth'
    filter_col : `str`
        Default 'filter'
    atm_err : `float`
        Minimum error in photometry centroids introduced by the atmosphere
        (arcseconds). Default 0.01.
    rmag : `float`
        r-band magnitude of the fiducual star that is being used (mag).
    SedTemplate : `str`
        The SED template to use for fiducia star colors,
        passed to rubin_scheduler.utils.stellarMags.
        Default 'flat'
    tol : `float`
        Tolerance for how well curve_fit needs to work before
        believing the covariance result.
        Default 0.05.

    Returns
    -------
    metricValue : `float`
        Returns the correlation coefficient between the best-fit parallax
        amplitude and DCR amplitude.
        The RA and Dec offsets are fit simultaneously.
        Values close to zero are good, values close to +/- 1 are bad.
        Experience with fitting Monte Carlo simulations suggests the
        astrometric fits start becoming poor around a correlation of 0.7.
    """

    def __init__(
        self,
        metric_name="ParallaxDcrDegenMetric",
        seeing_col="seeingFwhmGeom",
        m5_col="fiveSigmaDepth",
        atm_err=0.01,
        rmag=20.0,
        sed_template="flat",
        filter_col="filter",
        tol=0.05,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.seeing_col = seeing_col
        self.filter_col = filter_col
        self.tol = tol
        units = "Correlation"
        # just put all the columns that all the stackers will need here?
        cols = [
            "ra_pi_amp",
            "dec_pi_amp",
            "ra_dcr_amp",
            "dec_dcr_amp",
            seeing_col,
            m5_col,
        ]
        super(ParallaxDcrDegenMetric, self).__init__(cols, metric_name=metric_name, units=units, **kwargs)
        self.mags = {}
        if sed_template == "flat":
            for f in ["u", "g", "r", "i", "z", "y"]:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(sed_template, rmag=rmag)
        self.atm_err = atm_err

    def _positions(self, x, a, b):
        """Function to find parallax and dcr amplitudes

        x should be a vector with [[parallax_x1, parallax_x2...,
        parallax_y1, parallax_y2...],
        [dcr_x1, dcr_x2..., dcr_y1, dcr_y2...]]
        """
        result = a * x[0, :] + b * x[1, :]
        return result

    def run(self, data_slice, slice_point=None):
        # The idea here is that we calculate position errors (in RA and Dec)
        # for all observations. Then we generate arrays of the parallax
        # offsets (delta RA parallax = ra_pi_amp, etc) and the DCR offsets
        # (delta RA DCR = ra_dcr_amp, etc), and just add them together into one
        # RA (and Dec) offset. Then, we try to fit for how we combined these
        # offsets, but while considering the astrometric noise. If we can
        # figure out that we just added them together
        # (i.e. the curve_fit result is [a=1, b=1] for the function
        # _positions above) then we should be able to disentangle the
        # parallax and DCR offsets when fitting 'for real'.
        # compute SNR for all observations
        snr = np.zeros(len(data_slice), dtype="float")
        for filt in np.unique(data_slice[self.filter_col]):
            in_filt = np.where(data_slice[self.filter_col] == filt)
            snr[in_filt] = mafUtils.m52snr(self.mags[filt], data_slice[self.m5_col][in_filt])
        # Compute the centroiding uncertainties
        # Note that these centroiding uncertainties depend on the physical
        # size of the PSF, thus we are using seeingFwhmGeom for these metrics,
        # not seeingFwhmEff.
        position_errors = mafUtils.astrom_precision(data_slice[self.seeing_col], snr, self.atm_err)
        # Construct the vectors of RA/Dec offsets. xdata is the "input data".
        # ydata is the "output".
        xdata = np.empty((2, data_slice.size * 2), dtype=float)
        xdata[0, :] = np.concatenate((data_slice["ra_pi_amp"], data_slice["dec_pi_amp"]))
        xdata[1, :] = np.concatenate((data_slice["ra_dcr_amp"], data_slice["dec_dcr_amp"]))
        ydata = np.sum(xdata, axis=0)
        # Use curve_fit to compute covariance between parallax
        # and dcr amplitudes
        # Set the initial guess slightly off from the correct [1,1] to
        # make sure it iterates.
        popt, pcov = curve_fit(
            self._positions,
            xdata,
            ydata,
            p0=[1.1, 0.9],
            sigma=np.concatenate((position_errors, position_errors)),
            absolute_sigma=True,
        )
        # Catch if the fit failed to converge on the correct solution.
        if np.max(np.abs(popt - np.array([1.0, 1.0]))) > self.tol:
            return self.badval
        # Covariance between best fit parallax amplitude and DCR amplitude.
        cov = pcov[1, 0]
        # Convert covariance between parallax and DCR amplitudes to normalized
        # correlation
        perr = np.sqrt(np.diag(pcov))
        correlation = cov / (perr[0] * perr[1])
        result = correlation
        # This can throw infs.
        if np.isinf(result):
            result = self.badval
        return result


def calc_dist_cosines(ra1, dec1, ra2, dec2):
    """Calculates distance on a sphere using spherical law of cosines.

    Note: floats can be replaced by numpy arrays of RA/Dec.
    For very small distances, rounding errors may cause distance errors.

    Parameters
    ----------
    ra1, dec1 : `float`, `float`
        RA and Dec of one point. (radians)
    ra2, dec2 : `float`, `float`
        RA and Dec of another point. (radians)

    Returns
    -------
    distance : `float`
        Angular distance between the points in radians.
    """
    # This formula can have rounding errors for case where distances are small.
    # Oh, the joys of wikipedia -
    # http://en.wikipedia.org/wiki/Great-circle_distance
    # For the purposes of these calculations, this is probably accurate enough.
    D = np.sin(dec2) * np.sin(dec1) + np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1)
    D = np.arccos(D)
    return D


class RadiusObsMetric(BaseMetric):
    """Evaluate slice point radial position in the focal plane of each visit,
    reducing to the mean, rms and full range of these radial distances.
    """

    def __init__(
        self, metric_name="radiusObs", ra_col="fieldRA", dec_col="fieldDec", units="radians", **kwargs
    ):
        self.ra_col = ra_col
        self.dec_col = dec_col
        super(RadiusObsMetric, self).__init__(
            col=[self.ra_col, self.dec_col], metric_name=metric_name, units=units, **kwargs
        )

    def run(self, data_slice, slice_point):
        ra = slice_point["ra"]
        dec = slice_point["dec"]
        distances = calc_dist_cosines(
            ra,
            dec,
            np.radians(data_slice[self.ra_col]),
            np.radians(data_slice[self.dec_col]),
        )
        distances = np.degrees(distances)
        return distances

    def reduce_mean(self, distances):
        return np.mean(distances)

    def reduce_rms(self, distances):
        return np.std(distances)

    def reduce_full_range(self, distances):
        return np.max(distances) - np.min(distances)
