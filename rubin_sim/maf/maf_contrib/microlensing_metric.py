__all__ = (
    "generate_microlensing_slicer",
    "MicrolensingMetric",
    "microlensing_amplification",
    "microlensing_amplification_fsfb",
    "info_peak_before_t0",
    "fisher_matrix",
    "coefficients_pspl",
    "coefficients_fsfb",
)

import os
from copy import deepcopy

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import hpid2_ra_dec

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
from rubin_sim.maf.utils import m52snr


# Via Natasha Abrams nsabrams@college.harvard.edu
def microlensing_amplification(t, impact_parameter=1, crossing_time=1825.0, peak_time=100, blending_factor=1):
    """The microlensing amplification

    Parameters
    ----------
    t : `float`
        The time of observation (days)
    impact_parameter : `float`
        The impact paramter (0 means big amplification)
    crossing_time : `float`
        Einstein crossing time (days)
    peak_time : `float`
        The peak time (days)
    blending_factor: `float`
        The blending factor where 1 is unblended
    """

    lightcurve_u = np.sqrt(impact_parameter**2 + ((t - peak_time) ** 2 / crossing_time**2))
    amplified_mag = (lightcurve_u**2 + 2) / (
        lightcurve_u * np.sqrt(lightcurve_u**2 + 4)
    ) * blending_factor + (1 - blending_factor)

    return amplified_mag


def microlensing_amplification_fsfb(t, impact_parameter=1, crossing_time=1825.0, peak_time=100, fs=20, fb=20):
    """The microlensing amplification in terms of source flux and blend flux

    Parameters
    ----------
    t : `float`
        The time of observation (days)
    impact_parameter : `float`
        The impact paramter (0 means big amplification)
    crossing_time : `float`
        Einstein crossing time (days)
    peak_time : `float`
        The peak time (days)
    """

    lightcurve_u = np.sqrt(impact_parameter**2 + ((t - peak_time) ** 2 / crossing_time**2))
    amplified_mag = (lightcurve_u**2 + 2) / (lightcurve_u * np.sqrt(lightcurve_u**2 + 4)) * fs + fb

    return amplified_mag


def info_peak_before_t0(impact_parameter=1, crossing_time=100.0):
    """Time of Maximum Information before peak

    The point of maximum information before peak indicates when in a
    lightcurve with constant snr the contribution to the
    characterization of the event timescale tE is at maximum. After that
    point in timem the timescale characterization is decisively improved.
    via Markus Hundertmark markus.hundertmark@uni-heidelberg.de

    Parameters
    ----------
    impact_parameter : `float`
        The impact paramter (0 means big amplification)
    crossing_time : `float`
        Einstein crossing time (days)
    """

    optimal_time = (
        crossing_time
        * np.sqrt(
            -(impact_parameter**2) + np.sqrt(9 * impact_parameter**4 + 36 * impact_parameter**2 + 4) - 2
        )
        / 2
    )
    return np.array(optimal_time)


def coefficients_pspl(t, t0, te, u0, fs, fb):
    """Coefficients for calculating the first derivatives wrt t0,te,u0
    in the Fisher matrix in an effective way, optimized with sympy and cse
    via Markus Hundertmark markus.hundertmark@uni-heidelberg.de

    Parameters
    ----------
    t : `float`
        The time of observation (days usually as JD or HJD)
    u0 : `float`
        The impact parameter (0 means high magnification)
    te : `float`
        Einstein crossing time (days)
    t0 : `float`
        Time of peak (days usually as JD or HJD)
    fs : `float`
        Source flux (counts/s but here fs = 10.**(-0.4*mag_source)
        in the respective passband)
    fb : `float`
        Blend flux (counts/s but here fs = 10.**(-0.4*mag_blend)
        in the respective passband)
    """
    x0 = te ** (-2)
    x1 = (t - t0) ** 2
    x2 = u0**2 + x0 * x1
    x3 = 1 / np.sqrt(x2)
    x4 = 2 * x3
    x5 = x2 + 4
    x6 = fs / np.sqrt(x5)
    x7 = x1 / te**3
    x8 = x6 * x7
    x9 = x2 + 2
    x10 = x9 / x2 ** (3 / 2)
    x11 = fs * x3 * x9 / x5 ** (3 / 2)
    x12 = u0 * x6
    x13 = x0 * (-2 * t + 2 * t0)
    x14 = x13 * x6

    c0 = x10 * x8 + x11 * x7 - x4 * x8
    c1 = -u0 * x11 - x10 * x12 + x12 * x4
    c2 = -1 / 2 * x10 * x14 - 1 / 2 * x11 * x13 + x14 * x3

    return [c2, c1, c0]


def coefficients_fsfb(t, t0, te, u0, fs, fb):
    """Coefficients for calculating the first derivatives wrt fs,fb
    in the Fisher (information) matrix in an effective way, optimized
    with sympy and cse this function needs to be evaluated for each
    passband respectively.
    via Markus Hundertmark markus.hundertmark@uni-heidelberg.de

    Parameters
    ----------
    t : `float`
        The time of observation (days usually as JD or HJD)
    u0 : `float`
        The impact parameter (0 means high magnification)
    te : `float`
        Einstein crossing time (days)
    t0 : `float`
        Time of peak (days usually as JD or HJD)
    fs : `float`
        Source flux (counts/s but here fs = 10.**(-0.4*mag_source)
        in the respective passband)
    fb : `float`
        Blend flux (counts/s but here fs = 10.**(-0.4*mag_blend)
        in the respective passband)
    """
    x0 = u0**2 + (t - t0) ** 2 / te**2
    c0 = (x0 + 2) / (np.sqrt(x0) * np.sqrt(x0 + 4))
    c1 = 1
    return [c1, c0]


def fisher_matrix(t, t0, te, u0, fs, fb, snr, filters="ugriz", filter_name="i"):
    """The Fisher (information) matrix relying on first derivatives
    wrt t0,te,u0,fs,fb, relying on with cse optimized coefficents.
    This implementation for the non-linear magnification is subject to the
    respective underlying assumptions, i.e. relatively small uncertainties.
    NB the Fisher matrix is using first derivatives, alternatively one
    could consider second derivatives the resulting covariance matrix as
    inverse of the Fisher matrix is a Cramer Rao lower bound of the
    respective uncertainty.
    The function returns the full information matrix for all t.

    via Markus Hundertmark markus.hundertmark@uni-heidelberg.de

    Parameters
    ----------
    t : `float`
        The time of observation (days usually as JD or HJD)
    u0 : `float`
        The impact parameter (0 means high magnification)
    te : `float`
        Einstein crossing time (days)
    t0 : `float`
        Time of peak (days usually as JD or HJD)
    fs : `float`
        Source flux (counts/s but here fs = 10.**(-0.4*mag_source)
        in the respective passband)
    fb : `float`
        Blend flux (counts/s but here fs = 10.**(-0.4*mag_blend)
        in the respective passband)
    snr : `float`
        Signal to noise ratio in order to infer the flux uncertainty
    """
    # indices are 0:te, 1:u0, 2:t0
    # 3:fs_i, 4:fb_i, 5:fs_i+1, 6: fb_i+1, etc. for ugrizy
    filters = filters
    npars = 2 * len(filters) + 3
    fis_mat = np.zeros((npars, npars))
    filter_index = np.where(filters == filter_name)[0]
    # initialize result matrix
    for i in range(len(t)):
        t_value = t[i]
        # err = flux_err[i]
        matrix = np.zeros((npars, npars))
        derivatives = coefficients_pspl(t_value, t0, te, u0, fs, fb)
        fsfb_derivatives = 2 * len(filters) * [0]
        fbdiff, fsdiff = coefficients_fsfb(t_value, t0, te, u0, fs, fb)
        np.array(fsfb_derivatives)[filter_index * 2] = fsdiff
        np.array(fsfb_derivatives)[filter_index * 2 + 1] = fbdiff
        derivatives = derivatives + fsfb_derivatives
        for idx in range(npars):
            for jdx in range(idx, npars):
                matrix[idx, jdx] = derivatives[idx] * derivatives[jdx]
        # restore symmetrical entries
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))

        # microlenisng_amplification function to replace
        usqr = (t_value - t0) ** 2 / te**2 + u0**2
        mu = (usqr + 2.0) / (usqr * (usqr + 4)) ** 0.5
        # may need to check shape of snr and t
        flux_err = (fs * mu + fb) / snr[i]

        matrix = matrix / (flux_err**2)
        fis_mat = fis_mat + matrix
    return fis_mat


class MicrolensingMetric(metrics.BaseMetric):
    """
    Quantifies detectability of Microlensing events.
    Can also return the number of datapoints within two crossing times
    of the peak of event.

    Parameters
    ----------
    metric_calc : `str`
        Type of metric.
        If metric_calc == 'detect', returns the number of microlensing events
        detected within certain parameters.
        If metric_calc == 'Npts', returns the number of points
        within two crossing times of the peak of the vent.
    pts_needed : `int`
        Number of an object's lightcurve points required to be above the
        5-sigma limiting depth before it is considered detected.
    time_before_peak: `int` or `str`
        Number of days before lightcurve peak to qualify event as triggered.
        If time_before_peak == 'optimal', the number of days before the
        lightcurve peak is the time of maximal information.
    detect: `bool`
        By default we trigger which only looks at points before the peak
        of the lightcurve. When detect = True, observations on either side
        of the lightcurve are considered.

    Notes
    -----
    Expects slice_point to have keys of:
        peak_time : float (days)
        crossing_time : float (days)
        impact_parameter : float (positive)
        blending_factors (optional): float (between 0 and 1)
    """

    def __init__(
        self,
        metric_name="MicrolensingMetric",
        metric_calc="detect",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        night_col="night",
        pts_needed=2,
        mag_dict=None,
        detect_sigma=3.0,
        time_before_peak=0,
        detect=False,
        **kwargs,
    ):
        self.metric_calc = metric_calc
        if metric_calc == "detect":
            self.units = "Detected, 0 or 1"
        elif metric_calc == "Npts":
            self.units = "Npts within 2tE"
        elif metric_calc == "Fisher":
            self.units = "Characterized, 0 or 1"
        else:
            raise Exception('metric_calc must be "detect", "Npts" or "Fisher".')
        if mag_dict is None:
            # Mean mag of stars in each filter from TRIstarDensity map
            mag_dict = {
                "u": 25.2,
                "g": 25.0,
                "r": 24.5,
                "i": 23.4,
                "z": 22.8,
                "y": 22.5,
            }
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.pts_needed = pts_needed
        self.detect_sigma = detect_sigma
        self.time_before_peak = time_before_peak
        self.detect = detect
        self.mags = mag_dict

        cols = [self.mjd_col, self.m5_col, self.filter_col, self.night_col]
        super(MicrolensingMetric, self).__init__(
            col=cols, units=self.units, metric_name=metric_name + "_" + self.metric_calc, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        if self.detect is True and self.time_before_peak > 0:
            raise Exception("When detect = True, time_before_peak must be zero")

        # Generate the lightcurve for this object
        # make t a kind of simple way
        t = data_slice[self.mjd_col] - np.min(data_slice[self.night_col])
        t = t - t.min()

        #
        # Try for if a blending factor slice was added
        if "apparent_m_no_blend_u" in slice_point:
            amplitudes = np.zeros(len(t))
            individual_mags = True
        else:
            if "blending_factor" in slice_point:
                amplitudes = microlensing_amplification(
                    t,
                    impact_parameter=slice_point["impact_parameter"],
                    crossing_time=slice_point["crossing_time"],
                    peak_time=slice_point["peak_time"],
                    blending_factor=slice_point["blending_factor"],
                )
                individual_mags = False
            else:
                amplitudes = microlensing_amplification(
                    t,
                    impact_parameter=slice_point["impact_parameter"],
                    crossing_time=slice_point["crossing_time"],
                    peak_time=slice_point["peak_time"],
                )
                individual_mags = False

        filters = np.unique(data_slice[self.filter_col])
        amplified_mags = amplitudes * 0

        npars = 2 * len(filters) + 3
        fis_mat = np.zeros((npars, npars))
        fis_mat_wzeros = deepcopy(fis_mat)

        for filtername in filters:
            infilt = np.where(data_slice[self.filter_col] == filtername)[0]

            if individual_mags is True:
                fs = slice_point["apparent_m_no_blend_{}".format(filtername)]
                fb = slice_point["apparent_m_{}".format(filtername)] - fs
                amplitudes = microlensing_amplification_fsfb(
                    t,
                    impact_parameter=slice_point["impact_parameter"],
                    crossing_time=slice_point["crossing_time"],
                    peak_time=slice_point["peak_time"],
                    fs=fs,
                    fb=fb,
                )
                self.mags[filtername] = slice_point["apparent_m_{}".format(filtername)]
                amplified_mags[infilt] = self.mags[filtername] - 2.5 * np.log10(amplitudes[infilt])

            else:
                amplified_mags[infilt] = self.mags[filtername] - 2.5 * np.log10(amplitudes[infilt])

        # The SNR of each point in the light curve
        snr = m52snr(amplified_mags, data_slice[self.m5_col])
        # The magnitude uncertainties that go with amplified mags
        mag_uncert = 2.5 * np.log10(1 + 1.0 / snr)

        n_pre = []
        n_post = []
        for filtername in filters:
            if self.metric_calc == "detect":
                if self.time_before_peak == "optimal":
                    time_before_peak_optimal = info_peak_before_t0(
                        slice_point["impact_parameter"], slice_point["crossing_time"]
                    )
                    # observations pre-peak and in the given filter
                    infilt = np.where(
                        (data_slice[self.filter_col] == filtername)
                        & (t < (slice_point["peak_time"] - time_before_peak_optimal))
                    )[0]

                else:
                    # observations pre-peak and in the given filter
                    infilt = np.where(
                        (data_slice[self.filter_col] == filtername)
                        & (t < (slice_point["peak_time"] - self.time_before_peak))
                    )[0]

                # observations post-peak and in the given filter
                outfilt = np.where(
                    (data_slice[self.filter_col] == filtername) & (t > slice_point["peak_time"])
                )[0]
                # Broadcast to calc the mag_i - mag_j
                diffs = amplified_mags[infilt] - amplified_mags[infilt][:, np.newaxis]
                diffs_uncert = np.sqrt(mag_uncert[infilt] ** 2 + mag_uncert[infilt][:, np.newaxis] ** 2)
                diffs_post = amplified_mags[outfilt] - amplified_mags[outfilt][:, np.newaxis]
                diffs_post_uncert = np.sqrt(
                    mag_uncert[outfilt] ** 2 + mag_uncert[outfilt][:, np.newaxis] ** 2
                )

                # Calculating this as a catalog-level detection. In theory,
                # we could have a high SNR template image, so there would be
                # little to no additional uncertianty from the subtraction.

                sigma_above = np.abs(diffs) / diffs_uncert
                sigma_above_post = np.abs(diffs_post) / diffs_post_uncert
                # divide by 2 because array has i,j and j,i
                n_above = np.size(np.where(sigma_above > self.detect_sigma)[0]) / 2
                n_pre.append(n_above)
                n_above_post = np.size(np.where(sigma_above_post > self.detect_sigma)[0]) / 2
                n_post.append(n_above_post)

            elif self.metric_calc == "Npts":
                # observations pre-peak and in the given filter within 2tE
                infilt = np.where(
                    (data_slice[self.filter_col] == filtername)
                    & (t < (slice_point["peak_time"]))
                    & (t > (slice_point["peak_time"] - slice_point["crossing_time"]))
                    & (snr > self.detect_sigma)
                )[0]
                # observations post-peak and in the given filter within 2tE
                outfilt = np.where(
                    (data_slice[self.filter_col] == filtername)
                    & (t > (slice_point["peak_time"]))
                    & (t < (slice_point["peak_time"] + slice_point["crossing_time"]))
                    & (snr > self.detect_sigma)
                )[0]
                n_pre.append(len(infilt))
                n_post.append(len(outfilt))

            elif self.metric_calc == "Fisher":
                if individual_mags is True:
                    fs = slice_point["apparent_m_no_blend_{}".format(filtername)]
                    fb = slice_point["apparent_m_{}".format(filtername)] - fs
                else:
                    fs = self.mags[filtername]
                    # assuming that in populated areas of the galaxy
                    # the blend fraction is 0.5
                    fb = fs

                fis_mat = fis_mat_wzeros + fisher_matrix(
                    t[infilt],
                    t0=slice_point["peak_time"],
                    te=slice_point["crossing_time"],
                    u0=slice_point["impact_parameter"],
                    fs=fs,
                    fb=fb,
                    snr=snr[infilt],
                    filters=filters,
                    filter_name=filtername,
                )

                # We remove entries if there's no data in a filter
                # which would keep it from being inverted
                # detectable information in dimensions:
                fis_mat_wzeros = deepcopy(fis_mat)
                mask_idx = np.where(np.diag(fis_mat) == 0)
                fis_mat = np.delete(fis_mat, mask_idx[0], 0)
                fis_mat = np.delete(fis_mat, mask_idx[0], 1)
                try:
                    cov = np.linalg.inv(fis_mat)
                    sigmat_e_t_e = cov[0, 0] ** 0.5 / slice_point["crossing_time"]
                    # sigmau0_u0 = cov[1, 1] ** 0.5
                    #             / slice_point["impact_parameter"]
                    # corr_btwn_t_eu0 = cov[0, 1] /
                    #    (slice_point["crossing_time"]
                    #    * slice_point["impact_parameter"])
                except np.linalg.LinAlgError:
                    sigmat_e_t_e = np.inf
                    # sigmau0_u0 = np.inf
                    # corr_btwn_t_eu0 = np.inf

        npts = np.sum(n_pre)
        npts_post = np.sum(n_post)
        if self.metric_calc == "detect":
            if self.detect is True:
                if npts >= self.pts_needed and npts_post >= self.pts_needed:
                    return 1
                else:
                    return 0
            else:
                if npts >= self.pts_needed:
                    return 1
                else:
                    return 0
        elif self.metric_calc == "Npts":
            return npts + npts_post
        elif self.metric_calc == "Fisher":
            # cutoff of sigmat_e_t_e < 0.1 for
            # values determined by the 3sigma of
            # pubished planet candidates
            return sigmat_e_t_e


def generate_microlensing_slicer(
    min_crossing_time=1,
    max_crossing_time=10,
    t_start=1,
    t_end=3652,
    n_events=10000,
    seed=42,
    nside=128,
    filtername="r",
):
    """Generate a UserPointSlicer with a population of microlensing events.
    To be used with MicrolensingMetric

    Parameters
    ----------
    min_crossing_time : `float`
        The minimum crossing time for the events generated (days)
    max_crossing_time : `float`
        The max crossing time for the events generated (days)
    t_start : `float`
        The night to start generating peaks (days)
    t_end : `float`
        The night to end generating peaks (days)
    n_events : `int`
        Number of microlensing events to generate
    seed : `float`
        Random number seed
    nside : `int`
        HEALpix nside, used to pick which stellar density map to load
    filtername : `str`
        The filter to use for the stellar density map

    Returns
    -------
    slicer : `maf.UserPointsSlicer`
        A slicer populated by microlensing events
        (each slice_point is a different event)
    """
    # Seed the random number generator and generate random parameters
    rng = np.random.default_rng(seed)
    crossing_times = rng.uniform(low=min_crossing_time, high=max_crossing_time, size=n_events)
    peak_times = rng.uniform(low=t_start, high=t_end, size=n_events)
    impact_paramters = rng.uniform(low=0, high=1, size=n_events)

    map_dir = os.path.join(get_data_dir(), "maps", "TriMaps")
    data = np.load(os.path.join(map_dir, "TRIstarDensity_%s_nside_%i.npz" % (filtername, nside)))
    star_density = data["starDensity"].copy()
    # magnitude bins
    bins = data["bins"].copy()
    data.close()

    star_mag = 22
    bin_indx = np.where(bins[1:] >= star_mag)[0].min()
    density_used = star_density[:, bin_indx].ravel()
    order = np.argsort(density_used)
    # I think the model might have a few outliers at the extreme,
    # let's truncate it a bit
    density_used[order[-10:]] = density_used[order[-11]]

    # now, let's draw N from that distribution squared
    dist = density_used[order] ** 2
    cumm_dist = np.cumsum(dist)
    cumm_dist = cumm_dist / np.max(cumm_dist)
    uniform_draw = rng.uniform(size=n_events)
    indexes = np.floor(np.interp(uniform_draw, cumm_dist, np.arange(cumm_dist.size)))
    hp_ids = order[indexes.astype(int)]
    gal_l, gal_b = hpid2_ra_dec(nside, hp_ids, nest=True)
    c = SkyCoord(l=gal_l * u.deg, b=gal_b * u.deg, frame="galactic").transform_to("icrs")
    ra, dec = c.ra.deg, c.dec.deg
    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slice_points["peak_time"] = peak_times
    slicer.slice_points["crossing_time"] = crossing_times
    slicer.slice_points["impact_parameter"] = impact_paramters

    return slicer
