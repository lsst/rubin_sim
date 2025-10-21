__all__ = ("XrbLc", "XRBPopMetric", "generate_xrb_pop_slicer")

import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import SURVEY_START_MJD
from scipy.stats import loguniform

from rubin_sim.maf.utils import m52snr
from rubin_sim.phot_utils import DustValues

from ..metrics import BaseMetric
from ..slicers import UserPointsSlicer


class XrbLc:
    """Synthesize XRB outburst lightcurves."""

    def __init__(self, seed=42):
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

        self.rng = np.random.default_rng(seed)

    def lmxb_abs_mags(self, size=1):
        """Return LMXB absolute magnitudes per LSST filter.

        Absolute magnitude relation is taken from Casares 2018
        (2018MNRAS.473.5195C)
        Colors are taken from M. Johnson+ 2019
        (2019MNRAS.484...19J)

        Parameters
        ----------
        size : `int`
            Number of samples to generate.

        Returns
        -------
        abs_mags : `list` [`dict`]
            Absolute magnitudes for each LSST filter.
        Porbs: `array` [`float`]
            Randomized orbital periods in days
        """

        # Derive random orbital periods from the sample in Casares 18 Table 4
        # Since there are significant outliers from a single Gaussian sample,
        # take random choices with replacement, then perturb them fractionally
        catalog__porb = np.array(
            [
                33.85,
                6.4713,
                2.5445,
                1.7557,
                1.5420,
                0.5213,
                0.4326,
                0.3441,
                0.3230,
                0.3205,
                0.2852,
                0.2740,
                0.2122,
                0.1699,
                0.1352,
                0.117,
                0.1006,
            ]
        )

        sample__porbs = self.rng.choice(catalog__porb, size=size)
        sample__porbs *= self.rng.uniform(low=0.5, high=1.5, size=size)

        # lmxb_abs_mag_r = 4.6 # johnson+18
        # Casares 18
        lmxb_abs_mags_r = 4.64 - 3.69 * np.log10(sample__porbs)

        return [
            {
                "u": lmxb_abs_mag_r + 4.14,
                "g": lmxb_abs_mag_r + 3.24,
                "r": lmxb_abs_mag_r,
                "i": lmxb_abs_mag_r + 0.33,
                "z": lmxb_abs_mag_r + 1.05,
                "y": lmxb_abs_mag_r + 2.36,
            }
            for lmxb_abs_mag_r in lmxb_abs_mags_r
        ], sample__porbs

    def outburst_params(self, size=1):
        """Return a parameters at random characterizing the outburst.

        Uses distributions from Chen, Shrader, & Livio 1997 (ApJ 491, 312).

        Returns
        -------
        params : `list` [`dict`]
            Rise, decay, and amplitude parameters.
        """

        # rise timescale (Fig. 11): 50% 1-2 day lognormal,
        #                           50% loguniform 0.5-50 days

        flip = self.rng.uniform(size=size)
        w = flip > 0.5
        tau_rise = np.zeros(size) * np.nan
        tau_rise[w] = loguniform.rvs(0.5, 50, size=np.sum(w))
        tau_rise[~w] = self.rng.lognormal(0, 0.2, size=np.sum(~w)) + 0.3

        # amplitude = rng.lognormal(9,1,size=1)[0]
        # delta_mag = -2.5 * np.log10(amplitude)
        # The Chen+ 10**4 X-ray flux amplitudes imply -10 optical delta mags,
        # which are generally larger than observed.
        # We'll use a fixed -6 delta mag for simplicity
        delta_mag = -6
        amplitude = 10 ** (-0.4 * delta_mag) * np.ones(size)

        tau_decay = self.rng.lognormal(2.9, 0.65, size=size)

        duration = (tau_rise + tau_decay) + np.log(amplitude)

        abs_mags, porbs = self.lmxb_abs_mags(size=size)

        return [
            {
                "tau_rise": tr,
                "tau_decay": td,
                "amplitude": amp,
                "outburst_duration": dur,
                "abs_mag": abs_mag,
                "orbital_period": Porb,
            }
            for (tr, td, amp, dur, abs_mag, Porb) in zip(
                tau_rise, tau_decay, amplitude, duration, abs_mags, porbs
            )
        ]

    def fred(self, t, amplitude, tau_rise, tau_decay):
        """Fast-rise, exponential decay function.

        Amplitude is defined at the peak time = sqrt(tau_rise*tau_decay).

        See e.g., Tarnopolski 2021 for discussion.

        Parameters
        ----------
        t : `array` [`float`]
            The times relative to the start of the outburst
        amplitude : `float`
            Peak amplitude
        tau_rise : `float`
            E-folding time for the rise
        tau_decay : `float`
            E-folding time for the decay
        """
        return amplitude * np.exp(2 * np.sqrt(tau_rise / tau_decay)) * np.exp(-tau_rise / t - t / tau_decay)

    def lightcurve(self, t, filtername, params):
        """Generate an XRB outburst lightcurve for given times
        and a single filter.

        Uses a simple fast-rise, exponential decay with parameters taken from
        Chen, Shrader, & Livio 1997 (ApJ 491, 312).

        For now we ignore the late time linear decay (Tetarenko+2018a,b, and
        references therein.)

        Parameters
        ----------
        t : `array` [`float`]
            The times relative to the start of the outburst
        filtername : `str`
            The filter. one of ugrizy
        params : `dict`
            parameters for the FRED lightcurve.

        Returns
        -------
        lc : `array`
            Magnitudes of the outburst at the specified times in
            the given filter
        """

        # fill lightcurve with nondetections
        lc = np.ones(len(t)) * 99

        # FRED
        woutburst = (t >= 0) & (t <= params["outburst_duration"])
        if np.sum(woutburst):
            lc[woutburst] = params["abs_mag"][filtername] + -2.5 * np.log10(
                self.fred(
                    t[woutburst],
                    params["amplitude"],
                    params["tau_rise"],
                    params["tau_decay"],
                )
            )

        return lc

    def detectable_duration(self, params, ebv, distance):
        """Determine time range an outburst is detectable with
        perfect sampling.

        Does not consider visibility constraints.

        Parameters
        ----------
        params : `dict`
            lightcurve parameters for XrbLc
        ebv : `float`
            E(B-V)
        distance : `float`
            distance in kpc

        Returns
        ----------
        visible_start_time : `float`
            first time relative to outburst start that the outburst
            could be detected
        visible_end_time : `float`
            last time relative to outburst start that the outburst
            could be detected
        """

        nmodelt = 10000
        t = np.linspace(0, params["outburst_duration"], nmodelt)

        lsst_single_epoch_depth = {
            "u": 23.9,
            "g": 25.0,
            "r": 24.7,
            "i": 24.0,
            "z": 23.3,
            "y": 22.1,
        }

        visible_start_time = np.inf
        visible_end_time = -np.inf

        for filtername in ["u", "g", "r", "i", "z", "y"]:
            mags = self.lightcurve(t, filtername, params)

            # Apply dust extinction on the light curve
            a_x = self.ax1[filtername] * ebv
            mags += a_x
            distmod = 5 * np.log10(distance * 1.0e3) - 5.0
            mags += distmod

            wdetectable = np.where(mags <= lsst_single_epoch_depth[filtername])[0]
            if len(wdetectable) == 0:
                continue

            if t[wdetectable[0]] < visible_start_time:
                visible_start_time = t[wdetectable[0]]

            if t[wdetectable[-1]] >= visible_end_time:
                visible_end_time = t[wdetectable[-1]]

        if np.isinf(visible_start_time):
            return np.nan, np.nan

        return visible_start_time, visible_end_time


class XRBPopMetric(BaseMetric):
    """Evaluate whether a given XRB would be detectable.

    Includes a variety of detection criteria options, including if the
    XRB is possible to detect, if it is detected at least pts_needed times,
    or if it is detected pts_early times within t_early days of the start of
    the outburst.

    Parameters
    ----------
    pts_needed : `int`, opt
        Minimum number of detections, for simple `detected` option.
    mjd0 : `float`, opt
        Start of survey.
    output_lc : `bool`, opt
        If True, output lightcurve points.
        If False, just return metric values.
    """

    def __init__(
        self,
        metric_name="XRBPopMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        night_col="night",
        pts_needed=2,
        pts_early=2,
        t_early=7,
        mjd0=SURVEY_START_MJD,
        output_lc=False,
        badval=-666,
        **kwargs,
    ):
        # maps = ["DustMap"]
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.pts_needed = pts_needed
        self.pts_early = pts_early
        self.t_early = t_early
        # `bool` variable, if True the light curve will be exported
        self.output_lc = output_lc

        self.lightcurves = XrbLc()
        self.mjd0 = mjd0

        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

        cols = [self.mjd_col, self.m5_col, self.filter_col, self.night_col]
        super(XRBPopMetric, self).__init__(
            col=cols,
            units="Detected, 0 or 1",
            metric_name=metric_name,
            # maps=maps,
            badval=badval,
            **kwargs,
        )
        self.comment = "Number or characterization of XRBs."

    def _ever_detect(self, where_detected):
        """Simple detection criteria: detect at least a certain number
        of times.
        """
        # Detected data points
        return np.size(where_detected) >= self.pts_needed

    def _number_of_detections(self, where_detected):
        """Count total number of detections."""
        return len(where_detected)

    def _early_detect(self, where_detected, time, early_window_days=7.0, n_early_detections=2):
        """Detection near the start of the outburst.

        Parameters
        ----------
        where_detected : `array`
            indexes corresponding to 5 sigma detections
        mags : `array`
            magnitudes obtained interpolating models on the data_slice
        time : `array`
            relative times
        early_window_days : `float`
            time since start of outburst
        n_early_detections : `int`
            number of required early detections
        """

        return np.sum(time[where_detected] <= early_window_days) >= n_early_detections

    def _mean_time_between_detections(self, t):
        """Calculate the mean time between detections over the
        visible interval.

        Parameters
        ----------
        t : `array`
            Times of detections, bracketed by the start and
            end visibility times

        Return
        ----------
        med_dt : `float`
             separation between observations.
        """

        return np.mean(np.sort(np.diff(t)))

    def _possible_to_detect(self, visible_duration):
        """Return True if the outburst is ever bright enough
        for LSST to detect.

        Parameters
        ----------
        visible_duration : `float`
            Length of time the outburst is above LSST's fiducial limiting mag.
            May be nan.

        Return
        ----------
        detectable : `bool`
             Return True if outburst is ever detectable by LSST.
        """

        return ~np.isnan(visible_duration)

    def run(self, data_slice, slice_point=None):
        result = {}
        t = data_slice[self.mjd_col] - self.mjd0 - slice_point["start_time"]
        mags = np.zeros(t.size, dtype=float)

        for filtername in np.unique(data_slice[self.filter_col]):
            infilt = np.where(data_slice[self.filter_col] == filtername)
            mags[infilt] = self.lightcurves.lightcurve(t[infilt], filtername, slice_point["outburst_params"])
            # Apply dust extinction on the light curve
            a_x = self.ax1[filtername] * slice_point["ebv"]
            mags[infilt] += a_x

            distmod = 5 * np.log10(slice_point["distance"] * 1.0e3) - 5.0
            mags[infilt] += distmod

        # Find the detected points
        where_detected = np.where((mags < data_slice[self.m5_col]))[0]
        # Magnitude uncertainties with Gaussian approximation
        snr = m52snr(mags, data_slice[self.m5_col])
        mags_unc = 2.5 * np.log10(1.0 + 1.0 / snr)

        result["possible_to_detect"] = self._possible_to_detect(slice_point["visible_duration"])
        result["ever_detect"] = self._ever_detect(where_detected)
        result["early_detect"] = self._early_detect(where_detected, t, self.t_early, self.pts_early)
        result["number_of_detections"] = self._number_of_detections(where_detected)

        if result["number_of_detections"] > 1:
            result["mean_time_between_detections"] = self._mean_time_between_detections(
                [
                    slice_point["visible_start_time"],
                    *t[where_detected].tolist(),
                    slice_point["visible_end_time"],
                ]
            )
        else:
            result["mean_time_between_detections"] = -999

        # Export the light curve
        if self.output_lc is True:
            wdet = mags < data_slice[self.m5_col]
            mags[~wdet] = 99.0
            result["lc"] = [
                data_slice[self.mjd_col],
                mags,
                mags_unc,
                data_slice[self.m5_col],
                data_slice[self.filter_col],
            ]
            result["lc_colnames"] = ("t", "mag", "mag_unc", "maglim", "filter")

        return result

    def reduce_possible_to_detect(self, metric):
        return metric["possible_to_detect"]

    def reduce_ever_detect(self, metric):
        return metric["ever_detect"]

    def reduce_early_detect(self, metric):
        return metric["early_detect"]

    def reduce_number_of_detections(self, metric):
        return metric["number_of_detections"]

    def reduce_mean_time_between_detections(self, metric):
        tt = metric["mean_time_between_detections"]
        if tt < 0:
            return self.badval
        else:
            return tt


def generate_xrb_pop_slicer(t_start=1, t_end=3652, n_events=10000, seed=42):
    """Generate a population of XRB events, and put the info about them
    into a UserPointSlicer object.

    Parameters
    ----------
    t_start : `float`
        The night to start an XRB outburst on (days; default 1)
    t_end : `float`
        The final night of XRBs events (days; default 3652)
    n_events : `int`
        The number of XRB outbursts to generate (default 10000)
    seed : `float`
        The seed passed to np.random (default 42)
    """

    rng = np.random.default_rng(seed)

    datadir = get_data_dir()
    xrb_sample = np.genfromtxt(datadir + "/maf/xrb/sample_xrb_positions.csv.gz", delimiter=",")

    nsamples, nfields = xrb_sample.shape

    xrb_lc_gen = XrbLc()

    # select a random subsample
    event_idxs = rng.choice(np.arange(nsamples), size=n_events, replace=True)
    ra = xrb_sample[event_idxs, 0]
    dec = xrb_sample[event_idxs, 1]
    distance_kpc = xrb_sample[event_idxs, 2]
    ebv = xrb_sample[event_idxs, 3]

    start_times = rng.uniform(low=t_start, high=t_end, size=n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slice_points["start_time"] = start_times
    slicer.slice_points["distance"] = distance_kpc
    # use our own 3-d dust map extinctions
    slicer.slice_points["ebv"] = ebv
    # generate random parameters for this event
    slicer.slice_points["outburst_params"] = xrb_lc_gen.outburst_params(size=n_events)

    # determine detectable durations
    visible_starts = []
    visible_ends = []
    visible_durations = []
    for idx, param in enumerate(slicer.slice_points["outburst_params"]):
        start, end = xrb_lc_gen.detectable_duration(param, ebv[idx], distance_kpc[idx])
        visible_starts.append(start)
        visible_ends.append(end)
        visible_durations.append(end - start)

    slicer.slice_points["visible_start_time"] = visible_starts
    slicer.slice_points["visible_end_time"] = visible_ends
    slicer.slice_points["visible_duration"] = visible_durations

    return slicer
