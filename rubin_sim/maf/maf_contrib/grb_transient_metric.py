__all__ = ("GRBTransientMetric",)


import numpy as np

import rubin_sim.maf.metrics as metrics

# Gamma-ray burst afterglow metric
# ebellm@caltech.edu


class GRBTransientMetric(metrics.BaseMetric):
    """Evaluate the likelihood of detecting a GRB optical counterpart.

    Detections for an on-axis GRB afterglows decaying as
    F(t) = F(1min)((t-t0)/1min)^-alpha.  No jet break, for now.

    Derived from TransientMetric, but calculated with reduce functions to
    enable-band specific counts.
    Burst parameters taken from 2011PASP..123.1034J.

    Simplifications:
    * no color variation or evolution encoded yet.
    * no jet breaks.
    * not treating off-axis events.

    Parameters
    ----------
    alpha : `float`,
        temporal decay index
        Default = 1.0
    apparent_mag_1min_mean : `float`,
        mean magnitude at 1 minute after burst
        Default = 15.35
    apparent_mag_1min_sigma : `float`,
        std of magnitudes at 1 minute after burst
        Default = 1.59
    trans_duration : `float`, optional
        How long the transient lasts (days). Default 10.
    survey_duration : `float`, optional
        Length of survey (years).
        Default 10.
    survey_start : `float`, optional
        MJD for the survey start date.
        Default None (uses the time of the first observation).
    detect_m5_plus : `float`, optional
        An observation will be used if the light curve magnitude is brighter
        than m5+detect_m5_plus.
        Default 0.
    n_per_filter : `int`, optional
        Number of separate detections of the light curve above the
        detect_m5_plus theshold (in a single filter) for the light curve
        to be counted.
        Default 1.
    n_filters : `int`, optional
        Number of filters that need to be observed n_per_filter times,
        with differences min_delta_mag,
        for an object to be counted as detected.
        Default 1.
    min_delta_mag : `float`, optional
       magnitude difference between detections in the same filter required
       for second+ detection to be counted.
       For example, if min_delta_mag = 0.1 mag and two consecutive observations
       differ only by 0.05 mag, those two detections will only count as one.
       (Better would be a SNR-based discrimination of lightcurve change.)
       Default 0.
    n_phase_check : `int`, optional
        Sets the number of phases that should be checked.
        One can imagine pathological cadences where many objects pass the
        detection criteria, but would not if the observations were offset
        by a phase-shift.
        Default 1.
    """

    def __init__(
        self,
        alpha=1,
        apparent_mag_1min_mean=15.35,
        apparent_mag_1min_sigma=1.59,
        metric_name="GRBTransientMetric",
        mjd_col="expMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        trans_duration=10.0,
        survey_duration=10.0,
        survey_start=None,
        detect_m5_plus=0.0,
        n_per_filter=1,
        n_filters=1,
        min_delta_mag=0.0,
        n_phase_check=1,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        super(GRBTransientMetric, self).__init__(
            col=[self.mjd_col, self.m5_col, self.filter_col],
            units="Fraction Detected",
            metric_name=metric_name,
            **kwargs,
        )
        self.alpha = alpha
        self.apparent_mag_1min_mean = apparent_mag_1min_mean
        self.apparent_mag_1min_sigma = apparent_mag_1min_sigma
        self.trans_duration = trans_duration
        self.survey_duration = survey_duration
        self.survey_start = survey_start
        self.detect_m5_plus = detect_m5_plus
        self.n_per_filter = n_per_filter
        self.n_filters = n_filters
        self.min_delta_mag = min_delta_mag
        self.n_phase_check = n_phase_check
        self.peak_time = 0.0
        self.reduce_order = {
            "Bandu": 0,
            "Bandg": 1,
            "Bandr": 2,
            "Bandi": 3,
            "Bandz": 4,
            "Bandy": 5,
            "Band1FiltAvg": 6,
            "BandanyNfilters": 7,
        }

    def light_curve(self, time, filters):
        """
        given the times and filters of an observation, return the magnitudes.
        """

        lc_mags = np.zeros(time.size, dtype=float)

        decline = np.where(time > self.peak_time)
        apparent_mag_1min = np.random.randn() * self.apparent_mag_1min_sigma + self.apparent_mag_1min_mean
        lc_mags[decline] += apparent_mag_1min + self.alpha * 2.5 * np.log10(
            (time[decline] - self.peak_time) * 24.0 * 60.0
        )

        # for key in self.peaks.keys():
        #    fMatch = np.where(filters == key)
        #    lc_mags[fMatch] += self.peaks[key]

        return lc_mags

    def run(self, data_slice, slice_point=None):
        """
        Calculate the detectability of a transient with the
        specified lightcurve.
        """
        # Total number of transients that could go off back-to-back
        n_trans_max = np.floor(self.survey_duration / (self.trans_duration / 365.25))
        tshifts = np.arange(self.n_phase_check) * self.trans_duration / float(self.n_phase_check)
        n_trans_max = 0
        for tshift in tshifts:
            # Compute the total number of back-to-back transients
            # are possible to detect
            # given the survey duration and the transient duration.
            n_trans_max += np.floor(self.survey_duration / (self.trans_duration / 365.25))
            if tshift != 0:
                n_trans_max -= 1
            if self.survey_start is None:
                survey_start = data_slice[self.mjd_col].min()
            time = (data_slice[self.mjd_col] - survey_start + tshift) % self.trans_duration

            # Which lightcurve does each point belong to
            lc_number = np.floor((data_slice[self.mjd_col] - survey_start) / self.trans_duration)

            lc_mags = self.light_curve(time, data_slice[self.filter_col])

            # How many criteria needs to be passed
            detect_thresh = 0

            # Flag points that are above the SNR limit
            detected = np.zeros(data_slice.size, dtype=int)
            detected[np.where(lc_mags < data_slice[self.m5_col] + self.detect_m5_plus)] += 1

            bandcounter = {
                "u": 0,
                "g": 0,
                "r": 0,
                "i": 0,
                "z": 0,
                "y": 0,
                "any": 0,
            }  # define zeroed out counter

            # make sure things are sorted by time
            ord = np.argsort(data_slice[self.mjd_col])
            data_slice = data_slice[ord]
            detected = detected[ord]
            lc_number = lc_number[ord]
            lc_mags = lc_mags[ord]
            ulc_number = np.unique(lc_number)
            left = np.searchsorted(lc_number, ulc_number)
            right = np.searchsorted(lc_number, ulc_number, side="right")
            detect_thresh += self.n_filters

            # iterate over the lightcurves
            for le, ri in zip(left, right):
                wdet = np.where(detected[le:ri] > 0)
                ufilters = np.unique(data_slice[self.filter_col][le:ri][wdet])
                nfilts_lci = 0
                for filt_name in ufilters:
                    wdetfilt = np.where((data_slice[self.filter_col][le:ri] == filt_name) & detected[le:ri])

                    lc_points = lc_mags[le:ri][wdetfilt]
                    dlc = np.abs(np.diff(lc_points))

                    # number of detections in band, requiring that for
                    # nPerFilter > 1 that points have more than minDeltaMag
                    # change
                    nbanddet = np.sum(dlc > self.min_delta_mag) + 1
                    if nbanddet >= self.n_per_filter:
                        bandcounter[filt_name] += 1
                        nfilts_lci += 1
                if nfilts_lci >= self.n_filters:
                    bandcounter["any"] += 1

        bandfraction = {}
        for band in bandcounter.keys():
            bandfraction[band] = float(bandcounter[band]) / n_trans_max

        return bandfraction

    def reduce_band1_filt_avg(self, bandfraction):
        "Average fraction detected in single filter"
        return np.mean(list(bandfraction.values()))

    def reduce_bandany_nfilters(self, bandfraction):
        "Fraction of events detected in Nfilters or more"
        return bandfraction["any"]

    def reduce_bandu(self, bandfraction):
        return bandfraction["u"]

    def reduce_bandg(self, bandfraction):
        return bandfraction["g"]

    def reduce_bandr(self, bandfraction):
        return bandfraction["r"]

    def reduce_bandi(self, bandfraction):
        return bandfraction["i"]

    def reduce_bandz(self, bandfraction):
        return bandfraction["z"]

    def reduce_bandy(self, bandfraction):
        return bandfraction["y"]
