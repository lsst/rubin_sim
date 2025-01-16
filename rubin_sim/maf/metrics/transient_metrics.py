__all__ = ("TransientMetric",)


import numpy as np

from .base_metric import BaseMetric


class TransientMetric(BaseMetric):
    """
    Calculate what fraction of the transients would be detected. Best paired
    with a spatial slicer.
    We are assuming simple light curves with no color evolution.

    Parameters
    ----------
    trans_duration : float, optional
        How long the transient lasts (days). Default 10.
    peak_time : float, optional
        How long it takes to reach the peak magnitude (days). Default 5.
    rise_slope : float, optional
        Slope of the light curve before peak time (mags/day).
        This should be negative since mags are backwards (magnitudes decrease
        towards brighter fluxes).
        Default 0.
    decline_slope : float, optional
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
    survey_duration : float, optional
        Length of survey (years).
        Default 10.
    survey_start : float, optional
        MJD for the survey start date.
        Default None (uses the time of the first observation).
    detect_m5_plus : float, optional
        An observation will be used if the light curve magnitude is brighter
        than m5+detect_m5_plus.
        Default 0.
    n_pre_peak : int, optional
        Number of observations (in any filter(s)) to demand before peak_time,
        before saying a transient has been detected.
        Default 0.
    n_per_lc : int, optional
        Number of sections of the light curve that must be sampled above the
        detect_m5_plus theshold
        (in a single filter) for the light curve to be counted.
        For example, setting n_per_lc = 2 means a light curve  is only
        considered detected if there is at least 1 observation in the first
        half of the LC, and at least one in the second half of the LC.
        n_per_lc = 4 means each quarter of the light curve must be detected to
        count.
        Default 1.
    n_filters : int, optional
        Number of filters that need to be observed for an object to be counted
        as detected.
        Default 1.
    n_phase_check : int, optional
        Sets the number of phases that should be checked.
        One can imagine pathological cadences where many objects pass the
        detection criteria,
        but would not if the observations were offset by a phase-shift.
        Default 1.
    count_method : {'full' 'partialLC'}, defaults to 'full'
        Sets the method of counting max number of transients. if 'full', the
        only full light curves that fit the survey duration are counted. If
        'partialLC', then the max number of possible transients is taken to be
        the integer floor
    """

    def __init__(
        self,
        metric_name="TransientDetectMetric",
        mjd_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        trans_duration=10.0,
        peak_time=5.0,
        rise_slope=0.0,
        decline_slope=0.0,
        survey_duration=10.0,
        survey_start=None,
        detect_m5_plus=0.0,
        u_peak=20,
        g_peak=20,
        r_peak=20,
        i_peak=20,
        z_peak=20,
        y_peak=20,
        n_pre_peak=0,
        n_per_lc=1,
        n_filters=1,
        n_phase_check=1,
        count_method="full",
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        super(TransientMetric, self).__init__(
            col=[self.mjd_col, self.m5_col, self.filter_col],
            units="Fraction Detected",
            metric_name=metric_name,
            **kwargs,
        )
        self.peaks = {
            "u": u_peak,
            "g": g_peak,
            "r": r_peak,
            "i": i_peak,
            "z": z_peak,
            "y": y_peak,
        }
        self.trans_duration = trans_duration
        self.peak_time = peak_time
        self.rise_slope = rise_slope
        self.decline_slope = decline_slope
        self.survey_duration = survey_duration
        self.survey_start = survey_start
        self.detect_m5_plus = detect_m5_plus
        self.n_pre_peak = n_pre_peak
        self.n_per_lc = n_per_lc
        self.n_filters = n_filters
        self.n_phase_check = n_phase_check
        self.count_method = count_method

    def light_curve(self, time, filters):
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
        lc_mags = np.zeros(time.size, dtype=float)
        rise = np.where(time <= self.peak_time)
        lc_mags[rise] += self.rise_slope * time[rise] - self.rise_slope * self.peak_time
        decline = np.where(time > self.peak_time)
        lc_mags[decline] += self.decline_slope * (time[decline] - self.peak_time)
        for key in self.peaks:
            f_match = np.where(filters == key)
            lc_mags[f_match] += self.peaks[key]
        return lc_mags

    def run(self, data_slice, slice_point=None):
        """
        Calculate the detectability of a transient with the specified
        lightcurve.

        Parameters
        ----------
        data_slice : numpy.array
            Numpy structured array containing the data related to the visits
            provided by the slicer.
        slice_point : dict, optional
            Dictionary containing information about the slice_point currently
            active in the slicer.

        Returns
        -------
        float
            The total number of transients that could be detected.
        """
        # Total number of transients that could go off back-to-back
        if self.count_method == "partialLC":
            _n_trans_max = np.ceil(self.survey_duration / (self.trans_duration / 365.25))
        else:
            _n_trans_max = np.floor(self.survey_duration / (self.trans_duration / 365.25))
        tshifts = np.arange(self.n_phase_check) * self.trans_duration / float(self.n_phase_check)
        n_detected = 0
        n_trans_max = 0
        for tshift in tshifts:
            # Compute the total number of back-to-back transients are possible
            # to detect given the survey duration and the transient duration.
            n_trans_max += _n_trans_max
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
            detect_thresh += 1

            # If we demand points on the rise
            if self.n_pre_peak > 0:
                detect_thresh += 1
                ord = np.argsort(data_slice[self.mjd_col])
                data_slice = data_slice[ord]
                detected = detected[ord]
                lc_number = lc_number[ord]
                time = time[ord]
                ulc_number = np.unique(lc_number)
                left = np.searchsorted(lc_number, ulc_number)
                right = np.searchsorted(lc_number, ulc_number, side="right")
                # Note here I'm using np.searchsorted to basically do a
                # 'group by' might be clearer to use
                # scipy.ndimage.measurements.find_objects or pandas, but this
                # numpy function is known for being efficient.
                for le, ri in zip(left, right):
                    # Number of points where there are a detection
                    good = np.where(time[le:ri] < self.peak_time)
                    nd = np.sum(detected[le:ri][good])
                    if nd >= self.n_pre_peak:
                        detected[le:ri] += 1

            # Check if we need multiple points per light curve
            # or multiple filters
            if (self.n_per_lc > 1) | (self.n_filters > 1):
                # make sure things are sorted by time
                ord = np.argsort(data_slice[self.mjd_col])
                data_slice = data_slice[ord]
                detected = detected[ord]
                lc_number = lc_number[ord]
                time = time[ord]
                ulc_number = np.unique(lc_number)
                left = np.searchsorted(lc_number, ulc_number)
                right = np.searchsorted(lc_number, ulc_number, side="right")
                detect_thresh += self.n_filters

                for le, ri in zip(left, right):
                    points = np.where(detected[le:ri] > 0)
                    ufilters = np.unique(data_slice[self.filter_col][le:ri][points])
                    phase_sections = np.floor(time[le:ri][points] / self.trans_duration * self.n_per_lc)
                    for filt_name in ufilters:
                        good = np.where(data_slice[self.filter_col][le:ri][points] == filt_name)
                        if np.size(np.unique(phase_sections[good])) >= self.n_per_lc:
                            detected[le:ri] += 1

            # Find the unique number of light curves that passed the required
            # number of conditions
            n_detected += np.size(np.unique(lc_number[np.where(detected >= detect_thresh)]))

        # Rather than keeping a single "detected" variable, maybe make a mask
        # for each criteria, then reduce functions like: reduce_singleDetect,
        # reduce_NDetect, reduce_PerLC, reduce_perFilter.
        # The way I'm running now it would speed things up.

        return float(n_detected) / n_trans_max
