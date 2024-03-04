# Transient metric with input ascii SED.
# Builds upon and uses a significant amount of common code from the
# transientAsciiMetric by: fbb@nyu.edu, svalenti@lcogt.net
# Apart from the inherent differences between metrics, the code has changed in
# that many of the variable names have been changed for clarity, at the expense
# of brevity.
#
# Contact for this code:
# christian.setzer@fysik.su.se

__all__ = ("TransientAsciiSEDMetric",)

import os
from copy import deepcopy

import numpy as np

try:
    from sncosmo import Model, TimeSeriesSource, read_griddata_ascii
except ImportError:
    pass
from astropy.cosmology import Planck15 as cosmo  # noqa N813

from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils import m52snr


class TransientAsciiSEDMetric(BaseMetric):
    """
    Based on the transientMetric and transientAsciiMetric, uses an ascii
    input file and provides option to write out the light curves. Calculates
    what fraction of the transients would be detected. A spatial slicer is the
    preferred choice. The input SED should have the 3-column format
    (phase, wave, flux), and be scaled to a distance of 10pc from the observer.

    Parameters
    -----------
    ascii_file : `str`
        The ascii file containing the inputs for the SED. The file must
        contain three columns - ['phase', 'wave', 'flux'] -
        of phase/epoch (in days), wavelength (Angstroms), and
        flux (ergs/s/Angstrom).
    metric_name : `str`, optional
        Name of the metric, can be overwritten by user or child metric.
    survey_duration : `float`, optional
        Length of survey (years).
        Default 10 or maximum of timespan of observations.
    survey_start : `float`, optional
        MJD for the survey start date.
        Default None (uses the time of the first observation at each pointing).
    detect_snr : `dict`, optional
        An observation will be counted toward the discovery criteria if the
        light curve SNR is higher than detect_snr (specified per bandpass).
        Values must be provided for each filter which should be considered
        in the lightcurve.
        Default is {'u': 5, 'g': 5, 'r': 5, 'i': 5, 'z': 5, 'y': 5}
    z : `float`, optional
        Cosmological redshift at which to consider observations of the
        tranisent SED.
    num_pre_time : `int`, optional
        Number of observations (in any filter(s)) to demand before pre_time,
        before saying a transient has been detected.
        Default 0.
    pre_time : `float`, optional
        The time by which num_pre_time detections are required (in days).
        Default 5.0.
    num_filters : `int`, optional
        Number of filters that need to be observed for an object to be
        counted as detected. Default 1. (if num_per_lightcurve is 0, then
        this will be reset to 0).
    filter_time : `float`, optional
        The time within which observations in at least num_filters are
        required (in days). Default None (no time constraint).
    num_per_lightcurve : `int`, optional
        Number of sections of the light curve that must be sampled above
        the detect_snr theshold for the light curve to be counted.
        For example, num_per_lightcurve = 2 means a light curve is only
        considered detected if there is at least 1 observation in the first
        half of the LC, and at least one in the second half of the LC.
        num_per_lightcurve = 4 means each quarter of the light curve must
        be detected to count. Default 1.
    num_phases_to_run : `int`, optional
        Sets the number of phases that should be checked.
        One can imagine pathological cadences where many objects pass the
        detection criteria, but would not if the observations were offset
        by a phase-shift. Default 1.
    output_data : `bool`, optional
        If True, metric returns full lightcurve at each point. Note that
        this will potentially create a very large metric output data file.
        If False, metric returns the number of transients detected.
    """

    def __init__(
        self,
        ascii_file,
        metric_name="TransientAsciiSEDMetric",
        mjd_col="expMJD",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        survey_duration=10.0,
        survey_start=None,
        detect_snr={"u": 5, "g": 5, "r": 5, "i": 5, "z": 5, "y": 5},
        z=0.075,
        num_pre_time=2,
        pre_time=25.0,
        num_filters=2,
        filter_time=None,
        num_per_lightcurve=1,
        num_phases_to_run=5,
        output_data=False,
        **kwargs,
    ):
        # Set all initial attributes of the metric.
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.output_data = output_data
        self.z = z

        # If it is specified that the lightcurves should be returned rather
        # than the summary statistics, this changes the output of the metric,
        # and thus needs an alternate definition of the output type and units
        # when calling the init of the base class.
        if self.output_data:
            super_dict = {"units": "", "metricDtype": "object"}
        else:
            super_dict = {"units": "Number Detected"}

        super(TransientAsciiSEDMetric, self).__init__(
            col=[self.mjd_col, self.m5_col, self.filter_col],
            metric_name=metric_name,
            **super_dict,
            **kwargs,
        )
        # Continue setting the initial attributes.
        self.survey_duration = survey_duration
        self.survey_start = survey_start
        self.detect_snr = detect_snr
        self.num_pre_time = num_pre_time
        self.pre_time = pre_time
        self.num_filters = num_filters
        self.filter_time = filter_time
        self.num_per_lightcurve = num_per_lightcurve
        if self.num_per_lightcurve == 0:
            self.num_filters = 0
        self.num_phases_to_run = num_phases_to_run
        # Read ascii lightcurve template here.
        # It doesn't change per slice_point.
        self.read_sed(ascii_file)

    def read_sed(self, ascii_file):
        """
        Reads in an ascii file detailing the time evolution of an SED. Must be
        in the following format, 3 columns: phase, wavelength, flux.

        This will also set the source model and redshift the model according to
        initialization parameters.

        Parameters
        -----------
        ascii_file : `str`
            string containing the path to the ascii file containing the
            SED evolution.

        """
        # Sanity to check to make sure the provided string indeed points to a
        # file.
        if not os.path.isfile(ascii_file):
            raise IOError("Could not find SED ascii file %s" % (ascii_file))
        # Read in the SED text file with sncosmo for use with it's Model API.
        self.sed_phase, self.sed_wave, self.sed_flux = read_griddata_ascii(ascii_file)
        # Make the source model and set the transient Model to be used for
        # light curve generation and observation.
        self.make_model()
        # Given the redshifted SED set the transient duration.
        self.transient_duration = self.redshifted_model.maxtime() - self.redshifted_model.mintime()

    def make_model(self):
        """
        Wrapper function to take the phase, wave, and flux information from the
        provided ascii file and create an sncosmo Model object,
        and consistently redshift that model given initialization Parameters.
        This sets the transient model in rest frame, and transient model in
        observer frame,
        i.e., it is cosmologically redshifted.
        """
        # Set the source model with sncosmo API.
        source = TimeSeriesSource(self.sed_phase, self.sed_wave, self.sed_flux)
        # Create transient model from sncosmo API
        # Use deepcopy to make ensure full class is saved as attribute of new
        # class.
        self.transient_model = deepcopy(Model(source=source))
        # With the Model set, apply the cosmological redshift specified at
        # initialization.
        self.set_redshift()

    def set_redshift(self):
        """
        Function which takes the input desired observation redshift and
        cosmologically redshifts the source SED. This sets the redshifted_model
        which is what is needed for making observations.
        """

        z = self.z
        # Deepcopy the model to local variable to get proper behavior of model
        # object methods.
        redshifted_model = deepcopy(self.transient_model)
        # Compute the luminosity distance using Planck15 cosmological
        # parameters, by which to decrease the amplitude of the SED
        lumdist = cosmo.luminosity_distance(z).value * 1e6  # in pc
        # SED is assumed to be at 10pc so scale accordingly.
        amp = pow(np.divide(10.0, lumdist), 2)
        # Set the redshift of the SED, this stretches the wavelength
        # distribution.
        redshifted_model.set(z=z)
        # Separately set the amplitude, this rescales the flux at each
        # wavelength.
        redshifted_model.set(amplitude=amp)
        self.redshifted_model = redshifted_model

    def make_lightcurve(self, time, filters):
        """
        Compute light curve magnitudes from the source Model for the specified
        light curve phases, and filters.

        Parameters
        ----------
        time : `np.ndarray`, (N,)
            The times of the observations.
        filters : `list` [`str`]
            The filters of the observations. ['u','g','r',...] format.

        Returns
        -------
        light_curve_mags : `np.ndarray`, (N,)
             The magnitudes of the object at the times and in the filters of
             the observations.
        """
        # Use the redshifted source model.
        redshifted_model = deepcopy(self.redshifted_model)
        # initialize the light curve magnitudes array.
        light_curve_mags = np.zeros(time.size, dtype=float)

        # Iterate over the filters that are observed.
        for flt in list(set(filters)):
            # Find the phases of obesrvations in the current filter.
            flt_times = list(time[np.where(filters == flt)[0]])
            # Initialize the band magnitude list.
            filter_mag = []
            # Get all band magnitudes. Currently must be in loop, method fails
            # in the scipy interpolation if attempting to vectorize according
            # to sncosmo documentation.
            for obs_time in flt_times:
                filter_mag.append(redshifted_model.bandmag("lsst" + flt, "ab", obs_time))
            # Set light_curve_mags for array indices corresponding to
            # observations of the current filter.
            light_curve_mags[np.where(filters == flt)[0]] = np.array(filter_mag)
            self.light_curve_mags = light_curve_mags

    def evaluate_all_detection_criteria(self, data_slice):
        """
        Wrapper function to setup loop for each transient light curve and
        evaluate all detection criteria.

        Parameters
        -----------
        data_slice : `np.array`
            Numpy structured array containing the data related to the visits
            provided by the slicer.

        Returns
        --------
        transient_detected : `np.array`, (`bool`,)
            Array containing `bool` tracking variable whether transient is
            detected by passing all criteria.
        num_detected : `int`
            Scalar value of the number of transients that were detected in
            total between all phase shifts considered.

        """
        # Track whether each individual light curve was detected.
        # Start with the assumption that it is True, and if it fails
        # criteria then becomes False.
        self.transient_detected = np.ones(len(np.unique(self.transient_id)), dtype=bool)

        # Loop through each lightcurve and check if it meets requirements.
        for i, (start_ind, end_ind) in enumerate(zip(self.transient_start_index, self.transient_end_index)):
            t_id = i
            # If there were no observations at all for this lightcurve:
            if start_ind == end_ind:
                self.transient_detected[t_id] = False
                continue

            self.observation_epoch_above_thresh = self.observation_epoch[start_ind:end_ind][
                np.where(self.obs_above_snr_threshold[start_ind:end_ind])
            ]

            self.evaluate_pre_time_detection_criteria(t_id)
            # Check if previous condition passed.
            # If not, move to next transient.
            if not self.transient_detected[t_id]:
                continue

            self.evaluate_phase_section_detection_criteria(t_id)
            # Check if previous condition passed.
            # If not, move to next transient.
            if not self.transient_detected[t_id]:
                continue

            self.evaluate_number_filters_detection_criteria(data_slice, start_ind, end_ind, t_id)
            # Check if previous condition passed.
            # If not, move to next transient.
            if not self.transient_detected[t_id]:
                continue

            self.evaluate_filter_in_time_detection_criteria(t_id)
            # Check if previous condition passed.
            # If not, move to next transient.
            # Note: this last if block is technically unnecessary but if
            # further criteria are added then the if block should be copied
            # afterwards.
            if not self.transient_detected[t_id]:
                continue
            # Finished with current set of conditions

        # Find the unique number of light curves that passed the required
        # number of conditions
        self.num_detected += len(np.where(self.transient_detected == True)[0])

    def evaluate_pre_time_detection_criteria(self, t_id):
        """
        Function evaluate if the specfied number of observations of the current
        transient take place before the user speficied light curve phase by
        which these must be achieved.

        Parameters
        -----------
        t_id : `int`
            The transient id of the currently evaluated transient.
        """
        # If we did not get enough detections before pre_time, set
        # transient_detected to False.
        indices_pre_time = np.where(self.observation_epoch_above_thresh < self.pre_time)[0]
        if len(indices_pre_time) < self.num_pre_time:
            self.transient_detected[t_id] = False

    def evaluate_phase_section_detection_criteria(self, t_id):
        """
        Function to evaluate if the specified number of equal length sections
        of the current transient are detected given the user criteria.

        Parameters
        -----------
        t_id : `int`
            The transient id of the currently evaluated transient.
        """
        # If we did not get detections over enough sections of the
        # lightcurve, set tranisent_detected to False.
        detected_light_curve_sections = np.unique(
            np.floor(self.observation_epoch_above_thresh / self.transient_duration * self.num_per_lightcurve)
        )
        if len(detected_light_curve_sections) < self.num_per_lightcurve:
            self.transient_detected[t_id] = False

    def evaluate_number_filters_detection_criteria(self, data_slice, start_ind, end_ind, t_id):
        """
        Function to evaluate if the current transient passes the required
        number of detections in different filters.

        Parameters
        -----------
        data_slice : `np.array`, (N,)
            Numpy structured array containing the data related to the visits
            provided by the slicer.
        start_ind : `int`
            Starting index for observations of the specific transient being
            evaluated.
        end_ind : `int`
            Ending index for observations of the specific transient being
            evaluated.
        t_id : `int`
            The transient id of the currently evaluated transient.
        """
        # If we did not get detections in enough filters, set transient
        # detected to False.
        self.detected_filters = data_slice[start_ind:end_ind][
            np.where(self.obs_above_snr_threshold[start_ind:end_ind])
        ][self.filter_col]
        if len(np.unique(self.detected_filters)) < self.num_filters:
            self.transient_detected[t_id] = False

    def evaluate_filter_in_time_detection_criteria(self, t_id):
        """
        Function to evaluate whether the required detections in different
        filters take place within the specified time span for them to count
        towards detection of the transient.

        Parameters
        -----------
        t_id : `int`
            The transient id of the currently evaluted transient.
        """
        # If we did not get detections in enough filters within required
        # time, set transient_detected to False.
        if (self.filter_time is not None) and (self.num_filters > 1):
            final_filter_detection_ind = np.searchsorted(
                self.observation_epoch_above_thresh,
                self.observation_epoch_above_thresh + self.filter_time,
                "right",
            )
            final_filter_detection_ind = np.where(
                final_filter_detection_ind < len(self.observation_epoch_above_thresh) - 1,
                final_filter_detection_ind,
                len(self.observation_epoch_above_thresh) - 1,
            )
            is_detected = False
            for i, filt_end_ind in enumerate(final_filter_detection_ind):
                if len(np.unique(self.detected_filters[i:filt_end_ind])) >= self.num_filters:
                    is_detected = True
                    break
            if not is_detected:
                self.transient_detected[t_id] = False

    def setup_phase_shift_dependent_variables(self, time_shift, data_slice):
        """
        Wrapper function to initialize variables that will change for each
        phase shift that is considered.

        Parameters
        -----------
        time_shift : `float`
            The offset given the currently considered phase shift by which
            to cyclically shift the SED evolution.
        data_slice : `np.array`, (N,)
            Numpy structured array containing the data related to the visits
            provided by the slicer.

        Returns
        ----------
        max_num_transients : `int`
            Updated number of the total simulated transients.
        observation_epoch : `np.array`, (N,)
            Array of transient light curve phases of observations of
            transients within this phase shift cycle.
        transient_id : `np.array`, (N,)
            Array of all the transient ids within this phase shift cycle,
            regardless of whether it is observed. dtype int.
        transient_id_start : `int`
            Updated starting id for next phase shift loop.
        transient_start_index : `np.array`, (N,)
            Array of the indicies for each transient that are the start of
            their observations in the observation array. dtype int.
        transient_end_index: `np.array`, (N,)
            Array of the indicies for each transient that are the end of
            their observations in the observation array. dtype int.
        """
        # Update the maximum possible transients that could have been
        # observed during survey_duration.
        self.max_num_transients += np.ceil(self.survey_duration / (self.transient_duration / 365.25))
        # Calculate the observation epoch for each transient lightcurve.
        self.observation_epoch = (
            data_slice[self.mjd_col] - self.survey_start + time_shift
        ) % self.transient_duration
        # Identify the observations which belong to each distinct transient.
        self.transient_id = (
            np.floor((data_slice[self.mjd_col] - self.survey_start) / self.transient_duration)
            + self.transient_id_start
        )
        # Set the starting id number for the next phase shift
        self.transient_id_start = self.transient_id.max()
        # Find the set of uniquely observed transients
        unique_transient_id = np.unique(self.transient_id)
        # Find the starting index for each transient_id.
        self.transient_start_index = np.searchsorted(self.transient_id, unique_transient_id, side="left")
        # Find the ending index of each transient_id.
        self.transient_end_index = np.searchsorted(self.transient_id, unique_transient_id, side="right")

    def setup_run_metric_variables(self, data_slice):
        """
        Wrapper function to handle basic initialization of variables used
        to run this metric.

        Parameters
        -----------
        data_slice : `np.array`, (N,)
            Numpy structured array containing the data related to the visits
            provided by the slicer.

        Returns
        ---------
        data_slice : `np.array`, (N,)
            Now sorted in time.
        survey_duration : `float`
            Defaults to the maximum between the chosen slicer and the user
            specified duration given to the metric.
        survey_start : `float`
            Defaults to user specified, or metric default, however if it is
            not defined sets to the earliest time in the given slicer.
        """
        # Sort the entire data_slice in order of time.
        data_slice.sort(order=self.mjd_col)

        # Check that survey_duration is not larger than the time of
        # observations we obtained. If it is, then the max_num_transients will
        # not be accurate.
        data_slice_time_span = (data_slice[self.mjd_col].max() - data_slice[self.mjd_col].min()) / 365.25
        # Take the maximum time delta, either specified or from the slicer, to
        # be the survey duration.
        self.survey_duration = np.max([data_slice_time_span, self.survey_duration])

        # Set the survey start based on the slicer unless otherwise specified.
        if self.survey_start is None:
            self.survey_start = data_slice[self.mjd_col].min()
        return data_slice

    def initialize_phase_loop_variables(self, data_slice):
        """
        Wrapper function to initialize variables needed for checking all
        transietnts and phase shifts for detected transients.

        Parameters
        -----------
        data_slice : `np.array`, (N,)
            Numpy structured array containing the data related to the visits
            provided by the slicer.

        Returns
        ---------
        time_phase_shifts : `np.array`, (N,)
            The phase offsets over which to iterate detections given the
            specfied number of phases to run.
        num_detected : `int`
            Initialized variable for the number detected, set to zero.
        max_num_transients : `int`
            Initialized variable for the total transients that are simulated
            counting the multiplicity due to phase shifts.
        transient_id_start : `int`
            The starting id for simulated transients that are observed. This
            accounts for if the requested length of the data_slice and the
            number of simulated transient observations mismatch the number
            of transients that fit in the specified survey duration given
            the user specified survey start.
        """
        # Depending on the number of phase shifts to apply and check the
        # detectability, compute the necessary time shifts corresponding to
        # phase division of the lightcurves.
        self.time_phase_shifts = (
            np.arange(self.num_phases_to_run) * self.transient_duration / float(self.num_phases_to_run)
        )
        # Total number of transient which have reached detection thresholds.
        self.num_detected = 0
        # Total number of transients which could possibly be detected,
        # given survey duration and transient duration.
        self.max_num_transients = 0
        # Set this, in case survey_start was set to be much earlier than this
        # data (so we start counting at 0).
        self.transient_id_start = -1 * np.floor(
            (data_slice[self.mjd_col].min() - self.survey_start) / self.transient_duration
        )

    def evaluate_snr_thresholds(self, data_slice):
        """
        Take the given data_slice and the set SNR thresholds for observations
        to be considered in further detections and compute which observations
        pass.

        Parameters
        -----------
        data_slice : `np.array`, (N,)
            Numpy structured array containing the data related to the visits
            provided by the slicer.

        Returns
        --------
        obs_above_SNR_threshold: `np.array`, (N,)
            `bool` array corresponding to all observations and whether or
            not, given their filter specified SNR threshold, they pass this
            thresholding cut.
        """
        # Initilize array for observations below or above SNR threshold
        self.obs_above_snr_threshold = np.zeros(len(self.light_curve_SNRs), dtype=bool)
        # Identify which detections rise above the required SNR threshold
        # in each filter.
        for filt in np.unique(data_slice[self.filter_col]):
            # Find the indices for observations in current filter.
            filter_match = np.where(data_slice[self.filter_col] == filt)[0]
            # Find the subset of the above indices which are above SNR
            # threshold condition, otherwise set threshold bool to False.
            self.obs_above_snr_threshold[filter_match] = np.where(
                self.light_curve_SNRs[filter_match] >= self.detect_snr[filt],
                True,
                False,
            )

    def run(self, data_slice, slice_point=None):
        """
        Calculate the detectability of a transient with the specified SED.

        If self.output_data is True, then returns the full lightcurve for each
        object instead of the total number of transients that are detected.

        Parameters
        ----------
        data_slice : `np.array`, (N,)
            Numpy structured array containing the data related to the visits
            provided by the slicer.
        slice_point : `dict`, optional
            Dictionary containing information about the slice_point currently
            active in the slicer.

        Returns
        -------
        result : `float` or `dict`
            The fraction of transients that could be detected.
            (if output_data is False) Otherwise, a dictionary
            with arrays of 'transient_id', 'lcMag', 'detected', 'expMJD',
            'SNR', 'filter', 'epoch'
        """

        data_slice = self.setup_run_metric_variables(data_slice)
        self.initialize_phase_loop_variables(data_slice)

        # Consider each different 'phase shift' separately.
        # We then just have a series of lightcurves, taking place back-to-back.
        for time_shift in self.time_phase_shifts:
            self.setup_phase_shift_dependent_variables(time_shift, data_slice)

            # Generate the actual light curve magnitudes and SNR
            self.make_lightcurve(self.observation_epoch, data_slice[self.filter_col])
            self.light_curve_SNRs = m52snr(self.light_curve_mags, data_slice[self.m5_col])

            # Check observations above the defined threshold for detection.
            self.evaluate_snr_thresholds(data_slice)
            # With useable observations computed,
            # evaluate all detection criteria
            self.evaluate_all_detection_criteria(data_slice)

        if self.output_data:
            # Output all the light curves, regardless of detection threshold,
            # but indicate which were 'detected'.
            # Only returns for one phase shift, not all.
            return {
                "transient_id": self.transient_id,
                "expMJD": data_slice[self.mjd_col],
                "epoch": self.observation_epoch,
                "filter": data_slice[self.filter_col],
                "lcMag": self.light_curve_mags,
                "SNR": self.light_curve_SNRs,
                "detected": self.transient_detected,
            }
        else:
            return float(self.num_detected)
