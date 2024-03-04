__all__ = ("ChebyFits",)

import os
import warnings

import numpy as np

from .chebyshev_utils import chebfit, make_cheb_matrix, make_cheb_matrix_only_x
from .ooephemerides import PyOrbEphemerides, get_oorb_data_dir
from .orbits import Orbits


def three_sixty_to_neg(ra):
    """Wrap discontiguous RA values into more-contiguous results."""
    if (ra.min() < 100) and (ra.max() > 270):
        ra = np.where(ra > 270, ra - 360, ra)
    return ra


class ChebyFits:
    """Generates chebyshev coefficients for a provided set of orbits.

    Calculates true ephemerides using PyEphemerides, then fits these
    positions with a constrained Chebyshev Polynomial, using the routines
    in chebyshevUtils.py.

    Parameters
    ----------
    orbits_obj : `rubin_sim.moving_objects.Orbits`
        The orbits for which to fit chebyshev polynomial coefficients.
    t_start : `float`
        The starting point in time to fit coefficients. MJD.
    t_span : `float`
        The time span (starting at t_start) over which to fit coefficients
        (Days).
    time_scale : `str`, optional
        One of {'TAI', 'UTC', 'TT'}
        The timescale of the MJD time, t_start, and the time_scale
        that should be used with the chebyshev coefficients.
    obsCode : `int`, optional
        The observatory code of the location for which to generate
        ephemerides. Default I11 (Cerro Pachon).
    sky_tolerance : `float`, optional
        The desired tolerance in mas between ephemerides calculated by
        OpenOrb and fitted values.
        Default 2.5 mas.
    nCoeff_position : `int`, optional
        The number of Chebyshev coefficients to fit for the RA/Dec positions.
        Default 14.
    nCoeff_vmag : `int`, optional
        The number of Chebyshev coefficients to fit for the V magnitude values.
        Default 9.
    nCoeff_delta : `int`, optional
        The number of Chebyshev coefficients to fit for the distance
        between Earth/Object. Default 5.
    nCoeff_elongation : `int`, optional
        The number of Chebyshev coefficients to fit for the solar
        elongation. Default 5.
    ngran : `int`, optional
        The number of ephemeris points within each Chebyshev
        polynomial segment. Default 64.
    eph_file : `str`, optional
        The path to the JPL ephemeris file to use.
        Default is '$OORB_DATA/de405.dat'.
    n_decimal : `int`, optional
        The number of decimal places to allow in the segment length
        (and thus the times of the endpoints) can be limited to
        n_decimal places. Default 10.
        For LSST SIMS moving object database, this should be 13 decimal
        places for NEOs and 0 for all others.

    Notes
    -----
    Many chebyshev polynomials are used to fit one moving object over
    a given timeperiod; typically, the length of each segment is typically
    about 2 days for MBAs. The start and end of each segment must match
    exactly, and the entire segments must fit into the total timespan an
    integer number of times. This is accomplished by setting n_decimal to
    the number of decimal places desired in the 'time' value.
    For faster moving objects, this number needs be greater to allow for
    smaller subdivisions.
    It's tempting to allow flexibility to the point of not
    enforcing this non-overlap; however, then the resulting ephemeris
    may have multiple values depending on which polynomial segment was
    used to calculate the ephemeris.

    The length of each chebyshev polynomial is related to the number of
    ephemeris positions used to fit that polynomial by ngran:
    length = timestep * ngran
    The length of each polynomial is adjusted so that the residuals in
    RA/Dec position are less than sky_tolerance - default = 2.5mas.
    The polynomial length (and the resulting residuals) is affected
    by ngran (i.e. timestep).

    Default values are based on Yusra AlSayaad's work.
    """

    def __init__(
        self,
        orbits_obj,
        t_start,
        t_span,
        time_scale="TAI",
        obscode="I11",
        sky_tolerance=2.5,
        n_coeff_position=14,
        n_coeff_vmag=9,
        n_coeff_delta=5,
        n_coeff_elongation=6,
        ngran=64,
        eph_file=None,
        n_decimal=10,
    ):
        # Set up PyOrbEphemerides.
        if eph_file is None:
            self.eph_file = os.path.join(get_oorb_data_dir(), "de405.dat")
        else:
            self.eph_file = eph_file
        self.pyephems = PyOrbEphemerides(self.eph_file)
        # And then set orbits.
        self._set_orbits(orbits_obj)
        # Save input parameters.
        # We have to play some games with the start and end times,
        # using Decimal, in order to get the subdivision and times to
        # match exactly, up to n_decimal places.
        self.n_decimal = int(n_decimal)
        self.t_start = round(t_start, self.n_decimal)
        self.t_span = round(t_span, self.n_decimal)
        self.t_end = round(self.t_start + self.t_span, self.n_decimal)
        if time_scale.upper() == "TAI":
            self.time_scale = "TAI"
        elif time_scale.upper() == "UTC":
            self.time_scale = "UTC"
        elif time_scale.upper() == "TT":
            self.time_scale = "TT"
        else:
            raise ValueError("Do not understand time_scale; use TAI, UTC or TT.")
        self.obscode = obscode
        self.sky_tolerance = sky_tolerance
        self.n_coeff = {}
        self.n_coeff["position"] = int(n_coeff_position)
        self.n_coeff["geo_dist"] = int(n_coeff_delta)
        self.n_coeff["vmag"] = int(n_coeff_vmag)
        self.n_coeff["elongation"] = int(n_coeff_elongation)
        self.ngran = int(ngran)
        # Precompute multipliers (we only do this once).
        self._precompute_multipliers()
        # Initialize attributes to save the coefficients and residuals.
        self.coeffs = {
            "obj_id": [],
            "t_start": [],
            "t_end": [],
            "ra": [],
            "dec": [],
            "geo_dist": [],
            "vmag": [],
            "elongation": [],
        }
        self.resids = {
            "obj_id": [],
            "t_start": [],
            "t_end": [],
            "pos": [],
            "geo_dist": [],
            "vmag": [],
            "elongation": [],
        }
        self.failed = []

    def _set_orbits(self, orbits_obj):
        """Set the orbits, to be used to generate ephemerides.

        Parameters
        ----------
        orbits_obj : `rubin_sim.moving_objects.Orbits`
           The orbits to use to generate ephemerides.
        """
        if not isinstance(orbits_obj, Orbits):
            raise ValueError("Need to provide an Orbits object.")
        self.orbits_obj = orbits_obj
        self.pyephems.set_orbits(self.orbits_obj)

    def _precompute_multipliers(self):
        """Calculate multipliers for Chebyshev fitting.

        Calculate these once, rather than for each segment.
        """
        # The nPoints are predetermined here, based on Yusra's earlier work.
        # The weight is based on Newhall, X. X. 1989, Celestial Mechanics,
        # 45, p. 305-310
        self.multipliers = {}
        self.multipliers["position"] = make_cheb_matrix(self.ngran + 1, self.n_coeff["position"], weight=0.16)
        self.multipliers["vmag"] = make_cheb_matrix_only_x(self.ngran + 1, self.n_coeff["vmag"])
        self.multipliers["geo_dist"] = make_cheb_matrix_only_x(self.ngran + 1, self.n_coeff["geo_dist"])
        self.multipliers["elongation"] = make_cheb_matrix_only_x(self.ngran + 1, self.n_coeff["elongation"])

    def _length_to_timestep(self, length):
        """Convert chebyshev polynomial segment lengths to the
        corresponding timestep over the segment.

        Parameters
        ----------
        length : `float`
            The chebyshev polynomial segment length (nominally, days).

        Returns
        -------
        timestep : `float`
            The corresponding timestep, = length/ngran (nominally, days).
        """
        return length / self.ngran

    def make_all_times(self):
        """Using t_start and t_end, generate a numpy array containing
        times spaced at timestep = self.length/self.ngran.
        The expected use for this time array would be to generate
        ephemerides at each timestep.

        Returns
        -------
        times : `np.ndarray`
            Numpy array of times.
        """
        try:
            self.length
        except AttributeError:
            raise AttributeError("Need to set self.timestep first, using calcSegmentLength.")
        timestep = self._length_to_timestep(self.length)
        times = np.arange(self.t_start, self.t_end + timestep / 2, timestep)
        return times

    def generate_ephemerides(self, times, by_object=True):
        """Generate ephemerides using OpenOrb for all orbits.

        Parameters
        ----------
        times : `np.ndarray`
            The times to use for ephemeris generation.
        """
        return self.pyephems.generate_ephemerides(
            times,
            obscode=self.obscode,
            eph_mode="N",
            eph_type="basic",
            time_scale=self.time_scale,
            by_object=by_object,
        )

    def _round_length(self, length):
        """Modify length, to fit in an 'integer multiple' within the
        t_start/t_end, and to have the desired number of decimal values.

        Parameters
        ----------
        length : `float`
            The input length value to be rounded.

        Returns
        -------
        length : `float`
            The rounded length value.
        """
        length = round(length, self.n_decimal)
        length_in = length
        # Make length an integer value within the time interval,
        # to last decimal place accuracy.
        counter = 0
        prev_int_factor = 0
        num_tolerance = 10.0 ** (-1 * (self.n_decimal - 1))
        while ((self.t_span % length) > num_tolerance) and (length > 0) and (counter < 20):
            int_factor = int(self.t_span / length) + 1  # round up / ceiling
            if int_factor == prev_int_factor:
                int_factor = prev_int_factor + 1
            prev_int_factor = int_factor
            length = round(self.t_span / int_factor, self.n_decimal)
            counter += 1
        if (self.t_span % length) > num_tolerance or (length <= 0):
            # Add this entire segment into the failed list.
            for obj_id in self.orbits_obj.orbits["obj_id"].as_matrix():
                self.failed.append((obj_id, self.t_start, self.t_end))
            raise ValueError(
                "Could not find a suitable length for the timespan (start %f, span %f), "
                "starting with length %s, ending with length value %f"
                % (self.t_start, self.t_span, str(length_in), length)
            )
        return length

    def _test_residuals(self, length, cutoff=99):
        """Calculate the position residual, for a test case.
        Convenience function to make calcSegmentLength easier to read.
        """
        # The pos_resid used will be the 'cutoff' percentile of all
        # max residuals per object.
        max_pos_resids = np.zeros(len(self.orbits_obj), float)
        timestep = self._length_to_timestep(length)
        # Test for one segment near the start (would do at midpoint,
        # but for long timespans this is not efficient ..
        # a point near the start should be fine).
        times = np.arange(self.t_start, self.t_start + length + timestep / 2, timestep)
        # We must regenerate ephemerides here, because the timestep is
        # different each time.
        ephs = self.generate_ephemerides(times, by_object=True)
        # Look for the coefficients and residuals.
        for i, e in enumerate(ephs):
            coeff_ra, coeff_dec, max_pos_resids[i] = self._get_coeffs_position(e)
        # Find a representative value and return.
        pos_resid = np.percentile(max_pos_resids, cutoff)
        ratio = pos_resid / self.sky_tolerance
        return pos_resid, ratio

    def calc_segment_length(self, length=None):
        """Set the typical initial ephemeris timestep and segment length
        for all objects between t_start/t_end.

        Sets self.length.

        The segment length will fit into the time period between
        t_start/t_end an approximately integer multiple of times,
        and will only have a given number of decimal places.

        Parameters
        ----------
        length : `float`, optional
            If specified, this value for the length is used,
            instead of calculating it here.
        """
        # If length is specified, use it and do nothing else.
        if length is not None:
            length = self._round_length(length)
            pos_resid, ratio = self._test_residuals(length)
            if pos_resid > self.sky_tolerance:
                warnings.warn(
                    "Will set length and timestep, but this value of length "
                    "produces residuals (%f) > skyTolerance (%f)." % (pos_resid, self.sky_tolerance)
                )
            self.length = length
            return
        # Otherwise, calculate an appropriate length and timestep.
        # Give a guess at a very approximate segment length,
        # given the skyTolerance,
        # purposefully trying to overestimate this value.
        # The actual behavior of the residuals is not linear with
        # segment length.
        # There is a linear increase at low residuals
        # < ~2 mas / segment length < 2 days
        # Then at around 2 days the residuals blow up,
        # increasing rapidly to about 5000 mas
        #   (depending on orbit .. TNOs, for example, increase but
        #   only to about 300 mas, when the residuals resume ~linear growth
        #   out to 70 day segments if ngran=128)
        # Make an arbitrary cap on segment length at 60 days,
        # (25000 mas) ~.5 arcminute accuracy.
        max_length = 60
        max_iterations = 50
        if self.sky_tolerance < 5:
            # This is the cap of the low-linearity regime,
            # looping below will refine this value.
            length = 2.0
        elif self.sky_tolerance >= 5000:
            # Make a very rough guess.
            length = np.round((5000.0 / 20.0) * (self.sky_tolerance - 5000.0)) + 5.0
            length = np.min([max_length, int(length * 10) / 10.0])
        else:
            # Try to pick a length in the middle of the fast increase.
            length = 4.0
        # Tidy up some characteristics of "length":
        # make it fit an integer number of times into overall timespan.
        # and use a given number of decimal places
        # (easier for database storage).
        length = self._round_length(length)
        # Check the resulting residuals.
        pos_resid, ratio = self._test_residuals(length)
        counter = 0
        # Now should be relatively close.
        # Start to zero in using slope around the value.ngran
        while pos_resid > self.sky_tolerance and counter <= max_iterations and length > 0:
            length = length / 2
            length = self._round_length(length)
            pos_resid, ratio = self._test_residuals(length)
            counter += 1
        if counter > max_iterations or length <= 0:
            # Add this entire segment into the failed list.
            for obj_id in self.orbits_obj.orbits["obj_id"].as_matrix():
                self.failed.append((obj_id, self.t_start, self.t_end))
            error_message = "Could not find good segment length to meet skyTolerance %f" % (
                self.sky_tolerance
            )
            error_message += " milliarcseconds within %d iterations. " % (max_iterations)
            error_message += "Final residual was %f milli-arcseconds." % (pos_resid)
            raise ValueError(error_message)
        else:
            self.length = length

    def _get_coeffs_position(self, ephs):
        """Calculate coefficients for the ra/dec values of a
        single objects ephemerides.

        Parameters
        ----------
        times : `np.ndarray`
            The times of the ephemerides.
        ephs : `np.ndarray`
            The structured array returned by PyOrbEphemerides
            holding ephemeris values, for one object.

        Returns
        -------
        coeff_ra, coeff_dec, max_pos_resid : `np.ndarray`, `np.ndarray`,
        `np.ndarray`
            The ra coefficients, dec coefficients, and the positional error
            residuals between fit and ephemeris values, in mas.
        """
        dradt_coord = ephs["dradt"] / np.cos(np.radians(ephs["dec"]))
        coeff_ra, resid_ra, rms_ra_resid, max_ra_resid = chebfit(
            ephs["time"],
            three_sixty_to_neg(ephs["ra"]),
            dxdt=dradt_coord,
            x_multiplier=self.multipliers["position"][0],
            dx_multiplier=self.multipliers["position"][1],
            n_poly=self.n_coeff["position"],
        )
        coeff_dec, resid_dec, rms_dec_resid, max_dec_resid = chebfit(
            ephs["time"],
            ephs["dec"],
            dxdt=ephs["ddecdt"],
            x_multiplier=self.multipliers["position"][0],
            dx_multiplier=self.multipliers["position"][1],
            n_poly=self.n_coeff["position"],
        )
        max_pos_resid = np.max(np.sqrt(resid_dec**2 + (resid_ra * np.cos(np.radians(ephs["dec"]))) ** 2))
        # Convert position residuals to mas.
        max_pos_resid *= 3600.0 * 1000.0
        return coeff_ra, coeff_dec, max_pos_resid

    def _get_coeffs_other(self, ephs):
        """Calculate coefficients for the ra/dec values of a
        single objects ephemerides.

        Parameters
        ----------
        ephs : `np.ndarray`
            The structured array returned by PyOrbEphemerides
            holding ephemeris values, for one object.

        Returns
        -------
        coeffs, max_resids : `dict` of `float`
            Dictionary containing the coefficients for each of 'geo_dist',
            'vmag', 'elongation', and another dictionary containing the
            max residual values for each of 'geo_dist', 'vmag', 'elongation'.
        """
        coeffs = {}
        max_resids = {}
        for key, ephValue in zip(("geo_dist", "vmag", "elongation"), ("geo_dist", "magV", "solarelon")):
            coeffs[key], resid, rms, max_resids[key] = chebfit(
                ephs["time"],
                ephs[ephValue],
                dxdt=None,
                x_multiplier=self.multipliers[key],
                dx_multiplier=None,
                n_poly=self.n_coeff[key],
            )
        return coeffs, max_resids

    def calc_segments(self):
        """Run the calculation of all segments over the entire time span."""
        # First calculate ephemerides for all objects, over entire time span.
        # For some objects, we will end up recalculating the ephemeride values,
        # but most should be fine.
        times = self.make_all_times()
        ephs = self.generate_ephemerides(times)
        eps = self._length_to_timestep(self.length) / 4.0
        # Loop through each object to generate coefficients.
        for orbit_obj, e in zip(self.orbits_obj, ephs):
            t_segment_start = self.t_start
            # Cycle through all segments until we reach the end of the
            # period we're fitting.
            while t_segment_start < (self.t_end - eps):
                # Identify the subset of times and ephemerides
                # which are relevant for this segment
                # (at the default segment size).
                t_segment_end = round(t_segment_start + self.length, self.n_decimal)
                subset = np.where((times >= t_segment_start) & (times < t_segment_end + eps))
                self.calc_one_segment(orbit_obj, e[subset])
                t_segment_start = t_segment_end

    def calc_one_segment(self, orbit_obj, ephs):
        """Calculate the coefficients for a single Chebyshev segment,
        for a single object.

        Calculates the coefficients and residuals, and saves this
        information to self.coeffs, self.resids, and
        (if there are problems), self.failed.

        Parameters
        ----------
        orbit_obj : `rubin_sim.moving_objects.Orbits`
            The single Orbits object we're fitting at the moment.
        ephs : `np.ndarray`
            The ephemerides we're fitting at the moment
            (for the single object / single segment).
        """
        obj_id = orbit_obj.orbits.obj_id.iloc[0]
        t_segment_start = ephs["time"][0]
        t_segment_end = ephs["time"][-1]
        coeff_ra, coeff_dec, max_pos_resid = self._get_coeffs_position(ephs)
        if max_pos_resid > self.sky_tolerance:
            self._subdivide_segment(orbit_obj, ephs)
        else:
            coeffs, max_resids = self._get_coeffs_other(ephs)
            fit_failed = False
            for k in max_resids:
                if np.isnan(max_resids[k]):
                    fit_failed = True
            if fit_failed:
                warnings.warn(
                    "Fit failed for orbit_obj %s for times between %f and %f"
                    % (obj_id, t_segment_start, t_segment_end)
                )
                self.failed.append((orbit_obj.orbits["obj_id"], t_segment_start, t_segment_end))
            else:
                # Consolidate items into the tracked coefficient values.
                self.coeffs["obj_id"].append(obj_id)
                self.coeffs["t_start"].append(t_segment_start)
                self.coeffs["t_end"].append(t_segment_end)
                self.coeffs["ra"].append(coeff_ra)
                self.coeffs["dec"].append(coeff_dec)
                self.coeffs["geo_dist"].append(coeffs["geo_dist"])
                self.coeffs["vmag"].append(coeffs["vmag"])
                self.coeffs["elongation"].append(coeffs["elongation"])
                # Consolidate items into the tracked residual values.
                self.resids["obj_id"].append(obj_id)
                self.resids["t_start"].append(t_segment_start)
                self.resids["t_end"].append(t_segment_end)
                self.resids["pos"].append(max_pos_resid)
                self.resids["geo_dist"].append(max_resids["geo_dist"])
                self.resids["vmag"].append(max_resids["geo_dist"])
                self.resids["elongation"].append(max_resids["elongation"])

    def _subdivide_segment(self, orbit_obj, ephs):
        """Subdivide a segment, then calculate the segment coefficients.

        Parameters
        ----------
        orbit_obj : `rubin_sim.moving_objects.Orbits`
            The single Orbits object we're fitting at the moment.
        ephs : `np.ndarray`
            The ephemerides we're fitting at the moment
            (for the single object / single segment).
        """
        new_cheby = ChebyFits(
            orbit_obj,
            ephs["time"][0],
            (ephs["time"][-1] - ephs["time"][0]),
            time_scale=self.time_scale,
            obscode=self.obscode,
            sky_tolerance=self.sky_tolerance,
            n_coeff_position=self.n_coeff["position"],
            n_coeff_vmag=self.n_coeff["vmag"],
            n_coeff_delta=self.n_coeff["geo_dist"],
            n_coeff_elongation=self.n_coeff["elongation"],
            ngran=self.ngran,
            eph_file=self.eph_file,
            n_decimal=self.n_decimal,
        )
        try:
            new_cheby.calc_segment_length()
        except ValueError as ve:
            # Could not find a good segment length.
            warningmessage = "Objid %s, segment %f to %f " % (
                orbit_obj.orbits.obj_id.iloc[0],
                ephs["time"][0],
                ephs["time"][-1],
            )
            warningmessage += " - error: %s" % (ve)
            warnings.warn(warningmessage)
            self.failed += new_cheby.failed
            return
        new_cheby.calc_segments()
        # Add subdivided segment values into tracked values here.
        for k in self.coeffs:
            self.coeffs[k] += new_cheby.coeffs[k]
        for k in self.resids:
            self.resids[k] += new_cheby.resids[k]
        self.failed += new_cheby.failed

    def write(self, coeff_file, resid_file, failed_file, append=False):
        """Write coefficients, residuals and failed fits to disk.

        Parameters
        ----------
        coeff_file : `str`
            The filename for the coefficient values.
        resid_file : `str`
            The filename for the residual values.
        failed_file : `str`
            The filename to write the failed fit information
            (if failed objects exist).
        append : `bool`, optional
            Flag to append (or overwrite) the output files.
        """

        warnings.warn(
            "Writing cheby fit values may have cross-platform issues. Consider passing values directly"
        )

        if append:
            open_mode = "aw"
        else:
            open_mode = "w"
        # Write a header to the coefficients file, if writing to a new file:
        if (not append) or (not os.path.isfile(coeff_file)):
            header = "obj_id t_start t_end "
            header += " ".join(["ra_%d" % x for x in range(self.n_coeff["position"])]) + " "
            header += " ".join(["dec_%d" % x for x in range(self.n_coeff["position"])]) + " "
            header += " ".join(["geo_dist_%d" % x for x in range(self.n_coeff["geo_dist"])]) + " "
            header += " ".join(["vmag_%d" % x for x in range(self.n_coeff["vmag"])]) + " "
            header += " ".join(["elongation_%d" % x for x in range(self.n_coeff["elongation"])])
        else:
            header = None
        if (not append) or (not os.path.isfile(resid_file)):
            resid_header = "obj_id segNum t_start t_end length pos geo_dist vmag elong"
        else:
            resid_header = None
        timeformat = "%." + "%s" % self.n_decimal + "f"
        with open(coeff_file, open_mode) as f:
            if header is not None:
                print(header, file=f)
            for i, (obj_id, t_start, t_end, cRa, cDec, cDelta, cVmag, cE) in enumerate(
                zip(
                    self.coeffs["obj_id"],
                    self.coeffs["t_start"],
                    self.coeffs["t_end"],
                    self.coeffs["ra"],
                    self.coeffs["dec"],
                    self.coeffs["geo_dist"],
                    self.coeffs["vmag"],
                    self.coeffs["elongation"],
                )
            ):
                print(
                    "%s %s %s %s %s %s %s %s"
                    % (
                        obj_id,
                        timeformat % t_start,
                        timeformat % t_end,
                        " ".join("%.14e" % j for j in cRa),
                        " ".join("%.14e" % j for j in cDec),
                        " ".join("%.7e" % j for j in cDelta),
                        " ".join("%.7e" % j for j in cVmag),
                        " ".join("%.7e" % j for j in cE),
                    ),
                    file=f,
                )

        with open(resid_file, open_mode) as f:
            if resid_header is not None:
                print(resid_header, file=f)
            for i, (obj_id, t_start, t_end, rPos, rDelta, rVmag, rE) in enumerate(
                zip(
                    self.resids["obj_id"],
                    self.resids["t_start"],
                    self.resids["t_end"],
                    self.resids["pos"],
                    self.resids["geo_dist"],
                    self.resids["vmag"],
                    self.resids["elongation"],
                )
            ):
                print(
                    "%s %i %.14f %.14f %.14f %.14e %.14e %.14e %.14e"
                    % (
                        obj_id,
                        i + 1,
                        t_start,
                        t_end,
                        (t_end - t_start),
                        rPos,
                        rDelta,
                        rVmag,
                        rE,
                    ),
                    file=f,
                )

        if len(self.failed) > 0:
            with open(failed_file, open_mode) as f:
                for i, failed in enumerate(self.failed):
                    print(" ".join([str(x) for x in failed]), file=f)
