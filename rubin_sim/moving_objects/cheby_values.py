import os

import numpy as np
import pandas as pd

from .chebyshev_utils import chebeval

__all__ = ("ChebyValues", )


class ChebyValues:
    """Calculates positions, velocities, deltas, vmags and elongations,
    given a series of coefficients generated by ChebyFits.
    """

    def __init__(self):
        self.coeffs = {}
        self.coeff_keys = [
            "obj_id",
            "t_start",
            "t_end",
            "ra",
            "dec",
            "geo_dist",
            "vmag",
            "elongation",
        ]
        self.ephemeris_keys = [
            "ra",
            "dradt",
            "dec",
            "ddecdt",
            "geo_dist",
            "vmag",
            "elongation",
        ]

    def set_coefficients(self, cheby_fits):
        """Set coefficients using a ChebyFits object.
        (which contains a dictionary of obj_id, t_start, t_end, ra, dec, delta, vmag, and elongation lists).

        Parameters
        ----------
        cheby_fits : `rubin_sim.movingObjects.chebyFits`
            ChebyFits object, with attribute 'coeffs' - a dictionary of lists of coefficients.
        """
        self.coeffs = cheby_fits.coeffs
        # Convert list of coefficients into numpy arrays.
        for k in self.coeffs:
            self.coeffs[k] = np.array(self.coeffs[k])
        # Check that expected values were received.
        missing_keys = set(self.coeff_keys) - set(self.coeffs)
        if len(missing_keys) > 0:
            raise ValueError("Expected to find key(s) %s in coefficients." % " ".join(list[missing_keys]))
        self.coeffs["meanRA"] = self.coeffs["ra"].swapaxes(0, 1)[0]
        self.coeffs["meanDec"] = self.coeffs["dec"].swapaxes(0, 1)[0]

    def read_coefficients(self, cheby_fits_file):
        """Read coefficients from output file written by ChebyFits.

        Parameters
        ----------
        cheby_fits_file : `str`
            The filename of the coefficients file.
        """
        if not os.path.isfile(cheby_fits_file):
            raise IOError("Could not find cheby_fits_file at %s" % (cheby_fits_file))
        # Read the coefficients file.
        coeffs = pd.read_table(cheby_fits_file, delim_whitespace=True)
        # The header line provides information on the number of coefficients for each parameter.
        datacols = coeffs.columns.values
        cols = {}
        coeff_cols = ["ra", "dec", "geo_dist", "vmag", "elongation"]
        for k in coeff_cols:
            cols[k] = [x for x in datacols if x.startswith(k)]
        # Translate dataframe to dictionary of numpy arrays
        # while consolidating RA/Dec/Delta/Vmag/Elongation coeffs.
        self.coeffs["obj_id"] = coeffs.obj_id.values
        self.coeffs["t_start"] = coeffs.t_start.values
        self.coeffs["t_end"] = coeffs.t_end.values
        for k in coeff_cols:
            self.coeffs[k] = np.empty([len(cols[k]), len(coeffs)], float)
            for i in range(len(cols[k])):
                self.coeffs[k][i] = coeffs["%s_%d" % (k, i)].values
        # Add the mean RA and Dec columns (before swapping the coefficients axes).
        self.coeffs["meanRA"] = self.coeffs["ra"][0]
        self.coeffs["meanDec"] = self.coeffs["dec"][0]
        # Swap the coefficient axes so that they are [segment, coeff].
        for k in coeff_cols:
            self.coeffs[k] = self.coeffs[k].swapaxes(0, 1)

    def _eval_segment(self, segment_idx, times, subset_segments=None, mask=True):
        """Evaluate the ra/dec/delta/vmag/elongation values for a given segment at a series of times.

        Parameters
        ----------
        segment_idx : `int`
            The index in (each of) self.coeffs for the segment.
            e.g. the first segment, for each object.
        times : `np.ndarray`
            The times at which to evaluate the segment.
        subset_segments : `np.ndarray`, optional
            Optionally specify a subset of the total segment indexes.
            This lets you pick out particular obj_ids.
        mask : `bool`, optional
            If True, returns NaNs for values outside the range of times in the segment.
            If False, extrapolates segment for times outside the segment time range.

        Returns
        -------
        ephemeris : `dict`
           Dictionary of RA, Dec, delta, vmag, and elongation values for the segment indicated,
           at the time indicated.
        """
        if subset_segments is None:
            subset_segments = np.ones(len(self.coeffs["obj_id"]), dtype=bool)
        t_start = self.coeffs["t_start"][subset_segments][segment_idx]
        t_end = self.coeffs["t_end"][subset_segments][segment_idx]
        t_scaled = times - t_start
        t_interval = np.array([t_start, t_end]) - t_start
        # Evaluate RA/Dec/Delta/Vmag/elongation.
        ephemeris = {}
        ephemeris["ra"], ephemeris["dradt"] = chebeval(
            t_scaled,
            self.coeffs["ra"][subset_segments][segment_idx],
            interval=t_interval,
            do_velocity=True,
            mask=mask,
        )
        ephemeris["dec"], ephemeris["ddecdt"] = chebeval(
            t_scaled,
            self.coeffs["dec"][subset_segments][segment_idx],
            interval=t_interval,
            do_velocity=True,
            mask=mask,
        )
        ephemeris["dradt"] = ephemeris["dradt"] * np.cos(np.radians(ephemeris["dec"]))
        for k in ("geo_dist", "vmag", "elongation"):
            ephemeris[k], _ = chebeval(
                t_scaled,
                self.coeffs[k][subset_segments][segment_idx],
                interval=t_interval,
                do_velocity=False,
                mask=mask,
            )
        return ephemeris

    def get_ephemerides(self, times, obj_ids=None, extrapolate=False):
        """Find the ephemeris information for 'obj_ids' at 'time'.

        Implicit in how this is currently written is that the segments are all expected to cover the
        same start/end time range across all objects.
        They do not have to have the same segment length for all objects.

        Parameters
        ----------
        times : `float` or `np.ndarray`
            The time to calculate ephemeris positions.
        obj_ids : `np.ndarray`, opt
            The object ids for which to generate ephemerides. If None, then just uses all objects.
        extrapolate : `bool`, opt
            If True, extrapolate beyond ends of segments if time outside of segment range.
            If False, return ValueError if time is beyond range of segments.

        Returns
        -------
        ephemerides : `np.ndarray`
            The ephemeris positions for all objects.
            Note that these may not be sorted in the same order as obj_ids.
        """
        if isinstance(times, float) or isinstance(times, int):
            times = np.array([times], float)
        ntimes = len(times)
        ephemerides = {}
        # Find subset of segments which match obj_id, if specified.
        if obj_ids is None:
            obj_match = np.ones(len(self.coeffs["obj_id"]), dtype=bool)
            ephemerides["obj_id"] = np.unique(self.coeffs["obj_id"])
        else:
            if isinstance(obj_ids, str) or isinstance(obj_ids, int):
                obj_ids = np.array([obj_ids])
            obj_match = np.in1d(self.coeffs["obj_id"], obj_ids)
            ephemerides["obj_id"] = obj_ids
        # Now find ephemeris values.
        ephemerides["time"] = np.zeros((len(ephemerides["obj_id"]), ntimes), float) + times
        for k in self.ephemeris_keys:
            ephemerides[k] = np.zeros((len(ephemerides["obj_id"]), ntimes), float)
        for it, t in enumerate(times):
            # Find subset of segments which contain the appropriate time.
            # Look for simplest subset first.
            segments = np.where(
                (self.coeffs["t_start"][obj_match] <= t) & (self.coeffs["t_end"][obj_match] > t)
            )[0]
            if len(segments) == 0:
                seg_start = self.coeffs["t_start"][obj_match].min()
                seg_end = self.coeffs["t_end"][obj_match].max()
                if seg_start > t or seg_end < t:
                    if not extrapolate:
                        for k in self.ephemeris_keys:
                            ephemerides[k][:, it] = np.nan
                    else:
                        # Find the segments to use to extrapolate the times.
                        if seg_start > t:
                            segments = np.where(self.coeffs["t_start"][obj_match] == seg_start)[0]
                        if seg_end < t:
                            segments = np.where(self.coeffs["t_end"][obj_match] == seg_end)[0]
                elif seg_end == t:
                    # Not extrapolating, but outside the simple match case above.
                    segments = np.where(self.coeffs["t_end"][obj_match] == seg_end)[0]
            for i, segmentIdx in enumerate(segments):
                ephemeris = self._eval_segment(segmentIdx, t, obj_match, mask=False)
                for k in self.ephemeris_keys:
                    ephemerides[k][i][it] = ephemeris[k]
                ephemerides["obj_id"][i] = self.coeffs["obj_id"][obj_match][segmentIdx]
        if obj_ids is not None:
            if set(ephemerides["obj_id"]) != set(obj_ids):
                raise ValueError(
                    "Did not find expected match between obj_ids provided and ephemeride obj_ids."
                )
        return ephemerides
