__all__ = (
    "IntRounded",
    "int_binned_stat",
    "smallest_signed_angle",
    "SchemaConverter",
    "HpInComcamFov",
    "HpInLsstFov",
    "hp_kd_tree",
    "match_hp_resolution",
    "TargetoO",
    "SimTargetooServer",
    "set_default_nside",
    "restore_scheduler",
    "warm_start",
    "empty_observation",
    "scheduled_observation",
    "gnomonic_project_toxy",
    "gnomonic_project_tosky",
    "raster_sort",
    "run_info_table",
    "inrange",
    "season_calc",
    "create_season_offset",
)

import datetime
import os
import socket
import sqlite3

import healpy as hp
import matplotlib.path as mplPath
import numpy as np
import pandas as pd

import rubin_sim.version as rsVersion
from rubin_sim.utils import _build_tree, _hpid2_ra_dec, _xyz_from_ra_dec, xyz_angular_radius


def smallest_signed_angle(a1, a2):
    """
    via https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    """
    two_pi = 2.0 * np.pi
    x = a1 % two_pi
    y = a2 % two_pi
    a = (x - y) % two_pi
    b = (y - x) % two_pi
    result = b + 0
    alb = np.where(a < b)[0]
    result[alb] = -1.0 * a[alb]
    return result


class IntRounded:
    """
    Class to help force comparisons be made on scaled up integers,
    preventing machine precision issues cross-platforms

    Parameters
    ----------
    inval : number-like thing
        Some number that we want to compare
    scale : float (1e5)
        How much to scale inval before rounding and converting to an int.
    """

    def __init__(self, inval, scale=1e5):
        self.initial = inval
        self.value = np.round(inval * scale).astype(int)
        self.scale = scale

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __repr__(self):
        return str(self.initial)

    def __add__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial + other.initial, scale=out_scale)
        return result

    def __sub__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial - other.initial, scale=out_scale)
        return result

    def __mul__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial * other.initial, scale=out_scale)
        return result

    def __div__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = IntRounded(self.initial / other.initial, scale=out_scale)
        return result


def set_default_nside(nside=None):
    """
    Utility function to set a default nside value across the scheduler.

    XXX-there might be a better way to do this.

    Parameters
    ----------
    nside : int (None)
        A valid healpixel nside.
    """
    if not hasattr(set_default_nside, "nside"):
        if nside is None:
            nside = 32
        set_default_nside.nside = nside
    if nside is not None:
        set_default_nside.nside = nside
    return set_default_nside.nside


def restore_scheduler(observation_id, scheduler, observatory, in_obs, filter_sched=None, fast=True):
    """Put the scheduler and observatory in the state they were in. Handy for checking reward fucnction

    Parameters
    ----------
    observation_id : int
        The ID of the last observation that should be completed
    scheduler : rubin_sim.scheduler.scheduler object
        Scheduler object.
    observatory : rubin_sim.scheduler.observatory.Model_observatory
        The observaotry object
    in_obs : np.array or str
        Array of observations (formated like rubin_sim.scheduler.empty_observation). If a string,
        assumed to be a file and SchemaConverter is used to load it.
    filter_sched : rubin_sim.scheduler.scheduler object
        The filter scheduler. Note that we don't look up the official end of the previous night,
        so there is potential for the loaded filters to not match.
    fast : bool (True)
        If True, loads observations and passes them as an array to the `add_observations_array`
        method. If False, passes observations individually with `add_observation` method.
    """
    if type(in_obs) == str:
        sc = SchemaConverter()
        # load up the observations
        observations = sc.opsim2obs(in_obs)
    else:
        observations = in_obs
    good_obs = np.where(observations["ID"] <= observation_id)[0]
    observations = observations[good_obs]

    # replay the observations back into the scheduler
    # In the future, may be able to replace this with a
    # faster .add_observations_array method.

    if fast:
        scheduler.add_observations_array(observations)
        obs = observations[-1]
    else:
        for obs in observations:
            scheduler.add_observation(obs)

    if filter_sched is not None:
        # We've assumed the filter scheduler doesn't have any filters
        # May need to call the add_observation method on it if that
        # changes.

        # Make sure we have mounted the right filters for the night
        # XXX--note, this might not be exact, but should work most of the time.
        mjd_start_night = np.min(observations["mjd"][np.where(observations["night"] == obs["night"])])
        observatory.mjd = mjd_start_night
        conditions = observatory.return_conditions()
        filters_needed = filter_sched(conditions)
    else:
        filters_needed = ["u", "g", "r", "i", "y"]

    # update the observatory
    observatory.mjd = obs["mjd"] + observatory.observatory.visit_time(obs) / 3600.0 / 24.0
    observatory.obs_id_counter = obs["ID"] + 1
    observatory.observatory.parked = False
    observatory.observatory.current_ra_rad = obs["RA"]
    observatory.observatory.current_dec_rad = obs["dec"]
    observatory.observatory.current_rot_sky_pos_rad = obs["rotSkyPos"]
    observatory.observatory.cumulative_azimuth_rad = obs["cummTelAz"]
    observatory.observatory.current_filter = obs["filter"]
    observatory.observatory.mounted_filters = filters_needed
    # Note that we haven't updated last_az_rad, etc, but those values should be ignored.

    return scheduler, observatory


def int_binned_stat(ids, values, statistic=np.mean):
    """
    Like scipy.binned_statistic, but for unique int ids
    """

    uids = np.unique(ids)
    order = np.argsort(ids)

    ordered_ids = ids[order]
    ordered_values = values[order]

    left = np.searchsorted(ordered_ids, uids, side="left")
    right = np.searchsorted(ordered_ids, uids, side="right")

    stat_results = []
    for le, ri in zip(left, right):
        stat_results.append(statistic(ordered_values[le:ri]))

    return uids, np.array(stat_results)


def gnomonic_project_toxy(ra1, dec1, r_acen, deccen):
    """Calculate x/y projection of ra1/dec1 in system with center at r_acen, deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(deccen) * np.sin(dec1) + np.cos(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)
    x = np.cos(dec1) * np.sin(ra1 - r_acen) / cosc
    y = (np.cos(deccen) * np.sin(dec1) - np.sin(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, r_acen, deccen):
    """Calculate RA/dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/dec in radians."""
    denom = np.cos(deccen) - y * np.sin(deccen)
    RA = r_acen + np.arctan2(x, denom)
    dec = np.arctan2(np.sin(deccen) + y * np.cos(deccen), np.sqrt(x * x + denom * denom))
    return RA, dec


def match_hp_resolution(in_map, nside_out, unseen2nan=True):
    """Utility to convert healpix map resolution if needed and change hp.UNSEEN values to
    np.nan.

    Parameters
    ----------
    in_map : np.array
        A valie healpix map
    nside_out : int
        The desired resolution to convert in_map to
    unseen2nan : bool (True)
        If True, convert any hp.UNSEEN values to np.nan
    """
    current_nside = hp.npix2nside(np.size(in_map))
    if current_nside != nside_out:
        out_map = hp.ud_grade(in_map, nside_out=nside_out)
    else:
        out_map = in_map
    if unseen2nan:
        out_map[np.where(out_map == hp.UNSEEN)] = np.nan
    return out_map


def raster_sort(x0, order=["x", "y"], xbin=1.0):
    """XXXX--depriciated, use tsp instead.

    Do a sort to scan a grid up and down. Simple starting guess to traveling salesman.

    Parameters
    ----------
    x0 : array
    order : list
        Keys for the order x0 should be sorted in.
    xbin : float (1.)
        The bin_size to round off the first coordinate into

    returns
    -------
    array sorted so that it rasters up and down.
    """
    coords = x0.copy()
    bins = np.arange(
        coords[order[0]].min() - xbin / 2.0,
        coords[order[0]].max() + 3.0 * xbin / 2.0,
        xbin,
    )
    # digitize my bins
    coords[order[0]] = np.digitize(coords[order[0]], bins)
    order1 = np.argsort(coords, order=order)
    coords = coords[order1]
    places_to_invert = np.where(np.diff(coords[order[-1]]) < 0)[0]
    if np.size(places_to_invert) > 0:
        places_to_invert += 1
        indx = np.arange(coords.size)
        index_sorted = np.zeros(indx.size, dtype=int)
        index_sorted[0 : places_to_invert[0]] = indx[0 : places_to_invert[0]]

        for i, inv_pt in enumerate(places_to_invert[:-1]):
            if i % 2 == 0:
                index_sorted[inv_pt : places_to_invert[i + 1]] = indx[inv_pt : places_to_invert[i + 1]][::-1]
            else:
                index_sorted[inv_pt : places_to_invert[i + 1]] = indx[inv_pt : places_to_invert[i + 1]]

        if np.size(places_to_invert) % 2 != 0:
            index_sorted[places_to_invert[-1] :] = indx[places_to_invert[-1] :][::-1]
        else:
            index_sorted[places_to_invert[-1] :] = indx[places_to_invert[-1] :]
        return order1[index_sorted]
    else:
        return order1


class SchemaConverter:
    """
    Record how to convert an observation array to the standard opsim schema
    """

    def __init__(self):
        # Conversion dictionary, keys are opsim schema, values are observation dtype names
        self.convert_dict = {
            "observationId": "ID",
            "night": "night",
            "observationStartMJD": "mjd",
            "observationStartLST": "lmst",
            "numExposures": "nexp",
            "visitTime": "visittime",
            "visitExposureTime": "exptime",
            "proposalId": "survey_id",
            "fieldId": "field_id",
            "fieldRA": "RA",
            "fieldDec": "dec",
            "altitude": "alt",
            "azimuth": "az",
            "filter": "filter",
            "airmass": "airmass",
            "skyBrightness": "skybrightness",
            "cloud": "clouds",
            "seeingFwhm500": "FWHM_500",
            "seeingFwhmGeom": "FWHM_geometric",
            "seeingFwhmEff": "FWHMeff",
            "fiveSigmaDepth": "fivesigmadepth",
            "slewTime": "slewtime",
            "slewDistance": "slewdist",
            "paraAngle": "pa",
            "rotTelPos": "rotTelPos",
            "rotTelPos_backup": "rotTelPos_backup",
            "rotSkyPos": "rotSkyPos",
            "rotSkyPos_desired": "rotSkyPos_desired",
            "moonRA": "moonRA",
            "moonDec": "moonDec",
            "moonAlt": "moonAlt",
            "moonAz": "moonAz",
            "moonDistance": "moonDist",
            "moonPhase": "moonPhase",
            "sunAlt": "sunAlt",
            "sunAz": "sunAz",
            "solarElong": "solarElong",
            "note": "note",
            "target": "target",
        }
        # Column(s) not bothering to remap:  'observationStartTime': None,
        self.inv_map = {v: k for k, v in self.convert_dict.items()}
        # angles to convert
        self.angles_rad2deg = [
            "fieldRA",
            "fieldDec",
            "altitude",
            "azimuth",
            "slewDistance",
            "paraAngle",
            "rotTelPos",
            "rotSkyPos",
            "rotSkyPos_desired",
            "rotTelPos_backup",
            "moonRA",
            "moonDec",
            "moonAlt",
            "moonAz",
            "moonDistance",
            "sunAlt",
            "sunAz",
            "sunRA",
            "sunDec",
            "solarElong",
            "cummTelAz",
        ]
        # Put LMST into degrees too
        self.angles_hours2deg = ["observationStartLST"]

    def obs2opsim(self, obs_array, filename=None, info=None, delete_past=False):
        """convert an array of observations into a pandas dataframe with Opsim schema"""
        if delete_past:
            try:
                os.remove(filename)
            except OSError:
                pass

        df = pd.DataFrame(obs_array)
        df = df.rename(index=str, columns=self.inv_map)
        for colname in self.angles_rad2deg:
            df[colname] = np.degrees(df[colname])
        for colname in self.angles_hours2deg:
            df[colname] = df[colname] * 360.0 / 24.0

        if filename is not None:
            con = sqlite3.connect(filename)
            df.to_sql("observations", con, index=False)
            if info is not None:
                df = pd.DataFrame(info)
                df.to_sql("info", con)
        else:
            return df

    def opsim2obs(self, filename):
        """convert an opsim schema dataframe into an observation array."""

        con = sqlite3.connect(filename)
        df = pd.read_sql("select * from observations;", con)
        for key in self.angles_rad2deg:
            df[key] = np.radians(df[key])
        for key in self.angles_hours2deg:
            df[key] = df[key] * 24.0 / 360.0

        df = df.rename(index=str, columns=self.convert_dict)

        blank = empty_observation()
        final_result = np.empty(df.shape[0], dtype=blank.dtype)
        # XXX-ugh, there has to be a better way.
        for key in df.columns:
            final_result[key] = df[key].values

        return final_result


def empty_observation():
    """Return a numpy array that could be a handy observation record

    Returns
    -------
    empty_observation : `np.array`

    The numpy fields have the following labels. These fields are required to be set to be a valid observation
    the model observatory can execute.
    RA : `float`
       The Right Acension of the observation (center of the field) (Radians)
    dec : `float`
       Declination of the observation (Radians)
    mjd : `float`
       Modified Julian Date at the start of the observation (time shutter opens)
    exptime : `float`
       Total exposure time of the visit (seconds)
    filter : `str`
        The filter used. Should be one of u, g, r, i, z, y.
    rotSkyPos : `float`
        The rotation angle of the camera relative to the sky E of N (Radians). Will probably be overridden if rotTelPos is not np.nan.
    rotTelPos : `float`
        The rotation angle of the camera relative to the telescope (radians). Set to np.nan to force rotSkyPos to be used.
    rotSkyPos_desired : `float`
        If both rotSkyPos and rotTelPos are None/NaN, then rotSkyPos_desired is used. If rotSkyPos_desired results in a valid
        rotTelPos, rotSkyPos is set to rotSkyPos_desired. Otherwise, something else will happen--XXX.
    nexp : `int`
        Number of exposures in the visit.
    flush_by_mjd : `float`
        If we hit this MJD, we should flush the queue and refill it.
    note : `str` (optional)
        Usually good to set the note field so one knows which survey object generated the observation.
    target : `str` (optional)
        A note about what target is being observed.

    Additional Fields
    -----------------
    Lots of additional fields that get filled in by the model observatory when the observation is completed.
    See documentation at: https://rubin-sim.lsst.io/rs_scheduler/output_schema.html

    """

    names = [
        "ID",
        "RA",
        "dec",
        "mjd",
        "flush_by_mjd",
        "exptime",
        "filter",
        "rotSkyPos",
        "rotSkyPos_desired",
        "nexp",
        "airmass",
        "FWHM_500",
        "FWHMeff",
        "FWHM_geometric",
        "skybrightness",
        "night",
        "slewtime",
        "visittime",
        "slewdist",
        "fivesigmadepth",
        "alt",
        "az",
        "pa",
        "clouds",
        "moonAlt",
        "sunAlt",
        "note",
        "target",
        "field_id",
        "survey_id",
        "block_id",
        "lmst",
        "rotTelPos",
        "rotTelPos_backup",
        "moonAz",
        "sunAz",
        "sunRA",
        "sunDec",
        "moonRA",
        "moonDec",
        "moonDist",
        "solarElong",
        "moonPhase",
        "cummTelAz",
        "scripted_id",
    ]

    types = [
        int,
        float,
        float,
        float,
        float,
        float,
        "U40",
        float,
        float,
        int,
        float,
        float,
        float,
        float,
        float,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        "U40",
        "U40",
        int,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
    ]
    result = np.zeros(1, dtype=list(zip(names, types)))
    return result


def scheduled_observation(n=1):
    """Make an array to hold pre-scheduling observations

    Returns
    -------
    result : np.array


    things to fill in
    ------
    mjd_tol : `float`
        The tolerance on how early an observation can execute (days). Observation will be considered valid to attempt
        when mjd-mjd_tol < current MJD < flush_by_mjd (and other conditions below pass)
    dist_tol : `float`
        The angular distance an observation can be away from the specified RA,Dec and
        still count as completing the observation (radians).
    alt_min : `float`
        The minimum altitude to consider executing the observation (radians).
    alt_max : `float`
        The maximuim altitude to try observing (radians).
    HA_max : `float`
        Hour angle limit. Constraint is such that for hour angle running from 0 to 24 hours,
        the target RA,Dec must be greather than HA_max and less than HA_min. Set HA_max to 0 for
        no limit. (hours)
    HA_min : `float`
        Hour angle limit. Constraint is such that for hour angle running from 0 to 24 hours,
        the target RA,Dec must be greather than HA_max and less than HA_min. Set HA_min to 24 for
        no limit. (hours)
    sun_alt_max : float
        The sun must be below sun_alt_max to execute. (radians)
    observed : `bool`
        If set to True, scheduler will probably consider this a completed observation an never attempt it.

    """

    # Standard things from the usual observations
    names = [
        "ID",
        "RA",
        "dec",
        "mjd",
        "flush_by_mjd",
        "exptime",
        "filter",
        "rotSkyPos",
        "rotTelPos",
        "rotTelPos_backup",
        "rotSkyPos_desired",
        "nexp",
        "note",
        "target",
    ]
    types = [
        int,
        float,
        float,
        float,
        float,
        float,
        "U1",
        float,
        float,
        float,
        float,
        int,
        "U40",
        "U40",
    ]
    names += [
        "mjd_tol",
        "dist_tol",
        "alt_min",
        "alt_max",
        "HA_max",
        "HA_min",
        "sun_alt_max",
        "observed",
        "scripted_id",
    ]
    types += [float, float, float, float, float, float, float, bool, int]
    result = np.zeros(n, dtype=list(zip(names, types)))
    return result


def hp_kd_tree(nside=None, leafsize=100, scale=1e5):
    """
    Generate a KD-tree of healpixel locations

    Parameters
    ----------
    nside : int
        A valid healpix nside
    leafsize : int (100)
        Leafsize of the kdtree

    Returns
    -------
    tree : scipy kdtree
    """
    if nside is None:
        nside = set_default_nside()

    hpid = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2_ra_dec(nside, hpid)
    return _build_tree(ra, dec, leafsize, scale=scale)


class HpInLsstFov:
    """
    Return the healpixels within a pointing. A very simple LSST camera model with
    no chip/raft gaps.
    """

    def __init__(self, nside=None, fov_radius=1.75, scale=1e5):
        """
        Parameters
        ----------
        fov_radius : float (1.75)
            Radius of the filed of view in degrees
        scale : float (1e5)
            How many sig figs to round off to. Useful for ensuring identical results
            cross-ploatform where float precision can vary.
        """
        if nside is None:
            nside = set_default_nside()

        self.tree = hp_kd_tree(nside=nside, scale=scale)
        self.radius = np.round(xyz_angular_radius(fov_radius) * scale).astype(int)
        self.scale = scale

    def __call__(self, ra, dec, **kwargs):
        """
        Parameters
        ----------
        ra : float, array
            RA in radians
        dec : float, array
            Dec in radians

        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """
        x, y, z = _xyz_from_ra_dec(ra, dec)
        x = np.round(x * self.scale).astype(int)
        y = np.round(y * self.scale).astype(int)
        z = np.round(z * self.scale).astype(int)

        if np.size(x) == 1:
            indices = self.tree.query_ball_point((np.max(x), np.max(y), np.max(z)), self.radius)
        else:
            indices = self.tree.query_ball_point(np.vstack([x, y, z]).T, self.radius)
        return indices


class HpInComcamFov:
    """
    Return the healpixels within a ComCam pointing. Simple camera model
    with no chip gaps.
    """

    def __init__(self, nside=None, side_length=0.7):
        """
        Parameters
        ----------
        side_length : float (0.7)
            The length of one side of the square field of view (degrees).
        """
        if nside is None:
            nside = set_default_nside()
        self.nside = nside
        self.tree = hp_kd_tree(nside=nside)
        self.side_length = np.radians(side_length)
        self.inner_radius = xyz_angular_radius(side_length / 2.0)
        self.outter_radius = xyz_angular_radius(side_length / 2.0 * np.sqrt(2.0))
        # The positions of the raft corners, unrotated
        self.corners_x = np.array(
            [
                -self.side_length / 2.0,
                -self.side_length / 2.0,
                self.side_length / 2.0,
                self.side_length / 2.0,
            ]
        )
        self.corners_y = np.array(
            [
                self.side_length / 2.0,
                -self.side_length / 2.0,
                -self.side_length / 2.0,
                self.side_length / 2.0,
            ]
        )

    def __call__(self, ra, dec, rot_sky_pos=0.0):
        """
        Parameters
        ----------
        ra : float
            RA in radians
        dec : float
            Dec in radians
        rot_sky_pos : float
            The rotation angle of the camera in radians
        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """
        x, y, z = _xyz_from_ra_dec(np.max(ra), np.max(dec))
        # Healpixels within the inner circle
        indices = self.tree.query_ball_point((x, y, z), self.inner_radius)
        # Healpixels withing the outer circle
        indices_all = np.array(self.tree.query_ball_point((x, y, z), self.outter_radius))
        indices_to_check = indices_all[np.in1d(indices_all, indices, invert=True)]

        cos_rot = np.cos(rot_sky_pos)
        sin_rot = np.sin(rot_sky_pos)
        x_rotated = self.corners_x * cos_rot - self.corners_y * sin_rot
        y_rotated = self.corners_x * sin_rot + self.corners_y * cos_rot

        # Draw the square that we want to check if points are in.
        bb_path = mplPath.Path(
            np.array(
                [
                    [x_rotated[0], y_rotated[0]],
                    [x_rotated[1], y_rotated[1]],
                    [x_rotated[2], y_rotated[2]],
                    [x_rotated[3], y_rotated[3]],
                    [x_rotated[0], y_rotated[0]],
                ]
            )
        )

        ra_to_check, dec_to_check = _hpid2_ra_dec(self.nside, indices_to_check)

        # Project the indices to check to the tangent plane, see if they fall inside the polygon
        x, y = gnomonic_project_toxy(ra_to_check, dec_to_check, ra, dec)
        for i, xcheck in enumerate(x):
            # I wonder if I can do this all at once rather than a loop?
            if bb_path.contains_point((x[i], y[i])):
                indices.append(indices_to_check[i])

        return np.array(indices)


def run_info_table(observatory, extra_info=None):
    """
    Make a little table for recording the information about a run
    """

    observatory_info = observatory.get_info()
    if extra_info is not None:
        for key in extra_info:
            observatory_info.append([key, extra_info[key]])
    observatory_info = np.array(observatory_info)

    n_feature_entries = 3

    names = ["Parameter", "Value"]
    dtypes = ["|U200", "|U200"]
    result = np.zeros(observatory_info[:, 0].size + n_feature_entries, dtype=list(zip(names, dtypes)))

    # Fill in info about the run
    result[0]["Parameter"] = "Date, ymd"
    now = datetime.datetime.now()
    result[0]["Value"] = "%i, %i, %i" % (now.year, now.month, now.day)

    result[1]["Parameter"] = "hostname"
    result[1]["Value"] = socket.gethostname()

    result[2]["Parameter"] = "rubin_sim.__version__"
    result[2]["Value"] = rsVersion.__version__

    result[3:]["Parameter"] = observatory_info[:, 0]
    result[3:]["Value"] = observatory_info[:, 1]

    return result


def inrange(inval, minimum=-1.0, maximum=1.0):
    """
    Make sure values are within min/max
    """
    inval = np.array(inval)
    below = np.where(inval < minimum)
    inval[below] = minimum
    above = np.where(inval > maximum)
    inval[above] = maximum
    return inval


def warm_start(scheduler, observations, mjd_key="mjd"):
    """Replay a list of observations into the scheduler

    Parameters
    ----------
    scheduler : scheduler object
    observations : np.array
        An array of observation (e.g., from sqlite2observations)
    """

    # Check that observations are in order
    observations.sort(order=mjd_key)
    for observation in observations:
        scheduler.add_observation(observation)

    return scheduler


def season_calc(night, offset=0, modulo=None, max_season=None, season_length=365.25, floor=True):
    """
    Compute what season a night is in with possible offset and modulo
    using convention that night -365 to 0 is season -1.

    Parameters
    ----------
    night : int or array
        The night we want to convert to a season
    offset : float or array (0)
        Offset to be applied to night (days)
    modulo : int (None)
        If the season should be modulated (i.e., so we can get all even years)
        (seasons, years w/default season_length)
    max_season : int (None)
        For any season above this value (before modulo), set to -1
    season_length : float (365.25)
        How long to consider one season (nights)
    floor : bool (True)
        If true, take the floor of the season. Otherwise, returns season as a float
    """
    if np.size(night) == 1:
        night = np.ravel(np.array([night]))
    result = night + offset
    result = result / season_length
    if floor:
        result = np.floor(result)
    if max_season is not None:
        over_indx = np.where(IntRounded(result) >= IntRounded(max_season))

    if modulo is not None:
        neg = np.where(IntRounded(result) < IntRounded(0))
        result = result % modulo
        result[neg] = -1
    if max_season is not None:
        result[over_indx] = -1
    if floor:
        result = result.astype(int)
    return result


def create_season_offset(nside, sun_ra_rad):
    """
    Make an offset map so seasons roll properly
    """
    hpindx = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2_ra_dec(nside, hpindx)
    offset = ra - sun_ra_rad + 2.0 * np.pi
    offset = offset % (np.pi * 2)
    offset = offset * 365.25 / (np.pi * 2)
    offset = -offset - 365.25
    return offset


class TargetoO:
    """Class to hold information about a target of opportunity object

    Parameters
    ----------
    tooid : int
        Unique ID for the ToO.
    footprints : np.array
        np.array healpix maps. 1 for areas to observe, 0 for no observe.
    mjd_start : float
        The MJD the ToO starts
    duration : float
        Duration of the ToO (days).
    ra_rad_center : float
        RA of the estimated center of the event (radians).
    dec_rad_center : float
        Dec of the estimated center of the event (radians).
    """

    def __init__(
        self,
        tooid,
        footprint,
        mjd_start,
        duration,
        ra_rad_center=None,
        dec_rad_center=None,
    ):
        self.footprint = footprint
        self.duration = duration
        self.id = tooid
        self.mjd_start = mjd_start
        self.ra_rad_center = ra_rad_center
        self.dec_rad_center = dec_rad_center


class SimTargetooServer:
    """Wrapper to deliver a targetoO object at the right time"""

    def __init__(self, targeto_o_list):
        self.targeto_o_list = targeto_o_list
        self.mjd_starts = np.array([too.mjd_start for too in self.targeto_o_list])
        durations = np.array([too.duration for too in self.targeto_o_list])
        self.mjd_ends = self.mjd_starts + durations

    def __call__(self, mjd):
        in_range = np.where((mjd > self.mjd_starts) & (mjd < self.mjd_ends))[0]
        result = None
        if in_range.size > 0:
            result = [self.targeto_o_list[i] for i in in_range]
        return result
