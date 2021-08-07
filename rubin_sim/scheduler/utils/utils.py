import os
import sqlite3 as db
import datetime
import socket
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.path as mplPath
from rubin_sim.utils import _hpid2RaDec, xyz_angular_radius, _buildTree, _xyz_from_ra_dec
from rubin_sim.site_models import FieldsDatabase
import rubin_sim


def smallest_signed_angle(a1, a2):
    """
    via https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles"""
    TwoPi = 2.*np.pi
    x = a1 % TwoPi
    y = a2 % TwoPi
    a = (x - y) % TwoPi
    b = (y - x) % TwoPi
    result = b+0
    alb = np.where(a < b)[0]
    result[alb] = -1.*a[alb]
    return result

class int_rounded(object):
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
        result = int_rounded(self.initial + other.initial, scale=out_scale)
        return result

    def __sub__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = int_rounded(self.initial - other.initial, scale=out_scale)
        return result

    def __mul__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = int_rounded(self.initial * other.initial, scale=out_scale)
        return result

    def __div__(self, other):
        out_scale = np.min([self.scale, other.scale])
        result = int_rounded(self.initial / other.initial, scale=out_scale)
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
    if not hasattr(set_default_nside, 'nside'):
        if nside is None:
            nside = 32
        set_default_nside.nside = nside
    if nside is not None:
        set_default_nside.nside = nside
    return set_default_nside.nside


def restore_scheduler(observationId, scheduler, observatory, filename, filter_sched=None):
    """Put the scheduler and observatory in the state they were in. Handy for checking reward fucnction

    Parameters
    ----------
    observationId : int
        The ID of the last observation that should be completed
    scheduler : rubin_sim.scheduler.scheduler object
        Scheduler object.
    observatory : rubin_sim.schedler.observatory.Model_observatory
        The observaotry object
    filename : str
        The output sqlite dayabase to use
    filter_sched : rubin_sim.scheduler.scheduler object
        The filter scheduler. Note that we don't look up the official end of the previous night,
        so there is potential for the loaded filters to not match.
    """
    sc = schema_converter()
    # load up the observations
    observations = sc.opsim2obs(filename)
    good_obs = np.where(observations['ID'] <= observationId)[0]
    observations = observations[good_obs]

    # replay the observations back into the scheduler
    for obs in observations:
        scheduler.add_observation(obs)
        if filter_sched is not None:
            filter_sched.add_observation(obs)

    if filter_sched is not None:
        # Make sure we have mounted the right filters for the night
        # XXX--note, this might not be exact, but should work most of the time.
        mjd_start_night = np.min(observations['mjd'][np.where(observations['night'] == obs['night'])])
        observatory.mjd = mjd_start_night
        conditions = observatory.return_conditions()
        filters_needed = filter_sched(conditions)
    else:
        filters_needed = ['u', 'g', 'r', 'i', 'y']

    # update the observatory
    observatory.mjd = obs['mjd'] + observatory.observatory.visit_time(obs)/3600./24.
    observatory.observatory.parked = False
    observatory.observatory.current_RA_rad = obs['RA']
    observatory.observatory.current_dec_rad = obs['dec']
    observatory.observatory.current_rotSkyPos_rad = obs['rotSkyPos']
    observatory.observatory.cumulative_azimuth_rad = obs['cummTelAz']
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

    left = np.searchsorted(ordered_ids, uids, side='left')
    right = np.searchsorted(ordered_ids, uids, side='right')

    stat_results = []
    for le, ri in zip(left, right):
        stat_results.append(statistic(ordered_values[le:ri]))

    return uids, np.array(stat_results)


def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, RAcen, Deccen):
    """Calculate RA/Dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/Dec in radians."""
    denom = np.cos(Deccen) - y * np.sin(Deccen)
    RA = RAcen + np.arctan2(x, denom)
    Dec = np.arctan2(np.sin(Deccen) + y * np.cos(Deccen), np.sqrt(x*x + denom*denom))
    return RA, Dec


def match_hp_resolution(in_map, nside_out, UNSEEN2nan=True):
    """Utility to convert healpix map resolution if needed and change hp.UNSEEN values to
    np.nan.

    Parameters
    ----------
    in_map : np.array
        A valie healpix map
    nside_out : int
        The desired resolution to convert in_map to
    UNSEEN2nan : bool (True)
        If True, convert any hp.UNSEEN values to np.nan
    """
    current_nside = hp.npix2nside(np.size(in_map))
    if current_nside != nside_out:
        out_map = hp.ud_grade(in_map, nside_out=nside_out)
    else:
        out_map = in_map
    if UNSEEN2nan:
        out_map[np.where(out_map == hp.UNSEEN)] = np.nan
    return out_map


def raster_sort(x0, order=['x', 'y'], xbin=1.):
    """XXXX--depriciated, use tsp instead.

    Do a sort to scan a grid up and down. Simple starting guess to traveling salesman.

    Parameters
    ----------
    x0 : array
    order : list
        Keys for the order x0 should be sorted in.
    xbin : float (1.)
        The binsize to round off the first coordinate into

    returns
    -------
    array sorted so that it rasters up and down.
    """
    coords = x0.copy()
    bins = np.arange(coords[order[0]].min()-xbin/2., coords[order[0]].max()+3.*xbin/2., xbin)
    # digitize my bins
    coords[order[0]] = np.digitize(coords[order[0]], bins)
    order1 = np.argsort(coords, order=order)
    coords = coords[order1]
    places_to_invert = np.where(np.diff(coords[order[-1]]) < 0)[0]
    if np.size(places_to_invert) > 0:
        places_to_invert += 1
        indx = np.arange(coords.size)
        index_sorted = np.zeros(indx.size, dtype=int)
        index_sorted[0:places_to_invert[0]] = indx[0:places_to_invert[0]]

        for i, inv_pt in enumerate(places_to_invert[:-1]):
            if i % 2 == 0:
                index_sorted[inv_pt:places_to_invert[i+1]] = indx[inv_pt:places_to_invert[i+1]][::-1]
            else:
                index_sorted[inv_pt:places_to_invert[i+1]] = indx[inv_pt:places_to_invert[i+1]]

        if np.size(places_to_invert) % 2 != 0:
            index_sorted[places_to_invert[-1]:] = indx[places_to_invert[-1]:][::-1]
        else:
            index_sorted[places_to_invert[-1]:] = indx[places_to_invert[-1]:]
        return order1[index_sorted]
    else:
        return order1


class schema_converter(object):
    """
    Record how to convert an observation array to the standard opsim schema
    """
    def __init__(self):
        # Conversion dictionary, keys are opsim schema, values are observation dtype names
        self.convert_dict = {'observationId': 'ID', 'night': 'night',
                             'observationStartMJD': 'mjd',
                             'observationStartLST': 'lmst', 'numExposures': 'nexp',
                             'visitTime': 'visittime', 'visitExposureTime': 'exptime',
                             'proposalId': 'survey_id', 'fieldId': 'field_id',
                             'fieldRA': 'RA', 'fieldDec': 'dec', 'altitude': 'alt', 'azimuth': 'az',
                             'filter': 'filter', 'airmass': 'airmass', 'skyBrightness': 'skybrightness',
                             'cloud': 'clouds', 'seeingFwhm500': 'FWHM_500',
                             'seeingFwhmGeom': 'FWHM_geometric', 'seeingFwhmEff': 'FWHMeff',
                             'fiveSigmaDepth': 'fivesigmadepth', 'slewTime': 'slewtime',
                             'slewDistance': 'slewdist', 'paraAngle': 'pa', 'rotTelPos': 'rotTelPos',
                             'rotSkyPos': 'rotSkyPos', 'moonRA': 'moonRA',
                             'moonDec': 'moonDec', 'moonAlt': 'moonAlt', 'moonAz': 'moonAz',
                             'moonDistance': 'moonDist', 'moonPhase': 'moonPhase',
                             'sunAlt': 'sunAlt', 'sunAz': 'sunAz', 'solarElong': 'solarElong', 'note':'note'}
        # Column(s) not bothering to remap:  'observationStartTime': None,
        self.inv_map = {v: k for k, v in self.convert_dict.items()}
        # angles to converts
        self.angles_rad2deg = ['fieldRA', 'fieldDec', 'altitude', 'azimuth', 'slewDistance',
                               'paraAngle', 'rotTelPos', 'rotSkyPos', 'moonRA', 'moonDec',
                               'moonAlt', 'moonAz', 'moonDistance', 'sunAlt', 'sunAz', 'solarElong',
                               'cummTelAz']
        # Put LMST into degrees too
        self.angles_hours2deg = ['observationStartLST']

    def obs2opsim(self, obs_array, filename=None, info=None, delete_past=False):
        """convert an array of observations into a pandas dataframe with Opsim schema
        """
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
            df[colname] = df[colname] * 360./24.

        if filename is not None:
            con = db.connect(filename)
            df.to_sql('observations', con, index=False)
            if info is not None:
                df = pd.DataFrame(info)
                df.to_sql('info', con)

    def opsim2obs(self, filename):
        """convert an opsim schema dataframe into an observation array.
        """

        con = db.connect(filename)
        df = pd.read_sql('select * from observations;', con)
        for key in self.angles_rad2deg:
            df[key] = np.radians(df[key])
        for key in self.angles_hours2deg:
            df[key] = df[key] * 24./360.

        df = df.rename(index=str, columns=self.convert_dict)

        blank = empty_observation()
        final_result = np.empty(df.shape[0], dtype=blank.dtype)
        # XXX-ugh, there has to be a better way.
        for i, key in enumerate(df.columns):
            if key in self.inv_map.keys():
                final_result[key] = df[key].values

        return final_result


def empty_observation():
    """Return a numpy array that could be a handy observation record

    XXX:  Should this really be "empty visit"? Should we have "visits" made
    up of multple "observations" to support multi-exposure time visits?

    XXX-Could add a bool flag for "observed". Then easy to track all proposed
    observations. Could also add an mjd_min, mjd_max for when an observation should be observed.
    That way we could drop things into the queue for DD fields.

    XXX--might be nice to add a generic "sched_note" str field, to record any metadata that
    would be useful to the scheduler once it's observed. and/or observationID.

    Returns
    -------
    numpy array


    The numpy fields have the following structure
    RA : float
       The Right Acension of the observation (center of the field) (Radians)
    dec : float
       Declination of the observation (Radians)
    mjd : float
       Modified Julian Date at the start of the observation (time shutter opens)
    exptime : float
       Total exposure time of the visit (seconds)
    filter : str
        The filter used. Should be one of u, g, r, i, z, y.
    rotSkyPos : float
        The rotation angle of the camera relative to the sky E of N (Radians)
    nexp : int
        Number of exposures in the visit.
    airmass : float
        Airmass at the center of the field
    FWHMeff : float
        The effective seeing FWHM at the center of the field. (arcsec)
    skybrightness : float
        The surface brightness of the sky background at the center of the
        field. (mag/sq arcsec)
    night : int
        The night number of the observation (days)
    flush_by_mjd : float
        If we hit this MJD, we should flush the queue and refill it.
    cummTelAz : float
        The cummulative telescope rotation in azimuth
    """

    names = ['ID', 'RA', 'dec', 'mjd', 'flush_by_mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp',
             'airmass', 'FWHM_500', 'FWHMeff', 'FWHM_geometric', 'skybrightness', 'night',
             'slewtime', 'visittime', 'slewdist', 'fivesigmadepth',
             'alt', 'az', 'pa', 'clouds', 'moonAlt', 'sunAlt', 'note',
             'field_id', 'survey_id', 'block_id',
             'lmst', 'rotTelPos', 'moonAz', 'sunAz', 'sunRA', 'sunDec', 'moonRA', 'moonDec',
             'moonDist', 'solarElong', 'moonPhase', 'cummTelAz']

    types = [int, float, float, float, float, float, 'U1', float, int,
             float, float, float, float, float, int,
             float, float, float, float,
             float, float, float, float, float, float, 'U40',
             int, int, int,
             float, float, float, float, float, float, float, float,
             float, float, float, float]
    result = np.zeros(1, dtype=list(zip(names, types)))
    return result


def scheduled_observation():
    """Make an array for pre-scheduling observations

    mjd_tol : float
        The tolerance on how early an observation can execute (days).

    """

    # Standard things from the usual observations
    names = ['ID', 'RA', 'dec', 'mjd', 'flush_by_mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp',
             'note']
    types = [int, float, float, float, float, float, 'U1', float, float, 'U40']
    names += ['mjd_tol', 'dist_tol', 'alt_min', 'alt_max', 'HA_max', 'HA_min', 'observed']
    types += [float, float, float, float, float, float, bool]
    result = np.zeros(1, dtype=list(zip(names, types)))
    return result


def read_fields():
    """Read in the Field coordinates

    Returns
    -------
    fields : `numpy.array`
        With RA and dec in radians.
    """
    query = 'select fieldId, fieldRA, fieldDEC from Field;'
    fd = FieldsDatabase()
    fields = np.array(list(fd.get_field_set(query)))
    # order by field ID
    fields = fields[fields[:,0].argsort()]

    names = ['RA', 'dec']
    types = [float, float]
    result = np.zeros(np.size(fields[:, 1]), dtype=list(zip(names, types)))
    result['RA'] = np.radians(fields[:, 1])
    result['dec'] = np.radians(fields[:, 2])

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
    ra, dec = _hpid2RaDec(nside, hpid)
    return _buildTree(ra, dec, leafsize, scale=scale)


class hp_in_lsst_fov(object):
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
        """
        if nside is None:
            nside = set_default_nside()

        self.tree = hp_kd_tree(nside=nside, scale=scale)
        self.radius = np.round(xyz_angular_radius(fov_radius)*scale).astype(int)
        self.scale = scale

    def __call__(self, ra, dec, **kwargs):
        """
        Parameters
        ----------
        ra : float
            RA in radians
        dec : float
            Dec in radians

        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """

        x, y, z = _xyz_from_ra_dec(np.max(ra), np.max(dec))
        x = np.round(x * self.scale).astype(int)
        y = np.round(y * self.scale).astype(int)
        z = np.round(z * self.scale).astype(int)

        indices = self.tree.query_ball_point((x, y, z), self.radius)
        return np.array(indices)


class hp_in_comcam_fov(object):
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
        self.inner_radius = xyz_angular_radius(side_length/2.)
        self.outter_radius = xyz_angular_radius(side_length/2.*np.sqrt(2.))
        # The positions of the raft corners, unrotated
        self.corners_x = np.array([-self.side_length/2., -self.side_length/2., self.side_length/2.,
                                  self.side_length/2.])
        self.corners_y = np.array([self.side_length/2., -self.side_length/2., -self.side_length/2.,
                                  self.side_length/2.])

    def __call__(self, ra, dec, rotSkyPos=0.):
        """
        Parameters
        ----------
        ra : float
            RA in radians
        dec : float
            Dec in radians
        rotSkyPos : float
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

        cos_rot = np.cos(rotSkyPos)
        sin_rot = np.sin(rotSkyPos)
        x_rotated = self.corners_x*cos_rot - self.corners_y*sin_rot
        y_rotated = self.corners_x*sin_rot + self.corners_y*cos_rot

        # Draw the square that we want to check if points are in.
        bbPath = mplPath.Path(np.array([[x_rotated[0], y_rotated[0]],
                                       [x_rotated[1], y_rotated[1]],
                                       [x_rotated[2], y_rotated[2]],
                                       [x_rotated[3], y_rotated[3]],
                                       [x_rotated[0], y_rotated[0]]]))

        ra_to_check, dec_to_check = _hpid2RaDec(self.nside, indices_to_check)

        # Project the indices to check to the tangent plane, see if they fall inside the polygon
        x, y = gnomonic_project_toxy(ra_to_check, dec_to_check, ra, dec)
        for i, xcheck in enumerate(x):
            # I wonder if I can do this all at once rather than a loop?
            if bbPath.contains_point((x[i], y[i])):
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

    names = ['Parameter', 'Value']
    dtypes = ['|U200', '|U200']
    result = np.zeros(observatory_info[:, 0].size + n_feature_entries,
                      dtype=list(zip(names, dtypes)))

    # Fill in info about the run
    result[0]['Parameter'] = 'Date, ymd'
    now = datetime.datetime.now()
    result[0]['Value'] = '%i, %i, %i' % (now.year, now.month, now.day)

    result[1]['Parameter'] = 'hostname'
    result[1]['Value'] = socket.gethostname()

    result[2]['Parameter'] = 'rubin_sim.__version__'
    result[2]['Value'] = rubin_sim.__version__

    result[3:]['Parameter'] = observatory_info[:, 0]
    result[3:]['Value'] = observatory_info[:, 1]

    return result


def inrange(inval, minimum=-1., maximum=1.):
    """
    Make sure values are within min/max
    """
    inval = np.array(inval)
    below = np.where(inval < minimum)
    inval[below] = minimum
    above = np.where(inval > maximum)
    inval[above] = maximum
    return inval


def warm_start(scheduler, observations, mjd_key='mjd'):
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
    result = result/season_length
    if floor:
        result = np.floor(result)
    if max_season is not None:
        over_indx = np.where(int_rounded(result) >= int_rounded(max_season))

    if modulo is not None:
        neg = np.where(int_rounded(result) < int_rounded(0))
        result = result % modulo
        result[neg] = -1
    if max_season is not None:
        result[over_indx] = -1
    if floor:
        result = result.astype(int)
    return result


def create_season_offset(nside, sun_RA_rad):
    """
    Make an offset map so seasons roll properly
    """
    hpindx = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2RaDec(nside, hpindx)
    offset = ra - sun_RA_rad + 2.*np.pi
    offset = offset % (np.pi*2)
    offset = offset * 365.25/(np.pi*2)
    offset = -offset - 365.25
    return offset


class TargetoO(object):
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
    """
    def __init__(self, tooid, footprint, mjd_start, duration):
        self.footprint = footprint
        self.duration = duration
        self.id = tooid
        self.mjd_start = mjd_start


class Sim_targetoO_server(object):
    """Wrapper to deliver a targetoO object at the right time
    """

    def __init__(self, targetoO_list):
        self.targetoO_list = targetoO_list
        self.mjd_starts = np.array([too.mjd_start for too in self.targetoO_list])
        durations = np.array([too.duration for too in self.targetoO_list])
        self.mjd_ends = self.mjd_starts + durations

    def __call__(self, mjd):
        in_range = np.where((mjd > self.mjd_starts) & (mjd < self.mjd_ends))[0]
        result = None
        if in_range.size > 0:
            result = [self.targetoO_list[i] for i in in_range]
        return result
