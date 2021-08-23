from rubin_sim.utils import _raDec2Hpid, _approx_RaDec2AltAz, _angularSeparation, _approx_altaz2pa
import numpy as np
from rubin_sim.scheduler.utils import int_rounded
import copy

__all__ = ["Base_detailer", "Zero_rot_detailer", "Comcam_90rot_detailer", "Close_alt_detailer",
           "Take_as_pairs_detailer", "Twilight_triple_detailer", "Spider_rot_detailer",
           "Flush_for_sched_detailer", 'Filter_nexp']


class Base_detailer(object):
    """
    A Detailer is an object that takes a list of proposed observations and adds "details" to them. The
    primary purpose is that the Markov Decision Process does an excelent job selecting RA,Dec,filter
    combinations, but we may want to add additional logic such as what to set the camera rotation angle
    to, or what to use for an exposure time. We could also modify the order of the proposed observations.
    For Deep Drilling Fields, a detailer could be useful for computing dither positions and modifying
    the exact RA,Dec positions.
    """

    def __init__(self, nside=32):
        """
        """
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : np.array
            An array with information about the input observation
        indx : np.array
            The indices of the healpix map that the observation overlaps with
        """
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)

    def __call__(self, observation_list, conditions):
        """
        Parameters
        ----------
        observation_list : list of observations
            The observations to detail.
        conditions : rubin_sim.scheduler.conditions object

        Returns
        -------
        List of observations.
        """

        return observation_list


class Zero_rot_detailer(Base_detailer):
    """
    Detailer to set the camera rotation to be apporximately zero in rotTelPos.
    Because it can never be written too many times:
    rotSkyPos = rotTelPos - ParallacticAngle
    But, wait, what? Is it really the other way?
    """

    def __call__(self, observation_list, conditions):

        # XXX--should I convert the list into an array and get rid of this loop?
        for obs in observation_list:
            alt, az = _approx_RaDec2AltAz(obs['RA'], obs['dec'], conditions.site.latitude_rad,
                                          conditions.site.longitude_rad, conditions.mjd)
            obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs['rotSkyPos'] = obs_pa

        return observation_list


class Spider_rot_detailer(Base_detailer):
    """
    Set the camera rotation to +/- 45 degrees so diffraction spikes align along chip rows
    and columns
    """

    def __call__(self, observation_list, conditions):
        indx = int(conditions.night % 2)
        rotTelPos = np.radians([45., 315.][indx])

        for obs in observation_list:
            obs['rotSkyPos'] = np.nan
            obs['rotTelPos'] = rotTelPos

        return observation_list


class Comcam_90rot_detailer(Base_detailer):
    """
    Detailer to set the camera rotation so rotSkyPos is 0, 90, 180, or 270 degrees. Whatever
    is closest to rotTelPos of zero.
    """

    def __call__(self, observation_list, conditions):
        favored_rotSkyPos = np.radians([0., 90., 180., 270., 360.]).reshape(5, 1)
        obs_array =np.concatenate(observation_list)
        alt, az = _approx_RaDec2AltAz(obs_array['RA'], obs_array['dec'], conditions.site.latitude_rad,
                                      conditions.site.longitude_rad, conditions.mjd)
        parallactic_angle = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
        # If we set rotSkyPos to parallactic angle, rotTelPos will be zero. So, find the
        # favored rotSkyPos that is closest to PA to keep rotTelPos as close as possible to zero.
        ang_diff = np.abs(parallactic_angle - favored_rotSkyPos)
        min_indxs = np.argmin(ang_diff, axis=0)
        # can swap 360 and zero if needed?
        final_rotSkyPos = favored_rotSkyPos[min_indxs]
        # Set all the observations to the proper rotSkyPos
        for rsp, obs in zip(final_rotSkyPos, observation_list):
            obs['rotSkyPos'] = rsp

        return observation_list


class Close_alt_detailer(Base_detailer):
    """
    re-order a list of observations so that the closest in altitude to the current pointing is first.

    Parameters
    ----------
    alt_band : float (10)
        The altitude band to try and stay in (degrees)
    """
    def __init__(self, alt_band=10.):
        super(Close_alt_detailer, self).__init__()
        self.alt_band = int_rounded(np.radians(alt_band))

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)
        alt, az = _approx_RaDec2AltAz(obs_array['RA'], obs_array['dec'], conditions.site.latitude_rad,
                                      conditions.site.longitude_rad, conditions.mjd)
        alt_diff = np.abs(alt - conditions.telAlt)
        in_band = np.where(int_rounded(alt_diff) <= self.alt_band)[0]
        if in_band.size == 0:
            in_band = np.arange(alt.size)

        # Find the closest in angular distance of the points that are in band
        ang_dist = _angularSeparation(az[in_band], alt[in_band], conditions.telAz, conditions.telAlt)
        good = np.min(np.where(ang_dist == ang_dist.min())[0])
        indx = in_band[good]
        result = observation_list[indx:] + observation_list[:indx]
        return result


class Flush_for_sched_detailer(Base_detailer):
    """Update the flush-by MJD to be before any scheduled observations

    Parameters
    ----------
    tol : float
         How much before to flush (minutes)
    """
    def __init__(self, tol=2.5):
        super(Flush_for_sched_detailer, self).__init__()
        self.tol = tol/24./60.  # To days

    def __call__(self, observation_list, conditions):
        if np.size(conditions.scheduled_observations) > 0:
            new_flush = np.min(conditions.scheduled_observations) - self.tol
            for obs in observation_list:
                if obs['flush_by_mjd'] > new_flush:
                    obs['flush_by_mjd'] = new_flush
        return observation_list


class Filter_nexp(Base_detailer):
    """Demand one filter always be taken as a certain number of exposures
    """
    def __init__(self, filtername='u', nexp=1, exptime=None):
        super(Filter_nexp, self).__init__()
        self.filtername = filtername
        self.nexp = nexp
        self.exptime = exptime

    def __call__(self, observation_list, conditions):
        for obs in observation_list:
            if obs['filter'] == self.filtername:
                obs['nexp'] = self.nexp
                if self.exptime is not None:
                    obs['exptime'] = self.exptime
        return observation_list


class Take_as_pairs_detailer(Base_detailer):
    def __init__(self, filtername='r', exptime=None, nexp_dict=None):
        """
        """
        super(Take_as_pairs_detailer, self).__init__()
        self.filtername = filtername
        self.exptime = exptime
        self.nexp_dict = nexp_dict

    def __call__(self, observation_list, conditions):
        paired = copy.deepcopy(observation_list)
        if self.exptime is not None:
            for obs in paired:
                obs['exptime'] = self.exptime
        for obs in paired:
            obs['filter'] = self.filtername
            if self.nexp_dict is not None:
                obs['nexp'] = self.nexp_dict[self.filtername]
        if conditions.current_filter == self.filtername:
            for obs in paired:
                obs['note'] = obs['note'][0] + ', a'
            for obs in observation_list:
                obs['note'] = obs['note'][0] + ', b'
            result = paired + observation_list
        else:
            for obs in paired:
                obs['note'] = obs['note'][0] + ', b'
            for obs in observation_list:
                obs['note'] = obs['note'][0] + ', a'
            result = observation_list + paired
        # XXX--maybe a temp debugging thing, label what part of sequence each observation is.
        for i, obs in enumerate(result):
            obs['survey_id'] = i
        return result


class Twilight_triple_detailer(Base_detailer):
    def __init__(self, slew_estimate=5.0, n_repeat=3):
        super(Twilight_triple_detailer, self).__init__()
        self.slew_estimate = slew_estimate
        self.n_repeat = n_repeat

    def __call__(self, observation_list, conditions):

        obs_array = np.concatenate(observation_list)

        # Estimate how much time is left in the twilgiht block
        potential_times = np.array([conditions.sun_n18_setting - conditions.mjd,
                                   conditions.sun_n12_rising - conditions.mjd])

        potential_times = np.min(potential_times[np.where(potential_times > 0)]) * 24.*3600.

        # How long will observations take?
        cumulative_slew = np.arange(obs_array.size) * self.slew_estimate
        cumulative_expt = np.cumsum(obs_array['exptime'])
        cumulative_time = cumulative_slew + cumulative_expt
        # If we are way over, truncate the list before doing the triple
        if np.max(cumulative_time) > potential_times:
            max_indx = np.where(cumulative_time/self.n_repeat <= potential_times)[0]
            if np.size(max_indx) == 0:
                # Very bad magic number fudge
                max_indx = 3
            else:
                max_indx = np.max(max_indx)
                if max_indx == 0:
                    max_indx += 1
            observation_list = observation_list[0:max_indx]

        # Repeat the observations n times
        out_obs = []
        for i in range(self.n_repeat):
            out_obs.extend(copy.deepcopy(observation_list))

        return out_obs
