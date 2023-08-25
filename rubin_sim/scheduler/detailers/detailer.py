__all__ = (
    "BaseDetailer",
    "ZeroRotDetailer",
    "Comcam90rotDetailer",
    "Rottep2RotspDesiredDetailer",
    "CloseAltDetailer",
    "TakeAsPairsDetailer",
    "TwilightTripleDetailer",
    "SpiderRotDetailer",
    "FlushForSchedDetailer",
    "FilterNexp",
    "FixedSkyAngleDetailer",
)

import copy

import numpy as np

from rubin_sim.scheduler.utils import IntRounded
from rubin_sim.utils import _angular_separation, _approx_altaz2pa, _approx_ra_dec2_alt_az


class BaseDetailer:
    """
    A Detailer is an object that takes a list of proposed observations and adds "details" to them. The
    primary purpose is that the Markov Decision Process does an excelent job selecting RA,Dec,filter
    combinations, but we may want to add additional logic such as what to set the camera rotation angle
    to, or what to use for an exposure time. We could also modify the order of the proposed observations.
    For Deep Drilling Fields, a detailer could be useful for computing dither positions and modifying
    the exact RA,Dec positions.
    """

    def __init__(self, nside=32):
        """"""
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside

    def add_observations_array(self, observations_array, observations_hpid):
        """Like add_observation, but for loading a whole array of observations at a time"""

        for feature in self.survey_features:
            self.survey_features[feature].add_observations_array(observations_array, observations_hpid)

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : `np.array`
            An array with information about the input observation
        indx : `np.array`
            The indices of the healpix map that the observation overlaps with
        """
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)

    def __call__(self, observation_list, conditions):
        """
        Parameters
        ----------
        observation_list : `list` of observations
            The observations to detail.
        conditions : `rubin_sim.scheduler.conditions` object

        Returns
        -------
        List of observations.
        """

        return observation_list


class Rottep2RotspDesiredDetailer(BaseDetailer):
    """Convert all the rotTelPos values to rotSkyPos_desired"""

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)

        alt, az = _approx_ra_dec2_alt_az(
            obs_array["RA"],
            obs_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)

        rot_sky_pos_desired = (obs_array["rotTelPos"] - obs_pa) % (2.0 * np.pi)

        for obs, rotsp_d in zip(observation_list, rot_sky_pos_desired):
            obs["rotTelPos_backup"] = obs["rotTelPos"] + 0
            obs["rotTelPos"] = np.nan
            obs["rotSkyPos"] = np.nan
            obs["rotSkyPos_desired"] = rotsp_d

        return observation_list


class ZeroRotDetailer(BaseDetailer):
    """
    Detailer to set the camera rotation to be apporximately zero in rotTelPos.
    Because it can never be written too many times:
    rotSkyPos = rotTelPos - ParallacticAngle
    But, wait, what? Is it really the other way?
    """

    def __call__(self, observation_list, conditions):
        # XXX--should I convert the list into an array and get rid of this loop?
        for obs in observation_list:
            alt, az = _approx_ra_dec2_alt_az(
                obs["RA"],
                obs["dec"],
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
            )
            obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs["rotSkyPos"] = obs_pa

        return observation_list


class SpiderRotDetailer(BaseDetailer):
    """
    Set the camera rotation to +/- 45 degrees so diffraction spikes align along chip rows
    and columns
    """

    def __call__(self, observation_list, conditions):
        indx = int(conditions.night % 2)
        rot_tel_pos = np.radians([45.0, 315.0][indx])

        for obs in observation_list:
            obs["rotSkyPos"] = np.nan
            obs["rot_tel_pos"] = rot_tel_pos

        return observation_list


class Comcam90rotDetailer(BaseDetailer):
    """
    Detailer to set the camera rotation so rotSkyPos is 0, 90, 180, or 270 degrees. Whatever
    is closest to rotTelPos of zero.
    """

    def __call__(self, observation_list, conditions):
        favored_rot_sky_pos = np.radians([0.0, 90.0, 180.0, 270.0, 360.0]).reshape(5, 1)
        obs_array = np.concatenate(observation_list)
        alt, az = _approx_ra_dec2_alt_az(
            obs_array["RA"],
            obs_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        parallactic_angle = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
        # If we set rotSkyPos to parallactic angle, rotTelPos will be zero. So, find the
        # favored rotSkyPos that is closest to PA to keep rotTelPos as close as possible to zero.
        ang_diff = np.abs(parallactic_angle - favored_rot_sky_pos)
        min_indxs = np.argmin(ang_diff, axis=0)
        # can swap 360 and zero if needed?
        final_rot_sky_pos = favored_rot_sky_pos[min_indxs]
        # Set all the observations to the proper rotSkyPos
        for rsp, obs in zip(final_rot_sky_pos, observation_list):
            obs["rotSkyPos"] = rsp

        return observation_list


class FixedSkyAngleDetailer(BaseDetailer):
    """Detailer to force a specific sky angle.

    Parameters
    ----------
    sky_angle : `float`, optional
        Desired sky angle (default = 0, in degrees).
    """

    def __init__(self, sky_angle=0.0, nside=32):
        super().__init__(nside=nside)

        self.sky_angle = np.radians(sky_angle)

    def __call__(self, observation_list, conditions):
        for observation in observation_list:
            observation["rotSkyPos"] = self.sky_angle

        return observation_list


class CloseAltDetailer(BaseDetailer):
    """
    re-order a list of observations so that the closest in altitude to the current pointing is first.

    Parameters
    ----------
    alt_band : `float` (10)
        The altitude band to try and stay in (degrees)
    """

    def __init__(self, alt_band=10.0):
        super(CloseAltDetailer, self).__init__()
        self.alt_band = IntRounded(np.radians(alt_band))

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)
        alt, az = _approx_ra_dec2_alt_az(
            obs_array["RA"],
            obs_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        alt_diff = np.abs(alt - conditions.tel_alt)
        in_band = np.where(IntRounded(alt_diff) <= self.alt_band)[0]
        if in_band.size == 0:
            in_band = np.arange(alt.size)

        # Find the closest in angular distance of the points that are in band
        ang_dist = _angular_separation(az[in_band], alt[in_band], conditions.tel_az, conditions.tel_alt)
        if np.size(ang_dist) == 1:
            good = 0
        else:
            good = np.min(np.where(ang_dist == ang_dist.min())[0])
        indx = in_band[good]
        result = observation_list[indx:] + observation_list[:indx]
        return result


class FlushForSchedDetailer(BaseDetailer):
    """Update the flush-by MJD to be before any scheduled observations

    Parameters
    ----------
    tol : `float`
         How much before to flush (minutes)
    """

    def __init__(self, tol=2.5):
        super(FlushForSchedDetailer, self).__init__()
        self.tol = tol / 24.0 / 60.0  # To days

    def __call__(self, observation_list, conditions):
        if np.size(conditions.scheduled_observations) > 0:
            new_flush = np.min(conditions.scheduled_observations) - self.tol
            for obs in observation_list:
                if obs["flush_by_mjd"] > new_flush:
                    obs["flush_by_mjd"] = new_flush
        return observation_list


class FilterNexp(BaseDetailer):
    """Demand one filter always be taken as a certain number of exposures"""

    def __init__(self, filtername="u", nexp=1, exptime=None):
        super(FilterNexp, self).__init__()
        self.filtername = filtername
        self.nexp = nexp
        self.exptime = exptime

    def __call__(self, observation_list, conditions):
        for obs in observation_list:
            if obs["filter"] == self.filtername:
                obs["nexp"] = self.nexp
                if self.exptime is not None:
                    obs["exptime"] = self.exptime
        return observation_list


class TakeAsPairsDetailer(BaseDetailer):
    def __init__(self, filtername="r", exptime=None, nexp_dict=None):
        """"""
        super(TakeAsPairsDetailer, self).__init__()
        self.filtername = filtername
        self.exptime = exptime
        self.nexp_dict = nexp_dict

    def __call__(self, observation_list, conditions):
        paired = copy.deepcopy(observation_list)
        if self.exptime is not None:
            for obs in paired:
                obs["exptime"] = self.exptime
        for obs in paired:
            obs["filter"] = self.filtername
            if self.nexp_dict is not None:
                obs["nexp"] = self.nexp_dict[self.filtername]
        if conditions.current_filter == self.filtername:
            for obs in paired:
                obs["note"] = obs["note"][0] + ", a"
            for obs in observation_list:
                obs["note"] = obs["note"][0] + ", b"
            result = paired + observation_list
        else:
            for obs in paired:
                obs["note"] = obs["note"][0] + ", b"
            for obs in observation_list:
                obs["note"] = obs["note"][0] + ", a"
            result = observation_list + paired
        # XXX--maybe a temp debugging thing, label what part of sequence each observation is.
        for i, obs in enumerate(result):
            obs["survey_id"] = i
        return result


class TwilightTripleDetailer(BaseDetailer):
    def __init__(self, slew_estimate=5.0, n_repeat=3, update_note=True):
        super(TwilightTripleDetailer, self).__init__()
        self.slew_estimate = slew_estimate
        self.n_repeat = n_repeat
        self.update_note = update_note

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)

        # Estimate how much time is left in the twilgiht block
        potential_times = np.array(
            [
                conditions.sun_n18_setting - conditions.mjd,
                conditions.sun_n12_rising - conditions.mjd,
            ]
        )

        potential_times = np.min(potential_times[np.where(potential_times > 0)]) * 24.0 * 3600.0

        # How long will observations take?
        cumulative_slew = np.arange(obs_array.size) * self.slew_estimate
        cumulative_expt = np.cumsum(obs_array["exptime"])
        cumulative_time = cumulative_slew + cumulative_expt
        # If we are way over, truncate the list before doing the triple
        if np.max(cumulative_time) > potential_times:
            max_indx = np.where(cumulative_time / self.n_repeat <= potential_times)[0]
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
            sub_list = copy.deepcopy(observation_list)
            if self.update_note:
                for obs in sub_list:
                    obs["note"][0] += ", %i" % i
            out_obs.extend(sub_list)
        return out_obs
