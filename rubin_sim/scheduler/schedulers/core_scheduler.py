from __future__ import absolute_import
from builtins import object
import numpy as np
import healpy as hp
from rubin_sim.utils import _hpid2RaDec
from rubin_sim.scheduler.utils import hp_in_lsst_fov, set_default_nside, hp_in_comcam_fov, int_rounded
from rubin_sim.utils import _approx_RaDec2AltAz, _approx_altaz2pa
import logging


__all__ = ['Core_scheduler']


class Core_scheduler(object):
    """Core scheduler that takes completed observations and observatory status and requests observations

    Parameters
    ----------
    surveys : list (or list of lists) of rubin_sim.scheduler.survey objects
        A list of surveys to consider. If multiple surveys return the same highest
        reward value, the survey at the earliest position in the list will be selected.
        Can also be a list of lists to make heirarchical priorities.
    nside : int
        A HEALpix nside value.
    camera : str ('LSST')
        Which camera to use for computing overlapping HEALpixels for an observation.
        Can be 'LSST' or 'comcam'
    conditions : a rubin_sim.scheduler.features.Conditions object (None)
        An object that hold the current conditions and derived values (e.g., 5-sigma depth). Will
        generate a default if set to None.
    """

    def __init__(self, surveys, nside=None, camera='LSST', rotator_limits=[85., 275.], log=None):
        """
        Parameters
        ----------
        surveys : list (or list of lists) of rubin_sim.scheduler.survey objects
            A list of surveys to consider. If multiple surveys return the same highest
            reward value, the survey at the earliest position in the list will be selected.
            Can also be a list of lists to make heirarchical priorities.
        nside : int
            A HEALpix nside value.
        camera : str ('LSST')
            Which camera to use for computing overlapping HEALpixels for an observation.
            Can be 'LSST' or 'comcam'
        rotator_limits : sequence of floats
        """
        if nside is None:
            nside = set_default_nside()

        if log is None:
            self.log = logging.getLogger(type(self).__name__)
        else:
            self.log = log.getChild(type(self).__name__)

        # initialize a queue of observations to request
        self.queue = []
        # The indices of self.survey_lists that provided the last addition(s) to the queue
        self.survey_index = [None, None]

        # If we have a list of survey objects, convert to list-of-lists
        if isinstance(surveys[0], list):
            self.survey_lists = surveys
        else:
            self.survey_lists = [surveys]
        self.nside = nside
        hpid = np.arange(hp.nside2npix(nside))
        self.ra_grid_rad, self.dec_grid_rad = _hpid2RaDec(nside, hpid)
        # Should just make camera a class that takes a pointing and returns healpix indices
        if camera == 'LSST':
            self.pointing2hpindx = hp_in_lsst_fov(nside=nside)
        elif camera == 'comcam':
            self.pointing2hpindx = hp_in_comcam_fov(nside=nside)
        else:
            raise ValueError('camera %s not implamented' % camera)

        # keep track of how many observations get flushed from the queue
        self.flushed = 0
        self.rotator_limits = np.sort(np.radians(rotator_limits))

    def flush_queue(self):
        """"
        Like it sounds, clear any currently queued desired observations.
        """
        self.queue = []
        self.survey_index = [None, None]

    def add_observation(self, observation):
        """
        Record a completed observation and update features accordingly.

        Parameters
        ----------
        observation : dict-like
            An object that contains the relevant information about a
            completed observation (e.g., mjd, ra, dec, filter, rotation angle, etc)
        """

        # Find the healpixel centers that are included in an observation
        indx = self.pointing2hpindx(observation['RA'], observation['dec'],
                                    rotSkyPos=observation['rotSkyPos'])
        for surveys in self.survey_lists:
            for survey in surveys:
                survey.add_observation(observation, indx=indx)

    def update_conditions(self, conditions_in):
        """
        Parameters
        ----------
        conditions : dict-like
            The current conditions of the telescope (pointing position, loaded filters, cloud-mask, etc)
        """
        # Add the current queue and scheduled queue to the conditions
        self.conditions = conditions_in
        # put the local queue in the conditions
        self.conditions.queue = self.queue

        # Check if any surveys have upcomming scheduled observations. Note that we are accumulating
        # all of the possible scheduled observations, so it's up to the user to make sure things don't
        # collide. The ideal implementation would be to have all the scheduled observations in a
        # single survey objects, presumably at the highest tier of priority.

        all_scheduled = []
        for sl in self.survey_lists:
            for sur in sl:
                scheduled = sur.get_scheduled_obs()
                if scheduled is not None:
                    all_scheduled.append(scheduled)
        if len(all_scheduled) == 0:
            self.conditions.scheduled_observations = []
        else:
            all_scheduled = np.sort(np.array(all_scheduled).ravel())
            # In case the surveys have not been removing executed observations
            all_scheduled = all_scheduled[np.where(all_scheduled >= self.conditions.mjd)]
            self.conditions.scheduled_observations = all_scheduled

    def _check_queue_mjd_only(self, mjd):
        """
        Check if there are things in the queue that can be executed using only MJD and not full conditions.
        This is primarly used by sim_runner to reduce calls calculating updated conditions when they are not
        needed.
        """
        result = False
        if len(self.queue) > 0:
            if (int_rounded(mjd) < int_rounded(self.queue[0]['flush_by_mjd'])) | (self.queue[0]['flush_by_mjd'] == 0):
                result = True
        return result

    def request_observation(self, mjd=None):
        """
        Ask the scheduler what it wants to observe next

        Parameters
        ----------
        mjd : float (None)
            The Modified Julian Date. If None, it uses the MJD from the conditions from the
            last conditions update.

        Returns
        -------
        observation object (ra,dec,filter,rotangle)
        Returns None if the queue fails to fill
        """
        if mjd is None:
            mjd = self.conditions.mjd
        if len(self.queue) == 0:
            self._fill_queue()

        if len(self.queue) == 0:
            return None
        else:
            # If the queue has gone stale, flush and refill. Zero means no flush_by was set.
            if (int_rounded(mjd) > int_rounded(self.queue[0]['flush_by_mjd'])) & (self.queue[0]['flush_by_mjd'] != 0):
                self.flushed += len(self.queue)
                self.flush_queue()
                self._fill_queue()
            if len(self.queue) == 0:
                return None
            observation = self.queue.pop(0)
            # If we are limiting the camera rotator
            if self.rotator_limits is not None:
                alt, az = _approx_RaDec2AltAz(observation['RA'], observation['dec'], self.conditions.site.latitude_rad,
                                              self.conditions.site.longitude_rad, mjd)
                obs_pa = _approx_altaz2pa(alt, az, self.conditions.site.latitude_rad)
                rotTelPos_expected = (obs_pa - observation['rotSkyPos']) % (2.*np.pi)
                if (int_rounded(rotTelPos_expected) > int_rounded(self.rotator_limits[0])) & (int_rounded(rotTelPos_expected) < int_rounded(self.rotator_limits[1])):
                    diff = np.abs(self.rotator_limits - rotTelPos_expected)
                    limit_indx = np.min(np.where(diff == np.min(diff))[0])
                    observation['rotSkyPos'] = (obs_pa - self.rotator_limits[limit_indx]) % (2.*np.pi)
            return observation

    def _fill_queue(self):
        """
        Compute reward function for each survey and fill the observing queue with the
        observations from the highest reward survey.
        """

        rewards = None
        for ns, surveys in enumerate(self.survey_lists):
            rewards = np.zeros(len(surveys))
            for i, survey in enumerate(surveys):
                rewards[i] = np.nanmax(survey.calc_reward_function(self.conditions))
            # If we have a good reward, break out of the loop
            if np.nanmax(rewards) > -np.inf:
                self.survey_index[0] = ns
                break
        if (np.nanmax(rewards) == -np.inf) | (np.isnan(np.nanmax(rewards))):
            self.flush_queue()
        else:
            to_fix = np.where(np.isnan(rewards) == True)
            rewards[to_fix] = -np.inf
            # Take a min here, so the surveys will be executed in the order they are
            # entered if there is a tie.
            self.survey_index[1] = np.min(np.where(rewards == np.nanmax(rewards)))
            # Survey return list of observations
            result = self.survey_lists[self.survey_index[0]][self.survey_index[1]].generate_observations(self.conditions)

            self.queue = result

        if len(self.queue) == 0:
            self.log.warning('Failed to fill queue')
