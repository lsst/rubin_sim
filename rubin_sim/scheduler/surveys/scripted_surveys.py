import numpy as np
from rubin_sim.scheduler.utils import (empty_observation, set_default_nside)
import rubin_sim.scheduler.features as features
from rubin_sim.scheduler.surveys import BaseSurvey
from rubin_sim.utils import _approx_RaDec2AltAz, _raDec2Hpid, _angularSeparation
import logging

log = logging.getLogger(__name__)

__all__ = ['Scripted_survey', 'Pairs_survey_scripted']


class Scripted_survey(BaseSurvey):
    """
    Take a set of scheduled observations and serve them up.
    """
    def __init__(self, basis_functions, reward=1e6, ignore_obs='dummy',
                 nside=None):
        """
        """
        if nside is None:
            nside = set_default_nside()

        self.extra_features = {}
        self.nside = nside
        self.reward_val = reward
        self.reward = -np.inf
        super(Scripted_survey, self).__init__(basis_functions=basis_functions,
                                              ignore_obs=ignore_obs, nside=nside)

    def add_observation(self, observation, indx=None, **kwargs):
        """Check if observation matches a scripted observation
        """

        # From base class
        checks = [io not in str(observation['note']) for io in self.ignore_obs]
        if all(checks):
            for feature in self.extra_features:
                self.extra_features[feature].add_observation(observation, **kwargs)
            for bf in self.basis_functions:
                bf.add_observation(observation, **kwargs)
            for detailer in self.detailers:
                detailer.add_observation(observation, **kwargs)
            self.reward_checked = False

            # was it taken in the right time window, and hasn't already been marked as observed.
            time_matches = np.where((observation['mjd'] > self.mjd_start) &
                                    (observation['mjd'] < self.obs_wanted['flush_by_mjd']) &
                                    (~self.obs_wanted['observed']) &
                                    (observation['note'] == self.obs_wanted['note']))[0]
            for match in time_matches:
                distances = _angularSeparation(self.obs_wanted[match]['RA'],
                                               self.obs_wanted[match]['dec'],
                                               observation['RA'], observation['dec'])
                if (distances < self.obs_wanted[match]['dist_tol']) & \
                   (self.obs_wanted[match]['filter'] == observation['filter']):
                    # Log it as observed.
                    self.obs_wanted['observed'][match] = True
                    self.scheduled_obs = self.obs_wanted['mjd'][~self.obs_wanted['observed']]
                    break

    def calc_reward_function(self, conditions):
        """If there is an observation ready to go, execute it, otherwise, -inf
        """
        observation = self._check_list(conditions)
        if observation is None:
            self.reward = -np.inf
        else:
            self.reward = self.reward_val
        return self.reward

    def _slice2obs(self, obs_row):
        """take a slice and return a full observation object
        """
        observation = empty_observation()
        for key in ['RA', 'dec', 'filter', 'exptime', 'nexp',
                    'note', 'rotSkyPos', 'flush_by_mjd']:
            observation[key] = obs_row[key]
        return observation

    def _check_alts_HA(self, observation, conditions):
        """Given scheduled observations, check which ones can be done in current conditions.

        Parameters
        ----------
        observation : np.array
            An array of scheduled observations. Probably generated with rubin_sim.scheduler.utils.scheduled_observation
        """
        # Just do a fast ra,dec to alt,az conversion.
        alt, az = _approx_RaDec2AltAz(observation['RA'], observation['dec'],
                                      conditions.site.latitude_rad, None,
                                      conditions.mjd,
                                      lmst=conditions.lmst)
        HA = conditions.lmst - observation['RA']*12./np.pi
        HA[np.where(HA > 24)] -= 24
        HA[np.where(HA < 0)] += 24
        in_range = np.where((alt < observation['alt_max']) & (alt > observation['alt_min']) &
                            ((HA > observation['HA_max']) | (HA < observation['HA_min'])))[0]
        return in_range

    def _check_list(self, conditions):
        """Check to see if the current mjd is good
        """

        # Scheduled observations that are in the right time window and have not been executed
        in_time_window = np.where((self.mjd_start < conditions.mjd) &
                                  (self.obs_wanted['flush_by_mjd'] > conditions.mjd) &
                                  (~self.obs_wanted['observed']))[0]

        if np.size(in_time_window) > 0:
            pass_checks = self._check_alts_HA(self.obs_wanted[in_time_window], conditions)
            matches = in_time_window[pass_checks]
        else:
            matches = []

        if np.size(matches) > 0:
            # XXX--could make this a list and just send out all the things that currently match
            # rather than one at a time
            observation = self._slice2obs(self.obs_wanted[matches[0]])
        else:
            observation = None
        return observation

    def set_script(self, obs_wanted):
        """
        Parameters
        ----------
        obs_wanted : np.array
            The observations that should be executed. Needs to have columns with dtype names:
            Should be from lsst.sim.scheduler.utils.scheduled_observation
        mjds : np.array
            The MJDs for the observaitons, should be same length as obs_list
        mjd_tol : float (15.)
            The tolerance to consider an observation as still good to observe (min)
        """

        self.obs_wanted = obs_wanted

        self.obs_wanted.sort(order='mjd')
        self.mjd_start = self.obs_wanted['mjd'] - self.obs_wanted['mjd_tol']
        # Here is the atribute that core scheduler checks to broadcast scheduled observations
        # in the conditions object.
        self.scheduled_obs = self.obs_wanted['mjd']

    def generate_observations_rough(self, conditions):
        observation = self._check_list(conditions)
        return [observation]


class Pairs_survey_scripted(Scripted_survey):
    """Check if incoming observations will need a pair in 30 minutes. If so, add to the queue
    """
    def __init__(self, basis_functions, filt_to_pair='griz',
                 dt=40., ttol=10., reward_val=101., note='scripted', ignore_obs='ack',
                 min_alt=30., max_alt=85., lat=-30.2444, moon_distance=30., max_slew_to_pair=15.,
                 nside=None):
        """
        Parameters
        ----------
        filt_to_pair : str (griz)
            Which filters to try and get pairs of
        dt : float (40.)
            The ideal gap between pairs (minutes)
        ttol : float (10.)
            The time tolerance when gathering a pair (minutes)
        """
        if nside is None:
            nside = set_default_nside()

        super(Pairs_survey_scripted, self).__init__(basis_functions=basis_functions,
                                                    ignore_obs=ignore_obs, min_alt=min_alt,
                                                    max_alt=max_alt, nside=nside)

        self.lat = np.radians(lat)
        self.note = note
        self.ttol = ttol/60./24.
        self.dt = dt/60./24.  # To days
        self.max_slew_to_pair = max_slew_to_pair  # in seconds
        self._moon_distance = np.radians(moon_distance)

        self.extra_features = {}
        self.extra_features['Pair_map'] = features.Pair_in_night(filtername=filt_to_pair)

        self.reward_val = reward_val
        self.filt_to_pair = filt_to_pair
        # list to hold observations
        self.observing_queue = []
        # make ignore_obs a list
        if type(self.ignore_obs) is str:
            self.ignore_obs = [self.ignore_obs]

    def add_observation(self, observation, indx=None, **kwargs):
        """Add an observed observation
        """
        # self.ignore_obs not in str(observation['note'])
        to_ignore = np.any([ignore in str(observation['note']) for ignore in self.ignore_obs])
        log.debug('[Pairs.add_observation]: %s: %s: %s', to_ignore, str(observation['note']), self.ignore_obs)
        log.debug('[Pairs.add_observation.queue]: %s', self.observing_queue)
        if not to_ignore:
            # Update my extra features:
            for feature in self.extra_features:
                if hasattr(self.extra_features[feature], 'add_observation'):
                    self.extra_features[feature].add_observation(observation, indx=indx)
            self.reward_checked = False

            # Check if this observation needs a pair
            # XXX--only supporting single pairs now. Just start up another scripted survey
            # to grab triples, etc? Or add two observations to queue at a time?
            # keys_to_copy = ['RA', 'dec', 'filter', 'exptime', 'nexp']
            if ((observation['filter'][0] in self.filt_to_pair) and
                    (np.max(self.extra_features['Pair_map'].feature[indx]) < 1)):
                obs_to_queue = empty_observation()
                for key in observation.dtype.names:
                    obs_to_queue[key] = observation[key]
                # Fill in the ideal time we would like this observed
                log.debug('Observation MJD: %.4f (dt=%.4f)', obs_to_queue['mjd'], self.dt)
                obs_to_queue['mjd'] += self.dt
                self.observing_queue.append(obs_to_queue)
        log.debug('[Pairs.add_observation.queue.size]: %i', len(self.observing_queue))
        for obs in self.observing_queue:
            log.debug('[Pairs.add_observation.queue]: %s', obs)

    def _purge_queue(self, conditions):
        """Remove any pair where it's too late to observe it
        """
        # Assuming self.observing_queue is sorted by MJD.
        if len(self.observing_queue) > 0:
            stale = True
            in_window = np.abs(self.observing_queue[0]['mjd']-conditions.mjd) < self.ttol
            log.debug('Purging queue')
            while stale:
                # If the next observation in queue is past the window, drop it
                if (self.observing_queue[0]['mjd'] < conditions.mjd) & (~in_window):
                    log.debug('Past the window: obs_mjd=%.4f (current_mjd=%.4f)',
                              self.observing_queue[0]['mjd'],
                              conditions.mjd)
                    del self.observing_queue[0]
                # If we are in the window, but masked, drop it
                elif (in_window) & (~self._check_mask(self.observing_queue[0], conditions)):
                    log.debug('Masked')
                    del self.observing_queue[0]
                # If in time window, but in alt exclusion zone
                elif (in_window) & (~self._check_alts(self.observing_queue[0], conditions)):
                    log.debug('in alt exclusion zone')
                    del self.observing_queue[0]
                else:
                    stale = False
                # If we have deleted everything, break out of where
                if len(self.observing_queue) == 0:
                    stale = False

    def _check_alts(self, observation, conditions):
        result = False
        # Just do a fast ra,dec to alt,az conversion. Can use LMST from a feature.

        alt, az = _approx_RaDec2AltAz(observation['RA'], observation['dec'],
                                      self.lat, None,
                                      conditions.mjd,
                                      lmst=conditions.lmst)
        in_range = np.where((alt < self.max_alt) & (alt > self.min_alt))[0]
        if np.size(in_range) > 0:
            result = True
        return result

    def _check_mask(self, observation, conditions):
        """Check that the proposed observation is not currently masked for some reason on the sky map.
        True if the observation is good to observe
        False if the proposed observation is masked
        """

        hpid = np.max(_raDec2Hpid(self.nside, observation['RA'], observation['dec']))
        skyval = conditions.M5Depth[observation['filter'][0]][hpid]

        if skyval > 0:
            return True
        else:
            return False

    def calc_reward_function(self, conditions):
        self._purge_queue(conditions)
        result = -np.inf
        self.reward = result
        log.debug('Pair - calc_reward_func')
        for indx in range(len(self.observing_queue)):

            check = self._check_observation(self.observing_queue[indx], conditions)
            log.debug('%s: %s', check, self.observing_queue[indx])
            if check[0]:
                result = self.reward_val
                self.reward = self.reward_val
                break
            elif not check[1]:
                break

        self.reward_checked = True
        return result

    def _check_observation(self, observation, conditions):

        delta_t = observation['mjd'] - conditions.mjd
        log.debug('Check_observation: obs_mjd=%.4f (current_mjd=%.4f, delta=%.4f, tol=%.4f)',
                  observation['mjd'],
                  conditions.mjd,
                  delta_t,
                  self.ttol)
        obs_hp = _raDec2Hpid(self.nside, observation['RA'], observation['dec'])
        slewtime = conditions.slewtime[obs_hp[0]]
        in_slew_window = slewtime <= self.max_slew_to_pair or delta_t < 0.
        in_time_window = np.abs(delta_t) < self.ttol

        if conditions.current_filter is None:
            infilt = True
        else:
            infilt = conditions.current_filter in self.filt_to_pair

        is_observable = self._check_mask(observation, conditions)
        valid = in_time_window & infilt & in_slew_window & is_observable
        log.debug('Pair - observation: %s ' % observation)
        log.debug('Pair - check[%s]: in_time_window[%s] infilt[%s] in_slew_window[%s] is_observable[%s]' %
                  (valid, in_time_window, infilt, in_slew_window, is_observable))

        return (valid,
                in_time_window,
                infilt,
                in_slew_window,
                is_observable)

    def generate_observations(self, conditions):
        # Toss anything in the queue that is too old to pair up:
        self._purge_queue(conditions)
        # Check for something I want a pair of
        result = []
        # if len(self.observing_queue) > 0:
        log.debug('Pair - call')
        for indx in range(len(self.observing_queue)):

            check = self._check_observation(self.observing_queue[indx], conditions)

            if check[0]:
                result = self.observing_queue.pop(indx)
                result['note'] = 'pair(%s)' % self.note
                # Make sure we don't change filter if we don't have to.
                if conditions.current_filter is not None:
                    result['filter'] = conditions.current_filter
                # Make sure it is observable!
                # if self._check_mask(result):
                result = [result]
                break
            elif not check[1]:
                # If this is not in time window and queue is chronological, none will be...
                break

        return result
