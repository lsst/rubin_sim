import numpy as np
from rubin_sim.scheduler.utils import (empty_observation, set_default_nside, scheduled_observation)
import rubin_sim.scheduler.features as features
from rubin_sim.scheduler.surveys import BaseSurvey
from rubin_sim.utils import _approx_RaDec2AltAz, _raDec2Hpid, _angularSeparation
import logging

log = logging.getLogger(__name__)

__all__ = ['Long_gap_survey']


class Long_gap_survey(BaseSurvey):
    """
    Parameters
    ----------
    blob_survey : xxx
        A survey object that we will want to take repeat measurments of sometime later in the evening
    scripted_survey : xxx
        A scripted survey object that will hold the 
    gap range : list of 2 floats
        The desired gap range (hours)
    long_name : str
        The string to put in the observation 'note' for the scripted observations
    scripted_tol : float
        The tolerance for when scripted observations can execute (hours)
    """
    def __init__(self, blob_survey, scripted_survey, gap_range=[2, 10], long_name='long',
                 scripted_tol=2., alt_min=20, alt_max=85., HA_min=-12, HA_max=12., flush_time=2.,
                 dist_tol=1., block_length=33.):
        self.blob_survey = blob_survey
        self.scripted_survey = scripted_survey
        self.night = -1
        self.gap_range = np.array(gap_range)/24.  # To days
        self.gap = 0.
        self.long_name = long_name
        self.scripted_tol = scripted_tol/24.  # To days
        self.alt_min = np.radians(alt_min)
        self.alt_max = np.radians(alt_max)
        self.HA_min = HA_min
        self.HA_max = HA_max
        self.flush_time = flush_time/24.
        self.dist_tol = np.radians(dist_tol)
        self.block_length = block_length/60/24.

    def add_observation(self, observation, **kwargs):
        self.blob_survey.add_observation(observation, **kwargs)
        self.scripted_survey.add_observation(observation, **kwargs)

    def _check_feasibility(self, conditions):
        f1 = self.blob_survey._check_feasibility(conditions)
        f2 = self.scripted_survey._check_feasibility(conditions)

        # If either one is able to go, we can observe
        result = (f1 | f2)
        return result

    def get_scheduled_obs(self):
        return self.scripted_survey.get_scheduled_obs()

    def calc_reward_function(self, conditions):
        if conditions.night != self.night:
            # Clear out the scheduled observations
            self.scripted_survey.clear_script()
            self.night = conditions.night + 0
            self.gap = np.random.uniform(self.gap_range.min(), self.gap_range.max())
            time_remaining = conditions.sun_n18_rising - conditions.mjd
            if self.gap > time_remaining:
                self.gap = time_remaining - self.block_length
            # XXX-need to reach into the blob and set what the gap is I guess

        self.r1 = self.blob_survey.calc_reward_function(conditions)
        self.r2 = self.scripted_survey.calc_reward_function(conditions)
        self.reward_checked = True
        return np.nanmax([np.nanmax(self.r1), np.nanmax(self.r2)])

    def generate_observations_rough(self, conditions):
        """
        """
        pass

    def generate_observations(self, conditions):

        if not self.reward_checked:
            self.reward = self.calc_reward_function(conditions)

        # Check for any pre-scheduled
        if self.r2 > -np.inf:
            observations = self.scripted_survey.generate_observations(conditions)
        else:
            observations = None

        # Check if it's a good time for a blob
        if observations is None:
            o1 = self.blob_survey.generate_observations(conditions)
            if np.size(o1) > 0:
                # Set the script to have things
                obs_array = np.concatenate(o1)
                # Reverse the order to try and get even more spread out gap times
                obs_array = obs_array[::-1]
                obs_array = obs_array[np.where(obs_array['filter'] == obs_array['filter'][0])[0]]
                obs_array['mjd'] = conditions.mjd + self.gap
                obs_array['note'] = self.long_name
                sched_array = scheduled_observation(n=obs_array.size)
                for dt in np.intersect1d(obs_array.dtype.names, sched_array.dtype.names):
                    sched_array[dt] = obs_array[dt]
                sched_array['mjd_tol'] = self.scripted_tol
                sched_array['alt_min'] = self.alt_min
                sched_array['alt_max'] = self.alt_max
                sched_array['HA_min'] = self.HA_min
                sched_array['HA_max'] = self.HA_max
                sched_array['flush_by_mjd'] = obs_array['mjd'] + self.flush_time
                sched_array['dist_tol'] = self.dist_tol

                self.scripted_survey.set_script(sched_array)
                observations = o1

        return observations
