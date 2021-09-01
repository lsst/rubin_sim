import numpy as np
from rubin_sim.scheduler.utils import (empty_observation, set_default_nside)
import rubin_sim.scheduler.features as features
from rubin_sim.scheduler.surveys import BaseSurvey
from rubin_sim.utils import _approx_RaDec2AltAz, _raDec2Hpid, _angularSeparation
import logging

log = logging.getLogger(__name__)

__all__ = ['Long_gap_survey']


class Long_gap_survey(BaseSurvey):
    """
    """
    def __init__(self, blob_survey, scripted_survey, gap_range=[2, 10]):
        self.blob_survey = blob_survey
        self.scripted_survey = scripted_survey
        self.night = -1
        self.gap_range = np.array(gap_range)/24.  # To days
        self.gap = 0.

    def add_observation(self, observation, **kwargs):
        self.blob_survey.add_observation(observation, **kwargs)
        self.scripted_survey.add_observation(observation, **kwargs)

    def _check_feasibility(self, conditions):
        f1 = self.blob_survey._check_feasibility(conditions)
        f2 = self.scripted_survey._check_feasibility(conditions)

        # If either one is able to go, we can observe
        result = (f1 | f2)
        return result

    def calc_reward_function(self, conditions):
        if conditions.night != self.night:
            # Clear out the scheduled observations
            self.scripted_survey.scheduled_obs = -1
            self.night = conditions.night

        r1 = self.blob_survey.calc_reward_function(conditions)
        r2 = self.scripted_survey(conditions)
        self.reward_checked = True
        return np.max([r1, r2])

    def generate_observations_rough(self, conditions):
        """
        """
        pass

    def generate_observations(self, conditions):

        o1 = self.blob_survey.generate_observations(conditions)
        if o1 is not None:
            # Set the script to have things
            obs_array = np.array(o1)
            obs_array = obs_array[np.where(obs_array['filter'] == obs_array['filter'][0])[0]]
            obs_array['mjd'] = conditions.mjd + self.gap
            self.scripted_survey.set_script(obs_array)
            observations = o1
        else:
            observations = self.scripted_survey(conditions)

        return observations
