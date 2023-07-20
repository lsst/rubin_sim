__all__ = ("TooMaster", "TooSurvey")

import copy

import healpy as hp
import numpy as np

from rubin_sim.scheduler.surveys import BaseSurvey, BlobSurvey


class TooMaster(BaseSurvey):
    """
    A target of opportunity class. Every time a new ToO comes in, it will spawn a new sub-survey.

    Parameters
    ----------
    example__to_o_survey : rubin_sim.scheduler.surveys.ToO_survey object
        A survey object that will be coppied and have a new target map set
        for each incoming ToO.
    """

    def __init__(self, example__to_o_survey):
        self.example__to_o_survey = example__to_o_survey
        self.surveys = []
        self.highest_reward = -np.inf
        self.scheduled_obs = None

    def add_observation(self, observation, indx=None):
        if len(self.surveys) > 0:
            for survey in self.surveys:
                survey.add_observation(observation, indx=indx)

    def _spawn_new_survey(self, too):
        """Create a new survey object for a ToO we haven't seen before.

        Parameters
        ----------
        too : rubin_sim.scheduler.utils.TargetoO object
        """
        new_survey = copy.deepcopy(self.example__to_o_survey)
        new_survey.set_id(too.id)
        new_survey.set_target_map(too.footprint)

        return new_survey

    def _check_survey_list(self, conditions):
        """There is a current ToO in the conditions."""

        running_ids = [survey.too_id for survey in self.surveys]
        current_ids = [too.id for too in conditions.targets_of_opportunity]

        # delete any ToO surveys that are no longer relevant
        self.surveys = [survey for survey in self.surveys if survey.too_id in current_ids]

        # Spawn new surveys that are needed
        new_surveys = []
        for too in conditions.targets_of_opportunity:
            if too.id not in running_ids:
                new_surveys.append(self._spawn_new_survey(too))
        self.surveys.extend(new_surveys)

    def calc_reward_function(self, conditions):
        # Catch if a new ToO has happened
        if conditions.targets_of_opportunity is not None:
            self._check_survey_list(conditions)

        if len(self.surveys) > 0:
            rewards = [np.nanmax(survey.calc_reward_function(conditions)) for survey in self.surveys]
            self.reward = np.nanmax(rewards)
            self.highest_reward = np.min(np.where(rewards == self.reward))
        else:
            self.reward = -np.inf
            self.highest_reward = None
        return self.reward

    def generate_observations(self, conditions):
        if self.reward > -np.inf:
            result = self.surveys[self.highest_reward].generate_observations(conditions)
            return result


class TooSurvey(BlobSurvey):
    """Survey class to catch incoming target of opportunity anouncements and try to observe them.

    The idea is that we can dynamically update the target footprint basis function, and add new features as more ToOs come in.

    Parameters
    ----------
    too_id : int (None)
        A unique integer ID for the ToO getting observed
    """

    def __init__(
        self,
        basis_functions,
        basis_weights,
        filtername1="r",
        filtername2=None,
        slew_approx=7.5,
        filter_change_approx=140.0,
        read_approx=2.0,
        exptime=30.0,
        nexp=2,
        ideal_pair_time=22.0,
        min_pair_time=15.0,
        search_radius=30.0,
        alt_max=85.0,
        az_range=180.0,
        flush_time=30.0,
        smoothing_kernel=None,
        nside=None,
        dither=True,
        seed=42,
        ignore_obs=None,
        survey_note="ToO",
        detailers=None,
        camera="LSST",
        too_id=None,
        survey_name=None,
    ):
        super(TooSurvey, self).__init__(
            basis_functions=basis_functions,
            basis_weights=basis_weights,
            filtername1=filtername1,
            filtername2=filtername2,
            slew_approx=slew_approx,
            filter_change_approx=filter_change_approx,
            read_approx=read_approx,
            exptime=exptime,
            nexp=nexp,
            ideal_pair_time=ideal_pair_time,
            min_pair_time=min_pair_time,
            search_radius=search_radius,
            alt_max=alt_max,
            az_range=az_range,
            flush_time=flush_time,
            smoothing_kernel=smoothing_kernel,
            nside=nside,
            dither=dither,
            seed=seed,
            ignore_obs=ignore_obs,
            survey_note=survey_note,
            detailers=detailers,
            camera=camera,
            survey_name=survey_name,
        )
        # Include the ToO id in the note
        self.survey_note_base = self.survey_note
        self.set_id(too_id)

    def set_id(self, newid):
        """Set the id"""
        self.too_id = newid
        self.survey_note = self.survey_note_base + ", " + str(newid)

    def set_target_map(self, newmap):
        """
        Expect one of the basis functions to be Footprint_nvis_basis_function
        """
        for basis_func in self.basis_functions:
            if hasattr(basis_func, "footprint"):
                basis_func.footprint = newmap

    def generate_observations_rough(self, conditions):
        # Always spin the tesselation before generating a new block.
        if self.dither:
            self._spin_fields(conditions)
        result = super(TooSurvey, self).generate_observations_rough(conditions)
        return result
