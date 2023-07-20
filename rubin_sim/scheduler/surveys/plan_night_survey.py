__all__ = ("PlanAheadSurvey",)

from copy import copy

import healpy as hp
import matplotlib.pylab as plt
import numpy as np

import rubin_sim.scheduler.basis_functions as bfs
from rubin_sim.scheduler import features
from rubin_sim.scheduler.utils import empty_observation, set_default_nside

from .surveys import BlobSurvey


class PlanAheadSurvey(BlobSurvey):
    """Have a survey object that can plan ahead if it will want to observer a blob later in the night

    Parameters
    ----------
    delta_mjd_tol : float
        The tolerance to alow on when to execute scheduled observations (days)
    minimum_sky_area : float
        The minimum sky area to demand before making a scheduled observation (square degrees)
    track_filters : str
        The filter name we want to prevent long gaps on
    in_season : float
        The distance in RA from the meridian at midnight to consider (hours). This is the half-width
    cadence : float
        Ignore gaps below this length (days)
    """

    def __init__(
        self,
        basis_functions,
        basis_weights,
        delta_mjd_tol=0.3 / 24.0,
        minimum_sky_area=200.0,
        track_filters="g",
        in_season=2.5,
        cadence=9,
        **kwargs,
    ):
        super(PlanAheadSurvey, self).__init__(basis_functions, basis_weights, **kwargs)
        # note that self.night is already being used for tracking tesselation.
        # So here's an attribute for seeing if the night has changed for cadence tracking
        self.night_cad = -100
        self.scheduled_obs = None
        self.delta_mjd_tol = delta_mjd_tol
        self.minimum_sky_area = minimum_sky_area  # sq degrees
        self.extra_features = {}
        self.extra_features["last_observed"] = features.Last_observed(filtername=track_filters)
        self.extra_basis_functions = {}
        self.extra_basis_functions["moon_mask"] = bfs.Moon_avoidance_basis_function()
        self.track_filters = track_filters
        self.in_season = in_season / 12.0 * np.pi  # to radians

        self.pix_area = hp.nside2pixarea(self.nside, degrees=True)
        self.cadence = cadence

    # def add_observation(self, observation, **kwargs):
    #    # If a relevant observation got made, recompute when we actually want to observe in the night
    #    if observation['filter'] in self.track_filters:
    #        if self.scheduled_obs is not None:
    #            # this will force a run of self.check_night on the next calc_reward_function
    #            self.night = observation['night'] - 1
    #    super(PlanAheadSurvey, self).add_observation(observation, **kwargs)

    def check_night(self, conditions):
        """"""
        delta_mjd = conditions.mjd - self.extra_features["last_observed"].feature
        moon_mask = self.extra_basis_functions["moon_mask"](conditions)

        pix_to_obs = np.where(
            (delta_mjd > self.cadence)
            & (np.abs(conditions.az_to_antisun) < self.in_season)
            & (moon_mask >= 0)
        )[0]

        area = np.size(pix_to_obs) * self.pix_area

        # If there are going to be some observations at a given time
        if area > self.minimum_sky_area:
            # Maybe just calculate the mean (with angles)
            # Via https://en.wikipedia.org/wiki/Mean_of_circular_quantities
            mean_ra = np.arctan2(
                np.mean(np.sin(conditions.ra[pix_to_obs])),
                np.mean(np.cos(conditions.ra[pix_to_obs])),
            )
            if mean_ra < 0:
                mean_ra += 2.0 * np.pi

            hour_angle = conditions.lmst - mean_ra * 12.0 / np.pi
            # This should be running -12 hours to +12 hours

            hour_angle[np.where(hour_angle < -12)] += 24
            hour_angle[np.where(hour_angle > 12)] -= 24

            if hour_angle < 0:
                self.scheduled_obs = conditions.mjd - hour_angle / 24.0
            else:
                self.scheduled_obs = conditions.mjd
        else:
            self.scheduled_obs = None

    def calc_reward_function(self, conditions):
        # Only compute if we will want to observe sometime in the night
        self.reward = -np.inf
        if self._check_feasibility(conditions):
            if self.night_cad != conditions.night:
                self.check_night(conditions)
                self.night_cad = copy(conditions.night)

            if self.scheduled_obs is not None:
                # If there are scheduled observations, and we are in the correct time window
                delta_mjd = conditions.mjd - self.scheduled_obs
                if (np.abs(delta_mjd) < self.delta_mjd_tol) & (self.scheduled_obs is not None):
                    # So, we think there's a region that has had a long gap and can be observed
                    # call the standard reward function
                    self.reward = super(PlanAheadSurvey, self).calc_reward_function(conditions)

        return self.reward

    def generate_observations(self, conditions):
        observations = super(PlanAheadSurvey, self).generate_observations(conditions)
        # We are providing observations, so clear the scheduled obs and reset the night so
        # self.check_night will get called again in case there's another blob that should be done
        # after this one completes
        self.scheduled_obs = None
        self.night_cad = conditions.night - 1
        return observations
