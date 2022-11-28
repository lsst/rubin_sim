import numpy as np
from rubin_sim.scheduler.utils import scheduled_observation
from rubin_sim.scheduler.surveys import BaseSurvey
from rubin_sim.utils import _approx_ra_dec2_alt_az
import logging

log = logging.getLogger(__name__)

__all__ = ["LongGapSurvey"]


class LongGapSurvey(BaseSurvey):
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
    after_meridian : bool (False)
        If True, force the scripted obsrevations to happen after they pass through the meridian.
        This can help make sure we don't hit the zenith exclusion zone.
    hour_step : float (0.5)
        The amount of time to step scheduled observations forward if they could try to execute in the
        zenith avoidance area (hours). Only used if `avoid_zenith` is True.
    ha_min(_max) : float (0,24)
        Trying to set so they don't acctually get used.

    """

    def __init__(
        self,
        blob_survey,
        scripted_survey,
        gap_range=[2, 10],
        long_name="long",
        scripted_tol=2.0,
        alt_min=20,
        alt_max=80.0,
        ha_min=24,
        ha_max=0.0,
        flush_time=2.0,
        dist_tol=1.0,
        block_length=33.0,
        reverse=True,
        seed=42,
        night_max=50000,
        avoid_zenith=True,
        hour_step=0.5,
    ):
        self.blob_survey = blob_survey
        self.scripted_survey = scripted_survey
        self.night = -1
        self.gap_range = np.array(gap_range) / 24.0  # To days
        rng = np.random.default_rng(seed)
        self.gaps = rng.uniform(self.gap_range.min(), self.gap_range.max(), night_max)
        self.gap = 0.0
        self.long_name = long_name
        self.scripted_tol = scripted_tol / 24.0  # To days
        self.alt_min = np.radians(alt_min)
        self.alt_max = np.radians(alt_max)
        self.ha_min = ha_min
        self.ha_max = ha_max
        self.flush_time = flush_time / 24.0
        self.dist_tol = np.radians(dist_tol)
        self.block_length = block_length / 60 / 24.0
        self.reverse = reverse
        self.avoid_zenith = avoid_zenith
        self.mjd_step = hour_step / 24.0

    def add_observation(self, observation, **kwargs):
        self.blob_survey.add_observation(observation, **kwargs)
        self.scripted_survey.add_observation(observation, **kwargs)

    def _check_feasibility(self, conditions):
        f1 = self.blob_survey._check_feasibility(conditions)
        f2 = self.scripted_survey._check_feasibility(conditions)

        # If either one is able to go, we can observe
        result = f1 | f2
        return result

    def get_scheduled_obs(self):
        return self.scripted_survey.get_scheduled_obs()

    def calc_reward_function(self, conditions):
        if conditions.night != self.night:
            # Clear out the scheduled observations
            self.scripted_survey.clear_script()
            self.night = conditions.night + 0
            self.gap = self.gaps[conditions.night]
            time_remaining = conditions.sun_n18_rising - conditions.mjd
            if self.gap > time_remaining:
                self.gap = time_remaining - self.block_length
            # XXX-need to reach into the blob and set what the gap is I guess

        self.r1 = self.blob_survey.calc_reward_function(conditions)
        self.r2 = self.scripted_survey.calc_reward_function(conditions)
        self.reward_checked = True
        return np.nanmax([np.nanmax(self.r1), np.nanmax(self.r2)])

    def generate_observations_rough(self, conditions):
        """"""
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
                if self.reverse:
                    obs_array = obs_array[::-1]
                obs_array = obs_array[
                    np.where(obs_array["filter"] == obs_array["filter"][0])[0]
                ]

                obs_array["mjd"] = conditions.mjd + self.gap
                obs_array["note"] = self.long_name
                sched_array = scheduled_observation(n=obs_array.size)
                for dt in np.intersect1d(
                    obs_array.dtype.names, sched_array.dtype.names
                ):
                    sched_array[dt] = obs_array[dt]
                sched_array["mjd_tol"] = self.scripted_tol
                sched_array["alt_min"] = self.alt_min
                sched_array["alt_max"] = self.alt_max
                sched_array["HA_min"] = self.ha_min
                sched_array["HA_max"] = self.ha_max
                sched_array["flush_by_mjd"] = obs_array["mjd"] + self.flush_time
                sched_array["dist_tol"] = self.dist_tol
                if self.avoid_zenith:
                    # when is the earliest we expect things could execute
                    earliest_mjd = sched_array["mjd"] - sched_array["mjd_tol"]
                    alts = []
                    mjds = np.arange(
                        np.min(earliest_mjd),
                        np.max(sched_array["mjd"]) + self.mjd_step,
                        self.mjd_step,
                    )
                    # Let's compute the alt of everything at earliest and scheduled
                    for mjd in mjds:
                        alt, az = _approx_ra_dec2_alt_az(
                            sched_array["RA"],
                            sched_array["dec"],
                            conditions.site.latitude_rad,
                            conditions.site.longitude_rad,
                            mjd,
                        )
                        alts.append(np.max(alt))

                    while np.max(alts) > self.alt_max:
                        alts = []
                        sched_array["mjd"] += self.mjd_step
                        sched_array["flush_by_mjd"] += self.mjd_step
                        earliest_mjd = sched_array["mjd"] - sched_array["mjd_tol"]
                        # Let's compute the alt of everything then
                        mjds = np.arange(
                            np.min(earliest_mjd),
                            np.max(sched_array["mjd"]) + self.mjd_step,
                            self.mjd_step,
                        )
                        # Let's compute the alt of everything at earliest and scheduled
                        for mjd in mjds:
                            alt, az = _approx_ra_dec2_alt_az(
                                sched_array["RA"],
                                sched_array["dec"],
                                conditions.site.latitude_rad,
                                conditions.site.longitude_rad,
                                mjd,
                            )
                            alts.append(np.max(alt))

                # See if we need to append things to the scripted survey object
                if self.scripted_survey.obs_wanted is not None:
                    sched_array = np.concatenate(
                        [self.scripted_survey.obs_wanted, sched_array]
                    )

                self.scripted_survey.set_script(sched_array)
                observations = o1

        return observations
