__all__ = ("DescDdf", "generate_desc_dd_surveys")

import copy
import random

import healpy as hp
import numpy as np

import rubin_sim.scheduler.basis_functions as basis_functions
from rubin_sim.scheduler.surveys import BaseSurvey
from rubin_sim.scheduler.utils import empty_observation


class DescDdf(BaseSurvey):
    """DDF survey based on Scolnic et al Cadence White Paper."""

    def __init__(
        self,
        basis_functions,
        RA,
        dec,
        sequences=None,
        exptime=30.0,
        nexp=1,
        ignore_obs=None,
        survey_name="DD_DESC",
        reward_value=101.0,
        readtime=2.0,
        filter_change_time=120.0,
        nside=None,
        flush_pad=30.0,
        seed=42,
        detailers=None,
    ):
        super(DescDdf, self).__init__(
            nside=nside,
            basis_functions=basis_functions,
            detailers=detailers,
            ignore_obs=ignore_obs,
        )

        self.ra = np.radians(RA)
        self.ra_hours = RA / 360.0 * 24.0
        self.dec = np.radians(dec)
        self.survey_name = survey_name
        self.reward_value = reward_value
        self.flush_pad = flush_pad / 60.0 / 24.0  # To days

        self.simple_obs = empty_observation()
        self.simple_obs["RA"] = np.radians(RA)
        self.simple_obs["dec"] = np.radians(dec)
        self.simple_obs["exptime"] = exptime
        self.simple_obs["nexp"] = nexp
        self.simple_obs["note"] = survey_name

        # Define the sequences we would like to do
        if sequences is None:
            self.sequences = [{"u": 2, "g": 2, "r": 4, "i": 8}, {"z": 25, "y": 4}, None]
        else:
            self.sequences = sequences

        self.approx_times = []
        for sequence in self.sequences:
            if sequence is None:
                self.approx_times.append(0)
            else:
                n_exp_in_seq = np.sum(list(sequence.values()))
                time_needed = filter_change_time * len(sequence.keys())
                time_needed += exptime * n_exp_in_seq
                time_needed += readtime * n_exp_in_seq * nexp
                self.approx_times.append(time_needed / 3600.0 / 24.0)

        # Track what we last tried to do
        # XXX-this should probably go into self.extra_features or something for consistency.
        self.sequence_index = 0
        self.last_night_observed = -100

    def check_continue(self, observation, conditions):
        # feasibility basis functions?
        """
        This method enables external calls to check if a given observations that belongs to this survey is
        feasible or not. This is called once a sequence has started to make sure it can continue.

        XXX--TODO:  Need to decide if we want to develope check_continue, or instead hold the
        sequence in the survey, and be able to check it that way.
        """

        result = True
        return result

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """

        # No more if we've already observed this night
        if self.last_night_observed == conditions.night:
            return False

        # Advance the sequence index if we have skipped a day intentionally
        if (self.sequences[self.sequence_index] is None) & (conditions.night - self.last_night_observed > 1):
            self.sequence_index = (self.sequence_index + 1) % len(self.sequences)

        # If we want to skip this day
        if self.sequences[self.sequence_index] is None:
            return False

        # The usual basis function checks
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        return result

    def calc_reward_function(self, conditions):
        result = -np.inf
        if self._check_feasibility(conditions):
            result = self.reward_value
        return result

    def generate_observations_rough(self, conditions):
        result = []
        for key in self.sequences[self.sequence_index]:
            # Just skip adding the z-band ones if it's not loaded
            if key in conditions.mounted_filters:
                for i in np.arange(self.sequences[self.sequence_index][key]):
                    temp_obs = self.simple_obs.copy()
                    temp_obs["filter"] = key
                    # XXX--need to set flush by mjd
                    result.append(temp_obs)
        for i, obs in enumerate(result):
            result[i]["flush_by_mjd"] = (
                conditions.mjd + self.approx_times[self.sequence_index] + self.flush_pad
            )
        # Just assuming this sequence gets observed.
        self.last_night_observed = conditions.night
        self.sequence_index = (self.sequence_index + 1) % len(self.sequences)
        return result


def desc_dd_bfs(RA, dec, survey_name, ha_limits, frac_total=0.0185):
    """
    Convienence function to generate all the feasibility basis functions
    """
    bfs = []
    bfs.append(basis_functions.Not_twilight_basis_function(sun_alt_limit=-18))
    bfs.append(basis_functions.Time_to_twilight_basis_function(time_needed=30.0))
    bfs.append(basis_functions.Hour_Angle_limit_basis_function(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.Rising_more_basis_function(RA=RA))
    bfs.append(basis_functions.Clouded_out_basis_function())

    return bfs


def generate_desc_dd_surveys(nside=None, nexp=1, detailers=None):
    surveys = []

    # ELAIS S1
    RA = 9.45
    dec = -44.0
    survey_name = "DD:ELAISS1"
    ha_limits = ([0.0, 1.18], [21.82, 24.0])
    bfs = desc_dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(
        DescDdf(
            bfs,
            RA,
            dec,
            survey_name=survey_name,
            reward_value=100,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    # XMM-LSS
    survey_name = "DD:XMM-LSS"
    RA = 35.708333
    dec = -4 - 45 / 60.0
    ha_limits = ([0.0, 1.3], [21.7, 24.0])
    bfs = desc_dd_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(
        DescDdf(
            bfs,
            RA,
            dec,
            survey_name=survey_name,
            reward_value=100,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )
    # Extended Chandra Deep Field South
    RA = 53.125
    dec = -28.0 - 6 / 60.0
    survey_name = "DD:ECDFS"
    ha_limits = [[0.5, 3.0], [20.0, 22.5]]
    bfs = desc_dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(
        DescDdf(
            bfs,
            RA,
            dec,
            survey_name=survey_name,
            reward_value=100,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    # COSMOS
    RA = 150.1
    dec = 2.0 + 10.0 / 60.0 + 55 / 3600.0
    survey_name = "DD:COSMOS"
    ha_limits = ([0.0, 1.5], [21.5, 24.0])
    bfs = desc_dd_bfs(RA, dec, survey_name, ha_limits)
    # have a special sequence for COSMOS
    sequences = [{"g": 2, "r": 4, "i": 8}, {"z": 25, "y": 4}]
    surveys.append(
        DescDdf(
            bfs,
            RA,
            dec,
            survey_name=survey_name,
            reward_value=100,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
            sequences=sequences,
        )
    )

    # Just do the two Euclid fields independently for now
    survey_name = "DD:EDFSa"
    RA = 58.97
    dec = -49.28
    ha_limits = ([0.0, 1.5], [23.0, 24.0])
    bfs = desc_dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(
        DescDdf(
            bfs,
            RA,
            dec,
            survey_name=survey_name,
            reward_value=100,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    survey_name = "DD:EDFSb"
    RA = 63.6
    dec = -47.60
    ha_limits = ([0.0, 1.5], [23.0, 24.0])
    bfs = desc_dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(
        DescDdf(
            bfs,
            RA,
            dec,
            survey_name=survey_name,
            reward_value=100,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    return surveys

    pass
