__all__ = ("DeepDrillingSurvey", "generate_dd_surveys", "dd_bfs")

import copy
import logging
import random

import numpy as np

import rubin_sim.scheduler.basis_functions as basis_functions
from rubin_sim.scheduler import features
from rubin_sim.scheduler.surveys import BaseSurvey
from rubin_sim.scheduler.utils import empty_observation
from rubin_sim.utils import ddf_locations

log = logging.getLogger(__name__)


class DeepDrillingSurvey(BaseSurvey):
    """A survey class for running deep drilling fields.

    Parameters
    ----------
    basis_functions : list of rubin_sim.scheduler.basis_function objects
        These should be feasibility basis functions.
    RA : float
        The RA of the field (degrees)
    dec : float
        The dec of the field to observe (degrees)
    sequence : list of observation objects or str (rgizy)
        The sequence of observations to take. Can be a string of list of obs objects.
    nvis : list of ints
        The number of visits in each filter. Should be same length as sequence.
    survey_name : str (DD)
        The name to give this survey so it can be tracked
    reward_value : float (101.)
        The reward value to report if it is able to start (unitless).
    readtime : float (2.)
        Readout time for computing approximate time of observing the sequence. (seconds)
    flush_pad : float (30.)
        How long to hold observations in the queue after they were expected to be completed (minutes).
    """

    def __init__(
        self,
        basis_functions,
        RA,
        dec,
        sequence="rgizy",
        nvis=[20, 10, 20, 26, 20],
        exptime=30.0,
        u_exptime=30.0,
        nexp=2,
        ignore_obs=None,
        survey_name="DD",
        reward_value=None,
        readtime=2.0,
        filter_change_time=120.0,
        nside=None,
        flush_pad=30.0,
        seed=42,
        detailers=None,
    ):
        super(DeepDrillingSurvey, self).__init__(
            nside=nside,
            basis_functions=basis_functions,
            detailers=detailers,
            ignore_obs=ignore_obs,
        )
        random.seed(a=seed)

        self.ra = np.radians(RA)
        self.ra_hours = RA / 360.0 * 24.0
        self.dec = np.radians(dec)
        self.survey_name = survey_name
        self.reward_value = reward_value
        self.flush_pad = flush_pad / 60.0 / 24.0  # To days
        self.filter_sequence = []
        if type(sequence) == str:
            self.observations = []
            for num, filtername in zip(nvis, sequence):
                for j in range(num):
                    obs = empty_observation()
                    obs["filter"] = filtername
                    if filtername == "u":
                        obs["exptime"] = u_exptime
                    else:
                        obs["exptime"] = exptime
                    obs["RA"] = self.ra
                    obs["dec"] = self.dec
                    obs["nexp"] = nexp
                    obs["note"] = survey_name
                    self.observations.append(obs)
        else:
            self.observations = sequence

        # Let's just make this an array for ease of use
        self.observations = np.concatenate(self.observations)
        order = np.argsort(self.observations["filter"])
        self.observations = self.observations[order]

        n_filter_change = np.size(np.unique(self.observations["filter"]))

        # Make an estimate of how long a seqeunce will take. Assumes no major rotational or spatial
        # dithering slowing things down.
        self.approx_time = (
            np.sum(self.observations["exptime"] + readtime * self.observations["nexp"]) / 3600.0 / 24.0
            + filter_change_time * n_filter_change / 3600.0 / 24.0
        )  # to days

        if self.reward_value is None:
            self.extra_features["Ntot"] = features.N_obs_survey()
            self.extra_features["N_survey"] = features.N_obs_survey(note=self.survey_name)

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

    def calc_reward_function(self, conditions):
        result = -np.inf
        if self._check_feasibility(conditions):
            if self.reward_value is not None:
                result = self.reward_value
            else:
                # XXX This might backfire if we want to have DDFs with different fractions of the
                # survey time. Then might need to define a goal fraction, and have the reward be the
                # number of observations behind that target fraction.
                result = self.extra_features["Ntot"].feature / (self.extra_features["N_survey"].feature + 1)
        return result

    def generate_observations_rough(self, conditions):
        result = []
        if self._check_feasibility(conditions):
            result = copy.deepcopy(self.observations)

            # Set the flush_by
            result["flush_by_mjd"] = conditions.mjd + self.approx_time + self.flush_pad

            # remove filters that are not mounted
            mask = np.isin(result["filter"], conditions.mounted_filters)
            result = result[mask]
            # Put current loaded filter first
            ind1 = np.where(result["filter"] == conditions.current_filter)[0]
            ind2 = np.where(result["filter"] != conditions.current_filter)[0]
            result = result[ind1.tolist() + (ind2.tolist())]

            # convert to list of array. Arglebargle, don't understand why I need a reshape there
            final_result = [
                row.reshape(
                    1,
                )
                for row in result
            ]
            result = final_result

        return result

    def __repr__(self):
        return f"<{self.__class__.__name__} survey_name='{self.survey_name}', RA={self.ra}, dec={self.dec} at {hex(id(self))}>"


def dd_bfs(
    RA,
    dec,
    survey_name,
    ha_limits,
    frac_total=0.0185 / 2.0,
    aggressive_frac=0.011 / 2.0,
    delays=[0.0, 0.5, 1.5],
    time_needed=62.0,
):
    """
    Convienence function to generate all the feasibility basis functions
    """
    sun_alt_limit = -18.0
    fractions = [0.00, aggressive_frac, frac_total]
    bfs = []
    bfs.append(basis_functions.NotTwilightBasisFunction(sun_alt_limit=sun_alt_limit))
    bfs.append(basis_functions.TimeToTwilightBasisFunction(time_needed=time_needed))
    bfs.append(basis_functions.HourAngleLimitBasisFunction(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.MoonDownBasisFunction())
    bfs.append(basis_functions.FractionOfObsBasisFunction(frac_total=frac_total, survey_name=survey_name))
    bfs.append(
        basis_functions.LookAheadDdfBasisFunction(
            frac_total,
            aggressive_frac,
            sun_alt_limit=sun_alt_limit,
            time_needed=time_needed,
            RA=RA,
            survey_name=survey_name,
            ha_limits=ha_limits,
        )
    )
    bfs.append(
        basis_functions.SoftDelayBasisFunction(fractions=fractions, delays=delays, survey_name=survey_name)
    )
    bfs.append(basis_functions.TimeToScheduledBasisFunction(time_needed=time_needed))

    return bfs


def generate_dd_surveys(
    nside=None,
    nexp=2,
    detailers=None,
    euclid_detailers=None,
    reward_value=100,
    frac_total=0.0185 / 2.0,
    aggressive_frac=0.011 / 2.0,
    exptime=30,
    u_exptime=30,
    nvis_master=[8, 20, 10, 20, 26, 20],
    delays=[0.0, 0.5, 1.5],
):
    """Utility to return a list of standard deep drilling field surveys.

    XXX-Someone double check that I got the coordinates right!

    """

    if euclid_detailers is None:
        euclid_detailers = detailers

    surveys = []

    locations = ddf_locations()

    # ELAIS S1
    survey_name = "DD:ELAISS1"
    RA = locations["ELAISS1"][0]
    dec = locations["ELAISS1"][1]
    ha_limits = ([0.0, 1.5], [21.5, 24.0])
    bfs = dd_bfs(
        RA,
        dec,
        survey_name,
        ha_limits,
        frac_total=frac_total,
        aggressive_frac=aggressive_frac,
        delays=delays,
    )
    surveys.append(
        DeepDrillingSurvey(
            bfs,
            RA,
            dec,
            sequence="urgizy",
            nvis=nvis_master,
            exptime=exptime,
            u_exptime=u_exptime,
            survey_name=survey_name,
            reward_value=reward_value,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    # XMM-LSS
    survey_name = "DD:XMM-LSS"
    RA = locations["XMM_LSS"][0]
    dec = locations["XMM_LSS"][1]
    ha_limits = ([0.0, 1.5], [21.5, 24.0])
    bfs = dd_bfs(
        RA,
        dec,
        survey_name,
        ha_limits,
        frac_total=frac_total,
        aggressive_frac=aggressive_frac,
        delays=delays,
    )

    surveys.append(
        DeepDrillingSurvey(
            bfs,
            RA,
            dec,
            sequence="urgizy",
            exptime=exptime,
            u_exptime=u_exptime,
            nvis=nvis_master,
            survey_name=survey_name,
            reward_value=reward_value,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    # Extended Chandra Deep Field South
    survey_name = "DD:ECDFS"
    RA = locations["ECDFS"][0]
    dec = locations["ECDFS"][1]
    ha_limits = [[0.5, 3.0], [20.0, 22.5]]
    bfs = dd_bfs(
        RA,
        dec,
        survey_name,
        ha_limits,
        frac_total=frac_total,
        aggressive_frac=aggressive_frac,
        delays=delays,
    )
    surveys.append(
        DeepDrillingSurvey(
            bfs,
            RA,
            dec,
            sequence="urgizy",
            nvis=nvis_master,
            exptime=exptime,
            u_exptime=u_exptime,
            survey_name=survey_name,
            reward_value=reward_value,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    # COSMOS
    survey_name = "DD:COSMOS"
    RA = locations["COSMOS"][0]
    dec = locations["COSMOS"][1]
    ha_limits = ([0.0, 2.5], [21.5, 24.0])
    bfs = dd_bfs(
        RA,
        dec,
        survey_name,
        ha_limits,
        frac_total=frac_total,
        aggressive_frac=aggressive_frac,
        delays=delays,
    )
    surveys.append(
        DeepDrillingSurvey(
            bfs,
            RA,
            dec,
            sequence="urgizy",
            nvis=nvis_master,
            exptime=exptime,
            u_exptime=u_exptime,
            survey_name=survey_name,
            reward_value=reward_value,
            nside=nside,
            nexp=nexp,
            detailers=detailers,
        )
    )

    # Euclid Fields
    # I can use the sequence kwarg to do two positions per sequence
    filters = "urgizy"
    nviss = nvis_master
    survey_name = "DD:EDFS"
    # Note the sequences need to be in radians since they are using observation objects directly
    # Coords from jc.cuillandre@cea.fr Oct 15, 2020
    r_as = np.radians([locations["EDFS_a"][0], locations["EDFS_b"][0]])
    decs = np.radians([locations["EDFS_a"][1], locations["EDFS_b"][1]])
    suffixes = [", a", ", b"]
    sequence = []

    for filtername, nvis in zip(filters, nviss):
        for ra, dec, suffix in zip(r_as, decs, suffixes):
            for num in range(nvis):
                obs = empty_observation()
                obs["filter"] = filtername
                if filtername == "u":
                    obs["exptime"] = u_exptime
                else:
                    obs["exptime"] = exptime
                obs["RA"] = ra
                obs["dec"] = dec
                obs["nexp"] = nexp
                obs["note"] = survey_name + suffix
                sequence.append(obs)

    ha_limits = ([0.0, 1.5], [22.5, 24.0])
    # And back to degrees for the basis function. Need to bump up the time needed since it's a double field.
    bfs = dd_bfs(
        np.degrees(r_as[0]),
        np.degrees(decs[0]),
        survey_name,
        ha_limits,
        frac_total=frac_total,
        aggressive_frac=aggressive_frac,
        delays=delays,
        time_needed=120.0,
    )
    surveys.append(
        DeepDrillingSurvey(
            bfs,
            np.degrees(r_as),
            np.degrees(decs),
            sequence=sequence,
            survey_name=survey_name,
            reward_value=reward_value,
            nside=nside,
            nexp=nexp,
            detailers=euclid_detailers,
        )
    )

    return surveys
