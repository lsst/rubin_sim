__all__ = (
    "BaseFeature",
    "BaseSurveyFeature",
    "NObsCount",
    "NObsSurvey",
    "LastObservation",
    "LastsequenceObservation",
    "LastFilterChange",
    "NObservations",
    "CoaddedDepth",
    "LastObserved",
    "NObsNight",
    "PairInNight",
    "RotatorAngle",
    "NObservationsSeason",
    "NObsCountSeason",
    "NObservationsCurrentSeason",
    "LastNObsTimes",
    "SurveyInNight",
    "NoteInNight",
    "NoteLastObserved",
)

import healpy as hp
import numpy as np
from scipy.stats import binned_statistic

from rubin_sim.scheduler import utils
from rubin_sim.scheduler.utils import IntRounded
from rubin_sim.skybrightness_pre import dark_sky
from rubin_sim.utils import _hpid2_ra_dec, calc_season, m5_flat_sed, ra_dec2_hpid


class BaseFeature:
    """
    Base class for features.
    """

    def __init__(self, **kwargs):
        # self.feature should be a float, bool, or healpix size numpy array, or numpy masked array
        self.feature = None

    def __call__(self):
        return self.feature


class BaseSurveyFeature(BaseFeature):
    """
    Feature that tracks progreess of the survey. Takes observations and updates self.feature
    """

    def add_observations_array(self, observations_array, observations_hpid):
        """ """
        print(self)
        raise NotImplementedError

    def add_observation(self, observation, indx=None, **kwargs):
        """
        Parameters
        ----------
        obsevation : dict-like
            Object that contains the information about the observation (ra, dec, filter, mjd, etc)
        indx : ints (None)
            The healpixel indices that the observation overlaps.
        """
        raise NotImplementedError


class SurveyInNight(BaseSurveyFeature):
    """Keep track of how many times a survey has executed in a night."""

    def __init__(self, survey_str=""):
        self.feature = 0
        self.survey_str = survey_str
        self.night = -1

    def add_observation(self, observation, indx=None):
        if observation["night"] != self.night:
            self.night = observation["night"]
            self.feature = 0

        if self.survey_str in observation["note"]:
            self.feature += 1


class NoteInNight(BaseSurveyFeature):
    """How many times a matching note has executed in the current night"""

    def __init__(self, notes=[]):
        self.feature = 0
        self.notes = notes
        self.current_night = -100

    def add_observations_array(self, observations_array, observations_hpid):
        if self.current_night != observations_array["night"][-1]:
            self.current_night = observations_array["night"][-1].copy()
            self.feature = 0
        indx = np.where(observations_array["night"] == observations_array["night"][-1])[0]
        for ind in indx:
            if observations_array["note"][ind] in self.notes:
                self.feature += 1

    def add_observation(self, observation, indx=None):
        if self.current_night != observation["night"]:
            self.current_night = observation["night"].copy()
            self.feature = 0
        if observation["note"] in self.notes:
            self.feature += 1


class NObsCount(BaseSurveyFeature):
    """Count the number of observations. Total number, not tracked over sky

    Parameters
    ----------
    filtername : str (None)
        The filter to count (if None, all filters counted)
    """

    def __init__(self, filtername=None, tag=None):
        self.feature = 0
        self.filtername = filtername
        # XXX--is "tag" actually used anywhere? Maybe should remove that.
        self.tag = tag

    def add_observations_array(self, observations_array, observations_hpid):
        if self.filtername is None:
            self.feature += np.size(observations_array)
        else:
            in_filt = np.where(observations_array["filter"] == self.filtername)[0]
            self.feature += np.size(in_filt)

    def add_observation(self, observation, indx=None):
        if (self.filtername is None) and (self.tag is None):
            # Track all observations
            self.feature += 1
        elif (
            (self.filtername is not None)
            and (self.tag is None)
            and (observation["filter"][0] in self.filtername)
        ):
            # Track all observations on a specified filter
            self.feature += 1
        elif (self.filtername is None) and (self.tag is not None) and (observation["tag"][0] in self.tag):
            # Track all observations on a specified tag
            self.feature += 1
        elif (
            (self.filtername is None)
            and (self.tag is not None)
            and
            # Track all observations on a specified filter on a specified tag
            (observation["filter"][0] in self.filtername)
            and (observation["tag"][0] in self.tag)
        ):
            self.feature += 1


class NObsCountSeason(BaseSurveyFeature):
    """Count the number of observations.

    Parameters
    ----------
    filtername : str (None)
        The filter to count (if None, all filters counted)
    """

    def __init__(
        self,
        season,
        nside=None,
        filtername=None,
        tag=None,
        season_modulo=2,
        offset=None,
        max_season=None,
        season_length=365.25,
    ):
        self.feature = 0
        self.filtername = filtername
        self.tag = tag
        self.season = season
        self.season_modulo = season_modulo
        if offset is None:
            self.offset = np.zeros(hp.nside2npix(nside), dtype=int)
        else:
            self.offset = offset
        self.max_season = max_season
        self.season_length = season_length

    def add_observation(self, observation, indx=None):
        season = utils.season_calc(
            observation["night"],
            modulo=self.season_modulo,
            offset=self.offset[indx],
            max_season=self.max_season,
            season_length=self.season_length,
        )
        if self.season in season:
            if (self.filtername is None) and (self.tag is None):
                # Track all observations
                self.feature += 1
            elif (
                (self.filtername is not None)
                and (self.tag is None)
                and (observation["filter"][0] in self.filtername)
            ):
                # Track all observations on a specified filter
                self.feature += 1
            elif (self.filtername is None) and (self.tag is not None) and (observation["tag"][0] in self.tag):
                # Track all observations on a specified tag
                self.feature += 1
            elif (
                (self.filtername is None)
                and (self.tag is not None)
                and
                # Track all observations on a specified filter on a specified tag
                (observation["filter"][0] in self.filtername)
                and (observation["tag"][0] in self.tag)
            ):
                self.feature += 1


class NObsSurvey(BaseSurveyFeature):
    """Count the number of observations.

     Parameters
    ----------
    note : str (None)
        Only count observations that have str in their note field
    """

    def __init__(self, note=None):
        self.feature = 0
        self.note = note

    def add_observation(self, observation, indx=None):
        # Track all observations
        if self.note is None:
            self.feature += 1
        else:
            if self.note in observation["note"]:
                self.feature += 1


class LastObservation(BaseSurveyFeature):
    """Track the last observation. Useful if you want to see when the
    last time a survey took an observation.

    Parameters
    ----------
    survey_name : str (None)
        Only records if the survey name matches (or survey_name set to None)
    """

    def __init__(self, survey_name=None):
        self.survey_name = survey_name
        # Start out with an empty observation
        self.feature = utils.empty_observation()

    def add_observations_array(self, observations_array, observations_hpid):
        if self.survey_name is not None:
            good = np.where(observations_array["note"] == self.survey_name)[0]
            if np.size(good) < 0:
                self.feature = observations_array[good[-1]]
        else:
            self.feature = observations_array[-1]

    def add_observation(self, observation, indx=None):
        if self.survey_name is not None:
            if self.survey_name in observation["note"]:
                self.feature = observation
        else:
            self.feature = observation


class LastsequenceObservation(BaseSurveyFeature):
    """When was the last observation"""

    def __init__(self, sequence_ids=""):
        self.sequence_ids = sequence_ids  # The ids of all sequence observations...
        # Start out with an empty observation
        self.feature = utils.empty_observation()

    def add_observation(self, observation, indx=None):
        if observation["survey_id"] in self.sequence_ids:
            self.feature = observation


class LastFilterChange(BaseSurveyFeature):
    """Record when the filter last changed."""

    def __init__(self):
        self.feature = {"mjd": 0.0, "previous_filter": None, "current_filter": None}

    def add_observation(self, observation, indx=None):
        if self.feature["current_filter"] is None:
            self.feature["mjd"] = observation["mjd"][0]
            self.feature["previous_filter"] = None
            self.feature["current_filter"] = observation["filter"][0]
        elif observation["filter"][0] != self.feature["current_filter"]:
            self.feature["mjd"] = observation["mjd"][0]
            self.feature["previous_filter"] = self.feature["current_filter"]
            self.feature["current_filter"] = observation["filter"][0]


class NObservations(BaseSurveyFeature):
    """
    Track the number of observations that have been made across the sky.

    Parameters
    ----------
    filtername : str ('r')
        String or list that has all the filters that can count.
    nside : int (32)
        The nside of the healpixel map to use

    """

    def __init__(self, filtername=None, nside=None, survey_name=None):
        if nside is None:
            nside = utils.set_default_nside()

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername
        self.survey_name = survey_name
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        if self.survey_name is not None:
            tmp = [name in self.survey_name for name in observations_hpid["note"]]
            valid_indx = valid_indx * np.array(tmp)
        data = observations_hpid[valid_indx]
        if np.size(data) > 0:
            result, _be, _bn = binned_statistic(
                data["hpid"], np.ones(data.size), statistic=np.sum, bins=self.bins
            )
            self.feature += result

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        indx : ints
            The indices of the healpixel map that have been observed by observation
        """

        if self.filtername is None or observation["filter"][0] in self.filtername:
            if self.survey_name is None or observation["note"] in self.survey_name:
                self.feature[indx] += 1


class NObservationsSeason(BaseSurveyFeature):
    """
    Track the number of observations that have been made across sky

    Parameters
    ----------
    season : int
        Only count observations in this season (year).
    filtername : str ('r')
        String or list that has all the filters that can count.
    nside : int (32)
        The nside of the healpixel map to use
    offset : int (0)
        The offset to use when computing the season (days)
    modulo : int (None)
        How to mod the years when computing season

    """

    def __init__(
        self,
        season,
        filtername=None,
        nside=None,
        offset=0,
        modulo=None,
        max_season=None,
        season_length=365.25,
    ):
        if offset is None:
            offset = np.zeros(hp.nside2npix(nside), dtype=int)
        if nside is None:
            nside = utils.set_default_nside()

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername
        self.offset = offset
        self.modulo = modulo
        self.season = season
        self.max_season = max_season
        self.season_length = season_length

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        indx : ints
            The indices of the healpixel map that have been observed by observation
        """

        observation_season = utils.season_calc(
            observation["night"],
            offset=self.offset[indx],
            modulo=self.modulo,
            max_season=self.max_season,
            season_length=self.season_length,
        )
        if self.season in observation_season:
            if self.filtername is None or observation["filter"][0] in self.filtername:
                self.feature[indx] += 1


class LargestN:
    def __init__(self, n):
        self.n = n

    def __call__(self, in_arr):
        if np.size(in_arr) < self.n:
            return -1
        result = in_arr[-self.n]
        return result


class LastNObsTimes(BaseSurveyFeature):
    """Record the last three observations for each healpixel"""

    def __init__(self, filtername=None, n_obs=3, nside=None):
        self.filtername = filtername
        self.n_obs = n_obs
        if nside is None:
            nside = utils.set_default_nside()
        self.feature = np.zeros((n_obs, hp.nside2npix(nside)), dtype=float)
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        # Assumes we're already sorted on mjd
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        data = observations_hpid[valid_indx]

        if np.size(data) > 0:
            for i in range(1, self.n_obs + 1):
                func = LargestN(i)
                result, _be, _bn = binned_statistic(data["hpid"], data["mjd"], statistic=func, bins=self.bins)
                # some_vals = np.where(np.sum(result, axis=1) > 0)[0]
                self.feature[-i, :] = result

    def add_observation(self, observation, indx=None):
        if self.filtername is None or observation["filter"][0] in self.filtername:
            self.feature[0:-1, indx] = self.feature[1:, indx]
            self.feature[-1, indx] = observation["mjd"]


class NObservationsCurrentSeason(BaseSurveyFeature):
    """Track how many observations have been taken in the current season that meet criteria"""

    def __init__(
        self,
        filtername=None,
        nside=None,
        seeing_fwhm_max=None,
        m5_penalty_max=None,
        mjd_start=1,
    ):
        self.filtername = filtername
        if nside is None:
            self.nside = utils.set_default_nside()
        else:
            self.nside = nside
        self.seeing_fwhm_max = seeing_fwhm_max
        self.m5_penalty_max = m5_penalty_max

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        if self.filtername is not None:
            self.dark_map = dark_sky(nside)[filtername]
        self.ones = np.ones(hp.nside2npix(self.nside))
        self.ra, self.dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
        self.season_map = calc_season(np.degrees(self.ra), mjd_start)
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def season_update(self, observation=None, conditions=None):
        """clear the map anywhere the season has rolled over"""
        if observation is not None:
            current_season = calc_season(np.degrees(self.ra), observation["mjd"])
        if conditions is not None:
            current_season = calc_season(np.degrees(self.ra), conditions.mjd)

        # If the season has changed anywhere, set that count to zero
        new_season = np.where((self.season_map - current_season) != 0)
        self.feature[new_season] = 0
        self.season_map = current_season

    def add_observations_array(self, observations_array, observations_hpid):
        self.season_update(observation=observations_array[-1])

        check1 = np.zeros(observations_array.size, dtype=bool)
        if self.seeing_fwhm_max is not None:
            check1[np.where(observations_array["FWHMeff"] <= self.seeing_fwhm_max)] = True
        else:
            check1[:] = True

        check2 = np.zeros(observations_array.size, dtype=bool)
        if self.m5_penalty_max is not None:
            hpid = ra_dec2_hpid(self.nside, observations_array["RA"], observations_array["dec"])
            penalty = self.dark_map[hpid] - observations_array["fivesigmadepth"]
            check2[np.where(penalty <= self.m5_penalty_max)] = True
        else:
            check2[:] = True

        if self.filtername is None:
            check3 = np.zeros(observations_array.size, dtype=bool)
            check3[np.where(observations_array["filter"] == self.filter)] = True
        else:
            check3 = np.ones(observations_array.size, dtype=bool)

        good_ids = observations_array[check1 & check2 % check3]["ID"]

        indx = np.in1d(observations_hpid["ID"], observations_array["ID"][good_ids])

        result, _be, _bn = binned_statistic(
            observations_hpid["hpid"][indx],
            observations_hpid["hpid"][indx],
            bins=self.bins,
            statistic=np.size,
        )
        self.feature += result

    def add_observation(self, observation, indx=None):
        self.season_update(observation=observation)

        if self.seeing_fwhm_max is not None:
            check1 = observation["FWHMeff"] <= self.seeing_fwhm_max
        else:
            check1 = True

        if self.m5_penalty_max is not None:
            hpid = ra_dec2_hpid(self.nside, observation["RA"], observation["dec"])
            penalty = self.dark_map[hpid] - observation["fivesigmadepth"]
            check2 = penalty <= self.m5_penalty_max
        else:
            check2 = True

        if check1 & check2:
            if self.filtername is None or observation["filter"][0] in self.filtername:
                self.feature[indx] += 1


class CoaddedDepth(BaseSurveyFeature):
    """
    Track the co-added depth that has been reached accross the sky

    Parameters
    ----------
    fwh_meff_limit : float (100)
        The effective FWHM of the seeing (arcsecond). Images will only be added to the
        coadded depth if the observation FWHM is less than or equal to the limit.  Default 100.
    """

    def __init__(self, filtername="r", nside=None, fwh_meff_limit=100.0):
        if nside is None:
            nside = utils.set_default_nside()
        self.filtername = filtername
        self.fwh_meff_limit = IntRounded(fwh_meff_limit)
        # Starting at limiting mag of zero should be fine.
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        if observation["filter"] == self.filtername:
            if IntRounded(observation["FWHMeff"]) <= self.fwh_meff_limit:
                m5 = m5_flat_sed(
                    observation["filter"],
                    observation["skybrightness"],
                    observation["FWHMeff"],
                    observation["exptime"],
                    observation["airmass"],
                )

                self.feature[indx] = 1.25 * np.log10(10.0 ** (0.8 * self.feature[indx]) + 10.0 ** (0.8 * m5))


class LastObserved(BaseSurveyFeature):
    """
    Track when a pixel was last observed. Assumes observations are added in chronological
    order.
    """

    def __init__(self, filtername="r", nside=None, fill=np.nan):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float) + fill
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        # Assumes we're already sorted on mjd
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        data = observations_hpid[valid_indx]

        if np.size(data) > 0:
            result, _be, _bn = binned_statistic(data["hpid"], data["mjd"], statistic=np.max, bins=self.bins)
            good = np.where(result > 0)
            self.feature[good] = result[good]

    def add_observation(self, observation, indx=None):
        if self.filtername is None:
            self.feature[indx] = observation["mjd"]
        elif observation["filter"][0] in self.filtername:
            self.feature[indx] = observation["mjd"]


class NoteLastObserved(BaseSurveyFeature):
    """Track the last time an observation with a particular `note` field was
    made.

    Parameters
    ----------
    note : str
        Substring to match an observation note field to keep track of.
    """

    def __init__(self, note, filtername=None):
        self.note = note
        self.filtername = filtername
        self.feature = None

    def add_observation(self, observation, indx=None):
        if self.note in observation["note"] and (
            self.filtername is None or self.filtername == observation["filter"]
        ):
            self.feature = observation["mjd"]


class NObsNight(BaseSurveyFeature):
    """
    Track how many times something has been observed in a night
    (Note, even if there are two, it might not be a good pair.)

    Parameters
    ----------
    filtername : string ('r')
        Filter to track.
    nside : int (32)
        Scale of the healpix map

    """

    def __init__(self, filtername="r", nside=None):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=int)
        self.night = None

    def add_observation(self, observation, indx=None):
        if observation["night"] != self.night:
            self.feature *= 0
            self.night = observation["night"]
        if (self.filtername == "") | (self.filtername is None):
            self.feature[indx] += 1
        elif observation["filter"][0] in self.filtername:
            self.feature[indx] += 1


class PairInNight(BaseSurveyFeature):
    """
    Track how many pairs have been observed within a night

    Parameters
    ----------
    gap_min : float (25.)
        The minimum time gap to consider a successful pair in minutes
    gap_max : float (45.)
        The maximum time gap to consider a successful pair (minutes)
    """

    def __init__(self, filtername="r", nside=None, gap_min=25.0, gap_max=45.0):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.indx = np.arange(self.feature.size)
        self.last_observed = LastObserved(filtername=filtername)
        self.gap_min = gap_min / (24.0 * 60)  # Days
        self.gap_max = gap_max / (24.0 * 60)  # Days
        self.night = 0
        # Need to keep a full record of times and healpixels observed in a night.
        self.mjd_log = []
        self.hpid_log = []

    def add_observations_array(self, observations_array, observations_hpid):
        # ok, let's just find the largest night and toss all those in one at a time
        most_recent_night = np.where(observations_hpid["night"] == np.max(observations_hpid["night"]))[0]
        obs_hpid = observations_hpid[most_recent_night]
        uid = np.unique(obs_hpid["ID"])
        for ind_id in uid:
            # maybe a faster searchsorted way to do this, but it'll work for now
            good = np.where(obs_hpid["ID"] == ind_id)[0]
            self.add_observation(observations_hpid[good][0], observations_hpid[good]["hpid"])

    def add_observation(self, observation, indx=None):
        if self.filtername is None:
            infilt = True
        else:
            infilt = observation["filter"][0] in self.filtername
        if infilt:
            if indx is None:
                indx = self.indx
            # Clear values if on a new night
            if self.night != observation["night"]:
                self.feature *= 0.0
                self.night = observation["night"]
                self.mjd_log = []
                self.hpid_log = []

            # record the mjds and healpixels that were observed
            self.mjd_log.extend([np.max(observation["mjd"])] * np.size(indx))
            self.hpid_log.extend(list(indx))

            # Look for the mjds that could possibly pair with observation
            tmin = observation["mjd"] - self.gap_max
            tmax = observation["mjd"] - self.gap_min
            mjd_log = np.array(self.mjd_log)
            left = np.searchsorted(mjd_log, tmin)
            right = np.searchsorted(mjd_log, tmax, side="right")
            # Now check if any of the healpixels taken in the time gap
            # match the healpixels of the observation.
            matches = np.in1d(indx, self.hpid_log[int(left) : int(right)])
            # XXX--should think if this is the correct (fastest) order to check things in.
            self.feature[np.array(indx)[matches]] += 1


class RotatorAngle(BaseSurveyFeature):
    """
    Track what rotation angles things are observed with.
    XXX-under construction
    """

    def __init__(self, filtername="r", binsize=10.0, nside=None):
        """"""
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        # Actually keep a histogram at each healpixel
        self.feature = np.zeros((hp.nside2npix(nside), 360.0 / binsize), dtype=float)
        self.bins = np.arange(0, 360 + binsize, binsize)

    def add_observation(self, observation, indx=None):
        if observation["filter"][0] == self.filtername:
            # I think this is how to broadcast things properly.
            self.feature[indx, :] += np.histogram(observation.rotSkyPos, bins=self.bins)[0]
