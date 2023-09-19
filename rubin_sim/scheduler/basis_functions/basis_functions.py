__all__ = (
    "BaseBasisFunction",
    "ConstantBasisFunction",
    "DelayStartBasisFunction",
    "TargetMapBasisFunction",
    "AvoidLongGapsBasisFunction",
    "AvoidFastRevists",
    "VisitRepeatBasisFunction",
    "M5DiffBasisFunction",
    "StrictFilterBasisFunction",
    "GoalStrictFilterBasisFunction",
    "FilterChangeBasisFunction",
    "SlewtimeBasisFunction",
    "AggressiveSlewtimeBasisFunction",
    "SkybrightnessLimitBasisFunction",
    "CablewrapUnwrapBasisFunction",
    "CadenceEnhanceBasisFunction",
    "CadenceEnhanceTrapezoidBasisFunction",
    "AzimuthBasisFunction",
    "AzModuloBasisFunction",
    "DecModuloBasisFunction",
    "MapModuloBasisFunction",
    "TemplateGenerateBasisFunction",
    "FootprintNvisBasisFunction",
    "ThirdObservationBasisFunction",
    "SeasonCoverageBasisFunction",
    "NObsPerYearBasisFunction",
    "CadenceInSeasonBasisFunction",
    "NearSunTwilightBasisFunction",
    "NObsHighAmBasisFunction",
    "GoodSeeingBasisFunction",
    "ObservedTwiceBasisFunction",
    "EclipticBasisFunction",
    "LimitRepeatBasisFunction",
    "VisitGap",
    "NGoodSeeingBasisFunction",
    "AvoidDirectWind",
    "BalanceVisits",
    "RewardNObsSequence",
    "FilterDistBasisFunction",
)

import warnings

import healpy as hp
import matplotlib.pylab as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_sim.scheduler import features, utils
from rubin_sim.scheduler.utils import IntRounded
from rubin_sim.site_models import SeeingModel
from rubin_sim.skybrightness_pre import dark_sky
from rubin_sim.utils import _hpid2_ra_dec, m5_flat_sed


class BaseBasisFunction:
    """Class that takes features and computes a reward function when called."""

    def __init__(self, nside=None, filtername=None, **kwargs):
        # Set if basis function needs to be recalculated if there is a new
        # observation
        self.update_on_newobs = True
        # Set if basis function needs to be recalculated if conditions change
        self.update_on_mjd = True
        # Dict to hold all the features we want to track
        self.survey_features = {}
        # Keep track of the last time the basis function was called. If mjd
        # doesn't change, use cached value
        self.mjd_last = None
        self.value = 0
        # list the attributes to compare to check if basis functions are equal.
        self.attrs_to_compare = []
        # Do we need to recalculate the basis function
        self.recalc = True
        # Basis functions don't technically all need an nside, but so many do
        #  might as well set it here
        if nside is None:
            self.nside = utils.set_default_nside()
        else:
            self.nside = nside

        self.filtername = filtername

    def add_observations_array(self, observations_array, observations_hpid):
        """Like add_observation, but for loading a whole array of observations at a time

        Parameters
        ----------
        observations_array_in : np.array
            An array of completed observations (with columns like rubin_sim.scheduler.utils.empty_observation).
            Should be sorted by MJD.
        observations_hpid_in : np.array
            Same as observations_array_in, but larger and with an additional column for HEALpix id. Each
            observation is listed mulitple times, once for every HEALpix it overlaps.
        """

        for feature in self.survey_features:
            self.survey_features[feature].add_observations_array(observations_array, observations_hpid)
        if self.update_on_newobs:
            self.recalc = True

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : `np.array`
            An array with information about the input observation
        indx : `np.array`
            The indices of the healpix map that the observation overlaps with
        """
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)
        if self.update_on_newobs:
            self.recalc = True

    def check_feasibility(self, conditions):
        """If there is logic to decide if something is feasible (e.g., only if
        moon is down), it can be calculated here.

        Helps prevent full __call__ from being called more than needed.
        """
        return True

    def _calc_value(self, conditions, **kwargs):
        self.value = 0
        # Update the last time we had an mjd
        self.mjd_last = conditions.mjd + 0
        self.recalc = False
        return self.value

    def __eq__(self):
        # XXX--to work on if we need to make a registry of basis functions.
        pass

    def __ne__(self):
        pass

    def __call__(self, conditions, **kwargs):
        """
        Parameters
        ----------
        conditions : `rubin_sim.scheduler.features.conditions` object
             Object that has attributes for all the current conditions.

        Return a reward healpix map or a reward scalar.
        """
        # If we are not feasible, return -inf
        if not self.check_feasibility(conditions):
            return -np.inf
        if self.recalc:
            self.value = self._calc_value(conditions, **kwargs)
        if self.update_on_mjd:
            if conditions.mjd != self.mjd_last:
                self.value = self._calc_value(conditions, **kwargs)
        return self.value

    def label(self):
        """Creata a label for this basis function.

        Returns
        -------
        label : `str`
            A string suitable for labeling the basis function in a plot or table.
        """
        label = self.__class__.__name__.replace("BasisFunction", "")

        if self.filtername is not None:
            label += f" {self.filtername}"

        label += f" @{id(self)}"

        return label


class ConstantBasisFunction(BaseBasisFunction):
    """Just add a constant"""

    def __call__(self, conditions, **kwargs):
        return 1


class DelayStartBasisFunction(BaseBasisFunction):
    """Force things to not run before a given night"""

    def __init__(self, nights_delay=365.25 * 5):
        super().__init__()
        self.nights_delay = nights_delay

    def check_feasibility(self, conditions):
        result = True
        if conditions.night < self.nights_delay:
            result = False
        return result


class FilterDistBasisFunction(BaseBasisFunction):
    """Track filter distribution, increase reward as fraction of observations in
    specified filter drops.
    """

    def __init__(self, filtername="r"):
        super(FilterDistBasisFunction, self).__init__(filtername=filtername)

        self.survey_features = {}
        # Count of all the observations
        self.survey_features["n_obs_count_all"] = features.NObsCount(filtername=None)
        # Count in filter
        self.survey_features["n_obs_count_in_filt"] = features.NObsCount(filtername=filtername)

    def _calc_value(self, conditions, indx=None):
        result = self.survey_features["n_obs_count_all"].feature / (
            self.survey_features["n_obs_count_in_filt"].feature + 1
        )
        return result


class NObsPerYearBasisFunction(BaseBasisFunction):
    """Reward areas that have not been observed N-times in the last year

    Parameters
    ----------
    filtername : `str` ('r')
        The filter to track
    footprint : `np.array`
        Should be a HEALpix map. Values of 0 or np.nan will be ignored.
    n_obs : `int` (3)
        The number of observations to demand
    season : `float` (300)
        The amount of time to allow pass before marking a region as "behind". Default 365.25 (days).
    season_start_hour : `float` (-2)
        When to start the season relative to RA 180 degrees away from the sun (hours)
    season_end_hour : `float` (2)
        When to consider a season ending, the RA relative to the sun + 180 degrees. (hours)
    night_max : float (365)
        Set value to zero after night_max is reached (days)
    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        footprint=None,
        n_obs=3,
        season=300,
        season_start_hour=-4.0,
        season_end_hour=2.0,
        night_max=365,
    ):
        super(NObsPerYearBasisFunction, self).__init__(nside=nside, filtername=filtername)
        self.footprint = footprint
        self.n_obs = n_obs
        self.season = season
        self.season_start_hour = (season_start_hour) * np.pi / 12.0  # To radians
        self.season_end_hour = season_end_hour * np.pi / 12.0  # To radians

        self.survey_features["last_n_mjds"] = features.LastNObsTimes(
            nside=nside, filtername=filtername, n_obs=n_obs
        )
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.out_footprint = np.where((footprint == 0) | np.isnan(footprint))
        self.night_max = night_max

    def _calc_value(self, conditions, indx=None):
        if conditions.night > self.night_max:
            return 0

        result = self.result.copy()
        behind_pix = np.where((conditions.mjd - self.survey_features["last_n_mjds"].feature[0]) > self.season)
        result[behind_pix] = 1

        # let's ramp up the weight depending on how far into the observing season the healpix is
        mid_season_ra = (conditions.sun_ra + np.pi) % (2.0 * np.pi)
        # relative RA
        relative_ra = (conditions.ra - mid_season_ra) % (2.0 * np.pi)
        relative_ra = (self.season_end_hour - relative_ra) % (2.0 * np.pi)
        # ok, now
        relative_ra[
            np.where(IntRounded(relative_ra) > IntRounded(self.season_end_hour - self.season_start_hour))
        ] = 0

        weight = relative_ra / (self.season_end_hour - self.season_start_hour)
        result *= weight

        # mask off anything outside the footprint
        result[self.out_footprint] = 0

        return result


class NGoodSeeingBasisFunction(BaseBasisFunction):
    """Try to get N "good seeing" images each observing season

    Parameters
    ----------
    seeing_fwhm_max : `float` (0.8)
        Value to consider as "good" threshold (arcsec).
    m5_penalty_max : `float` (0.5)
        The maximum depth loss that is considered acceptable (magnitudes)
    n_obs_desired : `int` (3)
        Number of good seeing observations to collect per season.
    mjd_start : float (1)
        The starting MJD.
    footprint : `np.array` (None)
        Only use area where footprint > 0. Should be a HEALpix map.
    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        seeing_fwhm_max=0.8,
        m5_penalty_max=0.5,
        n_obs_desired=3,
        footprint=None,
        mjd_start=1,
    ):
        super().__init__(nside=nside, filtername=filtername)
        self.seeing_fwhm_max = seeing_fwhm_max
        self.m5_penalty_max = m5_penalty_max
        self.n_obs_desired = n_obs_desired

        self.survey_features["N_good_seeing"] = features.NObservationsCurrentSeason(
            filtername=filtername,
            mjd_start=mjd_start,
            seeing_fwhm_max=seeing_fwhm_max,
            m5_penalty_max=m5_penalty_max,
            nside=nside,
        )
        self.result = np.zeros(hp.nside2npix(self.nside))
        if self.filtername is not None:
            self.dark_map = dark_sky(nside)[filtername]
        self.footprint = footprint

    def _calc_value(self, conditions, indx=None):
        result = 0
        # Need to update the feature to the current season
        self.survey_features["N_good_seeing"].season_update(conditions=conditions)

        m5_penalty = self.dark_map - conditions.m5_depth[self.filtername]
        potential_pixels = np.where(
            (m5_penalty <= self.m5_penalty_max)
            & (conditions.fwhm_eff[self.filtername] <= self.seeing_fwhm_max)
            & (self.survey_features["N_good_seeing"].feature < self.n_obs_desired)
            & (self.footprint > 0)
        )[0]

        if np.size(potential_pixels) > 0:
            result = self.result.copy()
            result[potential_pixels] = 1
        return result


class AvoidLongGapsBasisFunction(BaseBasisFunction):
    """Boost the reward on parts of the survey that haven't been observed for a
    while.
    """

    def __init__(
        self,
        filtername=None,
        nside=None,
        footprint=None,
        min_gap=4.0,
        max_gap=40.0,
        ha_limit=3.5,
    ):
        super(AvoidLongGapsBasisFunction, self).__init__(nside=nside, filtername=filtername)
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.filtername = filtername
        self.footprint = footprint
        self.ha_limit = 2.0 * np.pi * ha_limit / 24.0  # To radians
        self.survey_features = {}
        self.survey_features["last_observed"] = features.Last_observed(nside=nside, filtername=filtername)
        self.result = np.zeros(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        gap = conditions.mjd - self.survey_features["last_observed"].feature
        in_range = np.where((gap > self.min_gap) & (gap < self.max_gap) & (self.footprint > 0))
        result[in_range] = 1

        # mask out areas beyond the hour angle limit.
        out_ha = np.where((conditions.HA > self.ha_limit) & (conditions.HA < (2.0 * np.pi - self.ha_limit)))[
            0
        ]
        result[out_ha] = 0

        return result


class TargetMapBasisFunction(BaseBasisFunction):
    """Basis function that tracks number of observations and tries to match a
    specified spatial distribution

    Parameters
    ----------
    filtername: `str` ('r')
        The name of the filter for this target map.
    nside: `int` (default_nside)
        The healpix resolution.
    target_map : numpy array (None)
        A healpix map showing the ratio of observations desired for all points
        on the sky
    norm_factor : `float` (0.00010519)
        for converting target map to number of observations. Should be the area
        of the camera divided by the area of a healpixel divided by the sum of
        all your goal maps. Default value assumes LSST foV has 1.75 degree
        radius and the standard goal maps. If using mulitple filters, see
        rubin_sim.scheduler.utils.calc_norm_factor for a utility that computes
        norm_factor.
    out_of_bounds_val : `float` (-10.)
        Reward value to give regions where there are no observations requested
        (unitless).
    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        target_map=None,
        norm_factor=None,
        out_of_bounds_val=-10.0,
    ):
        super(TargetMapBasisFunction, self).__init__(nside=nside, filtername=filtername)

        if norm_factor is None:
            warnings.warn("No norm_factor set, use utils.calc_norm_factor if using multiple filters.")
            self.norm_factor = 0.00010519
        else:
            self.norm_factor = norm_factor

        self.survey_features = {}
        # Map of the number of observations in filter
        self.survey_features["n_obs"] = features.NObservations(filtername=filtername, nside=self.nside)
        # Count of all the observations
        self.survey_features["n_obs_count_all"] = features.NObsCount(filtername=None)
        if target_map is None:
            self.target_map = utils.generate_goal_map(filtername=filtername, nside=self.nside)
        else:
            self.target_map = target_map
        self.out_of_bounds_area = np.where(self.target_map == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.all_indx = np.arange(self.result.size)

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : `list` (None)
            Index values to compute, if None, full map is computed

        Returns
        -------
        Healpix reward map
        """
        result = self.result.copy()
        if indx is None:
            indx = self.all_indx

        # Find out how many observations we want now at those points
        goal_n = self.target_map[indx] * self.survey_features["n_obs_count_all"].feature * self.norm_factor

        result[indx] = goal_n - self.survey_features["n_obs"].feature[indx]
        result[self.out_of_bounds_area] = self.out_of_bounds_val

        return result


def az_rel_point(azs, point_az):
    az_rel_moon = (azs - point_az) % (2.0 * np.pi)
    if isinstance(azs, np.ndarray):
        over = np.where(az_rel_moon > np.pi)
        az_rel_moon[over] = 2.0 * np.pi - az_rel_moon[over]
    else:
        if az_rel_moon > np.pi:
            az_rel_moon = 2.0 * np.pi - az_rel_moon
    return az_rel_moon


class NObsHighAmBasisFunction(BaseBasisFunction):
    """Reward only reward/count observations at high airmass"""

    def __init__(
        self,
        nside=None,
        filtername="r",
        footprint=None,
        n_obs=3,
        season=300.0,
        am_limits=[1.5, 2.2],
        out_of_bounds_val=np.nan,
    ):
        super(NObsHighAmBasisFunction, self).__init__(nside=nside, filtername=filtername)
        self.footprint = footprint
        self.out_footprint = np.where((footprint == 0) | np.isnan(footprint))
        self.am_limits = am_limits
        self.season = season
        self.survey_features["last_n_mjds"] = features.Last_n_obs_times(
            nside=nside, filtername=filtername, n_obs=n_obs
        )

        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float) + out_of_bounds_val
        self.out_of_bounds_val = out_of_bounds_val

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : `np.array`
            An array with information about the input observation
        indx : `np.array`
            The indices of the healpix map that the observation overlaps with
        """

        # Only count the observations if they are at the airmass limits
        if (observation["airmass"] > np.min(self.am_limits)) & (
            observation["airmass"] < np.max(self.am_limits)
        ):
            for feature in self.survey_features:
                self.survey_features[feature].add_observation(observation, indx=indx)
            if self.update_on_newobs:
                self.recalc = True

    def check_feasibility(self, conditions):
        """If there is logic to decide if something is feasible (e.g., only if moon is down),
        it can be calculated here. Helps prevent full __call__ from being called more than needed.
        """
        result = True
        reward = self._calc_value(conditions)
        # If there are no non-NaN values, we're not feasible now
        if True not in np.isfinite(reward):
            result = False

        return result

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        behind_pix = np.where(
            (
                IntRounded(conditions.mjd - self.survey_features["last_n_mjds"].feature[0])
                > IntRounded(self.season)
            )
            & (IntRounded(conditions.airmass) > IntRounded(np.min(self.am_limits)))
            & (IntRounded(conditions.airmass) < IntRounded(np.max(self.am_limits)))
        )
        result[behind_pix] = 1
        result[self.out_footprint] = self.out_of_bounds_val

        # Update the last time we had an mjd
        self.mjd_last = conditions.mjd + 0
        self.recalc = False
        self.value = result

        return result


class EclipticBasisFunction(BaseBasisFunction):
    """Mark the area around the ecliptic"""

    def __init__(self, nside=None, distance_to_eclip=25.0):
        super(EclipticBasisFunction, self).__init__(nside=nside)
        self.distance_to_eclip = np.radians(distance_to_eclip)
        ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(self.nside)))
        self.result = np.zeros(ra.size)
        coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
        eclip_lat = coord.barycentrictrueecliptic.lat.radian
        good = np.where(np.abs(eclip_lat) < self.distance_to_eclip)
        self.result[good] += 1

    def __call__(self, conditions, indx=None):
        return self.result


class CadenceInSeasonBasisFunction(BaseBasisFunction):
    """Drive observations at least every N days in a given area

    Parameters
    ----------
    drive_map : np.array
        A HEALpix map with values of 1 where the cadence should be driven.
    filtername : `str`
        The filters that can count
    season_span : `float` (2.5)
        How long to consider a spot "in_season" (hours)
    cadence : `float` (2.5)
        How long to wait before activating the basis function (days)
    """

    def __init__(self, drive_map, filtername="griz", season_span=2.5, cadence=2.5, nside=None):
        super(CadenceInSeasonBasisFunction, self).__init__(nside=nside, filtername=filtername)
        self.drive_map = drive_map
        self.season_span = season_span / 12.0 * np.pi  # To radians
        self.cadence = cadence
        self.survey_features["last_observed"] = features.Last_observed(nside=nside, filtername=filtername)
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        ra_mid_season = (conditions.sunRA + np.pi) % (2.0 * np.pi)

        angle_to_mid_season = np.abs(conditions.ra - ra_mid_season)
        over = np.where(IntRounded(angle_to_mid_season) > IntRounded(np.pi))
        angle_to_mid_season[over] = 2.0 * np.pi - angle_to_mid_season[over]

        days_lag = conditions.mjd - self.survey_features["last_observed"].feature

        active_pix = np.where(
            (IntRounded(days_lag) >= IntRounded(self.cadence))
            & (self.drive_map == 1)
            & (IntRounded(angle_to_mid_season) < IntRounded(self.season_span))
        )
        result[active_pix] = 1.0

        return result


class SeasonCoverageBasisFunction(BaseBasisFunction):
    """Basis function to encourage N observations per observing season

    Parameters
    ----------
    footprint : healpix map (None)
        The footprint where one should demand coverage every season
    n_per_season : `int` (3)
        The number of observations to attempt to gather every season
    offset : healpix map
        The offset to apply when computing the current season over the sky. utils.create_season_offset
        is helpful for making this
    season_frac_start : `float` (0.5)
        Only start trying to gather observations after a season is fractionally this far over.
    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        footprint=None,
        n_per_season=3,
        offset=None,
        season_frac_start=0.5,
    ):
        super(SeasonCoverageBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.n_per_season = n_per_season
        self.footprint = footprint
        self.survey_features["n_obs_season"] = features.NObservationsCurrentSeason(
            filtername=filtername, nside=nside, offset=offset
        )
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.season_frac_start = season_frac_start
        self.offset = offset

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        season = utils.season_calc(conditions.night, offset=self.offset, floor=False)
        # Find the area that still needs observation
        feature = self.survey_features["n_obs_season"].feature
        not_enough = np.where(
            (self.footprint > 0)
            & (feature < self.n_per_season)
            & ((IntRounded(season - np.floor(season)) > IntRounded(self.season_frac_start)))
            & (season >= 0)
        )
        result[not_enough] = 1
        return result


class FootprintNvisBasisFunction(BaseBasisFunction):
    """Basis function to drive observations of a given footprint. Good to target of opportunity targets
    where one might want to observe a region 3 times.

    Parameters
    ----------
    footprint : `np.array`
        A healpix array (1 for desired, 0 for not desired) of the target footprint.
    nvis : `int` (1)
        The number of visits to try and gather
    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        footprint=None,
        nvis=1,
        out_of_bounds_val=np.nan,
    ):
        super(FootprintNvisBasisFunction, self).__init__(nside=nside, filtername=filtername)
        self.footprint = footprint
        self.nvis = nvis

        # Have a feature that tracks how many observations we have
        self.survey_features = {}
        # Map of the number of observations in filter
        self.survey_features["n_obs"] = features.n_observations(filtername=filtername, nside=self.nside)
        self.result = np.zeros(hp.nside2npix(nside))
        self.result.fill(out_of_bounds_val)
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        diff = IntRounded(self.footprint * self.nvis - self.survey_features["n_obs"].feature)

        result[np.where(diff > IntRounded(0))] = 1

        # Any spot where we have enough visits is out of bounds now.
        result[np.where(diff <= IntRounded(0))] = self.out_of_bounds_val
        return result


class ThirdObservationBasisFunction(BaseBasisFunction):
    """If there have been observations in two filters long enough ago, go for a third

    Parameters
    ----------
    gap_min : `float` (40.)
        The minimum time gap to consider a pixel good (minutes)
    gap_max : `float` (120)
        The maximum time to consider going for a pair (minutes)
    """

    def __init__(self, nside=32, filtername1="r", filtername2="z", gap_min=40.0, gap_max=120.0):
        super(ThirdObservationBasisFunction, self).__init__(nside=nside)
        self.filtername1 = filtername1
        self.filtername2 = filtername2
        self.gap_min = IntRounded(gap_min / 60.0 / 24.0)
        self.gap_max = IntRounded(gap_max / 60.0 / 24.0)

        self.survey_features = {}
        self.survey_features["last_obs_f1"] = features.Last_observed(filtername=filtername1, nside=nside)
        self.survey_features["last_obs_f2"] = features.Last_observed(filtername=filtername2, nside=nside)
        self.result = np.empty(hp.nside2npix(self.nside))
        self.result.fill(np.nan)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        d1 = IntRounded(conditions.mjd - self.survey_features["last_obs_f1"].feature)
        d2 = IntRounded(conditions.mjd - self.survey_features["last_obs_f2"].feature)
        good = np.where((d1 > self.gap_min) & (d1 < self.gap_max) & (d2 > self.gap_min) & (d2 < self.gap_max))
        result[good] = 1
        return result


class AvoidFastRevists(BaseBasisFunction):
    """Marks targets as unseen if they are in a specified time window in order to avoid fast revisits.

    Parameters
    ----------
    filtername: `str` ('r')
        The name of the filter for this target map.
    gap_min : `float` (25.)
        Minimum time for the gap (minutes).
    nside: `int` (default_nside)
        The healpix resolution.
    penalty_val : `float` (np.nan)
        The reward value to use for regions to penalize. Will be masked if set to np.nan (default).
    """

    def __init__(self, filtername="r", nside=None, gap_min=25.0, penalty_val=np.nan):
        super(AvoidFastRevists, self).__init__(nside=nside, filtername=filtername)

        self.filtername = filtername
        self.penalty_val = penalty_val

        self.gap_min = IntRounded(gap_min / 60.0 / 24.0)
        self.nside = nside

        self.survey_features = dict()
        self.survey_features["Last_observed"] = features.Last_observed(filtername=filtername, nside=nside)

    def _calc_value(self, conditions, indx=None):
        result = np.ones(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = IntRounded(conditions.mjd - self.survey_features["Last_observed"].feature[indx])
        bad = np.where(diff < self.gap_min)[0]
        result[indx[bad]] = self.penalty_val
        return result


class NearSunTwilightBasisFunction(BaseBasisFunction):
    """Reward looking into the twilight for NEOs at high airmass

    Parameters
    ----------
    max_airmass : `float` (2.5)
        The maximum airmass to try and observe (unitless)
    """

    def __init__(self, nside=None, max_airmass=2.5, penalty=np.nan):
        super(NearSunTwilightBasisFunction, self).__init__(nside=nside)
        self.max_airmass = IntRounded(max_airmass)
        self.result = np.empty(hp.nside2npix(self.nside))
        self.result.fill(penalty)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        good_pix = np.where(
            (conditions.airmass >= 1.0)
            & (IntRounded(conditions.airmass) < self.max_airmass)
            & (IntRounded(np.abs(conditions.az_to_sun)) < IntRounded(np.pi / 2.0))
        )
        result[good_pix] = conditions.airmass[good_pix] / self.max_airmass.initial
        return result


class VisitRepeatBasisFunction(BaseBasisFunction):
    """
    Basis function to reward re-visiting an area on the sky. Looking for Solar System objects.

    Parameters
    ----------
    gap_min : `float` (15.)
        Minimum time for the gap (minutes)
    gap_max : `float` (45.)
        Maximum time for a gap
    filtername : `str` ('r')
        The filter(s) to count with pairs
    npairs : `int` (1)
        The number of pairs of observations to attempt to gather
    """

    def __init__(self, gap_min=25.0, gap_max=45.0, filtername="r", nside=None, npairs=1):
        super(VisitRepeatBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.gap_min = IntRounded(gap_min / 60.0 / 24.0)
        self.gap_max = IntRounded(gap_max / 60.0 / 24.0)
        self.npairs = npairs

        self.survey_features = {}
        # Track the number of pairs that have been taken in a night
        self.survey_features["Pair_in_night"] = features.PairInNight(
            filtername=filtername, gap_min=gap_min, gap_max=gap_max, nside=nside
        )
        # When was it last observed
        # XXX--since this feature is also in Pair_in_night, I should just access that one!
        self.survey_features["Last_observed"] = features.LastObserved(filtername=filtername, nside=nside)

    def _calc_value(self, conditions, indx=None):
        result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = IntRounded(conditions.mjd - self.survey_features["Last_observed"].feature[indx])
        good = np.where(
            (diff >= self.gap_min)
            & (diff <= self.gap_max)
            & (self.survey_features["Pair_in_night"].feature[indx] < self.npairs)
        )[0]
        result[indx[good]] += 1.0
        return result


class M5DiffBasisFunction(BaseBasisFunction):
    """Basis function based on the 5-sigma depth.
    Look up the best depth a healpixel achieves, and compute
    the limiting depth difference given current conditions

    Parameters
    ----------
    fiducial_FWHMEff : `float`
        The zenith seeing to assume for "good" conditions
    """

    def __init__(self, filtername="r", fiducial_FWHMEff=0.7, nside=None):
        super().__init__(nside=nside, filtername=filtername)
        # The dark sky surface brightness values
        self.dark_sky = dark_sky(nside)[filtername]
        self.dark_map = None
        self.fiducial_FWHMEff = fiducial_FWHMEff
        self.filtername = filtername

    def _calc_value(self, conditions, indx=None):
        if self.dark_map is None:
            # compute the maximum altitude each HEALpix reaches,
            # this lets us determine the dark sky values with appropriate seeing
            # for each declination.
            min_z = np.abs(conditions.dec - conditions.site.latitude_rad)
            airmass_min = 1 / np.cos(min_z)
            airmass_min = np.where(airmass_min < 0, np.nan, airmass_min)
            sm = SeeingModel(filter_list=[self.filtername])
            fwhm_eff = sm(self.fiducial_FWHMEff, airmass_min)["fwhmEff"][0]
            self.dark_map = m5_flat_sed(
                self.filtername,
                musky=self.dark_sky,
                fwhm_eff=fwhm_eff,
                exp_time=30.0,
                airmass=airmass_min,
                nexp=1,
                tau_cloud=0,
            )

        # No way to get the sign on this right the first time.
        result = conditions.m5_depth[self.filtername] - self.dark_map
        return result


class StrictFilterBasisFunction(BaseBasisFunction):
    """Remove the bonus for staying in the same filter if certain conditions are met.

    If the moon rises/sets or twilight starts/ends, it makes a lot of sense to consider
    a filter change. This basis function rewards if it matches the current filter, the moon rises or sets,
    twilight starts or stops, or there has been a large gap since the last observation.

    Parameters
    ----------
    time_lag : `float` (10.)
        If there is a gap between observations longer than this, let the filter change (minutes)
    twi_change : `float` (-18.)
        The sun altitude to consider twilight starting/ending (degrees)
    note_free : `str` ('DD')
        No penalty for changing filters if the last observation note field includes string.
        Useful for giving a free filter change after deep drilling sequence
    """

    def __init__(self, time_lag=10.0, filtername="r", twi_change=-18.0, note_free="DD"):
        super(StrictFilterBasisFunction, self).__init__(filtername=filtername)

        self.time_lag = time_lag / 60.0 / 24.0  # Convert to days
        self.twi_change = np.radians(twi_change)

        self.survey_features = {}
        self.survey_features["Last_observation"] = features.LastObservation()
        self.note_free = note_free

    def _calc_value(self, conditions, **kwargs):
        # Did the moon set or rise since last observation?
        moon_changed = conditions.moon_alt * self.survey_features["Last_observation"].feature["moonAlt"] < 0

        # Are we already in the filter (or at start of night)?
        in_filter = (conditions.current_filter == self.filtername) | (conditions.current_filter is None)

        # Has enough time past?
        time_past = IntRounded(
            conditions.mjd - self.survey_features["Last_observation"].feature["mjd"]
        ) > IntRounded(self.time_lag)

        # Did twilight start/end?
        twi_changed = (conditions.sun_alt - self.twi_change) * (
            self.survey_features["Last_observation"].feature["sunAlt"] - self.twi_change
        ) < 0

        # Did we just finish a DD sequence
        was_dd = self.note_free in self.survey_features["Last_observation"].feature["note"]

        # Is the filter mounted?
        mounted = self.filtername in conditions.mounted_filters

        if (moon_changed | in_filter | time_past | twi_changed | was_dd) & mounted:
            result = 1.0
        else:
            result = 0.0

        return result


class GoalStrictFilterBasisFunction(BaseBasisFunction):
    """Remove the bonus for staying in the same filter if certain conditions are met.

    If the moon rises/sets or twilight starts/ends, it makes a lot of sense to consider
    a filter change. This basis function rewards if it matches the current filter, the moon rises or sets,
    twilight starts or stops, or there has been a large gap since the last observation.

    Parameters
    ----------
    time_lag_min : `float`
        Minimum time after a filter change for which a new filter change will receive zero reward, or
        be denied at all (see unseen_before_lag).
    time_lag_max : `float`
        Time after a filter change where the reward for changing filters achieve its maximum.
    time_lag_boost : `float`
        Time after a filter change to apply a boost on the reward.
    boost_gain : `float`
        A multiplier factor for the reward after time_lag_boost.
    unseen_before_lag : `bool`
        If True will make it impossible to switch filter before time_lag has passed.
    filtername : `str`
        The filter for which this basis function will be used.
    tag: `str` or None
        When using filter proportion use only regions with this tag to count for observations.
    twi_change : `float`
        Switch reward on when twilight changes.
    proportion : `float`
        The expected filter proportion distribution.
    aways_available: `bool`
        If this is true the basis function will aways be computed regardless of the feasibility.
        If False a more detailed feasibility check is performed.
        When set to False, it may speed up the computation process by avoiding to compute the
        reward functions paired with this bf, when observation is not feasible.

    """

    def __init__(
        self,
        time_lag_min=10.0,
        time_lag_max=30.0,
        time_lag_boost=60.0,
        boost_gain=2.0,
        unseen_before_lag=False,
        filtername="r",
        tag=None,
        twi_change=-18.0,
        proportion=1.0,
        aways_available=False,
    ):
        super(GoalStrictFilterBasisFunction, self).__init__(filtername=filtername)

        self.time_lag_min = time_lag_min / 60.0 / 24.0  # Convert to days
        self.time_lag_max = time_lag_max / 60.0 / 24.0  # Convert to days
        self.time_lag_boost = time_lag_boost / 60.0 / 24.0
        self.boost_gain = boost_gain
        self.unseen_before_lag = unseen_before_lag

        self.twi_change = np.radians(twi_change)
        self.proportion = proportion
        self.aways_available = aways_available

        self.survey_features = {}
        self.survey_features["Last_observation"] = features.Last_observation()
        self.survey_features["Last_filter_change"] = features.LastFilterChange()
        self.survey_features["n_obs_all"] = features.NObsCount(filtername=None)
        self.survey_features["n_obs"] = features.NObsCount(filtername=filtername, tag=tag)

    def filter_change_bonus(self, time):
        lag_min = self.time_lag_min
        lag_max = self.time_lag_max

        a = 1.0 / (lag_max - lag_min)
        b = -a * lag_min

        bonus = a * time + b
        # How far behind we are with respect to proportion?
        nobs = self.survey_features["n_obs"].feature
        nobs_all = self.survey_features["n_obs_all"].feature
        goal = self.proportion
        # need = 1. - nobs / nobs_all + goal if nobs_all > 0 else 1. + goal
        need = goal / nobs * nobs_all if nobs > 0 else 1.0
        # need /= goal
        if hasattr(time, "__iter__"):
            before_lag = np.where(time <= lag_min)
            bonus[before_lag] = -np.inf if self.unseen_before_lag else 0.0
            after_lag = np.where(time >= lag_max)
            bonus[after_lag] = 1.0 if time < self.time_lag_boost else self.boost_gain
        elif IntRounded(time) <= IntRounded(lag_min):
            return -np.inf if self.unseen_before_lag else 0.0
        elif IntRounded(time) >= IntRounded(lag_max):
            return 1.0 if IntRounded(time) < IntRounded(self.time_lag_boost) else self.boost_gain

        return bonus * need

    def check_feasibility(self, conditions):
        """
        This method makes a pre-check of the feasibility of this basis function.
        If a basis function returns False on the feasibility check, it won't computed at all.

        Returns
        -------
        feasibility : `bool`
        """

        # Make a quick check about the feasibility of this basis function.
        # If current filter is none, telescope is parked and we could, in principle, switch to any filter.
        # If this basis function computes reward for the current filter, then it is also feasible.
        # At last we check for an "aways_available" flag. Meaning, we force this basis function t
        # o be aways be computed.
        if (
            conditions.current_filter is None
            or conditions.current_filter == self.filtername
            or self.aways_available
        ):
            return True

        # If we arrive here,
        # we make some extra checks to make sure this bf is feasible and should be computed.

        # Did the moon set or rise since last observation?
        moon_changed = conditions.moon_alt * self.survey_features["Last_observation"].feature["moonAlt"] < 0

        # Are we already in the filter (or at start of night)?
        not_in_filter = conditions.current_filter != self.filtername

        # Has enough time past?
        lag = conditions.mjd - self.survey_features["Last_filter_change"].feature["mjd"]
        time_past = IntRounded(lag) > IntRounded(self.time_lag_min)

        # Did twilight start/end?
        twi_changed = (conditions.sun_alt - self.twi_change) * (
            self.survey_features["Last_observation"].feature["sun_alt"] - self.twi_change
        ) < 0

        # Did we just finish a DD sequence
        was_dd = self.survey_features["Last_observation"].feature["note"] == "DD"

        # Is the filter mounted?
        mounted = self.filtername in conditions.mounted_filters

        if (moon_changed | time_past | twi_changed | was_dd) & mounted & not_in_filter:
            return True
        else:
            return False

    def _calc_value(self, conditions, **kwargs):
        if conditions.current_filter is None:
            return 0.0  # no bonus if no filter is mounted
        # elif self.condition_features['Current_filter'].feature == self.filtername:
        #     return 0.  # no bonus if on the filter already

        # Did the moon set or rise since last observation?
        moon_changed = conditions.moon_alt * self.survey_features["Last_observation"].feature["moonAlt"] < 0

        # Are we already in the filter (or at start of night)?
        # not_in_filter = (self.condition_features['Current_filter'].feature != self.filtername)

        # Has enough time past?
        lag = conditions.mjd - self.survey_features["Last_filter_change"].feature["mjd"]
        time_past = lag > self.time_lag_min

        # Did twilight start/end?
        twi_changed = (conditions.sun_alt - self.twi_change) * (
            self.survey_features["Last_observation"].feature["sun_alt"] - self.twi_change
        ) < 0

        # Did we just finish a DD sequence
        was_dd = self.survey_features["Last_observation"].feature["note"] == "DD"

        # Is the filter mounted?
        mounted = self.filtername in conditions.mounted_filters

        if (moon_changed | time_past | twi_changed | was_dd) & mounted:
            result = self.filter_change_bonus(lag) if time_past else 0.0
        else:
            result = -100.0 if self.unseen_before_lag else 0.0

        return result


class FilterChangeBasisFunction(BaseBasisFunction):
    """Reward staying in the current filter."""

    def __init__(self, filtername="r"):
        super(FilterChangeBasisFunction, self).__init__(filtername=filtername)

    def _calc_value(self, conditions, **kwargs):
        if (conditions.current_filter == self.filtername) | (conditions.current_filter is None):
            result = 1.0
        else:
            result = 0.0
        return result


class SlewtimeBasisFunction(BaseBasisFunction):
    """Reward slews that take little time

    Parameters
    ----------
    max_time : `float` (135)
         The estimated maximum slewtime (seconds). Used to normalize so the basis function
         spans ~ -1-0 in reward units.
    """

    def __init__(self, max_time=135.0, filtername="r", nside=None):
        super(SlewtimeBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.maxtime = max_time
        self.nside = nside
        self.filtername = filtername

    def add_observation(self, observation, indx=None):
        # No tracking of observations in this basis function. Purely based on conditions.
        pass

    def _calc_value(self, conditions, indx=None):
        # If we are in a different filter, the FilterChangeBasisFunction will take it
        if conditions.current_filter != self.filtername:
            result = 0
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(conditions.slewtime) > 1:
                # Slewtime map can contain nans and/or infs - mask these with nans
                result = np.where(
                    np.isfinite(conditions.slewtime),
                    -conditions.slewtime / self.maxtime,
                    np.nan,
                )
            else:
                result = -conditions.slewtime / self.maxtime
        return result


class AggressiveSlewtimeBasisFunction(BaseBasisFunction):
    """Reward slews that take little time

    XXX--not sure how this is different from SlewtimeBasisFunction?
    Looks like it's checking the slewtime to the field position rather than the healpix maybe?
    """

    def __init__(self, max_time=135.0, order=1.0, hard_max=None, filtername="r", nside=None):
        super(AggressiveSlewtimeBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.maxtime = max_time
        self.hard_max = hard_max
        self.order = order
        self.result = np.zeros(hp.nside2npix(nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        # If we are in a different filter, the FilterChangeBasisFunction will take it
        if conditions.current_filter != self.filtername:
            result = 0.0
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(self.condition_features["slewtime"].feature) > 1:
                result = self.result.copy()
                result.fill(np.nan)

                good = np.where(np.bitwise_and(conditions.slewtime > 0.0, conditions.slewtime < self.maxtime))
                result[good] = ((self.maxtime - conditions.slewtime[good]) / self.maxtime) ** self.order
                if self.hard_max is not None:
                    not_so_good = np.where(conditions.slewtime > self.hard_max)
                    result[not_so_good] -= 10.0
                fields = np.unique(conditions.hp2fields[good])
                for field in fields:
                    hp_indx = np.where(conditions.hp2fields == field)
                    result[hp_indx] = np.min(result[hp_indx])
            else:
                result = (self.maxtime - conditions.slewtime) / self.maxtime
        return result


class SkybrightnessLimitBasisFunction(BaseBasisFunction):
    """Mask regions that are outside a sky brightness limit

    XXX--TODO:  This should probably go to the mask basis functions.

    Parameters
    ----------
    min : `float` (20.)
         The minimum sky brightness (mags).
    max : `float` (30.)
         The maximum sky brightness (mags).

    """

    def __init__(self, nside=None, filtername="r", sbmin=20.0, sbmax=30.0):
        super(SkybrightnessLimitBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.min = IntRounded(sbmin)
        self.max = IntRounded(sbmax)
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.result.fill(np.nan)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        good = np.where(
            np.bitwise_and(
                IntRounded(conditions.skybrightness[self.filtername]) > self.min,
                IntRounded(conditions.skybrightness[self.filtername]) < self.max,
            )
        )
        result[good] = 1.0

        return result


class CablewrapUnwrapBasisFunction(BaseBasisFunction):
    """
    Parameters
    ----------
    min_az : `float` (20.)
        The minimum azimuth to activate bf (degrees)
    max_az : `float` (82.)
        The maximum azimuth to activate bf (degrees)
    unwrap_until: `float` (90.)
        The window in which the bf is activated (degrees)
    """

    def __init__(
        self,
        nside=None,
        min_az=-270.0,
        max_az=270.0,
        min_alt=20.0,
        max_alt=82.0,
        activate_tol=20.0,
        delta_unwrap=1.2,
        unwrap_until=70.0,
        max_duration=30.0,
    ):
        super(CablewrapUnwrapBasisFunction, self).__init__(nside=nside)

        self.min_az = np.radians(min_az)
        self.max_az = np.radians(max_az)

        self.activate_tol = np.radians(activate_tol)
        self.delta_unwrap = np.radians(delta_unwrap)
        self.unwrap_until = np.radians(unwrap_until)

        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        # Convert to half-width for convienence
        self.nside = nside
        self.active = False
        self.unwrap_direction = 0.0  # either -1., 0., 1.
        self.max_duration = max_duration / 60.0 / 24.0  # Convert to days
        self.activation_time = None
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        current_abs_rad = np.radians(conditions.az)
        unseen = np.where(np.bitwise_or(conditions.alt < self.min_alt, conditions.alt > self.max_alt))
        result[unseen] = np.nan

        if (
            self.min_az + self.activate_tol < current_abs_rad < self.max_az - self.activate_tol
        ) and not self.active:
            return result
        elif self.active and self.unwrap_direction == 1 and current_abs_rad > self.min_az + self.unwrap_until:
            self.active = False
            self.unwrap_direction = 0.0
            self.activation_time = None
            return result
        elif (
            self.active and self.unwrap_direction == -1 and current_abs_rad < self.max_az - self.unwrap_until
        ):
            self.active = False
            self.unwrap_direction = 0.0
            self.activation_time = None
            return result
        elif self.activation_time is not None and conditions.mjd - self.activation_time > self.max_duration:
            self.active = False
            self.unwrap_direction = 0.0
            self.activation_time = None
            return result

        if not self.active:
            self.activation_time = conditions.mjd
            if current_abs_rad < 0.0:
                self.unwrap_direction = 1  # clock-wise unwrap
            else:
                self.unwrap_direction = -1  # counter-clock-wise unwrap

        self.active = True

        max_abs_rad = self.max_az
        min_abs_rad = self.min_az

        TWOPI = 2.0 * np.pi

        # Compute distance and accumulated az.
        norm_az_rad = np.divmod(conditions.az - min_abs_rad, TWOPI)[1] + min_abs_rad
        distance_rad = divmod(norm_az_rad - current_abs_rad, TWOPI)[1]
        get_shorter = np.where(distance_rad > np.pi)
        distance_rad[get_shorter] -= TWOPI
        accum_abs_rad = current_abs_rad + distance_rad

        # Compute wrap regions and fix distances
        mask_max = np.where(accum_abs_rad > max_abs_rad)
        distance_rad[mask_max] -= TWOPI
        mask_min = np.where(accum_abs_rad < min_abs_rad)
        distance_rad[mask_min] += TWOPI

        # Step-2: Repeat but now with compute reward to unwrap using specified delta_unwrap
        unwrap_current_abs_rad = current_abs_rad - (
            np.abs(self.delta_unwrap) if self.unwrap_direction > 0 else -np.abs(self.delta_unwrap)
        )
        unwrap_distance_rad = divmod(norm_az_rad - unwrap_current_abs_rad, TWOPI)[1]
        unwrap_get_shorter = np.where(unwrap_distance_rad > np.pi)
        unwrap_distance_rad[unwrap_get_shorter] -= TWOPI
        unwrap_distance_rad = np.abs(unwrap_distance_rad)

        if self.unwrap_direction < 0:
            mask = np.where(accum_abs_rad > unwrap_current_abs_rad)
        else:
            mask = np.where(accum_abs_rad < unwrap_current_abs_rad)

        # Finally build reward map
        result = (1.0 - unwrap_distance_rad / np.max(unwrap_distance_rad)) ** 2.0
        result[mask] = 0.0
        result[unseen] = np.nan

        return result


class CadenceEnhanceBasisFunction(BaseBasisFunction):
    """Drive a certain cadence

    Parameters
    ----------
    filtername : `str` ('gri')
        The filter(s) that should be grouped together
    supress_window : `list` of `float`
        The start and stop window for when observations should be repressed (days)
    apply_area : healpix map
        The area over which to try and drive the cadence. Good values as 1, no candece drive 0.
        Probably works as a bool array too."""

    def __init__(
        self,
        filtername="gri",
        nside=None,
        supress_window=[0, 1.8],
        supress_val=-0.5,
        enhance_window=[2.1, 3.2],
        enhance_val=1.0,
        apply_area=None,
    ):
        super(CadenceEnhanceBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.supress_window = np.sort(supress_window)
        self.supress_val = supress_val
        self.enhance_window = np.sort(enhance_window)
        self.enhance_val = enhance_val

        self.survey_features = {}
        self.survey_features["last_observed"] = features.Last_observed(filtername=filtername)

        self.empty = np.zeros(hp.nside2npix(self.nside), dtype=float)
        # No map, try to drive the whole area
        if apply_area is None:
            self.apply_indx = np.arange(self.empty.size)
        else:
            self.apply_indx = np.where(apply_area != 0)[0]

    def _calc_value(self, conditions, indx=None):
        # copy an empty array
        result = self.empty.copy()
        if indx is not None:
            ind = np.intersect1d(indx, self.apply_indx)
        else:
            ind = self.apply_indx
        if np.size(ind) == 0:
            result = 0
        else:
            mjd_diff = conditions.mjd - self.survey_features["last_observed"].feature[ind]
            to_supress = np.where(
                (IntRounded(mjd_diff) > IntRounded(self.supress_window[0]))
                & (IntRounded(mjd_diff) < IntRounded(self.supress_window[1]))
            )
            result[ind[to_supress]] = self.supress_val
            to_enhance = np.where(
                (IntRounded(mjd_diff) > IntRounded(self.enhance_window[0]))
                & (IntRounded(mjd_diff) < IntRounded(self.enhance_window[1]))
            )
            result[ind[to_enhance]] = self.enhance_val
        return result


# https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Trapezoid1D
def trapezoid(x, amplitude, x_0, width, slope):
    """One dimensional Trapezoid model function"""
    # Compute the four points where the trapezoid changes slope
    # x1 <= x2 <= x3 <= x4
    x2 = x_0 - width / 2.0
    x3 = x_0 + width / 2.0
    x1 = x2 - amplitude / slope
    x4 = x3 + amplitude / slope

    result = x * 0

    # Compute model values in pieces between the change points
    range_a = np.logical_and(x >= x1, x < x2)
    range_b = np.logical_and(x >= x2, x < x3)
    range_c = np.logical_and(x >= x3, x < x4)

    result[range_a] = slope * (x[range_a] - x1)
    result[range_b] = amplitude
    result[range_c] = slope * (x4 - x[range_c])

    return result


class CadenceEnhanceTrapezoidBasisFunction(BaseBasisFunction):
    """Drive a certain cadence, like CadenceEnhanceBasisFunction but with smooth transitions

    Parameters
    ----------
    filtername : `str` ('gri')
        The filter(s) that should be grouped together

    XXX--fill out doc string!
    """

    def __init__(
        self,
        filtername="gri",
        nside=None,
        delay_width=2,
        delay_slope=2.0,
        delay_peak=0,
        delay_amp=0.5,
        enhance_width=3.0,
        enhance_slope=2.0,
        enhance_peak=4.0,
        enhance_amp=1.0,
        apply_area=None,
        season_limit=None,
    ):
        super(CadenceEnhanceTrapezoidBasisFunction, self).__init__(nside=nside, filtername=filtername)

        self.delay_width = delay_width
        self.delay_slope = delay_slope
        self.delay_peak = delay_peak
        self.delay_amp = delay_amp
        self.enhance_width = enhance_width
        self.enhance_slope = enhance_slope
        self.enhance_peak = enhance_peak
        self.enhance_amp = enhance_amp

        self.season_limit = season_limit / 12 * np.pi  # To radians

        self.survey_features = {}
        self.survey_features["last_observed"] = features.Last_observed(filtername=filtername)

        self.empty = np.zeros(hp.nside2npix(self.nside), dtype=float)
        # No map, try to drive the whole area
        if apply_area is None:
            self.apply_indx = np.arange(self.empty.size)
        else:
            self.apply_indx = np.where(apply_area != 0)[0]

    def suppress_enhance(self, x):
        result = x * 0
        result -= trapezoid(x, self.delay_amp, self.delay_peak, self.delay_width, self.delay_slope)
        result += trapezoid(
            x,
            self.enhance_amp,
            self.enhance_peak,
            self.enhance_width,
            self.enhance_slope,
        )

        return result

    def season_calc(self, conditions):
        ra_mid_season = (conditions.sunRA + np.pi) % (2.0 * np.pi)
        angle_to_mid_season = np.abs(conditions.ra - ra_mid_season)
        over = np.where(IntRounded(angle_to_mid_season) > IntRounded(np.pi))
        angle_to_mid_season[over] = 2.0 * np.pi - angle_to_mid_season[over]

        return angle_to_mid_season

    def _calc_value(self, conditions, indx=None):
        # copy an empty array
        result = self.empty.copy()
        if indx is not None:
            ind = np.intersect1d(indx, self.apply_indx)
        else:
            ind = self.apply_indx
        if np.size(ind) == 0:
            result = 0
        else:
            mjd_diff = conditions.mjd - self.survey_features["last_observed"].feature[ind]
            result[ind] += self.suppress_enhance(mjd_diff)

        if self.season_limit is not None:
            radians_to_midseason = self.season_calc(conditions)
            outside_season = np.where(radians_to_midseason > self.season_limit)
            result[outside_season] = 0

        return result


class AzimuthBasisFunction(BaseBasisFunction):
    """Reward staying in the same azimuth range.
    Possibly better than using slewtime, especially when selecting a large area of sky.

    """

    def __init__(self, nside=None):
        super(AzimuthBasisFunction, self).__init__(nside=nside)

    def _calc_value(self, conditions, indx=None):
        az_dist = conditions.az - conditions.telAz
        az_dist = az_dist % (2.0 * np.pi)
        over = np.where(az_dist > np.pi)
        az_dist[over] = 2.0 * np.pi - az_dist[over]
        # Normalize sp between 0 and 1
        result = az_dist / np.pi
        return result


class AzModuloBasisFunction(BaseBasisFunction):
    """Try to replicate the Rothchild et al cadence forcing by only observing on limited az ranges per night.

    Parameters
    ----------
    az_limits : `list` of `float` pairs (None)
        The azimuth limits (degrees) to use.
    """

    def __init__(self, nside=None, az_limits=None, out_of_bounds_val=-1.0):
        super(AzModuloBasisFunction, self).__init__(nside=nside)
        self.result = np.ones(hp.nside2npix(self.nside))
        if az_limits is None:
            spread = 100.0 / 2.0
            self.az_limits = np.radians(
                [
                    [360 - spread, spread],
                    [90.0 - spread, 90.0 + spread],
                    [180.0 - spread, 180.0 + spread],
                ]
            )
        else:
            self.az_limits = np.radians(az_limits)
        self.mod_val = len(self.az_limits)
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        az_lim = self.az_limits[np.max(conditions.night) % self.mod_val]

        if az_lim[0] < az_lim[1]:
            out_pix = np.where(
                (IntRounded(conditions.az) < IntRounded(az_lim[0]))
                | (IntRounded(conditions.az) > IntRounded(az_lim[1]))
            )
        else:
            out_pix = np.where(
                (IntRounded(conditions.az) < IntRounded(az_lim[0]))
                | (IntRounded(conditions.az) > IntRounded(az_lim[1]))
            )[0]
        result[out_pix] = self.out_of_bounds_val
        return result


class DecModuloBasisFunction(BaseBasisFunction):
    """Emphasize dec bands on a nightly varying basis

    Parameters
    ----------
    dec_limits : `list` of `float` pairs (None)
        The azimuth limits (degrees) to use.
    """

    def __init__(self, nside=None, dec_limits=None, out_of_bounds_val=-1.0):
        super(DecModuloBasisFunction, self).__init__(nside=nside)

        npix = hp.nside2npix(nside)
        hpids = np.arange(npix)
        ra, dec = _hpid2_ra_dec(nside, hpids)

        self.results = []

        if dec_limits is None:
            self.dec_limits = np.radians([[-90.0, -32.8], [-32.8, -12.0], [-12.0, 35.0]])
        else:
            self.dec_limits = np.radians(dec_limits)
        self.mod_val = len(self.dec_limits)
        self.out_of_bounds_val = out_of_bounds_val

        for limits in self.dec_limits:
            good = np.where((dec >= limits[0]) & (dec < limits[1]))[0]
            tmp = np.zeros(npix)
            tmp[good] = 1
            self.results.append(tmp)

    def _calc_value(self, conditions, indx=None):
        night_index = np.max(conditions.night % self.mod_val)
        result = self.results[night_index]

        return result


class MapModuloBasisFunction(BaseBasisFunction):
    """Similar to Dec_modulo, but now use input masks

    Parameters
    ----------
    inmaps : `list` of hp arrays
    """

    def __init__(self, inmaps):
        nside = hp.npix2nside(np.size(inmaps[0]))
        super(MapModuloBasisFunction, self).__init__(nside=nside)
        self.maps = inmaps
        self.mod_val = len(inmaps)

    def _calc_value(self, conditions, indx=None):
        indx = np.max(conditions.night % self.mod_val)
        result = self.maps[indx]
        return result


class GoodSeeingBasisFunction(BaseBasisFunction):
    """Drive observations in good seeing conditions"""

    def __init__(
        self,
        nside=None,
        filtername="r",
        footprint=None,
        fwh_meff_limit=0.8,
        mag_diff=0.75,
    ):
        super(GoodSeeingBasisFunction, self).__init__(nside=nside)

        self.filtername = filtername
        self.fwh_meff_limit = IntRounded(fwh_meff_limit)
        if footprint is None:
            fp = utils.standard_goals(nside=nside)[filtername]
        else:
            fp = footprint
        self.out_of_bounds = np.where(fp == 0)[0]
        self.result = fp * 0

        self.mag_diff = IntRounded(mag_diff)
        self.survey_features = {}
        self.survey_features["coadd_depth_all"] = features.Coadded_depth(filtername=filtername, nside=nside)
        self.survey_features["coadd_depth_good"] = features.Coadded_depth(
            filtername=filtername, nside=nside, fwh_meff_limit=fwh_meff_limit
        )

    def _calc_value(self, conditions, **kwargs):
        # Seeing is "bad"
        if IntRounded(conditions.FWHMeff[self.filtername].min()) > self.fwh_meff_limit:
            return 0
        result = self.result.copy()

        diff = (
            self.survey_features["coadd_depth_all"].feature - self.survey_features["coadd_depth_good"].feature
        )
        # Where are there things we want to observe?
        good_pix = np.where(
            (IntRounded(diff) > self.mag_diff)
            & (IntRounded(conditions.FWHMeff[self.filtername]) <= self.fwh_meff_limit)
        )
        # Hm, should this scale by the mag differences? Probably.
        result[good_pix] = diff[good_pix]
        result[self.out_of_bounds] = 0

        return result


class TemplateGenerateBasisFunction(BaseBasisFunction):
    """Emphasize areas that have not been observed in a long time

    Parameters
    ----------
    day_gap : `float` (250.)
        How long to wait before boosting the reward (days)
    footprint : `np.array`(None)
        The indices of the healpixels to apply the boost to. Uses the default footprint if None
    """

    def __init__(self, nside=None, day_gap=250.0, filtername="r", footprint=None):
        super(TemplateGenerateBasisFunction, self).__init__(nside=nside)
        self.day_gap = day_gap
        self.filtername = filtername
        self.survey_features = {}
        self.survey_features["Last_observed"] = features.Last_observed(filtername=filtername)
        self.result = np.zeros(hp.nside2npix(self.nside))
        if footprint is None:
            fp = utils.standard_goals(nside=nside)[filtername]
        else:
            fp = footprint
        self.out_of_bounds = np.where(fp == 0)

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()
        overdue = np.where(
            (IntRounded(conditions.mjd - self.survey_features["Last_observed"].feature))
            > IntRounded(self.day_gap)
        )
        result[overdue] = 1
        result[self.out_of_bounds] = 0

        return result


class LimitRepeatBasisFunction(BaseBasisFunction):
    """Mask out pixels that haven't been observed in the night"""

    def __init__(self, nside=None, filtername="r", n_limit=2):
        super(LimitRepeatBasisFunction, self).__init__(nside=nside)
        self.filtername = filtername
        self.n_limit = n_limit
        self.survey_features = {}
        self.survey_features["n_obs"] = features.NObsNight(nside=nside, filtername=filtername)

        self.result = np.zeros(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()
        good_pix = np.where(self.survey_features["n_obs"].feature >= self.n_limit)[0]
        result[good_pix] = 1

        return result


class ObservedTwiceBasisFunction(BaseBasisFunction):
    """Mask out pixels that haven't been observed in the night"""

    def __init__(self, nside=None, filtername="r", n_obs_needed=2, n_obs_in_filt_needed=1):
        super(ObservedTwiceBasisFunction, self).__init__(nside=nside)
        self.n_obs_needed = n_obs_needed
        self.n_obs_in_filt_needed = n_obs_in_filt_needed
        self.filtername = filtername
        self.survey_features = {}
        self.survey_features["n_obs_infilt"] = features.NObsNight(nside=nside, filtername=filtername)
        self.survey_features["n_obs_all"] = features.NObsNight(nside=nside, filtername="")

        self.result = np.zeros(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()
        good_pix = np.where(
            (self.survey_features["n_obs_infilt"].feature >= self.n_obs_in_filt_needed)
            & (self.survey_features["n_obs_all"].feature >= self.n_obs_needed)
        )[0]
        result[good_pix] = 1

        return result


class VisitGap(BaseBasisFunction):
    """Basis function to create a visit gap based on the survey note field.

    Parameters
    ----------
    note : str
        Value of the observation "note" field to be masked.
    filter_names : list [str], optional
        List of filter names that will be considered when evaluating if the gap
        has passed.
    gap_min : float (optional)
        Time gap (default=25, in minutes).
    penalty_val : float or np.nan
        Value of the penalty to apply (default is np.nan).

    Notes
    -----
    When a list of filters is provided, all filters must be observed before the
    gap requirement will be activated, and once activated, only observations in
    these filters will be evaluated in context of whether the last observation
    was at least gap in the past.
    """

    def __init__(self, note, filter_names=None, gap_min=25.0, penalty_val=np.nan):
        super().__init__()
        self.penalty_val = penalty_val

        self.gap = gap_min / 60.0 / 24.0
        self.filter_names = filter_names

        self.survey_features = dict()
        if self.filter_names is not None:
            for filtername in self.filter_names:
                self.survey_features[f"NoteLastObserved::{filtername}"] = features.NoteLastObserved(
                    note=note, filtername=filtername
                )
        else:
            self.survey_features["NoteLastObserved"] = features.NoteLastObserved(note=note)

    def check_feasibility(self, conditions):
        notes_last_observed = [last_observed.feature for last_observed in self.survey_features.values()]

        if any([last_observed is None for last_observed in notes_last_observed]):
            return True

        after_gap = [conditions.mjd - last_observed > self.gap for last_observed in notes_last_observed]

        return all(after_gap)

    def _calc_value(self, conditions, indx=None):
        return 1.0 if self.check_feasibility(conditions) else self.penalty_val


class AvoidDirectWind(BaseBasisFunction):
    """Basis function to avoid direct wind.

    Parameters
    ----------
    wind_speed_maximum : float
        Wind speed to mark regions as unobservable (in m/s).
    """

    def __init__(self, wind_speed_maximum=20.0, nside=None):
        super().__init__(nside=nside)

        self.wind_speed_maximum = wind_speed_maximum

    def _calc_value(self, conditions, indx=None):
        reward_map = np.zeros(hp.nside2npix(self.nside))

        if conditions.wind_speed is None or conditions.wind_direction is None:
            return reward_map

        wind_pressure = conditions.wind_speed * np.cos(conditions.az - conditions.wind_direction)

        reward_map -= wind_pressure**2.0

        mask = wind_pressure > self.wind_speed_maximum

        reward_map[mask] = np.nan

        return reward_map


class BalanceVisits(BaseBasisFunction):
    """Balance visits across multiple surveys.

    Parameters
    ----------
    nobs_reference : int
        Expected number of observations across all interested surveys.
    note_survey : str
        Note value for the current survey.
    note_interest : str
        Substring with the name of interested surveys to be accounted.
    nside : int
        Healpix map resolution.

    Notes
    -----
    This basis function is designed to balance the reward of a group of
    surveys, such that the group get a reward boost based on the required
    collective number of observations.

    For example, if you have 3 surveys (e.g. SURVEY_A_REGION_1,
    SURVEY_A_REGION_2, SURVEY_A_REGION_3), when one of them is observed once
    (SURVEY_A_REGION_1) they all get a small reward boost proportianal to the
    collective number of observations (`nobs_reference`). Further observations
    of SURVEY_A_REGION_1 would now cause the other surveys to gain a reward
    boost in relative to it.
    """

    def __init__(self, nobs_reference, note_survey, note_interest, nside=None):
        super().__init__(nside=nside)

        self.nobs_reference = nobs_reference

        self.survey_features = {}
        self.survey_features["n_obs_survey"] = features.NObsSurvey(note=note_survey)
        self.survey_features["n_obs_survey_interest"] = features.NObsSurvey(note=note_interest)

    def _calc_value(self, conditions, indx=None):
        return (1 + np.floor(self.survey_features["n_obs_survey_interest"].feature / self.nobs_reference)) / (
            self.survey_features["n_obs_survey"].feature
            if self.survey_features["n_obs_survey"].feature > 0
            else 1
        )


class RewardNObsSequence(BaseBasisFunction):
    """Reward taking a sequence of observations.

    Parameters
    ----------
    n_obs_survey : int
        Number of observations to reward.
    note_survey : str
        The value of the observation note, to take into account.
    nside : int, optional
        Healpix map resolution (ignored).

    Notes
    -----
    This basis function is usefull when a survey is composed of more than one
    observation (e.g. in different filters) and one wants to make sure they are
    all taken together.
    """

    def __init__(self, n_obs_survey, note_survey, nside=None):
        super().__init__(nside=nside)

        self.n_obs_survey = n_obs_survey

        self.survey_features = {}
        self.survey_features["n_obs_survey"] = features.NObsSurvey(note=note_survey)

    def _calc_value(self, conditions, indx=None):
        return self.survey_features["n_obs_survey"].feature % self.n_obs_survey
