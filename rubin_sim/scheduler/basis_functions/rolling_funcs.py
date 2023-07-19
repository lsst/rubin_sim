__all__ = (
    "TargetMapModuloBasisFunction",
    "FootprintBasisFunction",
    "FootprintRollingBasisFunction",
)

import warnings

import healpy as hp
import matplotlib.pylab as plt
import numpy as np

from rubin_sim.scheduler import features, utils
from rubin_sim.scheduler.basis_functions import BaseBasisFunction


class FootprintBasisFunction(BaseBasisFunction):
    """Basis function that tries to maintain a uniformly covered footprint

    Parameters
    ----------
    filtername : str ('r')
        The filter for this footprint
    footprint : rubin_sim.scheduler.utils.Footprint object
        The desired footprint.
    all_footprints_sum : float (None)
        If using multiple filters, the sum of all the footprints. Needed to make sure basis functions are
        normalized properly across all fitlers.
    out_of_bounds_val : float (-10)
        The value to set the basis function for regions that are not in the footprint (default -10, np.nan is
        another good value to use)

    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        footprint=None,
        out_of_bounds_val=-10.0,
        window_size=6.0,
    ):
        super(FootprintBasisFunction, self).__init__(nside=nside, filtername=filtername)
        self.footprint = footprint

        self.survey_features = {}
        # All the observations in all filters
        self.survey_features["N_obs_all"] = features.NObservations(nside=nside, filtername=None)
        self.survey_features["N_obs"] = features.NObservations(nside=nside, filtername=filtername)

        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(footprint.get_footprint(self.filtername) <= 0)[0]
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):
        # Find out what the footprint object thinks we should have been observed
        desired_footprint_normed = self.footprint(conditions.mjd)[self.filtername]

        # Compute how many observations we should have on the sky
        desired = desired_footprint_normed * np.sum(self.survey_features["N_obs_all"].feature)
        result = desired - self.survey_features["N_obs"].feature
        result[self.out_of_bounds_area] = self.out_of_bounds_val
        return result


class FootprintRollingBasisFunction(BaseBasisFunction):
    """Let's get the rolling really right.

    Parameters
    ----------
    footprints : list of np.array
        List of HEALpix arrays. The footprints should all have matching sums and have the same nside.
    all_footprints_sum : float
        The sum of footprints over all filters.
    all_rolling_sum : float
        The sum (over all filters) of the region of the maps that changs.
    season_modulo : int (2)
        The modulo to pass to utils.season_calc.
    season_length : float (365.25)
        How long a season should be (days).
    max_season : int (None)
        If set, the season calc will return -1 for values greater than max_season
    day_offset : np.array (None)
        Offset to pass to utils.season_calc (days).

    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        footprints=None,
        all_footprints_sum=None,
        all_rolling_sum=None,
        out_of_bounds_val=-10,
        season_modulo=2,
        season_length=365.25,
        max_season=None,
        day_offset=None,
        window_size=6.0,
    ):
        super(FootprintRollingBasisFunction, self).__init__(nside=nside, filtername=filtername)

        # OK, going to find the parts of the map that are the same everywhere, and compute the
        # basis function the same as usual for those.
        same_footprint = np.ones(footprints[0].size, dtype=bool)
        for footprint in footprints[0:-1]:
            same_footprint = same_footprint & (footprint == footprints[-1])

        sum_footprints = footprints[0] * 0
        for footprint in footprints:
            sum_footprints += footprint
        self.constant_footprint_indx = np.where((same_footprint == True) & (sum_footprints > 0))[0]
        self.rolling_footprint_indx = np.where((same_footprint == False) & (sum_footprints > 0))[0]

        self.season_modulo = season_modulo
        self.season_length = season_length
        self.max_season = max_season
        self.day_offset = day_offset
        self.footprints = footprints

        self.max_day_offset = np.max(self.day_offset)

        self.all_footprints_sum = all_footprints_sum
        self.all_rolling_sum = all_rolling_sum

        self.survey_features = {}
        # Set a season for -1 (for before rolling or after max_season)
        # All the observations in the given filter
        self.survey_features["N_obs_%i" % -1] = features.N_observations(nside=nside, filtername=filtername)
        # All the observations in all filters
        self.survey_features["N_obs_all_%i" % -1] = features.N_observations(nside=nside, filtername=None)

        for i, temp in enumerate(footprints[0:-1]):
            # Observation in a season, in filtername
            self.survey_features["N_obs_%i" % i] = features.N_observations_season(
                i,
                filtername=filtername,
                nside=self.nside,
                modulo=season_modulo,
                offset=day_offset,
                max_season=max_season,
                season_length=season_length,
            )
            # Count of all the observations taken in a season
            self.survey_features["N_obs_all_%i" % i] = features.N_observations_season(
                i,
                filtername=None,
                modulo=season_modulo,
                offset=day_offset,
                nside=self.nside,
                max_season=max_season,
                season_length=season_length,
            )

        # Now I need to track the observations taken in each season.
        self.out_of_bounds_area = np.where(footprint <= 0)[0]
        self.out_of_bounds_val = out_of_bounds_val

        self.result = np.zeros(hp.nside2npix(nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        # Compute what season it is at each pixel
        seasons = conditions.season(
            modulo=self.season_modulo,
            max_season=self.max_season,
            season_length=self.season_length,
        )

        # Update the coverage of the sky so far
        # If RA to sun is zero, we are at phase np.pi/2.
        coverage_map_phase = (conditions.ra - conditions.sun_RA_start + np.pi / 2.0) % (2.0 * np.pi)
        zero = utils.step_line(0.0, 1.0, 365.25, phase=coverage_map_phase * 365.25 / 2 / np.pi)
        t_elapsed = conditions.mjd - conditions.mjd_start
        norm_coverage_raw = utils.step_line(
            t_elapsed, 1.0, 365.25, phase=coverage_map_phase * 365.25 / 2 / np.pi
        )
        norm_coverage_raw -= zero

        norm_coverage = norm_coverage_raw / np.max(norm_coverage_raw)
        norm_footprint = self.footprints[-1] * norm_coverage

        # Compute the constant parts of the footprint like before
        desired = (
            norm_footprint / self.all_footprints_sum * np.sum(self.survey_features["N_obs_all_-1"].feature)
        )
        result[self.constant_footprint_indx] = (
            desired[self.constant_footprint_indx]
            - self.survey_features["N_obs_-1"].feature[self.constant_footprint_indx]
        )

        # Now for the rolling sections
        norm_coverage = 0
        ns = np.floor((t_elapsed + self.max_day_offset) / 365.25)
        if np.max(ns) > -1:
            ns_on = ns / self.season_modulo
            ns_off = ns - ns_on
            norm_coverage = norm_coverage_raw - ns_off
            norm_coverage = norm_coverage / np.max(norm_coverage)

        for season in np.unique(seasons[self.rolling_footprint_indx]):
            if season == -1:
                nf = norm_footprint
            else:
                nf = self.footprints[season] * norm_coverage
            season_indx = np.where(seasons[self.rolling_footprint_indx] == season)[0]
            desired = (
                nf[self.rolling_footprint_indx[season_indx]]
                / self.all_rolling_sum
                * np.sum(self.survey_features["N_obs_all_%i" % season].feature[self.rolling_footprint_indx])
            )
            result[self.rolling_footprint_indx[season_indx]] = (
                desired
                - self.survey_features["N_obs_%i" % season].feature[self.rolling_footprint_indx][season_indx]
            )

        result[self.out_of_bounds_area] = self.out_of_bounds_val
        return result


class TargetMapModuloBasisFunction(BaseBasisFunction):
    """Basis function that tracks number of observations and tries to match a specified spatial distribution
    can enter multiple maps that will be used at different times in the survey

    Parameters
    ----------
    day_offset : np.array
        Healpix map that has the offset to be applied to each pixel when computing what season it is on.
    filtername : (string 'r')
        The name of the filter for this target map.
    nside: int (default_nside)
        The healpix resolution.
    target_maps : list of numpy array (None)
        healpix maps showing the ratio of observations desired for all points on the sky. Last map will be used
        for season -1. Probably shouldn't support going to season less than -1.
    norm_factor : float (0.00010519)
        for converting target map to number of observations. Should be the area of the camera
        divided by the area of a healpixel divided by the sum of all your goal maps. Default
        value assumes LSST foV has 1.75 degree radius and the standard goal maps. If using
        mulitple filters, see rubin_sim.scheduler.utils.calc_norm_factor for a utility
        that computes norm_factor.
    out_of_bounds_val : float (-10.)
        Reward value to give regions where there are no observations requested (unitless).
    season_modulo : int (2)
        The value to modulate the season by (years).
    max_season : int (None)
        For seasons higher than this value (pre-modulo), the final target map is used.

    """

    def __init__(
        self,
        day_offset=None,
        filtername="r",
        nside=None,
        target_maps=None,
        norm_factor=None,
        out_of_bounds_val=-10.0,
        season_modulo=2,
        max_season=None,
        season_length=365.25,
    ):
        super(TargetMapModuloBasisFunction, self).__init__(nside=nside, filtername=filtername)

        if norm_factor is None:
            warnings.warn("No norm_factor set, use utils.calc_norm_factor if using multiple filters.")
            self.norm_factor = 0.00010519
        else:
            self.norm_factor = norm_factor

        self.survey_features = {}
        # Map of the number of observations in filter

        for i, temp in enumerate(target_maps[0:-1]):
            self.survey_features["N_obs_%i" % i] = features.N_observations_season(
                i,
                filtername=filtername,
                nside=self.nside,
                modulo=season_modulo,
                offset=day_offset,
                max_season=max_season,
                season_length=season_length,
            )
            # Count of all the observations taken in a season
            self.survey_features["N_obs_count_all_%i" % i] = features.N_obs_count_season(
                i,
                filtername=None,
                season_modulo=season_modulo,
                offset=day_offset,
                nside=self.nside,
                max_season=max_season,
                season_length=season_length,
            )
        # Set the final one to be -1
        self.survey_features["N_obs_%i" % -1] = features.N_observations_season(
            -1,
            filtername=filtername,
            nside=self.nside,
            modulo=season_modulo,
            offset=day_offset,
            max_season=max_season,
            season_length=season_length,
        )
        self.survey_features["N_obs_count_all_%i" % -1] = features.N_obs_count_season(
            -1,
            filtername=None,
            season_modulo=season_modulo,
            offset=day_offset,
            nside=self.nside,
            max_season=max_season,
            season_length=season_length,
        )
        if target_maps is None:
            self.target_maps = utils.generate_goal_map(filtername=filtername, nside=self.nside)
        else:
            self.target_maps = target_maps
        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(self.target_maps[0] == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.all_indx = np.arange(self.result.size)

        # For computing what day each healpix is on
        if day_offset is None:
            self.day_offset = np.zeros(hp.nside2npix(self.nside), dtype=float)
        else:
            self.day_offset = day_offset

        self.season_modulo = season_modulo
        self.max_season = max_season
        self.season_length = season_length

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix reward map
        """

        result = self.result.copy()
        if indx is None:
            indx = self.all_indx

        # Compute what season it is at each pixel
        seasons = utils.season_calc(
            conditions.night,
            offset=self.day_offset,
            modulo=self.season_modulo,
            max_season=self.max_season,
            season_length=self.season_length,
        )

        composite_target = self.result.copy()[indx]
        composite_nobs = self.result.copy()[indx]

        composite_goal_n = self.result.copy()[indx]

        for season in np.unique(seasons):
            season_indx = np.where(seasons == season)[0]
            composite_target[season_indx] = self.target_maps[season][season_indx]
            composite_nobs[season_indx] = self.survey_features["N_obs_%i" % season].feature[season_indx]
            composite_goal_n[season_indx] = (
                composite_target[season_indx]
                * self.survey_features["N_obs_count_all_%i" % season].feature
                * self.norm_factor
            )

        result[indx] = composite_goal_n - composite_nobs[indx]
        result[self.out_of_bounds_area] = self.out_of_bounds_val

        return result
