__all__ = ("GreedySurvey", "BlobSurvey")

import warnings
from copy import copy

import healpy as hp
import matplotlib.pylab as plt
import numpy as np

from rubin_sim.scheduler.surveys import BaseMarkovSurvey
from rubin_sim.scheduler.utils import (
    IntRounded,
    empty_observation,
    gnomonic_project_toxy,
    int_binned_stat,
    set_default_nside,
    tsp_convex,
)
from rubin_sim.utils import _angular_separation, _approx_ra_dec2_alt_az, _hpid2_ra_dec, hp_grow_argsort


class GreedySurvey(BaseMarkovSurvey):
    """
    Select pointings in a greedy way using a Markov Decision Process.
    """

    def __init__(
        self,
        basis_functions,
        basis_weights,
        filtername="r",
        block_size=1,
        smoothing_kernel=None,
        nside=None,
        dither=True,
        seed=42,
        ignore_obs=None,
        survey_name=None,
        nexp=2,
        exptime=30.0,
        detailers=None,
        camera="LSST",
        area_required=None,
        fields=None,
    ):
        extra_features = {}

        if survey_name == "":
            survey_name = f"Greedy {filtername}"

        super(GreedySurvey, self).__init__(
            basis_functions=basis_functions,
            basis_weights=basis_weights,
            extra_features=extra_features,
            smoothing_kernel=smoothing_kernel,
            ignore_obs=ignore_obs,
            nside=nside,
            survey_name=survey_name,
            dither=dither,
            detailers=detailers,
            camera=camera,
            area_required=area_required,
            fields=fields,
        )
        self.filtername = filtername
        self.block_size = block_size
        self.nexp = nexp
        self.exptime = exptime

    def generate_observations_rough(self, conditions):
        """
        Just point at the highest reward healpix
        """
        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields(conditions)
            self.night = copy(conditions.night)

        # Let's find the best N from the fields
        order = np.argsort(self.reward)[::-1]
        # Crop off any NaNs
        order = order[~np.isnan(self.reward[order])]

        iter = 0
        while True:
            best_hp = order[iter * self.block_size : (iter + 1) * self.block_size]
            best_fields = np.unique(self.hp2fields[best_hp])
            observations = []
            for field in best_fields:
                obs = empty_observation()
                obs["RA"] = self.fields["RA"][field]
                obs["dec"] = self.fields["dec"][field]
                obs["rotSkyPos"] = 0.0
                obs["filter"] = self.filtername
                obs["nexp"] = self.nexp
                obs["exptime"] = self.exptime
                obs["field_id"] = -1
                obs["note"] = self.survey_name

                observations.append(obs)
                break
            iter += 1
            if len(observations) > 0 or (iter + 2) * self.block_size > len(order):
                break
        return observations


class BlobSurvey(GreedySurvey):
    """Select observations in large, mostly contiguous, blobs.

    Parameters
    ----------
    filtername1 : `str`
        The filter to observe in.
    filtername2 : `str`
        The filter to pair with the first observation. If set to None, no pair
        will be observed.
    slew_approx : `float`
        The approximate slewtime between neerby fields (seconds). Used to calculate
        how many observations can be taken in the desired time block.
    nexp : `int`
        The number of exposures to take in a visit.
    exp_dict : `dict`
        If set, should have keys of filtername and values of ints that are the nuber of exposures to take
        per visit. For estimating block time, nexp is still used.
    filter_change_approx : `float`
         The approximate time it takes to change filters (seconds).
    ideal_pair_time : `float`
        The ideal time gap wanted between observations to the same pointing (minutes)
    min_pair_time : `float`
        The minimum acceptable pair time (minutes)
    flush_time : `float`
        The time past the final expected exposure to flush the queue. Keeps observations
        from lingering past when they should be executed. (minutes)
    twilight_scale : `bool`
        Scale the block size to fill up to twilight. Set to False if running in twilight
    in_twilight : `bool`
        Scale the block size to stay within twilight time.
    check_scheduled : `bool`
        Check if there are scheduled observations and scale blob size to match
    min_area : `float`
        If set, demand the reward function have an area of so many square degrees before executing
    grow_blob : `bool`
        If True, try to grow the blob from the global maximum. Otherwise, just use a simple sort.
        Simple sort will not constrain the blob to be contiguous.
    max_radius_peak : `float`
        The maximum radius to demand things be within the maximum of the reward function. (degrees)
        Note that traveling salesman solver can have rare failures if this is set too large
        (probably issue with projection effects or something).
    """

    def __init__(
        self,
        basis_functions,
        basis_weights,
        filtername1="r",
        filtername2="g",
        slew_approx=7.5,
        filter_change_approx=140.0,
        read_approx=2.0,
        exptime=30.0,
        nexp=2,
        nexp_dict=None,
        ideal_pair_time=22.0,
        min_pair_time=15.0,
        flush_time=30.0,
        smoothing_kernel=None,
        nside=None,
        dither=True,
        seed=42,
        ignore_obs=None,
        survey_note="blob",
        detailers=None,
        camera="LSST",
        twilight_scale=True,
        in_twilight=False,
        check_scheduled=True,
        min_area=None,
        grow_blob=True,
        area_required=None,
        max_radius_peak=40.0,
        fields=None,
        survey_name="",
        **kwargs,
    ):
        self.filtername1 = filtername1
        self.filtername2 = filtername2

        super(BlobSurvey, self).__init__(
            basis_functions=basis_functions,
            basis_weights=basis_weights,
            filtername=None,
            block_size=0,
            smoothing_kernel=smoothing_kernel,
            dither=dither,
            seed=seed,
            ignore_obs=ignore_obs,
            nside=nside,
            detailers=detailers,
            camera=camera,
            area_required=area_required,
            fields=fields,
        )
        self.flush_time = flush_time / 60.0 / 24.0  # convert to days
        self.nexp = nexp
        self.nexp_dict = nexp_dict
        self.exptime = exptime
        self.slew_approx = slew_approx
        self.read_approx = read_approx
        self.hpids = np.arange(hp.nside2npix(self.nside))
        self.twilight_scale = twilight_scale
        self.in_twilight = in_twilight
        self.grow_blob = grow_blob
        self.max_radius_peak = np.radians(max_radius_peak)

        if self.twilight_scale & self.in_twilight:
            warnings.warn("Both twilight_scale and in_twilight are set to True. That is probably wrong.")

        self.min_area = min_area
        self.check_scheduled = check_scheduled
        # If we are taking pairs in same filter, no need to add filter change time.
        if filtername1 == filtername2:
            filter_change_approx = 0
        # Compute the minimum time needed to observe a blob (or observe, then repeat.)
        if filtername2 is not None:
            self.time_needed = (
                (min_pair_time * 60.0 * 2.0 + exptime + read_approx + filter_change_approx) / 24.0 / 3600.0
            )  # Days
        else:
            self.time_needed = (min_pair_time * 60.0 + exptime + read_approx) / 24.0 / 3600.0  # Days
        self.filter_set = set(filtername1)
        if filtername2 is None:
            self.filter2_set = self.filter_set
        else:
            self.filter2_set = set(filtername2)

        self.ra, self.dec = _hpid2_ra_dec(self.nside, self.hpids)

        self.survey_note = survey_note
        self.counter = 1  # start at 1, because 0 is default in empty observation
        self.min_pair_time = min_pair_time
        self.ideal_pair_time = ideal_pair_time

        self.pixarea = hp.nside2pixarea(self.nside, degrees=True)

        # If we are only using one filter, this could be useful
        if (self.filtername2 is None) | (self.filtername1 == self.filtername2):
            self.filtername = self.filtername1

    def _generate_survey_name(self):
        self.survey_name = f"Blob survey {self.filtername1}"
        if self.filtername2 is None:
            self.survey_name += f"_{self.filtername1}"
        else:
            self.survey_name += f"_{self.filtername2}"

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions.
        """
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result

        # If we need to check that the reward function has enough area available
        if self.min_area is not None:
            reward = 0
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions)
                reward += basis_value * weight
            max_reward_indx = np.min(np.where(reward == np.nanmax(reward)))
            distances = _angular_separation(
                self.ra, self.dec, self.ra[max_reward_indx], self.dec[max_reward_indx]
            )
            valid_pix = np.where((np.isnan(reward) == False) & (distances < self.max_radius_peak))[0]
            if np.size(valid_pix) * self.pixarea < self.min_area:
                result = False
        return result

    def _set_block_size(self, conditions):
        """
        Update the block size if it's getting near a break point.
        """

        # If we are trying to get things done before twilight
        if self.twilight_scale:
            available_time = conditions.sun_n18_rising - conditions.mjd
            available_time *= 24.0 * 60.0  # to minutes
            n_ideal_blocks = available_time / self.ideal_pair_time
        else:
            n_ideal_blocks = 4

        # If we are trying to get things done before a scheduled simulation
        if self.check_scheduled:
            if len(conditions.scheduled_observations) > 0:
                available_time = np.min(conditions.scheduled_observations) - conditions.mjd
                available_time *= 24.0 * 60.0  # to minutes
                n_blocks = available_time / self.ideal_pair_time
                if n_blocks < n_ideal_blocks:
                    n_ideal_blocks = n_blocks

        # If we are trying to complete before twilight ends or the night ends
        if self.in_twilight:
            at1 = conditions.sun_n12_rising - conditions.mjd
            at2 = conditions.sun_n18_setting - conditions.mjd
            times = np.array([at1, at2])
            times = times[np.where(times > 0)]
            available_time = np.min(times)
            available_time *= 24.0 * 60.0  # to minutes
            n_blocks = available_time / self.ideal_pair_time
            if n_blocks < n_ideal_blocks:
                n_ideal_blocks = n_blocks

        if n_ideal_blocks >= 3:
            self.nvisit_block = int(
                np.floor(
                    self.ideal_pair_time
                    * 60.0
                    / (self.slew_approx + self.exptime + self.read_approx * (self.nexp - 1))
                )
            )
        else:
            # Now we can stretch or contract the block size to allocate the remainder time until twilight starts
            # We can take the remaining time and try to do 1,2, or 3 blocks.
            possible_times = available_time / np.arange(1, 4)
            diff = np.abs(self.ideal_pair_time - possible_times)
            best_block_time = np.max(possible_times[np.where(diff == np.min(diff))])
            self.nvisit_block = int(
                np.floor(
                    best_block_time
                    * 60.0
                    / (self.slew_approx + self.exptime + self.read_approx * (self.nexp - 1))
                )
            )

        # The floor can set block to zero, make it possible to to just one
        if self.nvisit_block <= 0:
            self.nvisit_block = 1

    def calc_reward_function(self, conditions):
        # Set the number of observations we are going to try and take
        self._set_block_size(conditions)
        #  Computing reward like usual with basis functions and weights
        if self._check_feasibility(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value * weight
            if self.smoothing_kernel is not None:
                self.smooth_reward()
        else:
            self.reward = -np.inf
            return self.reward

        if self.area_required is not None:
            max_indices = np.where(self.reward == np.nanmax(self.reward))[0]
            if np.size(max_indices) == 0:
                # This is the case if everything is masked
                self.reward = -np.inf
            else:
                max_reward_indx = np.min(max_indices)
                distances = _angular_separation(
                    self.ra,
                    self.dec,
                    self.ra[max_reward_indx],
                    self.dec[max_reward_indx],
                )
                good_area = np.where((np.abs(self.reward) >= 0) & (distances < self.max_radius_peak))[
                    0
                ].size * hp.nside2pixarea(self.nside)
                if good_area < self.area_required:
                    self.reward = -np.inf

        self.reward_checked = True
        return self.reward

    def simple_order_sort(self):
        """Fall back if we can't link contiguous blobs in the reward map"""

        # Assuming reward has already been calcualted

        potential_hp = np.where(~np.isnan(self.reward) == True)

        # Note, using nanmax, so masked pixels might be included in the pointing.
        # I guess I should document that it's not "NaN pixels can't be observed", but
        # "non-NaN pixles CAN be observed", which probably is not intuitive.
        ufields, reward_by_field = int_binned_stat(
            self.hp2fields[potential_hp], self.reward[potential_hp], statistic=np.nanmax
        )
        # chop off any nans
        not_nans = np.where(~np.isnan(reward_by_field) == True)
        ufields = ufields[not_nans]
        reward_by_field = reward_by_field[not_nans]

        order = np.argsort(reward_by_field)
        ufields = ufields[order][::-1][0 : self.nvisit_block]
        self.best_fields = ufields

    def generate_observations_rough(self, conditions):
        """
        Find a good block of observations.
        """

        self.reward = self.calc_reward_function(conditions)

        # Mask off pixels that are far away from the maximum.
        max_reward_indx = np.min(np.where(self.reward == np.nanmax(self.reward)))
        distances = _angular_separation(
            self.ra, self.dec, self.ra[max_reward_indx], self.dec[max_reward_indx]
        )

        self.reward[np.where(distances > self.max_radius_peak)] = np.nan
        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields(conditions)
            self.night = copy(conditions.night)

        if self.grow_blob:
            # Note, returns highest first
            ordered_hp = hp_grow_argsort(self.reward)
            ordered_fields = self.hp2fields[ordered_hp]
            orig_order = np.arange(ordered_fields.size)
            # Remove duplicate field pointings
            _u_of, u_indx = np.unique(ordered_fields, return_index=True)
            new_order = np.argsort(orig_order[u_indx])
            best_fields = ordered_fields[u_indx[new_order]]

            if np.size(best_fields) < self.nvisit_block:
                # Let's fall back to the simple sort
                self.simple_order_sort()
            else:
                self.best_fields = best_fields[0 : self.nvisit_block]
        else:
            self.simple_order_sort()

        if len(self.best_fields) == 0:
            # everything was nans, or self.nvisit_block was zero
            return []

        # Let's find the alt, az coords of the points (right now, hopefully doesn't change much in time block)
        pointing_alt, pointing_az = _approx_ra_dec2_alt_az(
            self.fields["RA"][self.best_fields],
            self.fields["dec"][self.best_fields],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
            lmst=conditions.lmst,
        )

        # Let's find a good spot to project the points to a plane
        mid_alt = (np.max(pointing_alt) - np.min(pointing_alt)) / 2.0 + np.min(pointing_alt)

        # Code snippet from MAF for computing mean of angle accounting for wrap around
        # XXX-TODO: Maybe move this to sims_utils as a generally useful snippet.
        x = np.cos(pointing_az)
        y = np.sin(pointing_az)
        meanx = np.mean(x)
        meany = np.mean(y)
        angle = np.arctan2(meany, meanx)
        radius = np.sqrt(meanx**2 + meany**2)
        mid_az = angle % (2.0 * np.pi)
        if radius < 0.1:
            mid_az = np.pi

        # Project the alt,az coordinates to a plane. Could consider scaling things to represent
        # time between points rather than angular distance.
        pointing_x, pointing_y = gnomonic_project_toxy(pointing_az, pointing_alt, mid_az, mid_alt)
        # Round off positions so that we ensure identical cross-platform performance
        scale = 1e4
        pointing_x = np.round(pointing_x * scale).astype(int)
        pointing_y = np.round(pointing_y * scale).astype(int)
        # Now I have a bunch of x,y pointings. Drop into TSP solver to get an effiencent route
        towns = np.vstack((pointing_x, pointing_y)).T
        # Leaving optimize=False for speed. The optimization step doesn't usually improve much.
        better_order = tsp_convex(towns, optimize=False)
        # XXX-TODO: Could try to roll better_order to start at the nearest/fastest slew from current position.
        observations = []
        counter2 = 0
        approx_end_time = np.size(better_order) * (
            self.slew_approx + self.exptime + self.read_approx * (self.nexp - 1)
        )
        flush_time = conditions.mjd + approx_end_time / 3600.0 / 24.0 + self.flush_time
        for i, indx in enumerate(better_order):
            field = self.best_fields[indx]
            obs = empty_observation()
            obs["RA"] = self.fields["RA"][field]
            obs["dec"] = self.fields["dec"][field]
            obs["rotSkyPos"] = 0.0
            obs["filter"] = self.filtername1
            if self.nexp_dict is None:
                obs["nexp"] = self.nexp
            else:
                obs["nexp"] = self.nexp_dict[self.filtername1]
            obs["exptime"] = self.exptime
            obs["field_id"] = -1
            obs["note"] = "%s" % (self.survey_note)
            obs["block_id"] = self.counter
            obs["flush_by_mjd"] = flush_time
            # Add the mjd for debugging
            # obs['mjd'] = conditions.mjd
            # XXX temp debugging line
            obs["survey_id"] = i
            observations.append(obs)
            counter2 += 1

        result = observations
        return result
