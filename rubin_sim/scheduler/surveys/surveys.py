import numpy as np
from rubin_sim.scheduler.utils import (empty_observation, set_default_nside)
import healpy as hp
import matplotlib.pylab as plt
from rubin_sim.scheduler.surveys import BaseMarkovDF_survey
from rubin_sim.scheduler.utils import (int_binned_stat, int_rounded,
                                              gnomonic_project_toxy, tsp_convex)
import copy
from rubin_sim.utils import _angularSeparation, _hpid2RaDec, _approx_RaDec2AltAz, hp_grow_argsort
import warnings

__all__ = ['Greedy_survey', 'Blob_survey']


class Greedy_survey(BaseMarkovDF_survey):
    """
    Select pointings in a greedy way using a Markov Decision Process.
    """
    def __init__(self, basis_functions, basis_weights, filtername='r',
                 block_size=1, smoothing_kernel=None, nside=None,
                 dither=True, seed=42, ignore_obs=None, survey_name='',
                 nexp=2, exptime=30., detailers=None, camera='LSST', area_required=None):

        extra_features = {}

        super(Greedy_survey, self).__init__(basis_functions=basis_functions,
                                            basis_weights=basis_weights,
                                            extra_features=extra_features,
                                            smoothing_kernel=smoothing_kernel,
                                            ignore_obs=ignore_obs,
                                            nside=nside,
                                            survey_name=survey_name, dither=dither,
                                            detailers=detailers, camera=camera,
                                            area_required=area_required)
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
            self._spin_fields()
            self.night = conditions.night.copy()

        # Let's find the best N from the fields
        order = np.argsort(self.reward)[::-1]
        # Crop off any NaNs
        order = order[~np.isnan(self.reward[order])]

        iter = 0
        while True:
            best_hp = order[iter*self.block_size:(iter+1)*self.block_size]
            best_fields = np.unique(self.hp2fields[best_hp])
            observations = []
            for field in best_fields:
                obs = empty_observation()
                obs['RA'] = self.fields['RA'][field]
                obs['dec'] = self.fields['dec'][field]
                obs['rotSkyPos'] = 0.
                obs['filter'] = self.filtername
                obs['nexp'] = self.nexp
                obs['exptime'] = self.exptime
                obs['field_id'] = -1
                obs['note'] = self.survey_name

                observations.append(obs)
                break
            iter += 1
            if len(observations) > 0 or (iter+2)*self.block_size > len(order):
                break
        return observations


class Blob_survey(Greedy_survey):
    """Select observations in large, mostly contiguous, blobs.

    Parameters
    ----------
    filtername1 : str ('r')
        The filter to observe in.
    filtername2 : str ('g')
        The filter to pair with the first observation. If set to None, no pair
        will be observed.
    slew_approx : float (7.5)
        The approximate slewtime between neerby fields (seconds). Used to calculate
        how many observations can be taken in the desired time block.
    nexp : int (2)
        The number of exposures to take in a visit.
    exp_dict : dict (None)
        If set, should have keys of filtername and values of ints that are the nuber of exposures to take
        per visit. For estimating block time, nexp is still used.
    filter_change_approx : float (140.)
         The approximate time it takes to change filters (seconds).
    ideal_pair_time : float (22.)
        The ideal time gap wanted between observations to the same pointing (minutes)
    min_pair_time : float (15.)
        The minimum acceptable pair time (minutes)
    search_radius : float (30.)
        The radius around the reward peak to look for additional potential pointings (degrees)
    alt_max : float (85.)
        The maximum altitude to include (degrees).
    az_range : float (90.)
        The range of azimuths to consider around the peak reward value (degrees).
    flush_time : float (30.)
        The time past the final expected exposure to flush the queue. Keeps observations
        from lingering past when they should be executed. (minutes)
    twilight_scale : bool (True)
        Scale the block size to fill up to twilight. Set to False if running in twilight
    in_twilight : bool (False)
        Scale the block size to stay within twilight time.
    check_scheduled : bool (True)
        Check if there are scheduled observations and scale blob size to match
    min_area : float (None)
        If set, demand the reward function have an area of so many square degrees before executing
    grow_blob : bool (True)
        If True, try to grow the blob from the global maximum. Otherwise, just use a simple sort.
        Simple sort will not constrain the blob to be contiguous.
    """
    def __init__(self, basis_functions, basis_weights,
                 filtername1='r', filtername2='g',
                 slew_approx=7.5, filter_change_approx=140.,
                 read_approx=2., exptime=30., nexp=2, nexp_dict=None,
                 ideal_pair_time=22., min_pair_time=15.,
                 search_radius=30., alt_max=85., az_range=90.,
                 flush_time=30.,
                 smoothing_kernel=None, nside=None,
                 dither=True, seed=42, ignore_obs=None,
                 survey_note='blob', detailers=None, camera='LSST',
                 twilight_scale=True, in_twilight=False, check_scheduled=True, min_area=None,
                 grow_blob=True, area_required=None):

        if nside is None:
            nside = set_default_nside()

        super(Blob_survey, self).__init__(basis_functions=basis_functions,
                                          basis_weights=basis_weights,
                                          filtername=None,
                                          block_size=0, smoothing_kernel=smoothing_kernel,
                                          dither=dither, seed=seed, ignore_obs=ignore_obs,
                                          nside=nside, detailers=detailers, camera=camera,
                                          area_required=area_required)
        self.flush_time = flush_time/60./24.  # convert to days
        self.nexp = nexp
        self.nexp_dict = nexp_dict
        self.exptime = exptime
        self.slew_approx = slew_approx
        self.read_approx = read_approx
        self.hpids = np.arange(hp.nside2npix(self.nside))
        self.twilight_scale = twilight_scale
        self.in_twilight = in_twilight
        self.grow_blob = grow_blob

        if self.twilight_scale & self.in_twilight:
            warnings.warn('Both twilight_scale and in_twilight are set to True. That is probably wrong.')

        self.min_area = min_area
        self.check_scheduled = check_scheduled
        # If we are taking pairs in same filter, no need to add filter change time.
        if filtername1 == filtername2:
            filter_change_approx = 0
        # Compute the minimum time needed to observe a blob (or observe, then repeat.)
        if filtername2 is not None:
            self.time_needed = (min_pair_time*60.*2. + exptime + read_approx + filter_change_approx)/24./3600.  # Days
        else:
            self.time_needed = (min_pair_time*60. + exptime + read_approx)/24./3600.  # Days
        self.filter_set = set(filtername1)
        if filtername2 is None:
            self.filter2_set = self.filter_set
        else:
            self.filter2_set = set(filtername2)

        self.ra, self.dec = _hpid2RaDec(self.nside, self.hpids)

        self.survey_note = survey_note
        self.counter = 1  # start at 1, because 0 is default in empty observation
        self.filtername1 = filtername1
        self.filtername2 = filtername2
        self.search_radius = np.radians(search_radius)
        self.az_range = np.radians(az_range)
        self.alt_max = np.radians(alt_max)
        self.min_pair_time = min_pair_time
        self.ideal_pair_time = ideal_pair_time

        self.pixarea = hp.nside2pixarea(self.nside, degrees=True)

        # If we are only using one filter, this could be useful
        if (self.filtername2 is None) | (self.filtername1 == self.filtername2):
            self.filtername = self.filtername1

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
                reward += basis_value*weight
            valid_pix = np.where(np.isnan(reward) == False)[0]
            if np.size(valid_pix)*self.pixarea < self.min_area:
                result = False
        return result

    def _set_block_size(self, conditions):
        """
        Update the block size if it's getting near a break point.
        """

        # If we are trying to get things done before twilight
        if self.twilight_scale:
            available_time = conditions.sun_n18_rising - conditions.mjd
            available_time *= 24.*60.  # to minutes
            n_ideal_blocks = available_time / self.ideal_pair_time
        else:
            n_ideal_blocks = 4

        # If we are trying to get things done before a scheduled simulation
        if self.check_scheduled:
            if len(conditions.scheduled_observations) > 0:
                available_time = np.min(conditions.scheduled_observations) - conditions.mjd
                available_time *= 24.*60.  # to minutes
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
            available_time *= 24.*60.  # to minutes
            n_blocks = available_time / self.ideal_pair_time
            if n_blocks < n_ideal_blocks:
                n_ideal_blocks = n_blocks

        if n_ideal_blocks >= 3:
            self.nvisit_block = int(np.floor(self.ideal_pair_time*60. / (self.slew_approx + self.exptime +
                                                                         self.read_approx*(self.nexp - 1))))
        else:
            # Now we can stretch or contract the block size to allocate the remainder time until twilight starts
            # We can take the remaining time and try to do 1,2, or 3 blocks.
            possible_times = available_time / np.arange(1, 4)
            diff = np.abs(self.ideal_pair_time-possible_times)
            best_block_time = np.max(possible_times[np.where(diff == np.min(diff))])
            self.nvisit_block = int(np.floor(best_block_time*60. / (self.slew_approx + self.exptime +
                                                                    self.read_approx*(self.nexp - 1))))

        # The floor can set block to zero, make it possible to to just one
        if self.nvisit_block <= 0:
            self.nvisit_block = 1

    def calc_reward_function(self, conditions):
        """
        """
        # Set the number of observations we are going to try and take
        self._set_block_size(conditions)
        #  Computing reward like usual with basis functions and weights
        if self._check_feasibility(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value*weight
                # might be faster to pull this out into the feasabiliity check?
            if self.smoothing_kernel is not None:
                self.smooth_reward()

            # Apply max altitude cut
            too_high = np.where(int_rounded(conditions.alt) > int_rounded(self.alt_max))
            self.reward[too_high] = np.nan

            # Select healpixels within some radius of the max
            # This is probably faster with a kd-tree.

            max_hp = np.where(self.reward == np.nanmax(self.reward))[0]
            if np.size(max_hp) > 0:
                peak_reward = np.min(max_hp)
            else:
                # Everything is masked, so get out
                return -np.inf

            # Apply radius selection
            dists = _angularSeparation(self.ra[peak_reward], self.dec[peak_reward], self.ra, self.dec)
            out_hp = np.where(int_rounded(dists) > int_rounded(self.search_radius))
            self.reward[out_hp] = np.nan

            # Apply az cut
            az_centered = conditions.az - conditions.az[peak_reward]
            az_centered[np.where(az_centered < 0)] += 2.*np.pi

            az_out = np.where((int_rounded(az_centered) > int_rounded(self.az_range/2.)) &
                              (int_rounded(az_centered) < int_rounded(2.*np.pi-self.az_range/2.)))
            self.reward[az_out] = np.nan
        else:
            self.reward = -np.inf

        if self.area_required is not None:
            good_area = np.where(np.abs(self.reward) >= 0)[0].size * hp.nside2pixarea(self.nside)
            if good_area < self.area_required:
                self.reward = -np.inf

        #if ('twi' in self.survey_note) & (np.any(np.isfinite(self.reward))):
        #    import pdb ; pdb.set_trace()

        self.reward_checked = True
        return self.reward

    def simple_order_sort(self):
        """Fall back if we can't link contiguous blobs in the reward map
        """

        # Assuming reward has already been calcualted

        potential_hp = np.where(~np.isnan(self.reward) == True)

        # Note, using nanmax, so masked pixels might be included in the pointing.
        # I guess I should document that it's not "NaN pixels can't be observed", but
        # "non-NaN pixles CAN be observed", which probably is not intuitive.
        ufields, reward_by_field = int_binned_stat(self.hp2fields[potential_hp],
                                                   self.reward[potential_hp],
                                                   statistic=np.nanmax)
        # chop off any nans
        not_nans = np.where(~np.isnan(reward_by_field) == True)
        ufields = ufields[not_nans]
        reward_by_field = reward_by_field[not_nans]

        order = np.argsort(reward_by_field)
        ufields = ufields[order][::-1][0:self.nvisit_block]
        self.best_fields = ufields

    def generate_observations_rough(self, conditions):
        """
        Find a good block of observations.
        """

        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields()
            self.night = conditions.night.copy()

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
                self.best_fields = best_fields[0:self.nvisit_block]
        else:
            self.simple_order_sort()

        if len(self.best_fields) == 0:
            # everything was nans, or self.nvisit_block was zero
            return []

        # Let's find the alt, az coords of the points (right now, hopefully doesn't change much in time block)
        pointing_alt, pointing_az = _approx_RaDec2AltAz(self.fields['RA'][self.best_fields],
                                                        self.fields['dec'][self.best_fields],
                                                        conditions.site.latitude_rad,
                                                        conditions.site.longitude_rad,
                                                        conditions.mjd,
                                                        lmst=conditions.lmst)

        # Let's find a good spot to project the points to a plane
        mid_alt = (np.max(pointing_alt) - np.min(pointing_alt))/2.

        # Code snippet from MAF for computing mean of angle accounting for wrap around
        # XXX-TODO: Maybe move this to sims_utils as a generally useful snippet.
        x = np.cos(pointing_az)
        y = np.sin(pointing_az)
        meanx = np.mean(x)
        meany = np.mean(y)
        angle = np.arctan2(meany, meanx)
        radius = np.sqrt(meanx**2 + meany**2)
        mid_az = angle % (2.*np.pi)
        if radius < 0.1:
            mid_az = np.pi

        # Project the alt,az coordinates to a plane. Could consider scaling things to represent
        # time between points rather than angular distance.
        pointing_x, pointing_y = gnomonic_project_toxy(pointing_az, pointing_alt, mid_az, mid_alt)
        # Round off positions so that we ensure identical cross-platform performance
        scale = 1e6
        pointing_x = np.round(pointing_x*scale).astype(int)
        pointing_y = np.round(pointing_y*scale).astype(int)
        # Now I have a bunch of x,y pointings. Drop into TSP solver to get an effiencent route
        towns = np.vstack((pointing_x, pointing_y)).T
        # Leaving optimize=False for speed. The optimization step doesn't usually improve much.
        better_order = tsp_convex(towns, optimize=False)
        # XXX-TODO: Could try to roll better_order to start at the nearest/fastest slew from current position.
        observations = []
        counter2 = 0
        approx_end_time = np.size(better_order)*(self.slew_approx + self.exptime +
                                                 self.read_approx*(self.nexp - 1))
        flush_time = conditions.mjd + approx_end_time/3600./24. + self.flush_time
        for i, indx in enumerate(better_order):
            field = self.best_fields[indx]
            obs = empty_observation()
            obs['RA'] = self.fields['RA'][field]
            obs['dec'] = self.fields['dec'][field]
            obs['rotSkyPos'] = 0.
            obs['filter'] = self.filtername1
            if self.nexp_dict is None:
                obs['nexp'] = self.nexp
            else:
                obs['nexp'] = self.nexp_dict[self.filtername1]
            obs['exptime'] = self.exptime
            obs['field_id'] = -1
            obs['note'] = '%s' % (self.survey_note)
            obs['block_id'] = self.counter
            obs['flush_by_mjd'] = flush_time
            # Add the mjd for debugging
            # obs['mjd'] = conditions.mjd
            # XXX temp debugging line
            obs['survey_id'] = i
            observations.append(obs)
            counter2 += 1

        result = observations
        return result
