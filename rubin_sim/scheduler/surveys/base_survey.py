__all__ = ("BaseSurvey", "BaseMarkovSurvey")

from copy import copy, deepcopy

import healpy as hp
import numpy as np
import pandas as pd

from rubin_sim.scheduler.basis_functions.mask_basis_funcs import ZenithShadowMaskBasisFunction
from rubin_sim.scheduler.detailers import ZeroRotDetailer
from rubin_sim.scheduler.thomson import thetaphi2xyz, xyz2thetaphi
from rubin_sim.scheduler.utils import (
    HpInComcamFov,
    HpInLsstFov,
    comcam_tessellate,
    empty_observation,
    set_default_nside,
)
from rubin_sim.site_models import _read_fields


class BaseSurvey:
    """A baseclass for survey objects.

    Parameters
    ----------
    basis_functions : list
        List of basis_function objects
    extra_features : list XXX--should this be a dict for clarity?
        List of any additional features the survey may want to use
        e.g., for computing final dither positions.
    extra_basis_functions : dict of rubin_sim.scheduler.basis_function objects
        Extra basis function objects. Typically not psased in, but et in the __init__.
    ignore_obs : list of str (None)
        If an incoming observation has this string in the note, ignore it. Handy if
        one wants to ignore DD fields or observations requested by self. Take note,
        if a survey is called 'mysurvey23', setting ignore_obs to 'mysurvey2' will
        ignore it because 'mysurvey2' is a substring of 'mysurvey23'.
    detailers : list of rubin_sim.scheduler.detailers objects
        The detailers to apply to the list of observations.
    scheduled_obs : np.array
        An array of MJD values for when observations should execute.
    """

    def __init__(
        self,
        basis_functions,
        extra_features=None,
        extra_basis_functions=None,
        ignore_obs=None,
        survey_name=None,
        nside=None,
        detailers=None,
        scheduled_obs=None,
    ):
        if nside is None:
            nside = set_default_nside()
        if ignore_obs is None:
            ignore_obs = []

        if isinstance(ignore_obs, str):
            ignore_obs = [ignore_obs]

        self.nside = nside
        if survey_name is None:
            self._generate_survey_name()
        else:
            self.survey_name = survey_name
        self.ignore_obs = ignore_obs

        self.reward = None
        self.survey_index = None

        self.basis_functions = basis_functions

        if extra_features is None:
            self.extra_features = {}
        else:
            self.extra_features = extra_features
        if extra_basis_functions is None:
            self.extra_basis_functions = {}
        else:
            self.extra_basis_functions = extra_basis_functions

        self.reward_checked = False

        # Attribute to track if the reward function is up-to-date.
        self.reward_checked = False

        # If there's no detailers, add one to set rotation to near zero
        if detailers is None:
            self.detailers = [ZeroRotDetailer(nside=nside)]
        else:
            self.detailers = detailers

        # Scheduled observations
        self.scheduled_obs = scheduled_obs

    def _generate_survey_name(self):
        self.survey_name = ""

    def get_scheduled_obs(self):
        return self.scheduled_obs

    def add_observations_array(self, observations_array_in, observations_hpid_in):
        """Add an array of observations rather than one at a time

        Parameters
        ----------
        observations_array_in : np.array
            An array of completed observations (with columns like rubin_sim.scheduler.utils.empty_observation).
        observations_hpid_in : np.array
            Same as observations_array_in, but larger and with an additional column for HEALpix id. Each
            observation is listed mulitple times, once for every HEALpix it overlaps."""

        # Just to be sure things are sorted
        observations_array_in.sort(order="mjd")
        observations_hpid_in.sort(order="mjd")

        # Copy so we don't prune things for other survey objects
        observations_array = observations_array_in.copy()
        observations_hpid = observations_hpid_in.copy()

        for ig in self.ignore_obs:
            not_ignore = np.where(np.char.find(observations_array["note"], ig) == -1)[0]
            observations_array = observations_array[not_ignore]

            not_ignore = np.where(np.char.find(observations_hpid["note"], ig) == -1)[0]
            observations_hpid = observations_hpid[not_ignore]

        for feature in self.extra_features:
            self.extra_features[feature].add_observations_array(observations_array, observations_hpid)
        for bf in self.extra_basis_functions:
            self.extra_basis_functions[bf].add_observations_array(observations_array, observations_hpid)
        for bf in self.basis_functions:
            bf.add_observations_array(observations_array, observations_hpid)
        for detailer in self.detailers:
            detailer.add_observations_array(observations_array, observations_hpid)
        self.reward_checked = False

    def add_observation(self, observation, **kwargs):
        # Check each posible ignore string
        checks = [io not in str(observation["note"]) for io in self.ignore_obs]
        # ugh, I think here I have to assume observation is an array and not a dict.
        if all(checks):
            for feature in self.extra_features:
                self.extra_features[feature].add_observation(observation, **kwargs)
            for bf in self.extra_basis_functions:
                self.extra_basis_functions[bf].add_observation(observation, **kwargs)
            for bf in self.basis_functions:
                bf.add_observation(observation, **kwargs)
            for detailer in self.detailers:
                detailer.add_observation(observation, **kwargs)
            self.reward_checked = False

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """
        result = True
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        return result

    def calc_reward_function(self, conditions):
        """
        Parameters
        ----------
        conditions : rubin_sim.scheduler.features.Conditions object

        Returns
        -------
        reward : float (or array)

        """
        if self._check_feasibility(conditions):
            self.reward = 0
        else:
            # If we don't pass feasability
            self.reward = -np.inf

        self.reward_checked = True
        return self.reward

    def generate_observations_rough(self, conditions):
        """
        Returns
        -------
        one of:
            1) None
            2) A list of observations
        """
        # If the reward function hasn't been updated with the
        # latest info, calculate it
        if not self.reward_checked:
            self.reward = self.calc_reward_function(conditions)
        obs = empty_observation()
        return [obs]

    def generate_observations(self, conditions):
        observations = self.generate_observations_rough(conditions)
        for detailer in self.detailers:
            observations = detailer(observations, conditions)
        return observations

    def viz_config(self):
        # XXX--zomg, we should have a method that goes through all the objects and
        # makes plots/prints info so there can be a little notebook showing the config!
        pass

    def __repr__(self):
        try:
            repr = f"<{self.__class__.__name__} survey_name='{self.survey_name}' at {hex(id(self))}>"
        except AttributeError:
            repr = f"<{self.__class__.__name__} at {hex(id(self))}>"

        return repr

    def _reward_to_scalars(self, reward):
        try:
            pix_area = hp.nside2pixarea(self.nside, degrees=True)
        except AttributeError:
            pix_area = None

        if np.isscalar(reward):
            unmasked_area = pix_area * hp.nside2npix(self.nside)
        else:
            unmasked_area = pix_area * np.count_nonzero(reward > -np.inf)

        if np.isscalar(reward):
            scalar_reward = reward
        elif unmasked_area == 0:
            scalar_reward = -np.inf
        else:
            scalar_reward = np.nanmax(reward)

        return scalar_reward, unmasked_area

    def make_reward_df(self, conditions, accum=True):
        """Create a pandas.DataFrame describing the reward from the survey.

        Parameters
        ----------
        conditions : `rubin_sim.scheduler.features.Conditions`
            Conditions for which rewards are to be returned
        accum : `bool`
            Include accumulated reward (more compute intensive)
            Defaults to True

        Returns
        -------
        reward_df : `pandas.DataFrame`
            A table of surveys listing the rewards.
        """

        feasibility = []
        max_rewards = []
        basis_areas = []
        accum_rewards = []
        accum_areas = []
        bf_label = []
        bf_class = []
        basis_functions = []
        basis_weights = []

        try:
            full_basis_weights = self.basis_weights
        except AttributeError:
            full_basis_weights = [1.0 for df in self.basis_functions]

        short_labels = self.bf_short_labels

        # Only count the part of the sky high enough to observe.
        horizon_mask = np.where(conditions.alt > np.radians(20), 1, np.nan)
        _, scalar_area = self._reward_to_scalars(horizon_mask)

        for weight, basis_function in zip(full_basis_weights, self.basis_functions):
            bf_label.append(short_labels[basis_function.label()])
            bf_class.append(basis_function.__class__.__name__)
            bf_reward = basis_function(conditions)
            if np.isscalar(bf_reward):
                max_reward = bf_reward
                basis_area = scalar_area
            else:
                bf_reward = bf_reward * horizon_mask
                max_reward, basis_area = self._reward_to_scalars(bf_reward)

            max_rewards.append(max_reward)
            basis_areas.append(basis_area)

            this_feasibility = np.array(basis_function.check_feasibility(conditions)).any()
            feasibility.append(this_feasibility)

            if accum:
                basis_functions.append(basis_function)
                basis_weights.append(weight)
                test_survey = deepcopy(self)
                test_survey.basis_functions = basis_functions
                test_survey.basis_weights = basis_weights
                this_accum_reward = test_survey.calc_reward_function(conditions)
                accum_reward, accum_area = self._reward_to_scalars(this_accum_reward)
                accum_rewards.append(accum_reward)
                accum_areas.append(accum_area)

        reward_data = {
            "basis_function": bf_label,
            "basis_function_class": bf_class,
            "feasible": feasibility,
            "max_basis_reward": max_rewards,
            "basis_area": basis_areas,
            "basis_weight": full_basis_weights,
        }
        if accum:
            reward_data["max_accum_reward"] = accum_rewards
            reward_data["accum_area"] = accum_areas

        reward_df = pd.DataFrame(reward_data)

        return reward_df

    def reward_changes(self, conditions):
        """List the rewards for each basis function used by the survey.

        Parameters
        ----------
        conditions : `rubin_sim.scheduler.features.Conditions`
            Conditions for which rewards are to be returned

        Returns
        -------
        rewards : `list`
            A list of tuples, each with a basis function name and the
            maximum reward returned by that basis function for the
            provided conditions.
        """

        reward_values = []
        basis_functions = []
        basis_weights = []

        try:
            full_basis_weights = self.basis_weights
        except AttributeError:
            full_basis_weights = [1 for bf in self.basis_functions]

        for weight, basis_function in zip(full_basis_weights, self.basis_functions):
            test_survey = deepcopy(self)
            basis_functions.append(basis_function)
            test_survey.basis_functions = basis_functions
            basis_weights.append(weight)
            test_survey.basis_weights = basis_weights
            try:
                reward_values.append(np.nanmax(test_survey.calc_reward_function(conditions)))
            except IndexError:
                reward_values.append(None)

        bf_names = [bf.label() for bf in self.basis_functions]
        return list(zip(bf_names, reward_values))

    @property
    def bf_short_labels(self):
        try:
            long_labels = [bf.label() for bf in self.basis_functions]
        except AttributeError:
            return []

        label_bases = [label.split(" @")[0] for label in long_labels]
        duplicated_labels = set([label for label in label_bases if label_bases.count(label) > 1])
        short_labels = []
        label_count = {k: 0 for k in duplicated_labels}
        for label_base in label_bases:
            if label_base in duplicated_labels:
                label_count[label_base] += 1
                short_labels.append(f"{label_base} {label_count[label_base]}")
            else:
                short_labels.append(label_base)

        label_map = dict(zip(long_labels, short_labels))

        return label_map


def rotx(theta, x, y, z):
    """rotate the x,y,z points theta radians about x axis"""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xp = x
    yp = y * cos_t + z * sin_t
    zp = -y * sin_t + z * cos_t
    return xp, yp, zp


class BaseMarkovSurvey(BaseSurvey):
    """A Markov Decision Function survey object. Uses Basis functions to compute a
    final reward function and decide what to observe based on the reward. Includes
    methods for dithering and defaults to dithering nightly.

    Parameters
    ----------
    basis_function : list of rubin_sim.schuler.basis_function objects

    basis_weights : list of float
        Must be same length as basis_function
    seed : hashable
        Random number seed, used for randomly orienting sky tessellation.
    camera : str ('LSST')
        Should be 'LSST' or 'comcam'
    fields : np.array (None)
        An array of field positions. Should be numpy array with columns of "RA" and
        "dec" in radians. If none, site_models.read_fields or utils.comcam_tessellate is
        used to read field positions.
    area_required : float (None)
        The valid area that should be present in the reward function (square degrees).
    npositions : int (7305)
        The number of dither positions to pre-compute. Defaults to 7305 (so good for 20 years)
    """

    def __init__(
        self,
        basis_functions,
        basis_weights,
        extra_features=None,
        smoothing_kernel=None,
        ignore_obs=None,
        survey_name=None,
        nside=None,
        seed=42,
        dither=True,
        detailers=None,
        camera="LSST",
        fields=None,
        area_required=None,
        npositions=7305,
    ):
        super(BaseMarkovSurvey, self).__init__(
            basis_functions=basis_functions,
            extra_features=extra_features,
            ignore_obs=ignore_obs,
            survey_name=survey_name,
            nside=nside,
            detailers=detailers,
        )

        self.basis_weights = basis_weights
        # Check that weights and basis functions are same length
        if len(basis_functions) != np.size(basis_weights):
            raise ValueError("basis_functions and basis_weights must be same length.")

        self.camera = camera
        # Load the OpSim field tesselation and map healpix to fields
        if fields is None:
            if self.camera == "LSST":
                ra, dec = _read_fields()
                self.fields_init = np.empty(ra.size, dtype=list(zip(["RA", "dec"], [float, float])))
                self.fields_init["RA"] = ra
                self.fields_init["dec"] = dec
            elif self.camera == "comcam":
                self.fields_init = comcam_tessellate()
            else:
                ValueError('camera %s unknown, should be "LSST" or "comcam"' % camera)
        else:
            self.fields_init = fields
        self.fields = self.fields_init.copy()
        self.hp2fields = np.array([])
        self._hp2fieldsetup(self.fields["RA"], self.fields["dec"])

        if smoothing_kernel is not None:
            self.smoothing_kernel = np.radians(smoothing_kernel)
        else:
            self.smoothing_kernel = None

        if area_required is None:
            self.area_required = area_required
        else:
            self.area_required = area_required * (np.pi / 180.0) ** 2  # To steradians

        # Start tracking the night
        self.night = -1

        self.dither = dither

        # Generate and store rotation positions to use.
        # This way, if different survey objects are seeded the same, they will
        # use the same dither positions each night
        rng = np.random.default_rng(seed)
        self.lon = rng.random(npositions) * np.pi * 2
        # Make sure latitude points spread correctly
        # http://mathworld.wolfram.com/SpherePointPicking.html
        self.lat = np.arccos(2.0 * rng.random(npositions) - 1.0)
        self.lon2 = rng.random(npositions) * np.pi * 2

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        if self.area_required is not None:
            reward = self.calc_reward_function(conditions)
            good_pix = np.where(np.isfinite(reward) == True)[0]
            area = hp.nside2pixarea(self.nside) * np.size(good_pix)
            if area < self.area_required:
                return False
        return result

    def _hp2fieldsetup(self, ra, dec):
        """Map each healpixel to nearest field. This will only work if healpix
        resolution is higher than field resolution.
        """
        if self.camera == "LSST":
            pointing2hpindx = HpInLsstFov(nside=self.nside)
        elif self.camera == "comcam":
            pointing2hpindx = HpInComcamFov(nside=self.nside)

        self.hp2fields = np.zeros(hp.nside2npix(self.nside), dtype=int)
        for i in range(len(ra)):
            hpindx = pointing2hpindx(ra[i], dec[i], rotSkyPos=0.0)
            self.hp2fields[hpindx] = i

    def _spin_fields(self, conditions, lon=None, lat=None, lon2=None):
        """Spin the field tessellation to generate a random orientation

        The default field tesselation is rotated randomly in longitude, and then the
        pole is rotated to a random point on the sphere.

        Parameters
        ----------
        lon : float (None)
            The amount to initially rotate in longitude (radians). Will use a random value
            between 0 and 2 pi if None (default).
        lat : float (None)
            The amount to rotate in latitude (radians).
        lon2 : float (None)
            The amount to rotate the pole in longitude (radians).
        """
        if lon is None:
            lon = self.lon[conditions.night]
        if lat is None:
            lat = self.lat[conditions.night]
        if lon2 is None:
            lon2 = self.lon2[conditions.night]

        # rotate longitude
        ra = (self.fields_init["RA"] + lon) % (2.0 * np.pi)
        dec = copy(self.fields_init["dec"])

        # Now to rotate ra and dec about the x-axis
        x, y, z = thetaphi2xyz(ra, dec + np.pi / 2.0)
        xp, yp, zp = rotx(lat, x, y, z)
        theta, phi = xyz2thetaphi(xp, yp, zp)
        dec = phi - np.pi / 2
        ra = theta + np.pi

        # One more RA rotation
        ra = (ra + lon2) % (2.0 * np.pi)

        self.fields["RA"] = ra
        self.fields["dec"] = dec
        # Rebuild the kdtree with the new positions
        # XXX-may be doing some ra,dec to conversions xyz more than needed.
        self._hp2fieldsetup(ra, dec)

    def smooth_reward(self):
        """If we want to smooth the reward function."""
        if hp.isnpixok(self.reward.size):
            # Need to swap NaNs to hp.UNSEEN so smoothing doesn't spread mask
            reward_temp = copy(self.reward)
            mask = np.isnan(reward_temp)
            reward_temp[mask] = hp.UNSEEN
            self.reward_smooth = hp.sphtfunc.smoothing(reward_temp, fwhm=self.smoothing_kernel, verbose=False)
            self.reward_smooth[mask] = np.nan
            self.reward = self.reward_smooth
            # good = ~np.isnan(self.reward_smooth)
            # Round off to prevent strange behavior early on
            # self.reward_smooth[good] = np.round(self.reward_smooth[good], decimals=4)

    def calc_reward_function(self, conditions):
        self.reward_checked = True
        if self._check_feasibility(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value * weight
        else:
            # If not feasable, negative infinity reward
            self.reward = -np.inf
            return self.reward
        if self.smoothing_kernel is not None:
            self.smooth_reward()

        if self.area_required is not None:
            good_area = np.where(np.abs(self.reward) >= 0)[0].size * hp.nside2pixarea(self.nside)
            if good_area < self.area_required:
                self.reward = -np.inf

        return self.reward

    def generate_observations_rough(self, conditions):
        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields(conditions)
            self.night = copy(conditions.night)

        # XXX Use self.reward to decide what to observe.
        return None
