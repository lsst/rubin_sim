import numpy as np
from rubin_sim.scheduler.utils import (
    empty_observation,
    set_default_nside,
    hp_in_lsst_fov,
    read_fields,
    hp_in_comcam_fov,
    comcamTessellate,
)
import healpy as hp
from rubin_sim.scheduler.thomson import xyz2thetaphi, thetaphi2xyz
from rubin_sim.scheduler.detailers import Zero_rot_detailer

__all__ = ["BaseSurvey", "BaseMarkovDF_survey"]


class BaseSurvey(object):
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
        survey_name="",
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
            self.detailers = [Zero_rot_detailer(nside=nside)]
        else:
            self.detailers = detailers

        # Scheduled observations
        self.scheduled_obs = scheduled_obs

    def get_scheduled_obs(self):
        return self.scheduled_obs

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


def rotx(theta, x, y, z):
    """rotate the x,y,z points theta radians about x axis"""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xp = x
    yp = y * cos_t + z * sin_t
    zp = -y * sin_t + z * cos_t
    return xp, yp, zp


class BaseMarkovDF_survey(BaseSurvey):
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
        survey_name="",
        nside=None,
        seed=42,
        dither=True,
        detailers=None,
        camera="LSST",
        area_required=None,
        npositions=7305,
    ):

        super(BaseMarkovDF_survey, self).__init__(
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
        if self.camera == "LSST":
            self.fields_init = read_fields()
        elif self.camera == "comcam":
            self.fields_init = comcamTessellate()
        else:
            ValueError('camera %s unknown, should be "LSST" or "comcam"' % camera)
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

    def _hp2fieldsetup(self, ra, dec, leafsize=100):
        """Map each healpixel to nearest field. This will only work if healpix
        resolution is higher than field resolution.
        """
        if self.camera == "LSST":
            pointing2hpindx = hp_in_lsst_fov(nside=self.nside)
        elif self.camera == "comcam":
            pointing2hpindx = hp_in_comcam_fov(nside=self.nside)

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
        dec = self.fields_init["dec"] + 0

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
            reward_temp = self.reward + 0
            mask = np.isnan(reward_temp)
            reward_temp[mask] = hp.UNSEEN
            self.reward_smooth = hp.sphtfunc.smoothing(
                reward_temp, fwhm=self.smoothing_kernel, verbose=False
            )
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

            if np.any(np.isinf(self.reward)):
                self.reward = np.inf
        else:
            # If not feasable, negative infinity reward
            self.reward = -np.inf
            return self.reward
        if self.smoothing_kernel is not None:
            self.smooth_reward()

        if self.area_required is not None:
            good_area = np.where(np.abs(self.reward) >= 0)[0].size * hp.nside2pixarea(
                self.nside
            )
            if good_area < self.area_required:
                self.reward = -np.inf

        return self.reward

    def generate_observations_rough(self, conditions):

        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields(conditions)
            self.night = conditions.night.copy()

        # XXX Use self.reward to decide what to observe.
        return None
