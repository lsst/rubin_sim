"""Footprints: Take sky area maps and turn them into dynamic `footprint`
objects which understand seasons and time, in order to weight area on sky
appropriately for a given time.
"""
__all__ = (
    "ra_dec_hp_map",
    "calc_norm_factor",
    "calc_norm_factor_array",
    "StepLine",
    "Footprints",
    "Footprint",
    "StepSlopes",
    "ConstantFootprint",
    "BasePixelEvolution",
    "slice_wfd_area_quad",
    "slice_wfd_indx",
    "slice_quad_galactic_cut",
    "make_rolling_footprints",
)

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_sim.utils import _hpid2_ra_dec

from .utils import set_default_nside


def make_rolling_footprints(
    fp_hp=None,
    mjd_start=60218.0,
    sun_ra_start=3.27717639,
    nslice=2,
    scale=0.8,
    nside=32,
    wfd_indx=None,
    order_roll=0,
    n_cycles=None,
    n_constant_start=3,
    n_constant_end=6,
):
    """
    Generate rolling footprints

    Parameters
    ----------
    fp_hp : dict-like
        A dict with filtername keys and HEALpix map values
    mjd_start : `float`
        The starting date of the survey.
    sun_ra_start : `float`
        The RA of the sun at the start of the survey
    nslice : `int`
        How much to slice the sky up. Can be 2, 3, 4, or 6.
    scale : `float`
        The strength of the rolling, value of 1 is full power rolling.
        Zero is no rolling.
    wfd_indx : array of ints
        The indices of the HEALpix map that are to be included in the rolling.
    order_roll : `int`
        Change the order of when bands roll. Default 0.
    n_cycles : `int`
        Number of complete rolling cycles to attempt. If None, defaults to 3
        full cycles for nslice=2, 2 cycles for nslice=3 or 4, and 1 cycle for
        nslice=6.
    n_constant_start : `int`
        The number of constant non-rolling seasons to start with.
        Anything less than 3 results in rolling starting before the
        entire sky has had a constant year.
    n_constant_end : `int`
        The number of constant seasons to end the survey with. Defaults to 6.

    Returns
    -------
    Footprints object
    """

    nc_default = {2: 3, 3: 2, 4: 2, 6: 1}
    if n_cycles is None:
        n_cycles = nc_default[nslice]

    hp_footprints = fp_hp

    down = 1.0 - scale
    up = nslice - down * (nslice - 1)

    start = [1.0] * n_constant_start
    # After n_cycles, just go to no-rolling for 6 years.
    end = [1.0] * n_constant_end

    rolling = [up] + [down] * (nslice - 1)
    rolling = rolling * n_cycles

    rolling = np.roll(rolling, order_roll).tolist()

    all_slopes = [start + np.roll(rolling, i).tolist() + end for i in range(nslice)]

    fp_non_wfd = Footprint(mjd_start, sun_ra_start=sun_ra_start, nside=nside)
    rolling_footprints = []
    for i in range(nslice):
        step_func = StepSlopes(rise=all_slopes[i])
        rolling_footprints.append(
            Footprint(mjd_start, sun_ra_start=sun_ra_start, step_func=step_func, nside=nside)
        )

    wfd = hp_footprints["r"] * 0
    if wfd_indx is None:
        wfd_indx = np.where(hp_footprints["r"] == 1)[0]

    wfd[wfd_indx] = 1
    non_wfd_indx = np.where(wfd == 0)[0]

    split_wfd_indices = slice_quad_galactic_cut(hp_footprints, nslice=nslice, wfd_indx=wfd_indx)

    for key in hp_footprints:
        temp = hp_footprints[key] + 0
        temp[wfd_indx] = 0
        fp_non_wfd.set_footprint(key, temp)

        for i in range(nslice):
            # make a copy of the current filter
            temp = hp_footprints[key] + 0
            # Set the non-rolling area to zero
            temp[non_wfd_indx] = 0

            indx = split_wfd_indices[i]
            # invert the indices
            ze = temp * 0
            ze[indx] = 1
            temp = temp * ze
            rolling_footprints[i].set_footprint(key, temp)

    result = Footprints([fp_non_wfd] + rolling_footprints)
    return result


def slice_wfd_indx(target_map, nslice=2, wfd_indx=None):
    """
    simple map split
    """

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indx = np.where(target_map["r"] == 1)[0]
    wfd[wfd_indx] = 1
    wfd_accum = np.cumsum(wfd)
    split_wfd_indices = np.floor(np.max(wfd_accum) / nslice * (np.arange(nslice) + 1)).astype(int)
    split_wfd_indices = split_wfd_indices.tolist()
    split_wfd_indices = [0] + split_wfd_indices

    return split_wfd_indices


def slice_quad_galactic_cut(target_map, nslice=2, wfd_indx=None):
    """
    Helper function for generating rolling footprints

    Parameters
    ----------
    target_map : dict of HEALpix maps
        The final desired footprint as HEALpix maps. Keys are filter names
    nslice : `int`
        The number of slices to make, can be 2 or 3.
    wfd_indx : array of ints
        The indices of target_map that should be used for rolling.
        If None, assumes the rolling area should be where target_map['r'] == 1.
    """

    ra, dec = ra_dec_hp_map(nside=hp.npix2nside(target_map["r"].size))

    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    _, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg

    indx_north = np.intersect1d(np.where(gal_lat >= 0)[0], wfd_indx)
    indx_south = np.intersect1d(np.where(gal_lat < 0)[0], wfd_indx)

    splits_north = slice_wfd_area_quad(target_map, nslice=nslice, wfd_indx=indx_north)
    splits_south = slice_wfd_area_quad(target_map, nslice=nslice, wfd_indx=indx_south)

    slice_indx = []
    for j in np.arange(nslice):
        indx_temp = []
        for i in np.arange(j + 1, nslice * 2 + 1, nslice):
            indx_temp += indx_north[splits_north[i - 1] : splits_north[i]].tolist()
            indx_temp += indx_south[splits_south[i - 1] : splits_south[i]].tolist()
        slice_indx.append(indx_temp)

    return slice_indx


def slice_wfd_area_quad(target_map, nslice=2, wfd_indx=None):
    """
    Divide a healpix map in an intelligent way

    Parameters
    ----------
    target_map : dict of HEALpix arrays
        The input map to slice
    nslice : int
        The number of slices to divide the sky into (gets doubled).
    wfd_indx : array of int
        The indices of the healpix map to consider as part of the WFD area
        that will be split.
        If set to None, the pixels where target_map['r'] == 1 are
        considered as WFD.
    """
    nslice2 = nslice * 2

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indices = np.where(target_map["r"] == 1)[0]
    else:
        wfd_indices = wfd_indx
    wfd[wfd_indices] = 1
    wfd_accum = np.cumsum(wfd)
    split_wfd_indices = np.floor(np.max(wfd_accum) / nslice2 * (np.arange(nslice2) + 1)).astype(int)
    split_wfd_indices = split_wfd_indices.tolist()
    split_wfd_indices = [0] + split_wfd_indices

    return split_wfd_indices


class BasePixelEvolution:
    """Helper class that can be used to describe the time evolution of a
    HEALpix in a footprint.
    """

    def __init__(self, period=365.25, rise=1.0, t_start=0.0):
        self.period = period
        self.rise = rise
        self.t_start = t_start

    def __call__(self, mjd_in, phase):
        pass


class StepLine(BasePixelEvolution):
    """
    Parameters
    ----------
    period : `float`
        The period to use
    rise : `float`
        How much the curve should rise every period
    """

    def __call__(self, mjd_in, phase):
        t = mjd_in + phase - self.t_start
        n_periods = np.floor(t / (self.period))
        result = n_periods * self.rise
        tphased = t % self.period
        step_area = np.where(tphased > self.period / 2.0)[0]
        result[step_area] += (tphased[step_area] - self.period / 2) * self.rise / (0.5 * self.period)
        result[np.where(t < 0)] = 0
        return result


class StepSlopes(BasePixelEvolution):
    """
    Parameters
    ----------
    period : `float`
        The period to use - typically should be a year.
    rise : np.array-like
        How much the curve should rise each period.
    """

    def __call__(self, mjd_in, phase):
        steps = np.array(self.rise)
        t = mjd_in + phase - self.t_start
        season = np.floor(t / (self.period))
        season = season.astype(int)
        plateus = np.cumsum(steps) - steps[0]
        result = plateus[season]
        tphased = t % self.period
        step_area = np.where(tphased > self.period / 2.0)[0]
        result[step_area] += (
            (tphased[step_area] - self.period / 2) * steps[season + 1][step_area] / (0.5 * self.period)
        )
        result[np.where(t < 0)] = 0

        return result


class Footprint:
    """An object to compute the desired survey footprint at a given time

    Parameters
    ----------
    mjd_start : float
        The MJD the survey starts on
    sun_ra_start : float
        The RA of the sun at the start of the survey (radians)

    """

    def __init__(
        self,
        mjd_start,
        sun_ra_start=0,
        nside=32,
        filters={"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5},
        period=365.25,
        step_func=None,
    ):
        self.period = period
        self.nside = nside
        if step_func is None:
            step_func = StepLine()
        self.step_func = step_func
        self.mjd_start = mjd_start
        self.sun_ra_start = sun_ra_start
        self.npix = hp.nside2npix(nside)
        self.filters = filters
        self.ra, self.dec = _hpid2_ra_dec(self.nside, np.arange(self.npix))
        # Set the phase of each healpixel.
        # If RA to sun is zero, we are at phase np.pi/2.
        self.phase = (-self.ra + self.sun_ra_start + np.pi / 2) % (2.0 * np.pi)
        self.phase = self.phase * (self.period / 2.0 / np.pi)
        # Empty footprints to start
        self.out_dtype = list(zip(filters, [float] * len(filters)))
        self.footprints = np.zeros((len(filters), self.npix), dtype=float)
        self.estimate = np.zeros((len(filters), self.npix), dtype=float)
        self.current_footprints = np.zeros((len(filters), self.npix), dtype=float)
        self.zero = self.step_func(0.0, self.phase)
        self.mjd_current = None

    def set_footprint(self, filtername, values):
        self.footprints[self.filters[filtername], :] = values

    def get_footprint(self, filtername):
        return self.footprints[self.filters[filtername], :]

    def _update_mjd(self, mjd, norm=True):
        if mjd != self.mjd_current:
            self.mjd_current = mjd
            t_elapsed = mjd - self.mjd_start

            norm_coverage = self.step_func(t_elapsed, self.phase)
            norm_coverage -= self.zero
            self.current_footprints = self.footprints * norm_coverage
            c_sum = np.sum(self.current_footprints)
            if norm:
                if c_sum != 0:
                    self.current_footprints = self.current_footprints / c_sum

    def arr2struc(self, inarr):
        """Take an array and convert it to labeled struc array"""
        result = np.empty(self.npix, dtype=self.out_dtype)
        for key in self.filters:
            result[key] = inarr[self.filters[key]]
        # Argle bargel, why doesn't this view work?
        # struc = inarr.view(dtype=self.out_dtype).squeeze()
        return result

    def estimate_counts(self, mjd, nvisits=2.2e6, fov_area=9.6):
        """Estimate the counts we'll get after some time and visits"""
        pix_area = hp.nside2pixarea(self.nside, degrees=True)
        pix_per_visit = fov_area / pix_area
        self._update_mjd(mjd, norm=True)
        self.estimate = self.current_footprints * pix_per_visit * nvisits
        return self.arr2struc(self.estimate)

    def __call__(self, mjd, norm=True):
        """
        Parameters
        ----------
        mjd : `float`
            Current MJD.
        norm : `bool`
            If normalized, the footprint retains the same range of values
            over time.

        Returns
        -------
        current_footprints : `np.ndarray`, (6, N)
            A numpy structured array with the updated normalized number of
            observations that should be requested at each Healpix.
            Multiply by the number of HEALpix observations (all filter), to
            convert to the number of observations desired.
        """
        self._update_mjd(mjd, norm=norm)
        return self.arr2struc(self.current_footprints)


class ConstantFootprint(Footprint):
    def __init__(self, nside=32, filters={"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}):
        self.nside = nside
        self.filters = filters
        self.npix = hp.nside2npix(nside)
        self.footprints = np.zeros((len(filters), self.npix), dtype=float)
        self.out_dtype = list(zip(filters, [float] * len(filters)))
        self.to_return = self.arr2struc(self.footprints)

    def __call__(self, mjd, array=False):
        return self.to_return


class Footprints(Footprint):
    """An object to combine multiple Footprint objects."""

    def __init__(self, footprint_list):
        self.footprint_list = footprint_list
        self.mjd_current = None
        self.current_footprints = 0
        # Should probably run a check that all the footprints are compatible
        # (same nside, etc)
        self.npix = footprint_list[0].npix
        self.out_dtype = footprint_list[0].out_dtype
        self.filters = footprint_list[0].filters
        self.nside = footprint_list[0].nside

        self.footprints = np.zeros((len(self.filters), self.npix), dtype=float)
        for fp in self.footprint_list:
            self.footprints += fp.footprints

    def set_footprint(self, filtername, values):
        pass

    def _update_mjd(self, mjd, norm=True):
        if mjd != self.mjd_current:
            self.mjd_current = mjd
            self.current_footprints = 0.0
            for fp in self.footprint_list:
                fp._update_mjd(mjd, norm=False)
                self.current_footprints += fp.current_footprints
            c_sum = np.sum(self.current_footprints)
            if norm:
                if c_sum != 0:
                    self.current_footprints = self.current_footprints / c_sum


def ra_dec_hp_map(nside=None):
    """
    Return all the RA,dec points for the centers of a healpix map, in radians.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    return ra, dec


def calc_norm_factor(goal_dict, radius=1.75):
    """Calculate how to normalize a Target_map_basis_function.
    This is basically:
    the area of the fov / area of a healpixel  /
    the sum of all of the weighted-healpix values in the footprint.

    Parameters
    -----------
    goal_dict : dict of healpy maps
        The target goal map(s) being used
    radius : float (1.75)
        Radius of the FoV (degrees)

    Returns
    -------
    Value to use as Target_map_basis_function norm_factor kwarg
    """
    all_maps_sum = 0
    for key in goal_dict:
        good = np.where(goal_dict[key] > 0)
        all_maps_sum += goal_dict[key][good].sum()
    nside = hp.npix2nside(goal_dict[key].size)
    hp_area = hp.nside2pixarea(nside, degrees=True)
    norm_val = radius**2 * np.pi / hp_area / all_maps_sum
    return norm_val


def calc_norm_factor_array(goal_map, radius=1.75):
    """Calculate how to normalize a Target_map_basis_function.
    This is basically:
    the area of the fov / area of a healpixel  /
    the sum of all of the weighted-healpix values in the footprint.

    Parameters
    -----------
    goal_map : recarray of healpy maps
        The target goal map(s) being used
    radius : float
        Radius of the FoV (degrees)

    Returns
    -------
    Value to use as Target_map_basis_function norm_factor kwarg
    """
    all_maps_sum = 0
    for key in goal_map.dtype.names:
        good = np.where(goal_map[key] > 0)
        all_maps_sum += goal_map[key][good].sum()
    nside = hp.npix2nside(goal_map[key].size)
    hp_area = hp.nside2pixarea(nside, degrees=True)
    norm_val = radius**2 * np.pi / hp_area / all_maps_sum
    return norm_val
