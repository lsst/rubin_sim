"""Footprints: Some relevant LSST footprints, including utilities to build them.

The goal here is to make it easy to build typical target maps and then their associated combined
survey inputs (maps in each filter, including scaling between filters; the associated cloud and
sky brightness maps that would have limits for WFD, etc.).

For generic use for defining footprints from scratch, there is also a utility that simply generates
the healpix points across the sky, along with their corresponding RA/Dec/Galactic l,b/Ecliptic l,b values.
"""
__all__ = (
    "ra_dec_hp_map",
    "generate_all_sky",
    "get_dustmap",
    "wfd_healpixels",
    "wfd_no_gp_healpixels",
    "wfd_bigsky_healpixels",
    "wfd_no_dust_healpixels",
    "scp_healpixels",
    "nes_healpixels",
    "galactic_plane_healpixels",
    "magellanic_clouds_healpixels",
    "ConstantFootprint",
    "generate_goal_map",
    "standard_goals",
    "calc_norm_factor",
    "filter_count_ratios",
    "StepLine",
    "Footprints",
    "Footprint",
    "StepSlopes",
    "BasePixelEvolution",
    "combo_dust_fp",
    "slice_wfd_area_quad",
    "slice_wfd_indx",
    "slice_quad_galactic_cut",
    "make_rolling_footprints",
)

import os

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_sim.data import get_data_dir
from rubin_sim.utils import Site, _angular_separation, _hpid2_ra_dec, angular_separation

from .utils import IntRounded, set_default_nside


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
    mjd_start : float
        The starting date of the survey.
    sun_ra_start : float
        The RA of the sun at the start of the survey
    nslice : int (2)
        How much to slice the sky up. Can be 2, 3, 4, or 6.
    scale : float (0.8)
        The strength of the rolling, value of 1 is full power rolling, zero is no rolling.
    wfd_indx : array of ints (none)
        The indices of the HEALpix map that are to be included in the rolling.
    order_roll : int (0)
        Change the order of when bands roll. Default 0.
    n_cycles : int (None)
        Number of complete rolling cycles to attempt. If None, defaults to 3
        full cycles for nslice=2, 2 cycles for nslice=3 or 4, and 1 cycle for
        nslice=6.
    n_constant_start : int (3)
        The number of constant non-rolling seasons to start with. Anything less
        than 3 results in rolling starting before the entire sky has had a constant year.
    n_constant_end : int (6)
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
        non_wfd_indx = np.where(hp_footprints["r"] != 1)[0]

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
    nslice : int (2)
        The number of slices to make, can be 2 or 3.
    wfd_indx : array of ints
        The indices of target_map that should be used for rolling. If None, assumes
        the rolling area should be where target_map['r'] == 1.
    """

    ra, dec = ra_dec_hp_map(nside=hp.npix2nside(target_map["r"].size))

    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    gal_lon, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg

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
    nslice : int (2)
        The number of slices to divide the sky into (gets doubled). Default is 2
    wfd_indx : array of int (None)
        The indices of the healpix map to consider as part of the WFD area that will be split. If
        set to None, the pixels where target_map['r'] == 1 are considered as WFD.
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
    """Helper class that can be used to describe the time evolution of a HEALpix in a footprint"""

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
    period : float (365.25)
        The period to use
    rise : float (1.)
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
    period : float (365.25)
        The period to use
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
        # Set the phase of each healpixel. If RA to sun is zero, we are at phase np.pi/2.
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
        """take an array and convert it to labled struc array"""
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

    def __call__(self, mjd, array=False, norm=True):
        """
        Returns
        -------
        a numpy array with the normalized number of observations that should be at each HEALpix.
        Multiply by the number of HEALpix observations (all filters), to convert to the number of observations
        desired.
        """
        self._update_mjd(mjd, norm=norm)
        # if array:
        #    return self.current_footprints
        # else:
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
        # Should probably run a check that all the footprints are compatible (same nside, etc)
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


def get_dustmap(nside=None):
    if nside is None:
        nside = set_default_nside()
    ebv_data_dir = os.path.join(get_data_dir(), "scheduler")
    filename = "dust_maps/dust_nside_%i.npz" % nside
    dustmap = np.load(os.path.join(ebv_data_dir, filename))["ebvMap"]
    return dustmap


def generate_all_sky(nside=None, elevation_limit=20, mask=hp.UNSEEN):
    """Set up a healpix map over the entire sky.
    Calculate RA & Dec, Galactic l & b, Ecliptic l & b, for all healpixels.
    Calculate max altitude, to set to  areas which LSST cannot reach (set these to hp.unseen).

    This is intended to be a useful tool to use to set up target maps, beyond the standard maps
    provided in these utilities. Masking based on RA, Dec, Galactic or Ecliptic lat and lon is easier.

    Parameters
    ----------
    nside : int, optional
        Resolution for the healpix maps.
        Default None uses rubin_sim.scheduler.utils.set_default_nside to set default (often 32).
    elevation_limit : float, optional
        Elevation limit for map.
        Parts of the sky which do not reach this elevation limit will be set to mask.
    mask : float, optional
        Mask value for 'unreachable' parts of the sky, defined as elevation < 20.
        Note that the actual limits will be set elsewhere, using the observatory model.
        This limit is for use when understanding what the maps could look like.

    Returns
    -------
    dict of np.ndarray
        Returns map, RA/Dec, Gal l/b, Ecl l/b (each an np.ndarray IN RADIANS) in a dictionary.
    """
    if nside is None:
        nside = set_default_nside()

    # Calculate coordinates of everything.
    skymap = np.zeros(hp.nside2npix(nside), float)
    ra, dec = ra_dec_hp_map(nside=nside)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
    eclip_lat = coord.barycentrictrueecliptic.lat.deg
    eclip_lon = coord.barycentrictrueecliptic.lon.deg
    gal_lon = coord.galactic.l.deg
    gal_lat = coord.galactic.b.deg

    # Calculate max altitude (when on meridian).
    lsst_site = Site("LSST")
    elev_max = np.pi / 2.0 - np.abs(dec - lsst_site.latitude_rad)
    skymap = np.where(IntRounded(elev_max) >= IntRounded(np.radians(elevation_limit), skymap, mask))

    return {
        "map": skymap,
        "ra": np.degrees(ra),
        "dec": np.degrees(dec),
        "eclip_lat": eclip_lat,
        "eclip_lon": eclip_lon,
        "gal_lat": gal_lat,
        "gal_lon": gal_lon,
    }


def wfd_healpixels(nside=None, dec_min=-62.5, dec_max=3.6):
    """
    Define a region based on declination limits only.

    Parameters
    ----------
    nside : int, optional
        Resolution for the healpix maps.
        Default None uses rubin_sim.scheduler.utils.set_default_nside to set default (often 32).
    dec_min : float, optional
        Minimum declination of the region (deg). Default -62.5.
    dec_max : float, optional
        Maximum declination of the region (deg). Default 3.6.

    Returns
    -------
    np.ndarray
        Healpix map with regions in declination-limited 'wfd' region as 1.
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size, float)
    dec = IntRounded(dec)
    good = np.where((dec >= IntRounded(np.radians(dec_min))) & (dec <= IntRounded(np.radians(dec_max))))
    result[good] = 1
    return result


def wfd_no_gp_healpixels(
    nside,
    dec_min=-62.5,
    dec_max=3.6,
    center_width=10.0,
    end_width=4.0,
    gal_long1=290.0,
    gal_long2=70.0,
):
    """
    Define a wide fast deep region with a galactic plane limit.

    Parameters
    ----------
    nside : int, optional
        Resolution for the healpix maps.
        Default None uses rubin_sim.scheduler.utils.set_default_nside to set default (often 32).
    dec_min : float, optional
        Minimum declination of the region (deg).
    dec_max : float, optional
        Maximum declination of the region (deg).
    center_width : float, optional
        Width across the central part of the galactic plane region.
    end_width : float, optional
        Width across the remainder of the galactic plane region.
    gal_long1 : float, optional
        Longitude at which to start tapering from center_width to end_width.
    gal_long2 : float, optional
        Longitude at which to stop tapering from center_width to end_width.

    Returns
    -------
    np.ndarray
        Healpix map with regions in declination-limited 'wfd' region as 1.
    """
    wfd_dec = wfd_healpixels(nside, dec_min=dec_min, dec_max=dec_max)
    gp = galactic_plane_healpixels(
        nside=nside,
        center_width=center_width,
        end_width=end_width,
        gal_long1=gal_long1,
        gal_long2=gal_long2,
    )
    sky = np.where(wfd_dec - gp > 0, wfd_dec - gp, 0)
    return sky


def wfd_bigsky_healpixels(nside):
    sky = wfd_no_gp_healpixels(
        nside,
        dec_min=-72.25,
        dec_max=12.4,
        center_width=14.9,
        gal_long1=0,
        gal_long2=360,
    )
    return sky


def wfd_no_dust_healpixels(nside, dec_min=-72.25, dec_max=12.4, dust_limit=0.19):
    """Define a WFD region with a dust extinction limit.

    Parameters
    ----------
    nside : int, optional
        Resolution for the healpix maps.
        Default None uses rubin_sim.scheduler.utils.set_default_nside to set default (often 32).
    dec_min : float, optional
        Minimum dec of the region (deg). Default -72.5 deg.
    dec_max : float, optional.
        Maximum dec of the region (deg). Default 12.5 deg.
        1.75 is the FOV radius in deg.
    dust_limit : float, None
        Remove pixels with E(B-V) values greater than dust_limit from the footprint.

    Returns
    -------
    result : numpy array
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    dustmap = get_dustmap(nside)

    result = np.zeros(ra.size, float)
    # First set based on dec range.
    dec = IntRounded(dec)
    good = np.where((dec >= IntRounded(np.radians(dec_min))) & (dec <= IntRounded(np.radians(dec_max))))
    result[good] = 1
    # Now remove areas with dust extinction beyond the limit.
    result = np.where(dustmap >= dust_limit, 0, result)
    return result


def scp_healpixels(nside=None, dec_max=-60.0):
    """
    Define the South Celestial Pole region. Return a healpix map with SCP pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size, float)
    good = np.where(IntRounded(dec) < IntRounded(np.radians(dec_max)))
    result[good] = 1
    return result


def nes_healpixels(nside=None, min_eb=-30.0, max_eb=10.0, dec_min=2.8):
    """
    Define the North Ecliptic Spur region. Return a healpix map with NES pixels as 1.

    Parameters
    ----------
    nside : int
        A valid healpix nside
    min_eb : float (-30.)
        Minimum barycentric true ecliptic latitude (deg)
    max_eb : float (10.)
        Maximum barycentric true ecliptic latitude (deg)
    dec_min : float (2.8)
        Minimum dec in degrees

    Returns
    -------
    result : numpy array
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size, float)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    eclip_lat = IntRounded(eclip_lat)
    dec = IntRounded(dec)
    good = np.where(
        (eclip_lat > IntRounded(np.radians(min_eb)))
        & (eclip_lat < IntRounded(np.radians(max_eb)))
        & (dec > IntRounded(np.radians(dec_min)))
    )
    result[good] = 1

    return result


def galactic_plane_healpixels(nside=None, center_width=10.0, end_width=4.0, gal_long1=290.0, gal_long2=70.0):
    """
    Define a Galactic Plane region.

    Parameters
    ----------
    nside : int, optional
        Resolution for the healpix maps.
        Default None uses rubin_sim.scheduler.utils.set_default_nside to set default (often 32).
    center_width : float, optional
        Width at the center of the galactic plane region.
    end_width : float, optional
        Width at the remainder of the galactic plane region.
    gal_long1 : float, optional
        Longitude at which to start the GP region.
    gal_long2 : float, optional
        Longitude at which to stop the GP region.
        Order matters for gal_long1 / gal_long2!

    Returns
    -------
    np.ndarray
        Healpix map with galactic plane regions set to 1.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = ra_dec_hp_map(nside=nside)

    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    gal_lon, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg
    # Reject anything beyond the central width.
    sky = np.where(np.abs(gal_lat) < center_width, 1, 0)
    # Apply the galactic longitude cuts, so that plane goes between gal_long1 to gal_long2.
    # This is NOT the shortest distance between the angles.
    gp_length = (gal_long2 - gal_long1) % 360
    # If the length is greater than 0 then we can add additional cuts.
    if gp_length > 0:
        # First, remove anything outside the gal_long1/gal_long2 region.
        sky = np.where(IntRounded((gal_lon - gal_long1) % 360) < IntRounded(gp_length), sky, 0)
        # Add the tapers.
        # These slope from the center (gp_center @ center_width)
        # to the edge (gp_center + gp_length/2 @ end_width).
        half_width = gp_length / 2.0
        slope = (center_width - end_width) / half_width
        gp_center = (gal_long1 + half_width) % 360
        gp_dist = gal_lon - gp_center
        gp_dist = np.abs(np.where(IntRounded(gp_dist) > IntRounded(180), (180 - gp_dist) % 180, gp_dist))
        lat_limit = np.abs(center_width - slope * gp_dist)
        sky = np.where(IntRounded(np.abs(gal_lat)) < IntRounded(lat_limit), sky, 0)
    return sky


def magellanic_clouds_healpixels(nside=None, lmc_radius=10, smc_radius=5):
    """
    Define the Galactic Plane region. Return a healpix map with GP pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(hp.nside2npix(nside))

    lmc_ra = np.radians(80.893860)
    lmc_dec = np.radians(-69.756126)
    lmc_radius = np.radians(lmc_radius)

    smc_ra = np.radians(13.186588)
    smc_dec = np.radians(-72.828599)
    smc_radius = np.radians(smc_radius)

    dist_to_lmc = _angular_separation(lmc_ra, lmc_dec, ra, dec)
    lmc_pix = np.where(IntRounded(dist_to_lmc) < IntRounded(lmc_radius))
    result[lmc_pix] = 1

    dist_to_smc = _angular_separation(smc_ra, smc_dec, ra, dec)
    smc_pix = np.where(IntRounded(dist_to_smc) < IntRounded(smc_radius))
    result[smc_pix] = 1
    return result


def generate_goal_map(
    nside=None,
    nes_fraction=0.3,
    wfd_fraction=1.0,
    scp_fraction=0.4,
    gp_fraction=0.2,
    nes_min_eb=-30.0,
    nes_max_eb=10,
    nes_dec_min=3.6,
    scp_dec_max=-62.5,
    gp_center_width=10.0,
    gp_end_width=4.0,
    gp_long1=290.0,
    gp_long2=70.0,
    wfd_dec_min=-62.5,
    wfd_dec_max=3.6,
    generate_id_map=False,
):
    """
    Handy function that will put together a target map in the proper order.
    """
    if nside is None:
        nside = set_default_nside()

    # Note, some regions overlap, thus order regions are added is important.
    result = np.zeros(hp.nside2npix(nside), dtype=float)
    id_map = np.zeros(hp.nside2npix(nside), dtype=int)
    pid = 1
    prop_name_dict = dict()

    if nes_fraction > 0.0:
        nes = nes_healpixels(nside=nside, min_eb=nes_min_eb, max_eb=nes_max_eb, dec_min=nes_dec_min)
        result[np.where(nes != 0)] = 0
        result += nes_fraction * nes
        id_map[np.where(nes != 0)] = 1
        pid += 1
        prop_name_dict[1] = "NorthEclipticSpur"

    if wfd_fraction > 0.0:
        wfd = wfd_healpixels(nside=nside, dec_min=wfd_dec_min, dec_max=wfd_dec_max)
        result[np.where(wfd != 0)] = 0
        result += wfd_fraction * wfd
        id_map[np.where(wfd != 0)] = 3
        pid += 1
        prop_name_dict[3] = "WideFastDeep"

    if scp_fraction > 0.0:
        scp = scp_healpixels(nside=nside, dec_max=scp_dec_max)
        result[np.where(scp != 0)] = 0
        result += scp_fraction * scp
        id_map[np.where(scp != 0)] = 2
        pid += 1
        prop_name_dict[2] = "SouthCelestialPole"

    if gp_fraction > 0.0:
        gp = galactic_plane_healpixels(
            nside=nside,
            center_width=gp_center_width,
            end_width=gp_end_width,
            gal_long1=gp_long1,
            gal_long2=gp_long2,
        )
        result[np.where(gp != 0)] = 0
        result += gp_fraction * gp
        id_map[np.where(gp != 0)] = 4
        pid += 1
        prop_name_dict[4] = "GalacticPlane"

    if generate_id_map:
        return result, id_map, prop_name_dict
    else:
        return result


def standard_goals(nside=None):
    """
    A quick function to generate the "standard" goal maps. This is the traditional WFD/mini survey footprint.
    """
    if nside is None:
        nside = set_default_nside()

    result = {}
    result["u"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.0,
        wfd_fraction=0.31,
        scp_fraction=0.15,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["g"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.2,
        wfd_fraction=0.44,
        scp_fraction=0.15,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["r"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.46,
        wfd_fraction=1.0,
        scp_fraction=0.15,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["i"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.46,
        wfd_fraction=1.0,
        scp_fraction=0.15,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["z"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.4,
        wfd_fraction=0.9,
        scp_fraction=0.15,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["y"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.0,
        wfd_fraction=0.9,
        scp_fraction=0.15,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    return result


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


def filter_count_ratios(target_maps):
    """Given the goal maps, compute the ratio of observations we want in each filter.
    This is basically:
    per filter, sum the number of pixels in each map and return this per filter value, normalized
    so that the total sum across all filters is 1.
    """
    results = {}
    all_norm = 0.0
    for key in target_maps:
        good = target_maps[key] > 0
        results[key] = np.sum(target_maps[key][good])
        all_norm += results[key]
    for key in results:
        results[key] /= all_norm
    return results


def combo_dust_fp(
    nside=32,
    wfd_weights={"u": 0.31, "g": 0.44, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
    wfd_dust_weights={"u": 0.13, "g": 0.13, "r": 0.25, "i": 0.25, "z": 0.25, "y": 0.25},
    nes_dist_eclip_n=10.0,
    nes_dist_eclip_s=-30.0,
    nes_south_limit=-5,
    ses_dist_eclip=9.0,
    nes_weights={"u": 0, "g": 0.2, "r": 0.46, "i": 0.46, "z": 0.4, "y": 0},
    dust_limit=0.19,
    wfd_north_dec=12.4,
    wfd_south_dec=-72.25,
    mc_wfd=True,
    outer_bridge_l=240,
    outer_bridge_width=10.0,
    outer_bridge_alt=13.0,
    bulge_radius=17.0,
    north_weights={"g": 0.03, "r": 0.03, "i": 0.03},
    north_limit=30.0,
    smooth=True,
    fwhm=5.7,
):
    """
    Based on the Olsen et al Cadence White Paper

    XXX---need to refactor and get rid of all the magic numbers everywhere.
    """

    ebv_data_dir = get_data_dir()
    filename = "scheduler/dust_maps/dust_nside_%i.npz" % nside
    dustmap = np.load(os.path.join(ebv_data_dir, filename))["ebvMap"]

    if smooth:
        dustmap = hp.smoothing(dustmap, fwhm=np.radians(fwhm))

    # wfd covers -72.25 < dec < 12.4. Avoid galactic plane |b| > 15 deg
    wfd_north = wfd_north_dec
    wfd_south = wfd_south_dec

    ra, dec = np.degrees(ra_dec_hp_map(nside=nside))
    wfd_no_dust = np.zeros(ra.size)
    wfd_dust = np.zeros(ra.size)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    gal_lon, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg

    # let's make a first pass here
    wfd_no_dust[np.where((dec > wfd_south) & (dec < wfd_north) & (dustmap < dust_limit))] = 1.0

    wfd_dust[np.where((dec > wfd_south) & (dec < wfd_north) & (dustmap > dust_limit))] = 1.0
    wfd_dust[np.where(dec < wfd_south)] = 1.0

    # Fill in values for WFD and WFD_dusty
    result = {}
    for key in wfd_weights:
        result[key] = wfd_no_dust + 0.0
        result[key][np.where(result[key] == 1)] = wfd_weights[key]
        result[key][np.where(wfd_dust == 1)] = wfd_dust_weights[key]

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    eclip_lat = coord.barycentrictrueecliptic.lat.deg

    # Any part of the NES that is too low gets pumped up
    nes_indx = np.where(
        ((eclip_lat < nes_dist_eclip_n) & (eclip_lat > nes_dist_eclip_s)) & (dec > nes_south_limit)
    )
    nes_hp_map = ra * 0
    nes_hp_map[nes_indx] = 1
    for key in result:
        result[key][np.where((nes_hp_map > 0) & (result[key] < nes_weights[key]))] = nes_weights[key]

    if mc_wfd:
        mag_clouds = magellanic_clouds_healpixels(nside)
        mag_clouds_indx = np.where(mag_clouds > 0)[0]
    else:
        mag_clouds_indx = []
    for key in result:
        result[key][mag_clouds_indx] = wfd_weights[key]

    # Put in an outer disk bridge
    outer_disk = np.where(
        (gal_lon < (outer_bridge_l + outer_bridge_width))
        & (gal_lon > (outer_bridge_l - outer_bridge_width))
        & (np.abs(gal_lat) < outer_bridge_alt)
    )
    for key in result:
        result[key][outer_disk] = wfd_weights[key]

    # Make a bulge go WFD
    dist_to_bulge = angular_separation(gal_lon, gal_lat, 0.0, 0.0)
    bulge_pix = np.where(dist_to_bulge <= bulge_radius)
    for key in result:
        result[key][bulge_pix] = wfd_weights[key]

    # Set South ecliptic to the WFD values
    ses_indx = np.where((np.abs(eclip_lat) < ses_dist_eclip) & (dec < nes_south_limit))
    for key in result:
        result[key][ses_indx] = wfd_weights[key]

    # Let's paint all the north as non-zero
    for key in north_weights:
        north = np.where((dec < north_limit) & (result[key] == 0))
        result[key][north] = north_weights[key]

    return result
