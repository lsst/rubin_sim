__all__ = ("ebv_3d_hp", "get_x_at_nearest_y")

import os
import warnings

import healpy as hp
import numpy as np
from astropy.io import fits
from rubin_scheduler.data import get_data_dir

from rubin_sim.maf.utils import radec2pix


def ebv_3d_hp(
    nside,
    map_file=None,
    ra=None,
    dec=None,
    pixels=None,
    interp=False,
):
    """Reads and saves a 3d dust extinction file from disk, return extinction
    at  specified points (ra/dec/ or pixels).

    Parameters
    ----------
    nside : `int`
        Healpixel resolution (2^x).
    map_file : `str`, opt
        Path to dust map file.
    ra : `np.ndarray` or `float`, opt
        RA (can take numpy array).
        Default None sets up healpix array of nside. Radians.
    dec : `np.ndarray` or `float`, opt
        Dec (can take numpy array).
        Default None set up healpix array of nside. Radians.
    pixels : `np.ndarray`, opt
        Healpixel IDs, to sub-select particular healpix points.
        Default uses all points.
        Easiest way to access healpix values.
        Note that the pixels in the healpix array MUST come from a h
        ealpix grid with the same nside as the ebv_3d_hp map.
        Using different nsides can potentially fail silently.
    interp : `bool`, opt
        Should returned values be interpolated (True)
        or just nearest neighbor (False).
        Default False.
    """
    if (ra is None) & (dec is None) & (pixels is None):
        raise RuntimeError("Need to set ra,dec or pixels.")

    # Set up path to map data
    if map_file is None:
        if nside == 64:
            map_name = "merged_ebv3d_nside64_defaults.fits"
            map_file = os.path.join(get_data_dir(), "maps/DustMaps3D", map_name)
        elif nside == 128:
            map_name = "merged_ebv3d_nside128_defaults.fits"
            map_file = os.path.join(get_data_dir(), "maps/DustMaps3D", map_name)
        else:
            raise Exception(
                f"map_file not specified, and nside {nside} not one of 64 or 128. "
                "Please specify a known map_file, as a basis for interpolation."
            )
    else:
        # Check if user specified map name but not full map path
        if not os.path.isfile(map_file):
            test_path = os.path.join(get_data_dir(), "maps/DustMaps3D", map_file)
            if os.path.isfile(test_path):
                map_file = test_path
    # Keep track of what nside and what map_file we're using
    if not hasattr(ebv_3d_hp, "map_file"):
        ebv_3d_hp.mapFile = map_file
        ebv_3d_hp.nside = nside

    # Do we need to re-read from disk?
    if (not hasattr(ebv_3d_hp, "ebvs")) | (not hasattr(ebv_3d_hp, "dists")) | (map_file != ebv_3d_hp.mapFile):
        ebv_3d_hp.mapFile = map_file
        ebv_3d_hp.nside = nside
        # Read map from disk
        hdul = fits.open(map_file)
        # hpids = hdul[0].data
        dists = hdul[1].data
        ebvs = hdul[2].data
        hdr = hdul[0].header
        # Close map file
        hdul.close()

        # Check additional healpix information from the header
        map_nside = hdr["NSIDE"]
        nested = hdr["NESTED"]
        if nside != map_nside:
            warnings.warn(
                f"Map nside {map_nside} did not match expected nside {nside}; "
                f"Will use nside from map data."
            )
            if pixels is not None:
                # We're just going to raise an exception here
                # because this could mean bad things.
                raise ValueError(
                    f"Map nside {map_nside} did not match expected nside {nside}, "
                    f"and pixels provided; this can potentially indicate a serious "
                    f"error. Make nsides match or specify ra/dec instead of pixels."
                )
            nside = map_nside
        # Nested healpix data will not match the healpix arrays
        # for the slicers (at this time)
        if nested:
            warnings.warn("Map has nested (not ring order) data; will reorder.")
            for i in np.arange(0, len(dists[0])):
                dists[:, i] = hp.reorder(dists[:, i], "nest", "ring")
                ebvs[:, i] = hp.reorder(ebvs[:, i], "nest", "ring")
        ebv_3d_hp.dists = dists
        ebv_3d_hp.ebvs = ebvs
        print(f"Read map {map_file} from disk")

    if pixels is not None:
        # Assume if pixels is set, then interp is irrelevant
        dists = ebv_3d_hp.dists[pixels]
        ebvs = ebv_3d_hp.ebvs[pixels]
    # BUT, if we were provided ra/dec then have to see if we should interpolate
    else:
        if interp:  # find interpolated values at given ra/dec
            distlen = len(ebv_3d_hp.dists[0])
            dists = np.zeros([len(ra), distlen], float)
            ebvs = np.zeros([len(ra), distlen], float)
            for i in np.arange(0, distlen):
                dists[:, i] = hp.get_interp_val(ebv_3d_hp.dists[:, i], np.pi / 2.0 - dec, ra)
                ebvs[:, i] = hp.get_interp_val(ebv_3d_hp.ebvs[:, i], np.pi / 2.0 - dec, ra)
        else:  # just fine nearest neighbor pixel value and return those
            pixels = radec2pix(nside, ra, dec)
            dists = ebv_3d_hp.dists[pixels]
            ebvs = ebv_3d_hp.ebvs[pixels]

    return dists, ebvs


def get_x_at_nearest_y(x, y, x_goal):
    """Given a goal x value, find y values at the closest x value.

    This could be used to fetch the nearest ebv value at a given distance,
    or the nearest distance to a given m-Mo value, for example.

    Parameters
    ----------
    x : `np.array`
        Can be either a map with x at each point in the map (2d array) or
        the x at a single point of the map (1d array)
    y : `np.array`
        Can be either a map with y at each point in the map (2d array) or
        the y at a single point of the map (1d array) -
        but should match x dimensionality
    x_goal : `float'
        The goal x value to look for the nearest y value

    Returns
    -------
    x_closest, y_closest : `np.array`, `np.array`
        1-d array of x and y (single value or over map).
    """
    if x.ndim == 2:
        idx_min = np.argmin(np.abs(x - x_goal), axis=1)
        i_expand = np.expand_dims(idx_min, axis=-1)
        x_closest = np.take_along_axis(x, i_expand, axis=-1).squeeze()
        y_closest = np.take_along_axis(y, i_expand, axis=-1).squeeze()
    else:
        idx_min = np.argmin(np.abs(x - x_goal))
        x_closest = x[idx_min]
        y_closest = y[idx_min]
    return x_closest, y_closest
