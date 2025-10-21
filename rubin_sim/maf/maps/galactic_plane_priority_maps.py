__all__ = (
    "gp_priority_map_components_to_keys",
    "galplane_priority_map",
    "GalacticPlanePriorityMap",
)

import os
import warnings

import healpy as hp
import numpy as np
from astropy.io import fits
from rubin_scheduler.data import get_data_dir

from rubin_sim.maf.maps import BaseMap
from rubin_sim.maf.utils import radec2pix


def gp_priority_map_components_to_keys(filtername, science_map):
    """A convenience function to help keep the map key
    formats in sync in various places.
    """
    return f"galplane_priority_{science_map}:{filtername}"


def gp_priority_map_keys_to_components(mapname):
    s, f = mapname.replace("galplane_priority_", "").split(":")
    return f, s


def galplane_priority_map(
    nside=64,
    get_keys=False,
    ra=None,
    dec=None,
    pixels=None,
    interp=False,
    map_path=None,
    use_alt_maps=False,
):
    """Reads and saves the galactic plane priority maps.

    Parameters
    ----------
    nside : `int`
        Healpixel resolution (2^x). At present, this must be 64.
    get_keys : `bool`, opt
        Set this to True to retrieve *only* the keys
        (such as the science map names) for the maps.
        Default False.
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
        Note that the pixels in the healpix array MUST come from a
        healpix grid with the same nside as the galactic plane priority map.
        Using different nsides can potentially fail silently.
    interp : `bool`, opt
        Should returned values be interpolated (True)
        or just nearest neighbor (False).
        Default False.
    map_path : `str`, opt
        Path to directory containing dust map files.
        Default None, uses $RUBIN_SIM_DATA_DIR/maps.
    use_alt_maps : `bool`, opt
        Use the priority_GalPlane_footprint_alt_map_data_{ugrizysum}.fits
        files instead of the default
        priority_galPlane_footprint_map_data_{ugrizysum}.fits files.
    """
    # This is a function that will read the galactic plane priority map data
    # and hold onto it indefinitely
    # this also lets us use a range of slicers, as it will set the slice_point
    # data appropriately.

    # This function's primary goal is to return this information to the map,
    # to use for the slicer.
    # So you MUST specify ra/dec or pixels -- or only retireve the keys
    if get_keys is False:
        if (ra is None) & (dec is None) & (pixels is None):
            raise RuntimeError("Need to set ra,dec or pixels.")

    # This reads and stores the galactic plane priority maps
    # The galactic plane priority maps are only available in nside 64
    # There are several different versions of the map -
    # but we will almost always run all of the galactic plane metrics
    # together, so we'll just read them all at once here
    if nside != 64:
        raise RuntimeError("Currently only available with nside=64")

    # Do we need to read from disk?
    if not hasattr(galplane_priority_map, "maps"):
        galplane_priority_map.nside = 64
        galplane_priority_map.filterlist = ["u", "g", "r", "i", "z", "y", "sum"]
        if map_path is not None:
            data_dir = map_path
        else:
            data_dir = os.path.join(get_data_dir(), "maps", "GalacticPlanePriorityMaps")
        galplane_priority_map.maps = {}
        science_maps = []
        for f in galplane_priority_map.filterlist:
            if use_alt_maps:
                filename = f"priority_GalPlane_footprint_alt_map_data_{f}.fits"
            else:
                filename = f"priority_GalPlane_footprint_map_data_{f}.fits"
            mapfile = os.path.join(data_dir, filename)
            with fits.open(mapfile) as hdul:
                galplane_priority_map.maps[f] = hdul[1].data
                print(f"Read galplane priority map from {mapfile}")
                science_maps += list(galplane_priority_map.maps[f].dtype.names)
            science_maps = list(set(science_maps))
            galplane_priority_map.science_maps = science_maps

    galplane_priority_map.keys = [
        gp_priority_map_components_to_keys(f, s)
        for f in galplane_priority_map.filterlist
        for s in galplane_priority_map.science_maps
    ]

    if get_keys:
        return galplane_priority_map.keys

    if pixels is not None:
        # Assume if pixels is set, then interp is irrelevant
        maps = {}
        for k in galplane_priority_map.keys:
            f, s = gp_priority_map_keys_to_components(k)
            maps[k] = galplane_priority_map.maps[f][s][pixels]
    # BUT, if we were provided ra/dec then have to see if we should interpolate
    else:
        if interp:  # find interpolated values at given ra/dec
            maps = {}
            for k in galplane_priority_map.keys:
                f, s = gp_priority_map_keys_to_components(k)
                maps[k] = hp.get_interp_val(galplane_priority_map.maps[f][s], np.pi / 2.0 - dec, ra)
        else:  # just find nearest neighbor pixel value and return those
            pixels = radec2pix(nside, ra, dec)
            maps = {}
            for k in galplane_priority_map.keys:
                f, s = gp_priority_map_keys_to_components(k)
                maps[k] = galplane_priority_map.maps[f][s][pixels]
    return maps


class GalacticPlanePriorityMap(BaseMap):
    """Add the galactic plane priority map data to the slice points.

    This calls galactic_plane_priority_map to read the map data, and then
    assigns the appropriate values to each slice_point.
    If the slicer is an nside=64 healpix slicer, this is trivial.
    (other use-cases currently experimental and not supported).

    Add keys corresponding to each of the galplane priority map elements.

    Parameters
    ----------
    interp : `bool`, opt
        Interpolate the dust map at each slice_point (True)
        or just use the nearest value (False).
        Default is False.
    nside : `int`, opt
        Default nside value to read the dust map from disk.
        Primarily useful if the slicer is not a healpix slicer.
        Default 64.
    map_path : `str`, opt
        Define a path to the directory holding the dust map files.
        Default None, which uses RUBIN_SIM_DATA_DIR.
    """

    def __init__(self, interp=False, nside=64, map_path=None):
        self.keynames = galplane_priority_map(get_keys=True)
        self.interp = interp
        self.nside = nside
        self.map_path = map_path

    def run(self, slice_points):
        # If the slicer has nside,
        # it's a healpix slicer so we can read the map directly
        if "nside" in slice_points:
            if slice_points["nside"] != self.nside:
                warnings.warn(
                    f"Slicer value of nside {slice_points['nside']} different "
                    f"from map value {self.nside}, using slicer value"
                )
            maps = galplane_priority_map(
                slice_points["nside"],
                pixels=slice_points["sid"],
                map_path=self.map_path,
            )
            for key in self.keynames:
                slice_points[key] = maps[key]
        # Not a healpix slicer, look up values based on RA,dec
        # with possible interpolation
        else:
            maps = galplane_priority_map(
                self.nside,
                ra=slice_points["ra"],
                dec=slice_points["dec"],
                interp=self.interp,
                map_path=self.map_path,
            )
            for key in self.keynames:
                slice_points[key] = maps[key]

        return slice_points
