__all__ = ("eb_vhp",)

import os

import healpy as hp
import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.maf.utils import radec2pix


def eb_vhp(nside, ra=None, dec=None, pixels=None, interp=False, map_path=None):
    """Read in a healpix dust map and return values for given RA, Dec values.

    This is primarily a tool for the rubin_sim.maf.DustMap class.

    nside : `int`
        Healpixel resolution (2^x).
    ra : `np.ndarray` or `float`, opt
        RA (can take numpy array).
        Default None sets up healpix array of nside. Radians.
    dec : `np.ndarray` or `float`, opt
        Dec (can take numpy array).
        Default None set up healpix array of nside. Radians.
    pixels : `np.ndarray`, opt
        Healpixel IDs, to sub-select particular healpix points.
        Default uses all points.
        NOTE - to use a healpix map, set pixels and not ra/dec.
    interp : `bool`, opt
        Should returned values be interpolated (True)
        or just nearest neighbor (False)
    map_path : `str`, opt
        Path to directory containing dust map files.
    """

    if (ra is None) & (dec is None) & (pixels is None):
        raise RuntimeError("Need to set ra,dec or pixels.")

    # Load the map
    if map_path is not None:
        ebv_data_dir = map_path
    else:
        ebv_data_dir = os.path.join(get_data_dir(), "maps", "DustMaps")
    if not hasattr(eb_vhp, "nside"):
        eb_vhp.nside = nside

    if (not hasattr(eb_vhp, "dustmap")) | (eb_vhp.nside != nside):
        eb_vhp.nside = nside
        filename = "dust_nside_%i.npz" % eb_vhp.nside
        eb_vhp.dustMap = np.load(os.path.join(ebv_data_dir, filename))["ebvMap"]

    # If we are interpolating to arbitrary positions
    if interp:
        result = hp.get_interp_val(eb_vhp.dustMap, np.pi / 2.0 - dec, ra)
    else:
        # If we know the pixel indices we want
        if pixels is not None:
            result = eb_vhp.dustMap[pixels]
        # Look up
        else:
            pixels = radec2pix(eb_vhp.nside, ra, dec)
            result = eb_vhp.dustMap[pixels]

    return result
