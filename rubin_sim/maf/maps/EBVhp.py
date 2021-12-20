import numpy as np
import healpy as hp
import os
from rubin_sim.maf.utils import radec2pix
from rubin_sim.data import get_data_dir


__all__ = ["EBVhp"]


def EBVhp(nside, ra=None, dec=None, pixels=None, interp=False, mapPath=None):
    """
    Read in a healpix dust map and return values for given RA, Dec values

    nside: `int`
        Healpixel resolution (2^x).
    ra: `np.ndarray` or `float`, opt
        RA (can take numpy array). Default None sets up healpix array of nside.
    dec: `np.ndarray` or `float`, opt
        Dec (can take numpy array). Default None set up healpix array of nside.
    pixles: `np.ndarray`, opt
        Healpixel IDs, to sub-select particular healpix points. Default uses all points.
    interp: `bool`, opt
        Should returned values be interpolated (True) or just nearest neighbor(False)
    mapPath : `str`, opt
        Path to directory containing dust map files.
    """

    if (ra is None) & (dec is None) & (pixels is None):
        raise RuntimeError("Need to set ra,dec or pixels.")

    # Load the map
    if mapPath is not None:
        ebvDataDir = mapPath
    else:
        ebvDataDir = os.path.join(get_data_dir(), "maps", "DustMaps")
    if not hasattr(EBVhp, "nside"):
        EBVhp.nside = nside

    if (not hasattr(EBVhp, "dustmap")) | (EBVhp.nside != nside):
        EBVhp.nside = nside
        filename = "dust_nside_%i.npz" % EBVhp.nside
        EBVhp.dustMap = np.load(os.path.join(ebvDataDir, filename))["ebvMap"]

    # If we are interpolating to arbitrary positions
    if interp:
        result = hp.get_interp_val(EBVhp.dustMap, np.pi / 2.0 - dec, ra)
    else:
        # If we know the pixel indices we want
        if pixels is not None:
            result = EBVhp.dustMap[pixels]
        # Look up
        else:
            pixels = radec2pix(EBVhp.nside, ra, dec)
            result = EBVhp.dustMap[pixels]

    return result
