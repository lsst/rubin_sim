import numpy as np
import healpy as hp
import os
from rubin_sim.maf.utils import radec2pix
from rubin_sim.data import get_data_dir


__all__ = ['EBVhp']


def EBVhp(nside, ra=None, dec=None, pixels=None, interp=False):
    """
    Read in a healpix dust map and return values for given RA, Dec values

    nside: Healpixel resolution (2^x)
    ra: RA (can take numpy array)
    dec: Dec (can take numpy array)
    pixles: Healpixel IDs
    interp: Should returned values be interpolated (True) or just nearest neighbor(False)
    """

    if (ra is None) & (dec is None) & (pixels is None):
        raise RuntimeError("Need to set ra,dec or pixels.")

    # Load the map
    if not hasattr(EBVhp, 'nside'):
        EBVhp.nside = nside

    if (not hasattr(EBVhp, 'dustmap')) | (EBVhp.nside != nside):
        EBVhp.nside = nside
        ebvDataDir = os.path.join(get_data_dir(), 'maps')
        filename = 'DustMaps/dust_nside_%i.npz' % EBVhp.nside
        EBVhp.dustMap = np.load(os.path.join(ebvDataDir, filename))['ebvMap']

    # If we are interpolating to arbitrary positions
    if interp:
        result = hp.get_interp_val(EBVhp.dustMap, np.pi/2. - dec, ra)
    else:
        # If we know the pixel indices we want
        if pixels is not None:
            result = EBVhp.dustMap[pixels]
        # Look up
        else:
            pixels = radec2pix(EBVhp.nside, ra, dec)
            result = EBVhp.dustMap[pixels]

    return result
