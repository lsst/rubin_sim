__all__ = ("dark_sky",)

import os

import healpy as hp
import numpy as np

from rubin_sim.data import get_data_dir


def dark_sky(nside=32):
    """Load an array of HEALpix maps that have the darkest expected sky
    backgrounds per filter.

    Parameters
    ----------
    nside : `int` (32)
        Desired nside resolution (default=32).

    Returns
    -------
    dark_sky_data : `np.ndarray`
        Named array with dark sky data for each band.
    """
    if not hasattr(dark_sky, "data"):
        # Load up the data
        data_dir = get_data_dir()
        data = np.load(os.path.join(data_dir, "skybrightness_pre", "dark_maps.npz"))
        dark_sky.data = data["dark_maps"].copy()
        data.close()

    dark_sky_data = np.empty(hp.nside2npix(nside), dtype=dark_sky.data.dtype)

    for band in dark_sky_data.dtype.names:
        dark_sky_data[band] = hp.pixelfunc.ud_grade(dark_sky.data[band], nside_out=nside)

    return dark_sky_data
