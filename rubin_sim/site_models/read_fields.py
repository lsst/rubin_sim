__all__ = ("read_fields", "_read_fields")

import os

import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.utils import _ra_dec_from_xyz


def _read_fields(filename=None):
    """Read in field positions.

    Parameters
    ----------
    filename : str (None)
        File to read. Defaults to icover.3.5292.23.0.txt origianlly from:
        http://neilsloane.com/icosahedral.codes/index.html

    Returns
    -------
    ra, dec : np.arrays of RA,dec values in radians
    """
    if filename is None:
        filename = os.path.join(get_data_dir(), "site_models/icover.3.5292.23.0.txt")
    cov = np.genfromtxt(filename)
    x = cov[0::3]
    y = cov[1::3]
    z = cov[2::3]

    ra, dec = _ra_dec_from_xyz(x, y, z)
    return ra, dec


def read_fields(filename=None):
    """Read in field positions.

    Parameters
    ----------
    filename : str (None)
        File to read. Defaults to icover.3.5292.23.0.txt origianlly from:
        http://neilsloane.com/icosahedral.codes/index.html

    Returns
    -------
    ra, dec : np.arrays of RA,dec values in degrees
    """
    ra, dec = _read_fields(filename=filename)
    return np.degrees(ra), np.degrees(dec)
