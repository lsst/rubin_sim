"""
This file contains coordinate transformation methods and utilities for converting an ra,dec coordinate set to
cartesian coordinates and to grid id using a spatial tree.
"""
__all__ = ("_build_tree",)

import numpy as np
from scipy.spatial import cKDTree as kdTree

from rubin_sim.utils.coordinate_transformations import _xyz_from_ra_dec


def _build_tree(ra, dec, leafsize=100, scale=None):
    """
    Build KD tree on simDataRA/Dec and set radius (via set_rad) for matching.

    Parameters
    ----------
    ra, dec : float (or arrays)
        RA and Dec values (in radians).
    leafsize : int (100)
        The number of Ra/Dec pointings in each leaf node.
    scale : float (None)
        If set, the values are scaled up, rounded, and converted to integers. Useful for
        forcing a set precision and preventing machine precision differences
    """
    if np.any(np.abs(ra) > np.pi * 2.0) or np.any(np.abs(dec) > np.pi * 2.0):
        raise ValueError("Expecting RA and Dec values to be in radians.")
    x, y, z = _xyz_from_ra_dec(ra, dec)
    if scale is not None:
        x = np.round(x * scale).astype(int)
        y = np.round(y * scale).astype(int)
        z = np.round(z * scale).astype(int)
    data = list(zip(x, y, z))
    if np.size(data) > 0:
        try:
            tree = kdTree(data, leafsize=leafsize, balanced_tree=False, compact_nodes=False)
        except TypeError:
            tree = kdTree(data, leafsize=leafsize)
    else:
        raise ValueError("ra and dec should have length greater than 0.")

    return tree
