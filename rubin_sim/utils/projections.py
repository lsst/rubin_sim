__all__ = ("gnomonic_project_toxy", "gnomonic_project_tosky")

import numpy as np


def gnomonic_project_toxy(ra1, dec1, r_acen, deccen):
    """Calculate x/y projection of ra1/dec1 in system with center at r_acen, deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(deccen) * np.sin(dec1) + np.cos(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)
    x = np.cos(dec1) * np.sin(ra1 - r_acen) / cosc
    y = (np.cos(deccen) * np.sin(dec1) - np.sin(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, r_acen, deccen):
    """Calculate RA/dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/dec in radians."""
    denom = np.cos(deccen) - y * np.sin(deccen)
    RA = r_acen + np.arctan2(x, denom)
    dec = np.arctan2(np.sin(deccen) + y * np.cos(deccen), np.sqrt(x * x + denom * denom))
    return RA, dec
