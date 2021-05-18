import numpy as np

__all__ = ['gnomonic_project_toxy', 'gnomonic_project_tosky']


def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, RAcen, Deccen):
    """Calculate RA/Dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/Dec in radians."""
    denom = np.cos(Deccen) - y * np.sin(Deccen)
    RA = RAcen + np.arctan2(x, denom)
    Dec = np.arctan2(np.sin(Deccen) + y * np.cos(Deccen), np.sqrt(x*x + denom*denom))
    return RA, Dec
