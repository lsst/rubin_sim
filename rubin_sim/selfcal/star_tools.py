__all__ = ("stars_project", "assign_patches")

import numpy as np
from rubin_scheduler.utils import gnomonic_project_toxy


def stars_project(stars, visit):
    """
    Project the stars to x,y plane for a given visit.
    """
    xtemp, ytemp = gnomonic_project_toxy(
        np.radians(stars["ra"]),
        np.radians(stars["decl"]),
        np.radians(visit["ra"]),
        np.radians(visit["dec"]),
    )
    # Rotate the field using the visit rotSkyPos.
    # Hope I got that sign right...
    sin_rot = np.sin(np.radians(visit["rotSkyPos"]))
    cos_rot = np.cos(np.radians(visit["rotSkyPos"]))
    stars["x"] = cos_rot * xtemp + sin_rot * ytemp
    stars["y"] = -1.0 * sin_rot * xtemp + cos_rot * ytemp

    stars["radius"] = (stars["x"] ** 2 + stars["y"] ** 2) ** 0.5
    return stars


def assign_patches(stars, visit, n_patches=16, radius_fov=1.8):
    """
    Assign PatchIDs to everything.
    Assume that stars have already been projected to x,y
    """
    maxx, maxy = gnomonic_project_toxy(0.0, np.radians(radius_fov), 0.0, 0.0)
    nsides = n_patches**0.5

    # This should move all coords to  0 < x < nsides-1
    px = np.floor((stars["x"] + maxy) / (2.0 * maxy) * nsides)
    py = np.floor((stars["y"] + maxy) / (2.0 * maxy) * nsides)

    stars["sub_patch"] = px + py * nsides
    stars["patch_id"] = stars["sub_patch"] + visit["observationId"] * n_patches
    return stars
