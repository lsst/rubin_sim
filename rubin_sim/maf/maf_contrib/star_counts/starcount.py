#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Calculates the number of stars in a given direction and between a given set of distances. For use with Field Star Count metric
import math
import sys

import numpy as np
from scipy.optimize import fsolve

from . import coords, stellardensity

# from rubin_sim.coordUtils import AstronomyBase
skyarea = 41253.0
distancebins = 51


def star_vols(d1, d2, area):
    distance_edges = (np.linspace((d1**3.0), (d2**3.0), num=distancebins)) ** (1.0 / 3)
    volumeshell = (area / skyarea) * (4.0 * np.pi / 3) * (distance_edges[1:] ** 3 - distance_edges[:-1] ** 3)
    distances = ((distance_edges[1:] ** 3 + distance_edges[:-1] ** 3) / 2.0) ** (1.0 / 3)
    return volumeshell, distances


def starcount(eq_ra, eq_dec, d1, d2):
    volumes, distances = star_vols(d1, d2, 9.62)
    # b_deg, l_deg=coords.eq_gal2(eq_ra, eq_dec)
    # b_deg, l_deg=AstrometryBase.equatorialToGalactic(eq_ra, eq_dec)
    b_deg, l_deg = coords.eq_gal3(eq_ra, eq_dec)
    positions = [coords.gal_cyn(b_deg, l_deg, x) for x in distances]
    densities = [stellardensity.stellardensity(x[0], x[2]) for x in positions]
    totalcount = np.sum(np.asarray(volumes) * np.asarray(densities))
    return totalcount


if __name__ == "__main__":
    print(
        starcount(
            float(sys.argv[1]),
            float(sys.argv[2]),
            float(sys.argv[3]),
            float(sys.argv[4]),
        )
    )
