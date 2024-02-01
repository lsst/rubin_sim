#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Calculates the stellar density based off of
# Juric et al 2008 and Jackson et al 2002. For use with Field Star Count metric

import numpy as np

zsun = 25.0
rsun = 8000.0
density_rsun = 0.0364
f = 0.1


def diskprofile(R, Z, L, H):
    part1 = (-R / L) - (abs(Z + zsun) / H)
    part2 = np.exp(part1)
    part3 = np.exp(rsun / L)
    tot = density_rsun * part2 * part3
    return tot


def thindisk(R, Z):
    return diskprofile(R, Z, 2150.0, 245.0)


def thickdisk(R, Z):
    return diskprofile(R, Z, 3261.0, 743.0)


def bulge(R, Z):
    factor = 2 * (diskprofile(0, 0, 2150.0, 245.0) + diskprofile(0, 0, 3261.0, 743.0))
    distance = (R**2 + Z**2) ** 0.5
    expfunc = np.exp(-distance / 800)
    return factor * expfunc


def halo(R, Z):
    q_h = 0.64
    n_h = 2.77
    f_h = 0.001
    part1 = R**2.0 + (Z / q_h) ** 2.0
    part2 = rsun / np.power(part1, 0.5)
    part3 = np.power(part2, n_h)
    tot = density_rsun * f_h * part3
    return tot


def stellardensity(R, Z, rho=0):
    part1 = thindisk(R, Z)
    part2 = thickdisk(R, Z)
    tot_density = part1 / 1.1 + f / 1.1 * part2 + halo(R, Z) + bulge(R, Z)
    return tot_density


# Juric et al 2008
# Jackson et al 2002
