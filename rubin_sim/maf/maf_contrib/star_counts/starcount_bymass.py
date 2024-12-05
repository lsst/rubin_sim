#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Takes a given set of galactic coordinates and a stellar
# mass range, then calculates the number of stars within that range
# that will be fainter than mag 16, and have sufficiently low noise
# in the given band. For use with Field Star Count metric


import numpy as np
from scipy.optimize import newton

from . import abs_mag, spec_type
from .starcount import starcount

xi = 1.0
alpha = 2.35


def IMF(lower, upper):
    exp = alpha - 1.0
    part1 = xi / exp
    part2 = lower**-exp - upper**-exp
    part3 = part1 * part2
    return part3


def get_distance(apparent, absolute):
    part1 = (apparent - absolute) / 5.0
    part2 = 10 * 10**part1
    return part2


def noise_opt(m, band, sigma):
    part1 = mag_error(m, band)
    part2 = part1**2.0
    part3 = sigma**2.0
    total = part2 - part3
    return total


def mag_error(ap_mag, band, calcm5=0):  # apparent magnitude and band
    gamma = {"u": 0.037, "g": 0.038, "r": 0.039, "i": 0.039, "z": 0.040, "y": 0.040}
    m5 = {"u": 23.9, "g": 25.0, "r": 24.7, "i": 24.0, "z": 23.3, "y": 22.1}
    if calcm5 == 0:
        calcm5 = m5[band]
    X = 10.0 ** (0.4 * (ap_mag - calcm5))
    random_2 = np.sqrt((0.04 - gamma[band]) * X + gamma[band] * X * X)
    error_2 = random_2**2.0 + (0.005) ** 2.0
    mag_error = np.sqrt(error_2)
    return mag_error


def noise_calc(band):
    gamma = {"u": 0.037, "g": 0.038, "r": 0.039, "i": 0.039, "z": 0.040, "y": 0.040}
    m5 = {"u": 23.9, "g": 25.0, "r": 24.7, "i": 24.0, "z": 23.3, "y": 22.1}
    sigma = 0.03
    sigma_sys = 0.005

    def fun(x):
        sigma_sys**2
        +(0.04 - gamma[band]) * 10 ** (0.4 * (x - m5[band]))
        +gamma[band] * 10 ** (0.8 * (x - m5[band]))
        -(sigma**2)

    return newton(fun, 25)


def dist_calc(mass, band):
    # mass to spectral type
    # spectral type to absolute mag
    bands = ["z", "y", "i", "r", "g", "u"]
    output = abs_mag.abs_mag(spec_type.spec_type(mass))[0]
    indexvalue = bands.index(band)
    absolutemag = output[indexvalue][0]
    apparent = noise_calc(band)
    dist_min = get_distance(16, absolutemag)
    dist_max = get_distance(apparent, absolutemag)
    return dist_min, dist_max
    # abs mag to apparent mag ranges, > 16, noise dependent upper limit


def starcount_bymass(eq_ra, eq_dec, m1, m2, band):
    masses = np.linspace(m1, m2, num=20)
    totmass = IMF(m1, m2)
    totmass = IMF(0.2, 1.04)
    massbins = IMF(masses[:-1], masses[1:])
    massfractions = massbins / totmass
    distances = [dist_calc(x, band) for x in masses[:-1]]
    starcounts = [y * starcount(eq_ra, eq_dec, x[0], x[1]) for x, y in zip(distances, massfractions)]
    return sum(starcounts)
