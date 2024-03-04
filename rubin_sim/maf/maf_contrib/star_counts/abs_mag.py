#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Calculates absolute magnitudes as a function of spectral type.
# For use with Field Star Count metric
import sys

import numpy as np
from scipy.interpolate import interp1d


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]
        elif x > xs[-1]:
            return ys[-1]
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def abs_mag(spec_type):
    spec_range = [
        5,
        9,
        10,
        11,
        13,
        16,
        18,
        19,
        20,
        22,
        23,
        25,
        27,
        30,
        32,
        35,
        36,
        38,
        40,
        42,
        45,
        48,
        50,
        52,
        53,
        54,
        55,
        57,
        60,
        61,
        62,
        62.5,
        63,
        64,
        65,
        66,
    ]
    abs_u = [
        -5.8,
        -4.92,
        -3.78,
        -2.75,
        -1.51,
        -0.54,
        0.17,
        1.14,
        2.72,
        3.55,
        3.64,
        4.12,
        4.52,
        5.24,
        5.4,
        5.52,
        5.97,
        6.29,
        6.65,
        7.23,
        7.52,
        8.17,
        8.66,
        9.74,
        10.58,
        11.18,
        12.19,
        12.74,
        13.33,
        13.92,
        14.11,
        14.95,
        15.35,
        17.15,
        18.12,
        19.37,
    ]
    abs_g = [
        -5.41,
        -4.6,
        -3.53,
        -2.62,
        -1.69,
        -1.05,
        -0.58,
        0.28,
        1.63,
        2.39,
        2.49,
        2.92,
        3.31,
        4.08,
        4.28,
        4.43,
        4.83,
        5.07,
        5.35,
        5.86,
        6.03,
        6.57,
        6.98,
        7.79,
        8.48,
        8.97,
        9.65,
        10.18,
        10.68,
        11.2,
        11.52,
        12.21,
        12.68,
        14.02,
        15.07,
        16.38,
    ]
    abs_r = [
        -5.02,
        -4.28,
        -3.28,
        -2.49,
        -1.87,
        -1.56,
        -1.33,
        -0.58,
        0.54,
        1.23,
        1.34,
        1.72,
        2.1,
        2.92,
        3.16,
        3.34,
        3.69,
        3.85,
        4.05,
        4.49,
        4.54,
        4.97,
        5.3,
        5.84,
        6.38,
        6.76,
        7.11,
        7.62,
        8.03,
        8.48,
        8.93,
        9.47,
        10.01,
        10.89,
        12.02,
        13.39,
    ]
    abs_i = [
        -4.65,
        -3.93,
        -2.93,
        -2.16,
        -1.61,
        -1.33,
        -1.1,
        -0.35,
        0.72,
        1.4,
        1.49,
        1.83,
        2.18,
        2.93,
        3.13,
        3.31,
        3.61,
        3.75,
        3.91,
        4.33,
        4.38,
        4.78,
        5.08,
        5.6,
        6.06,
        6.38,
        6.71,
        7.08,
        7.39,
        7.7,
        8.08,
        8.46,
        8.81,
        9.38,
        10.29,
        11.27,
    ]
    abs_z = [
        -4.3,
        -3.57,
        -2.61,
        -1.9,
        -1.35,
        -1.1,
        -0.93,
        -0.17,
        0.89,
        1.55,
        1.64,
        1.93,
        2.26,
        3.02,
        3.15,
        3.33,
        3.62,
        3.72,
        3.89,
        4.29,
        4.32,
        4.72,
        5.01,
        5.49,
        5.93,
        6.22,
        6.51,
        6.78,
        7.01,
        7.24,
        7.56,
        7.83,
        8.14,
        8.62,
        9.35,
        10.03,
    ]
    abs_y = [
        -4.96,
        -4.18,
        -3.05,
        -2.22,
        -1.95,
        -1.5,
        -1.27,
        -0.61,
        0.32,
        0.96,
        0.99,
        1.38,
        1.79,
        2.51,
        2.57,
        2.74,
        3.04,
        3.11,
        3.36,
        3.7,
        3.76,
        4.16,
        4.61,
        4.98,
        5.43,
        5.8,
        6.07,
        6.38,
        6.45,
        6.86,
        6.97,
        7.25,
        7.52,
        7.82,
        8.39,
        8.6,
    ]
    f1 = interp1d(spec_range, abs_z)
    func_z = extrap1d(f1)
    f2 = interp1d(spec_range, abs_i)
    func_i = extrap1d(f2)
    f3 = interp1d(spec_range, abs_r)
    func_r = extrap1d(f3)
    f4 = interp1d(spec_range, abs_g)
    func_g = extrap1d(f4)
    f5 = interp1d(spec_range, abs_u)
    func_u = extrap1d(f5)
    f6 = interp1d(spec_range, abs_y)
    func_y = extrap1d(f6)
    abs_all = [
        func_z([spec_type])[0],
        func_y([spec_type])[0],
        func_i([spec_type])[0],
        func_r([spec_type])[0],
        func_g([spec_type])[0],
        func_u([spec_type])[0],
    ]
    bands = ["z", "y", "i", "r", "g", "u"]
    return abs_all, bands


if __name__ == "__main__":
    print(abs_mag(float(sys.argv[1]))[0])  # spectral type
