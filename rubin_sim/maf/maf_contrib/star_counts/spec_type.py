#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Calculates spectral types for stars on the main sequence as
# a function of stellar mass. For use with Field Star Count metric
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


def spec_type(mass):
    mass_range = [
        0.06,
        0.21,
        0.40,
        0.51,
        0.67,
        0.79,
        0.92,
        1.05,
        1.4,
        1.6,
        2.0,
        2.9,
        3.8,
        5.9,
        7.6,
        17.5,
        23,
        37,
        60,
        120,
    ]
    spec_range = [
        68,
        65,
        62,
        60,
        55,
        50,
        45,
        40,
        35,
        30,
        25,
        20,
        18,
        15,
        13,
        10,
        8,
        6,
        5,
        3,
    ]
    f = interp1d(mass_range, spec_range)
    f2 = extrap1d(f)
    # f2=interp1d(mass_range, spec_range, kind='slinear')
    if mass == 0:
        xnew = np.logspace(-2, 3, 100)
        import matplotlib.pyplot as plt

        plt.plot(mass_range, spec_range, "o", xnew, f2(xnew), "-")  # , xnew, f2(xnew),'--')
        plt.xscale("log")
        plt.show()
    return f2([mass])


if __name__ == "__main__":
    print(spec_type(float(sys.argv[1])))  # mass
