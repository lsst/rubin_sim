#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Provides the coordinate conversions between equatorial and galactic coordinates, as well as to galactic cylindrical coordinates. Two different functions are present that do the conversion, and a third that uses ephem package, for redundancy purposes. For use with Field Star Count metric

from __future__ import print_function
import numpy as np
import sys
from scipy.optimize import fsolve
import math

rad1 = np.radians(282.25)
rad2 = np.radians(62.6)
rad3 = np.radians(33.0)


def eq_gal(eq_ra, eq_dec):
    d = np.radians(eq_dec)
    a = np.radians(eq_ra)

    def equations(p):
        b, l, x = p
        f1a = np.cos(d) * (np.cos(a - rad1))
        f2a = np.sin(d) * np.sin(rad2) + np.cos(d) * np.sin(a - rad1) * np.cos(rad2)
        f3a = np.sin(d) * np.cos(rad2) - np.cos(d) * np.sin(a - rad1) * np.sin(rad2)
        f1 = np.cos(b) * np.cos(l - rad3) - f1a
        f2 = np.cos(b) * np.sin(l - rad3) - f2a
        f3 = np.sin(b) - f3a
        return (f1, f2, f3)

    b, l, x = fsolve(equations, (0, 0, 0))
    b_deg = np.degrees(b) % 360  # galactic latitude
    if b_deg >= 270:
        b_deg = b_deg - 360
    if b_deg > 90:
        b_deg = 180 - b_deg
        l = l + np.pi
    l_deg = np.degrees(l) % 360  # galactic longitude
    return b_deg, l_deg
    # http://scienceworld.wolfram.com/astronomy/GalacticCoordinates.html


def eq_gal2(eq_ra, eq_dec):
    d = np.radians(eq_dec)
    p = np.radians(eq_ra)
    AC = np.radians(90.0) - d
    AB = np.radians(62.8717)
    CAB = np.radians(192.8585) - p
    cos_bc = np.sin(d) * np.cos(AB) + np.cos(d) * np.sin(AB) * np.cos(CAB)
    BC = np.arccos(cos_bc)
    AD = np.radians(118.9362)
    CAD = np.radians(266.4051) - p
    cos_cd = np.sin(d) * np.cos(AD) + np.cos(d) * np.sin(AD) * np.cos(CAD)
    cos_cbd = cos_cd / np.sin(BC)
    if cos_cbd > 1:
        cos_cbd = 1
    elif cos_cbd < -1:
        cos_cbd = -1
    CBD = np.arccos(cos_cbd)
    b_deg = 90.0 - np.degrees(BC)
    ad = np.radians(90.0)
    cad = np.radians(282.8595) - p
    coscd = np.cos(cad) * np.cos(d)
    coscbd = np.cos(cad) * np.cos(d) / np.sin(BC)
    if coscbd > 1:
        coscbd = 1
    elif coscbd < -1:
        coscbd = -1
    cbd = np.arccos(coscbd)
    if cbd - CBD < 32.9319:
        l_deg = 360.0 - np.degrees(CBD)
    else:
        l_deg = np.degrees(CBD)
    return b_deg, l_deg


def eq_gal3(eq_ra, eq_dec):
    coordset = ephem.Equatorial(np.radians(eq_ra), np.radians(eq_dec), epoch="2000")
    g = ephem.Galactic(coordset)
    templon, templat = float(g.lon), float(g.lat)
    l_deg = np.degrees(templon)
    b_deg = np.degrees(templat)
    return b_deg, l_deg


def gal_cyn(b_deg, l_deg, dist):
    b_rad = np.radians(b_deg)
    l_rad = np.radians(l_deg)
    Z = np.sin(b_rad) * dist
    xy = np.cos(b_rad) * dist
    x = np.cos(l_rad) * xy
    y = np.sin(l_rad) * xy
    x_new = 8000.0 - x
    R = np.power(x_new**2 + y**2, 0.5)
    rho = np.arctan(y / x)
    return R, rho, Z


if __name__ == "__main__":
    gal_lat, gal_lon = eq_gal2(float(sys.argv[1]), float(sys.argv[2]))
    print(gal_lat, gal_lon)
    R, rho, z = gal_cyn(gal_lat, gal_lon, float(sys.argv[3]))
    print(R, rho, z)
