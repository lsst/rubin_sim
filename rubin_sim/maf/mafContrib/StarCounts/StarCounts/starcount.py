#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Calculates the number of stars in a given direction and between a given set of distances. For use with Field Star Count metric

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import sys
from scipy.optimize import fsolve
import math
from . import stellardensity
from . import coords
#from rubin_sim.coordUtils import AstronomyBase
skyarea=41253.
distancebins=51

def star_vols(D1, D2, area):
   distance_edges=(np.linspace((D1**3.), (D2**3.), num=distancebins))**(1./3)
   volumeshell=(area/skyarea)*(4.*np.pi/3)*(distance_edges[1:]**3-distance_edges[:-1]**3)
   distances=((distance_edges[1:]**3+distance_edges[:-1]**3)/2.)**(1./3)
   return volumeshell, distances

def starcount(eqRA, eqDEC, D1, D2):
   volumes, distances = star_vols(D1,D2,9.62)
   #b_deg, l_deg=coords.eq_gal2(eqRA, eqDEC)
   #b_deg, l_deg=AstrometryBase.equatorialToGalactic(eqRA, eqDEC)
   b_deg, l_deg=coords.eq_gal3(eqRA, eqDEC)
   positions=[coords.gal_cyn(b_deg, l_deg, x) for x in distances]
   densities=[stellardensity.stellardensity(x[0], x[2]) for x in positions]
   totalcount=np.sum(np.asarray(volumes)*np.asarray(densities))
   return totalcount


if __name__ == "__main__":
   print(starcount(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])))

