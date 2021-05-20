#!/usr/bin/env python

# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Description: Calculates the stellar density based off of Juric et al 2008 and Jackson et al 2002. For use with Field Star Count metric

from __future__ import print_function
import numpy as np
import sys
from scipy.optimize import fsolve
import math
Zsun=25.
Rsun=8000.
densityRsun=.0364
f=.1

def diskprofile(R, Z, L, H):
   part1=(-R/L)-(abs(Z+Zsun)/H)
   part2=np.exp(part1)
   part3=np.exp(Rsun/L)
   tot=densityRsun*part2*part3
   return tot

def thindisk(R, Z):
   return diskprofile(R, Z, 2150., 245.)


def thickdisk(R, Z):
   return diskprofile(R, Z, 3261., 743.)

def bulge(R, Z):
   factor=2*(diskprofile(0, 0, 2150., 245.)+diskprofile(0, 0, 3261., 743.))
   distance=(R**2+Z**2)**0.5
   expfunc=np.exp(-distance/800)
   return factor*expfunc

def halo(R, Z):
   qH=0.64
   nH=2.77
   fH=.001
   part1=R**2.+(Z/qH)**2.
   part2=Rsun/np.power(part1, 0.5)
   part3=np.power(part2, nH)
   tot=densityRsun*fH*part3
   return tot
   
def stellardensity(R, Z, rho=0):
   part1=thindisk(R, Z)
   part2=thickdisk(R, Z)
   tot_density=part1/1.1+f/1.1*part2+halo(R, Z)+bulge(R,Z)
   return tot_density

if __name__ == "__main__":
   print(stellardensity(float(sys.argv[1]), float(sys.argv[2])))


#Juric et al 2008
#Jackson et al 2002
