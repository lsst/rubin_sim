import numpy as np
from .baseStacker import BaseStacker
from .generalStackers import FiveSigmaStacker

__all__ = ['NEODistStacker']


class NEODistStacker(BaseStacker):
    """
    For each observation, find the max distance to a ~144 km NEO,
    also stack on the x,y position of the object.
    """
    colsAdded = ['MaxGeoDist', 'NEOHelioX', 'NEOHelioY']

    def __init__(self,
                 stepsize=.001, maxDist=3., minDist=.3, H=22, elongCol='solarElong',
                 filterCol='filter', sunAzCol='sunAz', azCol='azimuth', m5Col='fiveSigmaDepth'):
        """
        stepsize:  The stepsize to use when solving (in AU)
        maxDist: How far out to try and measure (in AU)
        H: Asteroid magnitude

        Adds columns:
        MaxGeoDist:  Geocentric distance to the NEO
        NEOHelioX: Heliocentric X (with Earth at x,y,z (0,1,0))
        NEOHelioY: Heliocentric Y (with Earth at (0,1,0))

        Note that both opsim v3 and v4 report solarElongation in degrees.
        """
        self.units = ['AU', 'AU', 'AU']
        # Also grab things needed for the HA stacker
        self.colsReq = [elongCol, filterCol, sunAzCol, azCol, m5Col]

        self.sunAzCol = sunAzCol
        self.elongCol = elongCol
        self.filterCol = filterCol
        self.azCol = azCol
        self.m5Col = m5Col

        self.H = H
        # Magic numbers (Ivezic '15, private comm.)that convert an asteroid
        # V-band magnitude to LSST filters:
        # V_5 = m_5 + (adjust value)
        self.limitingAdjust = {'u': -2.1, 'g': -0.5, 'r': 0.2, 'i': 0.4, 'z': 0.6, 'y': 0.6}
        self.deltas = np.arange(minDist, maxDist+stepsize, stepsize)
        self.G = 0.15

        # Magic numbers from  http://adsabs.harvard.edu/abs/2002AJ....124.1776J
        self.a1 = 3.33
        self.b1 = 0.63
        self.a2 = 1.87
        self.b2 = 1.22

    def _run(self, simData, cols_present=False):
        if cols_present:
            # This is a pretty rare stacker. Assume we need to rerun
            pass
        elongRad = np.radians(simData[self.elongCol])
        v5 = np.zeros(simData.size, dtype=float) + simData[self.m5Col]
        for filterName in self.limitingAdjust:
            fmatch = np.where(simData[self.filterCol] == filterName)
            v5[fmatch] += self.limitingAdjust[filterName]
        for i, elong in enumerate(elongRad):
            # Law of cosines:
            # Heliocentric Radius of the object
            R = np.sqrt(1.+self.deltas**2-2.*self.deltas*np.cos(elong))
            # Angle between sun and earth as seen by NEO
            alphas = np.arccos((1.-R**2-self.deltas**2)/(-2.*self.deltas*R))
            ta2 = np.tan(alphas/2.)
            phi1 = np.exp(-self.a1*ta2**self.b1)
            phi2 = np.exp(-self.a2*ta2**self.b2)

            alpha_term = 2.5*np.log10((1. - self.G)*phi1+self.G*phi2)
            appmag = self.H+5.*np.log10(R*self.deltas)-alpha_term
            # There can be some local minima/maxima when solving, so
            # need to find the *1st* spot where it is too faint, not the
            # last spot it is bright enough.
            tooFaint = np.where(appmag > v5[i])

            # Check that there is a minimum
            if np.size(tooFaint[0]) == 0:
                simData['MaxGeoDist'][i] = 0
            else:
                simData['MaxGeoDist'][i] = np.min(self.deltas[tooFaint])

        # Make coords in heliocentric
        interior = np.where(elongRad <= np.pi/2.)
        outer = np.where(elongRad > np.pi/2.)
        simData['NEOHelioX'][interior] = simData['MaxGeoDist'][interior]*np.sin(elongRad[interior])
        simData['NEOHelioY'][interior] = -simData['MaxGeoDist'][interior]*np.cos(elongRad[interior]) + 1.

        simData['NEOHelioX'][outer] = simData['MaxGeoDist'][outer]*np.sin(np.pi-elongRad[outer])
        simData['NEOHelioY'][outer] = simData['MaxGeoDist'][outer]*np.cos(np.pi-elongRad[outer]) + 1.

        # Flip the X coord if sun az is negative?
        if simData[self.azCol].min() < - np.pi/2.0:
            halfval = 180.
        else:
            halfval = np.pi
        flip = np.where(((simData[self.sunAzCol] > halfval) & (simData[self.azCol] > halfval)) |
                        ((simData[self.sunAzCol] < halfval) & (simData[self.azCol] > halfval)))

        simData['NEOHelioX'][flip] *= -1.

        return simData
