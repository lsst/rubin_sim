__all__ = ("NEODistStacker",)

import numpy as np

from .base_stacker import BaseStacker


class NEODistStacker(BaseStacker):
    """
    For each observation, find the max distance to a ~144 km NEO,
    also stack on the x,y position of the object.
    """

    cols_added = ["MaxGeoDist", "NEOHelioX", "NEOHelioY"]

    def __init__(
        self,
        stepsize=0.001,
        max_dist=3.0,
        min_dist=0.3,
        H=22,
        elong_col="solarElong",
        filter_col="filter",
        sun_az_col="sunAz",
        az_col="azimuth",
        m5_col="fiveSigmaDepth",
    ):
        """
        stepsize:  The stepsize to use when solving (in AU)
        max_dist: How far out to try and measure (in AU)
        H: Asteroid magnitude

        Adds columns:
        MaxGeoDist:  Geocentric distance to the NEO
        NEOHelioX: Heliocentric X (with Earth at x,y,z (0,1,0))
        NEOHelioY: Heliocentric Y (with Earth at (0,1,0))

        Note that both opsim v3 and v4 report solarElongation in degrees.
        """
        self.units = ["AU", "AU", "AU"]
        # Also grab things needed for the HA stacker
        self.cols_req = [elong_col, filter_col, sun_az_col, az_col, m5_col]

        self.sun_az_col = sun_az_col
        self.elong_col = elong_col
        self.filter_col = filter_col
        self.az_col = az_col
        self.m5_col = m5_col

        self.H = H
        # Magic numbers (Ivezic '15, private comm.)that convert an asteroid
        # V-band magnitude to LSST filters:
        # V_5 = m_5 + (adjust value)
        self.limiting_adjust = {
            "u": -2.1,
            "g": -0.5,
            "r": 0.2,
            "i": 0.4,
            "z": 0.6,
            "y": 0.6,
        }
        self.deltas = np.arange(min_dist, max_dist + stepsize, stepsize)
        self.G = 0.15

        # Magic numbers from  http://adsabs.harvard.edu/abs/2002AJ....124.1776J
        self.a1 = 3.33
        self.b1 = 0.63
        self.a2 = 1.87
        self.b2 = 1.22

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # This is a pretty rare stacker. Assume we need to rerun
            pass
        elong_rad = np.radians(sim_data[self.elong_col])
        v5 = np.zeros(sim_data.size, dtype=float) + sim_data[self.m5_col]
        for filter_name in self.limiting_adjust:
            fmatch = np.where(sim_data[self.filter_col] == filter_name)
            v5[fmatch] += self.limiting_adjust[filter_name]
        for i, elong in enumerate(elong_rad):
            # Law of cosines:
            # Heliocentric Radius of the object
            R = np.sqrt(1.0 + self.deltas**2 - 2.0 * self.deltas * np.cos(elong))
            # Angle between sun and earth as seen by NEO
            alphas = np.arccos((1.0 - R**2 - self.deltas**2) / (-2.0 * self.deltas * R))
            ta2 = np.tan(alphas / 2.0)
            phi1 = np.exp(-self.a1 * ta2**self.b1)
            phi2 = np.exp(-self.a2 * ta2**self.b2)

            alpha_term = 2.5 * np.log10((1.0 - self.G) * phi1 + self.G * phi2)
            appmag = self.H + 5.0 * np.log10(R * self.deltas) - alpha_term
            # There can be some local minima/maxima when solving, so
            # need to find the *1st* spot where it is too faint, not the
            # last spot it is bright enough.
            too_faint = np.where(appmag > v5[i])

            # Check that there is a minimum
            if np.size(too_faint[0]) == 0:
                sim_data["MaxGeoDist"][i] = 0
            else:
                sim_data["MaxGeoDist"][i] = np.min(self.deltas[too_faint])

        # Make coords in heliocentric
        interior = np.where(elong_rad <= np.pi / 2.0)
        outer = np.where(elong_rad > np.pi / 2.0)
        sim_data["NEOHelioX"][interior] = sim_data["MaxGeoDist"][interior] * np.sin(elong_rad[interior])
        sim_data["NEOHelioY"][interior] = (
            -sim_data["MaxGeoDist"][interior] * np.cos(elong_rad[interior]) + 1.0
        )

        sim_data["NEOHelioX"][outer] = sim_data["MaxGeoDist"][outer] * np.sin(np.pi - elong_rad[outer])
        sim_data["NEOHelioY"][outer] = sim_data["MaxGeoDist"][outer] * np.cos(np.pi - elong_rad[outer]) + 1.0

        # Flip the X coord if sun az is negative?
        if sim_data[self.az_col].min() < -np.pi / 2.0:
            halfval = 180.0
        else:
            halfval = np.pi
        flip = np.where(
            ((sim_data[self.sun_az_col] > halfval) & (sim_data[self.az_col] > halfval))
            | ((sim_data[self.sun_az_col] < halfval) & (sim_data[self.az_col] > halfval))
        )

        sim_data["NEOHelioX"][flip] *= -1.0

        return sim_data
