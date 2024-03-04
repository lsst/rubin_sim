__all__ = ("GalCoordsMap",)

from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_sim.maf.maps import BaseMap


class GalCoordsMap(BaseMap):
    """Add `gall` and `galb` (in radians) to the slice point dictionaries."""

    def __init__(self):
        self.keynames = ["gall", "galb"]

    def run(self, slice_points):
        coords = SkyCoord(ra=slice_points["ra"] * u.rad, dec=slice_points["dec"] * u.rad)
        gal = coords.galactic
        gall = gal.l.rad
        galb = gal.b.rad
        slice_points["gall"] = gall
        slice_points["galb"] = galb
        return slice_points
