__all__ = ("GalCoordsMap",)

from rubin_sim.maf.maps import BaseMap
from rubin_sim.utils import _galactic_from_equatorial


class GalCoordsMap(BaseMap):
    def __init__(self):
        self.keynames = ["gall", "galb"]

    def run(self, slice_points):
        gall, galb = _galactic_from_equatorial(slice_points["ra"], slice_points["dec"])
        slice_points["gall"] = gall
        slice_points["galb"] = galb
        return slice_points
