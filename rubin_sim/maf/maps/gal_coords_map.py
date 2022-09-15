from rubin_sim.utils import _galactic_from_equatorial
from rubin_sim.maf.maps import BaseMap

__all__ = ["GalCoordsMap"]


class GalCoordsMap(BaseMap):
    def __init__(self):
        self.keynames = ["gall", "galb"]

    def run(self, slicePoints):
        gall, galb = _galactic_from_equatorial(slicePoints["ra"], slicePoints["dec"])
        slicePoints["gall"] = gall
        slicePoints["galb"] = galb
        return slicePoints
