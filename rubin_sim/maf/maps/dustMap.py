from rubin_sim.maf.maps import BaseMap
from .EBVhp import EBVhp
import warnings

__all__ = ["DustMap"]


class DustMap(BaseMap):
    """
    Compute the E(B-V) for each point in a given spatial distribution of slicePoints.

    Primarily, this calls EBVhp to read a healpix map of E(B-V) values over the sky, then
    assigns ebv values to each slicePoint. If the slicer is a healpix slicer, this is trivial.

    Parameters
    ----------
    interp : `bool`, opt
        Interpolate the dust map at each slicePoint (True) or just use the nearest value (False).
        Default is False.
    nside : `int`, opt
        Default nside value to read the dust map from disk. Primarily useful if the slicer is not
        a healpix slicer.
        Default 128.
    mapPath : `str`, opt
        Define a path to the directory holding the dust map files.
        Default None, which uses RUBIN_SIM_DATA_DIR.
    """

    def __init__(self, interp=False, nside=128, mapPath=None):
        """
        interp: should the dust map be interpolated (True) or just use the nearest value (False).
        """
        self.keynames = ["ebv"]
        self.interp = interp
        self.nside = nside
        self.mapPath = mapPath

    def run(self, slicePoints):
        # If the slicer has nside, it's a healpix slicer so we can read the map directly
        if "nside" in slicePoints:
            if slicePoints["nside"] != self.nside:
                warnings.warn(
                    f"Slicer value of nside {slicePoints['nside']} different "
                    f"from map value {self.nside}, using slicer value"
                )
            slicePoints["ebv"] = EBVhp(
                slicePoints["nside"], pixels=slicePoints["sid"], mapPath=self.mapPath
            )
        # Not a healpix slicer, look up values based on RA,dec with possible interpolation
        else:
            slicePoints["ebv"] = EBVhp(
                self.nside,
                ra=slicePoints["ra"],
                dec=slicePoints["dec"],
                interp=self.interp,
                mapPath=self.mapPath,
            )

        return slicePoints
