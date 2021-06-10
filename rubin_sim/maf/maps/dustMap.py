from rubin_sim.maf.maps import BaseMap
from .EBVhp import EBVhp
import warnings

__all__ = ['DustMap']

class DustMap(BaseMap):
    """
    Compute the E(B-V) for each point in a given spatial distribution of slicePoints.
    """

    def __init__(self, interp=False, nside=128):
        """
        interp: should the dust map be interpolated (True) or just use the nearest value (False).
        """
        self.keynames = ['ebv']
        self.interp = interp
        self.nside = nside

    def run(self, slicePoints):
        # If the slicer has nside, it's a healpix slicer so we can read the map directly
        if 'nside' in slicePoints:
            if slicePoints['nside'] != self.nside:
                warnings.warn(f"Slicer value of nside {slicePoints['nside']} different "
                              f"from map value {self.nside}, using slicer value")
            slicePoints['ebv'] = EBVhp(slicePoints['nside'], pixels=slicePoints['sid'])
        # Not a healpix slicer, look up values based on RA,dec with possible interpolation
        else:
            slicePoints['ebv'] = EBVhp(self.nside, ra=slicePoints['ra'],
                                       dec=slicePoints['dec'], interp=self.interp)

        return slicePoints

