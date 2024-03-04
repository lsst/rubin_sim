__all__ = ("StellarDensityMap",)

import os

import healpy as hp
import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.maf.utils import radec2pix

from . import BaseMap


class StellarDensityMap(BaseMap):
    """Read and hold the cumulative stellar luminosity function for
    each slice point.

    The underlying stellar luminosity function map is nside = 64, and contains
    stars per sq degree at a series of magnitudes (the map contains
    `starLumFunc_<filter>` and `starMapBins_<filter>`).
    For slice points which do not match nside=64, the map uses the nearest
    healpix point on the nside=64 grid.

    The stellar luminosity function comes from the GalFast model.

    Parameters
    ----------
    startype : `str` ('allstars', 'wdstars')
        Load the luminosity function for all stars ('allstars'),
        which includes main-sequence stars
        white dwarfs, blue horozontal branch, RR Lyrae, and Cepheids.
        The 'wdstars' option only includes white dwarf stars.
    filtername : `str`
        Filter to use. Options of u,g,r,i,z,y
    """

    def __init__(self, startype="allstars", filtername="r", map_dir=None):
        if map_dir is not None:
            self.map_dir = map_dir
        else:
            self.map_dir = os.path.join(get_data_dir(), "maps", "StarMaps")
        self.filtername = filtername
        self.keynames = [
            f"starLumFunc_{self.filtername}",
            f"starMapBins_{self.filtername}",
        ]
        if startype == "allstars":
            self.startype = ""
        else:
            self.startype = startype + "_"

    def _read_map(self):
        filename = "starDensity_%s_%snside_64.npz" % (self.filtername, self.startype)
        star_map = np.load(os.path.join(self.map_dir, filename))
        self.star_map = star_map["starDensity"].copy()
        self.star_map_bins = star_map["bins"].copy()
        self.starmap_nside = hp.npix2nside(np.size(self.star_map[:, 0]))

    def run(self, slice_points):
        self._read_map()

        nside_match = False
        if "nside" in slice_points:
            if slice_points["nside"] == self.starmap_nside:
                slice_points[f"starLumFunc_{self.filtername}"] = self.star_map
                nside_match = True
        if not nside_match:
            # Compute the healpix for each slice_point on the nside=64 grid
            indx = radec2pix(self.starmap_nside, slice_points["ra"], slice_points["dec"])
            slice_points[f"starLumFunc_{self.filtername}"] = self.star_map[indx, :]

        slice_points[f"starMapBins_{self.filtername}"] = self.star_map_bins
        return slice_points
