import os
import numpy as np
import healpy as hp
from rubin_sim.utils import (
    _hpid2_ra_dec,
    _equatorial_from_galactic,
    _build_tree,
    _xyz_from_ra_dec,
)
from rubin_sim.data import get_data_dir
from . import BaseMap

__all__ = ["TrilegalDensityMap"]


class TrilegalDensityMap(BaseMap):
    """
    Return the cumulative stellar luminosity function for each slicepoint. Units of stars per sq degree.

    Parameters
    ----------
    filtername : `str`, opt
        Filter to use. Options of u,g,r,i,z,y. Default r.
    nside : `int`, opt
        The HEALpix nside (can be 64 or 128). Default 64.
    ext : `bool`, opt
        Use the full sky maps. Default True.
    """

    def __init__(self, filtername="r", nside=64, ext=True):
        self.mapDir = os.path.join(get_data_dir(), "maps", "TriMaps")
        self.filtername = filtername
        self.keynames = [
            f"starLumFunc_{self.filtername}",
            f"starMapBins_{self.filtername}",
        ]
        self.nside = nside
        self.ext = ext

    def _readMap(self):
        if self.ext:
            filename = "TRIstarDensity_%s_nside_%i_ext.npz" % (
                self.filtername,
                self.nside,
            )
        else:
            filename = "TRIstarDensity_%s_nside_%i.npz" % (self.filtername, self.nside)
        starMap = np.load(os.path.join(self.mapDir, filename))
        self.starMap = starMap["starDensity"].copy()
        self.starMapBins = starMap["bins"].copy()
        self.starmapNside = hp.npix2nside(np.size(self.starMap[:, 0]))
        # note, the trilegal maps are in galactic coordinates, and nested healpix.
        gal_l, gal_b = _hpid2_ra_dec(
            self.nside, np.arange(hp.nside2npix(self.nside)), nest=True
        )

        # Convert that to RA,dec. Then do nearest neighbor lookup.
        ra, dec = _equatorial_from_galactic(gal_l, gal_b)
        self.tree = _build_tree(ra, dec)

    def run(self, slicePoints):
        self._readMap()

        x, y, z = _xyz_from_ra_dec(slicePoints["ra"], slicePoints["dec"])

        dist, indices = self.tree.query(list(zip(x, y, z)))

        slicePoints["starLumFunc_%s" % self.filtername] = self.starMap[indices, :]
        slicePoints["starMapBins_%s" % self.filtername] = self.starMapBins
        return slicePoints
