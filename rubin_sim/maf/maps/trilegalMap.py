import os
import numpy as np
import healpy as hp
from rubin_sim.utils import _hpid2RaDec, _equatorialFromGalactic, _buildTree, _xyz_from_ra_dec
from rubin_sim.data import get_data_dir
from . import BaseMap

__all__ = ['TrilegalDensityMap']


class TrilegalDensityMap(BaseMap):
    """
    Return the cumulative stellar luminosity function for each slicepoint. Units of stars per sq degree.

    Parameters
    ----------
    filtername : str
        Filter to use. Options of u,g,r,i,z,y
    nside : int (64)
        The HEALpix nside (can be 64 or 128)
    ext : bool (False)
        Use the full sky maps
    """
    def __init__(self, filtername='r', nside=64, ext=True):
        self.mapDir = os.path.join(get_data_dir(), 'maps', 'TriMaps')
        self.filtername = filtername
        self.keynames = [f'starLumFunc_{self.filtername}', f'starMapBins_{self.filtername}']
        self.nside = nside
        self.ext = ext

    def _readMap(self):
        if self.ext:
            filename = 'TRIstarDensity_%s_nside_%i_ext.npz' % (self.filtername, self.nside)
        else:
            filename = 'TRIstarDensity_%s_nside_%i.npz' % (self.filtername, self.nside)
        starMap = np.load(os.path.join(self.mapDir, filename))
        self.starMap = starMap['starDensity'].copy()
        self.starMapBins = starMap['bins'].copy()
        self.starmapNside = hp.npix2nside(np.size(self.starMap[:, 0]))
        # note, the trilegal maps are in galactic coordinates, and nested healpix.
        gal_l, gal_b = _hpid2RaDec(self.nside, np.arange(hp.nside2npix(self.nside)), nest=True)

        # Convert that to RA,dec. Then do nearest neighbor lookup.
        ra, dec = _equatorialFromGalactic(gal_l, gal_b)
        self.tree = _buildTree(ra, dec)

    def run(self, slicePoints):
        self._readMap()

        x, y, z = _xyz_from_ra_dec(slicePoints['ra'], slicePoints['dec'])

        dist, indices = self.tree.query(list(zip(x, y, z)))

        slicePoints['starLumFunc_%s' % self.filtername] = self.starMap[indices, :]
        slicePoints['starMapBins_%s' % self.filtername] = self.starMapBins
        return slicePoints
