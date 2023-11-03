__all__ = ("TrilegalDensityMap",)

import os

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import _build_tree, _hpid2_ra_dec, _xyz_from_ra_dec

from . import BaseMap


class TrilegalDensityMap(BaseMap):
    """
    Return the cumulative stellar luminosity function for each slice_point. Units of stars per sq degree.

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
        self.map_dir = os.path.join(get_data_dir(), "maps", "TriMaps")
        self.filtername = filtername
        self.keynames = [
            f"starLumFunc_{self.filtername}",
            f"starMapBins_{self.filtername}",
        ]
        self.nside = nside
        self.ext = ext

    def _read_map(self):
        if self.ext:
            filename = "TRIstarDensity_%s_nside_%i_ext.npz" % (
                self.filtername,
                self.nside,
            )
        else:
            filename = "TRIstarDensity_%s_nside_%i.npz" % (self.filtername, self.nside)
        star_map = np.load(os.path.join(self.map_dir, filename))
        self.star_map = star_map["starDensity"].copy()
        self.star_map_bins = star_map["bins"].copy()
        self.starmap_nside = hp.npix2nside(np.size(self.star_map[:, 0]))
        # note, the trilegal maps are in galactic coordinates, and nested healpix.
        gal_l, gal_b = _hpid2_ra_dec(self.nside, np.arange(hp.nside2npix(self.nside)), nest=True)

        # Convert that to RA,dec. Then do nearest neighbor lookup.
        c = SkyCoord(l=gal_l * u.rad, b=gal_b * u.rad, frame="galactic").transform_to("icrs")
        ra = c.ra.rad
        dec = c.dec.rad

        self.tree = _build_tree(ra, dec)

    def run(self, slice_points):
        self._read_map()

        x, y, z = _xyz_from_ra_dec(slice_points["ra"], slice_points["dec"])

        dist, indices = self.tree.query(list(zip(x, y, z)))

        slice_points["starLumFunc_%s" % self.filtername] = self.star_map[indices, :]
        slice_points["starMapBins_%s" % self.filtername] = self.star_map_bins
        return slice_points
