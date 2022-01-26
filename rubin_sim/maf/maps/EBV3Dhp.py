import warnings
import os
import numpy as np
import healpy as hp
from astropy.io import fits

from rubin_sim.data import get_data_dir
from rubin_sim.photUtils import Dust_values
from rubin_sim.maf.utils import radec2pix

__all__ = ["EBV3Dhp"]


def EBV3Dhp(nside, ra=None, dec=None, pixels=None, interp=False, mapPath=None):
    """The EBV3dhp function reads in a healpix map of 3d dust extinction, and provides the tools
    to translate that map to the ra/dec/pixel locations.

    Parameters
    ----------
    nside: `int`
        Healpixel resolution (2^x).
    ra: `np.ndarray` or `float`, opt
        RA (can take numpy array). Default None sets up healpix array of nside. Radians.
    dec: `np.ndarray` or `float`, opt
        Dec (can take numpy array). Default None set up healpix array of nside. Radians.
    pixels: `np.ndarray`, opt
        Healpixel IDs, to sub-select particular healpix points. Default uses all points.
        NOTE -- to use a healpix map, set pixels and not ra/dec.
    interp: `bool`, opt
        Should returned values be interpolated (True) or just nearest neighbor (False)
    mapPath : `str`, opt
        Path to directory containing dust map files.
    """

    if (ra is None) & (dec is None) & (pixels is None):
        raise RuntimeError("Need to set ra,dec or pixels.")

    # Set up path to map data
    if mapPath is None:
        if nside == 64:
            mapName = "merged_ebv3d_nside64.fits"
            mapPath = os.path.join(get_data_dir(), "DustMaps3d", mapName)
        elif nside == 128:
            mapName = "merged_ebv3d_nside128_defaults.fits"
            mapPath = os.path.join(get_data_dir(), "DustMaps3d", mapName)
        else:
            raise Exception(
                f"mapPath not specified, and nside {nside} not one of 64 or 128. "
                "Please specify mapPath."
            )
    else:
        # Check if user specified map name but not full map path
        if not os.path.isfile(mapPath):
            testPath = os.path.join(get_data_dir(), "DustMaps3d", mapPath)
            if os.path.isfile(testPath):
                mapPath = testPath

    # Read map from disk
    hdul = fits.open(mapPath)
    #hpids = hdul[0].data
    dists = hdul[1].data
    ebvs = hdul[2].data
    hdr = hdul[0].header
    sfacs = None
    mask = None
    if len(hdul) > 3:
        sfacs = hdul[3].data
    if len(hdul) > 4:
        mask = hdul[4].data
    # Close map file
    hdul.close()

    # Check additional healpix information from the header
    map_nside = hdr["NSIDE"]
    nested = hdr["NESTED"]
    if nside != map_nside:
        warnings.warn(
            f"Map nside {map_nside} did not match expected nside {nside}; "
            f"Will use nside from map data."
        )
        nside = map_nside
    # Nested healpix data will not match the healpix arrays for the slicers (at this time)
    if nested:
        warnings.warn("Map has nested (not ring order) data; will have to reorder.")
        for i in np.arange(0, len(dists[0])):
            dists[:, i] = hp.reorder(dists[:, i], "nest", "ring")
            ebvs[:, i] = hp.reorder(ebvs[:, i], "nest", "ring")
            if sfacs is not None:
                sfacs[:, i] = hp.reorder(sfacs[:, i], "nest", "ring")
            if mask is not None:
                mask[:, i] = hp.reorder(mask[:, i], "nest", "ring")
        nested = False

    #
    # If we are interpolating to arbitrary positions
    if interp:
        distlen = len(dists[0])
        dists_interp = np.zeros([len(ra), distlen], float)
        ebvs_interp = np.zeros([len(ra), distlen], float)
        for i in np.arange(0, distlen):
            dists_interp[:, i] = hp.get_interp_val(dists[:, i], np.pi / 2.0 - dec, ra)
            ebvs_interp[:, i] = hp.reorder(ebvs[:, i], np.pi /  2.0, ra)
            if sfacs is not None:
                sfacs[:, i] = hp.reorder(sfacs[:, i], "nest", "ring")
            if mask is not None:
                mask[:, i] = hp.reorder(mask[:, i], "nest", "ring")

    else:
        # If we know the pixel indices we want
        if pixels is not None:
            result = EBVhp.dustMap[pixels]
        # Look up
        else:
            pixels = radec2pix(EBVhp.nside, ra, dec)
            result = EBVhp.dustMap[pixels]

    return result


    def getEBVatSightline(
        self, l=0.0, b=0.0, ebvMap=np.array([]), interp=False,
    ):

        """Utility - returns E(B-V) for one or more sightlines in Galactic
        coordinates. Takes as an argument a 2D healpix array of
        quantities (usually this will be reddening returned by
        getMapNearestDist() ). Also returns the nearest coords to the
        requested coords for debug purposes. Arguments:
        l, b = Galactic coordinates of the sight-line(s). Can be
        scalar or vector.
        ebvMap = 2D reddening map to use
        interp: Interpolate using healpy? If False, the nearest
        healpix is used instead.
        showEBVdist = Will usually be used only for debugging
        purposes. If True, this method will plot the run of E(B-V) vs
        distance for the nearest hpid (ignored if interp=True)
        """

        if np.size(ebvMap) < 1:
            return 0.0, -99.0, -99.0

        # find the coords on the sky of the requested sight line, and
        # convert this to healpix
        coo = SkyCoord(l * u.deg, b * u.deg, frame="galactic")

        # Equatorial coordinates of the requested position(s)
        ra = coo.icrs.ra.deg
        dec = coo.icrs.dec.deg

        if interp:
            ebvRet = hp.get_interp_val(ebvMap, ra, dec, lonlat=True)

            # For backwards compatibility with the "Test" return values
            lTest = np.copy(l)
            bTest = np.copy(b)

        else:
            hpid = hp.ang2pix(self.nside, ra, dec, nest=self.nested, lonlat=True)
            ebvRet = ebvMap[hpid]

            # For debugging: determine the coordinates at this nearest pixel
            raTest, decTest = hp.pix2ang(
                self.nside, hpid, nest=self.nested, lonlat=True
            )
            cooTest = SkyCoord(raTest * u.deg, decTest * u.deg, frame="icrs")
            lTest = cooTest.galactic.l.degree
            bTest = cooTest.galactic.b.degree
        return ebvRet, lTest, bTest

