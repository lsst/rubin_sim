"""A slicer that uses a Healpix grid to calculate metric values (at the center of each healpixel)."""

# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy (for histogram and power spectrum plotting)

import numpy as np
import healpy as hp
from rubin_sim.maf.plots.spatial_plotters import HealpixSkyMap, HealpixHistogram
from rubin_sim.utils import _galactic_from_equatorial

from .base_spatial_slicer import BaseSpatialSlicer


__all__ = ["HealpixSlicer"]


class HealpixSlicer(BaseSpatialSlicer):
    """
    A spatial slicer that evaluates pointings on a healpix-based grid.

    Note that a healpix slicer is intended to evaluate the sky on a spatial grid.
    Usually this grid will be something like RA as the lon_col and Dec as the lat_col.
    However, it could also be altitude and azimuth, in which case altitude as lat_col,
    and azimuth as lon_col.
    An additional alternative is to use HA/Dec, with lon_col=HA, lat_col=Dec.

    When plotting with RA/Dec, the default HealpixSkyMap can be used, corresponding to
    {'rot': (0, 0, 0), 'flip': 'astro'}.
    When plotting with alt/az, either the LambertSkyMap can be used (if Basemap is installed)
    or the HealpixSkyMap can be used with a modified plot_dict,
    {'rot': (90, 90, 90), 'flip': 'geo'}.
    When plotting with HA/Dec, only the HealpixSkyMap can be used, with a modified plot_dict of
    {'rot': (0, -30, 0), 'flip': 'geo'}.

    Parameters
    ----------
    nside : `int`, optional
        The nside parameter of the healpix grid. Must be a power of 2.
        Default 128.
    lon_col : `str`, optional
        Name of the longitude (RA equivalent) column to use from the input data.
        Default fieldRA
    lat_col : `str`, optional
        Name of the latitude (Dec equivalent) column to use from the input data.
        Default fieldDec
    lat_lon_deg : `bool`, optional
        Flag indicating whether the lat and lon values in the input data are in
        degrees (True) or radians (False).
        Default True.
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : `float`, optional
        Bad value flag, relevant for plotting. Default the hp.UNSEEN value (in order to properly flag
        bad data points for plotting with the healpix plotting routines). This should not be changed.
    use_cache : `bool`, optional
        Flag allowing the user to indicate whether or not to cache (and reuse) metric results
        calculated with the same set of simulated data pointings.
        This can be safely set to True for slicers not using maps and will result in increased speed.
        When calculating metric results using maps, the metadata at each individual ra/dec point may
        influence the metric results and so use_cache should be set to False.
        Default True.
    leafsize : `int`, optional
        Leafsize value for kdtree. Default 100.
    radius : `float`, optional
        Radius for matching in the kdtree. Equivalent to the radius of the FOV. Degrees.
        Default 2.45.
    use_camera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
        Default True.
    camera_footprint_file : `str`, optional
        Name of the camera footprint map to use. Can be None, which will use the default.
    rot_sky_pos_col_name : `str`, optional
        Name of the rotSkyPos column in the input  data. Only used if useCamera is True.
        Describes the orientation of the camera orientation compared to the sky.
        Default rotSkyPos.
    """

    def __init__(
        self,
        nside=128,
        lon_col="fieldRA",
        lat_col="fieldDec",
        lat_lon_deg=True,
        verbose=True,
        badval=hp.UNSEEN,
        use_cache=True,
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
        rot_sky_pos_col_name="rotSkyPos",
    ):
        """Instantiate and set up healpix slicer object."""
        super().__init__(
            verbose=verbose,
            lon_col=lon_col,
            lat_col=lat_col,
            badval=badval,
            radius=radius,
            leafsize=leafsize,
            use_camera=use_camera,
            camera_footprint_file=camera_footprint_file,
            rot_sky_pos_col_name=rot_sky_pos_col_name,
            lat_lon_deg=lat_lon_deg,
        )
        # Valid values of nside are powers of 2.
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not (hp.isnsideok(nside)):
            raise ValueError("Valid values of nside are powers of 2.")
        self.nside = int(nside)
        self.pixArea = hp.nside2pixarea(self.nside)
        self.nslice = hp.nside2npix(self.nside)
        self.spatialExtent = [0, self.nslice - 1]
        self.shape = self.nslice
        if self.verbose:
            print(
                "Healpix slicer using NSIDE=%d, " % (self.nside)
                + "approximate resolution %f arcminutes"
                % (hp.nside2resol(self.nside, arcmin=True))
            )
        # Set variables so slicer can be re-constructed
        self.slicer_init = {
            "nside": nside,
            "lon_col": lon_col,
            "lat_col": lat_col,
            "radius": radius,
        }
        self.use_cache = use_cache
        if use_cache:
            # use_cache set the size of the cache for the memoize function in sliceMetric.
            binRes = hp.nside2resol(nside)  # Pixel size in radians
            # Set the cache size to be ~2x the circumference
            self.cacheSize = int(np.round(4.0 * np.pi / binRes))
        # Set up slice_point metadata.
        self.slice_points["nside"] = nside
        self.slice_points["sid"] = np.arange(self.nslice)
        self.slice_points["ra"], self.slice_points["dec"] = self._pix2radec(
            self.slice_points["sid"]
        )
        gall, galb = _galactic_from_equatorial(
            self.slice_points["ra"], self.slice_points["dec"]
        )
        self.slice_points["gall"] = gall
        self.slice_points["galb"] = galb

        # Set the default plotting functions.
        self.plot_funcs = [HealpixSkyMap, HealpixHistogram]

    def __eq__(self, other_slicer):
        """Evaluate if two slicers are equivalent."""
        # If the two slicers are both healpix slicers, check nsides value.
        result = False
        if isinstance(other_slicer, HealpixSlicer):
            if other_slicer.nside == self.nside:
                if (
                    other_slicer.lon_col == self.lon_col
                    and other_slicer.lat_col == self.lat_col
                ):
                    if other_slicer.radius == self.radius:
                        if other_slicer.useCamera == self.useCamera:
                            if other_slicer.rotSkyPosColName == self.rotSkyPosColName:
                                if np.all(other_slicer.shape == self.shape):
                                    if other_slicer.use_cache == self.use_cache:
                                        result = True
        return result

    def _pix2radec(self, islice):
        """Given the pixel number / sliceID, return the RA/Dec of the pointing, in radians."""
        # Calculate RA/Dec in RADIANS of pixel in this healpix slicer.
        # Note that ipix could be an array,
        # in which case RA/Dec values will be an array also.
        lat, ra = hp.pix2ang(self.nside, islice)
        # Move dec to +/- 90 degrees
        dec = np.pi / 2.0 - lat
        return ra, dec
