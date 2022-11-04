import numpy as np
from rubin_sim.maf.plots.spatial_plotters import BaseSkyMap, BaseHistogram
from .base_spatial_slicer import BaseSpatialSlicer
from rubin_sim.utils import _galactic_from_equatorial

__all__ = ["UserPointsSlicer"]


class UserPointsSlicer(BaseSpatialSlicer):
    """A spatial slicer that evaluates pointings overlapping user-provided list of points.

    Parameters
    ----------
    ra : `list` or `numpy.ndarray`
        User-selected RA points, in degrees. Stored internally in radians.
    dec : `list` or `numpy.ndarray`
        User-selected Dec points, in degrees. Stored internally in radians.
    lonCol : `str`, optional
        Name of the longitude (RA equivalent) column to use from the input data.
        Default fieldRA
    latCol : `str`, optional
        Name of the latitude (Dec equivalent) column to use from the input data.
        Default fieldDec
    latLonDeg : `bool`, optional
        Flag indicating whether the lon and lat values will be in degrees (True) or radians (False).
        Default True (appropriate for opsim v4).
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : `float`, optional
        Bad value flag, relevant for plotting. Default -666.
    leafsize : `int`, optional
        Leafsize value for kdtree. Default 100.
    radius : `float`, optional
        Radius for matching in the kdtree. Equivalent to the radius of the FOV. Degrees.
        Default 2.45.
    useCamera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
        Default True.
    camera_footprint_file : `str`, optional
        Name of the camera footprint map to use. Can be None, which will use the default.
    rotSkyPosColName : `str`, optional
        Name of the rotSkyPos column in the input  data. Only used if useCamera is True.
        Describes the orientation of the camera orientation compared to the sky.
        Default rotSkyPos.
    """

    def __init__(
        self,
        ra,
        dec,
        lon_col="fieldRA",
        lat_col="fieldDec",
        lat_lon_deg=True,
        verbose=True,
        badval=-666,
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
        rot_sky_pos_col_name="rotSkyPos",
    ):
        super().__init__(
            lon_col=lon_col,
            lat_col=lat_col,
            lat_lon_deg=lat_lon_deg,
            verbose=verbose,
            badval=badval,
            radius=radius,
            leafsize=leafsize,
            use_camera=use_camera,
            camera_footprint_file=camera_footprint_file,
            rot_sky_pos_col_name=rot_sky_pos_col_name,
        )
        # check that ra and dec are iterable, if not, they are probably naked numbers, wrap in list
        if not hasattr(ra, "__iter__"):
            ra = [ra]
        if not hasattr(dec, "__iter__"):
            dec = [dec]
        if len(ra) != len(dec):
            raise ValueError("RA and Dec must be the same length")
        ra = np.radians(ra)
        dec = np.radians(dec)
        self.slicePoints["sid"] = np.arange(np.size(ra))
        self.slicePoints["ra"] = np.array(ra)
        self.slicePoints["dec"] = np.array(dec)
        gall, galb = _galactic_from_equatorial(
            self.slicePoints["ra"], self.slicePoints["dec"]
        )
        self.slicePoints["gall"] = gall
        self.slicePoints["galb"] = galb

        self.nslice = np.size(ra)
        self.shape = self.nslice
        self.spatial_extent = [0, self.nslice - 1]
        self.slicer_init = {
            "ra": ra,
            "dec": dec,
            "lon_col": lon_col,
            "lat_col": lat_col,
            "radius": radius,
        }
        self.plot_funcs = [BaseSkyMap, BaseHistogram]

    def __eq__(self, other_slicer):
        """Evaluate if two slicers are equivalent."""
        result = False
        # check the slicePoints
        for key in other_slicer.slicePoints:
            if key in self.slicePoints.keys():
                if not np.array_equal(
                    other_slicer.slicePoints[key], self.slicePoints[key]
                ):
                    return False
            else:
                return False
        if isinstance(other_slicer, UserPointsSlicer):
            if other_slicer.nslice == self.nslice:
                if np.array_equal(
                    other_slicer.slicePoints["ra"], self.slicePoints["ra"]
                ) and np.array_equal(
                    other_slicer.slicePoints["dec"], self.slicePoints["dec"]
                ):
                    if (
                        other_slicer.lonCol == self.lonCol
                        and other_slicer.latCol == self.latCol
                    ):
                        if other_slicer.radius == self.radius:
                            if other_slicer.useCamera == self.useCamera:
                                if (
                                    other_slicer.rotSkyPosColName
                                    == self.rotSkyPosColName
                                ):
                                    if np.array_equal(other_slicer.shape, self.shape):
                                        result = True
        return result
