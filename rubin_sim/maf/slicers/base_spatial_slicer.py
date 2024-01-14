# The base class for all spatial slicers.
# Slicers are 'data slicers' at heart; spatial slicers slice data by RA/Dec and
# return the relevant indices in the sim_data to the metric.
# The primary things added here are the methods to slice the data
# (for any spatial slicer) as this uses a KD-tree built on spatial
# (RA/Dec type) indexes.

__all__ = ("BaseSpatialSlicer",)

import warnings
from functools import wraps

import numpy as np
import rubin_scheduler.utils as simsUtils

from rubin_sim.maf.plots.spatial_plotters import BaseHistogram, BaseSkyMap

from .base_slicer import BaseSlicer


def rotate(x, y, rotation_angle_rad):
    """rotate 2d points around the origin (0,0)"""
    cos_rad = np.cos(rotation_angle_rad)
    sin_rad = np.sin(rotation_angle_rad)
    qx = cos_rad * x - sin_rad * y
    qy = sin_rad * x + cos_rad * y
    return qx, qy


class BaseSpatialSlicer(BaseSlicer):
    """Base spatial slicer object, contains additional functionality
    for spatial slicing, including setting up and traversing a kdtree
    containing the simulated data points.

    Parameters
    ----------
    lon_col : `str`, optional
        Name of the longitude (RA equivalent) column.
    lat_col : `str`, optional
        Name of the latitude (Dec equivalent) column.
    rot_sky_pos_col_name : `str`, optional
        Name of the rotSkyPos column in the input  data.
        Only used if use_camera is True.
        Describes the orientation of the camera orientation on the sky.
    lat_lon_deg : `bool`, optional
        Flag indicating whether lat and lon values from input data are
        in degrees (True) or radians (False).
    verbose : `bool`, optional
        Verbosity flag - noisy or quiet.
    badval : `float`, optional
        Bad value flag, relevant for plotting.
    leafsize : `int`, optional
        Leafsize value for kdtree.
    radius : `float`, optional
        Radius for matching in the kdtree.
        Equivalent to the radius of the FOV, in degrees.
    use_camera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
    camera_footprint_file : `str`, optional
        Name of the camera footprint map to use.
        Can be None, which will use the default file.
    """

    def __init__(
        self,
        lon_col="fieldRA",
        lat_col="fieldDec",
        rot_sky_pos_col_name="rotSkyPos",
        lat_lon_deg=True,
        verbose=True,
        badval=-666,
        leafsize=100,
        radius=2.45,
        use_camera=True,
        camera_footprint_file=None,
    ):
        super().__init__(verbose=verbose, badval=badval)
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.latLonDeg = lat_lon_deg
        self.rotSkyPosColName = rot_sky_pos_col_name
        self.columns_needed = [lon_col, lat_col, rot_sky_pos_col_name]
        self.use_camera = use_camera
        self.camera_footprint_file = camera_footprint_file
        self.radius = radius
        self.leafsize = leafsize
        self.slicer_init = {
            "lon_col": self.lon_col,
            "lat_col": self.lat_col,
            "radius": self.radius,
            "badval": self.badval,
            "use_camera": self.use_camera,
        }
        # RA and Dec are required slice_point info for any spatial slicer.
        # slice_point RA/Dec are in radians.
        self.slice_points["sid"] = None
        self.slice_points["ra"] = None
        self.slice_points["dec"] = None
        self.nslice = None
        self.shape = None
        self.plot_funcs = [BaseHistogram, BaseSkyMap]

    def setup_slicer(self, sim_data, maps=None):
        """Use sim_data[self.lon_col] and sim_data[self.lat_col]
        (in radians) to set up KDTree.

        Parameters
        -----------
        sim_data : `numpy.ndarray`
            The simulated data, including the location of each pointing.
        maps : `list` of `rubin_sim.maf.maps` objects, optional
            List of maps (such as dust extinction) that will run to build up
            additional data at each slice_point. This additional data
            is available to metrics via the slice_point dictionary.
        """
        if maps is not None:
            if self.cache_size != 0 and len(maps) > 0:
                warnings.warn(
                    "Warning:  Loading maps but cache on." "Should probably set use_cache=False in slicer."
                )
            self._run_maps(maps)
        self._set_rad(self.radius)

        if self.latLonDeg:
            self.data_ra = np.radians(sim_data[self.lon_col])
            self.data_dec = np.radians(sim_data[self.lat_col])
            self.data_rot = np.radians(sim_data[self.rotSkyPosColName])
        else:
            self.data_ra = sim_data[self.lon_col]
            self.data_dec = sim_data[self.lat_col]
            self.data_rot = sim_data[self.rotSkyPosColName]
        if self.use_camera:
            self._setupLSSTCamera()

        self._build_tree(self.data_ra, self.data_dec, self.leafsize)

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Return indexes for relevant opsim data at slice_point
            (slice_point=lon_col/lat_col value .. usually ra/dec).
            """

            # Build dict for slice_point info
            slice_point = {"sid": islice}
            sx, sy, sz = simsUtils._xyz_from_ra_dec(
                self.slice_points["ra"][islice], self.slice_points["dec"][islice]
            )
            # Query against tree.
            indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)

            if (self.use_camera) & (len(indices) > 0):
                # Find the indices *of those indices*
                # which fall in the camera footprint
                camera_idx = self.camera(
                    self.slice_points["ra"][islice],
                    self.slice_points["dec"][islice],
                    self.data_ra[indices],
                    self.data_dec[indices],
                    self.data_rot[indices],
                )
                indices = np.array(indices)[camera_idx]

            # Loop through all the slice_point keys.
            # If the first dimension of slice_point[key] has the same shape
            # as the slicer, assume it is information per slice_point.
            # Otherwise, pass the whole slice_point[key] information.
            # Useful for stellar LF maps where we want to pass only the
            # relevant LF and the bins that go with it.
            for key in self.slice_points:
                if len(np.shape(self.slice_points[key])) == 0:
                    keyShape = 0
                else:
                    keyShape = np.shape(self.slice_points[key])[0]
                if keyShape == self.nslice:
                    slice_point[key] = self.slice_points[key][islice]
                else:
                    slice_point[key] = self.slice_points[key]
            return {"idxs": indices, "slice_point": slice_point}

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def _setupLSSTCamera(self):
        """If we want to include the camera chip gaps, etc."""
        self.camera = simsUtils.LsstCameraFootprint(
            units="radians", footprint_file=self.camera_footprint_file
        )

    def _build_tree(self, sim_dataRa, sim_dataDec, leafsize=100):
        """Build KD tree on sim_dataRA/Dec.

        sim_dataRA, sim_dataDec = RA and Dec values (in radians).
        leafsize = the number of Ra/Dec pointings in each leaf node.
        """
        self.opsimtree = simsUtils._build_tree(sim_dataRa, sim_dataDec, leafsize)

    def _set_rad(self, radius=1.75):
        """Set radius (in degrees) for kdtree search."""
        self.rad = simsUtils.xyz_angular_radius(radius)
