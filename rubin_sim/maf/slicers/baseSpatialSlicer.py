# The base class for all spatial slicers.
# Slicers are 'data slicers' at heart; spatial slicers slice data by RA/Dec and
#  return the relevant indices in the simData to the metric.
# The primary things added here are the methods to slice the data (for any spatial slicer)
#  as this uses a KD-tree built on spatial (RA/Dec type) indexes.

import warnings
import numpy as np
from functools import wraps
from rubin_sim.maf.plots.spatialPlotters import BaseHistogram, BaseSkyMap
import rubin_sim.utils as simsUtils
from .baseSlicer import BaseSlicer
from rubin_sim.data import get_data_dir
import os

__all__ = ['BaseSpatialSlicer']


def rotate(x, y, rotation_angle_rad):
    """rotate 2d points around the origin (0,0)
    """
    cos_rad = np.cos(rotation_angle_rad)
    sin_rad = np.sin(rotation_angle_rad)
    qx = cos_rad * x - sin_rad * y
    qy = sin_rad * x + cos_rad * y
    return qx, qy

class BaseSpatialSlicer(BaseSlicer):
    """Base spatial slicer object, contains additional functionality for spatial slicing,
    including setting up and traversing a kdtree containing the simulated data points.

    Parameters
    ----------
    lonCol : str, optional
        Name of the longitude (RA equivalent) column to use from the input data.
        Default fieldRA
    latCol : str, optional
        Name of the latitude (Dec equivalent) column to use from the input data.
        Default fieldDec
    latLonDeg : `bool`, optional
        Flag indicating whether lat and lon values from input data are in degrees (True) or radians (False).
        Default True.
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : float, optional
        Bad value flag, relevant for plotting. Default -666.
    leafsize : int, optional
        Leafsize value for kdtree. Default 100.
    radius : float, optional
        Radius for matching in the kdtree. Equivalent to the radius of the FOV. Degrees.
        Default 1.75.
    useCamera : `bool`, optional
        Flag to indicate whether to use the LSST camera footprint or not.
        Default False.
    rotSkyPosColName : str, optional
        Name of the rotSkyPos column in the input  data. Only used if useCamera is True.
        Describes the orientation of the camera orientation compared to the sky.
        Default rotSkyPos.
    """
    def __init__(self, lonCol='fieldRA', latCol='fieldDec', latLonDeg=True,
                 verbose=True, badval=-666, leafsize=100, radius=2.45,
                 useCamera=True, rotSkyPosColName='rotSkyPos'):
        super().__init__(verbose=verbose, badval=badval)
        self.lonCol = lonCol
        self.latCol = latCol
        self.latLonDeg = latLonDeg
        self.rotSkyPosColName = rotSkyPosColName
        self.columnsNeeded = [lonCol, latCol]
        self.useCamera = useCamera
        if useCamera:
            self.columnsNeeded.append(rotSkyPosColName)
        self.slicer_init = {'lonCol': lonCol, 'latCol': latCol,
                            'radius': radius, 'badval': badval,
                            'useCamera': useCamera}
        self.radius = radius
        self.leafsize = leafsize
        self.useCamera = useCamera
        # RA and Dec are required slicePoint info for any spatial slicer. Slicepoint RA/Dec are in radians.
        self.slicePoints['sid'] = None
        self.slicePoints['ra'] = None
        self.slicePoints['dec'] = None
        self.nslice = None
        self.shape = None
        self.plotFuncs = [BaseHistogram, BaseSkyMap]

    def setupSlicer(self, simData, maps=None):
        """Use simData[self.lonCol] and simData[self.latCol] (in radians) to set up KDTree.

        Parameters
        -----------
        simData : numpy.recarray
            The simulated data, including the location of each pointing.
        maps : list of rubin_sim.maf.maps objects, optional
            List of maps (such as dust extinction) that will run to build up additional metadata at each
            slicePoint. This additional metadata is available to metrics via the slicePoint dictionary.
            Default None.
        """
        if maps is not None:
            if self.cacheSize != 0 and len(maps) > 0:
                warnings.warn('Warning:  Loading maps but cache on.'
                              'Should probably set useCache=False in slicer.')
            self._runMaps(maps)
        self._setRad(self.radius)
        if self.useCamera:
            self.data_ra = simData[self.lonCol]
            self.data_dec = simData[self.latCol]
            self.data_rot = simData[self.rotSkyPosColName]
            if self.latLonDeg:
                self.data_ra = np.radians(self.data_ra)
                self.data_dec = np.radians(self.data_dec)
                self.data_rot = np.radians(self.data_rot)
            self._setupLSSTCamera()
        if self.latLonDeg:
            self._buildTree(np.radians(simData[self.lonCol]),
                            np.radians(simData[self.latCol]), self.leafsize)
        else:
            self._buildTree(simData[self.lonCol], simData[self.latCol], self.leafsize)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=lonCol/latCol value .. usually ra/dec)."""

            # Build dict for slicePoint info
            slicePoint = {}
            sx, sy, sz = simsUtils._xyz_from_ra_dec(self.slicePoints['ra'][islice],
                                                    self.slicePoints['dec'][islice])
            # Query against tree.
            indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)

            if (self.useCamera) & (len(indices) > 0):
                x_proj, y_proj = simsUtils.gnomonic_project_toxy(self.slicePoints['ra'][islice],
                                                                 self.slicePoints['dec'][islice],
                                                                 self.data_ra[indices],
                                                                 self.data_dec[indices])
                # rotate them by rotskypos
                # XXX---of course I have no idea if this should be a positive or
                # negative rotation. Whatever, the focal plane is pretty symetric, so whatever.
                x_proj, y_proj = rotate(x_proj, y_proj, self.data_rot[indices])

                # look up which points are good
                x_indx = np.round((x_proj - self.x_camera[0])/self.plate_scale).astype(int)
                y_indx = np.round((y_proj - self.x_camera[0])/self.plate_scale).astype(int)
                in_range = np.where((x_indx >= 0) & (x_indx < self.indx_max) &
                                    (y_indx >= 0) & (y_indx < self.indx_max))[0]
                indices = np.array(indices)[in_range]
                x_indx = x_indx[in_range]
                y_indx = y_indx[in_range]
                # reduce the indices down to only the ones that fall on silicon.
                # self.camera_fov is an array of `bool` values
                map_val = self.camera_fov[x_indx, y_indx]
                indices = indices[map_val].tolist()

            # Loop through all the slicePoint keys. If the first dimension of slicepoint[key] has
            # the same shape as the slicer, assume it is information per slicepoint.
            # Otherwise, pass the whole slicePoint[key] information. Useful for stellar LF maps
            # where we want to pass only the relevant LF and the bins that go with it.
            for key in self.slicePoints:
                if len(np.shape(self.slicePoints[key])) == 0:
                    keyShape = 0
                else:
                    keyShape = np.shape(self.slicePoints[key])[0]
                if (keyShape == self.nslice):
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs': indices, 'slicePoint': slicePoint}
        setattr(self, '_sliceSimData', _sliceSimData)

    def _setupLSSTCamera(self):
        """If we want to include the camera chip gaps, etc"""
        filename = os.path.join(get_data_dir(), 'maf/fov_map.npz')
        _temp = np.load(filename)
        self.camera_fov = _temp['image'].copy()
        self.x_camera = np.radians(_temp['x'].copy())
        _temp.close()
        self.plate_scale = self.x_camera[1] - self.x_camera[0]
        self.indx_max = len(self.x_camera)

    def _buildTree(self, simDataRa, simDataDec, leafsize=100):
        """Build KD tree on simDataRA/Dec using utility function from mafUtils.

        simDataRA, simDataDec = RA and Dec values (in radians).
        leafsize = the number of Ra/Dec pointings in each leaf node."""
        self.opsimtree = simsUtils._buildTree(simDataRa,
                                              simDataDec,
                                              leafsize)

    def _setRad(self, radius=1.75):
        """Set radius (in degrees) for kdtree search using utility function from mafUtils."""
        self.rad = simsUtils.xyz_angular_radius(radius)
