import numpy as np
import healpy as hp
from .healpixSlicer import HealpixSlicer
import warnings
from functools import wraps
import rubin_sim.utils as simsUtils
import matplotlib.path as mplPath
from rubin_sim.maf.utils.mafUtils import gnomonic_project_toxy


__all__ = ['HealpixComCamSlicer']


# The names of the chips in the central raft, aka, ComCam
center_raft_chips = ['R:2,2 S:0,0', 'R:2,2 S:0,1', 'R:2,2 S:0,2',
                     'R:2,2 S:1,0', 'R:2,2 S:1,1', 'R:2,2 S:1,2',
                     'R:2,2 S:2,0', 'R:2,2 S:2,1', 'R:2,2 S:2,2']


class HealpixComCamSlicer(HealpixSlicer):
    """Slicer that uses the ComCam footprint to decide if observations overlap a healpixel center
    """

    def __init__(self, nside=128, lonCol='fieldRA',
                 latCol='fieldDec', latLonDeg=True, verbose=True, badval=hp.UNSEEN,
                 useCache=True, leafsize=100,
                 useCamera=False, rotSkyPosColName='rotSkyPos',
                 mjdColName='observationStartMJD', chipNames=center_raft_chips, side_length=0.7):
        """
        Parameters
        ----------
        side_length : float (0.7)
            How large is a side of the raft (degrees)
        """
        radius = side_length/2.*np.sqrt(2.)
        super(HealpixComCamSlicer, self).__init__(nside=nside, lonCol=lonCol, latCol=latCol,
                                                  latLonDeg=latLonDeg,
                                                  verbose=verbose, badval=badval, useCache=useCache,
                                                  leafsize=leafsize, radius=radius, useCamera=useCamera,
                                                  rotSkyPosColName=rotSkyPosColName,
                                                  mjdColName=mjdColName, chipNames=chipNames)
        self.side_length = np.radians(side_length)
        self.corners_x = np.array([-self.side_length/2., -self.side_length/2., self.side_length/2.,
                                  self.side_length/2.])
        self.corners_y = np.array([self.side_length/2., -self.side_length/2., -self.side_length/2.,
                                  self.side_length/2.])
        # Need the rotation even if not using the camera
        self.columnsNeeded.append(rotSkyPosColName)
        self.columnsNeeded = list(set(self.columnsNeeded))

        # The 3D search radius for things inside the raft
        self.side_radius = simsUtils.xyz_angular_radius(side_length/2.)

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
            self._setupLSSTCamera()
            self._presliceFootprint(simData)
        else:
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
            if self.useCamera:
                indices = self.sliceLookup[islice]
                slicePoint['chipNames'] = self.chipNames[islice]
            else:
                sx, sy, sz = simsUtils._xyz_from_ra_dec(self.slicePoints['ra'][islice],
                                                        self.slicePoints['dec'][islice])
                # Anything within half the side length is good no matter what rotation angle
                # the camera is at
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.side_radius)
                # Now the larger radius. Need to make it an array for easy subscripting
                initial_indices = np.array(self.opsimtree.query_ball_point((sx, sy, sz), self.rad), dtype=int)
                # remove the indices we already know about
                initial_indices = initial_indices[np.in1d(initial_indices, indices, invert=True)]

                if self.latLonDeg:
                    lat = np.radians(simData[self.latCol][initial_indices])
                    lon = np.radians(simData[self.lonCol][initial_indices])
                    cos_rot = np.cos(np.radians(simData[self.rotSkyPosColName][initial_indices]))
                    sin_rot = np.cos(np.radians(simData[self.rotSkyPosColName][initial_indices]))
                else:
                    lat = simData[self.latCol][initial_indices]
                    lon = simData[self.lonCol][initial_indices]
                    cos_rot = np.cos(simData[self.rotSkyPosColName][initial_indices])
                    sin_rot = np.sin(simData[self.rotSkyPosColName][initial_indices])
                # loop over the observations that might be overlapping the healpix, check each
                for i, ind in enumerate(initial_indices):
                    # Rotate the camera
                    x_rotated = self.corners_x*cos_rot[i] - self.corners_y*sin_rot[i]
                    y_rotated = self.corners_x*sin_rot[i] + self.corners_y*cos_rot[i]
                    # How far is the pointing center from the healpix center
                    xshift, yshift = gnomonic_project_toxy(lon[i], lat[i], self.slicePoints['ra'][islice],
                                                           self.slicePoints['dec'][islice])

                    x_rotated += xshift
                    y_rotated += yshift
                    # Use matplotlib to make a polygon
                    bbPath = mplPath.Path(np.array([[x_rotated[0], y_rotated[0]],
                                                   [x_rotated[1], y_rotated[1]],
                                                   [x_rotated[2], y_rotated[2]],
                                                   [x_rotated[3], y_rotated[3]],
                                                   [x_rotated[0], y_rotated[0]]]))
                    # Check if the slicepoint is inside the image corners and append to list
                    if bbPath.contains_point((0., 0.)):
                        indices.append(ind)

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
