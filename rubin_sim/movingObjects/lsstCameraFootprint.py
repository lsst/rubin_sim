import numpy as np
import warnings
import os
from rubin_sim.data import get_data_dir
from rubin_sim.utils import gnomonic_project_toxy

__all__ = ['LsstCameraFootprint']


def rotate(x, y, rotation_angle_rad):
    """rotate 2d points around the origin (0,0)
    """
    cos_rad = np.cos(rotation_angle_rad)
    sin_rad = np.sin(rotation_angle_rad)
    qx = cos_rad * x - sin_rad * y
    qy = sin_rad * x + cos_rad * y
    return qx, qy


class LsstCameraFootprint(object):
    """
    Class to provide the capability for identifying observations within an LSST camera footprint.
    """
    def __init__(self, obsRA='fieldRA', obsDec='fieldDec'):
        self.obsRA = obsRA
        self.obsDec = obsDec
        filename = os.path.join(get_data_dir(), 'maf/fov_map.npz')
        _temp = np.load(filename)
        self.camera_fov = _temp['image'].copy()
        self.x_camera = np.radians(_temp['x'].copy())
        _temp.close()
        self.plate_scale = self.x_camera[1] - self.x_camera[0]
        self.indx_max = len(self.x_camera)

    def __call__(self, ephems, obsData, epoch=2000.0, rmax=2.1, timeCol='observationStartMJD',
                 rotSkyCol='rotSkyPos'):
        """Determine which observations are within the actual camera footprint for a series of observations.

        Parameters
        ----------
        ephems : np.recarray
            Ephemerides for the objects, with RA and Dec as 'ra' and 'dec' columns (in degrees).
        obsData : np.recarray
            Observation pointings, with RA and Dec as 'ra' and 'dec' columns (in degrees).
            The telescope rotation angle should be in 'rotSkyPos' (in degrees), and the time of each
            pointing should be in the 'expMJD' column.
        rmax : float (2.1)
            The maximum radius from the center of the field to consider
        epoch: float, optional
            The epoch of the ephemerides and pointing data. Default 2000.0.

        Returns
        -------
        np.ndarray
            Returns the indexes of the numpy array of the object observations
            which are inside the fov and land on a science chip
        """
        # See if the object is within 'rFov' of the center of the boresight.

        x_proj, y_proj = gnomonic_project_toxy(np.radians(ephems['ra']),
                                               np.radians(ephems['dec']),
                                               np.radians(obsData[self.obsRA]),
                                               np.radians(obsData[self.obsDec]))
        # rotate them by rotskypos
        # XXX---of course I have no idea if this should be a positive or
        # negative rotation. Whatever, the focal plane is pretty symetric, so whatever.
        x_proj, y_proj = rotate(x_proj, y_proj, np.radians(obsData[rotSkyCol]))

        # look up which points are good
        x_indx = np.round((x_proj - self.x_camera[0])/self.plate_scale).astype(int)
        y_indx = np.round((y_proj - self.x_camera[0])/self.plate_scale).astype(int)
        in_range = np.where((x_indx >= 0) & (x_indx < self.indx_max) &
                            (y_indx >= 0) & (y_indx < self.indx_max))[0]
        x_indx = x_indx[in_range]
        y_indx = y_indx[in_range]
        indices = in_range
        # reduce the indices down to only the ones that fall on silicon.
        # self.camera_fov is an array of `bool` values
        map_val = self.camera_fov[x_indx, y_indx]
        indices = indices[map_val].astype(int)

        return indices
