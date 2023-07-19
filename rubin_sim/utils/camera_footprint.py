__all__ = ("LsstCameraFootprint",)

import os

import numpy as np

from rubin_sim.data import get_data_dir

from .projections import gnomonic_project_toxy


def rotate(x, y, rotation_angle_rad):
    """rotate 2d points around the origin (0,0)"""
    cos_rad = np.cos(rotation_angle_rad)
    sin_rad = np.sin(rotation_angle_rad)
    qx = cos_rad * x - sin_rad * y
    qy = sin_rad * x + cos_rad * y
    return qx, qy


class LsstCameraFootprint:
    """
    Identify point on the sky within an LSST camera footprint.

    Parameters
    ----------
    units : `str`, opt
        Units for the object RA/Dec and boresight RA/Dec/rotation values.
        Default 'degrees'.  If not degrees, assumes incoming values are in radians.
    footprint_file : `str` or None, opt
        Location for the camera footprint map.
        Default None loads the default map from $RUBIN_SIM_DATA_DIR/maf/fov_map.npz

    """

    def __init__(self, units="degrees", footprint_file=None):
        if footprint_file is None:
            footprint_file = os.path.join(get_data_dir(), "maf", "fov_map.npz")
        _temp = np.load(footprint_file)
        # Units refers to the incoming ra/dec values in the _call__ method
        # Internally, radians are used to calculate the footprint
        self.units = units
        self.camera_fov = _temp["image"].copy()
        self.x_camera = np.radians(_temp["x"].copy())
        _temp.close()
        self.plate_scale = self.x_camera[1] - self.x_camera[0]
        self.indx_max = len(self.x_camera)

    def __call__(self, obj_ra, obj_dec, boresight_ra, boresight_dec, boresight_rot_sky_pos):
        """Determine which observations are within the actual camera footprint for a series of observations.

        Parameters
        ----------
        obj_ra : `np.ndarray`
            RA values for the objects (in the field of view?)
        obj_dec : `np.ndarray`
            Dec values for the objects
        boresight_ra : `float`
            RA value for the pointing
        boresight_dec : `float`
            Dec value for the pointing
        boresight_rot_sky_pos : `float`
            RotSkyPos value for the pointing.

        Returns
        -------
        indices : `np.ndarray`
            Returns the indexes of the numpy array of the object observations
            which are inside the fov and land on a science chip.
            Applying this to the input array (e.g. obj_ra[indices]) indicates the positions of
            the objects which fell onto active silicon.
        """
        if self.units == "degrees":
            x_proj, y_proj = gnomonic_project_toxy(
                np.radians(obj_ra),
                np.radians(obj_dec),
                np.radians(boresight_ra),
                np.radians(boresight_dec),
            )
            # rotate them by rotskypos
            # TODO: look up whether this is a positive or negative rotation
            #  in the observatory documentation
            x_proj, y_proj = rotate(x_proj, y_proj, np.radians(boresight_rot_sky_pos))

        else:
            x_proj, y_proj = gnomonic_project_toxy(obj_ra, obj_dec, boresight_ra, boresight_dec)
            x_proj, y_proj = rotate(x_proj, y_proj, boresight_rot_sky_pos)

        # look up which points are good
        x_indx = np.round((x_proj - self.x_camera[0]) / self.plate_scale).astype(int)
        y_indx = np.round((y_proj - self.x_camera[0]) / self.plate_scale).astype(int)
        in_range = np.where(
            (x_indx >= 0) & (x_indx < self.indx_max) & (y_indx >= 0) & (y_indx < self.indx_max)
        )[0]
        x_indx = x_indx[in_range]
        y_indx = y_indx[in_range]
        indices = in_range
        # reduce the indices down to only the ones that fall on silicon.
        # self.camera_fov is an array of `bool` values
        map_val = self.camera_fov[x_indx, y_indx]
        indices = indices[map_val]
        return indices
