__all__ = ("HealpixComCamSlicer",)

import warnings
from functools import wraps

import matplotlib.path as mplPath
import numpy as np
import rubin_scheduler.utils as simsUtils

from .healpix_slicer import HealpixSlicer

# The names of the chips in the central raft, aka, ComCam
center_raft_chips = [
    "R:2,2 S:0,0",
    "R:2,2 S:0,1",
    "R:2,2 S:0,2",
    "R:2,2 S:1,0",
    "R:2,2 S:1,1",
    "R:2,2 S:1,2",
    "R:2,2 S:2,0",
    "R:2,2 S:2,1",
    "R:2,2 S:2,2",
]


class HealpixComCamSlicer(HealpixSlicer):
    """Slicer that uses the ComCam footprint to decide if observations overlap a healpixel center"""

    def __init__(
        self,
        nside=128,
        lon_col="fieldRA",
        lat_col="fieldDec",
        lat_lon_deg=True,
        verbose=True,
        badval=np.nan,
        use_cache=True,
        leafsize=100,
        use_camera=False,
        rot_sky_pos_col_name="rotSkyPos",
        mjdColName="observationStartMJD",
        chipNames=center_raft_chips,
        side_length=0.7,
    ):
        """
        Parameters
        ----------
        side_length : float (0.7)
            How large is a side of the raft (degrees)
        """
        radius = side_length / 2.0 * np.sqrt(2.0)
        super(HealpixComCamSlicer, self).__init__(
            nside=nside,
            lon_col=lon_col,
            lat_col=lat_col,
            lat_lon_deg=lat_lon_deg,
            verbose=verbose,
            badval=badval,
            use_cache=use_cache,
            leafsize=leafsize,
            radius=radius,
            use_camera=use_camera,
            rot_sky_pos_col_name=rot_sky_pos_col_name,
            mjdColName=mjdColName,
            chipNames=chipNames,
        )
        self.side_length = np.radians(side_length)
        self.corners_x = np.array(
            [
                -self.side_length / 2.0,
                -self.side_length / 2.0,
                self.side_length / 2.0,
                self.side_length / 2.0,
            ]
        )
        self.corners_y = np.array(
            [
                self.side_length / 2.0,
                -self.side_length / 2.0,
                -self.side_length / 2.0,
                self.side_length / 2.0,
            ]
        )
        # Need the rotation even if not using the camera
        self.columns_needed.append(rot_sky_pos_col_name)
        self.columns_needed = list(set(self.columns_needed))

        # The 3D search radius for things inside the raft
        self.side_radius = simsUtils.xyz_angular_radius(side_length / 2.0)

    def setup_slicer(self, sim_data, maps=None):
        """Use sim_data[self.lon_col] and sim_data[self.lat_col] (in radians) to set up KDTree.

        Parameters
        -----------
        sim_data : numpy.recarray
            The simulated data, including the location of each pointing.
        maps : list of rubin_sim.maf.maps objects, optional
            List of maps (such as dust extinction) that will run to build up additional metadata at each
            slice_point. This additional metadata is available to metrics via the slice_point dictionary.
            Default None.
        """
        if maps is not None:
            if self.cache_size != 0 and len(maps) > 0:
                warnings.warn(
                    "Warning:  Loading maps but cache on." "Should probably set use_cache=False in slicer."
                )
            self._run_maps(maps)
        self._setRad(self.radius)
        if self.use_camera:
            self._setupLSSTCamera()
            self._presliceFootprint(sim_data)
        else:
            if self.latLonDeg:
                self._build_tree(
                    np.radians(sim_data[self.lon_col]),
                    np.radians(sim_data[self.lat_col]),
                    self.leafsize,
                )
            else:
                self._build_tree(sim_data[self.lon_col], sim_data[self.lat_col], self.leafsize)

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Return indexes for relevant opsim data at slice_point
            (slice_point=lon_col/lat_col value .. usually ra/dec)."""

            # Build dict for slice_point info
            slice_point = {"sid": islice}
            if self.use_camera:
                indices = self.sliceLookup[islice]
                slice_point["chipNames"] = self.chipNames[islice]
            else:
                sx, sy, sz = simsUtils._xyz_from_ra_dec(
                    self.slice_points["ra"][islice], self.slice_points["dec"][islice]
                )
                # Anything within half the side length is good no matter what rotation angle
                # the camera is at
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.side_radius)
                # Now the larger radius. Need to make it an array for easy subscripting
                initial_indices = np.array(self.opsimtree.query_ball_point((sx, sy, sz), self.rad), dtype=int)
                # remove the indices we already know about
                initial_indices = initial_indices[np.in1d(initial_indices, indices, invert=True)]

                if self.latLonDeg:
                    lat = np.radians(sim_data[self.lat_col][initial_indices])
                    lon = np.radians(sim_data[self.lon_col][initial_indices])
                    cos_rot = np.cos(np.radians(sim_data[self.rotSkyPosColName][initial_indices]))
                    sin_rot = np.cos(np.radians(sim_data[self.rotSkyPosColName][initial_indices]))
                else:
                    lat = sim_data[self.lat_col][initial_indices]
                    lon = sim_data[self.lon_col][initial_indices]
                    cos_rot = np.cos(sim_data[self.rotSkyPosColName][initial_indices])
                    sin_rot = np.sin(sim_data[self.rotSkyPosColName][initial_indices])
                # loop over the observations that might be overlapping the healpix, check each
                for i, ind in enumerate(initial_indices):
                    # Rotate the camera
                    x_rotated = self.corners_x * cos_rot[i] - self.corners_y * sin_rot[i]
                    y_rotated = self.corners_x * sin_rot[i] + self.corners_y * cos_rot[i]
                    # How far is the pointing center from the healpix center
                    xshift, yshift = simsUtils.gnomonic_project_toxy(
                        lon[i],
                        lat[i],
                        self.slice_points["ra"][islice],
                        self.slice_points["dec"][islice],
                    )

                    x_rotated += xshift
                    y_rotated += yshift
                    # Use matplotlib to make a polygon
                    bbPath = mplPath.Path(
                        np.array(
                            [
                                [x_rotated[0], y_rotated[0]],
                                [x_rotated[1], y_rotated[1]],
                                [x_rotated[2], y_rotated[2]],
                                [x_rotated[3], y_rotated[3]],
                                [x_rotated[0], y_rotated[0]],
                            ]
                        )
                    )
                    # Check if the slice_point is inside the image corners and append to list
                    if bbPath.contains_point((0.0, 0.0)):
                        indices.append(ind)

            # Loop through all the slice_point keys. If the first dimension of slice_point[key] has
            # the same shape as the slicer, assume it is information per slice_point.
            # Otherwise, pass the whole slice_point[key] information. Useful for stellar LF maps
            # where we want to pass only the relevant LF and the bins that go with it.

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
