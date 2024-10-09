__all__ = ("HealpixSDSSSlicer",)

from functools import wraps

import matplotlib.path as mplPath
import numpy as np
from rubin_scheduler.utils import _xyz_from_ra_dec, gnomonic_project_toxy

from rubin_sim.maf.plots import HealpixSDSSSkyMap

from .healpix_slicer import HealpixSlicer


class HealpixSDSSSlicer(HealpixSlicer):
    """For use with SDSS stripe 82 square images"""

    def __init__(
        self,
        nside=128,
        lon_col="RA1",
        lat_col="Dec1",
        verbose=True,
        use_cache=True,
        radius=17.0 / 60.0,
        leafsize=100,
        **kwargs,
    ):
        """Using one corner of the chip as the spatial key and the
        diagonal as the radius."""
        super().__init__(
            verbose=verbose,
            lon_col=lon_col,
            lat_col=lat_col,
            radius=radius,
            leafsize=leafsize,
            use_cache=use_cache,
            nside=nside,
        )
        self.cornerLables = ["RA1", "Dec1", "RA2", "Dec2", "RA3", "Dec3", "RA4", "Dec4"]
        self.plot_funcs = [
            HealpixSDSSSkyMap,
        ]

    def setup_slicer(self, sim_data, maps=None):
        """
        Use sim_data[self.lon_col] and sim_data[self.lat_col]
        (in radians) to set up KDTree.
        """
        self._run_maps(maps)
        self._build_tree(sim_data[self.lon_col], sim_data[self.lat_col], self.leafsize)
        self._set_rad(self.radius)
        self.corners = sim_data[self.cornerLables]

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Return indexes for relevant opsim data at slice_point
            (slice_point=lon_col/lat_col value .. usually ra/dec)."""
            sx, sy, sz = _xyz_from_ra_dec(self.slice_points["ra"][islice], self.slice_points["dec"][islice])
            # Query against tree.
            initIndices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
            # Loop through all the images and check if the slice_point
            # is inside the corners of the chip
            indices = []
            # Gnomic project all the corners that are near the slice point,
            # centered on slice point
            x1, y1 = gnomonic_project_toxy(
                self.corners["RA1"][initIndices],
                self.corners["Dec1"][initIndices],
                self.slice_points["ra"][islice],
                self.slice_points["dec"][islice],
            )
            x2, y2 = gnomonic_project_toxy(
                self.corners["RA2"][initIndices],
                self.corners["Dec2"][initIndices],
                self.slice_points["ra"][islice],
                self.slice_points["dec"][islice],
            )
            x3, y3 = gnomonic_project_toxy(
                self.corners["RA3"][initIndices],
                self.corners["Dec3"][initIndices],
                self.slice_points["ra"][islice],
                self.slice_points["dec"][islice],
            )
            x4, y4 = gnomonic_project_toxy(
                self.corners["RA4"][initIndices],
                self.corners["Dec4"][initIndices],
                self.slice_points["ra"][islice],
                self.slice_points["dec"][islice],
            )

            for i, ind in enumerate(initIndices):
                # Use matplotlib to make a polygon on
                bbPath = mplPath.Path(
                    np.array(
                        [
                            [x1[i], y1[i]],
                            [x2[i], y2[i]],
                            [x3[i], y3[i]],
                            [x4[i], y4[i]],
                            [x1[i], y1[i]],
                        ]
                    )
                )
                # Check if the slice_point is inside the image corners
                # and append to list if it is
                if bbPath.contains_point((0.0, 0.0)) == 1:
                    indices.append(ind)

            return {
                "idxs": indices,
                "slice_point": {
                    "sid": self.slice_points["sid"][islice],
                    "ra": self.slice_points["ra"][islice],
                    "dec": self.slice_points["dec"][islice],
                },
            }

        setattr(self, "_slice_sim_data", _slice_sim_data)
