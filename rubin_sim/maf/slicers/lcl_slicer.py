from functools import wraps
from itertools import batched

import numpy as np

from .base_slicer import BaseSlicer

# from .base_spatial_slicer import BaseSpatialSlicer
from .healpix_slicer import HealpixSlicer


class LynxBaseSlicer(BaseSlicer):
    def __init__(self, batch_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra_col = "ra"
        self.dec_col = "dec"
        self.batch_size = batch_size

    def setup_slicer(self, sim_data, maps=None):
        """Set up the slicer using the provided simulation data.

        Parameters
        ----------
        sim_data : `numpy.ndarray`
            The simulation data to be sliced.
        maps : `dict`, optional
            Optional maps to be used in the slicing process.
        """
        indicies = range(len(sim_data))
        self.batches = list(batched(indicies, self.batch_size))

        self.shape = (len(self.batches),)
        self.nslice = self.shape[0]
        self.islice = 0

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            slice_point = {"sid": islice}
            indices = np.array(self.batches[islice])

            return {"idxs": indices, "slice_point": slice_point}

        setattr(self, "_slice_sim_data", _slice_sim_data)


class LynxSpatialSlicer(LynxBaseSlicer, HealpixSlicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lon_col="ra", lat_col="dec", rot_sky_pos_col_name=None, **kwargs)

    def setup_slicer(self, sim_data, maps=None):
        """Set up the slicer using the provided simulation data.

        Parameters
        ----------
        sim_data : `numpy.ndarray`
            The simulation data to be sliced.
        maps : `dict`, optional
            Optional maps to be used in the slicing process.
        """
        self.use_camera = False
        HealpixSlicer.setup_slicer(self, sim_data, maps=maps)
