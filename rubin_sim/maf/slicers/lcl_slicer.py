from functools import wraps
from itertools import batched

import numpy as np

from .base_slicer import BaseSlicer


class LynxBaseSlicer(BaseSlicer):
    def __init__(self, batch_size=1, *args, **kwargs):
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

        self.shape = (len(sim_data),)
        self.nslice = self.shape[0]
        self.islice = 0

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            slice_point = {"sid": islice}
            indices = np.array(self.batches[islice])

            return {"idxs": indices, "slice_point": slice_point}

        setattr(self, "_slice_sim_data", _slice_sim_data)
