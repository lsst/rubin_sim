# UniSlicer class.
# This slicer simply returns the indexes of all data points. No slicing done at all.

import numpy as np
from functools import wraps

from .base_slicer import BaseSlicer

__all__ = ["UniSlicer"]


class UniSlicer(BaseSlicer):
    """UniSlicer."""

    def __init__(self, verbose=True, badval=-666):
        """Instantiate unislicer."""
        super(UniSlicer, self).__init__(verbose=verbose, badval=badval)
        self.nslice = 1
        self.shape = self.nslice
        self.slicePoints["sid"] = np.array(
            [
                0,
            ],
            int,
        )
        self.plot_funcs = []

    def setup_slicer(self, sim_data, maps=None):
        """Use sim_data to set indexes to return."""
        self._runMaps(maps)
        sim_dataCol = sim_data.dtype.names[0]
        self.indices = np.ones(len(sim_data[sim_dataCol]), dtype="bool")

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Return all indexes in sim_data."""
            idxs = self.indices
            return {"idxs": idxs, "slicePoint": {"sid": islice}}

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def __eq__(self, other_slicer):
        """Evaluate if slicers are equivalent."""
        if isinstance(other_slicer, UniSlicer):
            return True
        else:
            return False
