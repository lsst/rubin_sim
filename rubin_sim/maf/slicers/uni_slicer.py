"""UniSlicer - no slicing at all, simply return all data points."""

__all__ = ("UniSlicer",)

from functools import wraps

import numpy as np

from .base_slicer import BaseSlicer


class UniSlicer(BaseSlicer):
    """UniSlicer."""

    def __init__(self, verbose=True, badval=-666):
        """Instantiate unislicer."""
        super(UniSlicer, self).__init__(verbose=verbose, badval=badval)
        self.nslice = 1
        self.shape = self.nslice
        self.slice_points["sid"] = np.array(
            [
                0,
            ],
            int,
        )
        self.plot_funcs = []

    def setup_slicer(self, sim_data, maps=None):
        """Use sim_data to set indexes to return."""
        self._run_maps(maps)
        sim_dataCol = sim_data.dtype.names[0]
        self.indices = np.ones(len(sim_data[sim_dataCol]), dtype="bool")

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Return all indexes in sim_data."""
            idxs = self.indices
            slice_point = {"sid": islice}
            for key in self.slice_points:
                if len(np.shape(self.slice_points[key])) == 0:
                    keyShape = 0
                else:
                    keyShape = np.shape(self.slice_points[key])[0]
                if keyShape == self.nslice:
                    slice_point[key] = self.slice_points[key][islice]
                else:
                    slice_point[key] = self.slice_points[key]
            return {"idxs": idxs, "slice_point": slice_point}

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def __eq__(self, other_slicer):
        """Evaluate if slicers are equivalent."""
        if isinstance(other_slicer, UniSlicer):
            return True
        else:
            return False
