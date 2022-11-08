from builtins import zip
from builtins import map
from builtins import range

# nd Slicer slices data on N columns in sim_data

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
from functools import wraps

from rubin_sim.maf.plots.nd_plotters import TwoDSubsetData, OneDSubsetData
from .base_slicer import BaseSlicer

__all__ = ["NDSlicer"]


class NDSlicer(BaseSlicer):
    """Nd slicer (N dimensions)"""

    def __init__(self, slice_col_list=None, verbose=True, bins_list=100):
        """Instantiate object.
        bins_list can be a list of numpy arrays with the respective slicepoints for slice_col_list,
         or a list of integers (one per column in slice_col_list) or a single value
            (repeated for all columns, default=100)."""
        super(NDSlicer, self).__init__(verbose=verbose)
        self.bins = None
        self.nslice = None
        self.slice_col_list = slice_col_list
        self.columns_needed = self.slice_col_list
        if self.slice_col_list is not None:
            self.n_d = len(self.slice_col_list)
        else:
            self.n_d = None
        self.bins_list = bins_list
        if not (isinstance(bins_list, float) or isinstance(bins_list, int)):
            if len(self.bins_list) != self.n_d:
                raise Exception(
                    "BinsList must be same length as slice_col_names, unless it is a single value"
                )
        self.slicer_init = {"slice_col_list": slice_col_list}
        self.plot_funcs = [TwoDSubsetData, OneDSubsetData]

    def setup_slicer(self, sim_data, maps=None):
        """Set up bins."""
        # Parse input bins choices.
        self.bins = []
        # If we were given a single number for the binsList, convert to list.
        if isinstance(self.bins_list, float) or isinstance(self.bins_list, int):
            self.bins_list = [self.bins_list for c in self.slice_col_list]
        # And then build bins.
        for bl, col in zip(self.bins_list, self.slice_col_list):
            if isinstance(bl, float) or isinstance(bl, int):
                slice_col = sim_data[col]
                bin_min = slice_col.min()
                bin_max = slice_col.max()
                if bin_min == bin_max:
                    warnings.warn(
                        "BinMin=BinMax for column %s: increasing bin_max by 1." % (col)
                    )
                    bin_max = bin_max + 1
                binsize = (bin_max - bin_min) / float(bl)
                bins = np.arange(bin_min, bin_max + binsize / 2.0, binsize, "float")
                self.bins.append(bins)
            else:
                self.bins.append(np.sort(bl))
        # Count how many bins we have total (not counting last 'RHS' bin values, as in oneDSlicer).
        self.nslice = (np.array(list(map(len, self.bins))) - 1).prod()
        # Set up slice metadata.
        self.slice_points["sid"] = np.arange(self.nslice)
        # Including multi-D 'leftmost' bin values
        bins_for_iteration = []
        for b in self.bins:
            bins_for_iteration.append(b[:-1])
        biniterator = itertools.product(*bins_for_iteration)
        self.slice_points["bins"] = []
        for b in biniterator:
            self.slice_points["bins"].append(b)
        # and multi-D 'leftmost' bin indexes corresponding to each sid
        self.slice_points["binIdxs"] = []
        bin_ids_for_iteration = []
        for b in self.bins:
            bin_ids_for_iteration.append(np.arange(len(b[:-1])))
        bin_id_iterator = itertools.product(*bin_ids_for_iteration)
        for bidx in bin_id_iterator:
            self.slice_points["binIdxs"].append(bidx)
        # Add metadata from maps.
        self._run_maps(maps)
        # Set up indexing for data slicing.
        self.sim_idxs = []
        self.lefts = []
        for slice_col_name, bins in zip(self.slice_col_list, self.bins):
            sim_idxs = np.argsort(sim_data[slice_col_name])
            sim_fields_sorted = np.sort(sim_data[slice_col_name])
            # "left" values are location where simdata == bin value
            left = np.searchsorted(sim_fields_sorted, bins[:-1], "left")
            left = np.concatenate(
                (
                    left,
                    np.array(
                        [
                            len(sim_idxs),
                        ]
                    ),
                )
            )
            # Add these calculated values into the class lists of sim_idxs and lefts.
            self.sim_idxs.append(sim_idxs)
            self.lefts.append(left)

        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Slice sim_data to return relevant indexes for slicepoint."""
            # Identify relevant pointings in each dimension.
            sim_idxs_list = []
            # Translate islice into indexes in each bin dimension
            bin_idxs = self.slice_points["binIdxs"][islice]
            for d, i in zip(list(range(self.n_d)), bin_idxs):
                sim_idxs_list.append(
                    set(self.sim_idxs[d][self.lefts[d][i] : self.lefts[d][i + 1]])
                )
            idxs = list(set.intersection(*sim_idxs_list))
            return {
                "idxs": idxs,
                "slice_point": {
                    "sid": islice,
                    "binLeft": self.slice_points["bins"][islice],
                    "binIdx": self.slice_points["binIdxs"][islice],
                },
            }

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def __eq__(self, other_slicer):
        """Evaluate if grids are equivalent."""
        if isinstance(other_slicer, NDSlicer):
            if other_slicer.n_d != self.n_d:
                return False
            for i in range(self.n_d):
                if not np.array_equal(
                    other_slicer.slice_points["bins"][i], self.slice_points["bins"][i]
                ):
                    return False
            return True
        else:
            return False
