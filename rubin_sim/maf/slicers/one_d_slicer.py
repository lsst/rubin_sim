__all__ = ("OneDSlicer",)

import warnings
from functools import wraps

import numpy as np

from rubin_sim.maf.plots.oned_plotters import OneDBinnedData
from rubin_sim.maf.stackers import ColInfo
from rubin_sim.maf.utils import optimal_bins

from .base_slicer import BaseSlicer


class OneDSlicer(BaseSlicer):
    """OneD Slicer allows the slicing of data into bins in a single dimension.

    Parameters
    ----------
    slice_col_name : `str`
        The name of the data column to base slicing on (i.e. 'airmass', etc.)
    slice_col_units : `str`, optional
        Set a name for the units of the sliceCol. Used for plotting labels.
    bins : np.ndarray, optional
        The data will be sliced into 'bins': this can be defined as an
        array here. Default None.
    bin_min : `float`, optional
    bin_max : `float`, optional
    bin_size : `float`, optional
        If bins is not defined, then bin_min/bin_max/bin_size can be chosen
        to anchor the slice points.
        Default None.
        Priority goes: bins >> bin_min/bin_max/bin_size >> data values
        (if none of the above are chosen).

    Notes
    -----
    All bins except for the last bin are half-open ([a, b)) while the
    last bin is ([a, b]).
    """

    def __init__(
        self,
        slice_col_name=None,
        slice_col_units=None,
        bins=None,
        bin_min=None,
        bin_max=None,
        bin_size=None,
        verbose=True,
        badval=0,
    ):
        super().__init__(verbose=verbose, badval=badval)
        if slice_col_name is None:
            raise ValueError("slice_col_name cannot be left None - choose a data column to group data by")
        self.slice_col_name = slice_col_name
        self.columns_needed = [slice_col_name]
        # We could try to set up the self.bins here --
        # but it's also possible that
        # these bin_min/max/size values have not been set and
        # should just be set from the data.
        self.bins = bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_size = bin_size
        # Forget binmin/max/stepsize if bins was set
        if self.bins is not None:
            if bin_min is not None or bin_max is not None or bin_size is not None:
                warnings.warning(
                    f"Both bins and one of the bin_min/bin_max/bin_size was specified. "
                    f"Using bins ({self.bins} values only."
                )
                self.bin_min = None
                self.bin_max = None
                self.bin_size = None
        # Set the column units
        if slice_col_units is not None:
            self.slice_col_units = slice_col_units
        # Try to determine the column units
        else:
            co = ColInfo()
            self.slice_col_units = co.get_units(self.slice_col_name)
        # Set slicer re-initialize values and default plotFunction
        self.slicer_init = {
            "slice_col_name": self.slice_col_name,
            "slice_col_units": slice_col_units,
            "badval": badval,
            "bin_min": self.bin_min,
            "bin_max": self.bin_max,
            "bin_size": self.bin_size,
            "bins": self.bins,
        }
        self.plot_funcs = [
            OneDBinnedData,
        ]

    def setup_slicer(self, sim_data, maps=None):
        """Set up bins in slicer.

        This happens AFTER sim_data is defined,
        thus typically in the MetricBundleGroup.
        This maps data into the bins;
        it's not a good idea to reuse a OneDSlicer as a result.
        """
        if "bins" in self.slice_points:
            warning_msg = "Warning: this OneDSlicer was already set up once. "
            warning_msg += (
                "Re-setting up a OneDSlicer is unpredictable; at the very least, it "
                "will change the mapping of the simulated data into the data slices. "
                "A safer choice is to use a separate OneDSlicer for each MetricBundle."
            )
            warnings.warn(warning_msg)
        slice_col = sim_data[self.slice_col_name]
        # Set bins from data or specified values,
        # if they were previously defined.
        if self.bins is None:
            # Set bin min/max values (could have been set in __init__)
            if self.bin_min is None:
                self.bin_min = np.nanmin(slice_col)
            if self.bin_max is None:
                self.bin_max = np.nanmax(slice_col)
            # Give warning if bin_min = bin_max,
            # and do something at least slightly reasonable.
            if self.bin_min == self.bin_max:
                warnings.warn(
                    "bin_min = bin_max (maybe your data is single-valued?). "
                    "Increasing bin_max by 1 (or 2*bin_size, if bin_size was set)."
                )
                if self.bin_size is not None:
                    self.bin_max = self.bin_max + 2 * self.bin_size
                else:
                    self.bin_max = self.bin_max + 1
            if self.bin_size is None:
                bins = optimal_bins(slice_col, self.bin_min, self.bin_max)
                nbins = np.round(bins)
                self.bin_size = (self.bin_max - self.bin_min) / float(nbins)
            # Set bins
            self.bins = np.arange(self.bin_min, self.bin_max + self.bin_size / 2.0, self.bin_size, "float")
        # nslice is used to stop iteration and should
        # reflect the usable length of the bins
        self.nslice = len(self.bins) - 1
        # and "shape" refers to the length of the datavalues,
        # and should be one less than # of bins because last
        # binvalue is RH edge only
        self.shape = self.nslice
        # Set slice_point metadata.
        self.slice_points["sid"] = np.arange(self.nslice)
        self.slice_points["bins"] = self.bins
        # Add metadata from map if needed.
        self._run_maps(maps)

        indxs = np.argsort(sim_data[self.slice_col_name])
        data_sorted = sim_data[self.slice_col_name][indxs]

        # Setting up slices such that left_edge <= data < right_edge
        # in each slice.
        left = np.searchsorted(data_sorted, self.bins[0:-1], "left")
        right = np.searchsorted(data_sorted, self.bins[1:], "left")

        self.sim_idxs = [indxs[le:ri] for le, ri in zip(left, right)]

        # Set up _slice_sim_data method for this class.
        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Slice sim_data on oneD sliceCol, to return relevant
            indexes for slice_point.
            """
            idxs = self.sim_idxs[islice]
            bin_left = self.bins[islice]
            bin_right = self.bins[islice + 1]
            return {
                "idxs": idxs,
                "slice_point": {
                    "sid": islice,
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                },
            }

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def __eq__(self, other_slicer):
        """Evaluate if slicers are equivalent."""
        result = False
        if isinstance(other_slicer, OneDSlicer):
            if self.slice_col_name == other_slicer.slice_col_name:
                # If slicer restored from disk or setup,
                # then 'bins' in slice_points dict.
                # This is preferred method to see if slicers are equal.
                if ("bins" in self.slice_points) & ("bins" in other_slicer.slice_points):
                    result = np.array_equal(other_slicer.slice_points["bins"], self.slice_points["bins"])
                # However, before we 'setup' the slicer with data,
                # the slicers could be equivalent.
                else:
                    if (self.bins is not None) and (other_slicer.bins is not None):
                        result = np.array_equal(self.bins, other_slicer.bins)
                    elif (
                        (self.bin_size is not None)
                        and (self.bin_min is not None) & (self.bin_max is not None)
                        and (other_slicer.bin_size is not None)
                        and (other_slicer.bin_min is not None)
                        and (other_slicer.bin_max is not None)
                    ):
                        if (
                            (self.bin_size == other_slicer.bin_size)
                            and (self.bin_min == other_slicer.bin_min)
                            and (self.bin_max == other_slicer.bin_max)
                        ):
                            result = True
        return result
