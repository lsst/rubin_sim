# oneDSlicer - slices based on values in one data column in sim_data.

import numpy as np
from functools import wraps
import warnings
from rubin_sim.maf.utils import optimal_bins
from rubin_sim.maf.stackers import ColInfo
from rubin_sim.maf.plots.oned_plotters import OneDBinnedData

from .base_slicer import BaseSlicer

__all__ = ["OneDSlicer"]


class OneDSlicer(BaseSlicer):
    """OneD Slicer allows the 'slicing' of data into bins in a single dimension.

    Parameters
    ----------
    slice_col_name : `str`
        The name of the data column to base slicing on (i.e. 'airmass', etc.)
    slice_col_units : `str`, optional
        Set a name for the units of the sliceCol. Used for plotting labels. Default None.
    bins : np.ndarray, optional
        The data will be sliced into 'bins': this can be defined as an array here. Default None.
    bin_min : `float`, optional
    bin_max : `float`, optional
    binsize : `float`, optional
        If bins is not defined, then bin_min/bin_max/binsize can be chosen to anchor the slice points.
        Default None.
        Priority goes: bins >> bin_min/bin_max/binsize >> data values (if none of the above are chosen).

    The bins act like numpy histogram bins: the last bin value is the end value of the last bin.
    All bins except for the last bin are half-open ([a, b)) while the last bin is ([a, b]).
    """

    def __init__(
        self,
        slice_col_name=None,
        slice_col_units=None,
        bins=None,
        bin_min=None,
        bin_max=None,
        binsize=None,
        verbose=True,
        badval=0,
    ):
        super().__init__(verbose=verbose, badval=badval)
        if slice_col_name is None:
            raise ValueError(
                "slice_col_name cannot be left None - choose a data column to group data by"
            )
        self.slice_col_name = slice_col_name
        self.columns_needed = [slice_col_name]
        self.bins = bins
        # Forget binmin/max/stepsize if bins was set
        if self.bins is not None:
            if bin_min is not None or bin_max is not None or binsize is not None:
                warnings.warning(
                    f"Both bins and one of the bin_min/bin_max/binsize was specified. "
                    f"Using bins ({self.bins} values only."
                )
            self.bin_min = self.bins.min()
            self.binMax = self.bins.max()
            self.binsize = np.diff(self.bins)
            if len(np.unique(self.binsize)) == 1:
                self.binsize = np.unique(self.binsize)
        else:
            self.bin_min = bin_min
            self.binMax = bin_max
            self.binsize = binsize
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
            "bin_max": self.binMax,
            "binsize": self.binsize,
        }
        self.plot_funcs = [
            OneDBinnedData,
        ]

    def setup_slicer(self, sim_data, maps=None):
        """
        Set up bins in slicer.
        This happens AFTER sim_data is defined, thus typically in the MetricBundleGroup.
        This maps data into the bins; it's not a good idea to reuse a OneDSlicer as a result.
        """
        if "bins" in self.slice_points:
            warning_msg = "Warning: this OneDSlicer was already set up once. "
            warning_msg += (
                "Re-setting up a OneDSlicer is unpredictable; at the very least, it "
                "will change the mapping of the simulated data into the data slices, "
                "and may result in poor binsize choices (although these may potentially be ok). "
            )
            warning_msg += (
                "A safer choice is to use a separate OneDSlicer for each MetricBundle."
            )
            warnings.warn(warning_msg)
        sliceCol = sim_data[self.slice_col_name]
        # Set bins from data or specified values, if they were previously defined.
        if self.bins is None:
            # Set bin min/max values (could have been set in __init__)
            if self.bin_min is None:
                self.bin_min = np.nanmin(sliceCol)
            if self.binMax is None:
                self.binMax = np.nanmax(sliceCol)
            # Give warning if bin_min = bin_max, and do something at least slightly reasonable.
            if self.bin_min == self.binMax:
                warnings.warn(
                    "bin_min = bin_max (maybe your data is single-valued?). "
                    "Increasing bin_max by 1 (or 2*binsize, if binsize was set)."
                )
                if self.binsize is not None:
                    self.binMax = self.binMax + 2 * self.binsize
                else:
                    self.binMax = self.binMax + 1
            if self.binsize is None:
                bins = optimal_bins(sliceCol, self.bin_min, self.binMax)
                nbins = np.round(bins)
                self.binsize = (self.binMax - self.bin_min) / float(nbins)
            # Set bins
            self.bins = np.arange(
                self.bin_min, self.binMax + self.binsize / 2.0, self.binsize, "float"
            )
        # Set nbins to be one less than # of bins because last binvalue is RH edge only
        self.nslice = len(self.bins) - 1
        self.shape = self.nslice
        # Set slice_point metadata.
        self.slice_points["sid"] = np.arange(self.nslice)
        self.slice_points["bins"] = self.bins
        # Add metadata from map if needed.
        self._run_maps(maps)
        # Set up data slicing.
        self.sim_idxs = np.argsort(sim_data[self.slice_col_name])
        simFieldsSorted = np.sort(sim_data[self.slice_col_name])
        # "left" values are location where simdata == bin value
        self.left = np.searchsorted(simFieldsSorted, self.bins[:-1], "left")
        self.left = np.concatenate(
            (
                self.left,
                np.array(
                    [
                        len(self.sim_idxs),
                    ]
                ),
            )
        )
        # Set up _slice_sim_data method for this class.
        @wraps(self._slice_sim_data)
        def _slice_sim_data(islice):
            """Slice sim_data on oneD sliceCol, to return relevant indexes for slicepoint."""
            idxs = self.sim_idxs[self.left[islice] : self.left[islice + 1]]
            return {
                "idxs": idxs,
                "slice_point": {"sid": islice, "binLeft": self.bins[islice]},
            }

        setattr(self, "_slice_sim_data", _slice_sim_data)

    def __eq__(self, other_slicer):
        """Evaluate if slicers are equivalent."""
        result = False
        if isinstance(other_slicer, OneDSlicer):
            if self.slice_col_name == other_slicer.slice_col_name:
                # If slicer restored from disk or setup, then 'bins' in slice_points dict.
                # This is preferred method to see if slicers are equal.
                if ("bins" in self.slice_points) & (
                    "bins" in other_slicer.slice_points
                ):
                    result = np.array_equal(
                        other_slicer.slice_points["bins"], self.slice_points["bins"]
                    )
                # However, before we 'setup' the slicer with data, the slicers could be equivalent.
                else:
                    if (self.bins is not None) and (other_slicer.bins is not None):
                        result = np.array_equal(self.bins, other_slicer.bins)
                    elif (
                        (self.binsize is not None)
                        and (self.bin_min is not None) & (self.binMax is not None)
                        and (other_slicer.binsize is not None)
                        and (other_slicer.bin_min is not None)
                        and (other_slicer.binMax is not None)
                    ):
                        if (
                            (self.binsize == other_slicer.binsize)
                            and (self.bin_min == other_slicer.bin_min)
                            and (self.binMax == other_slicer.binMax)
                        ):
                            result = True
        return result
