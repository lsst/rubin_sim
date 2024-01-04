# Base class for all 'Slicer' objects.
#
__all__ = ("SlicerRegistry", "BaseSlicer")

import inspect
import json
import warnings
from io import StringIO

import numpy as np
import numpy.ma as ma

from rubin_sim.maf.utils import get_date_version


class SlicerRegistry(type):
    """
    Meta class for slicers, to build a registry of slicer classes.
    """

    def __init__(cls, name, bases, dict):
        super(SlicerRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, "registry"):
            cls.registry = {}
        modname = inspect.getmodule(cls).__name__ + "."
        if modname.startswith("rubin_sim.maf.slicers"):
            modname = ""
        slicername = modname + name
        if slicername in cls.registry:
            raise Exception("Redefining metric %s! (there are >1 slicers with the same name)" % (slicername))
        if slicername not in ["BaseSlicer", "BaseSpatialSlicer"]:
            cls.registry[slicername] = cls

    def get_class(cls, slicername):
        return cls.registry[slicername]

    def help(cls, doc=False):
        for slicername in sorted(cls.registry):
            if not doc:
                print(slicername)
            if doc:
                print("---- ", slicername, " ----")
                print(inspect.getdoc(cls.registry[slicername]))


class BaseSlicer(metaclass=SlicerRegistry):
    """
    Base class for all slicers: sets required methods and
    implements common functionality.

    After first construction, the slicer should be ready for
    `setup_slicer` which defines slice_points, allowing the slicer to "slice"
    data and generate plots.
    After init after a restore: everything necessary for using slicer for
    plotting or saving/restoring metric data should be present, although
    the slicer does not need to be able to slice data again and generally
    will not be able to do so.

    Parameters
    ----------
    verbose: `bool`, optional
        True/False flag to send extra output to screen.
    badval: `int` or `float`, optional
        The value the Slicer uses to fill masked metric data values
    """

    def __init__(self, verbose=False, badval=-666):
        self.verbose = verbose
        self.badval = badval
        # Set the cache_size.
        # Currently only healpixSlicers (and their derivatives) use the cache.
        # The size of the cache is set directly by those slicers.
        self.cache_size = 0
        # Set length of Slicer. This determines the endpoint for iteration.
        self.nslice = None
        # Set the length of the data (metric) values.
        # This is often but not necessarily the same as nslice.
        self.shape = None
        self.slice_points = {}
        self.slicer_name = self.__class__.__name__
        self.columns_needed = []
        # Create a dict that saves how to re-init the slicer.
        # This may not be the whole set of args/kwargs, may only be
        # those which carry useful metadata or are necesary for init.
        self.slicer_init = {"badval": badval}
        self.plot_funcs = []

    def _run_maps(self, maps):
        """Add map metadata to slice_points."""
        if maps is not None:
            for m in maps:
                self.slice_points = m.run(self.slice_points)

    def setup_slicer(self, sim_data, maps=None):
        """Set up Slicer for data slicing.

        Set up internal parameters necessary for slicer to slice data
        and generates indexes on sim_data.
        Also sets _slice_sim_data for a particular slicer.

        Parameters
        -----------
        sim_data : `np.recarray`
            The simulated data to be sliced.
        maps : `list` of `rubin_sim.maf.maps` objects, optional.
            Maps to apply at each slice_point,
            to add to the slice_point metadata.
        """
        raise NotImplementedError()

    def get_slice_points(self):
        """Return the slice_point metadata, for all slice points."""
        return self.slice_points

    def __len__(self):
        """Return nslice, the number of slice_points in the slicer."""
        return self.nslice

    def __iter__(self):
        """Iterate over the slices."""
        self.islice = 0
        return self

    def __next__(self):
        """Returns results of self._slice_sim_data when iterating over slicer.

        Results of self._slice_sim_data should be dictionary of
        {'idxs': the data indexes relevant for this slice of the slicer,
        'slice_point': the metadata for the slice_point, which always
        includes 'sid' key for ID of slice_point.}
        """
        if self.islice >= self.nslice:
            raise StopIteration
        islice = self.islice
        self.islice += 1
        return self._slice_sim_data(islice)

    def __getitem__(self, islice):
        return self._slice_sim_data(islice)

    def __eq__(self, other_slicer):
        """Evaluate if two slicers are equivalent."""
        raise NotImplementedError()

    def __ne__(self, other_slicer):
        """Evaluate if two slicers are not equivalent."""
        if self == other_slicer:
            return False
        else:
            return True

    def _slice_sim_data(self, slice_point):
        """Slice the simulation data appropriately for the slicer.

        Given the identifying slice_point metadata
        The slice of data returned will be the indices of the
        numpy rec array (the sim_data) which are appropriate for the metric
        to be working on, for that slice_point.
        """
        raise NotImplementedError('This method is set up by "setup_slicer" - run that first.')

    def write_data(
        self,
        outfilename,
        metric_values,
        metric_name="",
        sim_data_name="",
        constraint=None,
        info_label="",
        plot_dict=None,
        display_dict=None,
        summary_values=None,
    ):
        """
        Save metric values along with the information required to
        re-build the slicer.

        Parameters
        -----------
        outfilename : `str`
            The output file name.
        metric_values : `np.ma.MaskedArray` or `np.ndarray`
            The metric values to save to disk.
        metric_name : `str`
            Name of the metric as configured when run
        sim_data_name : `str`
            Name of the simulation metric run on
        constraint : `str`
            Constraint used to subselect data
        info_label : `str`
            Descriptive additional information
        plot_dict : `dict`
            Dictionary of plotting parameters
        display_dict : `dict`
            Dictionary of display parameters, including caption
        summary_values : `dict`
            Dictionary of summary statistics
        """
        header = {}
        header["metric_name"] = metric_name
        header["constraint"] = constraint
        header["info_label"] = info_label
        header["sim_data_name"] = sim_data_name
        date, version_info = get_date_version()
        header["date_ran"] = date
        if display_dict is None:
            display_dict = {"group": "Ungrouped"}
        header["display_dict"] = display_dict
        header["plot_dict"] = plot_dict
        header["summary_values"] = summary_values
        for key in version_info:
            header[key] = version_info[key]
        if hasattr(metric_values, "mask"):  # If it is a masked array
            data = metric_values.data
            mask = metric_values.mask
            fill = metric_values.fill_value
        else:
            data = metric_values
            mask = None
            fill = None
        # npz file acts like dictionary: each keyword/value pair
        # below acts as a dictionary in loaded NPZ file.
        np.savez(
            outfilename,
            # header saved as dictionary
            header=header,
            # metric data values
            metric_values=data,
            # metric mask values
            mask=mask,
            # metric badval/fill val
            fill=fill,
            # dictionary of instantiation parameters
            slicer_init=self.slicer_init,
            # class name
            slicer_name=self.slicer_name,
            # slice_point metadata saved (is a dictionary)
            slice_points=self.slice_points,
            slicer_n_slice=self.nslice,
            slicer_shape=self.shape,
        )

    def output_json(
        self,
        metric_values,
        metric_name="",
        sim_data_name="",
        info_label="",
        plot_dict=None,
    ):
        """
        Send metric data to JSON streaming API,
        along with a little bit of metadata.

        This method will only work for metrics where the
        metricDtype is float or int, as JSON will not interpret more
        complex data properly. These values can't be plotted anyway though.

        Parameters
        -----------
        metric_values : `np.ma.MaskedArray` or `np.ndarray`
            The metric values.
        metric_name : `str`, optional
            The name of the metric.
        sim_data_name : `str`, optional
            The name of the simulated data source.
        info_label : `str`, optional
            Some additional information about this metric
            and how it was calculated.
        plot_dict : `dict`, optional.
            The plot_dict for this metric bundle.

        Returns
        --------
        io : `StringIO`
            StringIO object containing a header dictionary with
            metric_name/metadata/sim_data_name/slicer_name,
            and plot labels from plot_dict, and metric values/data for plot.
            if oneDSlicer,
            the data is [ [bin_left_edge, value], [bin_left_edge, value]..].
            if a spatial slicer,
            the data is [ [lon, lat, value], [lon, lat, value] ..].
        """
        # Bail if this is not a good data type for JSON.
        if not (metric_values.dtype == "float") or (metric_values.dtype == "int"):
            warnings.warn("Cannot generate JSON.")
            io = StringIO()
            json.dump(["Cannot generate JSON for this file."], io)
            return None
        # Else put everything together for JSON output.
        if plot_dict is None:
            plot_dict = {}
            plot_dict["units"] = ""
        # Preserve some of the metadata for the plot.
        header = {}
        header["metric_name"] = metric_name
        header["info_label"] = info_label
        header["sim_data_name"] = sim_data_name
        header["slicer_name"] = self.slicer_name
        header["slicer_len"] = int(self.nslice)
        # Set some default plot labels if appropriate.
        if "title" in plot_dict:
            header["title"] = plot_dict["title"]
        else:
            header["title"] = "%s %s: %s" % (sim_data_name, info_label, metric_name)
        if "xlabel" in plot_dict:
            header["xlabel"] = plot_dict["xlabel"]
        else:
            if hasattr(self, "slice_col_name"):
                header["xlabel"] = "%s (%s)" % (
                    self.slice_col_name,
                    self.slice_col_units,
                )
            else:
                header["xlabel"] = "%s" % metric_name
                if "units" in plot_dict:
                    header["xlabel"] += " (%s)" % (plot_dict["units"])
        if "ylabel" in plot_dict:
            header["ylabel"] = plot_dict["ylabel"]
        else:
            if hasattr(self, "slice_col_name"):
                header["ylabel"] = "%s" % metric_name
                if "units" in plot_dict:
                    header["ylabel"] += " (%s)" % (plot_dict["units"])
            else:
                # If it's not a oneDslicer and no ylabel given, don't need one.
                pass
        # Bundle up slicer and metric info.
        metric = []
        # If metric values is a masked array.
        if hasattr(metric_values, "mask"):
            if "ra" in self.slice_points:
                # Spatial slicer.
                # Translate ra/dec to lon/lat in degrees and
                # output with metric value.
                for ra, dec, value, mask in zip(
                    self.slice_points["ra"],
                    self.slice_points["dec"],
                    metric_values.data,
                    metric_values.mask,
                ):
                    if not mask:
                        lon = ra * 180.0 / np.pi
                        lat = dec * 180.0 / np.pi
                        metric.append([lon, lat, value])
            elif "bins" in self.slice_points:
                # OneD slicer. Translate bins into bin/left and
                # output with metric value.
                for i in range(len(metric_values)):
                    bin_left = self.slice_points["bins"][i]
                    value = metric_values.data[i]
                    mask = metric_values.mask[i]
                    if not mask:
                        metric.append([bin_left, value])
                    else:
                        metric.append([bin_left, 0])
                metric.append([self.slice_points["bins"][i + 1], 0])
            elif self.slicer_name == "UniSlicer":
                metric.append([metric_values[0]])
        # Else:
        else:
            if "ra" in self.slice_points:
                for ra, dec, value in zip(self.slice_points["ra"], self.slice_points["dec"], metric_values):
                    lon = ra * 180.0 / np.pi
                    lat = dec * 180.0 / np.pi
                    metric.append([lon, lat, value])
            elif "bins" in self.slice_points:
                for i in range(len(metric_values)):
                    bin_left = self.slice_points["bins"][i]
                    value = metric_values[i]
                    metric.append([bin_left, value])
                metric.append(self.slice_points["bins"][i + 1][0])
            elif self.slicer_name == "UniSlicer":
                metric.append([metric_values[0]])
        # Write out JSON output.
        io = StringIO()
        json.dump([header, metric], io)
        return io

    def read_data(self, infilename):
        """
        Read metric data from disk, along with the info to
        rebuild the slicer (minus new slicing capability).

        Parameters
        -----------
        infilename: `str`
            The filename containing the metric data.

        Returns
        -------
        metric_values, slicer, header : `np.ma.MaskedArray`,
        `rubin_sim.maf.slicer`, `dict`
            MetricValues stored in data file,
            the slicer basis for those metric values,
            and a dictionary containing header information
            (run_name, metadata, etc.).
        """
        import rubin_sim.maf.slicers as slicers

        # Allowing pickles here is required, because otherwise we cannot
        # restore data saved as objects.
        restored = np.load(infilename, allow_pickle=True)
        if "slicer_name" not in restored:
            metric_values, slicer, header = self.read_backwards_compatible(restored, infilename)
            return metric_values, slicer, header
        # This is the standard behavior and will be the
        # sole behavior at a future release point.
        # Get metadata and other sim_data info.
        header = restored["header"][()]
        if "dateRan" in header:
            header["date_ran"] = header["dateRan"]
        # Get slicer information.
        slicer_init = restored["slicer_init"][()]
        slicer_name = str(restored["slicer_name"])
        slice_points = restored["slice_points"][()]
        slicer_nslice = restored["slicer_n_slice"]
        slicer_shape = restored["slicer_shape"]
        try:
            slicer = getattr(slicers, slicer_name)(**slicer_init)
        except TypeError:
            if self.verbose:
                warnings.warn(
                    f"Cannot use saved slicer init values; falling back to defaults for {infilename}"
                )
            slicer = getattr(slicers, slicer_name)()
        # Restore slice_point information.
        slicer.nslice = slicer_nslice
        slicer.slice_points = slice_points
        slicer.shape = slicer_shape
        # Get metric data set
        if restored["mask"][()] is None:
            metric_values = ma.MaskedArray(data=restored["metric_values"])
        else:
            metric_values = ma.MaskedArray(
                data=restored["metric_values"],
                mask=restored["mask"],
                fill_value=restored["fill"],
            )
        return metric_values, slicer, header

    def read_backwards_compatible(self, restored, infilename):
        """Read pre v1.0 metric files."""
        # Backwards compatibility for pre-v1.0 metric outputs.
        # To be deprecated at a future release.
        warnings.warn(
            "Reading pre-v1.0 metric data. To be deprecated in a future release.",
            FutureWarning,
        )
        import rubin_sim.maf.slicers as slicers

        header = restored["header"][()]
        header["metric_name"] = header["metricName"]
        header["sim_data_name"] = header["simDataName"]
        if "metadata" in header:
            header["info_label"] = header["metadata"]
        if "plotDict" in header:
            header["plot_dict"] = header["plotDict"]
        if "displayDict" in header:
            header["display_dict"] = header["displayDict"]
        if "dateRan" in header:
            header["date_ran"] = header["dateRan"]
        slicer_init = restored["slicer_init"][()]
        slicer_name = str(restored["slicerName"])
        slice_points = restored["slicePoints"][()]
        slicer_nslice = restored["slicerNSlice"]
        slicer_shape = restored["slicerShape"]
        # Slicer init update
        new = ["lat_col", "lon_col", "use_camera"]
        old = ["latCol", "lonCol", "useCamera"]
        for n, o in zip(new, old):
            if o in slicer_init:
                slicer_init[n] = slicer_init[o]
                del slicer_init[o]
        if "Hrange" in slicer_init:
            slicer_init["h_range"] = slicer_init["Hrange"]
            del slicer_init["Hrange"]
        new = ["bin_min", "bin_max", "bin_size", "slice_col_name", "slice_col_units"]
        old = ["binMin", "binMax", "binsize", "sliceColName", "sliceColUnits"]
        for n, o in zip(new, old):
            if o in slicer_init:
                slicer_init[n] = slicer_init[o]
                del slicer_init[o]
        # An earlier backwards compatibility issue -
        # map 'spatialkey1/spatialkey2' to 'lon_col/lat_col'.
        if "spatialkey1" in slicer_init:
            slicer_init["lon_col"] = slicer_init["spatialkey1"]
            del slicer_init["spatialkey1"]
        if "spatialkey2" in slicer_init:
            slicer_init["lat_col"] = slicer_init["spatialkey2"]
            del slicer_init["spatialkey2"]
        try:
            slicer = getattr(slicers, slicer_name)(**slicer_init)
        except TypeError:
            if self.verbose:
                warnings.warn(
                    f"Cannot use saved slicer init values; falling back to defaults for {infilename}"
                )
            slicer = getattr(slicers, slicer_name)()
        # Restore slice_point information.
        slicer.nslice = slicer_nslice
        slicer.slice_points = slice_points
        slicer.shape = slicer_shape
        # Get metric data set
        if restored["mask"][()] is None:
            metric_values = ma.MaskedArray(data=restored["metricValues"])
        else:
            metric_values = ma.MaskedArray(
                data=restored["metricValues"],
                mask=restored["mask"],
                fill_value=restored["fill"],
            )
        return metric_values, slicer, header
