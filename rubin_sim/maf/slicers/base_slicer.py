# Base class for all 'Slicer' objects.
#
import inspect
from io import StringIO
import json
import warnings
import numpy as np
import numpy.ma as ma
from rubin_sim.maf.utils import get_date_version
from six import with_metaclass

__all__ = ["SlicerRegistry", "BaseSlicer"]


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
            raise Exception(
                "Redefining metric %s! (there are >1 slicers with the same name)"
                % (slicername)
            )
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


class BaseSlicer(with_metaclass(SlicerRegistry, object)):
    """
    Base class for all slicers: sets required methods and implements common functionality.

    After first construction, the slicer should be ready for setup_slicer to define slicePoints, which will
    let the slicer 'slice' data and generate plots.
    After init after a restore: everything necessary for using slicer for plotting or
    saving/restoring metric data should be present (although slicer does not need to be able to
    slice data again and generally will not be able to).

    Parameters
    ----------
    verbose: `bool`, optional
        True/False flag to send extra output to screen.
        Default True.
    badval: int or float, optional
        The value the Slicer uses to fill masked metric data values
        Default -666.
    """

    def __init__(self, verbose=True, badval=-666):
        self.verbose = verbose
        self.badval = badval
        # Set cacheSize : each slicer will be able to override if appropriate.
        # Currently only the healpixSlice actually uses the cache: this is set in 'use_cache' flag.
        #  If other slicers have the ability to use the cache, they should add this flag and set the
        #  cacheSize in their __init__ methods.
        self.cacheSize = 0
        # Set length of Slicer.
        self.nslice = None
        self.shape = self.nslice
        self.slicePoints = {}
        self.slicerName = self.__class__.__name__
        self.columns_needed = []
        # Create a dict that saves how to re-init the slicer.
        #  This may not be the whole set of args/kwargs, but those which carry useful metadata or
        #   are absolutely necesary for init.
        # Will often be overwritten by individual slicer slicer_init dictionaries.
        self.slicer_init = {"badval": badval}
        self.plot_funcs = []
        # Note if the slicer needs OpSim field ID info
        self.needsFields = False
        # Set the y-axis range be on the two-d plot
        if self.nslice is not None:
            self.spatialExtent = [0, self.nslice - 1]

    def _run_maps(self, maps):
        """Add map metadata to slicePoints."""
        if maps is not None:
            for m in maps:
                self.slicePoints = m.run(self.slicePoints)

    def setup_slicer(self, sim_data, maps=None):
        """Set up Slicer for data slicing.

        Set up internal parameters necessary for slicer to slice data and generates indexes on sim_data.
        Also sets _slice_sim_data for a particular slicer.

        Parameters
        -----------
        sim_data : np.recarray
            The simulated data to be sliced.
        maps : list of rubin_sim.maf.maps objects, optional.
            Maps to apply at each slicePoint, to add to the slicePoint metadata. Default None.
        """
        # Typically args will be sim_data, but opsimFieldSlicer also uses fieldData.
        raise NotImplementedError()

    def get_slice_points(self):
        """Return the slicePoint metadata, for all slice points."""
        return self.slicePoints

    def __len__(self):
        """Return nslice, the number of slicePoints in the slicer."""
        return self.nslice

    def __iter__(self):
        """Iterate over the slices."""
        self.islice = 0
        return self

    def __next__(self):
        """Returns results of self._slice_sim_data when iterating over slicer.

        Results of self._slice_sim_data should be dictionary of
        {'idxs': the data indexes relevant for this slice of the slicer,
        'slicePoint': the metadata for the slicePoint, which always includes 'sid' key for ID of slicePoint.}
        """
        if self.islice >= self.nslice:
            raise StopIteration
        islice = self.islice
        self.islice += 1
        return self._slice_sim_data(islice)

    def __getitem__(self, islice):
        return self._slice_sim_data(islice)

    def __eq__(self, other_slicer):
        """
        Evaluate if two slicers are equivalent.
        """
        raise NotImplementedError()

    def __ne__(self, other_slicer):
        """
        Evaluate if two slicers are not equivalent.
        """
        if self == other_slicer:
            return False
        else:
            return True

    def _slice_sim_data(self, slicePoint):
        """
        Slice the simulation data appropriately for the slicer.

        Given the identifying slicePoint metadata
        The slice of data returned will be the indices of the numpy rec array (the sim_data)
        which are appropriate for the metric to be working on, for that slicePoint.
        """
        raise NotImplementedError(
            'This method is set up by "setup_slicer" - run that first.'
        )

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
    ):
        """
        Save metric values along with the information required to re-build the slicer.

        Parameters
        -----------
        outfilename : `str`
            The output file name.
        metric_values : `np.ma.MaskedArray` or `np.ndarray`
            The metric values to save to disk.
        """
        header = {}
        header["metric_name"] = metric_name
        header["constraint"] = constraint
        header["info_label"] = info_label
        header["sim_data_name"] = sim_data_name
        date, version_info = get_date_version()
        header["dateRan"] = date
        if display_dict is None:
            display_dict = {"group": "Ungrouped"}
        header["display_dict"] = display_dict
        header["plot_dict"] = plot_dict
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
        # npz file acts like dictionary: each keyword/value pair below acts as a dictionary in loaded NPZ file.
        np.savez(
            outfilename,
            header=header,  # header saved as dictionary
            metricValues=data,  # metric data values
            mask=mask,  # metric mask values
            fill=fill,  # metric badval/fill val
            slicer_init=self.slicer_init,  # dictionary of instantiation parameters
            slicerName=self.slicerName,  # class name
            slicePoints=self.slicePoints,  # slicePoint metadata saved (is a dictionary)
            slicerNSlice=self.nslice,
            slicerShape=self.shape,
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
        Send metric data to JSON streaming API, along with a little bit of metadata.

        This method will only work for metrics where the metricDtype is float or int,
        as JSON will not interpret more complex data properly. These values can't be plotted anyway though.

        Parameters
        -----------
        metric_values : np.ma.MaskedArray or np.ndarray
            The metric values.
        metric_name : str, optional
            The name of the metric. Default ''.
        sim_data_name : str, optional
            The name of the simulated data source. Default ''.
        info_label : str, optional
            Some additional information about this metric and how it was calculated. Default ''.
        plot_dict : dict, optional.
            The plot_dict for this metric bundle. Default None.

        Returns
        --------
        StringIO
            StringIO object containing a header dictionary with metric_name/metadata/sim_data_name/slicerName,
            and plot labels from plot_dict, and metric values/data for plot.
            if oneDSlicer, the data is [ [bin_left_edge, value], [bin_left_edge, value]..].
            if a spatial slicer, the data is [ [lon, lat, value], [lon, lat, value] ..].
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
        header["slicerName"] = self.slicerName
        header["slicerLen"] = int(self.nslice)
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
            if "ra" in self.slicePoints:
                # Spatial slicer. Translate ra/dec to lon/lat in degrees and output with metric value.
                for ra, dec, value, mask in zip(
                    self.slicePoints["ra"],
                    self.slicePoints["dec"],
                    metric_values.data,
                    metric_values.mask,
                ):
                    if not mask:
                        lon = ra * 180.0 / np.pi
                        lat = dec * 180.0 / np.pi
                        metric.append([lon, lat, value])
            elif "bins" in self.slicePoints:
                # OneD slicer. Translate bins into bin/left and output with metric value.
                for i in range(len(metric_values)):
                    binleft = self.slicePoints["bins"][i]
                    value = metric_values.data[i]
                    mask = metric_values.mask[i]
                    if not mask:
                        metric.append([binleft, value])
                    else:
                        metric.append([binleft, 0])
                metric.append([self.slicePoints["bins"][i + 1], 0])
            elif self.slicerName == "UniSlicer":
                metric.append([metric_values[0]])
        # Else:
        else:
            if "ra" in self.slicePoints:
                for ra, dec, value in zip(
                    self.slicePoints["ra"], self.slicePoints["dec"], metric_values
                ):
                    lon = ra * 180.0 / np.pi
                    lat = dec * 180.0 / np.pi
                    metric.append([lon, lat, value])
            elif "bins" in self.slicePoints:
                for i in range(len(metric_values)):
                    binleft = self.slicePoints["bins"][i]
                    value = metric_values[i]
                    metric.append([binleft, value])
                metric.append(self.slicePoints["bins"][i + 1][0])
            elif self.slicerName == "UniSlicer":
                metric.append([metric_values[0]])
        # Write out JSON output.
        io = StringIO()
        json.dump([header, metric], io)
        return io

    def read_data(self, infilename):
        """
        Read metric data from disk, along with the info to rebuild the slicer (minus new slicing capability).

        Parameters
        -----------
        infilename: str
            The filename containing the metric data.

        Returns
        -------
        np.ma.MaskedArray, rubin_sim.maf.slicer, dict
            MetricValues stored in data file, the slicer basis for those metric values, and a dictionary
            containing header information (run_name, metadata, etc.).
        """
        import rubin_sim.maf.slicers as slicers

        # Allowing pickles here is required, because otherwise we cannot restore data saved as objects.
        restored = np.load(infilename, allow_pickle=True)
        # Get metadata and other sim_data info.
        header = restored["header"][()]
        slicer_init = restored["slicer_init"][()]
        slicerName = str(restored["slicerName"])
        slicePoints = restored["slicePoints"][()]
        # Backwards compatibility issue - map 'spatialkey1/spatialkey2' to 'lonCol/latCol'.
        if "spatialkey1" in slicer_init:
            slicer_init["lonCol"] = slicer_init["spatialkey1"]
            del slicer_init["spatialkey1"]
        if "spatialkey2" in slicer_init:
            slicer_init["latCol"] = slicer_init["spatialkey2"]
            del slicer_init["spatialkey2"]
        try:
            slicer = getattr(slicers, slicerName)(**slicer_init)
        except TypeError:
            warnings.warn(
                "Cannot use saved slicer init values; falling back to defaults"
            )
            slicer = getattr(slicers, slicerName)()
        # Restore slicePoint metadata.
        slicer.nslice = restored["slicerNSlice"]
        slicer.slicePoints = slicePoints
        slicer.shape = restored["slicerShape"]
        # Get metric data set
        if restored["mask"][()] is None:
            metricValues = ma.MaskedArray(data=restored["metric_values"])
        else:
            metricValues = ma.MaskedArray(
                data=restored["metric_values"],
                mask=restored["mask"],
                fill_value=restored["fill"],
            )
        return metricValues, slicer, header
