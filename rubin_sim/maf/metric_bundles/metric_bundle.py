from builtins import zip
from builtins import object
import os
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import warnings

import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.maps as maps
import rubin_sim.maf.plots as plots
from rubin_sim.maf.stackers import ColInfo
import rubin_sim.maf.utils as utils

__all__ = ["MetricBundle", "create_empty_metric_bundle"]


def create_empty_metric_bundle():
    """Create an empty metric bundle.

    Returns
    -------
    MetricBundle
        An empty metric bundle, configured with just the :class:`BaseMetric` and :class:`BaseSlicer`.
    """
    return MetricBundle(metrics.BaseMetric(), slicers.BaseSlicer(), "")


class MetricBundle(object):
    """The MetricBundle is defined by a combination of a (single) metric, slicer and
    constraint - together these define a unique combination of an opsim benchmark.
    An example would be: a CountMetric, a HealpixSlicer, and a constraint 'filter="r"'.

    After the metric is evaluated over the slicePoints of the slicer, the resulting
    metric values are saved in the MetricBundle.

    The MetricBundle also saves the summary metrics to be used to generate summary
    statistics over those metric values, as well as the resulting summary statistic values.

    Plotting parameters and display parameters (for showMaf) are saved in the MetricBundle,
    as well as additional info_label such as the opsim run name, and relevant stackers and maps
    to apply when calculating the metric values.

    Parameters
    ----------
    metric : `~rubin_sim.maf.metric`
        The Metric class to run per slicePoint
    slicer : `~rubin_sim.maf.slicer`
        The Slicer to apply to the incoming visit data (the observations).
    constraint : `str` or None, opt
        A (sql-style) constraint to apply to the visit data, to apply a broad sub-selection.
    stacker_list : `list` of `~rubin_sim.maf.stacker`, opt
        A list of pre-configured stackers to use to generate additional columns per visit.
        These will be generated automatically if needed, but pre-configured versions will override these.
    run_name : `str`, opt
        The name of the simulation being run. This will be added to output files and plots.
        Setting it prevents file conflicts when running the same metric on multiple simulations, and
        provides a way to identify which simulation is being analyzed.
    metadata : `str`, opt
        A deprecated version of info_label (below).
        Values set by metadata will be used for info_label. If both are set, info_label is used.
    info_label : `str` or None, opt
        Information to add to the output metric data file name and plot labels.
        If this is not provided, it will be auto-generated from the constraint (if any).
        Setting this provides an easy way to specify different configurations of a metric, a slicer,
        or just to rewrite your constraint into friendlier terms.
        (i.e. a constraint like 'note not like "%DD%"' can become "non-DD" in the file name and plot labels
        by specifying info_label).
    plot_dict : `dict` of plotting parameters, opt
        Specify general plotting parameters, such as x/y/color limits.
    display_dict : `dict` of display parameters, opt
        Specify parameters for showMaf web pages, such as the side bar labels and figure captions.
        Keys: 'group', 'subgroup', 'caption', and 'order' (such as to set metrics in filter order, etc)
    summary_metrics : `list` of `~rubin_sim.maf.metrics`
        A list of summary metrics to run to summarize the primary metric, such as MedianMetric, etc.
    maps_list : `list` of `~rubin_sim.maf.maps`
        A list of pre-configured maps to use for the metric. This will be auto-generated if specified
        by the metric class, but pre-configured versions will override these.
    """

    colInfo = ColInfo()

    def __init__(
        self,
        metric,
        slicer,
        constraint=None,
        stacker_list=None,
        run_name="run name",
        metadata=None,
        info_label=None,
        plot_dict=None,
        display_dict=None,
        summary_metrics=None,
        maps_list=None,
        file_root=None,
        plot_funcs=None,
    ):
        # Set the metric.
        if not isinstance(metric, metrics.BaseMetric):
            raise ValueError("metric must be an rubin_sim.maf.metrics object")
        self.metric = metric
        # Set the slicer.
        if not isinstance(slicer, slicers.BaseSlicer):
            raise ValueError("slicer must be an rubin_sim.maf.slicers object")
        self.slicer = slicer
        # Set the constraint.
        self.constraint = constraint
        if self.constraint is None:
            self.constraint = ""
        # Set the stacker_list if applicable.
        if stacker_list is not None:
            if isinstance(stacker_list, stackers.BaseStacker):
                self.stacker_list = [
                    stacker_list,
                ]
            else:
                self.stacker_list = []
                for s in stacker_list:
                    if s is None:
                        pass
                    else:
                        if not isinstance(s, stackers.BaseStacker):
                            raise ValueError(
                                "stackerList must only contain rubin_sim.maf.stackers objs"
                            )
                        self.stacker_list.append(s)
        else:
            self.stacker_list = []
        # Set the 'maps' to apply to the slicer, if applicable.
        if maps_list is not None:
            if isinstance(maps_list, maps.BaseMap):
                self.known_cols = [
                    maps_list,
                ]
            else:
                self.maps_list = []
                for m in maps_list:
                    if not isinstance(m, maps.BaseMap):
                        raise ValueError(
                            "mapsList must only contain rubin_sim.maf.maps objects"
                        )
                    self.maps_list.append(m)
        else:
            self.maps_list = []
        # If the metric knows it needs a particular map, add it to the list.
        map_names = [mapName.__class__.__name__ for mapName in self.maps_list]
        if hasattr(self.metric, "maps"):
            for mapName in self.metric.maps:
                if mapName not in map_names:
                    if type(mapName) == str:
                        temp_map = getattr(maps, mapName)()
                        self.maps_list.append(temp_map)
                        map_names.append(mapName)
                    else:
                        self.maps_list.append(mapName)

        # Add the summary stats, if applicable.
        self.set_summary_metrics(summary_metrics)
        # Set the provenance/info_label.
        self._build_metadata(info_label, metadata)
        # Set the run name and build the output filename base (fileRoot).
        self.set_run_name(run_name)
        # Reset fileRoot, if provided.
        if file_root is not None:
            self.file_root = file_root
        # Determine the columns needed from the database.
        self._find_req_cols()
        # Set the plotting classes/functions.
        self.set_plot_funcs(plot_funcs)
        # Set the plot_dict and displayDicts.
        self.plotDict = {}
        self.set_plot_dict(plot_dict)
        # Update/set displayDict.
        self.displayDict = {}
        self.set_display_dict(display_dict)
        # This is where we store the metric values and summary stats.
        self.metricValues = None
        self.summary_values = None

    def _reset_metric_bundle(self):
        """Reset all properties of MetricBundle."""
        self.metric = None
        self.slicer = None
        self.constraint = None
        self.stacker_list = []
        self.summary_metrics = []
        self.plot_funcs = []
        self.maps_list = None
        self.run_name = "run name"
        self.info_label = ""
        self.db_cols = None
        self.file_root = None
        self.plotDict = {}
        self.displayDict = {}
        self.metricValues = None
        self.summary_values = None

    def _setup_metric_values(self):
        """Set up the numpy masked array to store the metric value data."""
        dtype = self.metric.metric_dtype
        # Can't store healpix slicer mask values in an int array.
        if dtype == "int":
            dtype = "float"
        if self.metric.shape == 1:
            shape = self.slicer.shape
        else:
            shape = (self.slicer.shape, self.metric.shape)
        self.metricValues = ma.MaskedArray(
            data=np.empty(shape, dtype),
            mask=np.zeros(shape, "bool"),
            fill_value=self.slicer.badval,
        )
        if hasattr(self.slicer, "mask"):
            self.metricValues.mask = self.slicer.mask

    def _build_metadata(self, info_label, metadata=None):
        """If no info_label is provided, process the constraint
        (by removing extra spaces, quotes, the word 'filter' and equal signs) to make a info_label version.
        e.g. 'filter = "r"' becomes 'r'
        """
        # Pass the deprecated version into info_label if info_label is not set
        if metadata is not None and info_label is None:
            warnings.warn(
                'Use of "metadata" as a kwarg is deprecated, please use info_label instead.'
                f" (copying {metadata} into info_label). "
            )
            info_label = metadata
        if info_label is None:
            self.info_label = (
                self.constraint.replace("=", "").replace("filter", "").replace("'", "")
            )
            self.info_label = self.info_label.replace('"', "").replace("  ", " ")
            self.info_label = self.info_label.strip(" ")
        else:
            self.info_label = info_label

    def _build_file_root(self):
        """
        Build an auto-generated output filename root (i.e. minus the plot type or .npz ending).
        """
        # Build basic version.
        self.file_root = "_".join(
            [
                self.run_name,
                self.metric.name,
                self.info_label,
                self.slicer.slicerName[:4].upper(),
            ]
        )
        # Sanitize output name if needed.
        self.file_root = utils.nameSanitize(self.file_root)

    def _find_req_cols(self):
        """Find the columns needed by the metrics, slicers, and stackers.
        If there are any additional stackers required, instatiate them and add them to
        the self.stackers list.
        (default stackers have to be instantiated to determine what additional columns
        are needed from database).
        """
        # Find all the columns needed by metric and slicer.
        known_cols = self.slicer.columns_needed + list(self.metric.col_name_arr)
        # For the stackers already set up, find their required columns.
        for s in self.stacker_list:
            known_cols += s.colsReq
        known_cols = set(known_cols)
        # Track sources of all of these columns.
        self.db_cols = set()
        newstackers = set()
        for col in known_cols:
            if self.colInfo.getDataSource(col) == self.colInfo.defaultDataSource:
                self.db_cols.add(col)
            else:
                # New default stackers could come from metric/slicer or stackers.
                newstackers.add(self.colInfo.getDataSource(col))
        # Remove already-specified stackers from default list.
        for s in self.stacker_list:
            if s.__class__ in newstackers:
                newstackers.remove(s.__class__)
        # Loop and check if stackers are introducing new columns or stackers.
        while len(newstackers) > 0:
            # Check for the sources of the columns in any of the new stackers.
            new_cols = []
            for s in newstackers:
                newstacker = s()
                new_cols += newstacker.colsReq
                self.stacker_list.append(newstacker)
            new_cols = set(new_cols)
            newstackers = set()
            for col in new_cols:
                if self.colInfo.getDataSource(col) == self.colInfo.defaultDataSource:
                    self.db_cols.add(col)
                else:
                    newstackers.add(self.colInfo.getDataSource(col))
            for s in self.stacker_list:
                if s.__class__ in newstackers:
                    newstackers.remove(s.__class__)
        # A Bit of cleanup.
        # Remove 'metricdata' from dbcols if it ended here by default.
        if "metricdata" in self.db_cols:
            self.db_cols.remove("metricdata")
        if "None" in self.db_cols:
            self.db_cols.remove("None")

    def set_summary_metrics(self, summary_metrics):
        """Set (or reset) the summary metrics for the metricbundle.

        Parameters
        ----------
        summary_metrics : List[BaseMetric]
            Instantiated summary metrics to use to calculate summary statistics for this metric.
        """
        if summary_metrics is not None:
            if isinstance(summary_metrics, metrics.BaseMetric):
                self.summary_metrics = [summary_metrics]
            else:
                self.summary_metrics = []
                for s in summary_metrics:
                    if not isinstance(s, metrics.BaseMetric):
                        raise ValueError(
                            "SummaryStats must only contain rubin_sim.maf.metrics objects"
                        )
                    self.summary_metrics.append(s)
        else:
            # Add identity metric to unislicer metric values (to get them into resultsDB).
            if self.slicer.slicerName == "UniSlicer":
                self.summary_metrics = [metrics.IdentityMetric()]
            else:
                self.summary_metrics = []

    def set_plot_funcs(self, plot_funcs):
        """Set or reset the plotting functions.

        The default is to use all the plotFuncs associated with the slicer, which
        is what happens in self.plot if setPlotFuncs is not used to override self.plotFuncs.

        Parameters
        ----------
        plot_funcs : List[BasePlotter]
            The plotter or plotters to use to generate visuals for this metric.
        """
        if plot_funcs is not None:
            if plot_funcs is isinstance(plot_funcs, plots.BasePlotter):
                self.plot_funcs = [plot_funcs]
            else:
                self.plot_funcs = []
                for pFunc in plot_funcs:
                    if not isinstance(pFunc, plots.BasePlotter):
                        raise ValueError(
                            "plotFuncs should contain instantiated "
                            + "rubin_sim.maf.plotter objects."
                        )
                    self.plot_funcs.append(pFunc)
        else:
            self.plot_funcs = []
            for pFunc in self.slicer.plot_funcs:
                if isinstance(pFunc, plots.BasePlotter):
                    self.plot_funcs.append(pFunc)
                else:
                    self.plot_funcs.append(pFunc())

    def set_plot_dict(self, plot_dict):
        """Set or update any property of plot_dict.

        Parameters
        ----------
        plot_dict : dict
            A dictionary of plotting parameters.
            The usable keywords vary with each rubin_sim.maf.plots Plotter.
        """
        # Don't auto-generate anything here - the plotHandler does it.
        if plot_dict is not None:
            self.plotDict.update(plot_dict)
        # Check for bad zp or normVal values.
        if "zp" in self.plotDict:
            if self.plotDict["zp"] is not None:
                if not np.isfinite(self.plotDict["zp"]):
                    warnings.warn(
                        "Warning! Plot zp for %s was infinite: removing zp from plot_dict"
                        % (self.file_root)
                    )
                    del self.plotDict["zp"]
        if "normVal" in self.plotDict:
            if self.plotDict["normVal"] == 0:
                warnings.warn(
                    "Warning! Plot normalization value for %s was 0: removing normVal from plot_dict"
                    % (self.file_root)
                )
                del self.plotDict["normVal"]

    def set_display_dict(self, display_dict=None, results_db=None):
        """Set or update any property of displayDict.

        Parameters
        ----------
        display_dict : Optional[dict]
            Dictionary of display parameters for showMaf.
            Expected keywords: 'group', 'subgroup', 'order', 'caption'.
            'group', 'subgroup', and 'order' control where the metric results are shown on the showMaf page.
            'caption' provides a caption to use with the metric results.
            These values are saved in the results database.
        results_db : Optional[ResultsDb]
            A MAF results database, used to save the display parameters.
        """
        # Set up a temporary dictionary with the default values.
        tmp_display_dict = {"group": None, "subgroup": None, "order": 0, "caption": None}
        # Update from self.displayDict (to use existing values, if present).
        tmp_display_dict.update(self.displayDict)
        # And then update from any values being passed now.
        if display_dict is not None:
            tmp_display_dict.update(display_dict)
        # Reset self.displayDict to this updated dictionary.
        self.displayDict = tmp_display_dict
        # If we still need to auto-generate a caption, do it.
        if self.displayDict["caption"] is None:
            if self.metric.comment is None:
                caption = self.metric.name + " calculated on a %s basis" % (
                    self.slicer.slicerName
                )
                if self.constraint != "" and self.constraint is not None:
                    caption += " using a subset of data selected via %s." % (
                        self.constraint
                    )
                else:
                    caption += "."
            else:
                caption = self.metric.comment
            if "zp" in self.plotDict:
                caption += " Values plotted with a zeropoint of %.2f." % (
                    self.plotDict["zp"]
                )
            if "normVal" in self.plotDict:
                caption += " Values plotted with a normalization value of %.2f." % (
                    self.plotDict["normVal"]
                )
            self.displayDict["caption"] = caption
        if results_db:
            # Update the display values in the resultsDb.
            metric_id = results_db.update_metric(
                self.metric.name,
                self.slicer.slicerName,
                self.run_name,
                self.constraint,
                self.info_label,
                None,
            )
            results_db.update_display(metric_id, self.displayDict)

    def set_run_name(self, run_name, update_file_root=True):
        """Set (or reset) the runName. FileRoot will be updated accordingly if desired.

        Parameters
        ----------
        run_name: str
            Run Name, which will become part of the fileRoot.
        fileRoot: bool, optional
            Flag to update the fileRoot with the runName. Default True.
        """
        self.run_name = run_name
        if update_file_root:
            self._build_file_root()

    def write_db(self, results_db=None, outfile_suffix=None):
        """Write the metric_values to the database"""
        if outfile_suffix is not None:
            outfile = self.file_root + "_" + outfile_suffix + ".npz"
        else:
            outfile = self.file_root + ".npz"
        if results_db is not None:
            metric_id = results_db.update_metric(
                self.metric.name,
                self.slicer.slicerName,
                self.run_name,
                self.constraint,
                self.info_label,
                outfile,
            )
            results_db.update_display(metric_id, self.displayDict)

    def write(self, comment="", out_dir=".", outfile_suffix=None, results_db=None):
        """Write metric_values (and associated info_label) to disk.

        Parameters
        ----------
        comment : Optional[str]
            Any additional comments to add to the output file
        out_dir : Optional[str]
            The output directory
        outfile_suffix : Optional[str]
            Additional suffix to add to the output files (typically a numerical suffix for movies)
        results_db : Optional[ResultsDb]
            Results database to store information on the file output
        """
        if outfile_suffix is not None:
            outfile = self.file_root + "_" + outfile_suffix + ".npz"
        else:
            outfile = self.file_root + ".npz"
        self.slicer.write_data(
            os.path.join(out_dir, outfile),
            self.metricValues,
            metricName=self.metric.name,
            sim_dataName=self.run_name,
            constraint=self.constraint,
            info_label=self.info_label + comment,
            displayDict=self.displayDict,
            plotDict=self.plotDict,
        )
        if results_db is not None:
            self.write_db(results_db=results_db)

    def output_json(self):
        """Set up and call the baseSlicer outputJSON method, to output to IO string.

        Returns
        -------
        io
           IO object containing JSON data representing the metric bundle data.
        """
        io = self.slicer.output_json(
            self.metricValues,
            metric_name=self.metric.name,
            simDataName=self.run_name,
            info_label=self.info_label,
            plot_dict=self.plotDict,
        )
        return io

    def read(self, filename):
        """Read metric_values and associated info_label from disk.
        Overwrites any data currently in metricbundle.

        Parameters
        ----------
        filename : str
           The file from which to read the metric bundle data.
        """
        if not os.path.isfile(filename):
            raise IOError("%s not found" % filename)

        self._reset_metric_bundle()
        # Set up a base slicer to read data (we don't know type yet).
        baseslicer = slicers.BaseSlicer()
        # Use baseslicer to read file.
        metric_values, slicer, header = baseslicer.read_data(filename)
        self.slicer = slicer
        self.metricValues = metric_values
        self.metricValues.fill_value = slicer.badval
        # It's difficult to reinstantiate the metric object, as we don't
        # know what it is necessarily -- the metric_name can be changed.
        self.metric = metrics.BaseMetric()
        # But, for plot label building, we do need to try to recreate the
        #  metric name and units.
        self.metric.units = ""
        if header is not None:
            self.metric.name = header["metric_name"]
            if "plot_dict" in header:
                if "units" in header["plot_dict"]:
                    self.metric.units = header["plot_dict"]["units"]
            self.run_name = header["simDataName"]
            try:
                self.constraint = header["constraint"]
            except KeyError:
                self.constraint = header["sqlconstraint"]
            # Handle potential old datafile, where 'metadata' may be used instead of info_label
            # use metadata if it's there
            if "metadata" in header:
                self.info_label = header["metadata"]
            # and then use info_label if it's there instead
            if "info_label" in header:
                self.info_label = header["info_label"]
            if "plot_dict" in header:
                self.set_plot_dict(header["plot_dict"])
            if "displayDict" in header:
                self.set_display_dict(header["displayDict"])
        if self.info_label is None:
            self._build_metadata()
        path, head = os.path.split(filename)
        self.file_root = head.replace(".npz", "")
        self.set_plot_funcs(None)

    @classmethod
    def load(cls, filename):
        """Create a metric bundle and load its content from disk.

        Parameters
        ----------
        filename : str
           The file from which to read the metric bundle data.
        """
        metric_bundle = cls(metrics.BaseMetric(), slicers.BaseSlicer(), "")
        metric_bundle.read(filename)
        return metric_bundle

    def compute_summary_stats(self, results_db=None):
        """Compute summary statistics on metric_values, using summaryMetrics (metricbundle list).

        Parameters
        ----------
        results_db : Optional[ResultsDb]
            ResultsDb object to use to store the summary statistic values on disk.
        """
        if self.summary_values is None:
            self.summary_values = {}
        if self.summary_metrics is not None:
            # Build array of metric values, to use for (most) summary statistics.
            rarr_std = np.array(
                list(zip(self.metricValues.compressed())),
                dtype=[("metricdata", self.metricValues.dtype)],
            )
            for m in self.summary_metrics:
                # The summary metric colname should already be set to 'metricdata', but in case it's not:
                m.colname = "metricdata"
                summary_name = m.name.replace(" metricdata", "").replace(" None", "")
                if hasattr(m, "mask_val"):
                    # summary metric requests to use the mask value, as specified by itself,
                    #  rather than skipping masked vals.
                    rarr = np.array(
                        list(zip(self.metricValues.filled(m.mask_val))),
                        dtype=[("metricdata", self.metricValues.dtype)],
                    )
                else:
                    rarr = rarr_std
                if np.size(rarr) == 0:
                    summary_val = self.slicer.badval
                else:
                    summary_val = m.run(rarr)
                self.summary_values[summary_name] = summary_val
                # Add summary metric info to results database, if applicable.
                if results_db:
                    metric_id = results_db.update_metric(
                        self.metric.name,
                        self.slicer.slicerName,
                        self.run_name,
                        self.constraint,
                        self.info_label,
                        None,
                    )
                    results_db.update_summary_stat(
                        metric_id, summary_name=summary_name, summary_value=summary_val
                    )

    def reduce_metric(
        self,
        reduce_func,
        reduce_func_name=None,
        reduce_plot_dict=None,
        reduce_display_dict=None,
    ):
        """Run 'reduceFunc' (any function that operates on self.metric_values).
        Typically reduceFunc will be the metric reduce functions, as they are tailored to expect the
        metric_values format.
        reduceDisplayDict and reducePlotDicts are displayDicts and plotDicts to be
        applied to the new metricBundle.

        Parameters
        ----------
        reduce_func : Func
            Any function that will operate on self.metric_values (typically metric.reduce* function).
        reduce_plot_dict : Optional[dict]
            Plot dictionary for the results of the reduce function.
        reduce_display_dict : Optional[dict]
            Display dictionary for the results of the reduce function.

        Returns
        -------
        MetricBundle
           New metric bundle, inheriting info_label from this metric bundle, but containing the new
           metric values calculated with the 'reduceFunc'.
        """
        # Generate a name for the metric values processed by the reduceFunc.
        if reduce_func_name is not None:
            r_name = reduce_func_name.replace("reduce", "")
        else:
            r_name = reduce_func.__name__.replace("reduce", "")
        reduce_name = self.metric.name + "_" + r_name
        # Set up metricBundle to store new metric values, and add plot_dict/displayDict.
        newmetric = deepcopy(self.metric)
        newmetric.name = reduce_name
        newmetric.metricDtype = "float"
        if reduce_plot_dict is not None:
            if "units" in reduce_plot_dict:
                newmetric.units = reduce_plot_dict["units"]
        newmetric_bundle = MetricBundle(
            metric=newmetric,
            slicer=self.slicer,
            stacker_list=self.stacker_list,
            constraint=self.constraint,
            info_label=self.info_label,
            run_name=self.run_name,
            plot_dict=None,
            plot_funcs=self.plot_funcs,
            display_dict=None,
            summary_metrics=self.summary_metrics,
            maps_list=self.maps_list,
            file_root="",
        )
        # Build a new output file root name.
        newmetric_bundle._build_file_root()
        # Add existing plot_dict (except for title/xlabels etc) into new plot_dict.
        for k, v in self.plotDict.items():
            if k not in newmetric_bundle.plotDict:
                newmetric_bundle.plotDict[k] = v
        # Update newmetric_bundle's plot dictionary with any set explicitly by reducePlotDict.
        newmetric_bundle.set_plot_dict(reduce_plot_dict)
        # Copy the parent metric's display dict into the reduce display dict.
        newmetric_bundle.set_display_dict(self.displayDict)
        # Set the reduce function display 'order' (this is set in the BaseMetric
        # by default, but can be overriden in a metric).
        order = newmetric.reduceOrder[r_name]
        newmetric_bundle.displayDict["order"] = order
        # And then update the newmetric_bundle's display dictionary with any set
        # explicitly by reduceDisplayDict.
        newmetric_bundle.set_display_dict(reduce_display_dict)
        # Set up new metricBundle's metric_values masked arrays, copying metricValue's mask.
        newmetric_bundle.metricValues = ma.MaskedArray(
            data=np.empty(len(self.slicer), "float"),
            mask=self.metricValues.mask.copy(),
            fill_value=self.slicer.badval,
        )
        # Fill the reduced metric data using the reduce function.
        for i, (mVal, mMask) in enumerate(
            zip(self.metricValues.data, self.metricValues.mask)
        ):
            if not mMask:
                val = reduce_func(mVal)
                newmetric_bundle.metricValues.data[i] = val
                if val == newmetric.badval:
                    newmetric_bundle.metricValues.mask[i] = True

        return newmetric_bundle

    def plot(self, plotHandler=None, plotFunc=None, outfileSuffix=None, savefig=False):
        """
        Create all plots available from the slicer. plotHandler holds the output directory info, etc.

        Parameters
        ----------
        plotHandler : Optional[PlotHandler]
           The plotHandler saves the output location and resultsDb connection for a set of plots.
        plotFunc : Optional[BasePlotter]
           Any plotter function. If not specified, the plotters in self.plotFuncs will be used.
        outfileSuffix : Optional[str]
           Optional string to append to the end of the plot output files.
           Useful when creating sequences of images for movies.
        savefig : Optional[bool]
           Flag indicating whether or not to save the figure to disk. Default is False.

        Returns
        -------
        dict
            Dictionary of plotType:figure number key/value pairs, indicating what plots were created
            and what matplotlib figure numbers were used.
        """
        # Generate a plotHandler if none was set.
        if plotHandler is None:
            plotHandler = plots.PlotHandler(savefig=savefig)
        # Make plots.
        if plotFunc is not None:
            if isinstance(plotFunc, plots.BasePlotter):
                plotFunc = plotFunc
            else:
                plotFunc = plotFunc()

        plotHandler.setMetricBundles([self])
        plotHandler.setPlotDicts(plotDicts=[self.plotDict], reset=True)
        madePlots = {}
        if plotFunc is not None:
            fignum = plotHandler.plot(plotFunc, outfileSuffix=outfileSuffix)
            madePlots[plotFunc.plotType] = fignum
        else:
            for plotFunc in self.plot_funcs:
                fignum = plotHandler.plot(plotFunc, outfileSuffix=outfileSuffix)
                madePlots[plotFunc.plotType] = fignum
        return madePlots
