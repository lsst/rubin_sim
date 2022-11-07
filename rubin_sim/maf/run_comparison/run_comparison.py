import os
import warnings
import glob
import numpy as np
import pandas as pd
from rubin_sim.maf.db import ResultsDb
import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.plots as plots

__all__ = ["RunComparison"]


class RunComparison:
    """
    Class to read multiple results databases, find requested summary metric comparisons,
    and stores results in DataFrames in class.

    This class can operate either as:
    * define a single root directory, automatically (recursively) find all subdirectories that contain
    resultsDbs (in which case, leave run_dirs as None)
    * define the directories in which to search for resultsDb (no search for further subdirectories, and
    limits the search to the directories listed). In this case, the root directory can be specified
    (and then further directory paths are relative to this root directory) or defined as None, in which case
    the full path names must be specified for each directory).

    The run_names (simulation names) are fetched from the resultsDB directly. This relies on the user
    specifying the simulation name when the metrics are run.

    This class can also pull information from the resultsDb about where files for the metric data
    are located; this is helpful to re-read data from disk and plot of multiple runs in the same image.

    Parameters
    ----------
    base_dir : `str` or None
        The root directory containing all of the underlying runs and their subdirectories.
        If this is "None", then the full pathnames must be specified in run_dirs. If
        not None, then run_dirs (if specified) is assumed to contain relative pathnames.
    run_dirs : `list` or None
        A list of directories containing MAF outputs and resultsDB_sqlite.db files.
        If this is None, the base_dir is searched (recursively) for directories containing resultsDB files.
        The contents of run_dirs can be relative paths (in which case base_dir must be specified) or
        absolute paths (in which case base_dir must be None).
    default_results_db : `str`, opt
        This should be the expected name for the resultsDB files in each directory.
        Default is resultsDb_sqlite.db, which is also the default for the resultsDB class.
    """

    def __init__(
        self, base_dir=None, run_dirs=None, default_results_db="resultsDb_sqlite.db"
    ):
        self.default_results_db = default_results_db
        self.base_dir = base_dir
        if run_dirs is not None:
            if base_dir is not None:
                self.run_dirs = [os.path.join(self.base_dir, r) for r in run_dirs]
            else:
                self.run_dirs = run_dirs
            # Check if each of these specified run directories contain a resultsDb file
            run_dirs = []
            for r in self.run_dirs:
                if not (os.path.isfile(os.path.join(r, self.default_results_db))):
                    warnings.warn(
                        f"Could not find resultsDb file {self.default_results_db} in {r}"
                    )
                else:
                    run_dirs.append(r)
            self.run_dirs = run_dirs
        else:
            if self.base_dir is None:
                raise Exception(
                    "Both base_dir and run_dirs cannot be None - please specify "
                    "base_dir to search recursively for resultsDb files, or "
                    "run_dirs to search specific directories for resultsDb files."
                )
            # Find subdirectories with resultsDb files
            self.run_dirs = [
                r.replace(f"/{self.default_results_db}", "")
                for r in glob.glob(
                    self.base_dir + "/**/" + self.default_results_db, recursive=True
                )
            ]
        self._connect_to_results()
        # Class attributes to store the stats data:
        self.summary_stats = None  # summary stats
        self.normalized_stats = (
            None  # normalized (to baselineRun) version of the summary stats
        )
        self.baseline_run = None  # name of the baseline run

    def _connect_to_results(self):
        """
        Open access to all the results database files.
        Sets up dictionary of connections.
        """
        # Open access to all results database files in self.run_dirs
        self.runresults = {}
        # Make a look-up table for simulation runName - runDir.
        # This is really only used in case the user wants to double-check which runs are represented.
        self.run_names = {}
        for rdir in self.run_dirs:
            # Connect to resultsDB
            self.runresults[rdir] = ResultsDb(out_dir=rdir)
            # Get simulation names
            self.run_names[rdir] = self.runresults[rdir].getSimDataName()

    def close(self):
        """
        Close all connections to the results database files.
        """
        self.__del__()

    def __del__(self):
        for r in self.runresults:
            self.runresults[r].close()

    def build_metric_dict(
        self, metric_name_like=None, metric_info_label_like=None, slicer_name_like=None
    ):
        """Return a metric dictionary based on finding all metrics which match 'like' the various kwargs.
        Note that metrics may not be present in all runDirs, and may not all have summary statistics.

        Parameters
        ----------
        metric_name_like: `str`, optional
            Metric name like this -- i.e. will look for metrics which match metric_name like "value".
        metricMetadataLike: `str`, optional
            Metric Metadata like this.
        slicer_name_like: `str`, optional
            Slicer name like this.

        Returns
        -------
        m_dict : `dict`
            Dictionary of union of metric bundle information across all directories.
            Key = self-created metric 'name', value = Dict{metric_name, metric_metadata, slicer_name}
        """
        if (
            metric_name_like is None
            and metric_info_label_like is None
            and slicer_name_like is None
        ):
            get_all = True
        else:
            get_all = False
        m_dict = {}

        # Go through each results database and gather up all of the available metric bundles
        for r in self.run_dirs:
            if get_all:
                m_ids = self.runresults[r].getAllMetricIds()
            else:
                m_ids = self.runresults[r].getMetricIdLike(
                    metric_name_like=metric_name_like,
                    metric_info_label_like=metric_info_label_like,
                    slicer_name_like=slicer_name_like,
                )
            for m_id in m_ids:
                info = self.runresults[r].getMetricInfo(m_id)
                metric_name = info["metric_name"][0]
                metric_metadata = info["metricInfoLabel"][0]
                slicer_name = info["slicer_name"][0]
                # Build a hash from the metric Name, metadata, slicer --
                # this will automatically remove duplicates
                hash = ResultsDb.buildSummaryName(
                    metric_name, metric_metadata, slicer_name, None
                )
                m_dict[hash] = {
                    "metric_name": metric_name,
                    "metricInfoLabel": metric_metadata,
                    "slicer_name": slicer_name,
                }
        return m_dict

    def _find_summary_stats(
        self,
        metric_name,
        metric_info_label=None,
        slicer_name=None,
        summary_name=None,
        verbose=False,
    ):
        """
        Look for summary metric values matching metric_name (and optionally metricMetadata, slicer_name
        and summary_name) among the results databases.
        Note that some metrics may not be present in some runDirs.

        Parameters
        ----------
        metric_name : `str`
            The name of the original metric.
        metricMetadata : `str`, optional
            The metric metadata specifying the metric desired (optional).
        slicer_name : `str`, optional
            The slicer name specifying the metric desired (optional).
        summary_name : `str`, optional
            The name of the summary statistic desired (optional).
        verbose : `bool`, optional
            Issue warnings resulting from not finding the summary stat information
            (such as if it was never calculated) will not be issued.   Default False.

        Returns
        -------
        summaryStats: `pd.DataFrame`
            <index>   <metric_name>  (possibly additional metricNames - multiple summary stats or metadata..)
             run_name    value
        """
        summary_values = {}
        for r in self.run_dirs:
            # Look for this metric/metadata/slicer/summary stat name combo in this resultsDb.
            m_id = self.runresults[r].get_metric_id(
                metric_name=metric_name,
                metric_info_label=metric_info_label,
                slicer_name=slicer_name,
            )
            # Note that we may have more than one matching summary metric value per resultsDb.
            stats = self.runresults[r].getSummaryStats(
                m_id, summary_name=summary_name, withSimName=True
            )
            for i in range(len(stats["summary_name"])):
                name = stats["summary_name"][i]
                run_name = stats["simDataName"][i]
                if run_name not in summary_values:
                    summary_values[run_name] = {}
                summary_values[run_name][name] = stats["summaryValue"][i]
            if len(stats) == 0 and verbose:
                warnings.warn(
                    "Warning: Found no metric results for %s %s %s %s in run %s"
                    % (metric_name, metric_info_label, slicer_name, summary_name, r)
                )
        # Make DataFrame for stat values
        stats = pd.DataFrame(summary_values).T
        return stats

    def add_summary_stats(self, metric_dict=None, verbose=False):
        """
        Combine the summary statistics of a set of metrics into a pandas
        dataframe that is indexed by the opsim run name.

        Parameters
        ----------
        metric_dict: `dict`, optional
            A dictionary of metrics with all of the information needed to query
            a results database.  The metric/metadata/slicer/summary values referred to
            by a metric_dict value could be unique but don't have to be.
            If None (default), then fetches all metric results.
            (This can be slow if there are a lot of metrics.)
        verbose : `bool`, optional
            Issue warnings resulting from not finding the summary stat information
            (such as if it was never calculated) will not be issued.   Default False.


        Sets self.summary_stats
        """
        if metric_dict is None:
            metric_dict = self.build_metric_dict()
        for m_name, metric in metric_dict.items():
            # In general this will not be present (if only auto-built metric dictionary)
            # But the summaryMetric could be specified (if only 'Medians' were desired, etc.)
            if "summaryMetric" not in metric:
                metric["summaryMetric"] = None
            temp_stats = self._find_summary_stats(
                metric_name=metric["metric_name"],
                metric_info_label=metric["metricInfoLabel"],
                slicer_name=metric["slicerName"],
                summary_name=metric["summaryMetric"],
                verbose=verbose,
            )
            if self.summary_stats is None:
                self.summary_stats = temp_stats
            else:
                self.summary_stats = self.summary_stats.join(
                    temp_stats, how="outer", lsuffix="_x"
                )

        self.summary_stats.index.name = "run_name"
        self.summary_stats.columns.name = "metric"

    def __call__(self, **kwargs):
        """Convenience method to wrap up returning all summary stats only."""
        self.add_summary_stats(**kwargs)
        return self.summary_stats

    def get_file_names(self, metric_name, metric_info_label=None, slicer_name=None):
        """Find the locations of a given metric in all available directories.

        Parameters
        ----------
        metric_name : `str`
            The name of the original metric.
        metricMetadata : `str`, optional
            The metric metadata specifying the metric desired (optional).
        slicer_name : `str`, optional
            The slicer name specifying the metric desired (optional).

        Returns
        -------
        filepaths: `dict`
            Keys: runName, Value: path to file
        """
        filepaths = {}
        for r in self.run_dirs:
            m_id = self.runresults[r].get_metric_id(
                metric_name=metric_name,
                metric_info_label=metric_info_label,
                slicer_name=slicer_name,
            )
            if len(m_id) > 0:
                if len(m_id) > 1:
                    warnings.warn(
                        "Found more than one metric data file matching "
                        + "metric_name %s metric_info_label %s and slicer_name %s"
                        % (metric_name, metric_info_label, slicer_name)
                        + " Skipping this combination."
                    )
                else:
                    filename = self.runresults[r].getMetricDataFiles(metricId=m_id)
                    filepaths[r] = os.path.join(r, filename[0])
        return filepaths

    # Plot actual metric values (skymaps or histograms or power spectra) (values not stored in class).
    def read_metric_data(self, metric_name, metric_info_label, slicer_name):
        # Get the names of the individual files for all runs.
        # Dictionary, keyed by run name.
        filenames = self.get_file_names(metric_name, metric_info_label, slicer_name)
        mname = ResultsDb.buildSummaryName(
            metric_name, metric_info_label, slicer_name, None
        )
        bundle_dict = {}
        for r in filenames:
            b = mb.create_empty_metric_bundle()
            b.read(filenames[r])
            hash = b.run_name + " " + mname
            bundle_dict[hash] = b
        return bundle_dict, mname

    def plot_metric_data(
        self,
        bundle_dict,
        plot_func,
        user_plot_dict=None,
        layout=None,
        out_dir=None,
        savefig=False,
    ):
        if user_plot_dict is None:
            user_plot_dict = {}

        ph = plots.PlotHandler(out_dir=out_dir, savefig=savefig)
        ph.setMetricBundles(bundle_dict)

        plot_dicts = [{} for b in bundle_dict]
        # Depending on plot_func, overplot or make many subplots.
        if plot_func.plotType == "SkyMap":
            # Note that we can only handle 9 subplots currently due
            # to how subplot identification (with string) is handled.
            if len(bundle_dict) > 9:
                raise ValueError("Please try again with < 9 subplots for skymap.")
            # Many subplots.
            if "color_min" not in user_plot_dict:
                color_min = 100000000
                for b in bundle_dict:
                    if "zp" not in bundle_dict[b].plotDict:
                        tmp = bundle_dict[b].metricValues.compressed().min()
                        color_min = min(tmp, color_min)
                    else:
                        color_min = bundle_dict[b].plotDict["color_min"]
                user_plot_dict["color_min"] = color_min
            if "color_max" not in user_plot_dict:
                color_max = -100000000
                for b in bundle_dict:
                    if "zp" not in bundle_dict[b].plotDict:
                        tmp = bundle_dict[b].metricValues.compressed().max()
                        color_max = max(tmp, color_max)
                    else:
                        color_max = bundle_dict[b].plotDict["color_max"]
                user_plot_dict["color_max"] = color_max
            for i, (pdict, bundle) in enumerate(zip(plot_dicts, bundle_dict.values())):
                # Add user provided dictionary.
                pdict.update(user_plot_dict)
                # Set subplot information.
                if layout is None:
                    ncols = int(np.ceil(np.sqrt(len(bundle_dict))))
                    nrows = int(np.ceil(len(bundle_dict) / float(ncols)))
                else:
                    ncols = layout[0]
                    nrows = layout[1]
                pdict["subplot"] = int(str(nrows) + str(ncols) + str(i + 1))
                pdict["title"] = bundle.run_name
                # For the subplots we do not need the label
                pdict["label"] = ""
                pdict["legendloc"] = None
                if "suptitle" not in user_plot_dict:
                    pdict["suptitle"] = ph._buildTitle()
        elif plot_func.plotType == "Histogram":
            # Put everything on one plot.
            if "x_min" not in user_plot_dict:
                x_min = 100000000
                for b in bundle_dict:
                    if "zp" not in bundle_dict[b].plotDict:
                        tmp = bundle_dict[b].metricValues.compressed().min()
                        x_min = min(tmp, x_min)
                    else:
                        x_min = bundle_dict[b].plotDict["x_min"]
                user_plot_dict["x_min"] = x_min
            if "x_max" not in user_plot_dict:
                x_max = -100000000
                for b in bundle_dict:
                    if "zp" not in bundle_dict[b].plotDict:
                        tmp = bundle_dict[b].metricValues.compressed().max()
                        x_max = max(tmp, x_max)
                    else:
                        x_max = bundle_dict[b].plotDict["x_max"]
                user_plot_dict["x_max"] = x_max
            for i, pdict in enumerate(plot_dicts):
                pdict.update(user_plot_dict)
                # Legend and title will automatically be ok, I think.
        elif plot_func.plotType == "BinnedData":
            # Put everything on one plot.
            if "y_min" not in user_plot_dict:
                y_min = 100000000
                for b in bundle_dict:
                    tmp = bundle_dict[b].metricValues.compressed().min()
                    y_min = min(tmp, y_min)
                user_plot_dict["y_min"] = y_min
            if "y_max" not in user_plot_dict:
                y_max = -100000000
                for b in bundle_dict:
                    tmp = bundle_dict[b].metricValues.compressed().max()
                    y_max = max(tmp, y_max)
                user_plot_dict["y_max"] = y_max
            if "x_min" not in user_plot_dict:
                x_min = 100000000
                for b in bundle_dict:
                    tmp = bundle_dict[b].slicer.slicePoints["bins"].min()
                    x_min = min(tmp, x_min)
                user_plot_dict["x_min"] = x_min
            if "x_max" not in user_plot_dict:
                x_max = -100000000
                for b in bundle_dict:
                    tmp = bundle_dict[b].slicer.slicePoints["bins"].max()
                    x_max = max(tmp, x_max)
                user_plot_dict["x_max"] = x_max
            for i, pdict in enumerate(plot_dicts):
                pdict.update(user_plot_dict)
                # Legend and title will automatically be ok, I think.
        ph.plot(plot_func, plot_dicts=plot_dicts)
