import os
import warnings
import glob
import pandas as pd
from rubin_sim.maf.db import ResultsDb

__all__ = ["RunComparison"]


class RunComparison:
    """
    Class to read multiple results databases, find requested summary metric comparisons,
    and stores results in DataFrames in class.

    This class can operate either as:
    * define a single root directory, automatically (recursively) find all subdirectories that contain
    resultsDbs (in which case, leave run_dirs as None)
    * define the directories in which to search for results_db (no search for further subdirectories, and
    limits the search to the directories listed). In this case, the root directory can be specified
    (and then further directory paths are relative to this root directory) or defined as None, in which case
    the full path names must be specified for each directory).

    The run_names (simulation names) are fetched from the resultsDB directly. This relies on the user
    specifying the simulation name when the metrics are run.

    This class can also pull information from the results_db about where files for the metric data
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
            # Check if each of these specified run directories contain a results_db file
            run_dirs = []
            for r in self.run_dirs:
                if not (os.path.isfile(os.path.join(r, self.default_results_db))):
                    warnings.warn(
                        f"Could not find results_db file {self.default_results_db} in {r}"
                    )
                else:
                    run_dirs.append(r)
            self.run_dirs = run_dirs
        else:
            if self.base_dir is None:
                raise Exception(
                    "Both base_dir and run_dirs cannot be None - please specify "
                    "base_dir to search recursively for results_db files, or "
                    "run_dirs to search specific directories for results_db files."
                )
            # Find subdirectories with results_db files
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
        # Make a look-up table for simulation run_name - runDir.
        # This is really only used in case the user wants to double-check which runs are represented.
        self.run_names = {}
        for rdir in self.run_dirs:
            # Connect to resultsDB
            self.runresults[rdir] = ResultsDb(out_dir=rdir)
            # Get simulation names
            self.run_names[rdir] = self.runresults[rdir].get_run_name()

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
        metric_info_label_like: `str`, optional
            Metric info label like this.
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
                m_ids = self.runresults[r].get_all_metric_ids()
            else:
                m_ids = self.runresults[r].get_metric_id_like(
                    metric_name_like=metric_name_like,
                    metric_info_label_like=metric_info_label_like,
                    slicer_name_like=slicer_name_like,
                )
            for m_id in m_ids:
                info = self.runresults[r].get_metric_info(m_id)
                metric_name = info["metric_name"][0]
                metric_info_label = info["metric_info_label"][0]
                slicer_name = info["slicer_name"][0]
                # Build a hash from the metric Name, metadata, slicer --
                # this will automatically remove duplicates
                hash = ResultsDb.build_summary_name(
                    metric_name, metric_info_label, slicer_name, None
                )
                m_dict[hash] = {
                    "metric_name": metric_name,
                    "metric_info_label": metric_info_label,
                    "slicer_name": slicer_name,
                }
        return m_dict

    def _find_summary_stats(
        self,
        metric_name,
        metric_info_label=None,
        slicer_name=None,
        summary_name=None,
        summary_name_like=None,
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
        metric_info_label : `str`, optional
            The metric metadata specifying the metric desired (optional).
        slicer_name : `str`, optional
            The slicer name specifying the metric desired (optional).
        summary_name : `str`, optional
            The name of the summary statistic desired (optional).
        summary_name_like : `str`, optional
            Wildcard matching name of the summary statistic desired (optional).
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
            # Look for this metric/metadata/slicer/summary stat name combo in this results_db.
            m_id = self.runresults[r].get_metric_id(
                metric_name=metric_name,
                metric_info_label=metric_info_label,
                slicer_name=slicer_name,
            )
            # Note that we may have more than one matching summary metric value per results_db.
            stats = self.runresults[r].get_summary_stats(
                m_id,
                summary_name=summary_name,
                summary_name_like=summary_name_like,
                with_sim_name=True,
            )
            for i in range(len(stats["summary_name"])):
                name = stats["summary_name"][i]
                run_name = stats["run_name"][i]
                if run_name not in summary_values:
                    summary_values[run_name] = {}
                summary_values[run_name][name] = stats["summary_value"][i]
            if len(stats) == 0 and verbose:
                warnings.warn(
                    "Warning: Found no metric results for %s %s %s %s in run %s"
                    % (metric_name, metric_info_label, slicer_name, summary_name, r)
                )
        # Make DataFrame for stat values
        stats = pd.DataFrame(summary_values).T
        return stats

    def add_summary_stats(
        self, metric_dict=None, summary_name_like=None, verbose=False
    ):
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
        summary_name_like : `str`, optional
            Optionally restrict summary stats to names like this.
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
            if "summary_metric" not in metric:
                metric["summary_metric"] = None
            temp_stats = self._find_summary_stats(
                metric_name=metric["metric_name"],
                metric_info_label=metric["metric_info_label"],
                slicer_name=metric["slicer_name"],
                summary_name_like=summary_name_like,
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
        metric_info_label : `str`, optional
            The metric info label specifying the metric desired (optional).
        slicer_name : `str`, optional
            The slicer name specifying the metric desired (optional).

        Returns
        -------
        filepaths: `dict`
            Keys: run_name, Value: path to file
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
                    filename = self.runresults[r].get_metric_datafiles(metricId=m_id)
                    filepaths[r] = os.path.join(r, filename[0])
        return filepaths
