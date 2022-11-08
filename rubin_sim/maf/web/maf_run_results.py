from builtins import zip
from builtins import object
import os
import re
from collections import OrderedDict
import numpy as np
import rubin_sim.maf.db as db
import rubin_sim.maf.metric_bundles as metricBundles

__all__ = ["MafRunResults"]


class MafRunResults(object):
    """
    Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages.

    Deals with a single MAF run (one output directory, one results_db) only.
    """

    def __init__(self, out_dir, run_name=None, results_db=None):
        """
        Instantiate the (individual run) layout visualization class.

        This class provides methods used by our jinja2 templates to help interact
        with the outputs of MAF.
        """
        self.out_dir = os.path.relpath(out_dir, ".")
        self.run_name = run_name
        # Set the config summary filename, if available.
        self.config_summary = os.path.join(self.out_dir, "configSummary.txt")
        if not os.path.isfile(self.config_summary):
            self.config_summary = "Config Summary Not Available"
        # if the config summary existed and we don't know the run_name, find it.
        elif self.run_name is None:
            # Read the config file to get the run_name.
            with open(self.config_summary, "r") as myfile:
                config = myfile.read()
            spot = config.find("RunName")
            # If we found the run_name, use that.
            if spot != -1:
                self.run_name = config[spot:].split("\n")[0][8:]
            # Otherwise, set it to be not available.
            else:
                self.run_name = "RunName not available"

        self.config_details = os.path.join(self.out_dir, "configDetails.txt")
        if not os.path.isfile(self.config_details):
            self.config_details = "Config Details Not Available."

        # Read in the results database.
        if results_db is None:
            results_db = os.path.join(self.out_dir, "resultsDb_sqlite.db")
        database = db.ResultsDb(database=results_db)

        # Get the metric and display info (1-1 match)
        self.metrics = database.getMetricDisplayInfo()
        self.metrics = self.sort_metrics(self.metrics)

        # Get the plot and stats info (many-1 metric match)
        skip_stats = ["Completeness@Time", "Completeness H", "FractionPop "]
        self.stats = database.getSummaryStats(summaryNameNotLike=skip_stats)
        self.plots = database.getPlotFiles()

        # Pull up the names of the groups and subgroups.
        groups = sorted(np.unique(self.metrics["displayGroup"]))
        self.groups = OrderedDict()
        for g in groups:
            group_metrics = self.metrics[np.where(self.metrics["displayGroup"] == g)]
            self.groups[g] = sorted(np.unique(group_metrics["displaySubgroup"]))

        self.summary_stat_order = [
            "Id",
            "Identity",
            "Median",
            "Mean",
            "Rms",
            "RobustRms",
            "N(-3Sigma)",
            "N(+3Sigma)",
            "Count",
            "25th%ile",
            "75th%ile",
            "Min",
            "Max",
        ]
        # Add in the table fraction sorting to summary stat ordering.
        table_fractions = [
            x
            for x in list(np.unique(self.stats["summaryMetric"]))
            if x.startswith("TableFraction")
        ]
        if len(table_fractions) > 0:
            for x in (
                "TableFraction 0 == P",
                "TableFraction 1 == P",
                "TableFraction 1 < P",
            ):
                if x in table_fractions:
                    table_fractions.remove(x)
            table_fractions = sorted(table_fractions)
            self.summary_stat_order.append("TableFraction 0 == P")
            for table_frac in table_fractions:
                self.summary_stat_order.append(table_frac)
            self.summary_stat_order.append("TableFraction 1 == P")
            self.summary_stat_order.append("TableFraction 1 < P")

        self.plot_order = ["SkyMap", "Histogram", "PowerSpectrum", "Combo"]

    # Methods to deal with metricIds

    def convert_select_to_metrics(self, group_list, metric_id_list):
        """
        Convert the lists of values returned by 'select metrics' template page
        into an appropriate dataframe of metrics (in sorted order).
        """
        metric_ids = set()
        for group_subgroup in group_list:
            group = group_subgroup.split("_")[0]
            subgroup = group_subgroup.split("_")[-1].replace("+", " ")
            m_ids = self.metric_ids_in_subgroup(group, subgroup)
            for m_id in m_ids:
                metric_ids.add(m_id)
        for m_id in metric_id_list:
            m_id = int(m_id)
            metric_ids.add(m_id)
        metric_ids = list(metric_ids)
        metrics = self.metric_ids_to_metrics(metric_ids)
        metrics = self.sort_metrics(metrics)
        return metrics

    def get_json(self, metric):
        """
        Return the JSON string containing the data for a particular metric.
        """
        if len(metric) > 1:
            return None
        metric = metric[0]
        filename = metric["metricDataFile"]
        if filename.upper() == "NULL":
            return None
        datafile = os.path.join(self.out_dir, filename)
        # Read data back into a  bundle.
        m_b = metricBundles.create_empty_metric_bundle()
        m_b.read(datafile)
        io = m_b.output_json()
        if io is None:
            return None
        return io.getvalue()

    def get_npz(self, metric):
        """
        Return the npz data.
        """
        if len(metric) > 1:
            return None
        metric = metric[0]
        filename = metric["metricDataFile"]
        if filename.upper() == "NULL":
            return None
        else:
            datafile = os.path.join(self.out_dir, filename)
            return datafile

    def get_results_db(self):
        """
        Return the summary results sqlite filename.

        Note that this assumes the resultsDB is stored in 'resultsDB_sqlite.db'.
        """
        return os.path.join(self.out_dir, "resultsDb_sqlite.db")

    def metric_ids_in_subgroup(self, group, subgroup):
        """
        Return the metric_ids within a given group/subgroup.
        """
        metrics = self.metrics_in_subgroup(group, subgroup)
        metric_ids = list(metrics["metric_id"])
        return metric_ids

    def metric_ids_to_metrics(self, metric_ids, metrics=None):
        """
        Return an ordered numpy array of metrics matching metric_ids.
        """
        if metrics is None:
            metrics = self.metrics
        # this should be faster with pandas (and self.metrics.query('metric_id in @metric_ids'))
        metrics = metrics[np.in1d(metrics["metric_id"], metric_ids)]
        return metrics

    def metrics_to_metric_ids(self, metrics):
        """
        Return a list of the metric Ids corresponding to a subset of metrics.
        """
        return list(metrics["metric_id"])

    # Methods to deal with metrics in numpy recarray.

    def sort_metrics(
        self,
        metrics,
        order=(
            "displayGroup",
            "displaySubgroup",
            "baseMetricNames",
            "slicerName",
            "displayOrder",
            "metricInfoLabel",
        ),
    ):
        """
        Sort the metrics by order specified by 'order'.

        Default is to sort by group, subgroup, metric name, slicer, display order, then info_label.
        Returns sorted numpy array.
        """
        if len(metrics) > 0:
            metrics = np.sort(metrics, order=order)
        return metrics

    def metrics_in_group(self, group, metrics=None, sort=True):
        """
        Given a group, return the metrics belonging to this group, in display order.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics[np.where(metrics["displayGroup"] == group)]
        if sort:
            metrics = self.sort_metrics(metrics)
        return metrics

    def metrics_in_subgroup(self, group, subgroup, metrics=None):
        """
        Given a group and subgroup, return a dataframe of the metrics belonging to these
        group/subgroups, in display order.

        If 'metrics' is provided, then only consider this subset of metrics.
        """
        metrics = self.metrics_in_group(group, metrics, sort=False)
        if len(metrics) > 0:
            metrics = metrics[np.where(metrics["displaySubgroup"] == subgroup)]
            metrics = self.sort_metrics(metrics)
        return metrics

    def metrics_to_subgroups(self, metrics):
        """
        Given an array of metrics, return an ordered dict of their group/subgroups.
        """
        group_list = sorted(np.unique(metrics["displayGroup"]))
        groups = OrderedDict()
        for group in group_list:
            groupmetrics = self.metrics_in_group(group, metrics, sort=False)
            groups[group] = sorted(np.unique(groupmetrics["displaySubgroup"]))
        return groups

    def metrics_with_plot_type(self, plot_type="SkyMap", metrics=None):
        """
        Return an array of metrics with plot=plot_type (optional, metric subset).
        """
        # Allow some variation in plot_type names for backward compatibility,
        #  even if plot_type is  a list.
        if not isinstance(plot_type, list):
            plot_type = [plot_type]
        plot_types = []
        for p_t in plot_type:
            plot_types.append(p_t)
            if p_t.endswith("lot"):
                plot_types.append(p_t[:-4])
            else:
                plot_types.append(p_t.lower() + "Plot")
        if metrics is None:
            metrics = self.metrics
        # Identify the plots with the right plot_type, get their IDs.
        plot_match = self.plots[np.in1d(self.plots["plot_type"], plot_types)]
        # Convert those potentially matching metricIds to metrics, using the subset info.
        metrics = self.metric_ids_to_metrics(plot_match["metric_id"], metrics)
        return metrics

    def unique_metric_names(self, metrics=None, baseonly=True):
        """
        Return a list of the unique metric names, preserving the order of 'metrics'.
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            sort_name = "baseMetricNames"
        else:
            sort_name = "metric_name"
        metric_names = list(np.unique(metrics[sort_name]))
        return metric_names

    def metrics_with_summary_stat(self, summary_stat_name="Identity", metrics=None):
        """
        Return metrics with summary stat matching 'summary_stat_name' (optional, metric subset).
        """
        if metrics is None:
            metrics = self.metrics
        # Identify the potentially matching stats.
        stats = self.stats[np.in1d(self.stats["summaryMetric"], summary_stat_name)]
        # Identify the subset of relevant metrics.
        metrics = self.metric_ids_to_metrics(stats["metric_id"], metrics)
        # Re-sort metrics because at this point, probably want displayOrder + info_label before metric name.
        metrics = self.sort_metrics(
            metrics,
            order=[
                "displayGroup",
                "displaySubgroup",
                "slicerName",
                "displayOrder",
                "metricInfoLabel",
                "baseMetricNames",
            ],
        )
        return metrics

    def metrics_with_stats(self, metrics=None):
        """
        Return metrics that have any summary stat.
        """
        if metrics is None:
            metrics = self.metrics
        # Identify metricIds which are also in stats.
        metrics = metrics[np.in1d(metrics["metric_id"], self.stats["metric_id"])]
        metrics = self.sort_metrics(
            metrics,
            order=[
                "displayGroup",
                "displaySubgroup",
                "slicerName",
                "displayOrder",
                "metricInfoLabel",
                "baseMetricNames",
            ],
        )
        return metrics

    def unique_slicer_names(self, metrics=None):
        """
        For an array of metrics, return the unique slicer names.
        """
        if metrics is None:
            metrics = self.metrics
        return list(np.unique(metrics["slicerName"]))

    def metrics_with_slicer(self, slicer, metrics=None):
        """
        For an array of metrics, return the subset which match a particular 'slicername' value.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics[np.where(metrics["slicerName"] == slicer)]
        return metrics

    def unique_metric_name_and_info_label(self, metrics=None):
        """
        For an array of metrics, return the unique metric names + info_label combo in same order.
        """
        if metrics is None:
            metrics = self.metrics
        metric_info_label = []
        for metric_name, info_label in zip(
            metrics["metric_name"], metrics["metric_info_label"]
        ):
            metricinfo = " ".join([metric_name, info_label])
            if metricinfo not in metric_info_label:
                metric_info_label.append(metricinfo)
        return metric_info_label

    def unique_metric_info_label(self, metrics=None):
        """
        For an array of metrics, return a list of the unique info_label.
        """
        if metrics is None:
            metrics = self.metrics
        return list(np.unique(metrics["metricInfoLabel"]))

    def metrics_with_info_label(self, info_label, metrics=None):
        """
        For an array of metrics, return the subset which match a particular 'info_label' value.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics[np.where(metrics["metricInfoLabel"] == info_label)]
        return metrics

    def metrics_with_metric_name(self, metric_name, metrics=None, baseonly=True):
        """
        Return all metrics which match metric_name (default, only the 'base' metric name).
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            metrics = metrics[np.where(metrics["baseMetricNames"] == metric_name)]
        else:
            metrics = metrics[np.where(metrics["metric_name"] == metric_name)]
        return metrics

    def metric_info(self, metric=None, with_data_link=False, with_slicer_name=True):
        """
        Return a dict with the metric info we want to show on the webpages.

        Currently : MetricName / Slicer/ InfoLabel / datafile (for download)
        Used to build a lot of tables in showMaf.
        """
        metric_info = OrderedDict()
        if metric is None:
            metric_info["MetricName"] = ""
            if with_slicer_name:
                metric_info["Slicer"] = ""
            metric_info["InfoLabel"] = ""
            if with_data_link:
                metric_info["Data"] = []
                metric_info["Data"].append([None, None])
            return metric_info
        # Otherwise, do this for real (not a blank).
        metric_info["MetricName"] = metric["metric_name"]
        if with_slicer_name:
            metric_info["Slicer"] = metric["slicerName"]
        metric_info["InfoLabel"] = metric["metricInfoLabel"]
        if with_data_link:
            metric_info["Data"] = []
            metric_info["Data"].append(metric["metricDataFile"])
            metric_info["Data"].append(
                os.path.join(self.out_dir, metric["metricDataFile"])
            )
        return metric_info

    def caption_for_metric(self, metric):
        """
        Return the caption for a given metric.
        """
        caption = metric["displayCaption"]
        if caption == "NULL":
            return ""
        else:
            return caption

    # Methods for plots.

    def plots_for_metric(self, metric):
        """
        Return a numpy array of the plots which match a given metric.
        """
        return self.plots[np.where(self.plots["metric_id"] == metric["metric_id"])]

    def plot_dict(self, plots=None):
        """
        Given an array of plots (for a single metric usually).
        Returns an ordered dict with 'plot_type' for interfacing with jinja2 templates.
        plot_dict == {'SkyMap': {'plot_file': [], 'thumbFile', []}, 'Histogram': {}..}

        If no plot of a particular type, the plot_file and thumbFile are empty lists.
        Calling with plots=None returns a blank plot_dict.
        """
        plot_dict = OrderedDict()
        # Go through plots in 'plotOrder'.
        if plots is None:
            for p in self.plot_order:
                plot_dict[p] = {}
                plot_dict[p]["plot_file"] = ""
                plot_dict[p]["thumbFile"] = ""
        else:
            plot_types = list(np.unique(plots["plot_type"]))
            for p in self.plot_order:
                if p in plot_types:
                    plot_dict[p] = {}
                    plotmatch = plots[np.where(plots["plot_type"] == p)]
                    plot_dict[p]["plot_file"] = []
                    plot_dict[p]["thumbFile"] = []
                    for pm in plotmatch:
                        plot_dict[p]["plot_file"].append(self.get_plotfile(pm))
                        plot_dict[p]["thumbFile"].append(self.get_thumbfile(pm))
                    plot_types.remove(p)
            # Round up remaining plots.
            for p in plot_types:
                plot_dict[p] = {}
                plotmatch = plots[np.where(plots["plot_type"] == p)]
                plot_dict[p]["plot_file"] = []
                plot_dict[p]["thumbFile"] = []
                for pm in plotmatch:
                    plot_dict[p]["plot_file"].append(self.get_plotfile(pm))
                    plot_dict[p]["thumbFile"].append(self.get_thumbfile(pm))
        return plot_dict

    def get_thumbfile(self, plot):
        """
        Return the thumbnail file name for a given plot.
        """
        thumbfile = os.path.join(self.out_dir, plot["thumbFile"])
        return thumbfile

    def get_plotfile(self, plot):
        """
        Return the filename for a given plot.
        """
        plot_file = os.path.join(self.out_dir, plot["plot_file"])
        return plot_file

    def order_plots(self, sky_plots):
        """
        sky_plots = numpy array of skymap plots.

        Returns an ordered list of plotDicts.

        The goal is to lay out the skymaps in a 3x2 grid on the MultiColor page, in ugrizy order.
        If a plot for a filter is missing, add a gap. (i.e. if there is no u, keep a blank spot).
        If there are other plots, with multiple filters or no filter info, they are added to the end.
        If sky_plots includes multiple plots in the same filter, just goes back to displayOrder.
        """
        ordered_sky_plots = []
        if len(sky_plots) == 0:
            return ordered_sky_plots

        order_list = ["u", "g", "r", "i", "z", "y"]
        blank_plot_dict = self.plot_dict(None)

        # Look for filter names in the plot filenames.
        too_many_plots = False
        for f in order_list:
            pattern = "_" + f + "_"
            matches = np.array(
                [bool(re.search(pattern, x)) for x in sky_plots["plot_file"]]
            )
            match_sky_plot = sky_plots[matches]
            # in pandas: match_sky_plot = sky_plots[sky_plots.plot_file.str.contains(pattern)]
            if len(match_sky_plot) == 1:
                ordered_sky_plots.append(self.plot_dict(match_sky_plot))
            elif len(match_sky_plot) == 0:
                ordered_sky_plots.append(blank_plot_dict)
            else:
                # If we found more than one plot in the same filter, we just go back to displayOrder.
                too_many_plots = True
                break

        if too_many_plots is False:
            # Add on any additional non-filter plots (e.g. joint completeness)
            #  that do NOT match original _*_ pattern.
            pattern = "_[ugrizy]_"
            nonmatches = np.array(
                [bool(re.search(pattern, x)) for x in sky_plots["plot_file"]]
            )
            nonmatch_sky_plots = sky_plots[nonmatches == False]
            if len(nonmatch_sky_plots) > 0:
                for sky_plot in nonmatch_sky_plots:
                    ordered_sky_plots.append(self.plot_dict(np.array([sky_plot])))

        elif too_many_plots:
            metrics = self.metrics[
                np.in1d(self.metrics["metric_id"], sky_plots["metric_id"])
            ]
            metrics = self.sort_metrics(metrics, order=["displayOrder"])
            ordered_sky_plots = []
            for m in metrics:
                sky_plot = sky_plots[np.where(sky_plots["metric_id"] == m["metric_id"])]
                ordered_sky_plots.append(self.plot_dict(sky_plot))

        # Pad out to make sure there are rows of 3
        while len(ordered_sky_plots) % 3 != 0:
            ordered_sky_plots.append(blank_plot_dict)

        return ordered_sky_plots

    def get_sky_maps(self, metrics=None, plot_type="SkyMap"):
        """
        Return a numpy array of the plots with plot_type=plot_type, optionally for subset of metrics.
        """
        if metrics is None:
            metrics = self.metrics
        # Match the plots to the metrics required.
        plot_metric_match = self.plots[
            np.in1d(self.plots["metric_id"], metrics["metric_id"])
        ]
        # Match the plot type (which could be a list)
        plot_match = plot_metric_match[
            np.in1d(plot_metric_match["plot_type"], plot_type)
        ]
        return plot_match

    # Set of methods to deal with summary stats.

    def stats_for_metric(self, metric, stat_name=None):
        """
        Return a numpy array of summary statistics which match a given metric(s).

        Optionally specify a particular stat_name that you want to match.
        """
        stats = self.stats[np.where(self.stats["metric_id"] == metric["metric_id"])]
        if stat_name is not None:
            stats = stats[np.where(stats["summaryMetric"] == stat_name)]
        return stats

    def stat_dict(self, stats):
        """
        Returns an ordered dictionary with statName:statValue for an array of stats.

        Note that if you pass 'stats' from multiple metrics with the same summary names, they
        will be overwritten in the resulting dictionary!
        So just use stats from one metric, with unique summaryMetric names.
        """
        # Result = dict with key == summary stat name, value = summary stat value.
        sdict = OrderedDict()
        statnames = self.order_stat_names(stats)
        for n in statnames:
            match = stats[np.where(stats["summaryMetric"] == n)]
            # We're only going to look at the first value; and this should be a float.
            sdict[n] = match["summaryValue"][0]
        return sdict

    def order_stat_names(self, stats):
        """
        Given an array of stats, return a list containing all the unique 'summaryMetric' names
        in a default ordering (identity-count-mean-median-rms..).
        """
        names = list(np.unique(stats["summaryMetric"]))
        # Add some default sorting:
        namelist = []
        for nord in self.summary_stat_order:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist

    def all_stat_names(self, metrics):
        """
        Given an array of metrics, return a list containing all the unique 'summaryMetric' names
        in a default ordering.
        """
        names = np.unique(
            self.stats["summaryMetric"][
                np.in1d(self.stats["metric_id"], metrics["metric_id"])
            ]
        )
        names = list(names)
        # Add some default sorting.
        namelist = []
        for nord in self.summary_stat_order:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist
