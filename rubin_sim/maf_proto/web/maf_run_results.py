__all__ = ("MafRunResults", "read_stat_table", "read_plot_table", "crop_df", "chop_stat_table")

import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import sqlite3


def read_stat_table(db_file):
    """Read the stats table
    """
    con = sqlite3.connect(db_file)
    result = pd.read_sql("select * from stats;", con)
    con.close()
    return result


def read_plot_table(db_file):
    con = sqlite3.connect(db_file)
    result = pd.read_sql("select * from plots;", con)
    con.close()
    return result


def crop_df(df_in):
    """Crop down a dataframe for display
    """
    # All have the same subset
    if np.size(np.unique(df_in["observations_subset"])) == 1:
        out_df = pd.DataFrame()
        out_df["cols"] = df_in["summary_name"] + " " + df_in["metric: unit"]
        # Put the data subset as the row title
        out_df[np.unique(df_in["observations_subset"])[0]] = df_in["value"]
        out_df.set_index('cols', inplace=True)
        out_df = out_df.transpose()

    else:
        rows = []
        to_bunch = df_in["observations_subset"] + " " + df_in["metric: name"]
        try:

            bunches = np.unique(to_bunch).tolist()
        except:
            import pdb ; pdb.set_trace()

        for bunch in bunches:
            out_df = pd.DataFrame()
            indx = np.where(to_bunch == bunch)[0]
            out_df["cols"] = df_in["summary_name"].iloc[indx]
            out_df[bunch] = df_in["value"].iloc[indx]
            out_df.set_index("cols", inplace=True)
            out_df = out_df.transpose()
            rows.append(out_df)
        out_df = pd.concat(rows)

    return out_df


def chop_stat_table(df_in):
    """Divide up a stat table for each one to be displayed
    """

    # Replace any None values
    keys = ["group", "subgroup", "observations_subset", "metric: name"]
    for key in keys:
        indx = np.where(df_in[key].values == None)[0]
        df_in.loc[indx, key] = ""

    g_p_sg = df_in["group"] + "," + df_in["subgroup"]
    up_p_sg = np.unique(g_p_sg)

    out_frames = []
    for grouping in up_p_sg:
        indx = np.where(g_p_sg == grouping)[0]
        temp_df = df_in.iloc[indx]
        temp_df = crop_df(temp_df)
        # Folks hate it when I do this. But it's so convienent.
        temp_df.title = grouping
        out_frames.append(temp_df)
    return out_frames


class MafRunResults:
    """Read and serve the MAF resultsDb_sqlite.db database for the
    show_maf jinja2 templates.

    Deals with a single MAF run (one output directory, one results_db) only.

    Parameters
    ----------
    out_dir : `str`
        The location of the results database for this run.
    run_name : `str`, optional
        The name of the opsim run.
        If None, simply stays blank on show_maf display pages.
    results_db : `str`, optional
        The path to the sqlite database in `out_dir`.
        If None, uses the default of `resultsDb_sqlite.db`.
    """

    def __init__(self, out_dir, run_name=None, results_db=None):
        self.out_dir = os.path.relpath(out_dir, ".")
        self.run_name = run_name

        # Read in the results database.
        if results_db is None:
            results_db = os.path.join(self.out_dir, "maf_results.db")

        # Get the plot and stats info (many-1 metric match)
        self.stats = read_stat_table(results_db)
        self.plots = read_plot_table(results_db)

        # Pull up the names of the groups and subgroups.
        group_list = np.unique(self.plots["group"].tolist() + self.stats["group"].tolist())

        self.groups = {}
        for group in group_list:
            indx1 = np.where(self.stats["group"] == group)[0]
            indx2 = np.where(self.plots["group"] == group)[0]
            self.groups[group] = list(set(self.stats["subgroup"].values[indx1].tolist() +
                                          self.plots["subgroup"].values[indx2].tolist()))

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

        self.plot_order = ["SkyMap", "Histogram", "PowerSpectrum", "Combo"]

    def get_results_db(self):
        """
        Return the summary results sqlite filename, as long as the
        results data is named `resultsDb_sqlite.db`.
        """
        return os.path.join(self.out_dir, "resultsDb_sqlite.db")


    def metrics_with_plot_type(self, plot_type="SkyMap", metrics=None):
        """
        Return an array of metrics with plot=plot_type
        (optionally also within a metric subset).
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
        plot_match = self.plots[np.isin(self.plots["plot_type"], plot_types)]
        # Convert those potentially matching metricIds to metrics,
        # using the subset info.
        metrics = self.metric_ids_to_metrics(plot_match["metric_id"], metrics)
        return metrics


    def plot_dict(self, plots=None):
        """
        Given an array of plots (for a single metric usually).
        Returns an ordered dict with 'plot_type' for interfacing with
        jinja2 templates.
        plot_dict ==
        {'SkyMap': {'plot_file': [], 'thumb_file', []}, 'Histogram': {}..}

        If no plot of a particular type, the plot_file and thumb_file
        are empty lists.
        Calling with plots=None returns a blank plot_dict.
        """
        plot_dict = OrderedDict()
        # Go through plots in 'plotOrder'.
        if plots is None:
            for p in self.plot_order:
                plot_dict[p] = {}
                plot_dict[p]["plot_file"] = ""
                plot_dict[p]["thumb_file"] = ""
        else:
            plot_types = list(np.unique(plots["plot_type"]))
            for p in self.plot_order:
                if p in plot_types:
                    plot_dict[p] = {}
                    plotmatch = plots[np.where(plots["plot_type"] == p)]
                    plot_dict[p]["plot_file"] = []
                    plot_dict[p]["thumb_file"] = []
                    for pm in plotmatch:
                        plot_dict[p]["plot_file"].append(self.get_plot_file(pm))
                        plot_dict[p]["thumb_file"].append(self.get_thumb_file(pm))
                    plot_types.remove(p)
            # Round up remaining plots.
            for p in plot_types:
                plot_dict[p] = {}
                plotmatch = plots[np.where(plots["plot_type"] == p)]
                plot_dict[p]["plot_file"] = []
                plot_dict[p]["thumb_file"] = []
                for pm in plotmatch:
                    plot_dict[p]["plot_file"].append(self.get_plot_file(pm))
                    plot_dict[p]["thumb_file"].append(self.get_thumb_file(pm))
        return plot_dict

    def get_thumb_file(self, plot):
        """
        Return the thumbnail file name for a given plot.
        """
        thumb_file = os.path.join(self.out_dir, plot["thumb_file"])
        return thumb_file

    def get_plot_file(self, plot):
        """
        Return the filename for a given plot.
        """
        plot_file = os.path.join(self.out_dir, plot["plot_file"])
        return plot_file

    def order_plots(self, sky_plots):
        """
        sky_plots = numpy array of skymap plots.

        Returns an ordered list of plotDicts.

        The goal is to lay out the skymaps in a 3x2 grid on the MultiColor
        page, in ugrizy order.
        If a plot for a filter is missing, add a gap. (i.e. if there is no
        u band plot, keep a blank spot).
        If there are other plots, with multiple filters or no filter
        info, they are added to the end.
        If sky_plots includes multiple plots in the same filter,
        just goes back to displayOrder.
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
            matches = np.array([bool(re.search(pattern, x)) for x in sky_plots["plot_file"]])
            match_sky_plot = sky_plots[matches]
            if len(match_sky_plot) == 1:
                ordered_sky_plots.append(self.plot_dict(match_sky_plot))
            elif len(match_sky_plot) == 0:
                ordered_sky_plots.append(blank_plot_dict)
            else:
                # If we found more than one plot in the same filter,
                # we just go back to displayOrder.
                too_many_plots = True
                break

        if too_many_plots is False:
            # Add on any additional non-filter plots (e.g. joint completeness)
            #  that do NOT match original _*_ pattern.
            pattern = "_[ugrizy]_"
            nonmatches = np.array([bool(re.search(pattern, x)) for x in sky_plots["plot_file"]])
            nonmatch_sky_plots = sky_plots[~nonmatches]
            if len(nonmatch_sky_plots) > 0:
                for sky_plot in nonmatch_sky_plots:
                    ordered_sky_plots.append(self.plot_dict(np.array([sky_plot])))

        elif too_many_plots:
            metrics = self.metrics[np.isin(self.metrics["metric_id"], sky_plots["metric_id"])]
            metrics = self.sort_metrics(metrics, order=["display_order"])
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
        Return a numpy array of the plots with plot_type=plot_type,
        optionally for subset of metrics.
        """
        if metrics is None:
            metrics = self.metrics
        # Match the plots to the metrics required.
        plot_metric_match = self.plots[np.isin(self.plots["metric_id"], metrics["metric_id"])]
        # Match the plot type (which could be a list)
        plot_match = plot_metric_match[np.isin(plot_metric_match["plot_type"], plot_type)]
        return plot_match

    # Set of methods to deal with summary stats.

    def stats_for_metric(self, metric, stat_name=None):
        """
        Return a numpy array of summary statistics which match a
        given metric(s).

        Optionally specify a particular stat_name that you want to match.
        """
        stats = self.stats[np.where(self.stats["metric_id"] == metric["metric_id"])]
        if stat_name is not None:
            stats = stats[np.where(stats["summary_metric"] == stat_name)]
        return stats

    def stat_dict(self, stats):
        """
        Returns an ordered dictionary with statName:statValue
        for an array of stats.

        Note that if you pass 'stats' from multiple metrics with the same
        summary names, they will be overwritten in the resulting dictionary!
        So just use stats from one metric, with unique summary_metric names.
        """
        # Result = dict with key
        # == summary stat name, value = summary stat value.
        sdict = OrderedDict()
        statnames = self.order_stat_names(stats)
        for n in statnames:
            match = stats[np.where(stats["summary_metric"] == n)]
            # We're only going to look at the first value;
            # and this should be a float.
            sdict[n] = match["summary_value"][0]
        return sdict

    def order_stat_names(self, stats):
        """
        Given an array of stats, return a list containing all the unique
        'summary_metric' names in a default ordering
        (identity-count-mean-median-rms..).
        """
        names = list(np.unique(stats["summary_metric"]))
        # Add some default sorting:
        namelist = []
        for nord in self.summary_stat_order:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist

    def summary_table(self, group, subgroup):
        indx = np.where((self.stats["group"].values == group) & (self.stats["subgroup"].values == subgroup))[0]

        # if the observation subset is the same, no need to have that as a column




    def all_stat_names(self, subgroup):
        """
        """
        indx = np.where(self.stats["subgroup"] == subgroup)

        names = self.stats["summary_name"].values[indx].tolist()
        # Add some default sorting.
        namelist = []
        for nord in self.summary_stat_order:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist
