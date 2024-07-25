__all__ = ("FOPlot", "SummaryHistogram")


import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import rubin_sim.maf.metrics as metrics

from .plot_handler import BasePlotter


class FOPlot(BasePlotter):
    """
    Special plotter to generate and label fO plots.
    """

    def __init__(self):
        self.plot_type = "FO"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": "Number of visits",
            "ylabel": "Area (1000s of square degrees)",
            "scale": None,
            "asky": 18000.0,
            "n_visits": 825,
            "x_min": 0,
            "x_max": None,
            "y_min": 0,
            "y_max": None,
            "linewidth": 2,
            "reflinewidth": 2,
            "figsize": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        """
        Parameters
        ----------
        metric_value : `numpy.ma.MaskedArray`
            The metric values from the bundle.
        slicer : `rubin_sim.maf.slicers.TwoDSlicer`
            The slicer.
        user_plot_dict: `dict`
            Dictionary of plot parameters set by user
            (overrides default values).
        fig : `matplotlib.figure.Figure`
            Matplotlib figure number to use. Default = None, starts new figure.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
           Figure with the plot.
        """
        if not hasattr(slicer, "nside"):
            raise ValueError("FOPlot to be used with healpix or healpix derived slicers.")

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        if plot_dict["scale"] is None:
            plot_dict["scale"] = hp.nside2pixarea(slicer.nside, degrees=True) / 1000.0

        if fig is None:
            fig, ax = plt.subplots(figsize=plot_dict["figsize"])

        # Expect metric_value to be something like number of visits
        cumulative_area = np.arange(1, metric_value.compressed().size + 1)[::-1] * plot_dict["scale"]
        ax.plot(
            np.sort(metric_value.compressed()),
            cumulative_area,
            "k-",
            linewidth=plot_dict["linewidth"],
            zorder=0,
        )
        # This results in calculating the summary stats in two places ..
        # not the ideal choice but easiest for most uses in this case.
        asky = plot_dict["asky"]
        n_visits = plot_dict["n_visits"]
        rarr = np.array(list(zip(metric_value.compressed())), dtype=[("fO", metric_value.dtype)])
        f_o_area = metrics.FOArea(col="fO", n_visit=n_visits, norm=False, nside=slicer.nside).run(rarr)
        f_o_nv = metrics.FONv(col="fO", asky=asky, norm=False, nside=slicer.nside).run(rarr)

        ax.axvline(x=n_visits, linewidth=plot_dict["reflinewidth"], color="b", linestyle=":")
        ax.axhline(y=asky / 1000.0, linewidth=plot_dict["reflinewidth"], color="r", linestyle=":")
        # Add lines for nvis_median and f_o_area:
        # note if these are -666 (badval), they will 'disappear'
        nvis_median = f_o_nv["value"][np.where(f_o_nv["name"] == "MedianNvis")][0]

        ax.axvline(
            x=nvis_median,
            linewidth=plot_dict["reflinewidth"],
            color="b",
            alpha=0.5,
            linestyle="-",
            label=f"f$_0$ Med. Nvis. (@ {asky/1000 :.0f}K sq deg) = {nvis_median :.0f} visits",
        )
        ax.axhline(
            y=f_o_area / 1000.0,
            linewidth=plot_dict["reflinewidth"],
            color="r",
            alpha=0.5,
            linestyle="-",
            label=f"f$_0$ Area (@ {n_visits :.0f} visits) = {f_o_area/1000 :.01f}K sq deg",
        )
        ax.legend(loc="upper right", fontsize="small", numpoints=1, framealpha=1.0)

        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        ax.set_title(plot_dict["title"])

        x_min = plot_dict["x_min"]
        x_max = plot_dict["x_max"]
        y_min = plot_dict["y_min"]
        y_max = plot_dict["y_max"]
        if (x_min is not None) or (x_max is not None):
            ax.set_xlim([x_min, x_max])
        if (y_min is not None) or (y_max is not None):
            ax.set_ylim([y_min, y_max])
        return fig


class SummaryHistogram(BasePlotter):
    """
    Special plotter to summarize metrics which return a set of values
    at each slice_point, e.g. a histogram the metric result per slicepoint.
    (example: the results of with the rubin_sim.maf.metrics.TgapsMetric).
    Essentially, this collapses the metric value over the sky and
    plots a summarized version (reduced to a single value per point
    according to the plot_dict['metricReduce'] metric).
    """

    def __init__(self):
        self.plot_type = "SummaryHistogram"
        self.object_plotter = True
        self.default_plot_dict = {
            "title": None,
            "xlabel": None,
            "ylabel": "Count",
            "label": None,
            "cumulative": False,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "color": "b",
            "linestyle": "-",
            "histStyle": True,
            "grid": True,
            "metricReduce": metrics.SumMetric(),
            "yscale": None,
            "xscale": None,
            "bins": None,
            "figsize": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        """
        Parameters
        ----------
        metric_value : `numpy.ma.MaskedArray`
            Handles 'object' datatypes for the masked array.
        slicer : `rubin_sim.maf.slicer`
            Any MAF slicer.
        user_plot_dict: `dict`
            Dictionary of plot parameters set by user to override defaults.
            'metricReduce' (a `rubin_sim.maf.metric`) indicates how to
            marginalize the metric values calculated at each point to a
            single series of values over the sky.
            'histStyle' (True/False) indicates whether to plot the
            results as a step histogram (True) or as a series of values (False)
            'bins' (np.ndarray) sets the x values for the resulting plot
            and should generally match the bins used with the metric.
        fig : `matplotlib.figure.Figure`
            Matplotlib figure to use. Default starts a new figure.

        Returns
        -------
        fig: `matplotlib.figure.Figure`
           Matplotlib figure used to create the plot.
        """
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        # Combine the metric values across all slice_points.
        if not isinstance(plot_dict["metricReduce"], metrics.BaseMetric):
            raise ValueError("Expected plot_dict[metricReduce] to be a MAF metric object.")
        # Check that there is data to plot
        if np.size(metric_value.compressed()) == 0:
            raise ValueError(f"Did not find any data to plot in {self.plot_type}.")
        # Get the data type
        dt = metric_value.compressed()[0].dtype
        # Change an array of arrays to a 2-d array of correct dtype
        m_v = np.array(metric_value.compressed().tolist(), dtype=[("metric_value", dt)])
        # Make an array to hold the combined result
        final_hist = np.zeros(m_v.shape[1], dtype=float)
        metric = plot_dict["metricReduce"]
        metric.colname = "metric_value"
        # Loop over each bin and use the selected metric to combine the results
        for i in np.arange(final_hist.size):
            final_hist[i] = metric.run(m_v[:, i])

        if plot_dict["cumulative"]:
            final_hist = final_hist.cumsum()

        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        bins = plot_dict["bins"]
        if plot_dict["histStyle"]:
            leftedge = bins[:-1]
            rightedge = bins[1:]

            x = np.vstack([leftedge, rightedge]).T.flatten()
            y = np.vstack([final_hist, final_hist]).T.flatten()

        else:
            # Could use this to plot things like FFT
            x = bins[:-1]
            y = final_hist
        # Make the plot.
        plt.plot(
            x,
            y,
            linestyle=plot_dict["linestyle"],
            label=plot_dict["label"],
            color=plot_dict["color"],
        )
        # Add labels.
        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        plt.title(plot_dict["title"])
        plt.grid(plot_dict["grid"], alpha=0.3)
        # Set y and x limits, if provided.
        if plot_dict["x_min"] is not None:
            plt.xlim(left=plot_dict["x_min"])
        elif bins[0] == 0:
            plt.xlim(left=0)
        if plot_dict["x_max"] is not None:
            plt.xlim(right=plot_dict["x_max"])
        if plot_dict["y_min"] is not None:
            plt.ylim(bottom=plot_dict["y_min"])
        elif final_hist.min() == 0:
            plot_dict["y_min"] = 0
        if plot_dict["y_max"] is not None:
            plt.ylim(top=plot_dict["y_max"])

        if plot_dict["yscale"] is not None:
            plt.yscale(plot_dict["yscale"])
        if plot_dict["xscale"] is not None:
            plt.xscale(plot_dict["xscale"])

        return fig
