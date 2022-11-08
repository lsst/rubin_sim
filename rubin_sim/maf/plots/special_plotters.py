from builtins import zip
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import rubin_sim.maf.metrics as metrics

from .plot_handler import BasePlotter

__all__ = ["FOPlot", "SummaryHistogram"]


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
            "Asky": 18000.0,
            "Nvisits": 825,
            "x_min": 0,
            "x_max": None,
            "yMin": 0,
            "yMax": None,
            "linewidth": 2,
            "reflinewidth": 2,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fignum=None):
        """
        Parameters
        ----------
        metric_value : numpy.ma.MaskedArray
            The metric values calculated with the 'Count' metric and a healpix slicer.
        slicer : rubin_sim.maf.slicers.HealpixSlicer
        user_plot_dict: dict
            Dictionary of plot parameters set by user (overrides default values).
            Note that Asky and Nvisits values set here and in the slicer should be consistent,
            for plot labels and summary statistic values to be consistent.
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        if not hasattr(slicer, "nside"):
            raise ValueError(
                "FOPlot to be used with healpix or healpix derived slicers."
            )
        fig = plt.figure(fignum)
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        if plot_dict["scale"] is None:
            plot_dict["scale"] = hp.nside2pixarea(slicer.nside, degrees=True) / 1000.0

        # Expect metric_value to be something like number of visits
        cumulative_area = (
            np.arange(1, metric_value.compressed().size + 1)[::-1] * plot_dict["scale"]
        )
        plt.plot(
            np.sort(metric_value.compressed()),
            cumulative_area,
            "k-",
            linewidth=plot_dict["linewidth"],
            zorder=0,
        )
        # This is breaking the rules and calculating the summary stats in two places.
        # Could just calculate summary stats and pass in labels.
        rarr = np.array(
            list(zip(metric_value.compressed())), dtype=[("fO", metric_value.dtype)]
        )
        f_o_area = metrics.fOArea(
            col="fO", Asky=plot_dict["Asky"], norm=False, nside=slicer.nside
        ).run(rarr)
        f_o_nv = metrics.fONv(
            col="fO", Nvisit=plot_dict["Nvisits"], norm=False, nside=slicer.nside
        ).run(rarr)

        plt.axvline(
            x=plot_dict["Nvisits"], linewidth=plot_dict["reflinewidth"], color="b"
        )
        plt.axhline(
            y=plot_dict["Asky"] / 1000.0, linewidth=plot_dict["reflinewidth"], color="r"
        )
        # Add lines for nvis_median and f_o_area: note if these are -666 (badval),
        # the default x_min/y_min values will just leave them off the edges of the plot.
        nvis_median = f_o_nv["value"][np.where(f_o_nv["name"] == "MedianNvis")]
        # Note that Nvis is the number of visits (it's not an area) - so goes on number axis
        plt.axvline(
            x=nvis_median,
            linewidth=plot_dict["reflinewidth"],
            color="b",
            alpha=0.5,
            linestyle=":",
            label=r"f$_0$ Median Nvisits=%.0f" % nvis_median,
        )
        plt.axhline(
            y=f_o_area / 1000.0,
            linewidth=plot_dict["reflinewidth"],
            color="r",
            alpha=0.5,
            linestyle=":",
            label="f$_0$ Area=%.0f" % f_o_area,
        )
        plt.legend(loc="lower left", fontsize="small", numpoints=1)

        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        plt.title(plot_dict["title"])

        x_min = plot_dict["x_min"]
        x_max = plot_dict["x_max"]
        y_min = plot_dict["y_min"]
        y_max = plot_dict["y_max"]
        if (x_min is not None) or (x_max is not None):
            plt.xlim([x_min, x_max])
        if (y_min is not None) or (y_max is not None):
            plt.ylim([y_min, y_max])
        return fig.number


class SummaryHistogram(BasePlotter):
    """
    Special plotter to summarize metrics which return a set of values at each slicepoint,
    such as if a histogram was calculated at each slicepoint
    (e.g. with the rubin_sim.maf.metrics.TgapsMetric).
    Effectively marginalizes the calculated values over the sky, and plots the a summarized
    version (reduced to a single according to the plot_dict['metricReduce'] metric).
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
            "yMin": None,
            "yMax": None,
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

    def __call__(self, metric_value, slicer, user_plot_dict, fignum=None):
        """
        Parameters
        ----------
        metric_value : numpy.ma.MaskedArray
            Handles 'object' datatypes for the masked array.
        slicer : rubin_sim.maf.slicers
            Any MAF slicer.
        user_plot_dict: dict
            Dictionary of plot parameters set by user (overrides default values).
            'metricReduce' (an rubin_sim.maf.metric) indicates how to marginalize the metric values
            calculated at each point to a single series of values over the sky.
            'histStyle' (True/False) indicates whether to plot the results as a step histogram (True)
            or as a series of values (False)
            'bins' (np.ndarray) sets the x values for the resulting plot and should generally match
            the bins used with the metric.
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        # Combine the metric values across all slice_points.
        if not isinstance(plot_dict["metricReduce"], metrics.BaseMetric):
            raise ValueError(
                "Expected plot_dict[metricReduce] to be a MAF metric object."
            )
        # Check that there is data to plot
        if np.size(metric_value.compressed()) == 0:
            raise ValueError(f"Did not find any data to plot in {self.plot_type}.")
        # Get the data type
        dt = metric_value.compressed()[0].dtype
        # Change an array of arrays (dtype=object) to a 2-d array of correct dtype
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

        fig = plt.figure(fignum, figsize=plot_dict["figsize"])
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
        if plot_dict["yMin"] is not None:
            plt.ylim(bottom=plot_dict["yMin"])
        elif final_hist.min() == 0:
            plot_dict["yMin"] = 0
        if plot_dict["yMax"] is not None:
            plt.ylim(top=plot_dict["yMax"])

        if plot_dict["yscale"] is not None:
            plt.yscale(plot_dict["yscale"])
        if plot_dict["xscale"] is not None:
            plt.xscale(plot_dict["xscale"])

        return fig.number
