__all__ = ("TwoDSubsetData", "OneDSubsetData")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .perceptual_rainbow import make_pr_cmap
from .plot_handler import BasePlotter

perceptual_rainbow = make_pr_cmap()


class TwoDSubsetData(BasePlotter):
    """
    Plot 2 axes from the slicer.sliceColList, identified by
    plot_dict['xaxis']/['yaxis'], given the metric_values at all
    slicepoints [sums over non-visible axes].
    """

    def __init__(self):
        self.plot_type = "2DBinnedData"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": None,
            "ylable": None,
            "units": None,
            "log_scale": False,
            "clims": None,
            "cmap": perceptual_rainbow,
            "cbar_format": None,
            "figsize": None,
        }

    def __call__(self, metric_values, slicer, user_plot_dict, fig=None):
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
        if slicer.slicer_name != "NDSlicer":
            raise ValueError("TwoDSubsetData plots ndSlicer metric values")
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        if "xaxis" not in plot_dict or "yaxis" not in plot_dict:
            raise ValueError("xaxis and yaxis must be specified in plot_dict")
        xaxis = plot_dict["xaxis"]
        yaxis = plot_dict["yaxis"]

        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        # Reshape the metric data so we can isolate the values to plot
        # (just new view of data, not copy).
        newshape = []
        for b in slicer.bins:
            newshape.append(len(b) - 1)
        newshape.reverse()
        md = metric_values.reshape(newshape)
        # Sum over other dimensions.
        # Note that masked values are not included in sum.
        sumaxes = list(range(slicer.nD))
        sumaxes.remove(xaxis)
        sumaxes.remove(yaxis)
        sumaxes = tuple(sumaxes)
        md = md.sum(sumaxes)
        # Plot the histogrammed data.
        # Plot data.
        x, y = np.meshgrid(slicer.bins[xaxis][:-1], slicer.bins[yaxis][:-1])
        if plot_dict["log_scale"]:
            norm = colors.LogNorm()
        else:
            norm = None
        if plot_dict["clims"] is None:
            im = plt.contourf(x, y, md, 250, norm=norm, extend="both", cmap=plot_dict["cmap"])
        else:
            im = plt.contourf(
                x,
                y,
                md,
                250,
                norm=norm,
                extend="both",
                cmap=plot_dict["cmap"],
                vmin=plot_dict["clims"][0],
                vmax=plot_dict["clims"][1],
            )
        xlabel = plot_dict["xlabel"]
        if xlabel is None:
            xlabel = slicer.sliceColList[xaxis]
        plt.xlabel(xlabel)
        ylabel = plot_dict["ylabel"]
        if ylabel is None:
            ylabel = slicer.sliceColList[yaxis]
        plt.ylabel(ylabel)
        cb = plt.colorbar(
            im,
            aspect=25,
            extend="both",
            orientation="horizontal",
            format=plot_dict["cbar_format"],
        )
        cb.set_label(plot_dict["units"])
        plt.title(plot_dict["title"])
        return fig


class OneDSubsetData(BasePlotter):
    """
    Plot a single axes from the sliceColList, identified by plot_dict['axis'],
    given the metric_values at all slicepoints [sums over non-visible axes].
    """

    def __init__(self):
        self.plot_type = "1DBinnedData"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "label": None,
            "units": None,
            "log_scale": False,
            "histRange": None,
            "filled": False,
            "alpha": 0.5,
            "cmap": perceptual_rainbow,
            "cbar_format": None,
            "figsize": None,
        }

    def __call__(self, metric_values, slicer, user_plot_dict, fig=None):
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
        if slicer.slicer_name != "NDSlicer":
            raise ValueError("TwoDSubsetData plots ndSlicer metric values")

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        if "axis" not in plot_dict:
            raise ValueError("axis for 1-d plot must be specified in plot_dict")

        if fig is None:
            fig = plt.Figure(figsize=plot_dict["figsize"])
        # Reshape the metric data so we can isolate the values to plot
        # (just new view of data, not copy).
        axis = plot_dict["axis"]
        newshape = []
        for b in slicer.bins:
            newshape.append(len(b) - 1)
        newshape.reverse()
        md = metric_values.reshape(newshape)
        # Sum over other dimensions.
        # Note that masked values are not included in sum.
        sumaxes = list(range(slicer.nD))
        sumaxes.remove(axis)
        sumaxes = tuple(sumaxes)
        md = md.sum(sumaxes)
        # Plot the histogrammed data.
        leftedge = slicer.bins[axis][:-1]
        width = np.diff(slicer.bins[axis])
        if plot_dict["filled"]:
            plt.bar(
                leftedge,
                md,
                width,
                label=plot_dict["label"],
                linewidth=0,
                alpha=plot_dict["alpha"],
                log=plot_dict["log_scale"],
            )
        else:
            x = np.ravel(list(zip(leftedge, leftedge + width)))
            y = np.ravel(list(zip(md, md)))
            if plot_dict["log_scale"]:
                plt.semilogy(x, y, label=plot_dict["label"])
            else:
                plt.plot(x, y, label=plot_dict["label"])
        plt.ylabel(plot_dict["ylabel"])
        xlabel = plot_dict["xlabel"]
        if xlabel is None:
            xlabel = slicer.sliceColName[axis]
            if plot_dict["units"] is not None:
                xlabel += " (" + plot_dict["units"] + ")"
        plt.xlabel(xlabel)
        if plot_dict["histRange"] is not None:
            plt.xlim(plot_dict["histRange"])
        plt.title(plot_dict["title"])
        return fig
