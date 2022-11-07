from builtins import zip
import numpy as np
import matplotlib.pyplot as plt
from rubin_sim.maf.utils import percentile_clipping

from .plot_handler import BasePlotter

__all__ = ["OneDBinnedData"]


class OneDBinnedData(BasePlotter):
    def __init__(self):
        self.plot_type = "BinnedData"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "label": None,
            "xlabel": None,
            "ylabel": None,
            "filled": False,
            "alpha": 0.5,
            "linestyle": "-",
            "linewidth": 1,
            "logScale": False,
            "percentileClip": None,
            "xMin": None,
            "xMax": None,
            "yMin": None,
            "yMax": None,
            "fontsize": None,
            "figsize": None,
            "grid": True,
        }

    def __call__(self, metric_values, slicer, user_plot_dict, fignum=None):
        """
        Plot a set of oneD binned metric data.
        """
        if slicer.slicerName != "OneDSlicer":
            raise ValueError("OneDBinnedData plotter is for use with OneDSlicer")
        if "bins" not in slicer.slicePoints:
            err_message = 'OneDSlicer must contain "bins" in slicePoints metadata.'
            err_message += " SlicePoints only contains keys %s." % (
                slicer.slicePoints.keys()
            )
            raise ValueError(err_message)
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        fig = plt.figure(fignum, figsize=plot_dict["figsize"])
        # Plot the histogrammed data.
        leftedge = slicer.slicePoints["bins"][:-1]
        width = np.diff(slicer.slicePoints["bins"])
        if plot_dict["filled"]:
            plt.bar(
                leftedge,
                metric_values.filled(),
                width,
                label=plot_dict["label"],
                linewidth=0,
                alpha=plot_dict["alpha"],
                log=plot_dict["logScale"],
                color=plot_dict["color"],
            )
        else:
            good = np.where(metric_values.mask == False)
            x = np.ravel(list(zip(leftedge[good], leftedge[good] + width[good])))
            y = np.ravel(list(zip(metric_values[good], metric_values[good])))
            if plot_dict["logScale"]:
                plt.semilogy(
                    x,
                    y,
                    label=plot_dict["label"],
                    color=plot_dict["color"],
                    linestyle=plot_dict["linestyle"],
                    linewidth=plot_dict["linewidth"],
                    alpha=plot_dict["alpha"],
                )
            else:
                plt.plot(
                    x,
                    y,
                    label=plot_dict["label"],
                    color=plot_dict["color"],
                    linestyle=plot_dict["linestyle"],
                    linewidth=plot_dict["linewidth"],
                    alpha=plot_dict["alpha"],
                )
        if "ylabel" in plot_dict:
            plt.ylabel(plot_dict["ylabel"], fontsize=plot_dict["fontsize"])
        if "xlabel" in plot_dict:
            plt.xlabel(plot_dict["xlabel"], fontsize=plot_dict["fontsize"])
        # Set y limits (either from values in args, percentile_clipping or compressed data values).
        if plot_dict["percentileClip"] is not None:
            y_min, y_max = percentile_clipping(
                metric_values.compressed(), percentile=plot_dict["percentileClip"]
            )
            if plot_dict["y_min"] is None:
                plot_dict["y_min"] = y_min
            if plot_dict["y_max"] is None:
                plot_dict["y_max"] = y_max

        if plot_dict["grid"]:
            plt.grid(plot_dict["grid"], alpha=0.3)

        if plot_dict["y_min"] is None and metric_values.filled().min() == 0:
            plot_dict["y_min"] = 0

        # Set y and x limits, if provided.
        if plot_dict["y_min"] is not None:
            plt.ylim(bottom=plot_dict["y_min"])
        if plot_dict["y_max"] is not None:
            plt.ylim(top=plot_dict["y_max"])
        if plot_dict["xMin"] is not None:
            plt.xlim(left=plot_dict["xMin"])
        if plot_dict["xMax"] is not None:
            plt.xlim(right=plot_dict["xMax"])
        plt.title(plot_dict["title"])
        return fig.number
