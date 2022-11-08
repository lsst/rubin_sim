import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from .plot_handler import BasePlotter
from .perceptual_rainbow import make_pr_cmap

__all__ = ["TwoDMap", "VisitPairsHist"]

perceptual_rainbow = make_pr_cmap()


class TwoDMap(BasePlotter):
    def __init__(self):
        self.plot_type = "TwoD"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "label": None,
            "logScale": False,
            "cbarFormat": None,
            "cbarTitle": "Count",
            "cmap": perceptual_rainbow,
            "percentileClip": None,
            "color_min": None,
            "color_max": None,
            "zp": None,
            "normVal": None,
            "cbar_edge": True,
            "nTicks": None,
            "aspect": "auto",
            "xextent": None,
            "origin": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fignum=None):
        """
        Parameters
        ----------
        metric_value : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.BaseSpatialSlicer
            (any spatial slicer)
        user_plot_dict: dict
            Dictionary of plot parameters set by user (overrides default values).
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        if "Healpix" in slicer.slicer_name:
            self.default_plot_dict["ylabel"] = "Healpix ID"
        elif "Opsim" in slicer.slicer_name:
            self.default_plot_dict["ylabel"] = "Field ID"
            self.default_plot_dict["origin"] = "lower"
        elif "User" in slicer.slicer_name:
            self.default_plot_dict["ylabel"] = "User Field ID"

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        # Don't clobber with None
        for key in user_plot_dict:
            if user_plot_dict[key] is not None:
                plot_dict[key] = user_plot_dict[key]

        if plot_dict["xextent"] is None:
            plot_dict["xextent"] = [0, metric_value[0, :].size]

        if plot_dict["logScale"]:
            norm = colors.LogNorm()
        else:
            norm = None

        # Mask out values below the color minimum so they show up as white
        if plot_dict["color_min"] is not None:
            low_vals = np.where(metric_value.data < plot_dict["color_min"])
            metric_value.mask[low_vals] = True

        figure = plt.figure(fignum)
        ax = figure.add_subplot(111)
        yextent = slicer.spatialExtent
        xextent = plot_dict["xextent"]
        extent = []
        extent.extend(xextent)
        extent.extend(yextent)
        image = ax.imshow(
            metric_value,
            vmin=plot_dict["color_min"],
            vmax=plot_dict["color_max"],
            aspect=plot_dict["aspect"],
            cmap=plot_dict["cmap"],
            norm=norm,
            extent=extent,
            interpolation="none",
            origin=plot_dict["origin"],
        )
        cb = plt.colorbar(image)

        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        ax.set_title(plot_dict["title"])
        cb.set_label(plot_dict["cbarTitle"])

        # Fix white space on pdf's
        if plot_dict["cbar_edge"]:
            cb.solids.set_edgecolor("face")
        return figure.number


class VisitPairsHist(BasePlotter):
    """
    Given an opsim2dSlicer, figure out what fraction of observations are in singles, pairs, triples, etc.
    """

    def __init__(self):
        self.plot_type = "TwoD"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": "N visits per night per field",
            "ylabel": "N Visits",
            "label": None,
            "color": "b",
            "xlim": [0, 20],
            "ylim": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fignum=None):
        """
        Parameters
        ----------
        metric_value : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.TwoDSlicer
        user_plot_dict: dict
            Dictionary of plot parameters set by user (overrides default values).
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        # Don't clobber with None
        for key in user_plot_dict:
            if user_plot_dict[key] is not None:
                plot_dict[key] = user_plot_dict[key]

        max_val = metric_value.max()
        bins = np.arange(0.5, max_val + 0.5, 1)

        vals, bins = np.histogram(metric_value, bins)
        xvals = (bins[:-1] + bins[1:]) / 2.0

        figure = plt.figure(fignum)
        ax = figure.add_subplot(111)
        ax.bar(xvals, vals * xvals, color=plot_dict["color"], label=plot_dict["label"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        ax.set_title(plot_dict["title"])
        ax.set_xlim(plot_dict["xlim"])
        ax.set_ylim(plot_dict["ylim"])

        return figure.number
