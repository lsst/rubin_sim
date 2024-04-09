__all__ = ("NeoDistancePlotter",)

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .plot_handler import BasePlotter


class NeoDistancePlotter(BasePlotter):
    """
    Special plotter to calculate and plot the maximum distance an H=22 NEO
    could be observable to, in any particular opsim observation.

    Parameters
    ----------
    step : `float`, optional
        Step size to use for radial bins. Default is 0.01 AU.
    eclip_max, eclip_min : `float`, `float`, optional
        Range of ecliptic latitude values to include when creating the plot.
    """

    def __init__(self, step=0.01, eclip_max=10.0, eclip_min=-10.0):
        self.plot_type = "neoxyPlotter"
        self.object_plotter = True

        self.default_plot_dict = {
            "title": None,
            "xlabel": "X (AU)",
            "ylabel": "Y (AU)",
            "x_min": -1.5,
            "x_max": 1.5,
            "y_min": -0.25,
            "y_max": 2.5,
            "units": "Count",
            "figsize": None,
        }
        self.filter2color = {
            "u": "purple",
            "g": "blue",
            "r": "green",
            "i": "cyan",
            "z": "orange",
            "y": "red",
        }
        self.filter_col_name = "filter"
        self.step = step
        self.eclip_max = np.radians(eclip_max)
        self.eclip_min = np.radians(eclip_min)

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
        in_plane = np.where(
            (metric_value[0]["eclipLat"] >= self.eclip_min) & (metric_value[0]["eclipLat"] <= self.eclip_max)
        )

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        if fig is None:
            fig = plt.figure(plot_dict["figsize"])
        ax = fig.add_subplot(111)

        planet_props = {"Earth": 1.0, "Venus": 0.72, "Mars": 1.52, "Mercury": 0.39}

        planets = []
        for prop in planet_props:
            planets.append(Ellipse((0, 0), planet_props[prop] * 2, planet_props[prop] * 2, fill=False))

        for planet in planets:
            ax.add_artist(planet)

        # Let's make a 2-d histogram in polar coords,
        # then convert and display in cartisian

        r_step = self.step
        rvec = np.arange(0, plot_dict["x_max"] + r_step, r_step)
        theta_step = np.radians(3.5)
        thetavec = np.arange(0, 2 * np.pi + theta_step, theta_step) - np.pi

        # array to hold histogram values
        H = np.zeros((thetavec.size, rvec.size), dtype=float)

        rgrid, thetagrid = np.meshgrid(rvec, thetavec)

        xgrid = rgrid * np.cos(thetagrid)
        ygrid = rgrid * np.sin(thetagrid)

        for dist, x, y in zip(
            metric_value[0]["MaxGeoDist"][in_plane],
            metric_value[0]["NEOHelioX"][in_plane],
            metric_value[0]["NEOHelioY"][in_plane],
        ):
            theta = np.arctan2(y - 1.0, x)
            diff = np.abs(thetavec - theta)
            theta_to_use = thetavec[np.where(diff == diff.min())]
            # This is a slow where-clause, should be possible to speed it up
            # using
            # np.searchsorted+clever slicing or hist2d to build up the map.
            good = np.where((thetagrid == theta_to_use) & (rgrid <= dist))
            H[good] += 1

        # Set the under value to white
        my_cmap = copy.copy(plt.cm.get_cmap("jet"))
        my_cmap.set_under("w")
        blah = ax.pcolormesh(xgrid, ygrid + 1, H, cmap=my_cmap, vmin=0.001, shading="auto")
        cb = plt.colorbar(blah, ax=ax)
        cb.set_label(plot_dict["units"])

        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        ax.set_title(plot_dict["title"])
        ax.set_ylim([plot_dict["y_min"], plot_dict["y_max"]])
        ax.set_xlim([plot_dict["x_min"], plot_dict["x_max"]])

        ax.plot([0], [1], marker="o", color="b")
        ax.plot([0], [0], marker="o", color="y")

        return fig
