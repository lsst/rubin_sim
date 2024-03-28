__all__ = ("HourglassPlot",)

import matplotlib.pyplot as plt
import numpy as np

from .plot_handler import BasePlotter


class HourglassPlot(BasePlotter):
    def __init__(self):
        self.plot_type = "Hourglass"
        self.object_plotter = True
        self.default_plot_dict = {
            "title": None,
            "xlabel": "Night - min(Night)",
            "ylabel": "Hours from local midnight",
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

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        """
        Generate the hourglass plot
        """
        if slicer.slicer_name != "HourglassSlicer":
            raise ValueError("HourglassPlot is for use with hourglass slicers")

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        ax = fig.add_subplot(111)
        pernight = metric_value[0]["pernight"]
        perfilter = metric_value[0]["perfilter"]

        y = (perfilter["mjd"] - perfilter["midnight"]) * 24.0
        dmin = np.floor(np.min(perfilter["mjd"]))
        for i in np.arange(0, perfilter.size, 2):
            ax.plot(
                [perfilter["mjd"][i] - dmin, perfilter["mjd"][i + 1] - dmin],
                [y[i], y[i + 1]],
                self.filter2color[perfilter["filter"][i]],
            )
        for i, key in enumerate(["u", "g", "r", "i", "z", "y"]):
            ax.text(
                1.05,
                0.9 - i * 0.07,
                key,
                color=self.filter2color[key],
                transform=ax.transAxes,
            )
        # ax.plot(pernight['mjd'] - dmin,
        # (pernight['twi6_rise'] - pernight['midnight']) * 24.,
        #        'blue', label=r'6$^\circ$ twilight')
        # ax.plot(pernight['mjd'] - dmin,
        # (pernight['twi6_set'] - pernight['midnight']) * 24.,
        #        'blue')
        ax.plot(
            pernight["mjd"] - dmin,
            (pernight["twi12_rise"] - pernight["midnight"]) * 24.0,
            "yellow",
            label=r"12$^\circ$ twilight",
        )
        ax.plot(
            pernight["mjd"] - dmin,
            (pernight["twi12_set"] - pernight["midnight"]) * 24.0,
            "yellow",
        )
        ax.plot(
            pernight["mjd"] - dmin,
            (pernight["twi18_rise"] - pernight["midnight"]) * 24.0,
            "red",
            label=r"18$^\circ$ twilight",
        )
        ax.plot(
            pernight["mjd"] - dmin,
            (pernight["twi18_set"] - pernight["midnight"]) * 24.0,
            "red",
        )
        ax.plot(
            pernight["mjd"] - dmin,
            pernight["moonPer"] / 100.0 - 7,
            "black",
            label="Moon",
        )
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        ax.set_title(plot_dict["title"])

        # draw things in with lines if we are only plotting one night
        if len(pernight["mjd"]) == 1:
            ax.axhline(
                (pernight["twi6_rise"] - pernight["midnight"]) * 24.0,
                color="blue",
                label=r"6$^\circ$ twilight",
            )
            ax.axhline((pernight["twi6_set"] - pernight["midnight"]) * 24.0, color="blue")
            ax.axhline(
                (pernight["twi12_rise"] - pernight["midnight"]) * 24.0,
                color="yellow",
                label=r"12$^\circ$ twilight",
            )
            ax.axhline((pernight["twi12_set"] - pernight["midnight"]) * 24.0, color="yellow")
            ax.axhline(
                (pernight["twi18_rise"] - pernight["midnight"]) * 24.0,
                color="red",
                label=r"18$^\circ$ twilight",
            )
            ax.axhline((pernight["twi18_set"] - pernight["midnight"]) * 24.0, color="red")

        return fig
