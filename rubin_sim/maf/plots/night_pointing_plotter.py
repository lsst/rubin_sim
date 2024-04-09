__all__ = ("NightPointingPlotter",)

import warnings

import matplotlib.pyplot as plt
import numpy as np

from .plot_handler import BasePlotter


class NightPointingPlotter(BasePlotter):
    def __init__(self, mjd_col="observationStartMJD", alt_col="alt", az_col="az"):
        # Just call it Hourglass so it gets treated the same way
        self.plot_type = "Hourglass"
        self.mjd_col = mjd_col
        self.alt_col = alt_col
        self.az_col = az_col
        self.object_plotter = True
        self.default_plot_dict = {
            "title": None,
            "xlabel": "MJD",
            "ylabels": ["Alt", "Az"],
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
        if fig is not None:
            warnings.warn("Always expect to create a new plot in " "NightPointingPlotter.")
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        mv = metric_value[0]

        u_filters = np.unique(mv["data_slice"]["filter"])
        for filt in u_filters:
            good = np.where(mv["data_slice"]["filter"] == filt)
            ax1.plot(
                mv["data_slice"][self.mjd_col][good],
                mv["data_slice"][self.alt_col][good],
                "o",
                color=self.filter2color[filt],
                markersize=5,
                alpha=0.5,
            )
            ax2.plot(
                mv["data_slice"][self.mjd_col][good],
                mv["data_slice"][self.az_col][good],
                "o",
                color=self.filter2color[filt],
                markersize=5,
                alpha=0.5,
            )

        good = np.where(np.degrees(mv["moon_alts"]) > -10.0)
        ax1.plot(
            mv["mjds"][good],
            np.degrees(mv["moon_alts"][good]),
            "ko",
            markersize=3,
            alpha=0.1,
        )
        ax2.plot(
            mv["mjds"][good],
            np.degrees(mv["moon_azs"][good]),
            "ko",
            markersize=3,
            alpha=0.1,
        )
        ax2.set_xlabel("MJD")
        ax1.set_ylabel("Altitude (deg)")
        ax2.set_ylabel("Azimuth (deg)")

        good = np.where(np.degrees(mv["sun_alts"]) > -20.0)
        ax1.plot(mv["mjds"][good], np.degrees(mv["sun_alts"][good]), "yo", markersize=3)
        ax2.plot(mv["mjds"][good], np.degrees(mv["sun_azs"][good]), "yo", markersize=3)

        ax1.set_ylim([-20.0, 90.0])
        ax2.set_ylim([0.0, 360.0])

        for i, key in enumerate(["u", "g", "r", "i", "z", "y"]):
            ax1.text(
                1.05,
                0.9 - i * 0.07,
                key,
                color=self.filter2color[key],
                transform=ax1.transAxes,
            )

        return fig
