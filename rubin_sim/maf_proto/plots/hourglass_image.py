__all__ = ("PlotHourglassImage",)

import copy
import warnings

import matplotlib.pylab as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import SURVEY_START_MJD

from .plots import BasePlot


def rubin_cm():
    """Handy rubin colormap"""
    filter2color = {
        "u": "purple",
        "g": "blue",
        "r": "green",
        "i": "cyan",
        "z": "orange",
        "y": "red",
    }
    viridis = plt.colormaps.get_cmap("viridis")
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0] = [0, 0, 0, 0]
    for i, bandname in enumerate(filter2color):
        newcolors[i + 1] = mcolors.to_rgba(filter2color[bandname])
    newcolors[7:] = [0, 0, 0, 0]
    newcmp = ListedColormap(newcolors)
    return newcmp


class PlotHourglassImage(BasePlot):
    """Plot the filters used as a function of time.
    Will totally fail in the arctic circle.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        filter_col="filter",
        night_col="night",
        delta_t=60.0,
        info=None,
        mjd_start=None,
        almanac=None,
        pixel_size=30.0,
    ):
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.delta_t = delta_t / 60.0 / 24.0
        self.pixel_size = pixel_size

        self.generated_plot_dict = self._gen_default_labels(info)
        if mjd_start is None:
            self.mjd_start = SURVEY_START_MJD
        else:
            self.mjd_start = mjd_start

        if almanac is None:
            self.almanac = Almanac()
        else:
            self.almanac = almanac

    def _gen_default_labels(self, info):
        """Generate any default label values"""
        result = {
            "title": None,
            "xlabel": "MJD - MJD start (Days)",
            "ylabel": "Hours from local midnight",
        }
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n" + info["observations_subset"]

        return result

    def _filter_to_color(self):
        filter2color = {
            "u": "purple",
            "g": "blue",
            "r": "green",
            "i": "cyan",
            "z": "orange",
            "y": "red",
        }
        return filter2color

    def __call__(
        self,
        data_slice,
        fig=None,
        ax=None,
        filter2color=None,
        title=None,
        xlabel=None,
        ylabel=None,
        figsize=None,
        hours_to_midnight_range=[-7.5, 7.5],
    ):

        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)

        if np.sum(np.isfinite(data_slice["night"])) == 0:
            warnings.warn("No valid data to plot, returning empty figure.")
            return fig

        if filter2color is None:
            filter2color = self._filter_to_color()

        plot_dict = copy.copy(self.generated_plot_dict)
        overrides = {"title": title, "xlabel": xlabel, "ylabel": ylabel}
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        data_slice.sort(order=self.mjd_col)
        unights, uindx = np.unique(data_slice[self.night_col], return_index=True)

        mjds = np.arange(np.min(data_slice[self.mjd_col]), np.max(data_slice[self.mjd_col]) + 2, 0.5)

        # Look up things from almanac
        sunsets = self.almanac.get_sunset_info(mjds)
        moon_info = self.almanac.get_sun_moon_positions(mjds)

        # Approx midnight values
        midnights_mjd = (
            sunsets["sun_n18_setting"] + (sunsets["sun_n18_rising"] - sunsets["sun_n18_setting"]) / 2.0
        )

        # Find time to midnight for each visit
        indx = np.searchsorted(midnights_mjd, data_slice["observationStartMJD"])
        d1 = data_slice["observationStartMJD"] - midnights_mjd[indx]
        d2 = data_slice["observationStartMJD"] - midnights_mjd[indx - 1]
        lower = np.where(np.abs(d1) < np.abs(d2))[0]
        upper = np.where(np.abs(d1) >= np.abs(d2))[0]

        midnight_vals = data_slice["observationStartMJD"] * 0
        midnight_vals[lower] = midnights_mjd[indx[lower]]
        midnight_vals[upper] = midnights_mjd[indx[upper] - 1]

        time_rel_midnight_hrs = (midnight_vals - data_slice["observationStartMJD"]) * 24
        time_rel_midnight_hrs_2 = (
            midnight_vals - data_slice["observationStartMJD"] + data_slice["visitExposureTime"]
        ) * 24

        y_vals = np.arange(hours_to_midnight_range[0], hours_to_midnight_range[1], self.pixel_size / 3600)
        y_indx = np.searchsorted(y_vals, time_rel_midnight_hrs)
        y_indx_2 = np.searchsorted(y_vals, time_rel_midnight_hrs_2)
        x_indx = (data_slice["night"] - data_slice["night"].min()).astype(int)

        image = np.zeros((x_indx.max() + 1, y_vals.size + 1))

        for i, bandname in enumerate(filter2color):
            in_band = np.where(data_slice["band"] == bandname)
            image[x_indx[in_band], y_indx[in_band]] = i + 1
            image[x_indx[in_band], y_indx_2[in_band]] = i + 1

        # Display the imag

        ax.imshow(
            image.T,
            cmap=rubin_cm(),
            vmin=0,
            vmax=256,
            origin="upper",
            aspect="auto",
            extent=[data_slice["night"].min(), data_slice["night"].max(), y_vals.min(), y_vals.max()],
        )

        names = [
            "mjd",
            "midnight",
            "moonPer",
            "twi6_rise",
            "twi6_set",
            "twi12_rise",
            "twi12_set",
            "twi18_rise",
            "twi18_set",
        ]
        types = ["float"] * len(names)
        pernight = np.zeros(len(midnights_mjd), dtype=list(zip(names, types)))
        pernight["midnight"] = midnights_mjd
        pernight["mjd"] = midnights_mjd

        pernight["twi12_rise"] = sunsets["sun_n12_rising"]
        pernight["twi12_set"] = sunsets["sun_n12_setting"]

        pernight["twi18_rise"] = sunsets["sun_n18_rising"]
        pernight["twi18_set"] = sunsets["sun_n18_setting"]

        pernight["moonPer"] = moon_info["moon_phase"]

        for i, key in enumerate(["u", "g", "r", "i", "z", "y"]):
            ax.text(
                1.05,
                0.9 - i * 0.07,
                key,
                color=filter2color[key],
                transform=ax.transAxes,
            )

        ax.plot(
            pernight["mjd"] - self.mjd_start,
            (pernight["twi12_rise"] - pernight["midnight"]) * 24.0,
            "yellow",
            label=r"12$^\circ$ twilight",
        )
        ax.plot(
            pernight["mjd"] - self.mjd_start,
            (pernight["twi12_set"] - pernight["midnight"]) * 24.0,
            "yellow",
        )
        ax.plot(
            pernight["mjd"] - self.mjd_start,
            (pernight["twi18_rise"] - pernight["midnight"]) * 24.0,
            "red",
            label=r"18$^\circ$ twilight",
        )
        ax.plot(
            pernight["mjd"] - self.mjd_start,
            (pernight["twi18_set"] - pernight["midnight"]) * 24.0,
            "red",
        )
        ax.plot(
            pernight["mjd"] - self.mjd_start,
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
