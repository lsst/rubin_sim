__all__ = ("PlotHourglass",)

import copy
import warnings

import matplotlib.pylab as plt
import numpy as np
from astroplan import Observer
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time
from rubin_scheduler.utils import SURVEY_START_MJD, Site

from .plots import BasePlot


class PlotHourglass(BasePlot):
    """Plot the filters used as a function of time.
    Must be used with the Hourglass Slicer.
    Will totally fail in the arctic circle.
    """

    def __init__(
        self,
        telescope="LSST",
        mjd_col="observationStartMJD",
        filter_col="filter",
        night_col="night",
        delta_t=60.0,
        info=None,
        mjd_start=None,
    ):
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.night_col = night_col
        self.telescope = Site(name=telescope)
        self.delta_t = delta_t / 60.0 / 24.0
        self.location = EarthLocation(
            lat=self.telescope.latitude_rad * u.rad,
            lon=self.telescope.longitude_rad * u.rad,
            height=self.telescope.height * u.m,
        )
        self.observer = Observer(location=self.location)
        self.generated_plot_dict = self._gen_default_labels(info)
        if mjd_start is None:
            self.mjd_start = SURVEY_START_MJD
        else:
            self.mjd_start = mjd_start

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
    ):

        if fig is None:
            fig, ax = plt.subplots()

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

        mjds = np.arange(np.min(data_slice[self.mjd_col]), np.max(data_slice[self.mjd_col]) + 1, 0.5)

        # Define the breakpoints as where either the filter changes OR
        # there's more than a 2 minute gap in observing
        good = np.where(
            (data_slice[self.filter_col] != np.roll(data_slice[self.filter_col], 1))
            | (
                np.abs(np.roll(data_slice[self.mjd_col], 1) - data_slice[self.mjd_col])
                > 120.0 / 3600.0 / 24.0
            )
        )[0]
        good = np.concatenate((good, [0], [len(data_slice[self.filter_col])]))
        good = np.unique(good)
        left = good[:-1]
        right = good[1:] - 1
        good = np.ravel(list(zip(left, right)))

        names = ["mjd", "midnight", "filter"]
        types = ["float", "float", (np.str_, 1)]
        perfilter = np.zeros((good.size), dtype=list(zip(names, types)))
        perfilter["mjd"] = data_slice[self.mjd_col][good]
        perfilter["filter"] = data_slice[self.filter_col][good]

        # Silence lots of ERFA warnings and astropy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # brute force compute midnight times for all days between
            # start and enc of data_slice
            times = Time(mjds, format="mjd")
            # let's just find the midnight before and after each of the
            # pre_night MJD values
            m_after = self.observer.midnight(times, "next")
            m_before = self.observer.midnight(times, "previous")
            try:
                midnights = np.unique(np.concatenate([m_before.mjd, m_after.mjd]).filled(np.nan))
            except AttributeError:
                midnights = np.unique(np.concatenate([m_before.mjd, m_after.mjd]))
            # calculating midnight can return nans? That seems bad.
            midnights = midnights[np.isfinite(midnights)]
            # chop off any repeats. Need to round because observe.midnight
            # values are not repeatable
            m10 = np.round(midnights * 10)
            _temp, indx = np.unique(m10, return_index=True)
            midnights = midnights[indx]
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
            pernight = np.zeros(len(midnights), dtype=list(zip(names, types)))
            pernight["midnight"] = midnights
            pernight["mjd"] = midnights

            # now for each perfilter, find the closes midnight
            indx = np.searchsorted(midnights, perfilter["mjd"])
            d1 = np.abs(perfilter["mjd"] - midnights[indx - 1])
            indx[np.where(indx >= midnights.size)] -= 1
            d2 = np.abs(perfilter["mjd"] - midnights[indx])

            perfilter["midnight"] = midnights[indx]
            temp_indx = np.where(d1 < d2)
            perfilter["midnight"][temp_indx] = midnights[indx - 1][temp_indx]
            mtime = Time(pernight["midnight"], format="mjd")

            pernight["twi12_rise"] = self.observer.twilight_morning_nautical(mtime, which="next").mjd
            pernight["twi12_set"] = self.observer.twilight_evening_nautical(mtime, which="previous").mjd

            pernight["twi18_rise"] = self.observer.twilight_morning_astronomical(mtime, which="next").mjd
            pernight["twi18_set"] = self.observer.twilight_evening_astronomical(mtime, which="previous").mjd

            aa = AltAz(location=self.location, obstime=mtime)
            moon_coords = get_body("moon", mtime).transform_to(aa)
            sun_coords = get_sun(mtime).transform_to(aa)
            ang_dist = sun_coords.separation(moon_coords)
            pernight["moonPer"] = ang_dist.deg / 180 * 100

            y = (perfilter["mjd"] - perfilter["midnight"]) * 24.0

        for i in np.arange(0, perfilter.size, 2):
            ax.plot(
                [perfilter["mjd"][i] - self.mjd_start, perfilter["mjd"][i + 1] - self.mjd_start],
                [y[i], y[i + 1]],
                filter2color[perfilter["filter"][i]],
            )
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
