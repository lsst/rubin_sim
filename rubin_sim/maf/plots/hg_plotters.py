"""Second generation hourglass plotting classes.
"""

# pylint: disable=too-many-arguments
# pylint: disable=super-init-not-called
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
__all__ = (
    "GeneralHourglassPlot",
    "CategoricalHourglassPlotMixin",
    "RangeHourglassCategoricalPlot",
    "MonthHourglassCategoricalPlot",
    "MonthHourglassUsePlot",
    "TimeUseHourglassPlotMixin",
    "MonthHourglassPlot",
    "YearHourglassCategoricalPlot",
    "YearHourglassPlot",
    "YearHourglassUsePlot",
)

import calendar
import copy
import logging
from collections import OrderedDict

import astropy.coordinates
import astropy.time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from rubin_scheduler.utils.ddf_locations import ddf_locations
from rubin_scheduler.utils.riseset import riseset_times

from .plot_handler import BasePlotter


class GeneralHourglassPlot(BasePlotter):
    """Make an hourglass plot

    Parameters
    ----------
    tz : `str`
        The timezone to use
    site : `str`
        The site name (sent to `astropy.coordinates.EarthLocation.of_site`)
    solar_time : `bool`
        Use solar time as the x axis (instead of civil time)?
    marked_ra : `dict`
        A dictionary of RA values (in deg) for which to label transit lines.

    A general feature of the hourglass plot is that you can pass more
    data (such as the metric values, calculated for the entire survey)
    to the plotter than are actually included in the plot. The plotter
    will pull out the select subset of data, defined by kwargs.

    """

    def __init__(
        self,
        tz="Chile/Continental",
        site="Cerro Pachon",
        solar_time=True,
        marked_ra=None,
    ):
        self.plot_type = "GeneralizedHourglass"
        self.object_plotter = False  # pylint: disable=invalid-name
        self.default_plot_dict = {
            "title": None,
            "xlabel": "Hours after midnight",
            "ylabel": "MJD",
            "x_min": -6,
            "x_max": 6,
            "figsize": None,
            "cmap": plt.get_cmap("viridis"),
            "colorbar": True,
            "legend": False,
            "legend_ncols": 2,
            "legend_bbox_to_anchor": (0.9, 0.0),
            "legend_loc": "lower right",
        }
        self.plot_dict = copy.copy(self.default_plot_dict)  # pylint: disable=invalid-name
        self.tz = tz  # pylint: disable=invalid-name
        if isinstance(site, str):
            self.site = astropy.coordinates.EarthLocation.of_site(site)
        else:
            self.site = site

        self.solar_time = solar_time
        self.color_map = {}
        self.marked_ra = {} if marked_ra is None else marked_ra

    def _add_axis_labels(self, ax):  # pylint: disable=invalid-name, no-self-use
        """Add title/x/y labels to one set of axes.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
           The axis to which to add labels
        """
        # Isolate code to add axis titles and labels because we need
        # to be able to call it independently of the actually plotting,
        # so that when we plot grids of axes on the same figure
        # we don't have overlaps and repeats.
        ax.set_title(self.plot_dict["title"])
        ax.set_xlabel(self.plot_dict["xlabel"])
        ax.set_ylabel(self.plot_dict["ylabel"])

    def _plot(self, fig, intervals):
        """Create axes for the data and add the data itself.

        This method does not add the legend or color bar. Use the
        __call__ method to combine all needed steps: handling creation
        of the intervals DataFrame, calling this method, and then
        adding any necessary legends or colorbars.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure on which to plot
        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.

        Returns
        -------
        color_mappable : `matplotlib.cm.ScalarMappable`
           map of scalar data to colors
        axes : numpy.array
           a numpy array of matplotlib axes.

        """

        # This is similar to the __call__ method
        # in other plotters, but isolates the making of the plot itself
        # from other things __call__ needs to do, such as reformat the
        # data and add figure-wide elements like the legend and color
        # bar, so that these tasks can be replaced independently in
        # subclases without needing to repeat the other elements of
        # call, which might not change.

        # Subclasses should override this method to change
        # how the plotting itself is done, as distinct from
        # restructuring the data or modifying other elements of the
        # figure such as the color bar and legend.

        ax = fig.add_axes([0, 0, 1, 1])  # pylint: disable=invalid-name
        color_mappable = self._plot_one_axis(intervals, ax)
        self._add_axis_labels(ax, self.plot_dict)
        return [color_mappable, np.array([ax])]

    def _plot_one_axis(self, intervals, ax, epoch_mjd=None):  # pylint: disable=invalid-name
        """Actually add the data to one axis of an hourglass plot

        includes adding the twilight/transit lines, does not include any
        supporting infrastructure or plot elements: it does not create
        the figure or axis, or add labels, color bars, or other
        elements of the plot that do not represent the data itself.

        Parameters
        ----------
        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.
        ax : `matplotlib.axes.Axes`
           the axes on which to plot data
        epoch_mjd : `float`
           the MJD used to set y limits for the plot
        """

        # Add content relevant to an individual axis.
        # It is separated from _plot so that it can be called
        # in different subplots, while _plot drives the the content
        # of all subplots.

        # Add the 'block' data
        if len(intervals) > 0:
            if self.solar_time:
                (
                    night_mjds,
                    hours_after_midnight,
                ) = _compute_hours_after_solar_midnight(intervals["mjd"].values, self.site)
            else:
                (
                    night_mjds,
                    hours_after_midnight,
                ) = _compute_hours_after_midnight(intervals["mjd"].values)

            color_mappable = self._plot_value_bars(
                intervals["value"],
                night_mjds,
                hours_after_midnight,
                intervals["duration"],
                epoch_mjd,
                ax,
            )
        else:
            color_mappable = None

        try:
            ax.set_ylim(self.plot_dict["y_max"], self.plot_dict["y_min"])
        except KeyError:
            ax.set_ylim(ax.get_ylim()[-1] + 1, -0.5)
        ax.set_xlim(self.plot_dict["x_min"], self.plot_dict["x_max"])

        start_mjd = ax.get_ylim()[1] + epoch_mjd
        end_mjd = ax.get_ylim()[0] + epoch_mjd
        # Add the twilight boundaries and moon transit lines
        _astron_hourglass(start_mjd + 0.5, end_mjd + 1, ax, self.site, self.solar_time)
        # Add the transits
        self._plot_transits(start_mjd + 0.5, end_mjd + 1, ax)

        return color_mappable

    def _plot_value_bars(  # pylint: disable=invalid-name
        self,
        values,
        night_mjds,
        hours_after_midnight,
        duration,
        epoch_mjd,
        ax,
    ):
        """Add the color bar blocks for the metric values.

        Parameters
        ----------
        values : `numpy.array`
           values to assign to the value bars
        night_mjds : `numpy.array`
           array if integer MJDs of the nights
           for each bar.
        hours_after_midnight : `numpy.array`
           array of times (hours after midnight) for the
           start times of each bar.
        duration : `numpy.array`
           array of durations (in hours) for each bar
        epoch_mjd : `int`
           the reference MJD of the vertical axis
           of the plot, for example the start of the
           month if the vertical axis is to be labeled
           with the day of month.
        ax : `matplotlib.axes.Axes`
           the axes on which to plot the bars

        Returns
        -------
        color_mappable : `matplotlib.cm.ScalarMappable`
           relates values to colors for the bars
           (see the matplotlib documentation)
        """

        # Drives the plotting of the elements that represent the metric itself,
        # changing as a function of time.

        # Find the colors to be used.
        colors, color_mappable = self._map_colors(values)

        # Add the bars.
        ax.bar(
            hours_after_midnight,
            height=0.8,
            width=duration,
            bottom=night_mjds - epoch_mjd - 0.4,
            color=colors,
            linewidth=0,
            align="edge",
        )

        return color_mappable

    def _map_colors(self, values):  # pylint: disable=invalid-name, no-self-use
        """Get colors corresponding to values

        Paramaters
        ----------
        values : `numpy.array`
           Values for which to get colors

        Returns
        -------
        colors : `numpy.array`
           colors corresponding to the provided values
        color_mappable : `matplotlib.cm.ScalarMappable`
           relates values to colors for the bars
           (see the matplotlib documentation)
        """

        # Manages the mapping of metric values to colors in the
        # horizonatal lines in the hourglass plot.
        cmap = self.plot_dict["cmap"]
        if values.max() == values.min():
            colors = cmap(0.5)
        else:
            try:
                color_limits = self.plot_dict["color_limits"]
            except KeyError:
                color_limits = (values.min(), values.max())
            colors = cmap((values - color_limits[0]) / (color_limits[1] - color_limits[0]))

        norm = mpl.colors.Normalize(vmin=color_limits[0], vmax=color_limits[1])
        color_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        color_mappable.set_array([])

        return colors, color_mappable

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        """Restructure the metric data to use, and build the figure.

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
        # Highest level driver for the plotter.
        # Prepares data structures and figure-wide elements

        # Building the figure entails plotting it the relevant
        # class._plot function and adding plot decorations with
        # the relevant class._add_colorbar and class._add_figure_legend
        # methods.

        intervals = pd.DataFrame(
            {
                "mjd": slicer.slice_points["mjd"],
                "duration": (slicer.slice_points["duration"] * u.s).to_value(
                    u.hour  # pylint: disable=no-member
                ),
                "value": metric_value,
            }
        )
        # Set up a copy of the plot dict to contain entries relevant
        # for this plot
        self.plot_dict = copy.copy(self.default_plot_dict)  # pylint: disable=invalid-name
        self.plot_dict.update(user_plot_dict)

        # Generate the figure
        if fig is None:
            fig = plt.figure(figsize=self.plot_dict["figsize"])

        # Add the plots
        color_mappable, axes = self._plot(fig, intervals)

        # add the colorbar, if requested
        if self.plot_dict["colorbar"] and color_mappable is not None:
            self._add_colorbar(fig, color_mappable, axes)

        # add the legend, if requested
        if self.plot_dict["legend"] and len(self.color_map) > 0:
            fig = self._add_figure_legend(fig, axes)

        return fig

    def _add_figure_legend(self, fig, axes):
        """Creates and adds the figure legend.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure on which to plot
        axes : `matplotlib.axes.Axes`
           axes from which to extract
           automatic labels.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
           The matplotlib figure
        """
        # Creates and adds the figure legend.  This method follows two
        # stages: first, it adds entries for each element in the color
        # map (defined in self.color_map).  Then, it adds any
        # additional elements of the plats that were given matplotlib
        # labels, but are not already in the legend by way of the
        # color map.  The user may limit the enties in the color map
        # by way of the legend_elements key in the plot_dict, which if
        # preset limits the elements of the key to only those
        # requested.

        # Build a dictionary and build lable and handle lists
        # later, so that we can ensure that the order of the
        # labels in the keys matches the order specified by the
        # user, if present.
        handle_dict = OrderedDict()

        # Label everything we have assigned a color
        for label in self.color_map:
            if "legend_elements" in self.plot_dict:
                if label not in self.plot_dict["legend_elements"]:
                    continue

            handle_dict[label] = mpl.patches.Patch(facecolor=self.color_map[label], label=label)

        # If we missed anything we have assigned a label while
        # plotting, automatically pick it up, but avoid
        # duplication
        auto_handles, auto_labels = axes.T.flatten()[-1].get_legend_handles_labels()
        for auto_handle, auto_label in zip(auto_handles, auto_labels):
            handle_dict[auto_label] = auto_handle

        # If the user specified which elements they wanted labels,
        # include only them (if present).
        # Otherwise, unclude all defined labels
        if "legend_elements" in self.plot_dict:
            desired_labels = self.plot_dict["legend_elements"]
        else:
            desired_labels = handle_dict.keys()

        labels = [lbl for lbl in desired_labels if lbl in handle_dict]
        handles = [handle_dict[lbl] for lbl in desired_labels if lbl in handle_dict]

        fig.legend(
            handles,
            labels,
            loc=self.plot_dict["legend_loc"],
            ncol=self.plot_dict["legend_ncols"],
            bbox_to_anchor=self.plot_dict["legend_bbox_to_anchor"],
        )

        return fig

    def _add_colorbar(self, fig, color_mappable, axes):  # pylint: disable=invalid-name, no-self-use
        """Add a colorbar.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure on which to plot
        color_mappable : `matplotlib.cm.ScalarMappable`
           map of scalar data to colors
        axes : `matplotlib.axes.Axes`
           The axes for which to add the colorbar
        """

        colorbar_kwargs = {"fraction": 0.05}
        if "colorbar_ticks" in self.plot_dict:
            colorbar_kwargs["ticks"] = self.plot_dict["colorbar_ticks"]
        fig.colorbar(color_mappable, ax=axes, **colorbar_kwargs)

    def _plot_dates(  # pylint: disable=invalid-name
        self,
        intervals,
        start_tstamp,
        end_tstamp,
        ax,
        epoch_tstamp=None,
    ):
        """Plot data on one axis for a range of dates.

        Parameters
        ----------
        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.
        start_tstamp : `pandas.Timestamp`
           The first time to include in the axes
        end_tstamp : `pandas.Timestamp`
           The last time to include in the axes
        ax : `matplotlib.axes.Axes`
           the axes on which to plot data
        epoch_tstamp : `float`
           the time used to set y limits for the plot


        Returns
        -------
        color_mappable : `matplotlib.cm.ScalarMappable`
           map of data to colors
        """
        if self.plot_dict is None:
            self.plot_dict = {}

        start_mjd = start_tstamp.to_julian_date() - 2400000.5
        end_mjd = end_tstamp.to_julian_date() - 2400000.5
        epoch_mjd = 0 if epoch_tstamp is None else epoch_tstamp.to_julian_date() - 2400000.5

        these_intervals = intervals.query(f"{start_mjd}<mjd<{end_mjd}")
        color_mappable = self._plot_one_axis(these_intervals, ax, epoch_mjd)
        return color_mappable

    def _plot_month(self, intervals, month, year, ax):
        """Plots data as function of date in the month.

        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.
        month : `int`
           Month number whose dates to plot
        year : `int`
           Year of the month to plot
        ax : `matplotlib.axes.Axes`
           the axes on which to plot data

        Returns
        -------
        color_mappable : `matplotlib.cm.ScalarMappable`
           map of data to colors
        """

        if self.plot_dict is None:
            self.plot_dict = {}

        start_tstamp = pd.Timestamp(year=year, month=month, day=1, hour=12, tz=self.tz)
        end_tstamp = start_tstamp + pd.DateOffset(months=1)
        epoch_tstamp = pd.Timestamp(year=year, month=month, day=1, hour=0, tz="UTC") - pd.DateOffset(days=1)
        color_mappable = self._plot_dates(
            intervals,
            start_tstamp,
            end_tstamp,
            ax,
            epoch_tstamp=epoch_tstamp,
        )
        ax.set_yticks(np.arange(1, 32, 7))
        return color_mappable

    def _plot_transits(self, start_mjd, end_mjd, ax):  # pylint: disable=invalid-name
        """Plot transit lines on a set of axes.

        Parameters
        ----------
        start_mjd : `int`
           Start time for the axes
        end_mjd: `int`
           End time for the exes
        ax: `matplotlib.axes.Axes`
           the axes on which to plot transit lines
        """
        # Add transit lines for each of the RA values
        # it self.marked_ra.
        mjds = np.arange(start_mjd, end_mjd + 1, 1)

        for field in self.marked_ra:
            ra = self.marked_ra[field]  # pylint: disable=invalid-name
            try:
                color = self.color_map[field]
            except KeyError:
                continue

            transit_mjds = _compute_coord_transit_mjds(mjds, ra, self.site)

            if self.solar_time:
                (
                    transit_nights,
                    transit_hams,
                ) = _compute_hours_after_solar_midnight(transit_mjds, self.site)
            else:
                transit_nights, transit_hams = _compute_hours_after_midnight(transit_mjds)

            ax.plot(
                transit_hams,
                transit_nights - start_mjd + 1,
                color=color,
                label=field + "-transit",
            )


#
# Plot appearance subclasses
#

# The following subclasses change the arrangement of dates on the
# hourglass plot.


class RangeHourglassPlot(GeneralHourglassPlot):
    """Make an hourglass plot for a range of dates

    Parameters
    ----------
    start_date : `str`
        The start date (in a format parseable by `pandas.Timestamp`)
    end_date : `str`
        The end date (in a format parseable by `pandas.Timestamp`)

    Keyward arguments are passed to `GeneralHourglassPlot`
    """

    def __init__(self, start_date, end_date, **kwargs):
        super().__init__(**kwargs)
        self.plot_type = "RangeHourglass"
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

    def _plot(self, fig, intervals):
        """Make the hourglass plot for a range of dates.

        This method does not add the legend or color bar. Use the
        __call__ method to handle creation of the intervals DataFrame,
        call this method, and then add any necessary legends or
        colorbars.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure on which to plot
        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.

        """
        ax = fig.add_axes([0, 0, 1, 1])  # pylint: disable=invalid-name
        color_mappable = self._plot_dates(intervals, self.start_date, self.end_date, ax)
        self._add_axis_labels(ax, self.plot_dict)
        return color_mappable, np.array([ax])


class MonthHourglassPlot(GeneralHourglassPlot):
    """Make an hourglass plot for a month

    Parameters
    ----------
    month : `int`
        The month number (1-12).
    year : `int`
        The year.

    Keyward arguments are passed to `GeneralHourglassPlot`

    Note that this pulls the chosen month's data out of the metric
    values calculated for the entire survey.  Unsubclassed, this class
    expects to obtain float/int data.  To use it with other data, its
    subclass that includes the appropriate mixin
    (MonthHourglassCategoricalPlot or MonthHourglassUsePlot) or a
    custom subclass.

    """

    def __init__(self, month, year, **kwargs):
        super().__init__(**kwargs)
        self.plot_type = "MonthHourglass"
        self.default_plot_dict["ylabel"] = "day of month"
        self.default_plot_dict["y_max"] = calendar.monthrange(year, month)[1] + 0.5
        self.default_plot_dict["y_min"] = 0.5
        self.month = month
        self.year = year

    def _plot(self, fig, intervals):
        """Make the hourglass plot for a month.

        This method does not add the legend or color bar. Use the
        __call__ method to handle creation of the intervals DataFrame,
        call this method, and then add any necessary legends or
        colorbars.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure on which to plot
        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.

        """

        ax = fig.add_axes([0, 0, 1, 1])  # pylint: disable=invalid-name
        color_mappable = self._plot_month(intervals, self.month, self.year, ax)
        self._add_axis_labels(ax)
        return color_mappable, np.array([ax])


class YearHourglassPlot(GeneralHourglassPlot):
    """Make an array of monthly hourglass plots for a year.

    Parameters
    ----------
    year : `int`
        The year.

    Keyward arguments are passed to `GeneralHourglassPlot`

    Note that this plot pulls the chosen year's data out of the metric
    values calculated for the entire survey.  Like the
    MonthHourglassPlot class, this class expects to obtain float/int
    data.  To use it with other data, its subclass that includes the
    appropriate mixin (YearHourglassCategoricalPlot or
    YearHourglassUsePlot) or a custom subclass.

    """

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        self.year = year
        self.plot_type = "YearHourglass"
        self.default_plot_dict["ylabel"] = "day of month"
        self.default_plot_dict["y_max"] = 31.5
        self.default_plot_dict["y_min"] = 0.5

    def _plot(self, fig, intervals):
        """Make a series of month plots, arranged for the entire year.

        This method does not add the legend or color bar. Use the
        __call__ method to handle creation of the intervals DataFrame,
        call this method, and then add any necessary legends or
        colorbars.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure on which to plot
        intervals : `pandas.DataFrame`
           data to be plotted, with "mjd", "duration", and "value" columns.
           "duration" should be in units of hours.

        """

        axes = fig.subplots(
            3,
            4,
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0.025, "hspace": 0},
        )

        for month, ax in zip(np.arange(1, 13), axes.T.flatten()):  # pylint: disable=invalid-name
            logging.info("Working on %s, %d", calendar.month_name[month], self.year)
            self.plot_dict["y_max"] = calendar.monthrange(self.year, month)[1] + 0.5
            color_mappable = self._plot_month(intervals, month, self.year, ax)
            ax.set_title("")
            ax.set_title(calendar.month_abbr[month], y=1, x=0.005, pad=-15, loc="left")

        self._add_figure_labels(fig, axes)
        return color_mappable, axes.ravel()

    def _add_figure_labels(self, fig, axes):  # pylint: disable=invalid-name, no-self-use
        """Generate the figure labels.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
           figure to which to add lables
        ax : `matplotlib.axes.Axes`
           The axis to which to add labels
        """
        axes[1, 0].set_ylabel(self.plot_dict["ylabel"])
        fig.suptitle(self.plot_dict["title"], y=0.9)
        fig.text(0.5, 0.05, self.plot_dict["xlabel"], ha="center")


#
# Mixin classes to change handling of metric data
#

# Mixin classes avoid the need to duplicate code for every appearance
# type subclass. Rather that need to duplicate code between
# RangeHourglassPlot, MonthHourglassPlot, and YearHourglassPlot to
# enable them all to support hourglass data, a single "mixin" class
# allows the needed modifications to be defined once, and then "mixed
# in" to each of these classes (see RangeHourglassCategoricalPlot,
# MonthHourglassCategoricalPlot, and YearHourglassCategoricalPlot
# below).


class CategoricalHourglassPlotMixin:
    """A mix-in to allow the HourglassPlot to accept categorical data,
    rather than float/int.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_plotter = True  # pylint: disable=invalid-name
        self.plot_type = (  # pylint: disable=invalid-name
            "Categorical" + self.plot_type
        )  # pylint: disable=invalid-name
        self.default_plot_dict["cmap"] = plt.get_cmap("tab10")
        self.default_plot_dict["assigned_colors"] = OrderedDict()
        self.default_plot_dict["legend"] = True
        self.default_plot_dict["colorbar"] = False
        if "marked_ra" not in kwargs:
            self.marked_ra = {}

    def _map_colors(self, values):
        cmap = self.plot_dict["cmap"]
        assigned_colors = self.plot_dict["assigned_colors"]
        self.color_map = {}
        if len(self.color_map.keys()) == 0:
            self.color_map = _assign_category_colors(
                np.unique(values),
                cmap,
                assigned_colors=assigned_colors,
            )

        colors = np.array(np.vectorize(self.color_map.get)(values)).T.tolist()
        color_mappable = None

        return colors, color_mappable


class RangeHourglassCategoricalPlot(CategoricalHourglassPlotMixin, RangeHourglassPlot):
    """Plot categorical data for a range of dates."""


class MonthHourglassCategoricalPlot(CategoricalHourglassPlotMixin, MonthHourglassPlot):
    """Plot categorical data for a Month."""


class YearHourglassCategoricalPlot(CategoricalHourglassPlotMixin, YearHourglassPlot):
    """Plot categorical data for a year."""


class TimeUseHourglassPlotMixin(CategoricalHourglassPlotMixin):
    """A mix-in to allow the HourglassPlot to accept categorical 'use' data
    rather than float/int."""

    def __init__(self, *args, **kwargs):
        """Customize hourglass plotter for time use"""
        super().__init__(*args, **kwargs)
        self.plot_type = "Use" + self.plot_type  # pylint: disable=invalid-name
        self.default_plot_dict["cmap"] = plt.get_cmap("tab10")
        self.default_plot_dict["colorbar"] = False
        self.default_plot_dict["assigned_colors"] = OrderedDict(
            (
                ("wide with only IR", 3),
                ("wide with u, g, or r", 0),
                ("greedy", 1),
            )
        )

        self.default_plot_dict["legend"] = True
        if "marked_ra" not in kwargs:
            self.marked_ra = {f: c[0] for f, c in ddf_locations().items()}


class MonthHourglassUsePlot(TimeUseHourglassPlotMixin, MonthHourglassPlot):
    """Plot categorical 'use' data for one month."""


class YearHourglassUsePlot(TimeUseHourglassPlotMixin, YearHourglassPlot):
    """Plot categorical 'use' data for one year."""


# internal functions & classes


def _assign_category_colors(uses, cmap, use_colors=None, assigned_colors=None):
    """Set a dictionary of nice colors for the use blocks.
    Options allow specifing pre-defined elements for some categories."""

    use_colors = OrderedDict() if use_colors is None else use_colors
    assigned_colors = OrderedDict() if assigned_colors is None else assigned_colors

    available_idxs = list(range(cmap.N))

    used_colors = set(use_colors.values())
    for cmap_idx in range(cmap.N):
        # Skip gray because it is confusing against background
        this_cmap = cmap(cmap_idx)
        if this_cmap[0] == this_cmap[1] and this_cmap[0] == this_cmap[2]:
            available_idxs.remove(cmap_idx)

        # If a color has been used already, don't reuse it for something else
        if cmap(cmap_idx) in used_colors:
            available_idxs.remove(cmap_idx)

    for use, cmap_idx in assigned_colors.items():
        if cmap_idx in available_idxs:
            available_idxs.remove(cmap_idx)
            use_colors[use] = cmap(cmap_idx)
        else:
            assert use_colors[use] == cmap(cmap_idx)

    for use in uses:
        if use not in use_colors:
            use_idx = available_idxs[0]
            use_colors[use] = cmap(use_idx)
            available_idxs.remove(use_idx)

    return use_colors


def _compute_hours_after_midnight(mjd, tz="Chile/Continental"):  # pylint: disable=invalid-name
    """Calculate the civil 'hours' value for the hourglass plot.

    Parameters
    ----------
    mjd : `numpy.array`
       array of Modified Julian Dates
    tz : `str`
       The timezone (to be passed to
       `pandas.DatetimeIndex.tz_convret`)

    Returns
    -------
    night_mjds : `numpy.array`
       array of interger nights for the dates of the provided mjds.
    hours_after_midnight : `numpy.array`
       array of hours after local civil midnight in the timezone
       provided, for each input mjd value.
    """

    utc_datetimes = pd.to_datetime(mjd + 2400000.5, unit="D", origin="julian").tz_localize("UTC")
    local_times = pd.DatetimeIndex(utc_datetimes).tz_convert(tz)
    night_mjds = np.floor(local_times.to_julian_date() - 2400001).astype(int).values
    hours_after_midnight = (local_times.to_julian_date().values - 2400001.5 - night_mjds) * 24
    return night_mjds, hours_after_midnight


def _compute_hours_after_solar_midnight(mjd, site, tz="Chile/Continental"):  # pylint: disable=invalid-name
    """Calculate the solar 'hours' value for the hourglass plot.

    Parameters
    ----------
    mjd : `numpy.array`
       array of Modified Julian Dates
    tz : `str`
       The timezone (to be passed to
       `pandas.DatetimeIndex.tz_convret`)

    Returns
    -------
    night_mjds : `numpy.array`
       array of interger nights for the dates of the provided mjds.
    hours_after_midnight : `numpy.array`
       array of hours after mean local solar midnight (antitransit),
       for each input mjd value.
    """

    times = astropy.time.Time(mjd, format="mjd", location=site)
    mean_solar_jd = times.ut1.mjd + site.lon.deg / 360
    mean_solar_time = astropy.coordinates.Angle(
        mean_solar_jd * 360, unit=u.deg  # pylint: disable=no-member
    ).wrap_at(
        180 * u.deg  # pylint: disable=no-member
    )
    hours_after_midnight = (
        mean_solar_time.to_value(u.deg) * 24 / 360.0  # pylint: disable=no-member  # pylint: disable=no-member
    )  # pylint: disable=no-member
    utc_datetimes = pd.to_datetime(mjd + 2400000.5, unit="D", origin="julian").tz_localize("UTC")
    local_times = pd.DatetimeIndex(utc_datetimes).tz_convert(tz)
    night_mjds = np.floor(local_times.to_julian_date() - 2400001).astype(int).values
    return night_mjds, hours_after_midnight


def _compute_coord_transit_mjds(mjds, ra, site):  # pylint: disable=invalid-name
    """Compute the coordinate transit times.

    Parameters
    ----------
    mjds : `numpy.array`
       array of Modified Julian Dates for the nights
       for which to calculate transit
    ra : `float`
       R.A. for which to calculate transit
       in degrees
    site : `astropy.coordinates.EarthLocation`
       location of the telescope site

    Return
    ------
    mjds : `numpy.array`
       array of MJDs of transit times.
    """
    # Used for transit lines for the hourglass plot.

    # Calculate the local sidereal time at mjds - in degrees (for RA
    # comparison)
    lsts = astropy.time.Time(mjds, format="mjd", location=site).sidereal_time("apparent").deg
    # pylint: disable=invalid-name, no-member
    ha = astropy.coordinates.Angle(lsts - ra, unit=u.deg).wrap_at(180 * u.deg)
    mjds = mjds - ha.to_value(u.deg) / 360
    return mjds


def _compute_moon_transit_mjds(mjds, site):
    """Compute the transit times for the moon,

    Parameters
    ----------
    mjds : `numpy.array`
       array of Modified Julian Dates for the nights
       for which to calculate transit
    site : `astropy.coordinates.EarthLocation`
       location of the telescope site

    Return
    ------
    mjds : `numpy.array`
       array of MJDs of transit times.
    """
    # for transit lines for the hourglass plot.

    for _ in np.arange(3):
        times = astropy.time.Time(mjds, format="mjd", location=site)
        moon_coords = astropy.coordinates.get_body("moon", times)
        # Calculate the local sidereal time at mjds
        lsts = astropy.time.Time(mjds, format="mjd", location=site).sidereal_time("apparent")
        # pylint: disable=invalid-name, no-member
        ha = (lsts - moon_coords.ra).wrap_at(180 * u.deg)
        mjds = mjds - ha.to_value(u.deg) / 360
    return mjds


def _astron_hourglass(
    start_mjd, end_mjd, ax, site, solar_time
):  # pylint: disable=invalid-name, too-many-locals, too-many-branches
    """Add the moon transit, night and twilight start/end lines.

    Parameters
    ----------
    start_mjd : `int`
       Start time for the axes
    end_mjd: `int`
       End time for the exes
    ax: `matplotlib.axes.Axes`
       the axes on which to plot transit lines
    site : `astropy.coordinates.EarthLocation`
       location of the telescope site
    solar_time : `bool`
       True to use mean local solar midnight
       False to use civil midnight

    Returns
    -------
    ax: `matplotlib.axes.Axes`
       the axes on transit lines were plotted
    """

    mjds = np.arange(start_mjd, end_mjd + 1, 1)
    start_mjd = mjds[0]

    # Moon transit

    cal_night_mjds = mjds - site.lon.deg / 360

    moon_transit_mjds = _compute_moon_transit_mjds(cal_night_mjds, site)

    if solar_time:
        (
            moon_transit_nights,
            moon_transit_hams,
        ) = _compute_hours_after_solar_midnight(moon_transit_mjds, site)
    else:
        moon_transit_nights, moon_transit_hams = _compute_hours_after_midnight(moon_transit_mjds)
    # Loop to avoid wrapping
    moon_lines = np.cumsum(np.diff(moon_transit_hams, prepend=moon_transit_hams[0]) < 0)
    moon_label = "moon"
    for moon_line in np.unique(moon_lines):
        these_hams = moon_transit_hams[moon_lines == moon_line]
        these_nights = moon_transit_nights[moon_lines == moon_line]
        ax.plot(
            these_hams,
            these_nights - start_mjd + 1,
            color="yellow",
            linewidth=8,
            alpha=0.5,
            label=moon_label,
        )
        moon_label = None

    # Moon rise and set
    for direction in ("up", "down"):
        moon_event_mjds = riseset_times(cal_night_mjds, direction, alt=0, body="moon")
        if solar_time:
            (
                moon_event_nights,
                moon_event_hams,
            ) = _compute_hours_after_solar_midnight(moon_event_mjds, site)
        else:
            moon_event_nights, moon_event_hams = _compute_hours_after_midnight(moon_event_mjds)
        # Loop to avoid wrapping
        moon_lines = np.cumsum(np.diff(moon_event_hams, prepend=moon_event_hams[0]) < 0)
        for moon_line in np.unique(moon_lines):
            these_hams = moon_event_hams[moon_lines == moon_line]
            these_nights = moon_event_nights[moon_lines == moon_line]
            ax.plot(
                these_hams,
                these_nights - start_mjd + 1,
                color="yellow",
                linestyle="dotted",
            )

    # Twilight
    guess_offset = {"up": 0.2, "down": -0.2}
    twilight_shade = {0: "0.7", -6: "0.6", -12: "0.4", -18: "black"}

    for alt in twilight_shade:
        pm_event_mjds = riseset_times(cal_night_mjds + guess_offset["down"], "down", alt=alt, body="sun")
        am_event_mjds = riseset_times(cal_night_mjds + guess_offset["up"], "up", alt=alt, body="sun")
        if solar_time:
            (
                pm_event_nights,
                pm_event_hams,
            ) = _compute_hours_after_solar_midnight(pm_event_mjds, site)
            (
                am_event_nights,
                am_event_hams,
            ) = _compute_hours_after_solar_midnight(am_event_mjds, site)
        else:
            (
                pm_event_nights,
                pm_event_hams,
            ) = _compute_hours_after_midnight(pm_event_mjds)
            (
                am_event_nights,
                am_event_hams,
            ) = _compute_hours_after_midnight(am_event_mjds)
        assert np.array_equal(am_event_nights, pm_event_nights)
        ax.fill_betweenx(
            am_event_nights - start_mjd + 1,
            pm_event_hams,
            am_event_hams,
            color=twilight_shade[alt],
            zorder=-10,
        )

    return ax
