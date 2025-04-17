__all__ = ("BasePlot", "PlotMoll", "PlotHist", "PlotHealHist", "PlotLine", "PlotLambert", "PlotFo")

import copy
import warnings

import healpy as hp
import matplotlib.pylab as plt
import numpy as np
import rubin_scheduler.utils as utils
from matplotlib import ticker

from rubin_sim.maf_proto.utils import optimal_bins, fO_calcs


class BasePlot(object):
    def __init__(self, info=None):

        self.info = info
        self.generated_plot_dict = self._gen_default_labels(info)

    def _gen_default_labels(self, info):
        """Generate any default label values"""
        result = {}
        return result

    def __call__(self, data, fig=None):
        if fig is None:
            fig, ax = plt.subplots()
        return fig


class PlotMoll(BasePlot):
    """Plot a mollweild projection of a HEALpix array."""

    def __init__(self, info=None):

        self.info = info
        self.moll_kwarg_dict = self._gen_default_labels(info)

    def _gen_default_labels(self, info):
        """ """
        result = {}
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n" + info["observations_subset"]

            if "metric: unit" in info.keys():
                result["unit"] = info["metric: unit"]
        return result

    def default_cb_params(self):
        cb_params = {
            "shrink": 0.75,
            "aspect": 25,
            "pad": 0.1,
            "orientation": "horizontal",
            "format": "%.1f",
            "extendrect": False,
            "extend": "neither",
            "labelsize": None,
            "n_ticks": 5,
            "cbar_edge": True,
            "fontsize": None,
            "label": None,
        }
        return cb_params

    def __call__(
        self,
        inarray,
        fig=None,
        add_grat=True,
        grat_params="default",
        cb_params="default",
        log=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        inarray : `np.array`
            numpy array with proper HEALpix size.
        fig : `matplotlib.Figure`
            A matplotlib figure object. Default of None
            creates a new figure.
        add_grat : `bool`
            Add gratacule to the plot. Default True.
        grat_params : `dict`
            Dictionary of kwargs to pass to healpy.graticule.
            Default of "default" generates a reasonable dict.
        cb_params : `dict`
            Dictionary of color bar parameters. Default of "default"
            uses PlotMoll.default_cb_params to construct defaults. Setting
            to None uses the healpy default colorbar. Setting
            cbar=False and cb_params=None should result in no colorbar.
        log : `bool`
            Set the colorbar to be log. Default False.
        **kwargs
            Kwargs sent to healpy.mollview. E.g.,
            title, unit, rot, min, max.

        """
        if fig is None:
            fig = plt.figure()

        # Nothing valid to plot, just return
        if np.sum(np.isfinite(inarray)) == 0:
            warnings.warn("No finite values to plot, returning empty figure")
            return fig

        if grat_params == "default":
            grat_params = {"dpar": 30, "dmer": 30}
        # Copy any auto-generated plot kwargs
        moll_kwarg_dict = copy.copy(self.moll_kwarg_dict)
        # Override if those things have been set with kwargs
        for key in kwargs:
            moll_kwarg_dict[key] = kwargs.get(key)

        # mollview seems to throw lots of warnings when using fig.number
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if cb_params is None:
                hp.mollview(inarray, **moll_kwarg_dict, fig=fig.number)
            else:
                hp.mollview(inarray, **moll_kwarg_dict, fig=fig.number, cbar=False)

        if add_grat:
            hp.graticule(**grat_params)
        self.ax = plt.gca()
        im = self.ax.get_images()[0]

        # Make sure cbar wasn't set to False
        if "cbar" in kwargs:
            cbar = kwargs["cbar"]
        else:
            cbar = True

        if cbar:
            if cb_params == "default":
                cb_params = self.default_cb_params()
            else:
                defaults = self.default_cb_params()
                for key in cb_params:
                    defaults[key] = cb_params[key]
                cb_params = defaults

            if cb_params["label"] is None:
                if "unit" in moll_kwarg_dict.keys():
                    cb_params["label"] = moll_kwarg_dict["unit"]

            cb = plt.colorbar(
                im,
                shrink=cb_params["shrink"],
                aspect=cb_params["aspect"],
                pad=cb_params["pad"],
                orientation=cb_params["orientation"],
                format=cb_params["format"],
                extendrect=cb_params["extendrect"],
                extend=cb_params["extend"],
            )
            cb.set_label(cb_params["label"], fontsize=cb_params["fontsize"])

            if cb_params["labelsize"] is not None:
                cb.ax.tick_params(labelsize=cb_params["labelsize"])
            if log:
                tick_locator = ticker.LogLocator(numticks=cb_params["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
            else:
                if cb_params["n_ticks"] is not None:
                    tick_locator = ticker.MaxNLocator(nbins=cb_params["n_ticks"])
                    cb.locator = tick_locator
                    cb.update_ticks()
            # If outputing to PDF, this fixes the colorbar white stripes
            if cb_params["cbar_edge"]:
                cb.solids.set_edgecolor("face")

        return fig


class PlotHist(BasePlot):

    def _gen_ylabel(self):
        return "#"

    def _gen_default_labels(self, info):
        """ """
        result = {"ylabel": self._gen_ylabel()}
        result["title"] = ""
        result["xlabel"] = ""
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n" + info["observations_subset"]
            if "metric: unit" in info.keys():
                result["xlabel"] = info["metric: unit"]

        return result

    def __call__(
        self,
        inarray,
        fig=None,
        ax=None,
        title=None,
        xlabel=None,
        ylabel=None,
        histtype="step",
        bins="optimal",
        **kwargs,
    ):
        """
        Parameters
        ----------
        inarray : `np.array`
            Vector to be histogrammed.
        fig : `matplotlib.Figure`
            Matplotlib Figure object to use. Default None
            will generate a new Figure.
        ax : `matplotlib.Axes`
            Matplotlib Axes object to use. Default None
            will generate a new axes
        title,xlabel,ylabel : `str`
            title to set on Axes. Default None will
            use auto-generated title from _gen_default_labels method.
        histtype : `str`
            Histogram type passed to matplotlib.hist. Default `step`.
        bins : `np.array`
            bins passed to matplotlib.hist. Default "optimal" will
            compute an "optimal" number of bins.
        **kwargs
            Additional keyword arguments passed to `matplotlib.hist`.
            E.g., range, log, align, cumulative, etc. Note histogram
            also passes through matplotlib.Patch properties like
            edgecolor, facecolor, linewidth.
        """

        if isinstance(bins, str):
            if bins == "optimal":
                bins = optimal_bins(inarray)

        overrides = {"title": title, "xlabel": xlabel, "ylabel": ylabel}
        plot_dict = copy.copy(self.generated_plot_dict)
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        if fig is None:
            fig, ax = plt.subplots()

        _n, _bins, _patches = ax.hist(inarray, histtype=histtype, bins=bins, **kwargs)
        ax.set_title(plot_dict["title"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])

        return fig


class PlotHealHist(PlotHist):
    """Make a histogram of a HEALpix array

    Parameters
    ----------
    info : `dict`
        Dictionary with information about reduction history
    scale : `float`
        Scaling used for plotting area on y-axis. Default of 1000
        results in 1000s of square degrees as the y-axis value
        and ylabel.
    """

    def __init__(self, info=None, scale=1000):
        self.scale = scale
        super().__init__(info=info)

    def _gen_ylabel(self):
        return "Area (%is of sq degrees)" % self.scale

    def __call__(self, inarray, fig=None, ax=None, histtype="step", bins="optimal", **kwargs):
        """
        Parameters
        ----------
        inarray : `np.array`
            HEALpix array to be histogrammed.
        fig : `matplotlib.Figure`
            Matplotlib figure to use. Default of None
            generates a new Figure
        ax : `matplotlib.Axes`
            Matplotlib Axes to use. If None, one will be created
        histtype : `str`
            histogram type passed to matplotlib.Ax.hist.
            Default "step".
        bins : `np.array`
            Bins passed through to matplotlib.Ax.hist.
            Default of "optimal" computes optimal bin sizes.
        **kwargs
            Additional kwargs passed to matplotlib.Ax.hist
        """

        pix_area = hp.nside2pixarea(hp.npix2nside(np.size(inarray)), degrees=True)
        weights = np.zeros(np.size(inarray)) + pix_area / self.scale
        super().__call__(inarray, fig=fig, ax=ax, histtype=histtype, weights=weights, bins=bins, **kwargs)


class PlotLine(BasePlot):
    def __init__(self, info=None):
        self.info = info
        self.generated_plot_dict = self._gen_default_labels(info)

    def _gen_default_labels(self, info):
        result = {"ylabel": ""}
        result["title"] = ""
        result["xlabel"] = ""
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n" + info["observations_subset"]

        return result

    def _gen_default_grid(self):
        return {"alpha": 0.5}

    def __call__(
        self, x, y, fig=None, ax=None, title=None, xlabel=None, ylabel=None, grid_params=None, **kwargs
    ):
        """
        Parameters
        ----------
        x : `np.array`
            Values to plot on the x-axis
        y : `np.array`
            Values to plot on the y axis
        fig : `matplotlib.Figure`
            Matplotlib figure to use. If None, one will be created
        ax : `matplotlib.Axes`
            Matplotlib Axes to use. If None, one will be created
        title : `str`
            String to use for plot title. If None, defaults generasted
            on init are used.
        xlabel : `str`
            String for the x-axis label. If None, defaults generasted
            on init are used.
        ylabel : `str`
            String for the y-axis label. If None, defaults generasted
            on init are used.
        grid_params : `dict`
            Dictionary of kwargs to pass to Axes.grid. If None,
            default of alpha=0.5 is used. Set to any non-dict value
            to turn off grid.
        **kwargs
            Additional kwargs passed through to matplotlib.Axes.plot

        Returns
        -------
        matplotlib.Figure
        """

        if fig is None:
            fig, ax = plt.subplots()

        if grid_params is None:
            grid_params = self._gen_default_grid()

        overrides = {"title": title, "xlabel": xlabel, "ylabel": ylabel}
        plot_dict = copy.copy(self.generated_plot_dict)
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        ax.plot(x, y, **kwargs)

        ax.set_title(plot_dict["title"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        if isinstance(grid_params, dict):
            ax.grid(**grid_params)

        return fig


def project_lambert(longitude, latitude):
    """Project from RA,dec to plane
    https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
    """

    # flipping the sign on latitude goes north pole or south pole centered
    r_polar = 2 * np.cos((np.pi / 2 + latitude) / 2.0)
    # Add pi/2 so north is up
    theta_polar = longitude + np.pi / 2

    x = r_polar * np.cos(theta_polar)
    y = r_polar * np.sin(theta_polar)
    return x, y


def draw_grat(ax):
    """Draw some graticule lines on an axis"""
    decs = np.radians(90.0 - np.array([20, 40, 60, 80]))
    ra = np.radians(np.arange(0, 361, 1))
    for dec in decs:
        temp_dec = ra * 0 + dec
        x, y = project_lambert(ra, temp_dec)
        ax.plot(x, y, "k--", alpha=0.5)

    ras = np.radians(np.arange(0, 360 + 45, 45))
    dec = np.radians(90.0 - np.arange(0, 81, 1))
    for ra in ras:
        temp_ra = dec * 0 + ra
        x, y = project_lambert(temp_ra, dec)
        ax.plot(x, y, "k--", alpha=0.5)

    for dec in decs:
        x, y = project_lambert(np.radians(45.0), dec)
        ax.text(x, y, "%i" % np.round(np.degrees(dec)))

    return ax


class PlotLambert(BasePlot):
    """Make a Lambertian projection"""

    def __init__(self, info=None):
        self.info = info
        self.generated_plot_dict = self._gen_default_labels(info)
        self.generated_cb_dict = self._gen_default_cb(info)

    def _gen_default_cb(self, info):

        result = {"labelsize": None, "format": "%i", "label": "#"}

        if info is not None:
            if "metric: unit" in info.keys():
                result["label"] = info["metric: unit"]

        return result

    def _gen_default_labels(self, info):
        result = {}
        result["title"] = ""
        result["xlabel"] = ""
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n" + info["observations_subset"]

        return result

    def _gen_default_grid(self):
        return {"alpha": 0.5}

    def __call__(
        self,
        values_in,
        latitudes=None,
        longitudes=None,
        fig=None,
        ax=None,
        title=None,
        alt_limit=10.0,
        levels=200,
        **kwargs,
    ):

        overrides = {"title": title}
        plot_dict = copy.copy(self.generated_plot_dict)
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        if fig is None:
            fig, ax = plt.subplots()

        if latitudes is None:
            # see if we can assume this is a HEALpix
            nside = hp.npix2nside(np.size(values_in))
            longitudes, latitudes = utils.hpid2_ra_dec(nside, np.arange(np.size(values_in)))
        x, y = project_lambert(np.radians(longitudes), np.radians(latitudes))

        if np.size(levels) == 1:
            level_step = (np.nanmax(values_in) - np.nanmin(values_in)) / levels
            levels = np.arange(np.nanmin(values_in), np.nanmax(values_in) + level_step, level_step)

        # non_finite below lowest level
        mod_val = copy.copy(values_in)
        finite = np.isfinite(values_in)
        mod_val[np.invert(finite)] = np.min(levels) - 1
        tcf = ax.tricontourf(
            x,
            y,
            mod_val,
            levels,
            **kwargs,
        )
        tcf.set_edgecolors("face")

        ax = draw_grat(ax)

        ax.set_xticks([])
        ax.set_yticks([])
        x, y = project_lambert(0, np.radians(alt_limit))
        max_val = np.max(np.abs([x, y]))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])

        # Pop in an extra line to raise the title a bit
        ax.set_title(plot_dict["title"])

        cb_params = self.generated_cb_dict

        cb = plt.colorbar(tcf, format=cb_params["format"])
        cb.set_label(cb_params["label"])
        if cb_params["labelsize"] is not None:
            cb.ax.tick_params(labelsize=cb_params["labelsize"])

        # If outputing to PDF, this fixes the colorbar white stripes
        cb.solids.set_edgecolor("face")

        return fig


class PlotFo(BasePlot):
    def __init__(self, info=None, scale=1000):
        self.info = info
        self.scale = scale
        self.generated_plot_dict = self._gen_default_labels(info)

    def _gen_default_labels(self, info):
        result = {"ylabel": "Area (%is of square degrees)" % self.scale}
        result["title"] = ""
        result["xlabel"] = "Number of Visits"
        if info is not None:
            if "run_name" in info.keys():
                result["title"] = info["run_name"]
            else:
                result["title"] = ""
            if "observations_subset" in info.keys():
                result["title"] += "\n" + info["observations_subset"]

        return result

    def __call__(self, nvisits_hparray, fig=None, ax=None, title=None, xlabel=None, ylabel=None, 
                 n_visits=750, asky=18000, reflinewidth=2, linewidth=3, color='k', 
                 xmin=0, xmax=1000, **kwargs):
        """
        Parameters
        ----------
        nvisits_hparray : `np.array`
            Healpix array with the number of visits per HEALpix.
        nvisits : `int`
            The number of visits to consider as the threshold for FO.
        asky : `float`
            Area of sky to use when calculating FO. Default 18000 (sq degrees)
        """
        
        pix_area = hp.nside2pixarea(hp.npix2nside(np.size(nvisits_hparray)), degrees=True)
        nvisits_hparray_finite = nvisits_hparray[np.isfinite(nvisits_hparray)]
        order = np.argsort(nvisits_hparray_finite)
        cumulative_area = np.arange(1, order.size + 1, 1) * pix_area

        if fig is None:
            fig, ax = plt.subplots()

        overrides = {"title": title, "xlabel": xlabel, "ylabel": ylabel}
        plot_dict = copy.copy(self.generated_plot_dict)
        for key in overrides:
            if overrides[key] is not None:
                plot_dict[key] = overrides[key]

        # Median number of visits in the top area
        fo_dict = fO_calcs(nvisits_hparray, asky=asky, n_visit=n_visits)
        nvis_median = fo_dict["Median N visits in top area"]
        f_o_area = fo_dict["Area above %i (sq deg)" % n_visits]

        ax.plot(nvisits_hparray_finite[order[::-1]], cumulative_area / self.scale, linewidth=linewidth,
                color=color, **kwargs)

        ax.set_title(plot_dict["title"])
        ax.set_xlabel(plot_dict["xlabel"])
        ax.set_ylabel(plot_dict["ylabel"])
        ax.set_xlim([xmin, xmax])

        ax.axvline(x=n_visits, linewidth=reflinewidth, color="b", linestyle=":")
        ax.axhline(y=asky / self.scale, linewidth=reflinewidth, color="r", linestyle=":")

        # Add in the reference lines.
        ax.axvline(
            x=nvis_median,
            linewidth=reflinewidth,
            color="b",
            alpha=0.5,
            linestyle="-",
            label=f"f$_0$ Med. Nvis. (@ {asky/1000 :.0f}K sq deg) = {nvis_median :.0f} visits",
        )

        ax.axhline(
            y=f_o_area / self.scale,
            linewidth=reflinewidth,
            color="r",
            alpha=0.5,
            linestyle="-",
            label=f"f$_0$ Area (@ {n_visits :.0f} visits) = {f_o_area/1000 :.01f}K sq deg",
        )

        ax.legend(loc="upper right", fontsize="small", numpoints=1, framealpha=1.0)

        return fig
