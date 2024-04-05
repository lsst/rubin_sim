__all__ = (
    "set_color_lims",
    "set_color_map",
    "HealpixSkyMap",
    "HealpixPowerSpectrum",
    "HealpixHistogram",
    "BaseHistogram",
    "BaseSkyMap",
    "HealpixSDSSSkyMap",
    "LambertSkyMap",
)

import copy
import numbers
import warnings

import healpy as hp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import colors, ticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from rubin_scheduler.utils import _healbin

from rubin_sim.maf.utils import optimal_bins, percentile_clipping

from .perceptual_rainbow import make_pr_cmap
from .plot_handler import BasePlotter, apply_zp_norm

perceptual_rainbow = make_pr_cmap()

base_default_plot_dict = {
    "title": None,
    "xlabel": None,
    "label": None,
    "log_scale": False,
    "percentile_clip": None,
    "norm_val": None,
    "zp": None,
    "cbar_orientation": "horizontal",
    "cbar_format": None,
    "cmap": perceptual_rainbow,
    "cbar_edge": True,
    "n_ticks": 10,
    "color_min": None,
    "color_max": None,
    "extend": "neither",
    "x_min": None,
    "x_max": None,
    "y_min": None,
    "y_max": None,
    "labelsize": None,
    "fontsize": None,
    "figsize": None,
    "subplot": 111,
    "mask_below": None,
}


def set_color_lims(metric_value, plot_dict, key_min="color_min", key_max="color_max"):
    """Set up x or color bar limits."""
    # Use plotdict values if available
    color_min = plot_dict[key_min]
    color_max = plot_dict[key_max]
    # If either is not set and we have data:
    if color_min is None or color_max is None:
        if np.size(metric_value.compressed()) > 0:
            # is percentile clipping set?
            if plot_dict["percentile_clip"] is not None:
                pc_min, pc_max = percentile_clipping(
                    metric_value.compressed(), percentile=plot_dict["percentile_clip"]
                )
                tempcolor_min = pc_min
                tempcolor_max = pc_max
            # If not, just use the data limits.
            else:
                tempcolor_min = metric_value.compressed().min()
                tempcolor_max = metric_value.compressed().max()
            # But make sure there is some range on the colorbar
            if tempcolor_min == tempcolor_max:
                tempcolor_min = tempcolor_min - 0.5
                tempcolor_max = tempcolor_max + 0.5
            tempcolor_min, tempcolor_max = np.sort([tempcolor_min, tempcolor_max])
        else:
            # There is no metric data to plot, but here we are.
            tempcolor_min = 0
            tempcolor_max = 1
    if color_min is None:
        color_min = tempcolor_min
    if color_max is None:
        color_max = tempcolor_max
    return [color_min, color_max]


def set_color_map(plot_dict):
    cmap = plot_dict["cmap"]
    if cmap is None:
        cmap = "perceptual_rainbow"
    if isinstance(cmap, str):
        cmap = getattr(cm, cmap)
    # Set background and masked pixel colors default healpy white and gray.
    cmap = copy.copy(cmap)
    cmap.set_over(cmap(1.0))
    cmap.set_under("w")
    cmap.set_bad("gray")
    return cmap


class HealpixSkyMap(BasePlotter):
    """
    Generate a sky map of healpix metric values using healpy's mollweide view.
    """

    def __init__(self):
        super(HealpixSkyMap, self).__init__()
        # Set the plot_type
        self.plot_type = "SkyMap"
        self.object_plotter = False
        # Set up the default plotting parameters.
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update(
            {
                "rot": (0, 0, 0),
                "flip": "astro",
                "coord": "C",
                "nside": 8,
                "reduce_func": np.mean,
                "visufunc": hp.mollview,
            }
        )
        # Note: for alt/az sky maps using the healpix plotter, you can use
        # {'rot': (90, 90, 90), 'flip': 'geo'}
        self.healpy_visufunc_params = {}
        self.ax = None
        self.im = None

    def __call__(self, metric_value_in, slicer, user_plot_dict, fig=None):
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
        # Override the default plotting parameters with user specified values.
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        self.healpy_visufunc = plot_dict["visufunc"]

        # Check if we have a valid HEALpix slicer
        if "Heal" in slicer.slicer_name:
            # Update the metric data with zeropoint or normalization.
            metric_value = apply_zp_norm(metric_value_in, plot_dict)
        else:
            # Bin the values up on a healpix grid.
            metric_value = _healbin(
                slicer.slice_points["ra"],
                slicer.slice_points["dec"],
                metric_value_in.filled(slicer.badval),
                nside=plot_dict["nside"],
                reduce_func=plot_dict["reduce_func"],
                fill_val=slicer.badval,
            )
            mask = np.zeros(metric_value.size)
            mask[np.where(metric_value == slicer.badval)] = 1

            metric_value = ma.array(metric_value, mask=mask)
            metric_value = apply_zp_norm(metric_value, plot_dict)

        if plot_dict["mask_below"] is not None:
            to_mask = np.where(metric_value <= plot_dict["mask_below"])[0]
            metric_value.mask[to_mask] = True
            badval = hp.UNSEEN
        else:
            badval = slicer.badval

        # Generate a full-sky plot.
        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        # Set up color bar limits.
        clims = set_color_lims(metric_value, plot_dict)
        cmap = set_color_map(plot_dict)
        # Set log scale?
        norm = None
        if plot_dict["log_scale"]:
            norm = "log"
        # Avoid trying to log scale when zero is in the range.
        if (norm == "log") & ((clims[0] <= 0 <= clims[1]) or (clims[0] >= 0 >= clims[1])):
            # Try something simple
            above = metric_value[np.where(metric_value > 0)]
            if len(above) > 0:
                clims[0] = above.max()
            # If still bad, give up and turn off norm
            if (clims[0] <= 0 <= clims[1]) or (clims[0] >= 0 >= clims[1]):
                norm = None
            warnings.warn(
                "Using norm was set to log, but color limits pass through 0. "
                "Adjusting so plotting doesn't fail"
            )
        if plot_dict["coord"] == "C":
            notext = True
        else:
            notext = False

        visufunc_params = {
            "title": plot_dict["title"],
            "cbar": False,
            "min": clims[0],
            "max": clims[1],
            "rot": plot_dict["rot"],
            "flip": plot_dict["flip"],
            "coord": plot_dict["coord"],
            "cmap": cmap,
            "norm": norm,
            "sub": plot_dict["subplot"],
            "fig": fig.number,
            "notext": notext,
        }
        # Keys to specify only if present in plot_dict
        for key in (
            "reso",
            "xsize",
            "lamb",
            "reuse_axes",
            "alpha",
            "badcolor",
            "bgcolor",
        ):
            if key in plot_dict:
                visufunc_params[key] = plot_dict[key]

        visufunc_params.update(self.healpy_visufunc_params)
        self.healpy_visufunc(metric_value.filled(badval), **visufunc_params)

        # Add colorbar
        # (not using healpy default colorbar because we want more tickmarks).
        self.ax = plt.gca()
        im = self.ax.get_images()[0]

        # Add a graticule (grid) over the globe.
        if "noGraticule" not in plot_dict:
            hp.graticule(dpar=30, dmer=30)

        # Add label.
        if plot_dict["label"] is not None:
            plt.figtext(0.8, 0.8, "%s" % (plot_dict["label"]))
        # Make a color bar. Suppress excessive colorbar warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The vertical colorbar is primarily aimed at the movie
            # but may be useful for other purposes
            if plot_dict["extend"] != "neither":
                extendrect = False
            else:
                extendrect = True
            if plot_dict["cbar_orientation"].lower() == "vertical":
                cb = plt.colorbar(
                    im,
                    shrink=0.5,
                    extendrect=extendrect,
                    extend=plot_dict["extend"],
                    location="right",
                    format=plot_dict["cbar_format"],
                )
            else:
                # Most of the time we just want a standard horizontal colorbar
                cb = plt.colorbar(
                    im,
                    shrink=0.75,
                    aspect=25,
                    pad=0.1,
                    orientation="horizontal",
                    format=plot_dict["cbar_format"],
                    extendrect=extendrect,
                    extend=plot_dict["extend"],
                )
            cb.set_label(plot_dict["xlabel"], fontsize=plot_dict["fontsize"])
            if plot_dict["labelsize"] is not None:
                cb.ax.tick_params(labelsize=plot_dict["labelsize"])
            if norm == "log":
                tick_locator = ticker.LogLocator(numticks=plot_dict["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
            if (plot_dict["n_ticks"] is not None) & (norm != "log"):
                tick_locator = ticker.MaxNLocator(nbins=plot_dict["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if plot_dict["cbar_edge"]:
            cb.solids.set_edgecolor("face")
        return fig


class HealpixPowerSpectrum(BasePlotter):
    def __init__(self):
        self.plot_type = "PowerSpectrum"
        self.object_plotter = False
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update({"maxl": None, "removeDipole": True, "linestyle": "-"})

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        """Generate and plot the power spectrum of metric_values
        (for metrics calculated on a healpix grid).
        """
        if "Healpix" not in slicer.slicer_name:
            raise ValueError("HealpixPowerSpectrum for use with healpix metricBundles.")
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        if plot_dict["subplot"] != "111":
            fig.add_subplot(plot_dict["subplot"])
        # If the mask is True everywhere (no data), just plot zeros
        if False not in metric_value.mask:
            return None
        if plot_dict["removeDipole"]:
            cl = hp.anafast(
                hp.remove_dipole(metric_value.filled(hp.UNSEEN)),
                lmax=plot_dict["maxl"],
            )
        else:
            cl = hp.anafast(metric_value.filled(hp.UNSEEN), lmax=plot_dict["maxl"])
        ell = np.arange(np.size(cl))
        if plot_dict["removeDipole"]:
            condition = ell > 1
        else:
            condition = ell > 0
        ell = ell[condition]
        cl = cl[condition]
        # Plot the results.
        plt.plot(
            ell,
            (cl * ell * (ell + 1)) / 2.0 / np.pi,
            color=plot_dict["color"],
            linestyle=plot_dict["linestyle"],
            label=plot_dict["label"],
        )
        if cl.max() > 0 and plot_dict["log_scale"]:
            plt.yscale("log")
        plt.xlabel(r"$l$", fontsize=plot_dict["fontsize"])
        plt.ylabel(r"$l(l+1)C_l/(2\pi)$", fontsize=plot_dict["fontsize"])
        if plot_dict["labelsize"] is not None:
            plt.tick_params(axis="x", labelsize=plot_dict["labelsize"])
            plt.tick_params(axis="y", labelsize=plot_dict["labelsize"])
        if plot_dict["title"] is not None:
            plt.title(plot_dict["title"])
        # Return figure number
        # (so we can reuse/add onto/save this figure if desired).
        return fig


class HealpixHistogram(BasePlotter):
    def __init__(self):
        self.plot_type = "Histogram"
        self.object_plotter = False
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update(
            {
                "ylabel": "Area (1000s of square degrees)",
                "bins": None,
                "bin_size": None,
                "cumulative": False,
                "scale": None,
                "linestyle": "-",
            }
        )
        self.base_hist = BaseHistogram()

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        """Histogram metric_value for all healpix points."""
        if "Healpix" not in slicer.slicer_name:
            raise ValueError("HealpixHistogram is for use with healpix slicer.")
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        if plot_dict["scale"] is None:
            plot_dict["scale"] = hp.nside2pixarea(slicer.nside, degrees=True) / 1000.0
        fig = self.base_hist(metric_value, slicer, plot_dict, fig=fig)
        return fig


class BaseHistogram(BasePlotter):
    def __init__(self):
        self.plot_type = "Histogram"
        self.object_plotter = False
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update(
            {
                "ylabel": "Count",
                "bins": None,
                "bin_size": None,
                "cumulative": False,
                "scale": 1.0,
                "yaxisformat": "%.3f",
                "linestyle": "-",
            }
        )

    def __call__(self, metric_value_in, slicer, user_plot_dict, fig=None):
        """
        Plot a histogram of metric_values
        (such as would come from a spatial slicer).
        """
        # Adjust metric values by zeropoint or norm_val,
        # and use 'compressed' version of masked array.
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        metric_value = apply_zp_norm(metric_value_in, plot_dict)
        # Toss any NaNs or infs
        metric_value = metric_value[np.isfinite(metric_value)]
        x_min, x_max = set_color_lims(metric_value, plot_dict, key_min="x_min", key_max="x_max")
        metric_value = metric_value.compressed()
        # Set up the bins for the histogram.
        # User specified 'bins' overrides 'bin_size'.
        # Note that 'bins' could be a single number or an array,
        # simply passed to plt.histogram.
        if plot_dict["bins"] is not None:
            bins = plot_dict["bins"]
        elif plot_dict["bin_size"] is not None:
            #  If generating a cumulative histogram,
            #  want to use full range of data (but with given bin_size).
            #  but if user set histRange to be wider than full range of data,
            #  then extend bins to cover this range,
            #  so we can make prettier plots.
            if plot_dict["cumulative"]:
                # Potentially, expand the range for the cumulative histogram.
                bmin = np.min([metric_value.min(), x_min])
                bmax = np.max([metric_value.max(), x_max])
                bins = np.arange(bmin, bmax + plot_dict["bin_size"] / 2.0, plot_dict["bin_size"])
            #  Otherwise, not cumulative so just use metric values,
            #  without potential expansion.
            else:
                bins = np.arange(
                    x_min,
                    x_max + plot_dict["bin_size"] / 2.0,
                    plot_dict["bin_size"],
                )
            # Catch edge-case where there is only 1 bin value
            if bins.size < 2:
                bins = np.arange(
                    bins.min() - plot_dict["bin_size"] * 2.0,
                    bins.max() + plot_dict["bin_size"] * 2.0,
                    plot_dict["bin_size"],
                )
        else:
            # If user did not specify bins or bin_size,
            # then we try to figure out a good number of bins.
            bins = optimal_bins(metric_value)
        # Generate plots.
        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        if (
            plot_dict["subplot"] != 111
            and plot_dict["subplot"] != (1, 1, 1)
            and plot_dict["subplot"] is not None
        ):
            ax = fig.add_subplot(plot_dict["subplot"])
        else:
            ax = plt.gca()
        # Check if any data falls within histRange,
        # because otherwise histogram generation will fail.
        if isinstance(bins, np.ndarray):
            condition = (metric_value >= bins.min()) & (metric_value <= bins.max())
        else:
            condition = (metric_value >= x_min) & (metric_value <= x_max)
        plot_value = metric_value[condition]
        if len(plot_value) == 0:
            # No data is within histRange/bins.
            # So let's just make a simple histogram anyway.
            n, b, p = plt.hist(
                metric_value,
                bins=50,
                histtype="step",
                cumulative=plot_dict["cumulative"],
                log=plot_dict["log_scale"],
                label=plot_dict["label"],
                color=plot_dict["color"],
            )
        else:
            # There is data to plot, and we've already ensured
            # x_min/x_max/bins are more than single value.
            n, b, p = plt.hist(
                metric_value,
                bins=bins,
                range=[x_min, x_max],
                histtype="step",
                log=plot_dict["log_scale"],
                cumulative=plot_dict["cumulative"],
                label=plot_dict["label"],
                color=plot_dict["color"],
            )
        hist_ylims = plt.ylim()
        if n.max() > hist_ylims[1]:
            plt.ylim(top=n.max())
        if n.min() < hist_ylims[0] and not plot_dict["log_scale"]:
            plt.ylim(bottom=n.min())
        # Fill in axes labels and limits.
        # Option to use 'scale' to turn y axis into area or other value.

        def mjr_formatter(y, pos):
            if not isinstance(plot_dict["scale"], numbers.Number):
                raise ValueError('plot_dict["scale"] must be a number to scale the y axis.')
            return plot_dict["yaxisformat"] % (y * plot_dict["scale"])

        ax.yaxis.set_major_formatter(FuncFormatter(mjr_formatter))
        # Set optional x, y limits.
        if "x_min" in plot_dict:
            plt.xlim(left=plot_dict["x_min"])
        if "x_max" in plot_dict:
            plt.xlim(right=plot_dict["x_max"])
        if "y_min" in plot_dict:
            plt.ylim(bottom=plot_dict["y_min"])
        if "y_max" in plot_dict:
            plt.ylim(top=plot_dict["y_max"])
        # Set/Add various labels.
        plt.xlabel(plot_dict["xlabel"], fontsize=plot_dict["fontsize"])
        plt.ylabel(plot_dict["ylabel"], fontsize=plot_dict["fontsize"])
        plt.title(plot_dict["title"])
        if plot_dict["labelsize"] is not None:
            plt.tick_params(axis="x", labelsize=plot_dict["labelsize"])
            plt.tick_params(axis="y", labelsize=plot_dict["labelsize"])
        # Return figure
        return fig


class BaseSkyMap(BasePlotter):
    def __init__(self):
        self.plot_type = "SkyMap"
        self.object_plotter = False  # unless 'metricIsColor' is true..
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update(
            {
                "projection": "aitoff",
                "radius": np.radians(1.75),
                "alpha": 1.0,
                "plotMask": False,
                "metricIsColor": False,
                "cbar": True,
                "raCen": 0.0,
                "mwZone": True,
                "bgcolor": "gray",
            }
        )

    def _plot_tissot_ellipse(self, lon, lat, radius, **kwargs):
        """Plot Tissot Ellipse/Tissot Indicatrix

        Parameters
        ----------
        lon : `float` or array_like
            longitude-like of ellipse centers (radians)
        lat : `float` or array_like
            latitude-like of ellipse centers (radians)
        radius : `float` or array_like
            radius of ellipses (radians)
        **kwargs : `dict`
            Keyword argument which will be passed to
            `matplotlib.patches.Ellipse`.

        Returns
        -------
        ellipses : `list` [ `matplotlib.patches.Ellipse` ]
            List of ellipses to add to the plot.

        # The code in this method adapted from astroML, which is BSD-licensed.
        # See http: //github.com/astroML/astroML for details.
        """
        # Code adapted from astroML, which is BSD-licensed.
        # See http: //github.com/astroML/astroML for details.
        ellipses = []
        for ll, bb, diam in np.broadcast(lon, lat, radius * 2.0):
            el = Ellipse((ll, bb), diam / np.cos(bb), diam, **kwargs)
            ellipses.append(el)
        return ellipses

    def _plot_ecliptic(self, ra_cen=0, ax=None):
        """
        Plot a red line at location of ecliptic.
        """
        if ax is None:
            ax = plt.gca()
        ecinc = 23.439291 * (np.pi / 180.0)
        ra_ec = np.arange(0, np.pi * 2.0, (np.pi * 2.0 / 360.0))
        dec_ec = np.sin(ra_ec) * ecinc
        lon = -(ra_ec - ra_cen - np.pi) % (np.pi * 2) - np.pi
        ax.plot(lon, dec_ec, "r.", markersize=1.8, alpha=0.4)

    def _plot_mw_zone(
        self,
        ra_cen=0,
        peak_width=np.radians(10.0),
        taper_length=np.radians(80.0),
        ax=None,
    ):
        """
        Plot blue lines to mark the milky way galactic exclusion zone.
        """
        if ax is None:
            ax = plt.gca()
        # Calculate galactic coordinates for mw location.
        step = 0.02
        gal_l = np.arange(-np.pi, np.pi + step / 2.0, step)
        val = peak_width * np.cos(gal_l / taper_length * np.pi / 2.0)
        gal_b1 = np.where(np.abs(gal_l) <= taper_length, val, 0)
        gal_b2 = np.where(np.abs(gal_l) <= taper_length, -val, 0)
        # Convert to ra/dec.
        # Convert to lon/lat and plot.
        c = SkyCoord(l=gal_l * u.rad, b=gal_b1 * u.rad, frame="galactic").transform_to("icrs")
        ra = c.ra.rad
        dec = c.dec.rad
        lon = -(ra - ra_cen - np.pi) % (np.pi * 2) - np.pi
        ax.plot(lon, dec, "b.", markersize=1.8, alpha=0.4)
        c = SkyCoord(l=gal_l * u.rad, b=gal_b2 * u.rad, frame="galactic").transform_to("icrs")
        ra = c.ra.rad
        dec = c.dec.rad
        lon = -(ra - ra_cen - np.pi) % (np.pi * 2) - np.pi
        ax.plot(lon, dec, "b.", markersize=1.8, alpha=0.4)

    def __call__(self, metric_value_in, slicer, user_plot_dict, fig=None):
        """
        Plot the sky map of metric_value for a generic spatial slicer.
        """
        if "ra" not in slicer.slice_points or "dec" not in slicer.slice_points:
            err_message = 'SpatialSlicer must contain "ra" and "dec" in slice_points metadata.'
            err_message += " SlicePoints only contains keys %s." % (slicer.slice_points.keys())
            raise ValueError(err_message)
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        metric_value = apply_zp_norm(metric_value_in, plot_dict)

        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        # other projections available include
        # ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
        ax = fig.add_subplot(plot_dict["subplot"], projection=plot_dict["projection"])
        # Set up valid datapoints and color_min/max values.
        if plot_dict["plotMask"]:
            # Plot all data points.
            good = np.ones(len(metric_value), dtype="bool")
        else:
            # Only plot points which are not masked.
            # Flip numpy ma mask where 'False' == 'good'.
            good = ~metric_value.mask

        # Add ellipses at RA/Dec locations - but don't add colors yet.
        lon = -(slicer.slice_points["ra"][good] - plot_dict["raCen"] - np.pi) % (np.pi * 2) - np.pi
        ellipses = self._plot_tissot_ellipse(
            lon,
            slicer.slice_points["dec"][good],
            plot_dict["radius"],
            rasterized=True,
        )
        if plot_dict["metricIsColor"]:
            current = None
            for ellipse, mVal in zip(ellipses, metric_value.data[good]):
                if mVal[3] > 1:
                    ellipse.set_alpha(1.0)
                    ellipse.set_facecolor((mVal[0], mVal[1], mVal[2]))
                    ellipse.set_edgecolor("k")
                    current = ellipse
                else:
                    ellipse.set_alpha(mVal[3])
                    ellipse.set_color((mVal[0], mVal[1], mVal[2]))
                ax.add_patch(ellipse)
            if current:
                ax.add_patch(current)
        else:
            # Determine color min/max values.
            # metricValue.compressed = non-masked points.
            clims = set_color_lims(metric_value, plot_dict)
            # Determine whether or not to use auto-log scale.
            if plot_dict["log_scale"] == "auto":
                if clims[0] > 0:
                    if np.log10(clims[1]) - np.log10(clims[0]) > 3:
                        plot_dict["log_scale"] = True
                    else:
                        plot_dict["log_scale"] = False
                else:
                    plot_dict["log_scale"] = False
            if plot_dict["log_scale"]:
                # Move min/max values to things that can be marked
                # on the colorbar.
                # clims[0] = 10 ** (int(np.log10(clims[0])))
                # clims[1] = 10 ** (int(np.log10(clims[1])))
                norml = colors.LogNorm()
                p = PatchCollection(
                    ellipses,
                    cmap=plot_dict["cmap"],
                    alpha=plot_dict["alpha"],
                    linewidth=0,
                    edgecolor=None,
                    norm=norml,
                    rasterized=True,
                )
            else:
                p = PatchCollection(
                    ellipses,
                    cmap=plot_dict["cmap"],
                    alpha=plot_dict["alpha"],
                    linewidth=0,
                    edgecolor=None,
                    rasterized=True,
                )
            p.set_array(metric_value.data[good])
            p.set_clim(clims)
            ax.add_collection(p)
            # Add color bar (with optional setting of limits)
            if plot_dict["cbar"]:
                if plot_dict["cbar_orientation"].lower() == "vertical":
                    cb = plt.colorbar(
                        p,
                        shrink=0.5,
                        extendrect=True,
                        location="right",
                        format=plot_dict["cbar_format"],
                    )
                else:
                    # Usually we just want a standard horizontal colorbar
                    cb = plt.colorbar(
                        p,
                        shrink=0.75,
                        aspect=25,
                        pad=0.1,
                        orientation="horizontal",
                        format=plot_dict["cbar_format"],
                        extendrect=True,
                    )
                # If outputing to PDF, this fixes the colorbar white stripes
                if plot_dict["cbar_edge"]:
                    cb.solids.set_edgecolor("face")
                cb.set_label(plot_dict["xlabel"], fontsize=plot_dict["fontsize"])
                cb.ax.tick_params(labelsize=plot_dict["labelsize"])
                tick_locator = ticker.MaxNLocator(nbins=plot_dict["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
        # Add ecliptic
        self._plot_ecliptic(plot_dict["raCen"], ax=ax)
        if plot_dict["mwZone"]:
            self._plot_mw_zone(plot_dict["raCen"], ax=ax)
        ax.grid(True, zorder=1)
        ax.xaxis.set_ticklabels([])
        if plot_dict["bgcolor"] is not None:
            ax.set_facecolor(plot_dict["bgcolor"])
        # Add label.
        if plot_dict["label"] is not None:
            plt.figtext(0.75, 0.9, "%s" % plot_dict["label"], fontsize=plot_dict["fontsize"])
        if plot_dict["title"] is not None:
            plt.text(
                0.5,
                1.09,
                plot_dict["title"],
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=plot_dict["fontsize"],
            )
        return fig


class HealpixSDSSSkyMap(BasePlotter):
    def __init__(self):
        self.plot_type = "SkyMap"
        self.object_plotter = False
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update(
            {
                "cbar_format": "%.2f",
                "raMin": -90,
                "raMax": 90,
                "raLen": 45,
                "decMin": -2.0,
                "decMax": 2.0,
            }
        )

    def __call__(self, metric_value_in, slicer, user_plot_dict, fig=None):
        """
        Plot the sky map of metric_value using healpy cartview plots
        in thin strips.
        raMin: Minimum RA to plot (deg)
        raMax: Max RA to plot (deg).
        Note raMin/raMax define the centers that will be plotted.
        raLen:  Length of the plotted strips in degrees
        decMin: minimum dec value to plot
        decMax: max dec value to plot
        metric_value_in: metric values
        """
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        metric_value = apply_zp_norm(metric_value_in, plot_dict)
        norm = None
        if plot_dict["log_scale"]:
            norm = "log"
        clims = set_color_lims(metric_value, plot_dict)
        cmap = set_color_map(plot_dict)
        racenters = np.arange(plot_dict["raMin"], plot_dict["raMax"], plot_dict["raLen"])
        nframes = racenters.size
        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        # Do not specify or use plot_dict['subplot']
        # because this is done in each call to hp.cartview.
        for i, racenter in enumerate(racenters):
            if i == 0:
                use_title = (
                    plot_dict["title"]
                    + " /n"
                    + "%i < RA < %i" % (racenter - plot_dict["raLen"], racenter + plot_dict["raLen"])
                )
            else:
                use_title = "%i < RA < %i" % (
                    racenter - plot_dict["raLen"],
                    racenter + plot_dict["raLen"],
                )
            hp.cartview(
                metric_value.filled(slicer.badval),
                title=use_title,
                cbar=False,
                min=clims[0],
                max=clims[1],
                flip="astro",
                rot=(racenter, 0, 0),
                cmap=cmap,
                norm=norm,
                lonra=[-plot_dict["raLen"], plot_dict["raLen"]],
                latra=[plot_dict["decMin"], plot_dict["decMax"]],
                sub=(nframes + 1, 1, i + 1),
                fig=fig,
            )
            hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar)
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.075])
        # Add label.
        if plot_dict["label"] is not None:
            plt.figtext(0.8, 0.9, "%s" % plot_dict["label"])
        # Make the colorbar as a seperate figure,
        # from http: //matplotlib.org/examples/api/colorbar_only.html
        cnorm = colors.Normalize(vmin=clims[0], vmax=clims[1])
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = mpl.colorbar.ColorbarBase(
                ax,
                cmap=cmap,
                norm=cnorm,
                orientation="horizontal",
                format=plot_dict["cbar_format"],
            )
            cb.set_label(plot_dict["xlabel"])
            cb.ax.tick_params(labelsize=plot_dict["labelsize"])
            if norm == "log":
                tick_locator = ticker.LogLocator(numticks=plot_dict["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
            if (plot_dict["n_ticks"] is not None) & (norm != "log"):
                tick_locator = ticker.MaxNLocator(nbins=plot_dict["n_ticks"])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if plot_dict["cbar_edge"]:
            cb.solids.set_edgecolor("face")
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


class LambertSkyMap(BasePlotter):
    """
    Use basemap and contour to make a Lambertian projection.
    Note that the plot_dict can include a 'basemap' key with a dictionary of
    arbitrary kwargs to use with the call to Basemap.
    """

    def __init__(self):
        self.plot_type = "SkyMap"
        self.object_plotter = False
        self.default_plot_dict = {}
        self.default_plot_dict.update(base_default_plot_dict)
        self.default_plot_dict.update(
            {
                "basemap": {
                    "projection": "nplaea",
                    "boundinglat": 1,
                    "lon_0": 180,
                    "resolution": None,
                    "celestial": False,
                    "round": False,
                },
                "levels": 200,
                "cbar_format": "%i",
                "norm": None,
            }
        )

    def __call__(self, metric_value_in, slicer, user_plot_dict, fig=None):
        if "ra" not in slicer.slice_points or "dec" not in slicer.slice_points:
            err_message = 'SpatialSlicer must contain "ra" and "dec" in slice_points metadata.'
            err_message += " SlicePoints only contains keys %s." % (slicer.slice_points.keys())
            raise ValueError(err_message)

        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)

        metric_value = apply_zp_norm(metric_value_in, plot_dict)
        clims = set_color_lims(metric_value, plot_dict)
        # Calculate the levels to use for the contour
        if np.size(plot_dict["levels"]) > 1:
            levels = plot_dict["levels"]
        else:
            step = (clims[1] - clims[0]) / plot_dict["levels"]
            levels = np.arange(clims[0], clims[1] + step, step)

        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        ax = fig.add_subplot(plot_dict["subplot"])

        x, y = project_lambert(slicer.slice_points["ra"], slicer.slice_points["dec"])
        # Contour the plot first to remove any anti-aliasing artifacts.
        # Doesn't seem to work though. See:
        # http://stackoverflow.com/questions/15822159/
        # aliasing-when-saving-matplotlib-filled-contour-plot-to-pdf-or-eps
        # tmpContour = m.contour(np.degrees(slicer.slice_points['ra']),
        #                        np.degrees(slicer.slice_points['dec']),
        #                        metric_value.filled(np.min(clims)-1), levels,
        #                        tri=True,
        #                        cmap=plot_dict['cmap'], ax=ax, latlon=True,
        #                        lw=1)

        # Set masked values to be below the lowest contour level.
        if plot_dict["norm"] == "log":
            z_val = metric_value.filled(np.min(clims) - 0.9)
            norm = colors.LogNorm(vmin=z_val.min(), vmax=z_val.max())
        else:
            norm = plot_dict["norm"]
        tcf = ax.tricontourf(
            x,
            y,
            metric_value.filled(np.min(clims) - 0.9),
            levels,
            cmap=plot_dict["cmap"],
            norm=norm,
        )

        ax = draw_grat(ax)

        ax.set_xticks([])
        ax.set_yticks([])
        alt_limit = 10.0
        x, y = project_lambert(0, np.radians(alt_limit))
        max_val = np.max(np.abs([x, y]))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])

        # Try to fix the ugly pdf contour problem
        for c in tcf.collections:
            c.set_edgecolor("face")

        cb = plt.colorbar(tcf, format=plot_dict["cbar_format"])
        cb.set_label(plot_dict["xlabel"])
        if plot_dict["labelsize"] is not None:
            cb.ax.tick_params(labelsize=plot_dict["labelsize"])
        # Pop in an extra line to raise the title a bit
        ax.set_title(plot_dict["title"] + "\n ")
        # If outputing to PDF, this fixes the colorbar white stripes
        if plot_dict["cbar_edge"]:
            cb.solids.set_edgecolor("face")
        return fig
