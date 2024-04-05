__all__ = ("MetricVsH", "MetricVsOrbit", "MetricVsOrbitPoints")


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .plot_handler import BasePlotter

# mag_sun = -27.1
# apparent r band magnitude of the sun.
# this sets the band for the magnitude limit.
# see http://www.ucolick.org/~cnaw/sun.html for apparent mags in other bands.
mag_sun = -26.74
# apparent V band magnitude of the Sun (our H mags translate to V band)
km_per_au = 1.496e8
m_per_km = 1000


class MetricVsH(BasePlotter):
    """
    Plot metric values versus H.
    Marginalize over metric values in each H bin using 'np_reduce'.
    """

    def __init__(self):
        self.plot_type = "MetricVsH"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": "H (mag)",
            "ylabel": None,
            "label": None,
            "np_reduce": None,
            "nbins": None,
            "albedo": None,
            "Hmark": None,
            "HmarkLinestyle": ":",
            "figsize": None,
        }
        self.min_hrange = 1.0

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        if "linestyle" not in user_plot_dict:
            user_plot_dict["linestyle"] = "-"
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        hvals = slicer.slice_points["H"]
        reduce_func = plot_dict["np_reduce"]
        if reduce_func is None:
            reduce_func = np.mean
        if hvals.shape[0] == 1:
            # We have a simple set of values to plot against H.
            # This may be due to running a summary metric,
            # such as completeness.
            m_vals = metric_value[0].filled()
        elif len(hvals) == slicer.shape[1]:
            # Using cloned H distribution.
            # Apply 'np_reduce' method directly to metric values,
            # and plot at matching H values.
            m_vals = reduce_func(metric_value.filled(), axis=0)
        else:
            # Probably each object has its own H value.
            hrange = hvals.max() - hvals.min()
            min_h = hvals.min()
            if hrange < self.min_hrange:
                hrange = self.min_hrange
                min_h = hvals.min() - hrange / 2.0
            nbins = plot_dict["nbins"]
            if nbins is None:
                nbins = 30
            stepsize = hrange / float(nbins)
            bins = np.arange(min_h, min_h + hrange + stepsize / 2.0, stepsize)
            # In each bin of H, calculate the 'np_reduce' value of the
            # corresponding metric_values.
            inds = np.digitize(hvals, bins)
            inds = inds - 1
            m_vals = np.zeros(len(bins), float)
            for i in range(len(bins)):
                match = metric_value[inds == i]
                if len(match) == 0:
                    m_vals[i] = slicer.badval
                else:
                    m_vals[i] = reduce_func(match.filled())
            hvals = bins
        # Plot the values.
        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        ax = plt.gca()
        ax.plot(
            hvals,
            m_vals,
            color=plot_dict["color"],
            linestyle=plot_dict["linestyle"],
            label=plot_dict["label"],
        )
        if "x_min" in plot_dict:
            ax.set_xlim(left=plot_dict["x_min"])
        if "x_max" in plot_dict:
            ax.set_xlim(right=plot_dict["x_max"])
        if "y_min" in plot_dict:
            ax.set_ylim(bottom=plot_dict["y_min"])
        if "y_max" in plot_dict:
            ax.set_ylim(top=plot_dict["y_max"])
        # Convert hvals to diameter, using 'albedo' - add these upper xticks
        albedo = plot_dict["albedo"]
        y = 1.0
        if albedo is not None:
            ax2 = ax.twiny()
            hmin, hmax = ax.get_xlim()
            dmax = 2.0 * np.sqrt(10 ** ((mag_sun - hmin - 2.5 * np.log10(albedo)) / 2.5))
            dmin = 2.0 * np.sqrt(10 ** ((mag_sun - hmax - 2.5 * np.log10(albedo)) / 2.5))
            dmax = dmax * km_per_au * m_per_km
            dmin = dmin * km_per_au * m_per_km
            ax2.set_xlim(dmax, dmin)
            ax2.set_xscale("log")
            dmid = (dmax - dmin) / 2 + dmin
            dmid = np.power(10, round(np.log10(dmid)))
            dmin = np.power(10, round(np.log10(dmin)))
            dmax = np.power(10, round(np.log10(dmax)))
            dticks = np.array([dmin, dmid, dmax])
            ax2.set_xticks(dticks)
            ax2.set_xlabel("D (m)", labelpad=-5, horizontalalignment="right")
            ax2.grid(False)
            plt.sca(ax)
            y = 1.1
        plt.grid(True)
        if plot_dict["Hmark"] is not None:
            plt.axvline(
                x=plot_dict["Hmark"],
                color="r",
                linestyle=plot_dict["HmarkLinestyle"],
                alpha=0.3,
            )
        plt.title(plot_dict["title"], y=y)
        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        plt.tight_layout()
        return fig


class MetricVsOrbit(BasePlotter):
    """
    Plot metric values (at a particular H value) vs. orbital parameters.
    Marginalize over metric values in each orbital bin using 'np_reduce'.
    """

    def __init__(self, xaxis="q", yaxis="e"):
        self.plot_type = "MetricVsOrbit_%s%s" % (xaxis, yaxis)
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": xaxis,
            "ylabel": yaxis,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "label": None,
            "cmap": cm.viridis,
            "np_reduce": None,
            "nxbins": None,
            "nybins": None,
            "levels": None,
            "h_val": None,
            "Hwidth": None,
            "figsize": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        xvals = slicer.slice_points["orbits"][plot_dict["xaxis"]]
        yvals = slicer.slice_points["orbits"][plot_dict["yaxis"]]
        # Set x/y bins.
        nxbins = plot_dict["nxbins"]
        nybins = plot_dict["nybins"]
        if nxbins is None:
            nxbins = 100
        if nybins is None:
            nybins = 100
        if "xbins" in plot_dict:
            xbins = plot_dict["xbins"]
        else:
            xbinsize = (xvals.max() - xvals.min()) / float(nxbins)
            xbins = np.arange(xvals.min(), xvals.max() + xbinsize / 2.0, xbinsize)
        if "ybins" in plot_dict:
            ybins = plot_dict["ybins"]
        else:
            ybinsize = (yvals.max() - yvals.min()) / float(nybins)
            ybins = np.arange(yvals.min(), yvals.max() + ybinsize / 2.0, ybinsize)
        nxbins = len(xbins)
        nybins = len(ybins)
        # Identify the relevant metric_values for the Hvalue we want to plot.
        hvals = slicer.slice_points["H"]
        hwidth = plot_dict["hwidth"]
        if hwidth is None:
            hwidth = 1.0
        if len(hvals) == slicer.shape[1]:
            if plot_dict["hval"] is None:
                hidx = int(len(hvals) / 2)
                hval = hvals[hidx]
            else:
                hval = plot_dict["hval"]
                hidx = np.where(np.abs(hvals - hval) == np.abs(hvals - hval).min())[0]
                hidx = hidx[0]
        else:
            if plot_dict["hval"] is None:
                hval = np.median(hvals)
                hidx = np.where(np.abs(hvals - hval) <= hwidth / 2.0)[0]
            else:
                hval = plot_dict["hvals"]
                hidx = np.where(np.abs(hvals - hval) <= hwidth / 2.0)[0]
        if len(hvals) == slicer.shape[1]:
            m_vals = np.swapaxes(metric_value, 1, 0)[hidx].filled()
        else:
            m_vals = metric_value[hidx].filled()
        # Calculate the np_reduce'd metric values at each x/y bin.
        if "color_min" in plot_dict:
            badval = plot_dict["color_min"] - 1
        else:
            badval = slicer.badval
        binvals = np.zeros((nybins, nxbins), dtype="float") + badval
        xidxs = np.digitize(xvals, xbins) - 1
        yidxs = np.digitize(yvals, ybins) - 1
        reduce_func = plot_dict["np_reduce"]
        if reduce_func is None:
            reduce_func = np.mean
        for iy in range(nybins):
            ymatch = np.where(yidxs == iy)[0]
            for ix in range(nxbins):
                xmatch = np.where(xidxs[ymatch] == ix)[0]
                match_vals = m_vals[ymatch][xmatch]
                if len(match_vals) > 0:
                    binvals[iy][ix] = reduce_func(match_vals)
        xi, yi = np.meshgrid(xbins, ybins)
        if "color_min" in plot_dict:
            v_min = plot_dict["color_min"]
        else:
            v_min = binvals.min()
        if "color_max" in plot_dict:
            v_max = plot_dict["color_max"]
        else:
            v_max = binvals.max()
        nlevels = plot_dict["levels"]
        if nlevels is None:
            nlevels = 200
        levels = np.arange(v_min, v_max, (v_max - v_min) / float(nlevels))
        plt.contourf(xi, yi, binvals, levels, extend="max", zorder=0, cmap=plot_dict["cmap"])
        cbar = plt.colorbar()
        label = plot_dict["label"]
        if label is None:
            label = ""
        cbar.set_label(label + " @ H=%.1f" % (hval))
        plt.title(plot_dict["title"])
        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        return fig


class MetricVsOrbitPoints(BasePlotter):
    """
    Plot metric values (at a particular H value) as function
    of orbital parameters, using points for each metric value.
    """

    def __init__(self, xaxis="q", yaxis="e"):
        self.plot_type = "MetricVsOrbit"
        self.object_plotter = False
        self.default_plot_dict = {
            "title": None,
            "xlabel": xaxis,
            "ylabel": yaxis,
            "label": None,
            "cmap": cm.viridis,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "h_val": None,
            "Hwidth": None,
            "foregroundPoints": True,
            "backgroundPoints": False,
            "figsize": None,
        }

    def __call__(self, metric_value, slicer, user_plot_dict, fig=None):
        plot_dict = {}
        plot_dict.update(self.default_plot_dict)
        plot_dict.update(user_plot_dict)
        if fig is None:
            fig = plt.figure(figsize=plot_dict["figsize"])
        xvals = slicer.slice_points["orbits"][plot_dict["xaxis"]]
        yvals = slicer.slice_points["orbits"][plot_dict["yaxis"]]
        # Identify the relevant metric_values for the Hvalue we want to plot.
        hvals = slicer.slice_points["H"]
        hwidth = plot_dict["hwidth"]
        if hwidth is None:
            hwidth = 1.0
        if len(hvals) == slicer.shape[1]:
            if plot_dict["hval"] is None:
                hidx = int(len(hvals) / 2)
                hval = hvals[hidx]
            else:
                hval = plot_dict["hval"]
                hidx = np.where(np.abs(hvals - hval) == np.abs(hvals - hval).min())[0]
                hidx = hidx[0]
        else:
            if plot_dict["hval"] is None:
                hval = np.median(hvals)
                hidx = np.where(np.abs(hvals - hval) <= hwidth / 2.0)[0]
            else:
                hval = plot_dict["hvals"]
                hidx = np.where(np.abs(hvals - hval) <= hwidth / 2.0)[0]
        if len(hvals) == slicer.shape[1]:
            m_vals = np.swapaxes(metric_value, 1, 0)[hidx]
        else:
            m_vals = metric_value[hidx]
        if "color_min" in plot_dict:
            v_min = plot_dict["color_min"]
        else:
            v_min = m_vals.min()
        if "color_max" in plot_dict:
            v_max = plot_dict["color_max"]
        else:
            v_max = m_vals.max()
        if plot_dict["backgroundPoints"]:
            # This isn't quite right for the condition .. but will do for now.
            condition = np.where(m_vals == 0)
            plt.plot(
                xvals[condition],
                yvals[condition],
                "r.",
                markersize=4,
                alpha=0.5,
                zorder=3,
            )
        if plot_dict["foregroundPoints"]:
            plt.scatter(
                xvals,
                yvals,
                c=m_vals,
                vmin=v_min,
                vmax=v_max,
                cmap=plot_dict["cmap"],
                s=15,
                alpha=0.8,
                zorder=0,
            )
            cbar = plt.colorbar()
            label = plot_dict["label"]
            if label is None:
                label = ""
        cbar.set_label(label + " @ H=%.1f" % (hval))
        plt.title(plot_dict["title"])
        plt.xlabel(plot_dict["xlabel"])
        plt.ylabel(plot_dict["ylabel"])
        return fig
