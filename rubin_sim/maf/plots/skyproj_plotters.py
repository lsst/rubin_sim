import abc
import warnings
from collections import defaultdict
from numbers import Integral
from weakref import WeakKeyDictionary

import astropy.units as u
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import skyproj
from astropy.coordinates import SkyCoord
from rubin_scheduler.utils import _healbin

from .plot_handler import BasePlotter, apply_zp_norm
from .spatial_plotters import set_color_lims, set_color_map


def compute_circle_points(
    center_ra,
    center_decl,
    radius=90.0,
    start_bear=0,
    end_bear=360,
    step=1,
):
    """Create points along a circle or arc on a sphere

    Parameters
    ----------
    center_ra : `float`
        R.A. of the center of the circle (deg.).
    center_decl : `float`
        Decl. of the center of the circle (deg.).
    radius : float, optional
        Radius of the circle (deg.), by default 90.0
    start_bear : int, optional
        Bearing (E. of N.) of the start of the circle (deg.), by default 0
    end_bear : int, optional
        Bearing (E. of N.) of the end of the circle (deg.), by default 360
    step : int, optional
        Spacing of the points along the circle (deg.), by default 1

    Returns
    -------
    circle : `pandas.DataFrame`
        DataFrame with points in the circle.
    """
    ras = []
    decls = []

    bearing_angles = np.arange(start_bear, end_bear + step, step) * u.deg
    center_coords = SkyCoord(center_ra * u.deg, center_decl * u.deg)
    circle_coords = center_coords.directional_offset_by(bearing_angles, radius * u.deg)
    bearings = bearing_angles.value.tolist()
    ras = circle_coords.ra.deg.tolist()
    decls = circle_coords.dec.deg.tolist()

    # de-normalize RA so there are no large jumps, which can
    # confuse cartopy + matplotlib

    previous_ra = ras[0]
    for ra_index in range(1, len(ras)):
        if ras[ra_index] - previous_ra > 180:
            ras[ra_index] -= 360
        if ras[ra_index] - previous_ra < -180:
            ras[ra_index] += 360
        previous_ra = ras[ra_index]

    circle = pd.DataFrame(
        {
            "bearing": bearings,
            "ra": ras,
            "decl": decls,
        }
    ).set_index("bearing")

    return circle


class SkyprojPlotter(BasePlotter, abc.ABC):
    """Base class for all plotters that use the skyproj module."""

    # Store a mapping between figures and existing SkyprojPlotters, but use
    # weak keys so that the figure can be cleaned up by the garbage collector,
    # and when it is entry in the dictionary automatically deleted.
    skyproj_instances = WeakKeyDictionary()

    default_decorations = ["ecliptic", "galactic_plane"]

    default_colorbar_kwargs = {
        "location": "bottom",
        "shrink": 0.75,
        "aspect": 25,
        "pad": 0.1,
        "orientation": "horizontal",
    }

    ecliptic_kwargs = {"edgecolor": "green"}
    galactic_plane_kwargs = {"edgecolor": "blue"}
    sun_kwargs = {"color": "yellow"}
    moon_kwargs = {"color": "orange"}
    horizon_kwargs = {"edgecolor": "black", "linewidth": 3}

    def __init__(self):
        super().__init__()
        # Customize our plotters members for our new plot

        self.plot_type = "Skyproj"
        self.default_plot_dict = {}
        self.default_plot_dict.update(
            {
                "subplot": 111,
                "skyproj": skyproj.MollweideSkyproj,
                "skyproj_kwargs": {"lon_0": 0},
                "decorations": self.default_decorations,
                "title": None,
                "xlabel": None,
                "label": None,
                "labelsize": None,
                "fontsize": None,
                "figsize": None,
            }
        )
        self._initialize_plot_dict({})
        self.figure = None

    def _initialize_plot_dict(self, user_plot_dict):
        # Use a defaultdict so that everything not explicitly set return None.
        def return_none():
            return None

        self.plot_dict = defaultdict(return_none)
        self.plot_dict.update(self.default_plot_dict)
        self.plot_dict.update(user_plot_dict)

    def _prepare_skyproj(self, fig=None):
        # Exctract elemens of plot_dict that need to be passed to plt.figure
        fig_kwargs = {k: v for k, v in self.plot_dict.items() if k in ["figsize"]}

        self.figure = plt.figure(**fig_kwargs) if fig is None else fig

        # Normalize subplot specification,
        # so that it can be set in either (single) integer form or tuple form.
        # This is needed so that the same subplot set by different ways will be
        # stored in the same key in the instance dictionary.
        subplot_in = self.plot_dict["subplot"]
        in_integral_form = isinstance(subplot_in, Integral) and len(str(subplot_in)) == 3
        subplot = tuple(map(int, str(subplot_in))) if in_integral_form else tuple(subplot_in)

        # If there is an existing instance of SkyprojPlotter (or one of its
        # subclasses), reuse it.
        # Otherwise, create a new one and store it so that it may be reused.
        if self.figure not in self.skyproj_instances:
            self.skyproj_instances[self.figure] = {}

        if subplot in self.skyproj_instances[self.figure]:
            existing_instance = self.skyproj_instances[self.figure][subplot]
            assert isinstance(existing_instance, self.plot_dict["skyproj"])
            self.skyproj = existing_instance
        else:
            # This instance of Axis will get deleted and replaced when the
            # corresponding SkyPlot class is created, but this instances is
            # needed so SkyPlot will put its instance in the correct relation
            # with its figure.
            ax = self.figure.add_subplot(*subplot)
            self.skyproj = self.plot_dict["skyproj"](ax=ax, **self.plot_dict["skyproj_kwargs"])
            self.skyproj_instances[self.figure][subplot] = self.skyproj

    def draw_circle(self, center_ra, center_decl, radius=90, **kwargs):
        """Draw a circle on the sphere.

        Parameters
        ----------
        center_ra : `float`
            R.A. of the center of the circle (deg.).
        center_decl : `float`
            Decl. of the center of the circle (deg.).
        radius : float, optional
            Radius of the circle (deg.), by default 90.0
        **kwargs
            Additional keyword arguments passed to
            `skyproj._Skyproj.draw_polygon`.
        """
        points = compute_circle_points(center_ra, center_decl, radius)
        self.skyproj.draw_polygon(points.ra, points.decl, **kwargs)

    def draw_ecliptic(self):
        """Draw the ecliptic the sphere.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to
            `skyproj._Skyproj.draw_polygon`.
        """
        ecliptic_pole = SkyCoord(lon=0 * u.degree, lat=90 * u.degree, frame="geocentricmeanecliptic").icrs
        self.draw_circle(ecliptic_pole.ra.deg, ecliptic_pole.dec.deg, **self.ecliptic_kwargs)

    def draw_galactic_plane(self):
        """Draw the galactic plane the sphere.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to
            `skyproj._Skyproj.draw_polygon`.
        """
        galactic_pole = SkyCoord(l=0 * u.degree, b=90 * u.degree, frame="galactic").icrs
        self.draw_circle(galactic_pole.ra.deg, galactic_pole.dec.deg, **self.galactic_plane_kwargs)

    def draw_zd(self, zd=90, **kwargs):
        """Draw a circle at a given zenith distance.

        Parameters
        ----------
        zd : `float`
            The zenith distance to draw, in degrees.
            Defaults to 90.
        **kwargs
            Keyword arguments passed to
            `skyproj._Skyproj.draw_polygon`.

        Notes
        -----
        This method relies on the site location and time (as an mjd) set in
        the ``model_observatory`` element of the ``plot_dict``, which
        should be of the class
        `rubin_scheduler.scheduler.model_observatory.ModelObservatory`.
        """
        if "model_observatory" not in self.plot_dict or self.plot_dict["model_observatory"] is None:
            warnings.warn("plot_dict['model_observatory'] must be set to plot a zenith distance circle")
            return

        model_observatory = self.plot_dict["model_observatory"]
        lmst = model_observatory.return_conditions().lmst * 360 / 24
        latitude = model_observatory.location.lat.deg
        self.draw_circle(lmst, latitude, zd, **kwargs)

    def draw_body(self, body="sun", **kwargs):
        """Mark the sun or moon.

        Parameters
        ----------
        body : `str`
            The name of the body to draw, either `sun` or `moon`.
            Defaults to `sun`
        **kwargs
            Keyword arguments passed to
            `skyproj._Skyproj.scatter`

        Notes
        -----
        This method relies on the site location and time (as an mjd) set in
        the ``model_observatory`` element of the ``plot_dict``, which
        should be of the class
        `rubin_scheduler.scheduler.model_observatory.ModelObservatory`.
        """
        if "model_observatory" not in self.plot_dict or self.plot_dict["model_observatory"] is None:
            warnings.warn(f"plot_dict['model_observatory'] must be set to plot the {body}")
            return

        mjd = self.plot_dict["model_observatory"].mjd
        model_observatory = self.plot_dict["model_observatory"]
        sun_moon_positions = model_observatory.almanac.get_sun_moon_positions(mjd)
        ra = np.degrees(sun_moon_positions[f"{body}_RA"].item())
        decl = np.degrees(sun_moon_positions[f"{body}_dec"].item())
        self.skyproj.scatter(ra, decl, **kwargs)

    def decorate(self):
        """Add decorations/annotations to the sky plot.

        Notes
        -----
        For decorations that depend on the time or location of the observer
        (e.g., the horizon and sun and moon positions)
        this method relies on the site location and time (as an mjd) set in
        the ``model_observatory`` element of the ``plot_dict``, which
        should be of the class
        `rubin_scheduler.scheduler.model_observatory.ModelObservatory`.
        Other decorations (e.g., the ecliptic and galactic plane) can
        be shown even when ``model_observatory`` is not set.
        """
        decorations = self.plot_dict["decorations"]

        if "ecliptic" in decorations:
            self.draw_ecliptic()

        if "galactic_plane" in decorations:
            self.draw_galactic_plane()

        if "sun" in decorations:
            self.draw_body("sun", **self.sun_kwargs)

        if "moon" in decorations:
            self.draw_body("moon", **self.moon_kwargs)

        if "horizon" in decorations:
            self.draw_zd(**self.horizon_kwargs)

    @abc.abstractmethod
    def draw(self, metric_values, slicer):
        raise NotImplementedError()

    def __call__(self, metric_values, slicer, user_plot_dict, fig=None):
        """Make a plot.

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
            Matplotlib figure.
            The default is ``None``, which starts new figure.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
           Figure with the plot.
        """
        self._initialize_plot_dict(user_plot_dict)
        self._prepare_skyproj(fig)
        self.draw(metric_values, slicer)
        self.decorate()

        # Do not show axis labels unless they are set explicitly
        # or specified as decorators.
        if self.plot_dict["xlabel"] is None:
            if "xlabel" not in self.plot_dict["decorations"]:
                self.skyproj.ax.set_xlabel("", visible=False)
        else:
            self.skyproj.ax.set_xlabel(self.plot_dict["xlabel"])

        if self.plot_dict["ylabel"] is None:
            if "ylabel" not in self.plot_dict["decorations"]:
                self.skyproj.ax.set_ylabel("", visible=False)
        else:
            self.skyproj.ax.set_ylabel(self.plot_dict["ylabel"])

        if self.plot_dict["label"] is not None:
            label_kwargs = {}
            if "fontsize" in self.plot_dict:
                label_kwargs["fontsize"] = self.plot_dict["fontsize"]
            if "label_loc" in self.plot_dict:
                label_kwargs["loc"] = self.plot_dict["label_loc"]
            self.skyproj.ax.set_title(self.plot_dict["label"], **label_kwargs)

        if self.plot_dict["title"] is not None:
            title_kwargs = {}
            if "fontsize" in self.plot_dict:
                title_kwargs["fontsize"] = self.plot_dict["fontsize"]
            self.figure.suptitle(self.plot_dict["title"], **title_kwargs)

        return self.figure


class HpxmapPlotter(SkyprojPlotter):
    default_hpixmap_kwargs = {"zoom": False}

    def __init__(self):
        super().__init__()
        self.default_plot_dict["colorbar"] = True
        self.object_plotter = False

    def draw_colorbar(self):
        """Add a color bar."""
        plot_dict = self.plot_dict

        colorbar_kwargs = {}
        colorbar_kwargs.update(self.default_colorbar_kwargs)

        if "extend" in self.plot_dict:
            colorbar_kwargs["extendrect"] = self.plot_dict["extend"] == "neither"

        if "cbar_format" in self.plot_dict:
            colorbar_kwargs["format"] = self.plot_dict["cbar_format"]

        if "cbar_orientation" in self.plot_dict:
            colorbar_kwargs["orientation"] = self.plot_dict["cbar_orientation"]
            match self.plot_dict["cbar_orientation"].lower():
                case "vertical":
                    colorbar_kwargs["shrink"] = 0.5
                    colorbar_kwargs["location"] = "right"
                case "horizontal":
                    colorbar_kwargs["location"] = "bottom"
                case _:
                    orientation = self.plot_dict["cbar_orientation"]
                    raise NotImplementedError(f"cbar_orintation {orientation} is not supported")

        colorbar = self.skyproj.draw_colorbar(**colorbar_kwargs)

        colorbar.set_label(self.plot_dict["xlabel"], fontsize=self.plot_dict["fontsize"])

        if "labelsize" in self.plot_dict and self.plot_dict["labelsize"] is not None:
            colorbar.ax.tick_params(labelsize=plot_dict["labelsize"])

        if "n_ticks" in self.plot_dict and self.plot_dict["n_ticks"] is not None:
            tick_locator = mpl.ticker.MaxNLocator(nbins=plot_dict["n_ticks"])
            colorbar.locator = tick_locator
            colorbar.update_ticks()

        # If outputing to PDF, this fixes the colorbar white stripes
        if "cbar_edge" in self.plot_dict and self.plot_dict["cbar_edge"]:
            colorbar.solids.set_edgecolor("face")

    def decorate(self):
        super().decorate()

        if self.plot_dict["colorbar"]:
            self.draw_colorbar()

    def _transform_to_healpix(self, metric_value_in, slicer):
        # Bin the values up on a healpix grid.
        metric_value = _healbin(
            slicer.slice_points["ra"],
            slicer.slice_points["dec"],
            metric_value_in.filled(slicer.badval),
            nside=self.plot_dict["nside"],
            reduce_func=self.plot_dict["reduce_func"],
            fill_val=slicer.badval,
        )
        mask = np.zeros(metric_value.size)
        mask[np.where(metric_value == slicer.badval)] = 1

        metric_value = ma.array(metric_value, mask=mask)
        return metric_value

    def _compatible_color_limits(self, norm, clims):
        # Log can't be 0
        compatible = (norm != "log") or (clims[0] < 0 and clims[1] < 0) or (clims[0] > 0 and clims[1] > 0)
        return compatible

    def draw(self, metric_values_in, slicer):
        """Draw the healpix map.

        Parameters
        ----------
        metric_value : `numpy.ma.MaskedArray`
            The metric values from the bundle.
        slicer : `rubin_sim.maf.slicers.TwoDSlicer`
            The slicer
        """
        kwargs = {}
        kwargs.update(self.default_hpixmap_kwargs)

        # Check if we have a valid HEALpix slicer
        if "Heal" not in slicer.slicer_name:
            hpix_map = self._transform_to_healpix(metric_values_in, slicer)
        else:
            hpix_map = metric_values_in.copy()

        hpix_map = apply_zp_norm(hpix_map, self.plot_dict)

        if "draw_hpxmap_kwargs" in self.plot_dict:
            kwargs.update(self.plot_dict["draw_hpxmap_kwargs"])

        try:
            kwargs["cmap"] = set_color_map(self.plot_dict)
        except AttributeError:
            # Probably here because an invalid cmap was set
            pass

        if self.plot_dict["log_scale"]:
            if "norm" in kwargs and kwargs["norm"] != "log":
                raise ValueError("Contradictory values for color scale normalization set.")
            kwargs["norm"] = "log"

        clims = set_color_lims(hpix_map, self.plot_dict)
        if "norm" not in kwargs or self._compatible_color_limits(kwargs["norm"], clims):
            kwargs["vmin"] = clims[0]
            kwargs["vmax"] = clims[1]
        else:
            warnings.warn(
                "Using log normalization, but color limits pass through 0. "
                "Adjusting so plotting doesn't fail"
            )

        # Filling in masked values must take place after automatic color limits
        # might be calculated!

        # This masking replicates the behavior of HealpixSkyMap, but is it
        # what we really want to do? Why not use the default handling of
        # masked values applied by skyproj?
        if "mask_below" in self.plot_dict and self.plot_dict["mask_below"] is not None:
            to_mask = np.where(hpix_map <= self.plot_dict["mask_below"])[0]
            hpix_map.mask[to_mask] = True
            hpix_map = hpix_map.filled(hp.UNSEEN)
        else:
            hpix_map = hpix_map.filled(slicer.badval)

        self.skyproj.draw_hpxmap(hpix_map, **kwargs)


class VisitPerimeterPlotter(SkyprojPlotter):
    default_visits_kwargs = {
        "edgecolor": "black",
        "linewidth": 0.2,
    }

    def __init__(self):
        super().__init__()
        self.plot_type = "SkyprojVisitPerimeter"
        self.object_plotter = True

    def draw(self, metric_values, slicer):
        """Draw the perimeters of visits.

        Parameters
        ----------
        metric_value : `numpy.ma.MaskedArray`
            The metric values from the bundle.
        slicer : `rubin_sim.maf.slicers.TwoDSlicer`
            The slicer
        """

        # If there are no visits, just return
        if metric_values is None or len(metric_values) == 0:
            return

        kwargs = {}
        kwargs.update(self.default_visits_kwargs)

        visits = metric_values[0]

        if "draw_polygon_kwargs" in self.plot_dict:
            kwargs.update(self.plot_dict["draw_polygon_kwargs"])

        camera_perimeter_func = self.plot_dict["camera_perimeter_func"]

        ras, decls = camera_perimeter_func(visits["fieldRA"], visits["fieldDec"], visits["rotSkyPos"])
        for visit_ras, visit_decls in zip(ras, decls):
            self.skyproj.draw_polygon(visit_ras, visit_decls, **kwargs)
