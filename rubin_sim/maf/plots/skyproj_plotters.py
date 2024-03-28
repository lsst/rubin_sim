from collections import defaultdict
from numbers import Integral
from weakref import WeakKeyDictionary

import matplotlib.pyplot as plt
import skyproj

from .plot_handler import BasePlotter
from .spatial_plotters import set_color_lims, set_color_map


class SkyprojPlotter(BasePlotter):
    # Store a mapping between figures and existing SkyprojPlotters, but use
    # weak keys so that the figure can be cleaned up by the garbage collector,
    # and when it is entry in the dictionary automatically deleted.
    skyproj_instances = WeakKeyDictionary()

    default_decorations = []

    default_colorbar_kwargs = {
        "location": "bottom",
        "shrink": 0.75,
        "aspect": 25,
        "pad": 0.1,
        "orientation": "horizontal",
    }

    def __init__(self):
        super().__init__()
        # Customize our plotters members for our new plot

        self.default_plot_dict = {}
        self.default_plot_dict.update(
            {
                "subplot": 111,
                "skyproj": skyproj.MollweideSkyproj,
                "skyproj_kwargs": {"lon_0": 0},
                "kind": "hpxmap",
                "decorations": self.default_decorations,
            }
        )
        self.initialize_plot_dict({})
        self.figure = None

    def initialize_plot_dict(self, user_plot_dict):
        # Use a defaultdict so that everything not explicitly set return None.
        def return_none():
            return None

        self.plot_dict = defaultdict(return_none)
        self.plot_dict.update(self.default_plot_dict)
        self.plot_dict.update(user_plot_dict)

    def prepare_skyproj(self, fignum=None):
        # Exctract elemens of plot_dict that need to be passed to plt.figure
        fig_kwargs = {k: v for k, v in self.plot_dict.items() if k in ["figsize"]}

        self.figure = plt.figure(fignum, **fig_kwargs)

        # Normalize subplot specification,
        # so that it can be in either (single) in form or tuple form.
        # This is needed so that the same subplot set by different ways will be
        # stored in the same key in the instance dictionary.
        subplot_in = self.plot_dict["subplot"]
        in_integral_form = isinstance(subplot_in, Integral) and 100 <= subplot_in <= 999
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
            self.skyproj = self.plot_dict["skyproj"](ax, **self.plot_dict["skyproj_kwargs"])
            self.skyproj_instances[self.figure][subplot] = self.skyproj

    def decorate(self):
        decorations = self.plot_dict["decorations"]
        return decorations

    def __call__(self, metric_values, slicer, user_plot_dict, fignum=None):
        self.initialize_plot_dict(user_plot_dict)
        print(self.plot_dict)
        self.prepare_skyproj(fignum)
        self.draw(metric_values, slicer)
        self.decorate()

        return self.figure.number


class HpxmapPlotter(SkyprojPlotter):
    default_hpixmap_kwargs = {"zoom": False}

    def __init__(self):
        super().__init__()
        self.plot_type = "SkyprojHpxmap"
        self.default_plot_dict["decorations"].append("colorbar")
        self.object_plotter = False

    def draw_colorbar(self):
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

        if self.plot_dict["labelsize"] is not None:
            colorbar.ax.tick_params(labelsize=plot_dict["labelsize"])

    def decorate(self):
        super().decorate()

        if "colorbar" in self.plot_dict["decorations"]:
            self.draw_colorbar()

    def draw(self, metric_values, slicer):
        kwargs = {}
        kwargs.update(self.default_hpixmap_kwargs)

        hpix_map = metric_values

        if "draw_hpxmap_kwargs" in self.plot_dict:
            kwargs.update(self.plot_dict["draw_hpxmap_kwargs"])

        try:
            kwargs["cmap"] = set_color_map(self.plot_dict)
        except AttributeError:
            # Probably here because an invalid cmap was set
            pass

        clims = set_color_lims(hpix_map, self.plot_dict)
        kwargs["vmin"] = clims[0]
        kwargs["vmax"] = clims[1]

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
        kwargs = {}
        kwargs.update(self.default_visits_kwargs)

        visits = metric_values[0]

        if "draw_polygon_kwargs" in self.plot_dict:
            kwargs.update(self.plot_dict["draw_polygon_kwargs"])

        camera_perimeter_func = self.plot_dict["camera_perimeter_func"]

        ras, decls = camera_perimeter_func(visits["fieldRA"], visits["fieldDec"], visits["rotSkyPos"])
        for visit_ras, visit_decls in zip(ras, decls):
            self.skyproj.draw_polygon(visit_ras, visit_decls, **kwargs)
