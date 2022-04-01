__all__ = ["BaseBokehPlotter"]

import abc
from copy import deepcopy
from collections import namedtuple

import bokeh
import bokeh.plotting

from rubin_sim.maf.plots import BasePlotter


class BaseBokehPlotter(BasePlotter, abc.ABC):
    _plotType = None
    required_data_source_columns = []
    default_glyph_args = {}
    default_figure_args = {}
    color_glyph_args = []

    def __init__(self):
        self.defaultPlotDict = {
            "glyph_args": self.default_glyph_args,
            "figure_args": self.default_figure_args,
        }

    @abc.abstractmethod
    def add_glyph(self, metric_bundle, glyph_args):
        pass

    def check_data_source(self, data_source):
        for slicer_column in self.required_data_source_columns:
            if slicer_column not in data_source.data:
                raise ValueError(
                    f"{slicer_column} not found in data_source: this MetricBundle is not compatible with this plot, probably because it has an incompatible slicer."
                )

    def refine_plot(self, plot, plotDict):
        pass

    @property
    def plotType(self):
        if self._plotType is None:
            return self.__class__.__name__

        return self._plotType

    def infer_plotDict(self, metric_bundle, plotDict={}, inplace=False):
        if not inplace:
            plotDict = deepcopy(plotDict)

        figure_args = plotDict.get('figure_args', {})
        glyph_args = plotDict.get('glyph_args', {})
        plotDict.update(self.defaultPlotDict)
        figure_args.update(self.defaultPlotDict.get('figure_args', {}))
        glyph_args.update(self.defaultPlotDict.get('glyph_args', {}))

        figure_args['title'] = f"{metric_bundle.runName} {metric_bundle.info_label}: {metric_bundle.metric.name}"

        # If we set the color in the plotDict and haven't given
        # values in the glyph args, set the glyph args to have
        # that color
        if 'color' in plotDict:
            for color_arg in self.color_glyph_args:
                if color_arg not in plotDict['glyph_args']:
                    glyph_args[color_arg] = plotDict['color']

        if 'legend_label' not in glyph_args:
            glyph_args['legend_label'] = metric_bundle.info_label

        plotDict['figure_args'] = figure_args
        plotDict['glyph_args'] = glyph_args

        return plotDict

    def __call__(self, metric_bundle, userPlotDict={}, plot=None):
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        if plot is None:
            figure_args = plotDict.get("figure_args", {})
            plot = bokeh.plotting.figure(**figure_args)

        glyph_args = {}
        glyph_args.update(self.default_glyph_args)
        user_glyph_args = plotDict.get("glyph_args", {})
        glyph_args.update(user_glyph_args)

        self.add_glyph(plot, metric_bundle, glyph_args)

        self.refine_plot(plot, plotDict)

        return plot
