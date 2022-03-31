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

    def __init__(self):
        self.defaultPlotDict = {
            "glyph_args": self.default_glyph_args,
            "figure_args": self.default_figure_args,
        }

    @abc.abstractmethod
    def create_glyph(self, metric_bundle, glyph_args):
        pass

    def check_data_source(self, data_source):
        for slicer_column in self.required_data_source_columns:
            if slicer_column not in data_source.data:
                raise ValueError(f"{slicer_column} not found in data_source: this MetricBundle is not compatible with this plot, probably because it has an incompatible slicer.")

    def refine_plot(self, plot, plotDict):
        pass

    @property
    def plotType(self):
        if self._plotType is None:
            return self.__class__.__name__
        
        return self._plotType

    def infer_plotDict(self, metric_bundle):
        plotDict = deepcopy(self.defaultPlotDict)
        plotDict.update({
            'figure_args': {
                'title': f"{metric_bundle.runName} {metric_bundle.info_label}: {metric_bundle.metric.name}"
            },
            'glyph_args': {},
        })
        return plotDict

    def __call__(self, metric_bundle, userPlotDict={}, plot=None):
        data_source = metric_bundle.make_column_data_source()
        self.check_data_source(data_source)

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
        glyph = self.create_glyph(metric_bundle, glyph_args)

        plot.add_glyph(data_source, glyph)

        self.refine_plot(plot, plotDict)

        return plot
