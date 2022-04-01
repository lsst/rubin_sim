import bokeh.models

from . import BaseBokehPlotter

__all__ = ["OneDBinnedQuad", "OneDBinnedStep"]

class OneDBinnedPlotter(BaseBokehPlotter):
    objectPlotter = False
    required_data_source_columns = ["bin_min", "bin_max"]

    def infer_plotDict(self, metric_bundle, plotDict={}, inplace=False):
        plotDict = super().infer_plotDict(metric_bundle, plotDict, inplace)
        plotDict["figure_args"].update(
            {
                "x_axis_label": f"{metric_bundle.slicer.sliceColName} ({metric_bundle.slicer.sliceColUnits})",
                "y_axis_label": f"{metric_bundle.metric.name} ({metric_bundle.metric.units})",
            }
        )
        return plotDict


class OneDBinnedQuad(OneDBinnedPlotter):
    color_glyph_args = ['line_color', 'fill_color']

    def add_glyph(self, plot, metric_bundle, glyph_args):
        data_source = metric_bundle.make_column_data_source()
        self.check_data_source(data_source)
        metric_column_name = metric_bundle.data_source_metric_column_name
        plot.quad(
            left="bin_min", right="bin_max", bottom=0, top=metric_column_name, source=data_source, **glyph_args
        )
        return plot


class OneDBinnedStep(OneDBinnedPlotter):
    color_glyph_args = ['line_color']
    default_glyph_args = {'mode': 'after'}

    def add_glyph(self, plot, metric_bundle, glyph_args):
        data_source = metric_bundle.make_column_data_source()
        self.check_data_source(data_source)
        metric_column_name = metric_bundle.data_source_metric_column_name

        if glyph_args['mode'] == 'after':
            x_column = 'bin_min'
        elif glyph_args['mode'] == 'before':
            x_column = 'bin _max'
        else:
            raise NotImplementedError(f"Step mode {glyph_args['mode']} not supported by {self.__class__}")

        plot.step(x="bin_min", y=metric_column_name, source=data_source, **glyph_args)

        return plot