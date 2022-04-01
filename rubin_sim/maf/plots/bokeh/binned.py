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

    def create_glyph(self, metric_bundle, glyph_args):
        metric_column_name = metric_bundle.data_source_metric_column_name
        glyph = bokeh.models.Quad(
            left="bin_min", right="bin_max", bottom=0, top=metric_column_name, **glyph_args
        )
        return glyph


class OneDBinnedStep(OneDBinnedPlotter):
    color_glyph_args = ['line_color']
    default_glyph_args = {'mode': 'after'}

    def create_glyph(self, metric_bundle, glyph_args):
        metric_column_name = metric_bundle.data_source_metric_column_name

        if glyph_args['mode'] == 'after':
            x_column = 'bin_min'
        elif glyph_args['mode'] == 'before':
            x_column = 'bin _max'
        else:
            raise NotImplementedError(f"Step mode {glyph_args['mode']} not supported by {self.__class__}")

        glyph = bokeh.models.Step(
            x="bin_min", y=metric_column_name, **glyph_args
        )
        return glyph
