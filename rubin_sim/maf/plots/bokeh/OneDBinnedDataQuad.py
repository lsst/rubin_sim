import bokeh.models

from . import BaseBokehPlotter

__all__ = ["OneDBinnedDataQuad"]


class OneDBinnedDataQuad(BaseBokehPlotter):
    objectPlotter = False
    required_data_source_columns = ["bin_min", "bin_max"]

    def infer_plotDict(self, metric_bundle):
        plotDict = super().infer_plotDict(metric_bundle)
        plotDict["figure_args"].update(
            {
                "x_axis_label": f"{metric_bundle.slicer.sliceColName} ({metric_bundle.slicer.sliceColUnits})",
                "y_axis_label": f"{metric_bundle.metric.name} ({metric_bundle.metric.units})",
            }
        )
        return plotDict

    def create_glyph(self, metric_bundle, glyph_args):
        metric_column_name = metric_bundle.data_source_metric_column_name
        glyph = bokeh.models.Quad(
            left="bin_min", right="bin_max", bottom=0, top=metric_column_name, **glyph_args
        )
        return glyph
