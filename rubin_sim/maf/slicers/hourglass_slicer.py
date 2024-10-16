__all__ = ("HourglassSlicer",)


from rubin_sim.maf.plots import HourglassPlot

from .uni_slicer import UniSlicer


class HourglassSlicer(UniSlicer):
    """Slicer to make the filter hourglass plots"""

    def __init__(self, verbose=True, badval=-666):
        # Inherits from UniSlicer, so nslice=1 and only one 'slice'.
        super(HourglassSlicer, self).__init__(verbose=verbose, badval=badval)
        self.columns_needed = []
        self.slicer_name = "HourglassSlicer"
        self.plot_funcs = [
            HourglassPlot,
        ]

    def write_data(self, outfilename, metric_values, metric_name="", **kwargs):
        """
        Override base write method: we don't want to save hourglass metric
        data.

        The data volume is too large.
        """
        pass

    def read_metric_data(self, infilename):
        """
        Override base read method to 'pass':
        we don't save or read hourglass metric data.

        The data volume is too large.
        """
        pass
