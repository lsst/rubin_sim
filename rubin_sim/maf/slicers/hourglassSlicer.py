import numpy as np
import matplotlib.pyplot as plt
import warnings
from rubin_sim.maf.plots import HourglassPlot
from .uniSlicer import UniSlicer

__all__ = ['HourglassSlicer']

class HourglassSlicer(UniSlicer):
    """Slicer to make the filter hourglass plots """

    def __init__(self, verbose=True, badval=-666):
        # Inherits from UniSlicer, so nslice=1 and only one 'slice'.
        super(HourglassSlicer,self).__init__(verbose=verbose, badval=badval)
        self.columnsNeeded=[]
        self.slicerName='HourglassSlicer'
        self.plotFuncs = [HourglassPlot,]

    def writeData(self, outfilename, metricValues, metricName='', **kwargs):
        """
        Override base write method: we don't want to save hourglass metric data.

        The data volume is too large.
        """
        pass

    def readMetricData(self, infilename):
        """
        Override base read method to 'pass': we don't save or read hourglass metric data.

        The data volume is too large.
        """
        pass
