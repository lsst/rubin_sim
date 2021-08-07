#####################################################################################################
# Purpose: Calculate the number of observations in a given dataslice.

# Humna Awan: humna.awan@rutgers.edu
# Last updated: 06/10/16
 #####################################################################################################
 
from rubin_sim.maf.metrics import BaseMetric

__all__= ['NumObsMetric']

class NumObsMetric(BaseMetric):
    """Calculate the number of observations per data slice.
     e.g. HealPix pixel when using HealPix slicer.

    Parameters
    -----------
    nightCol : `str`
        Name of the night column in the data; basically just need it to
        acccess the data for each visit. Default: 'night'.
    nside : `int`
        HEALpix resolution parameter. Default: 128
    """
    def __init__(self, nightCol='night', nside=128, metricName='NumObsMetric', **kwargs):
        self.nightCol = nightCol
        super(NumObsMetric, self).__init__(col=self.nightCol, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        return len(dataSlice)
