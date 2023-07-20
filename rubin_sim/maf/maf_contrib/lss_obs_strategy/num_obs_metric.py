#####################################################################################################
# Purpose: Calculate the number of observations in a given data_slice.

# Humna Awan: humna.awan@rutgers.edu
# Last updated: 06/10/16
#####################################################################################################
__all__ = ("NumObsMetric",)

from rubin_sim.maf.metrics import BaseMetric


class NumObsMetric(BaseMetric):
    """Calculate the number of observations per data slice.
     e.g. HealPix pixel when using HealPix slicer.

    Parameters
    -----------
    night_col : `str`
        Name of the night column in the data; basically just need it to
        acccess the data for each visit. Default: 'night'.
    nside : `int`
        HEALpix resolution parameter. Default: 128
    """

    def __init__(self, night_col="night", nside=128, metric_name="NumObsMetric", **kwargs):
        self.night_col = night_col
        super(NumObsMetric, self).__init__(col=self.night_col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        return len(data_slice)
