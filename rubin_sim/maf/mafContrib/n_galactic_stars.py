import numpy as np
from rubin_sim.maf.metrics.baseMetric import BaseMetric

__all__ = [' NGalStarsMetric']


class NGalStarsMetric(BaseMetric):
    """Use the stellar density maps and crowding metrics to find the total number of stars that we would
    expect in a catalog.
    """

    def __init__(self, filtername='r', filterCol='filter'):

        pass

    def run(self, dataSlice, slicePoint=None):

