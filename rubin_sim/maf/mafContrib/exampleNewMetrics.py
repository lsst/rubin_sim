# Example of a new metric added to the repo.
# ljones@astro.washington.edu

from rubin_sim.maf.metrics import BaseMetric
import numpy as np

__all__ = ['NightsWithNFiltersMetric']


class NightsWithNFiltersMetric(BaseMetric):
    """Count how many times more than NFilters are used within the same night, for this set of visits.
    """
    def __init__(self, nightCol='night', filterCol='filter', nFilters=3, **kwargs):
        """
        nightCol = the name of the column defining the night
        filterCol = the name of the column defining the filter
        nFilters = the minimum desired set of filters used in these visits
        """
        self.nightCol = nightCol
        self.filterCol = filterCol
        self.nFilters = nFilters
        super(NightsWithNFiltersMetric, self).__init__(col=[self.nightCol, self.filterCol],
                                                       **kwargs)

    def run(self, dataSlice, slicePoint=None):
        count = 0
        uniqueNights = np.unique(dataSlice[self.nightCol])
        for n in uniqueNights:
            condition = (dataSlice[self.nightCol] == n)
            uniqueFilters = np.unique(dataSlice[self.filterCol][condition])
            if len(uniqueFilters) > self.nFilters:
                count += 1
        return count
