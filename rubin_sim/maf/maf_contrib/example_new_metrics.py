# Example of a new metric added to the repo.
# ljones@astro.washington.edu

__all__ = ("NightsWithNFiltersMetric",)

import numpy as np

from rubin_sim.maf.metrics import BaseMetric


class NightsWithNFiltersMetric(BaseMetric):
    """Count how many times more than NFilters are used within the same night,
    for this set of visits.

    Parameters
    ----------
    n_filters : `int`, optional
        How many filters to look for, within the same night.
    """

    def __init__(self, night_col="night", filter_col="filter", n_filters=3, **kwargs):
        """
        night_col = the name of the column defining the night
        filter_col = the name of the column defining the filter
        n_filters = the minimum desired set of filters used in these visits
        """
        self.night_col = night_col
        self.filter_col = filter_col
        self.n_filters = n_filters
        super(NightsWithNFiltersMetric, self).__init__(col=[self.night_col, self.filter_col], **kwargs)

    def run(self, data_slice, slice_point=None):
        count = 0
        unique_nights = np.unique(data_slice[self.night_col])
        for n in unique_nights:
            condition = data_slice[self.night_col] == n
            unique_filters = np.unique(data_slice[self.filter_col][condition])
            if len(unique_filters) > self.n_filters:
                count += 1
        return count
