__all__ = ("StarCountMetric",)

import numpy as np

from rubin_sim.maf.metrics import BaseMetric

from .star_counts import *

# Example for CountMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Motivation: The distances to stars in LSST will be signficant enough that the structure of the Galaxy will be readily apparent because of its influence on the number of stars in a given field. Any metric concerned with the number of potential objects to be detected will need to feature not only the effects of the cadence but also the number of objects per field.
# This metric identifies the number of stars in a given field in a particular distance range. D1 and D2 are the close and far distances in parsecs.
# Requires StarCounts.StarCounts


class StarCountMetric(BaseMetric):
    """Find the number of stars in a given field between d1 and d2 in parsecs."""

    def __init__(self, **kwargs):
        self.d1 = kwargs.pop("D1", 100)
        self.d2 = kwargs.pop("D2", 1000)
        super(StarCountMetric, self).__init__(col=[], **kwargs)

    def run(self, data_slice, slice_point=None):
        self.dec_col = np.degrees(data_slice[0][3])
        self.ra_col = np.degrees(data_slice[0][2])
        return starcount.starcount(self.ra_col, self.dec_col, self.d1, self.d2)
