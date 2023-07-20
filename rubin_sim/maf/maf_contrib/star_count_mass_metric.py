__all__ = ("StarCountMassMetric",)

import numpy as np

from rubin_sim.maf.metrics import BaseMetric

from .star_counts import *

# Example for CountMassMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Motivation: The distances to stars in LSST will be signficant enough that the structure of the Galaxy will be readily apparent because of its influence on the number of stars in a given field. Any metric concerned with the number of potential objects to be detected will need to feature not only the effects of the cadence but also the number of objects per field.
# This metric identifies the number of stars in a given field in a particular mass range that will be fainter than the saturation limit of 16th magnitude and still bright enough to have noise less than 0.03 mag. M1 and M2 are the low and high limits of the mass range in solar masses. 'band' is the band for the observations to be made in.
# Requires StarCounts.StarCounts


class StarCountMassMetric(BaseMetric):
    """Find the number of stars in a given field in the mass range fainter than magnitude 16 and bright enough to have noise less than 0.03 in a given band. M1 and m2 are the upper and lower limits of the mass range. 'band' is the band to be observed."""

    def __init__(self, **kwargs):
        self.m1 = kwargs.pop("M1", 0.9)
        self.m2 = kwargs.pop("M2", 1.0)
        self.band = kwargs.pop("band", "i")
        super(StarCountMassMetric, self).__init__(col=[], **kwargs)

    def run(self, data_slice, slice_point=None):
        self.dec_col = np.degrees(data_slice[0][3])
        self.ra_col = np.degrees(data_slice[0][2])
        return starcount_bymass.starcount_bymass(self.ra_col, self.dec_col, self.m1, self.m2, self.band)
