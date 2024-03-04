__all__ = ("StarCountMassMetric",)

import numpy as np

from rubin_sim.maf.metrics import BaseMetric

from .star_counts import starcount_bymass

# Example for CountMassMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Motivation: The distances to stars in LSST will be significant enough that
# the structure of the Galaxy will be readily apparent because of its
# influence on the number of stars in a given field.
# Any metric concerned with the number of potential objects to be detected
# will need to feature not only the effects of the cadence but also the
# number of objects per field.
# This metric identifies the number of stars in a given field in a particular
# mass range that will be fainter than the saturation limit of 16th magnitude
# and still bright enough to have noise less than 0.03 mag.
# M1 and M2 are the low and high limits of the mass range in solar masses.
# 'band' is the band for the observations to be made in.
# Requires StarCounts.StarCounts

# NOTE
# There are stellar luminosity function maps available within MAF
# that may supersede these StarCount functions


class StarCountMassMetric(BaseMetric):
    """Find the number of stars in a given field in the mass range
    fainter than magnitude 16 and bright enough to have noise less than
    0.03 in a given band.
    M1 and m2 are the upper and lower limits of the mass range.
    'band' is the band to be observed.

    This metric uses the stellar distance and luminosity equations
    contributed by Mike Lund, which are based on the Galfast model.
    There are some imposed limitations on the expected magnitudes
    of the stars included for the metric, based on assuming saturation
    at 16th magnitude and not considering stars with magnitude
    uncertainties greater than 0.03 (based on photometry/m5 alone).

    Parameters
    ----------
    m1 : `float`
        Lower limit of the mass range.
    m2 : `float`
        Upper limit of the mass range.
    band : `str`
        Bandpass to consider.
    """

    def __init__(self, m1=0.9, m2=1.0, band="i", **kwargs):
        self.m1 = m1
        self.m2 = m2
        self.band = band
        super(StarCountMassMetric, self).__init__(col=[], **kwargs)

    def run(self, data_slice, slice_point=None):
        self.dec_col = np.degrees(data_slice[0][3])
        self.ra_col = np.degrees(data_slice[0][2])
        return starcount_bymass.starcount_bymass(self.ra_col, self.dec_col, self.m1, self.m2, self.band)
