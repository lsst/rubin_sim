# Angular spread metric:
# https://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread
# jmeyers314@gmail.com

import numpy as np
from rubin_sim.maf.metrics import BaseMetric

__all__ = ['AngularSpreadMetric']

class AngularSpreadMetric(BaseMetric):
    """Compute the angular spread statistic which measures uniformity of a distribution angles accounting for 2pi periodicity.

    The strategy is to first map angles into unit vectors on the unit circle, and then compute the
    2D centroid of those vectors.  A uniform distribution of angles will lead to a distribution of
    unit vectors with mean that approaches the origin.  In contrast, a delta function distribution
    of angles leads to a delta function distribution of unit vectors with a mean that lands on the
    unit circle.

    The angular spread statistic is then defined as 1 - R, where R is the radial offset of the mean
    of the unit vectors derived from the input angles.  R approaches 1 for a uniform distribution
    of angles and 0 for a delta function distribution of angles.

    The optional parameter `period` may be used to specificy periodicity other than 2 pi.
    """
    def __init__(self, col=None, period=2.0*np.pi, **kwargs):
        self.period = period
        super(AngularSpreadMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Unit vectors; unwrapped at specified period
        x = np.cos(dataSlice[self.colname] * 2.0*np.pi/self.period)
        y = np.sin(dataSlice[self.colname] * 2.0*np.pi/self.period)
        meanx = np.mean(x)
        meany = np.mean(y)
        # radial offset (i.e., length) of the mean unit vector
        R = np.hypot(meanx, meany)
        return 1.0 - R
