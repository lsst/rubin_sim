from __future__ import absolute_import
# Example for CountMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Motivation: The distances to stars in LSST will be signficant enough that the structure of the Galaxy will be readily apparent because of its influence on the number of stars in a given field. Any metric concerned with the number of potential objects to be detected will need to feature not only the effects of the cadence but also the number of objects per field.
# This metric identifies the number of stars in a given field in a particular distance range. D1 and D2 are the close and far distances in parsecs.
# Requires StarCounts.StarCounts

from rubin_sim.maf.metrics import BaseMetric
import numpy as np
from .StarCounts.StarCounts import *

__all__ = ['StarCountMetric']

class StarCountMetric(BaseMetric):
   """Find the number of stars in a given field between D1 and D2 in parsecs.
   """
   def __init__(self,**kwargs):
      self.D1=kwargs.pop('D1', 100)
      self.D2=kwargs.pop('D2', 1000)
      super(StarCountMetric, self).__init__(col=[], **kwargs)

   def run(self, dataSlice, slicePoint=None):
      self.DECCol=np.degrees(dataSlice[0][3])
      self.RACol=np.degrees(dataSlice[0][2])
      return starcount.starcount(self.RACol, self.DECCol, self.D1, self.D2)

