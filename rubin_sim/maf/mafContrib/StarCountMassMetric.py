from __future__ import absolute_import
# Example for CountMassMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 8/15/2015
# Motivation: The distances to stars in LSST will be signficant enough that the structure of the Galaxy will be readily apparent because of its influence on the number of stars in a given field. Any metric concerned with the number of potential objects to be detected will need to feature not only the effects of the cadence but also the number of objects per field.
# This metric identifies the number of stars in a given field in a particular mass range that will be fainter than the saturation limit of 16th magnitude and still bright enough to have noise less than 0.03 mag. M1 and M2 are the low and high limits of the mass range in solar masses. 'band' is the band for the observations to be made in.
# Requires StarCounts.StarCounts

from rubin_sim.maf.metrics import BaseMetric
import numpy as np
from .StarCounts.StarCounts import *

__all__ = ['StarCountMassMetric']

class StarCountMassMetric(BaseMetric):
   """Find the number of stars in a given field in the mass range fainter than magnitude 16 and bright enough to have noise less than 0.03 in a given band. M1 and M2 are the upper and lower limits of the mass range. 'band' is the band to be observed.
   """
   def __init__(self,**kwargs):
      self.M1=kwargs.pop('M1', 0.9)
      self.M2=kwargs.pop('M2', 1.0)
      self.band=kwargs.pop('band', 'i')
      super(StarCountMassMetric, self).__init__(col=[], **kwargs)

   def run(self, dataSlice, slicePoint=None):
      self.DECCol=np.degrees(dataSlice[0][3])
      self.RACol=np.degrees(dataSlice[0][2])
      return starcount_bymass.starcount_bymass(self.RACol, self.DECCol, self.M1, self.M2, self.band)

