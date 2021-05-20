# Example for TripletMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 9/6/2014
# Motivation: The detection of nonperiodic transient events can be thought of as most simply being accomplished by a set of three observations, one before the event occurs, a second after the event has begun, and a third to confirm the event is real.
# This metric identifies the number of triplets that will occur. DelMin and DelMax set the smallest and largest intervals that can occur between the first and second point and between the second and third point. This can be set to reflect the timescales for various events. RatioMax and RatioMin set constraints on how similar the two intervals must be. RatioMin can never be less than 1.

from rubin_sim.maf.metrics import BaseMetric
import numpy as np

__all__ = ['TripletMetric', 'TripletBandMetric']

class TripletMetric(BaseMetric):
   """Find the number of 'triplets' of three images taken in any band, based on user-selected minimum and maximum intervals (in hours),
   as well as constraining the ratio of the two exposures intervals.
   Triplets are not required to be consecutive observations and may be overlapping.
   """
   def __init__(self, TimeCol='expMJD', **kwargs):
      self.TimeCol=TimeCol
      self.delmin=kwargs.pop('DelMin', 1)/24. #convert minutes to hours
      self.delmax=kwargs.pop('DelMax', 12)/24. #convert minutes to hours
      self.ratiomax=kwargs.pop('RatioMax', 1000)
      self.ratiomin=kwargs.pop('RatioMin', 1)
      super(TripletMetric, self).__init__(col=[self.TimeCol], **kwargs)

   def run(self, dataSlice, slicePoint=None):
      times=dataSlice[self.TimeCol]
      times=times-49378 #change times to smaller numbers
      delmax=self.delmax
      delmin=self.delmin
      ratiomax=self.ratiomax
      ratiomin=self.ratiomin
      total=0
      #iterate over every exposure time
      for (counter, time) in enumerate(times):
         #calculate the window to look for all possible second points in
         minmax=[time+delmin, time+delmax]
         index2=np.where((minmax[0] < times) & (times < minmax[1]))[0]
         #iterate over every middle exposure
         for middleindex in index2:
            timeb=times[middleindex]
            #calculate the window to look for all possible third points in
            minmax2=[timeb+delmin, timeb+delmax]
            index3=np.where((times > minmax2[0]) & (times < minmax2[1]))[0]
            newadd=np.size(index3)
            total=total+newadd #add all triplets with same first two observations to total
      return total



class TripletBandMetric(BaseMetric):
   """Find the number of 'triplets' of three images taken in the same band, based on user-selected minimum and maximum intervals (in hours),
   as well as constraining the ratio of the two exposures intervals.
   Triplets are not required to be consecutive observations and may be overlapping.
   """
   def __init__(self, TimeCol='expMJD', FilterCol='filter', **kwargs):
      self.TimeCol=TimeCol
      self.FilterCol=FilterCol
      self.delmin=kwargs.pop('DelMin', 1)/24. #convert minutes to hours
      self.delmax=kwargs.pop('DelMax', 12)/24. #convert minutes to hours
      self.ratiomax=kwargs.pop('RatioMax', 1000)
      self.ratiomin=kwargs.pop('RatioMin', 1)
      super(TripletBandMetric, self).__init__(col=[self.TimeCol, self.FilterCol], **kwargs)
      self.reduceOrder = {'Bandu':0, 'Bandg':1, 'Bandr':2, 'Bandi':3, 'Bandz':4, 'Bandy':5}

   def run(self, dataSlice, slicePoint=None):
      times=dataSlice[self.TimeCol]
      times=times-49378 #change times to smaller numbers
      bands=dataSlice[self.FilterCol]
      bandset=['u','g','r','i','z','y'] #list of possible bands
      timedict={}
      delmax=self.delmax
      delmin=self.delmin
      ratiomax=self.ratiomax
      ratiomin=self.ratiomin
      bandcounter={'u':0, 'g':0, 'r':0, 'i':0, 'z':0, 'y':0} #define zeroed out counter
      #iterate over each bandpass
      for band in bandset:
         indexlist=np.where(band==bands)
         timedict[band]=times[indexlist]
         #create a data set of all exposures for a single band
         timeband=timedict[band]
         #iterate over every exposure time
         for (counter, time) in enumerate(timeband):
           #calculate the window to look for all possible second points in
           minmax=[time+delmin, time+delmax]
           index2=np.where((minmax[0] < timeband) & (timeband < minmax[1]))[0]
           #iterate over every middle exposure
           for middleindex in index2:
             timeb=timeband[middleindex]
             #calculate the window to look for all possible third points in
             minmax2=[timeb+delmin, timeb+delmax]
             index3=np.where((timeband > minmax2[0]) & (timeband < minmax2[1]))[0]
             #iterate over last exposure of triplet
             for lastindex in index3:
                timec=timeband[lastindex]
                #calculate intervals for T1 to T2 and T2 to T3, and take ratio
                delt1=timeb-time
                delt2=timec-timeb
                ratio=np.max([delt1, delt2])/np.min([delt1, delt2])
                #check if ratio is within restrictions (ratio should never be < 1 )
                if ratiomin < ratio < ratiomax:
                   bandcounter[band]=bandcounter[band]+1
      return bandcounter #return bandcounter dictionary

   def reduceBandall(self, bandcounter):
      return np.sum(list(bandcounter.values()))

   def reduceBandu(self, bandcounter):
      return bandcounter['u']

   def reduceBandg(self, bandcounter):
      return bandcounter['g']

   def reduceBandr(self, bandcounter):
      return bandcounter['r']

   def reduceBandi(self, bandcounter):
      return bandcounter['i']

   def reduceBandz(self, bandcounter):
      return bandcounter['z']

   def reduceBandy(self, bandcounter):
      return bandcounter['y']
