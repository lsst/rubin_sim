# Example for IntervalsBetweenObsMetric
# Somayeh Khakpash - Lehigh University
# Last edited : 10/21/2020
# Calculates statistics (mean or median or standard deviation) of intervals between observations during simultaneous windows/Inter-seasonal gap of another survey.
# SurveyIntervals is the list of the survey observing window/Inter-seasonal gap intervals. It should be in the format:
# SurveyIntervals = [ [YYYY-MM-DD, YYYY-MM-DD] , [YYYY-MM-DD, YYYY-MM-DD] , ... , [YYYY-MM-DD, YYYY-MM-DD] ]
# We are interested in calculating this metric in each of the LSST passbands.
# The difference between this metric and the VisitGapMetric metric is that VisitGapMetric calculates reduceFunc of gaps between observations of a dataslice throughout the whole 
# baseline, but IntervalsBetweenObsMetric calculates the gaps between observations during another survey observing window. This metric combined with surveys footprint
# overlap can determine how many often another survey footprint is observed by LSST during specific time intervals.


from __future__ import print_function
import numpy as np 
from astropy.time import Time
from rubin_sim.maf.metrics import BaseMetric

__all__ = ['IntervalsBetweenObsMetric']

class IntervalsBetweenObsMetric (BaseMetric):
    

    
    def __init__ (self,SurveyIntervals,Stat, metricName= 'IntervalsBetweenObsMetric', TimeCol='observationStartMJD', **kwargs):
        
        self.TimeCol = TimeCol
        self.metricName = metricName
        self.SurveyIntervals = SurveyIntervals
        self.Stat = Stat
        super(IntervalsBetweenObsMetric, self).__init__(col= TimeCol, metricName=metricName, **kwargs)


    def run (self, dataSlice, slicePoint=None):

        
        dataSlice.sort(order=self.TimeCol)
        obs_diff = []
        
        for interval in self.SurveyIntervals :
            
            start_interval = Time(interval[0]+' 00:00:00')
            end_interval = Time(interval[1]+' 00:00:00')
            index = dataSlice[self.TimeCol][np.where ((dataSlice[self.TimeCol]> start_interval.mjd) & (dataSlice[self.TimeCol]<end_interval.mjd))[0]]
            obs_diff_per_interval = np.diff(index) 
            obs_diff = obs_diff + obs_diff_per_interval.tolist()
            
        if self.Stat == 'mean':
            result = np.mean(obs_diff)
        
        elif self.Stat =='median' :
            result = np.median(obs_diff)
        
        elif self.Stat == 'std' : 
            result = np.std(obs_diff)  

        return result
