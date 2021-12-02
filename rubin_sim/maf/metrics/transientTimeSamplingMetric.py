################################################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric
import healpy as hp
import calcVisitIntervalMetric, calcSeasonVisibilityGapsMetric

class transientTimeSamplingMetric(BaseMetric):

    def __init__(self, cols=['observationStartMJD',],
                       metricName='calcVisitIntervalMetric',
                       **kwargs):
         """tau_obs is an array of minimum-required observation intervals for four categories
        of time variability"""
            
        self.mjdCol = 'observationStartMJD'
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])

        super(transientTimeSamplingMetric,self).__init__(col=cols, metricName=metricName)


    def run(self, dataSlice, slicePoint=None):
        
        metric_values1 = np.array(calcVisitIntervalMetric(dataSlice, slicePoint))
        metric_values2 = np.array(calcSeasonVisibilityGapsMetric(dataSlice, slicePoint))
        
        metric_values= metric_values1 * metric_values2 
    
        return metric_values