#Slightly modified version of the DESC contributed metric ExgalM5_with_cuts available in https://github.com/lsst/sims_maf/blob/master/python/lsst/sims/maf/metrics/weakLensingSystematicsMetric.py. 

import numpy as np
from lsst.sims.maf.metrics import ExgalM5
from lsst.sims.maf.metrics import BaseMetric

class ExgalM5_with_cuts_AGN(BaseMetric):
    
    def __init__(self, lsstFilter, m5Col='fiveSigmaDepth', units='mag', extinction_cut=1.0, \
                 filterCol='filter', metricName='ExgalM5_with_cuts_AGN', **kwargs):
            
        #Dust Extinction limit to which consider regions. If left unconstrained,
        #it ends up finding extremely shallow (m_lim=100) 5 sigma regions. Not
        #sure why though, but this is something also enforced in the ExgalM5_with_cuts
        #metric of DESC.
        self.extinction_cut = extinction_cut

        #Save the filter information.
        self.filterCol = filterCol
        self.lsstFilter = lsstFilter

        #This calculation is reliant on the ExgalM5 metric. So declare that here.
        self.exgalM5 = ExgalM5(m5Col=m5Col, units=units, lsstFilter=self.lsstFilter)
        
        #Initiate the metric.
        super(ExgalM5_with_cuts_AGN, self).__init__(
            col=[m5Col, filterCol], metricName=metricName, maps=self.exgalM5.maps, units=units,
            **kwargs)
        
    def run(self, dataSlice, slicePoint=None):
        # exclude areas with high extinction
        if slicePoint['ebv'] > self.extinction_cut:
            return self.badval
        
        dS = dataSlice[dataSlice[self.filterCol] == self.lsstFilter]
        mlim5 = self.exgalM5.run(dS, slicePoint)
        return mlim5