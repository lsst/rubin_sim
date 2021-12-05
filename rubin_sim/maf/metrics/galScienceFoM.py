################################################################################################
# Metric to evaluate the Galactic Science Figure of Merit
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
import galPlaneFootprintMetric, galPlaneTimePerFilter, transientTimeSamplingMetric

class galPlaneFoM(maf.BaseMetric):
    """Figure of Merit to evaluate the combination of survey footprint, cadence and filter selection for suitability
    for Galactic Plane science"""

    def __init__(self, cols=['fieldRA','fieldDec','filter',
                             'observationStartMJD','visitExposureTime','fiveSigmaDepth'],
                       metricName='galPlaneFoM',
                       **kwargs):

        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.m5Col = 'fiveSigmaDepth'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.magCuts = {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}
        self.mjdCol = 'observationStartMJD'
        self.exptCol = 'visitExposureTime'

        super().__init__(col=cols, metricName=metricName)


    def run(self, dataSlice, slicePoint=None):

        # Survey footprint metric produces a single value
        footprint_metric = galPlaneFootprintMetric.galPlaneFootprintMetric()
        footprint_metric_data = footprint_metric.run(dataSlice, slicePoint)

        # Cadence metric produces a set of four values for four time categories
        cadence_metric = transientTimeSamplingMetric.transientTimeSamplingMetric()
        cadence_metric_data = cadence_metric.run(dataSlice, slicePoint)

        # Filter selection metric produces a single value
        filters_metric = galPlaneTimePerFilterMetric.galPlaneTimePerFilter()
        filters_metric_data = filters_metric.run(dataSlice, slicePoint)

        # Note: Consider summing cadence_metric_values over all variability categories
        # to produce a single FoM value?
        fom = 0.0
        for t_metric in cadence_metric_data.metric_values:
            fom += footprint_metric_data.data * t_metric * filters_metric_data.data

        return fom
