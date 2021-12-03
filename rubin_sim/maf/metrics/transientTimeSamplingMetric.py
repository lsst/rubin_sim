################################################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
import calcVisitIntervalMetric, calcSeasonVisibilityGapsMetric

class transientTimeSamplingMetric(maf.BaseMetric):
    """Metric to evaluate how well a survey strategy will sample lightcurves,
    using a metric geared towards transient phenomena, to evaluate both the
    intervals between sequential observations and also the impact of gaps between
    observing seasons.

    Parameters
    ----------
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """
    def __init__(self, cols=['observationStartMJD',],
                       metricName='transientTimeSamplingMetric',
                       **kwargs):
         """tau_obs is an array of minimum-required observation intervals for four categories
        of time variability"""

        self.mjdCol = 'observationStartMJD'
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])

        super().__init__(col=cols, metricName=metricName)


    def run(self, dataSlice, slicePoint=None):

        metric_values1 = np.array(calcVisitIntervalMetric(dataSlice, slicePoint))
        metric_values2 = np.array(calcSeasonVisibilityGapsMetric(dataSlice, slicePoint))

        metric_values= metric_values1 * metric_values2

        return metric_values
