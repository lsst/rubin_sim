################################################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
import calcVisitIntervalMetric, calcSeasonVisibilityGapsMetric
import TauObsMetricData

class calcVisitIntervalMetric(maf.BaseMetric):
    """Metric to evaluate the intervals between sequential observations in a
    lightcurve relative to the scientifically desired sampling interval.

    Parameters
    ----------
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """

    def __init__(self, cols=['observationStartMJD','fiveSigmaDepth'],
                       metricName='calcVisitIntervalMetric',
                       **kwargs):
        """tau_obs is an array of minimum-required observation intervals for four categories
        of time variability"""

        self.mjdCol = 'observationStartMJD'
        self.m5Col = 'fiveSigmaDepth'
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])
        self.magLimit = 22.0

        super().__init__(col=cols, metricName=metricName, metricDtype='object')


    def run(self, dataSlice, slicePoint=None):

        metric_data = TauObsMetricData()

        # Select observations in the time sequence that fulfill the S/N requirements:
        match = np.where(dataSlice[self.m5Col] >= self.magLimit)[0]

        # Calculate the median time interval from the observation
        # sequence in the dataSlice
        tobs_ordered = dataSlice[self.mjdCol][match]
        tobs_ordered.sort()
        delta_tobs = tobs_ordered[1:] - tobs_ordered[0:-1]

        for i,tau in enumerate(self.tau_obs):
            metric_data.metric_values[i] = self.calc_interval_metric(delta_tobs, tau)

            # Normalize by the number of intervals in the lightcurve
            metric_data.metric_values[i] /= len(delta_tobs)

        return metric_data

    def calc_interval_metric(self,delta_tobs, tau):
        # Decay constant for metric value relationship as a function of
        # observation interval
        K = 1.0/tau
        m = np.zeros(len(delta_tobs))
        idx = np.where(delta_tobs <= tau)[0]
        m[idx] = 1.0
        idx = np.where(delta_tobs > tau)[0]
        m[idx] = np.exp(-K*(delta_tobs[idx] - tau))
        return m.sum()

    def reduceTau0(self, metric_data):
        return metric_data.metric_values[0]
    def reduceTau1(self, metric_data):
        return metric_data.metric_values[1]
    def reduceTau2(self, metric_data):
        return metric_data.metric_values[2]
    def reduceTau3(self, metric_data):
        return metric_data.metric_values[3]
    
