################################################################################################
# Metric to evaluate the transientTimeSamplingMetric
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import healpy as hp
import rubin_sim.maf as maf

class calcVisitIntervalMetric(maf.BaseMetric):
    """Metric to evaluate the intervals between sequential observations in a
    lightcurve relative to the scientifically desired sampling interval.

    Parameters
    ----------
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """

    def __init__(
        self,
        cols=["observationStartMJD", "fiveSigmaDepth"],
        metricName="calcVisitIntervalMetric",
        **kwargs
    ):
        """tau_obs is an array of minimum-required observation intervals for
        four categories of time variability"""

        self.mjdCol = "observationStartMJD"
        self.m5Col = "fiveSigmaDepth"
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])
        self.magLimit = 22.0

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def run(self, dataSlice, slicePoint=None):

        metric_data = {}

        # Select observations in the time sequence that fulfill the
        # S/N requirements:
        match = np.where(dataSlice[self.m5Col] >= self.magLimit)[0]

        # Calculate the median time interval from the observation
        # sequence in the dataSlice
        tobs_ordered = dataSlice[self.mjdCol][match]
        tobs_ordered.sort()
        delta_tobs = tobs_ordered[1:] - tobs_ordered[0:-1]

        for i, tau in enumerate(self.tau_obs):
            metric_data[tau] = self.calc_interval_metric(delta_tobs, tau)

            # Normalize by the number of intervals in the lightcurve
            metric_data[tau] /= len(delta_tobs)

        return metric_data

    def calc_interval_metric(self, delta_tobs, tau):
        # Decay constant for metric value relationship as a function of
        # observation interval
        K = 1.0 / tau
        m = np.zeros(len(delta_tobs))
        idx = np.where(delta_tobs <= tau)[0]
        m[idx] = 1.0
        idx = np.where(delta_tobs > tau)[0]
        m[idx] = np.exp(-K * (delta_tobs[idx] - tau))
        return m.sum()


class calcSeasonVisibilityGapsMetric(maf.BaseMetric):
    """Metric to evaluate the gap between sequential seasonal gaps in
    observations in a lightcurve relative to the scientifically desired
    sampling interval.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """

    def __init__(
        self,
        cols=[
            "fieldRA",
            "observationStartMJD",
        ],
        metricName="calcSeasonVisibilityGapsMetric",
        **kwargs
    ):

        """tau_obs is an array of minimum-required observation intervals for
        four categories of time variability"""

        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])
        self.ra_col = "fieldRA"
        self.mjdCol = "observationStartMJD"

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def calcSeasonGaps(self, dataSlice):
        """Given the RA of a field pointing, and time of observation, calculate
        the length of the gaps between observing seasons.

        Parameters
        ----------
        ra : float
            The RA (in degrees) of the point on the sky
        time : np.ndarray
            The times of the observations, in MJD
        Returns
        -------
        np.ndarray
            Time gaps in days between sequential observing seasons
        """

        seasons = maf.seasonMetrics.calcSeason(
            dataSlice[self.ra_col], dataSlice[self.mjdCol]
        )
        seasons.sort()
        firstOfSeason, lastOfSeason = maf.seasonMetrics.findSeasonEdges(seasons)

        ngaps = len(firstOfSeason) - 1
        season_gaps = (
            dataSlice[self.mjdCol][lastOfSeason[0 : ngaps - 1]]
            - dataSlice[self.mjdCol][firstOfSeason[1:ngaps]]
        )

        return season_gaps

    def run(self, dataSlice, slicePoint=None):
        season_gaps = self.calcSeasonGaps(dataSlice)

        # To avoid the intensive calculation of the exact visibility of every pointing
        # for 365d a year, we adopt the pre-calculated values for an example field in
        # the Galactic Bulge, which receives good, but not circumpolar, annual visibility.
        total_time_visible_days = 1975.1256 / 24.0
        expected_gap = 365.24 - total_time_visible_days

        metric_data = {}
        interval_metric = calcVisitIntervalMetric()
        for i, tau in enumerate(self.tau_obs):
            if tau >= expected_gap:
                metric_data[tau] = 0.0
                for t in season_gaps:
                    metric_data[
                    tau
                    ] += interval_metric.calc_interval_metric(np.array([t]), tau)
                metric_data[tau] /= 10.0

            else:
                metric_data[tau] = 1.0

        return metric_data

class transientTimeSamplingMetric(maf.BaseMetric):
    """Metric to evaluate how well a survey strategy will sample lightcurves,
    using a metric geared towards transient phenomena, to evaluate both the
    intervals between sequential observations and also the impact of gaps between
    observing seasons.

    Parameters
    ----------
    observationStartMJD : float, MJD timestamp of the start of a given observation
    """

    def __init__(
        self,
        cols=[
            "observationStartMJD",
            "fiveSigmaDepth",
        ],
        metricName="calcVisitIntervalMetric",
        **kwargs
    ):
        """tau_obs is an array of minimum-required observation intervals for
        four categories of time variability"""

        self.mjdCol = "observationStartMJD"
        self.m5Col = "fiveSigmaDepth"
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def run(self, dataSlice, slicePoint=None):

        metric1 = calcVisitIntervalMetric()
        m1 = metric1.run(dataSlice, slicePoint)
        metric2 = calcSeasonVisibilityGapsMetric()
        m2 = metric2.run(dataSlice, slicePoint)

        metric_data = {}
        for i, tau in enumerate(self.tau_obs):
            metric_data[tau] = m1[tau] * m2[tau]

        return metric_data
