__all__ = ("MDwarfCountMetric", "MDwarfMeasureMetric")

import numpy as np
from uncertainties import unumpy
from rubin_sim.maf.metrics.base_lcl_metric import BaseLCLMetric

class MDwarfCountMetric(BaseLCLMetric):
    """
    Metric based on lightcurvelynx parquet file.
    Counts the number of flares that have been observed at least once
    in u, g, or r by checks if the flare flux is significantly different
    than the quiescent state.
    Averages all the quiescent flux that we simulated pre-flare in order 
    to compare the fluxes and flux errors to the flare observation.

    Checks observations within 1 day before to 2 days after peak
    (to avoid extra compute in checking far-away data points)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data_slice, slice_point):
        """
        Calculate metric values.

        Parameters
        ----------
        data_slice : `numpy.ndarray`, (N,)
           Values passed to metric by the slicer, which the metric will
           use to calculate metric values at each slice_point.
        slice_point : `dict` or None
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel.

        Returns
        -------
        metricValue : `int` `float` or `object`
            The metric value at each slice_point.
        """

        data_slice = data_slice[0][0]  # lynne or max - does anyone know why?
        detected_mask = data_slice["snr"] > 10  # only using detected data points
        observed_points = data_slice[detected_mask]
        bands = ["u", "g", "r"]
        # for each band, see if any data points are observed in flare
        for band in bands:
            observations_in_filter = observed_points[observed_points["filter"] == band]
            pre_flare_mask = observations_in_filter["time_rel"] < -1  # more than one day pre-peak
            potential_flare_mask = (observations_in_filter["time_rel"] > -1) * (
                observations_in_filter["time_rel"] < 2
            )

            pre_flare_average = np.average(
                unumpy.uarray(
                    observations_in_filter[pre_flare_mask]["flux"],
                    observations_in_filter[pre_flare_mask]["fluxerr"],
                )
            )

            for i, obs in observations_in_filter[potential_flare_mask.values].iterrows():
                flux_diff = obs["flux"] - pre_flare_average.nominal_value
                error_combo = np.sqrt(obs["fluxerr"] ** 2 + pre_flare_average.std_dev**2)
                if np.abs(flux_diff / error_combo) > 3:
                    return True

        return False

class MDwarfMeasureMetric(BaseLCLMetric):
    """
    Counts the number of flares that have been detected
    according to Hilton (2011)'s criteria for confirmed flares:
    three consecutive epochs greater than 3 sigma and one greater
    than 5 sigma.
    Since we consider u, g, or r, we relax that from "consecutive"
    to "within the flare timescale" (in case, i.e., r is lower 
    than the threshold).
    Flare timescale could later be defined by flare FWHM and is now
    a threshold in number of days.

    Recommendation:
    Only pass in flares that have been detected by the count metric 
    to save compute!
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data_slice, slice_point):
        """
        Calculate metric values.

        Parameters
        ----------
        data_slice : `numpy.ndarray`, (N,)
           Values passed to metric by the slicer, which the metric will
           use to calculate metric values at each slice_point.
        slice_point : `dict` or None
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel.

        Returns
        -------
        metricValue : `int` `float` or `object`
            The metric value at each slice_point.
        """
        count_above_3 = 0
        count_above_5 = 0
        data_slice = data_slice[0][0]  # lynne or max - does anyone know why?
        detected_mask = data_slice["snr"] > 10  # only using detected data points
        observed_points = data_slice[detected_mask]
        bands = ["u", "g", "r"]
        # for each band, see if any data points are observed in flare
        for band in bands:
            observations_in_filter = observed_points[observed_points["filter"] == band]
            pre_flare_mask = observations_in_filter["time_rel"] < -1  # more than one day pre-peak
            potential_flare_mask = (observations_in_filter["time_rel"] > -1) * (
                observations_in_filter["time_rel"] < 2
            )

            pre_flare_average = np.average(
                unumpy.uarray(
                    observations_in_filter[pre_flare_mask]["flux"],
                    observations_in_filter[pre_flare_mask]["fluxerr"],
                )
            )

            for i, obs in observations_in_filter[potential_flare_mask.values].iterrows():
                flux_diff = obs["flux"] - pre_flare_average.nominal_value
                error_combo = np.sqrt(obs["fluxerr"] ** 2 + pre_flare_average.std_dev**2)
                if np.abs(flux_diff / error_combo) > 3:
                    count_above_3 += 1
                if np.abs(flux_diff / error_combo) > 5:
                    count_above_5 += 1
        if count_above_5 > 0 and count_above_3 > 2:
            return True
        else:
            return False
