__all__ = ("GRBAfterglowMetric")

import numpy as np
from rubin_sim.maf.metrics.base_lynx_metrics import BaseLynxMetric


class GRBAfterglowMetric(BaseLynxMetric):
    """
    Metric based on lightcurvelynx parquet file FOR NOW. hope to do it dynamically
    To recognize a GRB afterglow: Requires two observations in different filters 
    within 12 hours AND two observations in THE SAME filters showing magnitude change.
    As an additional metric, events are flagged as spectroscopically triggerable
    if they reach at least mag 21 in any filter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_dtype = object

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
        metric_value_recognize = False
        metric_value_spectrig = False
        data_slice = data_slice[0][0]  
        detected_mask = data_slice["snr"] > 5  # only using detected data points
        if np.sum(detected_mask)>0:
            
            observed_points = data_slice[detected_mask]

            t = observed_points["time_rel"].to_numpy()  
            filt = observed_points["filter"].to_numpy()
            close_in_time = np.abs(t[:, None] - t[None, :]) < 0.5  
            different_filter = filt[:, None] != filt[None, :]
            two_filters_close = np.any(close_in_time & different_filter)

            if two_filters_close:

                # for each band, see if any data points are observed
                for band in np.unique(filt):
                    observations_in_filter = observed_points[observed_points["filter"] == band]
    
                    #gonna try to account for this in the LCL half
                    
                    # potential_flare_mask = (observations_in_filter["time_rel"] > -1) * (
                    #     observations_in_filter["time_rel"] < 2
                    # )
                    # if not np.sum(potential_flare_mask):  # no potential flares
                    #     continue
    
                    if len(observations_in_filter)<2:  # want two+ in same filter
                        continue
                        
                    diff = observations_in_filter["flux"].to_numpy()[:, None] - observations_in_filter["flux"].to_numpy()[None, :]
                    err_combo = np.sqrt(observations_in_filter["fluxerr"].to_numpy()[:, None] ** 2 \
                                        + observations_in_filter["fluxerr"].to_numpy()[None, :] ** 2)
                    if np.any(np.abs(diff / err_combo) > 3):
                        metric_value_recognize = True
                        if np.any(observed_points["flux"])> 14454: #21 mags
                                metric_value_spectrig = True
                        break

        return {"metric_value_recognize": metric_value_recognize, "metric_value_spectrig": metric_value_spectrig}  # "first_true_band": None,

    def reduce_metric_value_recognize(self, metric_value):
        return metric_value["metric_value_recognize"]

    # def reduce_band(self, metric_value):
    #     return metric_value["first_true_band"]
    def reduce_metric_value_spectrig(self, metric_value):
        return metric_value["metric_value_spectrig"]
