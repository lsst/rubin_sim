# Base class for all metrics that use LightCurveLynx models.

from .base_metric import BaseMetric


class LynxBaseMetric(BaseMetric):
    """A base class for metrics that use LightCurveLynx models
    that provides support for evaluating the model on the given points.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to the parent class (BaseMetric).
    """
    # The columns from the Opsim database that are required for evaluating the model.
    _opsim_cols = [
        "fieldRA",
        "fieldDec",
        "filter",
        "visitExposureTime",
        "numExposures",
        "seeingFwhmEff",
        "skyBrightness",
        "observationStartMJD",
        "airmass",
    ]
    def __init__(self, **kwargs):
        super().__init__(col=self._opsim_cols, **kwargs)

    def get_lightcurve(self, data_slice, slice_point):
        """Calculate metric values.

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
        lightcurve : `Pandas DataFrame`
            A pandas data frame with the light curve information for the object.
        """
        try:
            from lightcurvelynx.utils.maf_api import MAFQueryTable, execute_maf_query
        except ImportError as e:
            raise ImportError(
                "LightCurveLynx is needed to run the LynxDetectionMetric. It is not installed by "
                "default. Install it with `pip install lightcurvelynx`."
            )

        # Convert the numpy table into a dictionary and use that to initialize the MAF query table.
        data_dict = {col: data_slice[col] for col in self._opsim_cols}
        maf_query_table = MAFQueryTable(data_dict)

        # Execute the MAF query to retrieve the lightcurve data for this slice point.
        lightcurve_data, _ = execute_maf_query(
            slice_point["lynx_model"],
            maf_query_table,
            graph_state = slice_point["lynx_params"],
        )
        return lightcurve_data


class LynxDetectionMetric(LynxBaseMetric):
    """A metric that counts the number of times an object's signal to noise
    is above the given threshold.

    Parameters
    ----------
    threshold : `float`
        The signal-to-noise ratio threshold above which detections are counted.
    **kwargs
        Additional keyword arguments passed to the parent class (BaseMetric).
    """
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def run(self, data_slice, slice_point=None):
        try:
            from lightcurvelynx.utils.post_process_results import lightcurve_compute_snr
        except ImportError as e:
            raise ImportError(
                "LightCurveLynx is needed to run the LynxDetectionMetric. It is not installed by "
                "default. Install it with `pip install lightcurvelynx`."
            )

        lightcurve = self.get_lightcurve(data_slice, slice_point)
        if lightcurve is None or len(lightcurve) == 0:
            return 0

        # Count the number of detections above the threshold.
        snr = lightcurve_compute_snr(lightcurve["flux"], lightcurve["fluxerr"])
        detections = snr > self.threshold
        return detections.sum()
