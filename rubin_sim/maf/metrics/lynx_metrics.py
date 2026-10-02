# Base class for all metrics that use LightCurveLynx models. The can either
# take a precomputed light curve (saved in the slicepoint) or compute it
# dynamically using LightCurveLynx.

import numpy as np
import numpy.ma as ma

from .base_metric import BaseMetric


class LynxBaseMetric(BaseMetric):
    """A base class for metrics that use LightCurveLynx models
    that provides support for evaluating the model on the given points.

    This class will first try to use the saved "lightcurve" data (in the
    slice_point) if available. Otherwise, it will call out to LightCurveLynx
    to compute the light curve.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to the parent class (BaseMetric).
    """

    # The columns from the Opsim database that are required.
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
        """Run the LightCurveLynx simulation and return the resulting light
        curve information in a pandas DataFrame with columns such as mjd,
        flux, and fluxerr.

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
            A pandas data frame with the light curve information for the
            object.
        """
        # If we have the precomputed lightcurve in the slice_point, use it.
        if slice_point is not None and "lightcurve" in slice_point:
            return slice_point["lightcurve"]

        # Make sure we can use the LightCurveLynx libraries.
        try:
            from lightcurvelynx.utils.maf_api import MAFQueryTable, execute_maf_query
        except ImportError:
            raise ImportError(
                "LightCurveLynx is needed to run the LynxDetectionMetric. "
                "It is not installed by default. Install it with "
                "`pip install lightcurvelynx`."
            )

        # Convert the numpy table into a dictionary and use that to
        # initialize the MAF query table.
        data_dict = {col: data_slice[col] for col in self._opsim_cols}
        maf_query_table = MAFQueryTable(data_dict)

        # Execute the MAF query to retrieve the lightcurve data for
        # this slice point.
        lightcurve_data, _ = execute_maf_query(
            slice_point["lynx_model"],
            maf_query_table,
            graph_state=slice_point["lynx_params"],
        )
        return lightcurve_data


class LynxDetectionMetric(LynxBaseMetric):
    """A metric that counts the number of times an object's signal to noise
    is above the given threshold.

    Parameters
    ----------
    threshold : `float`
        The signal-to-noise ratio threshold above which detections are
        counted.
    **kwargs
        Additional keyword arguments passed to the parent class (BaseMetric).
    """

    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def run(self, data_slice, slice_point=None):
        lightcurve = self.get_lightcurve(data_slice, slice_point)
        if lightcurve is None or len(lightcurve) == 0:
            return 0

        # Compute the signal-to-noise ratio for each observation, masking
        # out invalid values.
        flux = np.asarray(lightcurve["flux"])
        fluxerr = np.asarray(lightcurve["fluxerr"])
        valid_mask = (flux > 0) & (fluxerr > 0)
        snr = ma.masked_all(flux.shape)
        snr[valid_mask] = flux[valid_mask] / fluxerr[valid_mask]

        # Count the number of detections above the threshold.
        detections = snr > self.threshold
        return detections.sum()
