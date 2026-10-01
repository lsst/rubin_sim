# Base class for all metrics that use LightCurveLynx models.

from .base_metric import BaseMetric


class LynxBaseMetric(BaseMetric):
    """An abstract base class for metrics that use LightCurveLynx models
    that provides support for evaluating the model on the given points.

    Parameters
    ----------
    col : `str` or `list` [`str`]
        Names of the data columns that the metric will use.
        The columns required for each metric is tracked in the ColRegistry,
        and used to retrieve data from the opsim database.
        Can be a single string or a list.
    metric_name : `str`
        Name to use for the metric (optional - if not set, will be derived).
    maps : `list` [`rubin_sim.maf.maps`]
        The maps that the metric will need (passed from the slicer).
    units : `str`
        The units for the value returned by the metric (optional - if not set,
        will be derived from the ColInfo).
    metric_dtype : `str`
        The type of value returned by the metric - 'int', 'float', 'object'.
        If not set, will be derived by introspection.
    badval : `float`
        The value indicating "bad" values calculated by the metric.
    """
    _opsim_cols = [
        "visitExposureTime",  # seconds
        "filter",
        "numExposures",  # count
        "seeingFwhmEff",  # arcseconds
        "skyBrightness",  # mag per arcsec^2
        "observationStartMJD",  # days
        "zp_nJy",  # nJy
    ]
    def __init__(self):
        super().__init__(col=self._opsim_cols)

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
                "LightCurveLynx is needed to run the LynxSamplerSlicer. It is not installed by "
                "default. Install it with `pip install lightcurvelynx`."
            )

        # Convert the numpy table into a dictionary and use that to initialize the MAF query table.
        data_dict = {col: data_slice[col] for col in self._opsim_cols}
        maf_query_table = MAFQueryTable(data_dict)

        # Execute the MAF query to retrieve the lightcurve data for this slice point.
        lightcurve_data = execute_maf_query(
            slice_point["lynx_model"],
            maf_query_table,
            slice_point["lynx_params"],
        )
        return lightcurve_data["lightcurve"][0]
 