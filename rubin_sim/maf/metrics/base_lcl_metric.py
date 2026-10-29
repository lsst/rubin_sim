# Base class for all metrics that use LightCurveLynx models. The can either
# take a precomputed light curve (saved in the slicepoint) or compute it
# dynamically using LightCurveLynx.

from .base_metric import BaseMetric


class BaseLCLMetric(BaseMetric):
    """
    Base class for LightCurveLynx metrics, extending the BaseMetric class.

    This class can either read from the saved "lightcurve" data (in the
    data_slice) via the `LynxResultsSliceror` compute it dynamically using
    LightCurveLynx via the data by the `LynxResultsSlicer`.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to the parent class (BaseMetric).
    """

    # The columns from the Opsim database that are required for simulations.
    _simulate_cols = [
        "fieldRA",
        "fieldDec",
        "band",
        "visitExposureTime",
        "numExposures",
        "seeingFwhmEff",
        "skyBrightness",
        "observationStartMJD",
        "airmass",
    ]

    def __init__(self, *args, **kwargs):
        # Update the columns that we need to fetch from the Opsim database
        # to ensure we have all the necessary columns for the simulation.
        kwargs["col"] = list(set(kwargs.get("col", []) + self._simulate_cols))

        # Initialize the parent class with the updated keyword arguments.
        super().__init__(*args, **kwargs)

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
        # If the slice_point does not contain the necessary model information,
        # we assume the data came from the precomputed lightcurve in the data_slice.
        if slice_point is None or "lynx_model" not in slice_point or "lynx_params" not in slice_point:
            return data_slice[0][0]

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
        data_dict = {col: data_slice[col] for col in self._simulate_cols}
        maf_query_table = MAFQueryTable(data_dict)

        # Execute the MAF query to retrieve the lightcurve data for
        # this slice point.
        lightcurve_data, _ = execute_maf_query(
            slice_point["lynx_model"],
            maf_query_table,
            graph_state=slice_point["lynx_params"],
        )
        return lightcurve_data


class LCLObsCountMetric(BaseLCLMetric):
    """
    A simple test metric that will probably be removed later.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data_slice, slice_point):
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
        metricValue : `int` `float` or `object`
            The metric value at each slice_point."""
        lc = self.get_lightcurve(data_slice, slice_point)
        return len(lc["flux"])
