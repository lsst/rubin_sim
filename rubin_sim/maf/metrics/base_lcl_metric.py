from .base_metric import BaseMetric


class BaseLCLMetric(BaseMetric):
    """
    Base class for LCL metrics, extending the BaseMetric class.
    This class can be used as a foundation for creating specific LCL metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        return len(data_slice[0][0]["flux"])
