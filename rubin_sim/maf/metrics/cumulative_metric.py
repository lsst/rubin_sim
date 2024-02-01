__all__ = ("CumulativeMetric",)

import numpy as np

from .base_metric import BaseMetric


class CumulativeMetric(BaseMetric):
    """For plotting up the cumulative number of observations.
    Expected to be used with a UniSlicer or UserPointSlicer with one point.

    Parameters
    ----------
    interp_points : `np.array`, (N,)  or None
        The points to interpolate the cumulative number of observations to.
        If None, then the range of the data is used with a stepsize of 1.
    """

    def __init__(
        self,
        metric_name="Cumulative",
        time_col="observationStartMJD",
        night_col="night",
        interp_points=None,
        **kwargs,
    ):
        super().__init__(col=[time_col, night_col], metric_name=metric_name, metric_dtype="object", **kwargs)
        self.time_col = time_col
        self.night_col = night_col
        self.interp_points = interp_points
        self.plot_dict = {"xlabel": "MJD (days)", "ylabel": "N obs"}

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.time_col)
        if self.interp_points is None:
            interp_points = np.arange(
                data_slice[self.night_col].min(),
                data_slice[self.night_col].max() + 1,
                1,
            )
        else:
            interp_points = self.interp_points
        cumulative_number = np.arange(data_slice.size) + 1
        yresult = np.interp(interp_points, data_slice[self.night_col], cumulative_number)
        xresult = interp_points
        return {"x": xresult, "y": yresult, "plot_dict": self.plot_dict}
